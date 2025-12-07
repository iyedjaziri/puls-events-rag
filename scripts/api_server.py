"""
FastAPI Server
Production REST API for cultural event recommendation chatbot.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field, validator

from scripts.rag_system import EventRAGSystem
from scripts.data_extraction import OpenAgendaExtractor
from scripts.preprocessing import EventPreprocessor
from scripts.vectorization import EventVectorizer
import threading

# Load environment variables
load_dotenv()

# Global RAG system instance
rag_system: Optional[EventRAGSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global rag_system
    
    # Startup: Load RAG system
    logger.info("Initializing RAG system...")
    
    try:
        rag_system = EventRAGSystem(
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "faiss_index/events.index"),
            metadata_path=os.getenv("METADATA_PATH", "faiss_index/metadata.pkl"),
            config_path=os.getenv("CONFIG_PATH", "faiss_index/config.json"),
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            top_k=int(os.getenv("TOP_K", "5"))
        )
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to initialize RAG system on startup: {e}")
        logger.warning("Server starting in degraded mode. Use /rebuild to initialize index.")
        # Do not raise, allow server to start
        
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="PULS Events RAG API",
    description="Intelligent cultural event recommendation chatbot for Paris",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AskRequest(BaseModel):
    """Request model for /ask endpoint."""
    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Natural language question about cultural events",
        example="Quels concerts de jazz ont lieu ce week-end à Paris?"
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Number of events to retrieve (default: 5)"
    )
    
    @validator('question')
    def validate_question(cls, v):
        """Validate question is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v.strip()


class EventSource(BaseModel):
    """Event source metadata."""
    event_id: str
    title: str
    location: str
    city: str
    date: str
    link: str
    similarity_score: float


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""
    question: str
    answer: str
    sources: list[EventSource]
    response_time_ms: float


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    rag_loaded: bool
    index_size: Optional[int] = None
    message: Optional[str] = None


# API Endpoints
@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "PULS Events RAG API",
        "version": "1.0.0",
        "description": "Intelligent cultural event recommendation chatbot",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns system status and RAG system information.
    """
    if rag_system is None:
        return HealthResponse(
            status="unhealthy",
            rag_loaded=False,
            message="RAG system not initialized"
        )
        
    try:
        index_size = rag_system.index.ntotal
        return HealthResponse(
            status="healthy",
            rag_loaded=True,
            index_size=index_size,
            message=f"RAG system operational with {index_size} indexed vectors"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            rag_loaded=True,
            message=f"RAG system loaded but error: {str(e)}"
        )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question about cultural events.
    
    The system will:
    1. Retrieve relevant event chunks using semantic search
    2. Generate a contextual response using Mistral LLM
    3. Return recommendations with source events
    
    Example questions:
    - "Quels concerts de jazz ont lieu ce week-end?"
    - "Expositions d'art gratuites dans le Marais"
    - "Événements pour enfants samedi prochain"
    """
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
        
    try:
        # Call RAG system
        result = rag_system.ask(
            question=request.question,
            k=request.top_k,
            include_sources=True
        )
        
        # Convert sources to EventSource models
        sources = [EventSource(**source) for source in result["sources"]]
        
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            response_time_ms=result["response_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/stats", tags=["Stats"])
async def get_stats() -> Dict:
    """
    Get system statistics.
    
    Returns information about indexed events and performance.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
        
    try:
        return {
            "total_vectors": rag_system.index.ntotal,
            "total_chunks": len(rag_system.metadata),
            "embedding_dimension": rag_system.config.get("dimension", 384),
            "index_type": rag_system.config.get("index_type", "IVFFlat"),
            "nlist": rag_system.config.get("nlist", 100),
            "nprobe": rag_system.config.get("nprobe", 10)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@app.post("/rebuild", tags=["Admin"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Trigger a complete rebuild of the RAG index.
    
    This process:
    1. Extracts fresh data from Open Agenda
    2. Preprocesses and chunks the data
    3. Generates new embeddings and builds Faiss index
    4. Reloads the RAG system
    """
    def run_rebuild():
        try:
            logger.info("Starting index rebuild...")
            
            # 1. Extraction
            extractor = OpenAgendaExtractor()
            events = extractor.extract_events(save_path="data/raw/events_raw.json")
            
            # 2. Preprocessing
            preprocessor = EventPreprocessor()
            chunks = preprocessor.preprocess_events(events, save_path="data/processed/events_processed.json")
            
            # 3. Vectorization
            vectorizer = EventVectorizer()
            vectorizer.vectorize_events(chunks, save_path="faiss_index/events.index")
            
            # 4. Reload RAG
            global rag_system
            rag_system = EventRAGSystem(
                faiss_index_path=os.getenv("FAISS_INDEX_PATH", "faiss_index/events.index"),
                metadata_path=os.getenv("METADATA_PATH", "faiss_index/metadata.pkl"),
                config_path=os.getenv("CONFIG_PATH", "faiss_index/config.json"),
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                top_k=int(os.getenv("TOP_K", "5"))
            )
            logger.info("Index rebuild completed successfully")
            
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")

    background_tasks.add_task(run_rebuild)
    return {"message": "Index rebuild started in background"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
