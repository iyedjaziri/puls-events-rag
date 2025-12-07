"""
RAG System Module
Complete Retrieval-Augmented Generation pipeline for event recommendations.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from loguru import logger
from mistralai import Mistral
from sentence_transformers import SentenceTransformer


class EventRAGSystem:
    """RAG system for cultural event recommendations."""
    
    def __init__(
        self,
        faiss_index_path: str = "faiss_index/events.index",
        metadata_path: str = "faiss_index/metadata.pkl",
        config_path: str = "faiss_index/config.json",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        mistral_api_key: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize RAG system.
        
        Args:
            faiss_index_path: Path to Faiss index file
            metadata_path: Path to metadata pickle file
            config_path: Path to index configuration
            embedding_model: HuggingFace embedding model name
            mistral_api_key: Mistral AI API key
            top_k: Number of results to retrieve
        """
        self.top_k = top_k
        
        # Load Faiss index
        logger.info(f"Loading Faiss index from {faiss_index_path}")
        self.index = faiss.read_index(faiss_index_path)
        logger.info(f"Index loaded: {self.index.ntotal} vectors")
        
        # Load metadata
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Metadata loaded: {len(self.metadata)} entries")
        
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        # Set nprobe for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.get("nprobe", 10)
            
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("Embedding model loaded")
        
        # Initialize Mistral client
        if mistral_api_key:
            self.mistral_client = Mistral(api_key=mistral_api_key)
            logger.info("Mistral client initialized")
        else:
            self.mistral_client = None
            logger.warning("Mistral API key not provided - generation disabled")
            
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve most relevant event chunks for query.
        
        Args:
            query: Natural language query
            k: Number of results (default: self.top_k)
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        k = k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search Faiss index
        distances, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for idx, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index == -1:  # No more results
                break
                
            chunk_data = self.metadata[index]
            results.append({
                "rank": idx + 1,
                "chunk_id": chunk_data["chunk_id"],
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"],
                "similarity_score": float(distance)
            })
            
        return results
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        model: str = "mistral-small-latest"
    ) -> str:
        """
        Generate response using Mistral LLM.
        
        Args:
            query: User question
            retrieved_chunks: Retrieved context chunks
            model: Mistral model name
            
        Returns:
            Generated response
        """
        if not self.mistral_client:
            return "Error: Mistral client not initialized (missing API key)"
            
        # Build context from retrieved chunks
        context_parts = []
        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            context_parts.append(
                f"**{metadata['title']}**\n"
                f"Lieu: {metadata['location']}, {metadata['city']}\n"
                f"Date: {metadata['firstdate']} - {metadata['lastdate']}\n"
                f"Description: {chunk['text'][:500]}\n"
                f"Lien: {metadata['link']}\n"
            )
            
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """Tu es un expert en recommandation d'événements culturels à Paris.
        
Règles importantes:
- Réponds UNIQUEMENT en te basant sur le contexte fourni
- Recommande des événements spécifiques avec leurs titres, dates, lieux et liens
- Si aucun événement pertinent n'est trouvé, dis-le clairement
- Sois concis et précis
- Fournis toujours les liens vers les événements"""

        user_prompt = f"""Question: {query}

Contexte (événements pertinents):

{context}

Réponds à la question en recommandant les meilleurs événements du contexte."""

        # Call Mistral API
        try:
            response = self.mistral_client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            include_sources: Include source events in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved = self.retrieve(question, k=k)
        
        if not retrieved:
            return {
                "question": question,
                "answer": "Désolé, je n'ai trouvé aucun événement pertinent pour votre question.",
                "sources": [],
                "response_time_ms": (time.time() - start_time) * 1000
            }
            
        # Generate response
        answer = self.generate_response(question, retrieved)
        
        # Build sources
        sources = []
        if include_sources:
            seen_events = set()
            for chunk in retrieved:
                event_id = chunk["chunk_id"].split("_")[0]  # Extract event ID
                if event_id not in seen_events:
                    metadata = chunk["metadata"]
                    sources.append({
                        "event_id": event_id,
                        "title": metadata["title"],
                        "location": metadata["location"],
                        "city": metadata["city"],
                        "date": metadata["firstdate"],
                        "link": metadata["link"],
                        "similarity_score": chunk["similarity_score"]
                    })
                    seen_events.add(event_id)
                    
        response_time = (time.time() - start_time) * 1000
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "response_time_ms": round(response_time, 2)
        }


def main():
    """Example usage."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize RAG system
    rag = EventRAGSystem(
        faiss_index_path="faiss_index/events.index",
        metadata_path="faiss_index/metadata.pkl",
        config_path="faiss_index/config.json",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        top_k=5
    )
    
    # Test queries
    queries = [
        "Concerts de jazz ce week-end à Paris",
        "Expositions d'art gratuites",
        "Événements pour enfants dans le Marais"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print(f"{'='*60}\n")
        
        result = rag.ask(query)
        
        print(f"Réponse:\n{result['answer']}\n")
        print(f"Sources ({len(result['sources'])}):")
        for source in result['sources'][:3]:
            print(f"  - {source['title']}")
            print(f"    {source['location']}, {source['date']}")
            print(f"    {source['link']}\n")
        print(f"Temps de réponse: {result['response_time_ms']:.0f}ms")


if __name__ == "__main__":
    main()
