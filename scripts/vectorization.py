"""
Vectorization Module
Generate embeddings and build Faiss index for event chunks.
"""

import json
import pickle
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EventVectorizer:
    """Vectorize text chunks and build Faiss index."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu"
    ):
        """
        Initialize vectorizer.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cpu' or 'cuda')
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Dimension: {self.dimension}")
        
    def vectorize_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        texts = [chunk["text"] for chunk in chunks]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.astype("float32")
    
    def build_index(
        self,
        embeddings: np.ndarray,
        nlist: int = 100,
        nprobe: int = 10
    ) -> faiss.Index:
        """
        Build Faiss IVFFlat index.
        
        Args:
            embeddings: Matrix of embeddings
            nlist: Number of clusters (Voronoi cells)
            nprobe: Number of clusters to visit during search
            
        Returns:
            Trained and populated Faiss index
        """
        logger.info(f"Building Faiss index (nlist={nlist})")
        
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Train index
        logger.info("Training index...")
        # Use a subset for training if dataset is large, otherwise use all
        train_size = min(len(embeddings), 50000)
        index.train(embeddings[:train_size])
        
        # Add vectors
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        # Set nprobe
        index.nprobe = nprobe
        
        logger.info(f"Index built with {index.ntotal} vectors")
        return index
    
    def save_index(
        self,
        index: faiss.Index,
        chunks: List[Dict],
        output_dir: str = "faiss_index"
    ):
        """
        Save index and metadata to disk.
        
        Args:
            index: Faiss index
            chunks: List of chunk dictionaries (metadata)
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save index
        index_path = os.path.join(output_dir, "events.index")
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "wb") as f:
            pickle.dump(chunks, f)
            
        # Save config
        config_path = os.path.join(output_dir, "config.json")
        config = {
            "dimension": self.dimension,
            "ntotal": index.ntotal,
            "nlist": index.nlist,
            "nprobe": index.nprobe,
            "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info("Index and metadata saved successfully")


    def vectorize_events(
        self,
        chunks: List[Dict],
        save_path: str = "faiss_index/events.index"
    ):
        """
        Orchestrate full vectorization pipeline.
        
        Args:
            chunks: List of text chunks
            save_path: Path to save index (directory or file path)
        """
        # Generate embeddings
        embeddings = self.vectorize_chunks(chunks)
        
        # Build index
        index = self.build_index(embeddings)
        
        # Determine output directory
        if save_path.endswith(".index"):
            output_dir = os.path.dirname(save_path)
        else:
            output_dir = save_path
            
        if not output_dir:
            output_dir = "."
            
        # Save results
        self.save_index(index, chunks, output_dir)
        return index


def main():
    """Example usage."""
    # Load processed chunks
    input_path = "data/processed/events_processed.json"
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # Initialize vectorizer
    vectorizer = EventVectorizer()
    
    # Generate embeddings
    embeddings = vectorizer.vectorize_chunks(chunks)
    
    # Build index
    index = vectorizer.build_index(embeddings)
    
    # Save results
    vectorizer.save_index(index, chunks)


if __name__ == "__main__":
    main()
