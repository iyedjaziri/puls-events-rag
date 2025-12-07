"""
Test suite for vectorization module.
"""

import os
import pickle
import shutil
import numpy as np
import pytest
import faiss
from scripts.vectorization import EventVectorizer


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": "chunk_0",
            "text": "This is a test event about jazz music.",
            "metadata": {"title": "Jazz Concert"}
        },
        {
            "chunk_id": "chunk_1",
            "text": "Another event about art exhibition.",
            "metadata": {"title": "Art Show"}
        }
    ]


class TestEventVectorizer:
    """Test EventVectorizer class."""
    
    def test_init(self):
        """Test initialization."""
        vectorizer = EventVectorizer()
        assert vectorizer.model is not None
        assert vectorizer.dimension == 768  # MPNet dimension
        
    def test_vectorize_chunks(self, sample_chunks):
        """Test embedding generation."""
        vectorizer = EventVectorizer()
        embeddings = vectorizer.vectorize_chunks(sample_chunks)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)
        assert embeddings.dtype == "float32"
        
    def test_build_index(self, sample_chunks):
        """Test index building."""
        vectorizer = EventVectorizer()
        embeddings = vectorizer.vectorize_chunks(sample_chunks)
        
        index = vectorizer.build_index(embeddings, nlist=1, nprobe=1)
        
        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.ntotal == 2
        assert index.d == 768
        
    def test_save_index(self, sample_chunks, tmp_path):
        """Test saving index and metadata."""
        vectorizer = EventVectorizer()
        embeddings = vectorizer.vectorize_chunks(sample_chunks)
        index = vectorizer.build_index(embeddings, nlist=1)
        
        output_dir = tmp_path / "faiss_index"
        vectorizer.save_index(index, sample_chunks, output_dir=str(output_dir))
        
        assert (output_dir / "events.index").exists()
        assert (output_dir / "metadata.pkl").exists()
        assert (output_dir / "config.json").exists()
        
        # Verify metadata load
        with open(output_dir / "metadata.pkl", "rb") as f:
            loaded_chunks = pickle.load(f)
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0]["chunk_id"] == "chunk_0"
