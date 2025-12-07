"""
Test suite for RAG system.
"""

import json
import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import faiss
from scripts.rag_system import EventRAGSystem


@pytest.fixture
def mock_faiss_index(tmp_path):
    """Create a mock Faiss index and metadata."""
    # Create dummy data
    dimension = 768
    n_vectors = 5
    
    # Create index
    index = faiss.IndexFlatL2(dimension)
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    index.add(vectors)
    
    # Create metadata
    metadata = []
    for i in range(n_vectors):
        metadata.append({
            "chunk_id": f"chunk_{i}",
            "text": f"Event text {i}",
            "metadata": {
                "title": f"Event {i}",
                "location": "Paris",
                "city": "Paris",
                "firstdate": "2024-01-01",
                "lastdate": "2024-01-01",
                "link": "http://example.com"
            }
        })
        
    # Save files
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir()
    
    index_path = index_dir / "events.index"
    faiss.write_index(index, str(index_path))
    
    metadata_path = index_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
        
    config_path = index_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({"dimension": dimension, "nprobe": 1}, f)
        
    return str(index_path), str(metadata_path), str(config_path)


class TestEventRAGSystem:
    """Test EventRAGSystem class."""
    
    def test_init(self, mock_faiss_index):
        """Test initialization."""
        index_path, metadata_path, config_path = mock_faiss_index
        
        rag = EventRAGSystem(
            faiss_index_path=index_path,
            metadata_path=metadata_path,
            config_path=config_path
        )
        
        assert rag.index.ntotal == 5
        assert len(rag.metadata) == 5
        assert rag.embedding_model is not None
        
    def test_retrieve(self, mock_faiss_index):
        """Test retrieval logic."""
        index_path, metadata_path, config_path = mock_faiss_index
        
        rag = EventRAGSystem(
            faiss_index_path=index_path,
            metadata_path=metadata_path,
            config_path=config_path
        )
        
        results = rag.retrieve("test query", k=2)
        
        assert len(results) == 2
        assert "chunk_id" in results[0]
        assert "similarity_score" in results[0]
        
    @patch("scripts.rag_system.Mistral")
    def test_generate_response(self, mock_mistral, mock_faiss_index):
        """Test generation logic with mocked Mistral API."""
        index_path, metadata_path, config_path = mock_faiss_index
        
        # Mock Mistral response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated answer"
        mock_client.chat.complete.return_value = mock_response
        mock_mistral.return_value = mock_client
        
        rag = EventRAGSystem(
            faiss_index_path=index_path,
            metadata_path=metadata_path,
            config_path=config_path,
            mistral_api_key="fake_key"
        )
        
        retrieved = rag.retrieve("test query", k=1)
        answer = rag.generate_response("test query", retrieved)
        
        assert answer == "Generated answer"
        mock_client.chat.complete.assert_called_once()
        
    @patch("scripts.rag_system.Mistral")
    def test_ask(self, mock_mistral, mock_faiss_index):
        """Test full ask pipeline."""
        index_path, metadata_path, config_path = mock_faiss_index
        
        # Mock Mistral response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated answer"
        mock_client.chat.complete.return_value = mock_response
        mock_mistral.return_value = mock_client
        
        rag = EventRAGSystem(
            faiss_index_path=index_path,
            metadata_path=metadata_path,
            config_path=config_path,
            mistral_api_key="fake_key"
        )
        
        result = rag.ask("test query")
        
        assert result["answer"] == "Generated answer"
        assert len(result["sources"]) > 0
        assert "response_time_ms" in result
