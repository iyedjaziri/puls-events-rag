"""
Test suite for preprocessing module.
"""

import pytest
import json
from scripts.preprocessing import EventPreprocessor


@pytest.fixture
def sample_events():
    """Sample events for testing."""
    return [
        {
            "recordid": "test_event_1",
            "fields": {
                "title": "Concert de Jazz",
                "description": "<p>Un <b>excellent</b> concert de jazz   au Sunset.</p>",
                "location": "Sunset Jazz Club",
                "city": "Paris",
                "firstdate": "2024-11-15",
                "lastdate": "2024-11-15",
                "keywords": ["jazz", "music"],
                "categories": ["Concert"],
                "free": False,
                "link": "https://example.com/event1"
            }
        },
        {
            "recordid": "test_event_2",
            "fields": {
                "title": "Exposition d'Art",
                "description": "Une belle exposition d'art contemporain. " * 50,  # Long text
                "location": "Galerie Perrotin",
                "city": "Paris",
                "firstdate": "2024-11-20",
                "categories": ["Art"]
            }
        }
    ]


class TestEventPreprocessor:
    """Test EventPreprocessor class."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = EventPreprocessor(
            chunk_size=400,
            chunk_overlap=200,
            min_chunk_length=50
        )
        assert preprocessor.chunk_size == 400
        assert preprocessor.chunk_overlap == 200
        assert preprocessor.min_chunk_length == 50
    
    def test_clean_html(self):
        """Test HTML cleaning."""
        preprocessor = EventPreprocessor()
        
        # Test basic HTML removal
        html = "<p>This is <b>bold</b> text.</p>"
        cleaned = preprocessor._clean_html(html)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "This is bold text." in cleaned
        
        # Test script removal
        html_with_script = "<p>Text</p><script>alert('xss')</script>"
        cleaned = preprocessor._clean_html(html_with_script)
        assert "script" not in cleaned.lower()
        assert "alert" not in cleaned
    
    def test_normalize_text(self):
        """Test text normalization."""
        preprocessor = EventPreprocessor()
        
        # Test whitespace normalization
        text = "Multiple    spaces   and\n\nnewlines"
        normalized = preprocessor._normalize_text(text)
        assert "  " not in normalized
        assert "\n" not in normalized
        
        # Test quote normalization
        text_with_quotes = 'He said "hello" and \'goodbye\''
        normalized = preprocessor._normalize_text(text_with_quotes)
        assert '"' not in normalized
        assert "'" not in normalized
    
    def test_build_event_text(self, sample_events):
        """Test event text building."""
        preprocessor = EventPreprocessor()
        
        fields = sample_events[0]["fields"]
        text = preprocessor._build_event_text(fields)
        
        # Check title is included
        assert "Concert de Jazz" in text
        
        # Check HTML is cleaned
        assert "<p>" not in text
        assert "<b>" not in text
        
        # Check metadata is included
        assert "jazz" in text.lower()
        assert "music" in text.lower()
    
    def test_chunk_text(self):
        """Test text chunking."""
        preprocessor = EventPreprocessor(chunk_size=50, chunk_overlap=20)
        
        # Create long text
        text = " ".join([f"Sentence {i}." for i in range(100)])
        
        chunks = preprocessor._chunk_text(text)
        
        # Check chunks were created
        assert len(chunks) > 1
        
        # Check chunk sizes are reasonable
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count >= preprocessor.min_chunk_length / 5  # Rough estimate
    
    def test_preprocess_events(self, sample_events, tmp_path):
        """Test full preprocessing pipeline."""
        preprocessor = EventPreprocessor(
            chunk_size=100,
            chunk_overlap=50,
            min_chunk_length=20
        )
        
        # Process events
        save_path = tmp_path / "chunks.json"
        chunks = preprocessor.preprocess_events(
            sample_events,
            save_path=str(save_path)
        )
        
        # Check chunks were created
        assert len(chunks) > 0
        
        # Check chunk structure
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "event_id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            
            # Check metadata
            assert "title" in chunk["metadata"]
            assert "location" in chunk["metadata"]
            assert "firstdate" in chunk["metadata"]
        
        # Check file was saved
        assert save_path.exists()
        
        # Verify saved data can be loaded
        with open(save_path, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
        assert len(loaded_chunks) == len(chunks)
    
    def test_empty_description(self):
        """Test handling of events with missing descriptions."""
        preprocessor = EventPreprocessor()
        
        event_no_desc = {
            "recordid": "test_event_no_desc",
            "fields": {
                "title": "Event Title",
                "location": "Paris"
            }
        }
        
        # Should not crash, should use title
        text = preprocessor._build_event_text(event_no_desc["fields"])
        assert "Event Title" in text


def test_integration_preprocessing(sample_events, tmp_path):
    """Integration test for preprocessing."""
    preprocessor = EventPreprocessor()
    
    save_path = tmp_path / "processed.json"
    chunks = preprocessor.preprocess_events(
        sample_events,
        save_path=str(save_path)
    )
    
    # Verify output
    assert len(chunks) >= len(sample_events)
    assert save_path.exists()
    
    # Load and verify
    with open(save_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    assert len(loaded) == len(chunks)
    assert all("text" in c for c in loaded)
    assert all("metadata" in c for c in loaded)
