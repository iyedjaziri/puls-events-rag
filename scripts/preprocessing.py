"""
Preprocessing Module
Clean, normalize, and chunk event descriptions for vectorization.
"""

import json
import re
from typing import Dict, List

from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm


class EventPreprocessor:
    """Preprocess cultural event data for RAG pipeline."""
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 200,
        min_chunk_length: int = 50
    ):
        """
        Initialize preprocessor.
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_length: Minimum chunk length in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        
    def preprocess_events(
        self,
        events: List[Dict],
        save_path: str = None
    ) -> List[Dict]:
        """
        Preprocess and chunk events.
        
        Args:
            events: List of raw event records
            save_path: Path to save processed chunks
            
        Returns:
            List of text chunks with metadata
        """
        logger.info(f"Processing {len(events)} events")
        
        all_chunks = []
        chunk_id = 0
        
        for event in tqdm(events, desc="Preprocessing events"):
            # Extract event data
            fields = event.get("fields", {})
            event_id = event.get("recordid", f"event_{chunk_id}")
            
            # Build unified text
            text = self._build_event_text(fields)
            
            if not text or len(text) < self.min_chunk_length:
                continue
                
            # Create chunks
            chunks = self._chunk_text(text)
            
            # Build chunk metadata
            for chunk_text in chunks:
                chunk_data = {
                    "chunk_id": f"chunk_{chunk_id}",
                    "event_id": event_id,
                    "text": chunk_text,
                    "metadata": {
                        "title": fields.get("title_fr", fields.get("title", "")),
                        "location": fields.get("location_name", fields.get("location", "")),
                        "city": fields.get("location_city", fields.get("city", "")),
                        "firstdate": fields.get("firstdate_begin", fields.get("firstdate", "")),
                        "lastdate": fields.get("lastdate_end", fields.get("lastdate", "")),
                        "keywords": fields.get("keywords_fr", fields.get("keywords", [])),
                        "categories": fields.get("categories", []),
                        "free": "gratuit" in fields.get("conditions_fr", "").lower(),
                        "link": fields.get("canonicalurl", fields.get("link", "")),
                        "latitude": fields.get("location_coordinates", [None, None])[0],
                        "longitude": fields.get("location_coordinates", [None, None])[1]
                    }
                }
                all_chunks.append(chunk_data)
                chunk_id += 1
                
        logger.info(f"Created {len(all_chunks)} chunks from {len(events)} events")
        
        # Save processed chunks
        if save_path:
            self._save_chunks(all_chunks, save_path)
            
        return all_chunks
    
    def _build_event_text(self, fields: Dict) -> str:
        """Build unified text from event fields."""
        parts = []
        
        # Title (weighted higher)
        title = fields.get("title_fr", fields.get("title", ""))
        if title:
            parts.append(f"{title}. {title}.")  # Repeat for emphasis
            
        # Description
        description = fields.get("description_fr", fields.get("description", ""))
        if description:
            clean_desc = self._clean_html(description)
            parts.append(clean_desc)
            
        # Long description
        long_desc = fields.get("longdescription_fr", fields.get("longdescription", ""))
        if long_desc:
            clean_long = self._clean_html(long_desc)
            parts.append(clean_long)
            
        # Keywords
        keywords = fields.get("keywords_fr", fields.get("keywords", []))
        if isinstance(keywords, str):
            keywords = keywords.split(";")
        
        if keywords:
            parts.append(f"Mots-clés: {', '.join(keywords)}")
            
        # Categories
        categories = fields.get("categories", [])
        if categories:
            parts.append(f"Catégories: {', '.join(categories)}")
            
        # Location context
        location = fields.get("location_name", fields.get("location", ""))
        city = fields.get("location_city", fields.get("city", ""))
        if location or city:
            location_text = f"{location}, {city}" if location and city else (location or city)
            parts.append(f"Lieu: {location_text}")
            
        text = " ".join(parts)
        return self._normalize_text(text)
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text."""
        if not text:
            return ""
            
        # Parse HTML
        soup = BeautifulSoup(text, "lxml")
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get text
        text = soup.get_text()
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (whitespace, special characters)."""
        if not text:
            return ""
            
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace('"', '"')
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Uses simple word-based chunking with sentence boundary preservation.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                # Keep last sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    s_length = len(s.split())
                    if overlap_length + s_length <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += s_length
                    else:
                        break
                        
                current_chunk = overlap_sentences
                current_length = overlap_length
                
            current_chunk.append(sentence)
            current_length += sentence_length
            
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return [c for c in chunks if len(c) >= self.min_chunk_length]
    
    def _save_chunks(self, chunks: List[Dict], save_path: str):
        """Save processed chunks to JSON."""
        logger.info(f"Saving {len(chunks)} chunks to {save_path}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info("Chunks saved successfully")


def main():
    """Example usage."""
    # Load raw events
    with open("data/raw/events_raw.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    
    # Preprocess
    preprocessor = EventPreprocessor(
        chunk_size=400,
        chunk_overlap=200,
        min_chunk_length=50
    )
    
    chunks = preprocessor.preprocess_events(
        events,
        save_path="data/processed/events_processed.json"
    )
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"✓ Average chunk length: {sum(len(c['text'].split()) for c in chunks) / len(chunks):.0f} words")
    
    # Show sample chunk
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk:")
        print(f"  ID: {sample['chunk_id']}")
        print(f"  Event: {sample['metadata']['title']}")
        print(f"  Text preview: {sample['text'][:200]}...")


if __name__ == "__main__":
    main()
