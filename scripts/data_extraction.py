"""
Data Extraction Module
Extracts cultural events from Open Agenda API via OpenDataSoft endpoint.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from loguru import logger
from tqdm import tqdm


class OpenAgendaExtractor:
    """Extract cultural events from Open Agenda API."""
    
    def __init__(
        self,
        api_endpoint: str = "https://public.opendatasoft.com/api/records/1.0/search/",
        dataset: str = "evenements-publics-openagenda",
        rows_per_request: int = 100,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize Open Agenda API client.
        
        Args:
            api_endpoint: OpenDataSoft API endpoint
            dataset: Dataset identifier
            rows_per_request: Number of records per API call (max 100)
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries (seconds)
        """
        self.api_endpoint = api_endpoint
        self.dataset = dataset
        self.rows_per_request = min(rows_per_request, 100)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def extract_events(
        self,
        city: Optional[str] = "Paris",
        start_date: Optional[str] = None,
        max_events: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract events from Open Agenda API.
        
        Args:
            city: Filter by city (default: Paris)
            start_date: Filter events after this date (YYYY-MM-DD)
            max_events: Maximum number of events to extract
            save_path: Path to save raw JSON (optional)
            
        Returns:
            List of event dictionaries
        """
        # Calculate default start_date (12 months ago)
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
        logger.info(f"Extracting events from Open Agenda API")
        logger.info(f"Filters: city={city}, start_date>={start_date}")
        
        all_events = []
        offset = 0
        
        # First request to get total count
        params = self._build_params(city, start_date, offset=0)
        response_data = self._make_request(params)
        
        if response_data is None:
            logger.error("Failed to fetch data from API")
            return []
            
        total_count = response_data.get("nhits", 0)
        logger.info(f"Total events available: {total_count}")
        
        # Determine number of events to extract
        target_count = min(max_events, total_count) if max_events else total_count
        
        # Extract events with pagination
        with tqdm(total=target_count, desc="Extracting events") as pbar:
            while len(all_events) < target_count:
                params = self._build_params(city, start_date, offset=offset)
                response_data = self._make_request(params)
                
                if response_data is None:
                    break
                    
                records = response_data.get("records", [])
                if not records:
                    break
                    
                all_events.extend(records)
                pbar.update(len(records))
                
                offset += self.rows_per_request
                
                # Respect API rate limits
                time.sleep(0.5)
                
        logger.info(f"Extracted {len(all_events)} events")
        
        # Save raw data if path provided
        if save_path:
            self._save_events(all_events, save_path)
            
        return all_events
    
    def _build_params(
        self,
        city: Optional[str],
        start_date: str,
        offset: int
    ) -> Dict:
        """Build API request parameters."""
        params = {
            "dataset": self.dataset,
            "rows": self.rows_per_request,
            "start": offset,
            "sort": "-firstdate_begin"  # Sort by date descending
        }
        
        # Build query filter
        query_parts = []
        
        if city:
            query_parts.append(f'location_city:"{city}"')
            
        if start_date:
            query_parts.append(f'firstdate_begin>={start_date}')
            
        if query_parts:
            params["q"] = " AND ".join(query_parts)
            
        return params
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    self.api_endpoint,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All retry attempts failed")
                    return None
                    
    def _save_events(self, events: List[Dict], save_path: str):
        """Save events to JSON file."""
        logger.info(f"Saving {len(events)} events to {save_path}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        logger.info(f"Events saved successfully")


def main():
    """Example usage."""
    extractor = OpenAgendaExtractor()
    
    events = extractor.extract_events(
        city="Paris",
        max_events=3000,
        save_path="data/raw/events_raw.json"
    )
    
    print(f"\n✓ Extracted {len(events)} events")
    
    # Show sample event
    if events:
        sample = events[0]["fields"]
        print(f"\nSample event:")
        print(f"  Title: {sample.get('title', 'N/A')}")
        print(f"  Location: {sample.get('location', 'N/A')}")
        print(f"  Date: {sample.get('firstdate', 'N/A')}")
        print(f"  Categories: {sample.get('categories', [])}")


if __name__ == "__main__":
    main()
