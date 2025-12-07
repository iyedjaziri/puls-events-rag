"""
Test suite for API server.
"""

import pytest
from fastapi.testclient import TestClient
from scripts.api_server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test GET /health."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "rag_loaded" in data
    
    def test_health_returns_json(self, client):
        """Test health endpoint returns valid JSON."""
        response = client.get("/health")
        
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        
        assert isinstance(data, dict)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test GET /."""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "PULS Events RAG API"


class TestAskEndpoint:
    """Test ask endpoint."""
    
    def test_ask_valid_question(self, client):
        """Test POST /ask with valid question."""
        payload = {
            "question": "Quels concerts de jazz ce week-end?"
        }
        
        response = client.post("/ask", json=payload)
        
        # May return 503 if RAG not initialized in test
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "sources" in data
            assert "response_time_ms" in data
    
    def test_ask_question_too_short(self, client):
        """Test POST /ask with question too short."""
        payload = {
            "question": "Jazz?"
        }
        
        response = client.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_question_too_long(self, client):
        """Test POST /ask with question too long."""
        payload = {
            "question": "A" * 1000
        }
        
        response = client.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_empty_question(self, client):
        """Test POST /ask with empty question."""
        payload = {
            "question": "          "
        }
        
        response = client.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_missing_question(self, client):
        """Test POST /ask without question field."""
        payload = {}
        
        response = client.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_ask_with_top_k(self, client):
        """Test POST /ask with top_k parameter."""
        payload = {
            "question": "Concerts de jazz ce week-end?",
            "top_k": 10
        }
        
        response = client.post("/ask", json=payload)
        
        # May return 503 if RAG not initialized
        assert response.status_code in [200, 503]
    
    def test_ask_invalid_top_k(self, client):
        """Test POST /ask with invalid top_k."""
        payload = {
            "question": "Concerts de jazz?",
            "top_k": 100  # Too large
        }
        
        response = client.post("/ask", json=payload)
        
        assert response.status_code == 422  # Validation error


class TestStatsEndpoint:
    """Test stats endpoint."""
    
    def test_stats(self, client):
        """Test GET /stats."""
        response = client.get("/stats")
        
        # May return 503 if RAG not initialized
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "total_vectors" in data
            assert "embedding_dimension" in data


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")
        
        # CORS middleware should add headers
        assert "access-control-allow-origin" in response.headers


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation endpoints."""
    
    def test_openapi_json(self, client):
        """Test GET /openapi.json."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_docs_redirect(self, client):
        """Test GET /docs redirects to Swagger UI."""
        response = client.get("/docs", follow_redirects=False)
        
        # Should redirect or return docs page
        assert response.status_code in [200, 307]


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async behavior of endpoints."""
    
    async def test_concurrent_requests(self, client):
        """Test multiple concurrent requests."""
        import asyncio
        
        async def make_request():
            return client.get("/health")
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)


def test_error_handling(client):
    """Test API error handling."""
    # Invalid endpoint
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404
    
    # Invalid method
    response = client.get("/ask")
    assert response.status_code == 405  # Method not allowed
