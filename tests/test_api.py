#!/usr/bin/env python3
"""
API Test Suite for Multi-RAG System

This module contains tests for the Flask API endpoints and streaming functionality.
"""

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app, initialize_components
from config.settings import Config

class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock the global components
        with patch('api.app.initialize_components') as mock_init:
            mock_init.return_value = True
            
        # Mock global variables
        app.retriever = Mock()
        app.query_processor = Mock()
        app.llm_integrator = Mock()
        app.query_engine = Mock()
        app.streaming_handler = Mock()
    
    def test_health_check(self):
        """Test health check endpoint."""
        # Mock retriever statistics
        app.retriever.get_statistics.return_value = {
            'total_documents': 5,
            'total_elements': 100
        }
        
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('statistics', data)
        self.assertIn('timestamp', data)
    
    def test_health_check_uninitialized(self):
        """Test health check when system is not initialized."""
        # Temporarily set retriever to None
        original_retriever = app.retriever
        app.retriever = None
        
        try:
            response = self.client.get('/health')
            
            self.assertEqual(response.status_code, 503)
            
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'initializing')
            
        finally:
            app.retriever = original_retriever
    
    def test_upload_endpoint_no_file(self):
        """Test upload endpoint with no file."""
        response = self.client.post('/upload')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('No file provided', data['error'])
    
    def test_upload_endpoint_empty_filename(self):
        """Test upload endpoint with empty filename."""
        response = self.client.post('/upload', data={'file': (None, '')})
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('No file selected', data['error'])
    
    def test_upload_endpoint_invalid_file_type(self):
        """Test upload endpoint with invalid file type."""
        response = self.client.post('/upload', data={
            'file': (tempfile.NamedTemporaryFile(suffix='.txt'), 'test.txt')
        })
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Only PDF files are supported', data['error'])
    
    def test_upload_endpoint_success(self):
        """Test successful file upload."""
        # Mock successful document processing
        app.retriever.add_document.return_value = True
        
        # Create a temporary PDF-like file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b'%PDF-1.4 fake pdf content')
            tmp_file.flush()
            
            try:
                with open(tmp_file.name, 'rb') as f:
                    response = self.client.post('/upload', data={
                        'file': (f, 'test.pdf')
                    })
                
                self.assertEqual(response.status_code, 200)
                
                data = json.loads(response.data)
                self.assertIn('message', data)
                self.assertIn('filename', data)
                self.assertIn('successfully', data['message'])
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_query_endpoint_no_data(self):
        """Test query endpoint with no data."""
        response = self.client.post('/query')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Query is required', data['error'])
    
    def test_query_endpoint_empty_query(self):
        """Test query endpoint with empty query."""
        response = self.client.post('/query', 
                                  data=json.dumps({'query': ''}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_query_endpoint_success(self):
        """Test successful query processing."""
        # Mock query processing components
        app.query_processor.analyze_query.return_value = {
            'suggested_content_types': ['text']
        }
        
        # Mock retrieval results
        mock_element = Mock()
        mock_element.id = 'test_1'
        mock_element.content_type = 'text'
        mock_element.content = 'Test content'
        mock_element.source = 'test.pdf'
        mock_element.page_number = 1
        
        mock_result = Mock()
        mock_result.element = mock_element
        mock_result.score = 0.9
        
        app.retriever.retrieve.return_value = [mock_result]
        
        # Mock LLM response
        app.llm_integrator.generate_response.return_value = "Test response"
        
        response = self.client.post('/query',
                                  data=json.dumps({'query': 'test query'}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('response', data)
        self.assertIn('results', data)
        self.assertEqual(data['response'], 'Test response')
    
    def test_hybrid_search_endpoint(self):
        """Test hybrid search endpoint."""
        # Mock hybrid search results
        app.retriever.hybrid_search.return_value = {
            'text': [Mock()],
            'image': [],
            'table': []
        }
        
        response = self.client.post('/hybrid_search',
                                  data=json.dumps({'query': 'test query'}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertIn('text', data['results'])
        self.assertIn('image', data['results'])
        self.assertIn('table', data['results'])
    
    def test_statistics_endpoint(self):
        """Test statistics endpoint."""
        # Mock statistics
        app.retriever.get_statistics.return_value = {
            'total_documents': 10,
            'total_elements': 200,
            'content_type_distribution': {
                'text': 150,
                'image': 30,
                'table': 20
            }
        }
        
        response = self.client.get('/statistics')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('total_documents', data)
        self.assertIn('total_elements', data)
        self.assertIn('content_type_distribution', data)
    
    def test_clear_data_endpoint(self):
        """Test clear data endpoint."""
        app.retriever.clear_all_data.return_value = True
        
        response = self.client.post('/clear_data')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('cleared', data['message'])
    
    def test_chat_endpoint(self):
        """Test chat endpoint."""
        # Mock chat response
        app.llm_integrator.generate_chat_response.return_value = "Chat response"
        
        response = self.client.post('/chat',
                                  data=json.dumps({
                                      'message': 'Hello',
                                      'history': []
                                  }),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('response', data)
        self.assertEqual(data['response'], 'Chat response')

class TestStreamingEndpoints(unittest.TestCase):
    """Test streaming API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock streaming components
        app.streaming_handler = Mock()
        app.query_engine = Mock()
    
    def test_streaming_health_endpoint(self):
        """Test streaming health check endpoint."""
        response = self.client.get('/api/stream/health')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('components', data)
    
    def test_streaming_status_endpoint(self):
        """Test streaming status endpoint."""
        # Mock active streams
        app.streaming_handler.get_active_streams.return_value = {}
        app.query_engine.get_metrics.return_value = {
            'total_queries': 10,
            'successful_queries': 9
        }
        
        response = self.client.get('/api/stream/status')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('streaming_enabled', data)
        self.assertIn('active_streams', data)
        self.assertIn('query_engine_metrics', data)
    
    def test_cancel_stream_endpoint(self):
        """Test stream cancellation endpoint."""
        app.streaming_handler.cancel_stream.return_value = True
        
        response = self.client.post('/api/stream/cancel/test_stream_id')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('success', data)
        self.assertTrue(data['success'])

class TestAPIErrorHandling(unittest.TestCase):
    """Test API error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_404_error(self):
        """Test 404 error handling."""
        response = self.client.get('/nonexistent_endpoint')
        
        self.assertEqual(response.status_code, 404)
    
    def test_method_not_allowed(self):
        """Test method not allowed error."""
        response = self.client.get('/upload')  # Should be POST
        
        self.assertEqual(response.status_code, 405)
    
    def test_invalid_json(self):
        """Test invalid JSON handling."""
        response = self.client.post('/query',
                                  data='invalid json',
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)

class TestAPIPerformance(unittest.TestCase):
    """Test API performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Mock components for performance testing
        app.retriever = Mock()
        app.query_processor = Mock()
        app.llm_integrator = Mock()
        
        # Mock fast responses
        app.retriever.get_statistics.return_value = {'total_documents': 0}
        app.query_processor.analyze_query.return_value = {'suggested_content_types': ['text']}
        app.retriever.retrieve.return_value = []
        app.llm_integrator.generate_response.return_value = "Fast response"
    
    def test_health_check_performance(self):
        """Test health check response time."""
        start_time = time.time()
        
        response = self.client.get('/health')
        
        end_time = time.time()
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 1.0)  # Should respond within 1 second
    
    def test_query_performance(self):
        """Test query endpoint performance."""
        start_time = time.time()
        
        response = self.client.post('/query',
                                  data=json.dumps({'query': 'test query'}),
                                  content_type='application/json')
        
        end_time = time.time()
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 5.0)  # Should respond within 5 seconds
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = self.client.get('/health')
                results.put(response.status_code)
            except Exception as e:
                results.put(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        self.assertEqual(len(status_codes), 5)
        for status_code in status_codes:
            self.assertEqual(status_code, 200)
        
        # Should handle concurrent requests efficiently
        self.assertLess(total_time, 10.0)

class TestAPIIntegration(unittest.TestCase):
    """Integration tests for the API."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_full_workflow(self):
        """Test complete API workflow."""
        # This would test the full workflow from upload to query
        # For now, we'll test the structure
        
        # 1. Check health
        health_response = self.client.get('/health')
        self.assertIn(health_response.status_code, [200, 503])  # Either healthy or initializing
        
        # 2. Check statistics
        stats_response = self.client.get('/statistics')
        self.assertIn(stats_response.status_code, [200, 503])
        
        # 3. Test streaming health
        stream_health_response = self.client.get('/api/stream/health')
        self.assertIn(stream_health_response.status_code, [200, 503])

def run_api_tests():
    """Run all API tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAPIEndpoints,
        TestStreamingEndpoints,
        TestAPIErrorHandling,
        TestAPIPerformance,
        TestAPIIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == '__main__':
    # Run tests when script is executed directly
    result = run_api_tests()
    
    # Exit with appropriate code
    if result.wasSuccessful():
        print("\nAll API tests passed!")
        sys.exit(0)
    else:
        print(f"\nAPI tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)

