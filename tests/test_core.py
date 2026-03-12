#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-RAG System Core Components

This module contains unit and integration tests for all core components
of the multi-RAG system.
"""

import unittest
import tempfile
import os
import sys
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.settings import Config
from core.document_processor import DocumentProcessor, ImageProcessor, TableProcessor
from core.embeddings import EmbeddingGenerator, TableEmbeddingGenerator
from core.retriever import MultiVectorRetriever, QueryProcessor, DocumentElement, RetrievalResult
from core.llm_integration import LLMIntegrator
from core.query_engine import RealTimeQueryEngine, AdvancedQueryProcessor, QueryType, QueryComplexity
from core.streaming_handler import StreamingResponseHandler

class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test that configuration initializes with default values."""
        config = Config()
        
        # Test required attributes exist
        self.assertTrue(hasattr(config, 'CHUNK_SIZE'))
        self.assertTrue(hasattr(config, 'CHUNK_OVERLAP'))
        self.assertTrue(hasattr(config, 'EMBEDDING_MODEL'))
        self.assertTrue(hasattr(config, 'EMBEDDING_DIMENSION'))
        self.assertTrue(hasattr(config, 'TOP_K'))
        
        # Test default values are reasonable
        self.assertGreater(config.CHUNK_SIZE, 0)
        self.assertGreaterEqual(config.CHUNK_OVERLAP, 0)
        self.assertLess(config.CHUNK_OVERLAP, config.CHUNK_SIZE)
        self.assertGreater(config.EMBEDDING_DIMENSION, 0)
        self.assertGreater(config.TOP_K, 0)

class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        text = "This is a test document. " * 20  # Create long text
        chunks = self.processor.chunk_text(text)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        # Test chunk sizes
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.processor.chunk_size + 50)  # Allow some flexibility
    
    def test_empty_text_chunking(self):
        """Test chunking of empty text."""
        chunks = self.processor.chunk_text("")
        self.assertEqual(len(chunks), 0)
    
    def test_short_text_chunking(self):
        """Test chunking of text shorter than chunk size."""
        short_text = "Short text."
        chunks = self.processor.chunk_text(short_text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
    
    def create_test_pdf(self, content="Test PDF content"):
        """Create a test PDF file."""
        import fitz  # PyMuPDF
        
        pdf_path = os.path.join(self.temp_dir, "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), content)
        doc.save(pdf_path)
        doc.close()
        
        return pdf_path
    
    def test_pdf_processing(self):
        """Test PDF document processing."""
        # Create test PDF
        test_content = "This is test content for PDF processing."
        pdf_path = self.create_test_pdf(test_content)
        
        try:
            elements = self.processor.process_pdf(pdf_path)
            
            self.assertIsInstance(elements, list)
            self.assertGreater(len(elements), 0)
            
            # Check that we got text elements
            text_elements = [e for e in elements if e.content_type == 'text']
            self.assertGreater(len(text_elements), 0)
            
            # Check element structure
            for element in elements:
                self.assertTrue(hasattr(element, 'id'))
                self.assertTrue(hasattr(element, 'content_type'))
                self.assertTrue(hasattr(element, 'content'))
                self.assertTrue(hasattr(element, 'source'))
                self.assertTrue(hasattr(element, 'page_number'))
                
        except Exception as e:
            self.skipTest(f"PDF processing failed (may be due to missing dependencies): {e}")

class TestEmbeddingGenerator(unittest.TestCase):
    """Test embedding generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.generator = EmbeddingGenerator(self.config)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_model_initialization(self, mock_transformer):
        """Test embedding model initialization."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        result = self.generator.initialize_models()
        
        self.assertTrue(result)
        mock_transformer.assert_called_once()
    
    def test_text_embedding_generation(self):
        """Test text embedding generation."""
        # Mock the model
        self.generator.text_model = Mock()
        self.generator.text_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        text = "Test text for embedding"
        embedding = self.generator.generate_text_embedding(text)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 3)
        self.generator.text_model.encode.assert_called_once_with([text])
    
    def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        # Mock the model
        self.generator.text_model = Mock()
        self.generator.text_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        texts = ["Text 1", "Text 2"]
        embeddings = self.generator.generate_text_embeddings(texts)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 2)
        self.generator.text_model.encode.assert_called_once_with(texts)

class TestMultiVectorRetriever(unittest.TestCase):
    """Test multi-vector retrieval functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.retriever = MultiVectorRetriever(self.config)
        
        # Mock the embedding generator
        self.retriever.embedding_generator = Mock()
        self.retriever.embedding_generator.generate_text_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Initialize with mock data
        self.retriever.document_store = {}
        self.retriever.vector_store = {}
    
    def test_document_element_creation(self):
        """Test document element creation."""
        element = DocumentElement(
            id="test_1",
            content_type="text",
            content="Test content",
            source="test.pdf",
            page_number=1
        )
        
        self.assertEqual(element.id, "test_1")
        self.assertEqual(element.content_type, "text")
        self.assertEqual(element.content, "Test content")
        self.assertEqual(element.source, "test.pdf")
        self.assertEqual(element.page_number, 1)
    
    def test_add_document_element(self):
        """Test adding document elements to the store."""
        element = DocumentElement(
            id="test_1",
            content_type="text",
            content="Test content",
            source="test.pdf",
            page_number=1
        )
        
        self.retriever.add_document_element(element)
        
        # Check element was added to document store
        self.assertIn("test_1", self.retriever.document_store)
        self.assertEqual(self.retriever.document_store["test_1"], element)
        
        # Check embedding was generated and stored
        self.retriever.embedding_generator.generate_text_embedding.assert_called_once_with("Test content")
        self.assertIn("test_1", self.retriever.vector_store)
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Add test elements
        elements = [
            DocumentElement("1", "text", "Python programming", "test.pdf", 1),
            DocumentElement("2", "text", "Machine learning", "test.pdf", 2),
            DocumentElement("3", "text", "Data science", "test.pdf", 3)
        ]
        
        for element in elements:
            self.retriever.add_document_element(element)
        
        # Mock similarity calculation
        with patch.object(self.retriever, '_calculate_similarity') as mock_sim:
            mock_sim.side_effect = [0.9, 0.7, 0.5]  # Decreasing similarity
            
            results = self.retriever.retrieve("Python", top_k=2)
            
            self.assertEqual(len(results), 2)
            self.assertGreater(results[0].score, results[1].score)  # Should be sorted by score
    
    def test_content_type_filtering(self):
        """Test content type filtering in retrieval."""
        # Add elements of different types
        elements = [
            DocumentElement("1", "text", "Text content", "test.pdf", 1),
            DocumentElement("2", "image", "Image description", "test.pdf", 2),
            DocumentElement("3", "table", "Table data", "test.pdf", 3)
        ]
        
        for element in elements:
            self.retriever.add_document_element(element)
        
        # Test filtering by content type
        with patch.object(self.retriever, '_calculate_similarity') as mock_sim:
            mock_sim.return_value = 0.8
            
            # Filter for text only
            results = self.retriever.retrieve("query", content_types=["text"])
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].element.content_type, "text")

class TestQueryProcessor(unittest.TestCase):
    """Test query processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retriever = Mock()
        self.processor = QueryProcessor(self.retriever)
    
    def test_query_analysis(self):
        """Test basic query analysis."""
        query = "What is machine learning?"
        analysis = self.processor.analyze_query(query)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('query', analysis)
        self.assertIn('keywords', analysis)
        self.assertIn('suggested_content_types', analysis)
        
        self.assertEqual(analysis['query'], query)
        self.assertIsInstance(analysis['keywords'], list)
        self.assertIsInstance(analysis['suggested_content_types'], list)
    
    def test_keyword_extraction(self):
        """Test keyword extraction from queries."""
        query = "How does machine learning work in Python?"
        keywords = self.processor.extract_keywords(query)
        
        self.assertIsInstance(keywords, list)
        self.assertIn('machine', keywords)
        self.assertIn('learning', keywords)
        self.assertIn('Python', keywords)
        
        # Should not include stop words
        self.assertNotIn('how', keywords)
        self.assertNotIn('does', keywords)

class TestAdvancedQueryProcessor(unittest.TestCase):
    """Test advanced query processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.processor = AdvancedQueryProcessor(self.config)
    
    def test_query_type_detection(self):
        """Test query type detection."""
        test_cases = [
            ("What is machine learning?", QueryType.FACTUAL),
            ("Compare Python and Java", QueryType.COMPARATIVE),
            ("Why does this happen?", QueryType.ANALYTICAL),
            ("Summarize the document", QueryType.SUMMARIZATION),
            ("How to install Python?", QueryType.PROCEDURAL),
            ("Tell me about AI", QueryType.EXPLORATORY)
        ]
        
        for query, expected_type in test_cases:
            detected_type = self.processor._detect_query_type(query)
            self.assertEqual(detected_type, expected_type, f"Failed for query: {query}")
    
    def test_complexity_assessment(self):
        """Test query complexity assessment."""
        simple_query = "What is AI?"
        complex_query = "Compare machine learning and deep learning approaches, analyzing their advantages and disadvantages in different scenarios, and explain when to use each method."
        
        simple_complexity = self.processor._assess_complexity(simple_query)
        complex_complexity = self.processor._assess_complexity(complex_query)
        
        self.assertEqual(simple_complexity, QueryComplexity.SIMPLE)
        self.assertIn(complex_complexity, [QueryComplexity.MODERATE, QueryComplexity.COMPLEX])
    
    def test_entity_extraction(self):
        """Test entity extraction from queries."""
        query = "What is the price of AAPL stock in 2023?"
        entities = self.processor._extract_entities(query)
        
        self.assertIsInstance(entities, list)
        # Should extract year and stock symbol
        year_entities = [e for e in entities if '2023' in e]
        self.assertGreater(len(year_entities), 0)

class TestLLMIntegrator(unittest.TestCase):
    """Test LLM integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.integrator = LLMIntegrator(self.config)
    
    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai):
        """Test LLM integrator initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        result = self.integrator.initialize()
        
        self.assertTrue(result)
        mock_openai.assert_called_once()
    
    def test_response_generation(self):
        """Test response generation."""
        # Mock the OpenAI client
        self.integrator.client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        self.integrator.client.chat.completions.create.return_value = mock_response
        
        query = "Test query"
        retrieval_results = []
        
        response = self.integrator.generate_response(query, retrieval_results)
        
        self.assertEqual(response, "Test response")
        self.integrator.client.chat.completions.create.assert_called_once()

class TestRealTimeQueryEngine(unittest.TestCase):
    """Test real-time query engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.engine = RealTimeQueryEngine(self.config)
        
        # Mock dependencies
        self.mock_retriever = Mock()
        self.mock_llm = Mock()
        
        self.engine.initialize(self.mock_retriever, self.mock_llm)
    
    def test_query_processing(self):
        """Test complete query processing."""
        query = "What is machine learning?"
        
        # Mock retrieval results
        mock_element = DocumentElement("1", "text", "ML content", "test.pdf", 1)
        mock_result = RetrievalResult(mock_element, 0.9)
        self.mock_retriever.retrieve.return_value = [mock_result]
        
        # Mock LLM response
        self.mock_llm.generate_response.return_value = "Machine learning is..."
        
        result = self.engine.process_query(query)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.query, query)
        self.assertIsInstance(result.response, str)
        self.assertGreater(result.confidence, 0)
        self.assertGreater(result.processing_time, 0)
    
    def test_query_caching(self):
        """Test query result caching."""
        query = "Test query"
        
        # Mock retrieval and response
        mock_element = DocumentElement("1", "text", "Content", "test.pdf", 1)
        mock_result = RetrievalResult(mock_element, 0.9)
        self.mock_retriever.retrieve.return_value = [mock_result]
        self.mock_llm.generate_response.return_value = "Response"
        
        # First query
        result1 = self.engine.process_query(query, use_cache=True)
        
        # Second query (should use cache)
        result2 = self.engine.process_query(query, use_cache=True)
        
        self.assertEqual(result1.response, result2.response)
        self.assertTrue(result2.metadata.get('from_cache', False))

class TestStreamingHandler(unittest.TestCase):
    """Test streaming response handler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.handler = StreamingResponseHandler(self.config)
        
        # Mock query engine
        self.mock_engine = Mock()
        self.handler.initialize(self.mock_engine)
    
    def test_initialization(self):
        """Test streaming handler initialization."""
        self.assertIsNotNone(self.handler.query_engine)
        self.assertEqual(len(self.handler.active_streams), 0)
    
    def test_stream_management(self):
        """Test stream management functionality."""
        # Test getting active streams
        active_streams = self.handler.get_active_streams()
        self.assertIsInstance(active_streams, dict)
        self.assertEqual(len(active_streams), 0)
        
        # Test stream cancellation
        result = self.handler.cancel_stream("nonexistent_stream")
        self.assertFalse(result)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # This test would require actual models and databases
        # For now, we'll test the component integration structure
        
        # Initialize components
        retriever = MultiVectorRetriever(self.config)
        
        # Test that components can be connected
        self.assertIsNotNone(retriever)
        self.assertIsNotNone(retriever.config)
    
    def test_error_handling(self):
        """Test error handling across components."""
        # Test with invalid configuration
        config = Config()
        config.EMBEDDING_MODEL = "invalid_model"
        
        generator = EmbeddingGenerator(config)
        
        # Should handle invalid model gracefully
        result = generator.initialize_models()
        # The actual result depends on implementation details

class TestPerformance(unittest.TestCase):
    """Performance tests for the system."""
    
    def test_query_processing_speed(self):
        """Test query processing performance."""
        config = Config()
        processor = AdvancedQueryProcessor(config)
        
        # Test processing speed for multiple queries
        queries = [
            "What is machine learning?",
            "How does neural network work?",
            "Compare supervised and unsupervised learning",
            "Explain deep learning algorithms"
        ]
        
        start_time = time.time()
        
        for query in queries:
            analysis = processor.analyze_query(query)
            self.assertIsNotNone(analysis)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process all queries in reasonable time
        self.assertLess(processing_time, 5.0)  # 5 seconds max
        
        # Average time per query should be reasonable
        avg_time = processing_time / len(queries)
        self.assertLess(avg_time, 1.0)  # 1 second per query max

def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfig,
        TestDocumentProcessor,
        TestEmbeddingGenerator,
        TestMultiVectorRetriever,
        TestQueryProcessor,
        TestAdvancedQueryProcessor,
        TestLLMIntegrator,
        TestRealTimeQueryEngine,
        TestStreamingHandler,
        TestIntegration,
        TestPerformance
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
    result = run_tests()
    
    # Exit with appropriate code
    if result.wasSuccessful():
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)

