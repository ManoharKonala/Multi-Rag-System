import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from config.settings import Config
from .database import DatabaseManager
from .embeddings import EmbeddingGenerator, TableEmbeddingGenerator
from .document_processor import DocumentElement, DocumentProcessor, ImageProcessor, TableProcessor

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    element: DocumentElement
    score: float
    relevance_explanation: str

class MultiVectorRetriever:
    """Multi-vector retriever that handles different data modalities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.embedding_generator = EmbeddingGenerator(config)
        self.table_embedding_generator = TableEmbeddingGenerator(self.embedding_generator)
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.image_processor = ImageProcessor()
        self.table_processor = TableProcessor()
        
        # In-memory document store for quick access
        self.document_store = {}
        
    def initialize(self) -> bool:
        """Initialize the retriever and all its components."""
        try:
            # Initialize database connections
            if not self.db_manager.initialize():
                logger.error("Failed to initialize database connections")
                return False
            
            # Initialize embedding models
            if not self.embedding_generator.initialize_models():
                logger.error("Failed to initialize embedding models")
                return False
            
            logger.info("Successfully initialized MultiVectorRetriever")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MultiVectorRetriever: {e}")
            return False
    
    def add_document(self, file_path: str) -> bool:
        """Add a document to the retrieval system."""
        try:
            # Process the document
            elements = self.document_processor.process_pdf(file_path)
            
            if not elements:
                logger.warning(f"No elements extracted from document: {file_path}")
                return False
            
            # Process each element
            vector_data = []
            
            for element in elements:
                # Store element in document store
                self.document_store[element.id] = element
                
                # Generate embeddings based on content type
                embedding = None
                processed_content = element.content
                
                if element.content_type == 'text':
                    embedding = self.embedding_generator.generate_text_embedding(element.content)
                    
                elif element.content_type == 'image':
                    # Generate image description and embed it
                    description = self.image_processor.generate_image_description(element.content)
                    processed_content = description
                    embedding = self.embedding_generator.generate_text_embedding(description)
                    
                    # Update element metadata with description
                    element.metadata['description'] = description
                    
                elif element.content_type == 'table':
                    # Parse table data and generate summary embedding
                    table_data = json.loads(element.content)
                    summary = self.table_processor.generate_table_summary(table_data)
                    processed_content = summary
                    embedding = self.table_embedding_generator.generate_table_summary_embedding(table_data)
                    
                    # Store table in PostgreSQL
                    table_name = f"table_{element.id.replace('-', '_')}"
                    if 'data' in table_data and table_data['data']:
                        df = pd.DataFrame(table_data['data'])
                        self.db_manager.postgres.create_table_from_dataframe(df, table_name)
                        element.metadata['table_name'] = table_name
                    
                    # Update element metadata with summary
                    element.metadata['summary'] = summary
                
                if embedding:
                    vector_data.append({
                        'id': element.id,
                        'content': processed_content,
                        'content_type': element.content_type,
                        'source': element.source,
                        'metadata': json.dumps(element.metadata),
                        'embedding': embedding
                    })
            
            # Insert vectors into Milvus
            if vector_data:
                success = self.db_manager.milvus.insert_vectors(vector_data)
                if success:
                    logger.info(f"Successfully added document {file_path} with {len(vector_data)} elements")
                    return True
                else:
                    logger.error(f"Failed to insert vectors for document {file_path}")
                    return False
            else:
                logger.warning(f"No valid embeddings generated for document {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = None, content_types: List[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents based on a query."""
        try:
            if not top_k:
                top_k = self.config.TOP_K
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_text_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in Milvus
            search_results = self.db_manager.milvus.search_vectors(query_embedding, top_k * 2)  # Get more results for filtering
            
            # Filter by content types if specified
            if content_types:
                search_results = [r for r in search_results if r['content_type'] in content_types]
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_results[:top_k]:
                element_id = result['id']
                
                if element_id in self.document_store:
                    element = self.document_store[element_id]
                    
                    # Generate relevance explanation
                    explanation = self._generate_relevance_explanation(query, element, result['score'])
                    
                    retrieval_result = RetrievalResult(
                        element=element,
                        score=result['score'],
                        relevance_explanation=explanation
                    )
                    retrieval_results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(retrieval_results)} results for query: {query[:50]}...")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve results for query '{query}': {e}")
            return []
    
    def retrieve_by_content_type(self, query: str, content_type: str, top_k: int = None) -> List[RetrievalResult]:
        """Retrieve documents of a specific content type."""
        return self.retrieve(query, top_k, [content_type])
    
    def get_table_data(self, table_id: str) -> Optional[pd.DataFrame]:
        """Retrieve full table data from PostgreSQL."""
        try:
            if table_id in self.document_store:
                element = self.document_store[table_id]
                if element.content_type == 'table' and 'table_name' in element.metadata:
                    table_name = element.metadata['table_name']
                    return self.db_manager.postgres.get_table_data(table_name)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve table data for {table_id}: {e}")
            return None
    
    def hybrid_search(self, query: str, top_k: int = None) -> Dict[str, List[RetrievalResult]]:
        """Perform hybrid search across all content types."""
        try:
            if not top_k:
                top_k = self.config.TOP_K
            
            results = {
                'text': self.retrieve_by_content_type(query, 'text', top_k),
                'image': self.retrieve_by_content_type(query, 'image', top_k),
                'table': self.retrieve_by_content_type(query, 'table', top_k)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search for query '{query}': {e}")
            return {'text': [], 'image': [], 'table': []}
    
    def _generate_relevance_explanation(self, query: str, element: DocumentElement, score: float) -> str:
        """Generate an explanation for why this element is relevant to the query."""
        try:
            explanation_parts = []
            
            # Basic relevance score
            if score > 0.8:
                explanation_parts.append("High relevance")
            elif score > 0.6:
                explanation_parts.append("Medium relevance")
            else:
                explanation_parts.append("Low relevance")
            
            # Content type specific explanations
            if element.content_type == 'text':
                explanation_parts.append(f"text content from page {element.page_number}")
            elif element.content_type == 'image':
                explanation_parts.append(f"image from page {element.page_number}")
                if 'description' in element.metadata:
                    explanation_parts.append(f"described as: {element.metadata['description'][:100]}...")
            elif element.content_type == 'table':
                explanation_parts.append(f"table from page {element.page_number}")
                if 'summary' in element.metadata:
                    explanation_parts.append(f"containing: {element.metadata['summary'][:100]}...")
            
            # Source information
            explanation_parts.append(f"from {element.source}")
            
            return " - ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate relevance explanation: {e}")
            return f"Relevant content (score: {score:.2f})"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        try:
            stats = {
                'total_documents': len(set(element.source for element in self.document_store.values())),
                'total_elements': len(self.document_store),
                'content_type_distribution': {},
                'database_health': self.db_manager.health_check()
            }
            
            # Count elements by content type
            for element in self.document_store.values():
                content_type = element.content_type
                stats['content_type_distribution'][content_type] = stats['content_type_distribution'].get(content_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def clear_all_data(self) -> bool:
        """Clear all data from the retrieval system."""
        try:
            # Clear document store
            self.document_store.clear()
            
            # Clear Milvus collection
            self.db_manager.milvus.delete_collection()
            self.db_manager.milvus.create_collection()
            self.db_manager.milvus.load_collection()
            
            logger.info("Successfully cleared all data from retrieval system")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False
    
    def close(self):
        """Close all connections and clean up resources."""
        self.db_manager.close_connections()
        logger.info("Closed MultiVectorRetriever")

class QueryProcessor:
    """Processes and analyzes user queries to optimize retrieval."""
    
    def __init__(self, retriever: MultiVectorRetriever):
        self.retriever = retriever
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine the best retrieval strategy."""
        try:
            analysis = {
                'original_query': query,
                'query_type': self._classify_query_type(query),
                'suggested_content_types': self._suggest_content_types(query),
                'expanded_queries': self._expand_query(query),
                'confidence': 0.8  # Placeholder confidence score
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query '{query}': {e}")
            return {'original_query': query, 'query_type': 'general', 'suggested_content_types': ['text']}
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        # Image-related keywords
        if any(word in query_lower for word in ['image', 'picture', 'photo', 'chart', 'graph', 'diagram', 'figure']):
            return 'image_query'
        
        # Table-related keywords
        if any(word in query_lower for word in ['table', 'data', 'statistics', 'numbers', 'values', 'rows', 'columns']):
            return 'table_query'
        
        # General text query
        return 'text_query'
    
    def _suggest_content_types(self, query: str) -> List[str]:
        """Suggest which content types to search based on the query."""
        query_type = self._classify_query_type(query)
        
        if query_type == 'image_query':
            return ['image', 'text']  # Include text for image descriptions
        elif query_type == 'table_query':
            return ['table', 'text']  # Include text for table context
        else:
            return ['text', 'image', 'table']  # Search all types for general queries
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand the query with synonyms and related terms."""
        # This is a simplified implementation
        # In a real system, you might use word embeddings or a thesaurus
        
        expanded = [query]
        
        # Add some basic expansions
        query_lower = query.lower()
        
        # Synonym mapping (simplified)
        synonyms = {
            'chart': ['graph', 'diagram', 'figure'],
            'table': ['data', 'statistics', 'numbers'],
            'image': ['picture', 'photo', 'figure'],
            'performance': ['efficiency', 'speed', 'results'],
            'comparison': ['difference', 'versus', 'compare']
        }
        
        for word, syns in synonyms.items():
            if word in query_lower:
                for syn in syns:
                    expanded.append(query.replace(word, syn))
        
        return expanded[:3]  # Limit to 3 expanded queries

