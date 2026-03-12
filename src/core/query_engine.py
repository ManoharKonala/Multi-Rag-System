"""
Advanced Query Processing Engine for Multi-RAG System

This module provides sophisticated query processing capabilities including
query understanding, intent detection, and multi-step reasoning.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime

from .retriever import MultiVectorRetriever, RetrievalResult, QueryProcessor
from .llm_integration import LLMIntegrator
from config.settings import Config

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"

class QueryComplexity(Enum):
    """Complexity levels of queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class QueryAnalysis:
    """Analysis results for a user query."""
    original_query: str
    query_type: QueryType
    complexity: QueryComplexity
    intent: str
    entities: List[str]
    keywords: List[str]
    suggested_content_types: List[str]
    confidence: float
    processing_strategy: str
    sub_queries: List[str]

@dataclass
class QueryResult:
    """Complete result of query processing."""
    query: str
    analysis: QueryAnalysis
    retrieval_results: List[RetrievalResult]
    response: str
    confidence: float
    processing_time: float
    sources_used: List[str]
    reasoning_steps: List[str]
    metadata: Dict[str, Any]

class AdvancedQueryProcessor:
    """Advanced query processor with intent detection and multi-step reasoning."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Query patterns for intent detection
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what|who|when|where|which)\b',
                r'\b(define|definition|meaning)\b',
                r'\b(is|are|was|were)\b.*\?'
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|comparison|versus|vs|difference|different)\b',
                r'\b(better|worse|more|less|higher|lower)\b',
                r'\b(than|compared to)\b'
            ],
            QueryType.ANALYTICAL: [
                r'\b(analyze|analysis|why|how|explain|reason)\b',
                r'\b(cause|effect|impact|influence)\b',
                r'\b(trend|pattern|relationship)\b'
            ],
            QueryType.SUMMARIZATION: [
                r'\b(summarize|summary|overview|brief)\b',
                r'\b(main points|key points|highlights)\b',
                r'\b(in summary|overall)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|steps|process|procedure)\b',
                r'\b(guide|tutorial|instructions)\b',
                r'\b(method|approach|way to)\b'
            ],
            QueryType.EXPLORATORY: [
                r'\b(explore|investigation|research)\b',
                r'\b(tell me about|learn about)\b',
                r'\b(everything about|all about)\b'
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'numbers': r'\b\d+(?:\.\d+)?\b',
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
            'percentages': r'\b\d+(?:\.\d+)?%\b',
            'currencies': r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
            'organizations': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
            'technical_terms': r'\b[A-Z]{2,}\b'  # Acronyms
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Perform comprehensive analysis of a user query."""
        try:
            start_time = time.time()
            
            # Detect query type
            query_type = self._detect_query_type(query)
            
            # Assess complexity
            complexity = self._assess_complexity(query)
            
            # Extract intent
            intent = self._extract_intent(query, query_type)
            
            # Extract entities and keywords
            entities = self._extract_entities(query)
            keywords = self._extract_keywords(query)
            
            # Suggest content types
            suggested_content_types = self._suggest_content_types(query, query_type)
            
            # Determine processing strategy
            processing_strategy = self._determine_processing_strategy(query_type, complexity)
            
            # Generate sub-queries for complex queries
            sub_queries = self._generate_sub_queries(query, query_type, complexity)
            
            # Calculate confidence
            confidence = self._calculate_confidence(query, query_type, entities, keywords)
            
            analysis = QueryAnalysis(
                original_query=query,
                query_type=query_type,
                complexity=complexity,
                intent=intent,
                entities=entities,
                keywords=keywords,
                suggested_content_types=suggested_content_types,
                confidence=confidence,
                processing_strategy=processing_strategy,
                sub_queries=sub_queries
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Query analysis completed in {processing_time:.3f}s: {query_type.value}, {complexity.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query '{query}': {e}")
            # Return basic analysis as fallback
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.FACTUAL,
                complexity=QueryComplexity.SIMPLE,
                intent="general_inquiry",
                entities=[],
                keywords=query.split(),
                suggested_content_types=['text'],
                confidence=0.5,
                processing_strategy="basic_retrieval",
                sub_queries=[]
            )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query based on patterns."""
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score
        
        # Return the type with highest score, default to FACTUAL
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        else:
            return QueryType.FACTUAL
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess the complexity of a query."""
        # Factors that increase complexity
        complexity_indicators = {
            'length': len(query.split()),
            'questions': len(re.findall(r'\?', query)),
            'conjunctions': len(re.findall(r'\b(and|or|but|however|moreover|furthermore)\b', query.lower())),
            'comparatives': len(re.findall(r'\b(more|less|better|worse|compared|versus)\b', query.lower())),
            'conditionals': len(re.findall(r'\b(if|when|unless|provided|assuming)\b', query.lower())),
            'quantifiers': len(re.findall(r'\b(all|some|many|few|most|several)\b', query.lower()))
        }
        
        # Calculate complexity score
        score = 0
        score += min(complexity_indicators['length'] / 10, 3)  # Length factor (max 3)
        score += complexity_indicators['questions'] * 0.5
        score += complexity_indicators['conjunctions'] * 1
        score += complexity_indicators['comparatives'] * 1.5
        score += complexity_indicators['conditionals'] * 2
        score += complexity_indicators['quantifiers'] * 0.5
        
        if score <= 2:
            return QueryComplexity.SIMPLE
        elif score <= 5:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _extract_intent(self, query: str, query_type: QueryType) -> str:
        """Extract the specific intent from the query."""
        query_lower = query.lower()
        
        # Intent mapping based on query type and keywords
        intent_keywords = {
            'find_information': ['find', 'search', 'look for', 'locate'],
            'get_definition': ['define', 'definition', 'meaning', 'what is'],
            'compare_items': ['compare', 'difference', 'versus', 'vs'],
            'analyze_data': ['analyze', 'analysis', 'examine', 'study'],
            'summarize_content': ['summarize', 'summary', 'overview'],
            'get_instructions': ['how to', 'steps', 'guide', 'tutorial'],
            'explore_topic': ['explore', 'learn about', 'tell me about']
        }
        
        # Find matching intent
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        # Default intent based on query type
        default_intents = {
            QueryType.FACTUAL: 'find_information',
            QueryType.COMPARATIVE: 'compare_items',
            QueryType.ANALYTICAL: 'analyze_data',
            QueryType.SUMMARIZATION: 'summarize_content',
            QueryType.PROCEDURAL: 'get_instructions',
            QueryType.EXPLORATORY: 'explore_topic'
        }
        
        return default_intents.get(query_type, 'find_information')
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from the query."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append(f"{entity_type}:{match}")
        
        # Extract potential proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(proper_nouns)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _suggest_content_types(self, query: str, query_type: QueryType) -> List[str]:
        """Suggest which content types to search based on query analysis."""
        query_lower = query.lower()
        
        # Keywords that suggest specific content types
        image_keywords = ['chart', 'graph', 'diagram', 'figure', 'image', 'picture', 'visual', 'plot']
        table_keywords = ['table', 'data', 'statistics', 'numbers', 'values', 'rows', 'columns']
        
        suggested_types = ['text']  # Always include text
        
        # Check for image-related keywords
        if any(keyword in query_lower for keyword in image_keywords):
            suggested_types.append('image')
        
        # Check for table-related keywords
        if any(keyword in query_lower for keyword in table_keywords):
            suggested_types.append('table')
        
        # Query type specific suggestions
        if query_type == QueryType.COMPARATIVE:
            suggested_types.extend(['table', 'image'])
        elif query_type == QueryType.ANALYTICAL:
            suggested_types.extend(['table', 'image'])
        
        return list(set(suggested_types))  # Remove duplicates
    
    def _determine_processing_strategy(self, query_type: QueryType, complexity: QueryComplexity) -> str:
        """Determine the best processing strategy for the query."""
        if complexity == QueryComplexity.SIMPLE:
            return "basic_retrieval"
        elif complexity == QueryComplexity.MODERATE:
            if query_type in [QueryType.COMPARATIVE, QueryType.ANALYTICAL]:
                return "multi_step_retrieval"
            else:
                return "enhanced_retrieval"
        else:  # COMPLEX
            return "advanced_reasoning"
    
    def _generate_sub_queries(self, query: str, query_type: QueryType, complexity: QueryComplexity) -> List[str]:
        """Generate sub-queries for complex queries."""
        if complexity == QueryComplexity.SIMPLE:
            return []
        
        sub_queries = []
        
        # For comparative queries, break down into individual comparisons
        if query_type == QueryType.COMPARATIVE:
            # Extract items being compared
            comparison_words = re.findall(r'\b\w+\b(?=\s+(?:vs|versus|compared to|and))', query.lower())
            if len(comparison_words) >= 2:
                for word in comparison_words:
                    sub_queries.append(f"What is {word}?")
                    sub_queries.append(f"Characteristics of {word}")
        
        # For analytical queries, break down into components
        elif query_type == QueryType.ANALYTICAL:
            if 'why' in query.lower():
                sub_queries.append(query.replace('why', 'what causes'))
                sub_queries.append(query.replace('why', 'what factors influence'))
            elif 'how' in query.lower():
                sub_queries.append(query.replace('how', 'what process'))
                sub_queries.append(query.replace('how', 'what steps'))
        
        # For exploratory queries, generate related questions
        elif query_type == QueryType.EXPLORATORY:
            main_topic = self._extract_main_topic(query)
            if main_topic:
                sub_queries.extend([
                    f"What is {main_topic}?",
                    f"How does {main_topic} work?",
                    f"Applications of {main_topic}",
                    f"Benefits and limitations of {main_topic}"
                ])
        
        return sub_queries[:5]  # Limit to 5 sub-queries
    
    def _extract_main_topic(self, query: str) -> str:
        """Extract the main topic from a query."""
        # Simple heuristic: find the longest noun phrase
        words = query.split()
        
        # Look for patterns like "about X", "of X", etc.
        topic_indicators = ['about', 'of', 'regarding', 'concerning']
        for i, word in enumerate(words):
            if word.lower() in topic_indicators and i + 1 < len(words):
                return ' '.join(words[i+1:i+3])  # Take next 1-2 words
        
        # Fallback: return the last few meaningful words
        meaningful_words = [w for w in words if len(w) > 3 and w.lower() not in ['what', 'how', 'when', 'where', 'why']]
        if meaningful_words:
            return ' '.join(meaningful_words[-2:])
        
        return ""
    
    def _calculate_confidence(self, query: str, query_type: QueryType, entities: List[str], keywords: List[str]) -> float:
        """Calculate confidence in the query analysis."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on clear indicators
        if len(keywords) > 0:
            confidence += 0.1
        
        if len(entities) > 0:
            confidence += 0.1
        
        # Query type specific confidence adjustments
        query_lower = query.lower()
        type_indicators = self.query_patterns.get(query_type, [])
        
        for pattern in type_indicators:
            if re.search(pattern, query_lower):
                confidence += 0.1
                break
        
        # Length and structure indicators
        if 5 <= len(query.split()) <= 20:  # Optimal length
            confidence += 0.1
        
        if query.endswith('?'):  # Clear question format
            confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0

class RealTimeQueryEngine:
    """Real-time query processing engine with streaming capabilities."""
    
    def __init__(self, config: Config):
        self.config = config
        self.retriever = None
        self.llm_integrator = None
        self.query_processor = None
        self.advanced_processor = None
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_max_size = 100
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_processing_time': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }
    
    def initialize(self, retriever: MultiVectorRetriever, llm_integrator: LLMIntegrator) -> bool:
        """Initialize the query engine with required components."""
        try:
            self.retriever = retriever
            self.llm_integrator = llm_integrator
            self.query_processor = QueryProcessor(retriever)
            self.advanced_processor = AdvancedQueryProcessor(self.config)
            
            logger.info("Real-time query engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            return False
    
    def process_query(self, query: str, use_cache: bool = True, stream_response: bool = False) -> QueryResult:
        """Process a query with advanced analysis and retrieval."""
        try:
            start_time = time.time()
            self.metrics['total_queries'] += 1
            
            # Check cache first
            if use_cache and query in self.query_cache:
                self.metrics['cache_hits'] += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                cached_result = self.query_cache[query]
                cached_result.metadata['from_cache'] = True
                return cached_result
            
            # Analyze the query
            analysis = self.advanced_processor.analyze_query(query)
            
            # Retrieve relevant documents
            retrieval_results = self._perform_retrieval(analysis)
            
            # Generate response
            if stream_response:
                response = self._generate_streaming_response(analysis, retrieval_results)
            else:
                response = self._generate_response(analysis, retrieval_results)
            
            # Calculate confidence and extract reasoning
            confidence = self._calculate_response_confidence(analysis, retrieval_results, response)
            reasoning_steps = self._extract_reasoning_steps(analysis, retrieval_results)
            sources_used = [result.element.source for result in retrieval_results]
            
            # Create result
            processing_time = time.time() - start_time
            result = QueryResult(
                query=query,
                analysis=analysis,
                retrieval_results=retrieval_results,
                response=response,
                confidence=confidence,
                processing_time=processing_time,
                sources_used=list(set(sources_used)),
                reasoning_steps=reasoning_steps,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'from_cache': False,
                    'processing_strategy': analysis.processing_strategy
                }
            )
            
            # Cache the result
            if use_cache:
                self._cache_result(query, result)
            
            # Update metrics
            self.metrics['successful_queries'] += 1
            self._update_avg_processing_time(processing_time)
            
            logger.info(f"Query processed successfully in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.metrics['failed_queries'] += 1
            logger.error(f"Failed to process query '{query}': {e}")
            
            # Return error result
            return QueryResult(
                query=query,
                analysis=self.advanced_processor.analyze_query(query),
                retrieval_results=[],
                response=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time,
                sources_used=[],
                reasoning_steps=[],
                metadata={'error': str(e)}
            )
    
    def _perform_retrieval(self, analysis: QueryAnalysis) -> List[RetrievalResult]:
        """Perform retrieval based on query analysis."""
        try:
            if analysis.processing_strategy == "basic_retrieval":
                return self.retriever.retrieve(
                    analysis.original_query,
                    top_k=self.config.TOP_K,
                    content_types=analysis.suggested_content_types
                )
            
            elif analysis.processing_strategy == "multi_step_retrieval":
                # Perform multiple retrievals and combine results
                all_results = []
                
                # Main query
                main_results = self.retriever.retrieve(
                    analysis.original_query,
                    top_k=self.config.TOP_K // 2,
                    content_types=analysis.suggested_content_types
                )
                all_results.extend(main_results)
                
                # Sub-queries
                for sub_query in analysis.sub_queries[:2]:  # Limit to 2 sub-queries
                    sub_results = self.retriever.retrieve(
                        sub_query,
                        top_k=2,
                        content_types=analysis.suggested_content_types
                    )
                    all_results.extend(sub_results)
                
                # Remove duplicates and sort by score
                unique_results = {}
                for result in all_results:
                    if result.element.id not in unique_results:
                        unique_results[result.element.id] = result
                    elif result.score > unique_results[result.element.id].score:
                        unique_results[result.element.id] = result
                
                sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
                return sorted_results[:self.config.TOP_K]
            
            elif analysis.processing_strategy == "advanced_reasoning":
                # Use hybrid search for complex queries
                hybrid_results = self.retriever.hybrid_search(analysis.original_query, self.config.TOP_K)
                
                # Flatten and combine results from all content types
                all_results = []
                for content_type, results in hybrid_results.items():
                    all_results.extend(results)
                
                # Sort by score and return top results
                sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
                return sorted_results[:self.config.TOP_K]
            
            else:
                # Default to enhanced retrieval
                return self.retriever.retrieve(
                    analysis.original_query,
                    top_k=self.config.TOP_K,
                    content_types=analysis.suggested_content_types
                )
                
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def _generate_response(self, analysis: QueryAnalysis, retrieval_results: List[RetrievalResult]) -> str:
        """Generate a comprehensive response based on analysis and retrieval results."""
        try:
            if not retrieval_results:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or uploading more documents."
            
            # Use LLM integrator to generate response
            response = self.llm_integrator.generate_response(analysis.original_query, retrieval_results)
            
            # Enhance response based on query type
            if analysis.query_type == QueryType.COMPARATIVE:
                response = self._enhance_comparative_response(response, analysis, retrieval_results)
            elif analysis.query_type == QueryType.ANALYTICAL:
                response = self._enhance_analytical_response(response, analysis, retrieval_results)
            elif analysis.query_type == QueryType.SUMMARIZATION:
                response = self._enhance_summary_response(response, analysis, retrieval_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while generating the response. Please try again."
    
    def _generate_streaming_response(self, analysis: QueryAnalysis, retrieval_results: List[RetrievalResult]) -> str:
        """Generate a streaming response (placeholder for future implementation)."""
        # For now, return the regular response
        # In a full implementation, this would yield chunks of the response
        return self._generate_response(analysis, retrieval_results)
    
    def _enhance_comparative_response(self, response: str, analysis: QueryAnalysis, results: List[RetrievalResult]) -> str:
        """Enhance response for comparative queries."""
        # Add comparison structure if not present
        if "comparison" not in response.lower() and len(results) > 1:
            comparison_intro = "\n\nComparison Summary:\n"
            response += comparison_intro
        
        return response
    
    def _enhance_analytical_response(self, response: str, analysis: QueryAnalysis, results: List[RetrievalResult]) -> str:
        """Enhance response for analytical queries."""
        # Add analytical structure
        if "analysis" not in response.lower():
            analysis_intro = "\n\nAnalysis:\n"
            response += analysis_intro
        
        return response
    
    def _enhance_summary_response(self, response: str, analysis: QueryAnalysis, results: List[RetrievalResult]) -> str:
        """Enhance response for summarization queries."""
        # Ensure summary format
        if not response.startswith("Summary:") and not response.startswith("In summary"):
            response = "Summary:\n" + response
        
        return response
    
    def _calculate_response_confidence(self, analysis: QueryAnalysis, results: List[RetrievalResult], response: str) -> float:
        """Calculate confidence in the generated response."""
        confidence = analysis.confidence * 0.4  # Base from query analysis
        
        # Add confidence from retrieval results
        if results:
            avg_retrieval_score = sum(result.score for result in results) / len(results)
            confidence += avg_retrieval_score * 0.4
        
        # Add confidence from response quality
        response_quality = self._assess_response_quality(response)
        confidence += response_quality * 0.2
        
        return min(confidence, 1.0)
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess the quality of a generated response."""
        quality = 0.5  # Base quality
        
        # Length check
        if 50 <= len(response) <= 1000:
            quality += 0.2
        
        # Structure check
        if any(indicator in response.lower() for indicator in ['based on', 'according to', 'the document']):
            quality += 0.2
        
        # Completeness check
        if response.endswith('.') or response.endswith('!'):
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _extract_reasoning_steps(self, analysis: QueryAnalysis, results: List[RetrievalResult]) -> List[str]:
        """Extract reasoning steps from the query processing."""
        steps = []
        
        steps.append(f"1. Analyzed query as {analysis.query_type.value} with {analysis.complexity.value} complexity")
        steps.append(f"2. Identified intent: {analysis.intent}")
        
        if analysis.entities:
            steps.append(f"3. Extracted entities: {', '.join(analysis.entities[:3])}")
        
        if results:
            steps.append(f"4. Retrieved {len(results)} relevant documents")
            steps.append(f"5. Used {analysis.processing_strategy} strategy")
        
        steps.append("6. Generated response based on retrieved information")
        
        return steps
    
    def _cache_result(self, query: str, result: QueryResult):
        """Cache a query result."""
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query] = result
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time metric."""
        current_avg = self.metrics['avg_processing_time']
        total_queries = self.metrics['total_queries']
        
        if total_queries == 1:
            self.metrics['avg_processing_time'] = processing_time
        else:
            self.metrics['avg_processing_time'] = (current_avg * (total_queries - 1) + processing_time) / total_queries
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get query engine performance metrics."""
        return self.metrics.copy()
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")

