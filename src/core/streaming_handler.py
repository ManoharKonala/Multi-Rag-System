"""
Streaming Response Handler for Multi-RAG System

This module provides real-time streaming capabilities for query responses,
allowing for progressive response generation and real-time user feedback.
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue, Empty
from datetime import datetime

from .query_engine import RealTimeQueryEngine, QueryResult, QueryAnalysis
from .retriever import RetrievalResult
from config.settings import Config

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """Types of streaming events."""
    QUERY_RECEIVED = "query_received"
    ANALYSIS_COMPLETE = "analysis_complete"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_PROGRESS = "retrieval_progress"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    RESPONSE_STARTED = "response_started"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_COMPLETE = "response_complete"
    ERROR = "error"
    METADATA = "metadata"

@dataclass
class StreamEvent:
    """A single streaming event."""
    event_type: StreamEventType
    timestamp: str
    data: Dict[str, Any]
    sequence_id: int

class StreamingResponseHandler:
    """Handles streaming responses for real-time query processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.query_engine = None
        self.active_streams = {}
        self.stream_counter = 0
        self.lock = threading.Lock()
        
        # Streaming configuration
        self.chunk_size = 50  # Characters per chunk
        self.chunk_delay = 0.05  # Seconds between chunks
        self.max_concurrent_streams = 10
        
    def initialize(self, query_engine: RealTimeQueryEngine) -> bool:
        """Initialize the streaming handler."""
        try:
            self.query_engine = query_engine
            logger.info("Streaming response handler initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize streaming handler: {e}")
            return False
    
    async def stream_query_response(self, query: str, stream_id: Optional[str] = None) -> AsyncGenerator[StreamEvent, None]:
        """Stream a query response with real-time updates."""
        if not stream_id:
            with self.lock:
                self.stream_counter += 1
                stream_id = f"stream_{self.stream_counter}"
        
        try:
            # Check concurrent stream limit
            if len(self.active_streams) >= self.max_concurrent_streams:
                yield StreamEvent(
                    event_type=StreamEventType.ERROR,
                    timestamp=datetime.now().isoformat(),
                    data={"error": "Maximum concurrent streams reached"},
                    sequence_id=0
                )
                return
            
            # Register stream
            self.active_streams[stream_id] = {
                'start_time': time.time(),
                'query': query,
                'status': 'active'
            }
            
            sequence_id = 0
            
            # Event 1: Query received
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.QUERY_RECEIVED,
                timestamp=datetime.now().isoformat(),
                data={"query": query, "stream_id": stream_id},
                sequence_id=sequence_id
            )
            
            # Event 2: Start analysis
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.ANALYSIS_COMPLETE,
                timestamp=datetime.now().isoformat(),
                data={"status": "analyzing_query"},
                sequence_id=sequence_id
            )
            
            # Perform query analysis
            analysis = self.query_engine.advanced_processor.analyze_query(query)
            
            # Event 3: Analysis complete
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.ANALYSIS_COMPLETE,
                timestamp=datetime.now().isoformat(),
                data={
                    "analysis": {
                        "query_type": analysis.query_type.value,
                        "complexity": analysis.complexity.value,
                        "intent": analysis.intent,
                        "confidence": analysis.confidence,
                        "processing_strategy": analysis.processing_strategy
                    }
                },
                sequence_id=sequence_id
            )
            
            # Event 4: Start retrieval
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.RETRIEVAL_STARTED,
                timestamp=datetime.now().isoformat(),
                data={"content_types": analysis.suggested_content_types},
                sequence_id=sequence_id
            )
            
            # Perform retrieval with progress updates
            retrieval_results = []
            async for progress_event in self._stream_retrieval(analysis, sequence_id):
                sequence_id = progress_event.sequence_id
                if progress_event.event_type == StreamEventType.RETRIEVAL_COMPLETE:
                    retrieval_results = progress_event.data.get("results", [])
                yield progress_event
            
            # Event: Start response generation
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.RESPONSE_STARTED,
                timestamp=datetime.now().isoformat(),
                data={"sources_count": len(retrieval_results)},
                sequence_id=sequence_id
            )
            
            # Generate and stream response
            full_response = ""
            async for response_event in self._stream_response_generation(analysis, retrieval_results, sequence_id):
                sequence_id = response_event.sequence_id
                if response_event.event_type == StreamEventType.RESPONSE_CHUNK:
                    full_response += response_event.data.get("chunk", "")
                yield response_event
            
            # Event: Response complete with metadata
            sequence_id += 1
            processing_time = time.time() - self.active_streams[stream_id]['start_time']
            
            yield StreamEvent(
                event_type=StreamEventType.RESPONSE_COMPLETE,
                timestamp=datetime.now().isoformat(),
                data={
                    "full_response": full_response,
                    "processing_time": processing_time,
                    "sources_used": [result.element.source for result in retrieval_results],
                    "confidence": self._calculate_final_confidence(analysis, retrieval_results, full_response)
                },
                sequence_id=sequence_id
            )
            
        except Exception as e:
            logger.error(f"Error in streaming query response: {e}")
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                timestamp=datetime.now().isoformat(),
                data={"error": str(e)},
                sequence_id=sequence_id + 1
            )
        
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def _stream_retrieval(self, analysis: QueryAnalysis, start_sequence_id: int) -> AsyncGenerator[StreamEvent, None]:
        """Stream retrieval process with progress updates."""
        sequence_id = start_sequence_id
        
        try:
            # Simulate retrieval progress (in real implementation, this would be actual progress)
            total_steps = len(analysis.suggested_content_types)
            
            for i, content_type in enumerate(analysis.suggested_content_types):
                sequence_id += 1
                yield StreamEvent(
                    event_type=StreamEventType.RETRIEVAL_PROGRESS,
                    timestamp=datetime.now().isoformat(),
                    data={
                        "step": i + 1,
                        "total_steps": total_steps,
                        "current_content_type": content_type,
                        "progress_percentage": ((i + 1) / total_steps) * 100
                    },
                    sequence_id=sequence_id
                )
                
                # Small delay to simulate processing
                await asyncio.sleep(0.1)
            
            # Perform actual retrieval
            retrieval_results = self.query_engine._perform_retrieval(analysis)
            
            # Retrieval complete
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.RETRIEVAL_COMPLETE,
                timestamp=datetime.now().isoformat(),
                data={
                    "results_count": len(retrieval_results),
                    "results": retrieval_results,
                    "avg_score": sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
                },
                sequence_id=sequence_id
            )
            
        except Exception as e:
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                timestamp=datetime.now().isoformat(),
                data={"error": f"Retrieval error: {str(e)}"},
                sequence_id=sequence_id
            )
    
    async def _stream_response_generation(self, analysis: QueryAnalysis, retrieval_results: List[RetrievalResult], 
                                        start_sequence_id: int) -> AsyncGenerator[StreamEvent, None]:
        """Stream response generation with chunked output."""
        sequence_id = start_sequence_id
        
        try:
            # Generate full response
            full_response = self.query_engine._generate_response(analysis, retrieval_results)
            
            # Stream response in chunks
            for i in range(0, len(full_response), self.chunk_size):
                chunk = full_response[i:i + self.chunk_size]
                
                sequence_id += 1
                yield StreamEvent(
                    event_type=StreamEventType.RESPONSE_CHUNK,
                    timestamp=datetime.now().isoformat(),
                    data={
                        "chunk": chunk,
                        "chunk_index": i // self.chunk_size,
                        "total_chunks": (len(full_response) + self.chunk_size - 1) // self.chunk_size,
                        "progress_percentage": ((i + len(chunk)) / len(full_response)) * 100
                    },
                    sequence_id=sequence_id
                )
                
                # Delay between chunks for streaming effect
                await asyncio.sleep(self.chunk_delay)
            
        except Exception as e:
            sequence_id += 1
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                timestamp=datetime.now().isoformat(),
                data={"error": f"Response generation error: {str(e)}"},
                sequence_id=sequence_id
            )
    
    def _calculate_final_confidence(self, analysis: QueryAnalysis, results: List[RetrievalResult], response: str) -> float:
        """Calculate final confidence for the complete response."""
        return self.query_engine._calculate_response_confidence(analysis, results, response)
    
    def get_active_streams(self) -> Dict[str, Any]:
        """Get information about currently active streams."""
        with self.lock:
            return {
                stream_id: {
                    'query': info['query'],
                    'status': info['status'],
                    'duration': time.time() - info['start_time']
                }
                for stream_id, info in self.active_streams.items()
            }
    
    def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream."""
        with self.lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id]['status'] = 'cancelled'
                return True
            return False

class WebSocketStreamHandler:
    """Handles WebSocket streaming for real-time communication."""
    
    def __init__(self, streaming_handler: StreamingResponseHandler):
        self.streaming_handler = streaming_handler
        self.connections = {}
        self.connection_counter = 0
        
    async def handle_websocket_connection(self, websocket, path):
        """Handle a new WebSocket connection."""
        connection_id = f"ws_{self.connection_counter}"
        self.connection_counter += 1
        
        self.connections[connection_id] = {
            'websocket': websocket,
            'connected_at': datetime.now(),
            'queries_processed': 0
        }
        
        try:
            logger.info(f"New WebSocket connection: {connection_id}")
            
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'connection_id': connection_id,
                'timestamp': datetime.now().isoformat()
            }))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(websocket, connection_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': 'Invalid JSON message'
                    }))
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': str(e)
                    }))
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        
        finally:
            # Clean up connection
            if connection_id in self.connections:
                del self.connections[connection_id]
            logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def _handle_websocket_message(self, websocket, connection_id: str, data: Dict[str, Any]):
        """Handle a message from WebSocket client."""
        message_type = data.get('type')
        
        if message_type == 'query':
            query = data.get('query', '').strip()
            if not query:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'error': 'Empty query'
                }))
                return
            
            # Process query with streaming
            stream_id = f"{connection_id}_query_{self.connections[connection_id]['queries_processed']}"
            self.connections[connection_id]['queries_processed'] += 1
            
            async for event in self.streaming_handler.stream_query_response(query, stream_id):
                # Convert event to WebSocket message
                message = {
                    'type': 'stream_event',
                    'stream_id': stream_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp,
                    'data': event.data,
                    'sequence_id': event.sequence_id
                }
                
                await websocket.send(json.dumps(message, default=str))
        
        elif message_type == 'cancel_stream':
            stream_id = data.get('stream_id')
            if stream_id:
                success = self.streaming_handler.cancel_stream(stream_id)
                await websocket.send(json.dumps({
                    'type': 'stream_cancelled',
                    'stream_id': stream_id,
                    'success': success
                }))
        
        elif message_type == 'get_active_streams':
            active_streams = self.streaming_handler.get_active_streams()
            await websocket.send(json.dumps({
                'type': 'active_streams',
                'streams': active_streams
            }))
        
        else:
            await websocket.send(json.dumps({
                'type': 'error',
                'error': f'Unknown message type: {message_type}'
            }))

class ServerSentEventsHandler:
    """Handles Server-Sent Events for streaming responses."""
    
    def __init__(self, streaming_handler: StreamingResponseHandler):
        self.streaming_handler = streaming_handler
    
    async def handle_sse_request(self, query: str) -> AsyncGenerator[str, None]:
        """Handle Server-Sent Events request for streaming query response."""
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            # Stream query response
            async for event in self.streaming_handler.stream_query_response(query):
                # Format as SSE
                event_data = {
                    'type': 'stream_event',
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp,
                    'data': event.data,
                    'sequence_id': event.sequence_id
                }
                
                yield f"data: {json.dumps(event_data, default=str)}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'stream_complete', 'timestamp': datetime.now().isoformat()})}\n\n"
            
        except Exception as e:
            logger.error(f"SSE streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

class BatchStreamProcessor:
    """Processes multiple queries in batch with streaming updates."""
    
    def __init__(self, streaming_handler: StreamingResponseHandler):
        self.streaming_handler = streaming_handler
        self.batch_counter = 0
    
    async def process_batch_queries(self, queries: List[str], max_concurrent: int = 3) -> AsyncGenerator[Dict[str, Any], None]:
        """Process multiple queries concurrently with streaming updates."""
        self.batch_counter += 1
        batch_id = f"batch_{self.batch_counter}"
        
        try:
            # Send batch start event
            yield {
                'type': 'batch_started',
                'batch_id': batch_id,
                'total_queries': len(queries),
                'timestamp': datetime.now().isoformat()
            }
            
            # Process queries with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_query(query_index: int, query: str):
                async with semaphore:
                    query_id = f"{batch_id}_query_{query_index}"
                    
                    # Yield query start event
                    yield {
                        'type': 'query_started',
                        'batch_id': batch_id,
                        'query_index': query_index,
                        'query': query,
                        'query_id': query_id
                    }
                    
                    # Stream query processing
                    async for event in self.streaming_handler.stream_query_response(query, query_id):
                        yield {
                            'type': 'query_event',
                            'batch_id': batch_id,
                            'query_index': query_index,
                            'query_id': query_id,
                            'event': asdict(event)
                        }
            
            # Create tasks for all queries
            tasks = [
                process_single_query(i, query)
                for i, query in enumerate(queries)
            ]
            
            # Process all tasks concurrently
            completed_queries = 0
            async for task in asyncio.as_completed(tasks):
                async for result in task:
                    yield result
                    
                    if result.get('type') == 'query_event' and \
                       result.get('event', {}).get('event_type') == 'response_complete':
                        completed_queries += 1
                        
                        # Send progress update
                        yield {
                            'type': 'batch_progress',
                            'batch_id': batch_id,
                            'completed_queries': completed_queries,
                            'total_queries': len(queries),
                            'progress_percentage': (completed_queries / len(queries)) * 100
                        }
            
            # Send batch completion event
            yield {
                'type': 'batch_complete',
                'batch_id': batch_id,
                'total_queries': len(queries),
                'completed_queries': completed_queries,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            yield {
                'type': 'batch_error',
                'batch_id': batch_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

