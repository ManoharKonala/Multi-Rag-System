"""
Streaming API Routes for Multi-RAG System

This module provides Flask routes for real-time streaming capabilities
including Server-Sent Events and WebSocket-like functionality.
"""

from flask import Blueprint, request, Response, jsonify
import json
import asyncio
import logging
from datetime import datetime
from typing import Generator
import time

from core.streaming_handler import StreamingResponseHandler, ServerSentEventsHandler
from core.query_engine import RealTimeQueryEngine

logger = logging.getLogger(__name__)

# Create blueprint for streaming routes
streaming_bp = Blueprint('streaming', __name__, url_prefix='/api/stream')

# Global variables (will be set by main app)
streaming_handler = None
sse_handler = None
query_engine = None

def init_streaming_routes(sh: StreamingResponseHandler, qe: RealTimeQueryEngine):
    """Initialize streaming routes with required components."""
    global streaming_handler, sse_handler, query_engine
    streaming_handler = sh
    query_engine = qe
    sse_handler = ServerSentEventsHandler(streaming_handler)

@streaming_bp.route('/query', methods=['POST'])
def stream_query():
    """Stream a query response using Server-Sent Events."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        def generate_sse_response():
            """Generate Server-Sent Events for the query."""
            try:
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Stream the query response
                async def stream_async():
                    async for event_data in sse_handler.handle_sse_request(query):
                        yield event_data
                
                # Run the async generator
                for event_data in loop.run_until_complete(collect_async_generator(stream_async())):
                    yield event_data
                    
            except Exception as e:
                logger.error(f"Error in SSE generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            finally:
                loop.close()
        
        return Response(
            generate_sse_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream_query: {e}")
        return jsonify({'error': str(e)}), 500

@streaming_bp.route('/query/sync', methods=['POST'])
def stream_query_sync():
    """Stream a query response using synchronous chunked transfer."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        def generate_chunked_response():
            """Generate chunked response for the query."""
            try:
                # Process query synchronously
                result = query_engine.process_query(query, use_cache=True, stream_response=False)
                
                # Send initial metadata
                metadata = {
                    'type': 'metadata',
                    'query': query,
                    'analysis': {
                        'query_type': result.analysis.query_type.value,
                        'complexity': result.analysis.complexity.value,
                        'confidence': result.analysis.confidence
                    },
                    'sources_count': len(result.sources_used),
                    'timestamp': datetime.now().isoformat()
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                
                # Stream response in chunks
                response_text = result.response
                chunk_size = 50
                
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]
                    chunk_data = {
                        'type': 'chunk',
                        'chunk': chunk,
                        'chunk_index': i // chunk_size,
                        'total_chunks': (len(response_text) + chunk_size - 1) // chunk_size,
                        'progress': ((i + len(chunk)) / len(response_text)) * 100
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    time.sleep(0.05)  # Small delay for streaming effect
                
                # Send completion data
                completion_data = {
                    'type': 'complete',
                    'full_response': response_text,
                    'processing_time': result.processing_time,
                    'confidence': result.confidence,
                    'sources': result.sources_used,
                    'reasoning_steps': result.reasoning_steps
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                logger.error(f"Error in chunked response generation: {e}")
                error_data = {
                    'type': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return Response(
            generate_chunked_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream_query_sync: {e}")
        return jsonify({'error': str(e)}), 500

@streaming_bp.route('/batch', methods=['POST'])
def stream_batch_queries():
    """Stream processing of multiple queries."""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({'error': 'Queries list is required'}), 400
        
        queries = data['queries']
        if not isinstance(queries, list) or not queries:
            return jsonify({'error': 'Queries must be a non-empty list'}), 400
        
        max_concurrent = data.get('max_concurrent', 3)
        
        def generate_batch_response():
            """Generate batch processing response."""
            try:
                batch_id = f"batch_{int(time.time())}"
                
                # Send batch start event
                start_event = {
                    'type': 'batch_started',
                    'batch_id': batch_id,
                    'total_queries': len(queries),
                    'timestamp': datetime.now().isoformat()
                }
                yield f"data: {json.dumps(start_event)}\n\n"
                
                # Process queries sequentially (simplified version)
                results = []
                for i, query in enumerate(queries):
                    # Send query start event
                    query_start = {
                        'type': 'query_started',
                        'batch_id': batch_id,
                        'query_index': i,
                        'query': query,
                        'timestamp': datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(query_start)}\n\n"
                    
                    # Process query
                    try:
                        result = query_engine.process_query(query, use_cache=True)
                        results.append({
                            'query': query,
                            'success': True,
                            'response': result.response,
                            'confidence': result.confidence,
                            'processing_time': result.processing_time
                        })
                        
                        # Send query complete event
                        query_complete = {
                            'type': 'query_complete',
                            'batch_id': batch_id,
                            'query_index': i,
                            'result': results[-1],
                            'timestamp': datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(query_complete)}\n\n"
                        
                    except Exception as e:
                        results.append({
                            'query': query,
                            'success': False,
                            'error': str(e)
                        })
                        
                        # Send query error event
                        query_error = {
                            'type': 'query_error',
                            'batch_id': batch_id,
                            'query_index': i,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(query_error)}\n\n"
                    
                    # Send progress update
                    progress = {
                        'type': 'batch_progress',
                        'batch_id': batch_id,
                        'completed': i + 1,
                        'total': len(queries),
                        'progress_percentage': ((i + 1) / len(queries)) * 100
                    }
                    yield f"data: {json.dumps(progress)}\n\n"
                
                # Send batch complete event
                batch_complete = {
                    'type': 'batch_complete',
                    'batch_id': batch_id,
                    'results': results,
                    'successful_queries': len([r for r in results if r.get('success', False)]),
                    'failed_queries': len([r for r in results if not r.get('success', True)]),
                    'timestamp': datetime.now().isoformat()
                }
                yield f"data: {json.dumps(batch_complete)}\n\n"
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                error_event = {
                    'type': 'batch_error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
        
        return Response(
            generate_batch_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream_batch_queries: {e}")
        return jsonify({'error': str(e)}), 500

@streaming_bp.route('/status', methods=['GET'])
def get_streaming_status():
    """Get current streaming status and active streams."""
    try:
        if not streaming_handler:
            return jsonify({'error': 'Streaming handler not initialized'}), 500
        
        active_streams = streaming_handler.get_active_streams()
        metrics = query_engine.get_metrics() if query_engine else {}
        
        status = {
            'streaming_enabled': True,
            'active_streams': active_streams,
            'total_active_streams': len(active_streams),
            'query_engine_metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        return jsonify({'error': str(e)}), 500

@streaming_bp.route('/cancel/<stream_id>', methods=['POST'])
def cancel_stream(stream_id: str):
    """Cancel an active stream."""
    try:
        if not streaming_handler:
            return jsonify({'error': 'Streaming handler not initialized'}), 500
        
        success = streaming_handler.cancel_stream(stream_id)
        
        return jsonify({
            'success': success,
            'stream_id': stream_id,
            'message': 'Stream cancelled' if success else 'Stream not found or already completed'
        })
        
    except Exception as e:
        logger.error(f"Error cancelling stream {stream_id}: {e}")
        return jsonify({'error': str(e)}), 500

@streaming_bp.route('/health', methods=['GET'])
def streaming_health():
    """Health check for streaming endpoints."""
    try:
        health_status = {
            'streaming_handler': streaming_handler is not None,
            'query_engine': query_engine is not None,
            'sse_handler': sse_handler is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if all(health_status.values()):
            return jsonify({'status': 'healthy', 'components': health_status})
        else:
            return jsonify({'status': 'unhealthy', 'components': health_status}), 503
            
    except Exception as e:
        logger.error(f"Error in streaming health check: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Utility functions
async def collect_async_generator(async_gen):
    """Collect all items from an async generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items

def format_sse_data(data: dict) -> str:
    """Format data for Server-Sent Events."""
    return f"data: {json.dumps(data, default=str)}\n\n"

# Error handlers for streaming blueprint
@streaming_bp.errorhandler(404)
def streaming_not_found(error):
    return jsonify({'error': 'Streaming endpoint not found'}), 404

@streaming_bp.errorhandler(500)
def streaming_internal_error(error):
    return jsonify({'error': 'Internal streaming error'}), 500

