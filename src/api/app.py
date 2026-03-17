from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import logging
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Config
from core.retriever import MultiVectorRetriever, QueryProcessor
from core.llm_integration import LLMIntegrator
from core.query_engine import RealTimeQueryEngine
from core.streaming_handler import StreamingResponseHandler
from api.streaming_routes import streaming_bp, init_streaming_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins for development

# Configuration
config = Config()
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Ensure upload directories exist
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)

# Global variables for components
retriever = None
query_processor = None
llm_integrator = None
query_engine = None
streaming_handler = None

def initialize_components():
    """Initialize all system components."""
    global retriever, query_processor, llm_integrator, query_engine, streaming_handler
    
    try:
        # Initialize retriever
        retriever = MultiVectorRetriever(config)
        if not retriever.initialize():
            logger.error("Failed to initialize retriever")
            return False
        
        # Initialize query processor
        query_processor = QueryProcessor(retriever)
        
        # Initialize LLM integrator
        llm_integrator = LLMIntegrator(config)
        if not llm_integrator.initialize():
            logger.error("Failed to initialize LLM integrator")
            return False
        
        # Initialize real-time query engine
        query_engine = RealTimeQueryEngine(config)
        if not query_engine.initialize(retriever, llm_integrator):
            logger.error("Failed to initialize query engine")
            return False
        
        # Initialize streaming handler
        streaming_handler = StreamingResponseHandler(config)
        if not streaming_handler.initialize(query_engine):
            logger.error("Failed to initialize streaming handler")
            return False
        
        # Initialize streaming routes
        init_streaming_routes(streaming_handler, query_engine)
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

# Register streaming blueprint
app.register_blueprint(streaming_bp)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        if retriever:
            stats = retriever.get_statistics()
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'statistics': stats
            })
        else:
            return jsonify({
                'status': 'initializing',
                'timestamp': datetime.now().isoformat()
            }), 503
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a document."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the document
        if retriever:
            success = retriever.add_document(file_path)
            if success:
                return jsonify({
                    'message': 'Document uploaded and processed successfully',
                    'filename': filename,
                    'file_path': file_path
                })
            else:
                return jsonify({'error': 'Failed to process document'}), 500
        else:
            return jsonify({'error': 'System not initialized'}), 503
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_documents():
    """Query the document collection."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        top_k = data.get('top_k', config.TOP_K)
        content_types = data.get('content_types', None)
        
        if not retriever or not query_processor or not llm_integrator:
            return jsonify({'error': 'System not initialized'}), 503
        
        # Analyze the query
        query_analysis = query_processor.analyze_query(query)
        
        # Retrieve relevant documents
        if content_types:
            results = retriever.retrieve(query, top_k, content_types)
        else:
            # Use suggested content types from query analysis
            suggested_types = query_analysis.get('suggested_content_types', ['text'])
            results = retriever.retrieve(query, top_k, suggested_types)
        
        # Generate response using LLM
        response = llm_integrator.generate_response(query, results)
        
        # Format results for API response
        formatted_results = []
        for result in results:
            formatted_result = {
                'id': result.element.id,
                'content_type': result.element.content_type,
                'content': result.element.content[:500] + '...' if len(result.element.content) > 500 else result.element.content,
                'source': result.element.source,
                'page_number': result.element.page_number,
                'score': result.score,
                'relevance_explanation': result.relevance_explanation,
                'metadata': result.element.metadata
            }
            formatted_results.append(formatted_result)
        
        return jsonify({
            'query': query,
            'query_analysis': query_analysis,
            'results': formatted_results,
            'response': response,
            'total_results': len(results)
        })
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/hybrid_search', methods=['POST'])
def hybrid_search():
    """Perform hybrid search across all content types."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        top_k = data.get('top_k', config.TOP_K)
        
        if not retriever:
            return jsonify({'error': 'System not initialized'}), 503
        
        # Perform hybrid search
        results = retriever.hybrid_search(query, top_k)
        
        # Format results
        formatted_results = {}
        for content_type, type_results in results.items():
            formatted_results[content_type] = []
            for result in type_results:
                formatted_result = {
                    'id': result.element.id,
                    'content': result.element.content[:300] + '...' if len(result.element.content) > 300 else result.element.content,
                    'source': result.element.source,
                    'page_number': result.element.page_number,
                    'score': result.score,
                    'relevance_explanation': result.relevance_explanation
                }
                formatted_results[content_type].append(formatted_result)
        
        return jsonify({
            'query': query,
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/table/<table_id>', methods=['GET'])
def get_table_data(table_id):
    """Get full table data."""
    try:
        if not retriever:
            return jsonify({'error': 'System not initialized'}), 503
        
        table_df = retriever.get_table_data(table_id)
        if table_df is not None:
            return jsonify({
                'table_id': table_id,
                'data': table_df.to_dict('records'),
                'columns': table_df.columns.tolist(),
                'shape': table_df.shape
            })
        else:
            return jsonify({'error': 'Table not found'}), 404
            
    except Exception as e:
        logger.error(f"Failed to get table data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics."""
    try:
        if not retriever:
            return jsonify({'error': 'System not initialized'}), 503
        
        stats = retriever.get_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_data', methods=['POST'])
def clear_all_data():
    """Clear all data from the system."""
    try:
        if not retriever:
            return jsonify({'error': 'System not initialized'}), 503
        
        success = retriever.clear_all_data()
        if success:
            return jsonify({'message': 'All data cleared successfully'})
        else:
            return jsonify({'error': 'Failed to clear data'}), 500
            
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for conversational interaction."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        conversation_history = data.get('history', [])
        
        if not llm_integrator:
            return jsonify({'error': 'LLM not initialized'}), 503
        
        # Generate conversational response
        response = llm_integrator.chat(message, conversation_history)
        
        return jsonify({
            'message': message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize components on startup (required for both direct run and flask run)
if initialize_components():
    logger.info("All components initialized successfully")
else:
    logger.error("Failed to initialize components. Exiting.")
    sys.exit(1)

if __name__ == '__main__':
    logger.info("Starting Multi-RAG API server...")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )

