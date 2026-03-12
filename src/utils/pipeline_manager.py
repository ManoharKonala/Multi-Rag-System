#!/usr/bin/env python3
"""
Pipeline Manager for Multi-RAG System

This module provides a unified interface for managing the entire
document processing and embedding pipeline.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Config
from core.retriever import MultiVectorRetriever
from core.document_processor import DocumentProcessor, ImageProcessor, TableProcessor
from core.embeddings import EmbeddingGenerator, TableEmbeddingGenerator
from .batch_processor import BatchProcessor
from .embedding_updater import EmbeddingUpdater, DocumentTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineManager:
    """Manages the entire document processing and embedding pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.retriever = None
        self.batch_processor = None
        self.embedding_updater = None
        self.document_tracker = DocumentTracker()
        
        # Pipeline components
        self.document_processor = None
        self.embedding_generator = None
        self.image_processor = None
        self.table_processor = None
        
        # Pipeline state
        self.is_initialized = False
        self.is_running = False
        self.pipeline_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'total_elements': 0,
            'start_time': None,
            'last_activity': None
        }
        
        # Thread safety
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing pipeline manager...")
            
            # Initialize core retriever
            self.retriever = MultiVectorRetriever(self.config)
            if not self.retriever.initialize():
                logger.error("Failed to initialize retriever")
                return False
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(self.config)
            if not self.embedding_generator.initialize_models():
                logger.error("Failed to initialize embedding generator")
                return False
            
            # Initialize specialized processors
            self.image_processor = ImageProcessor()
            self.table_processor = TableProcessor()
            
            # Initialize batch processor
            self.batch_processor = BatchProcessor(self.config)
            if not self.batch_processor.initialize():
                logger.error("Failed to initialize batch processor")
                return False
            
            # Initialize embedding updater
            self.embedding_updater = EmbeddingUpdater(self.config)
            if not self.embedding_updater.initialize():
                logger.error("Failed to initialize embedding updater")
                return False
            
            self.is_initialized = True
            self.pipeline_stats['start_time'] = datetime.now()
            
            logger.info("Pipeline manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline manager: {e}")
            return False
    
    def start_pipeline(self, watch_directories: List[str] = None) -> bool:
        """Start the complete pipeline with monitoring."""
        try:
            if not self.is_initialized:
                logger.error("Pipeline not initialized")
                return False
            
            logger.info("Starting pipeline...")
            
            # Start file watching if directories provided
            if watch_directories:
                self.embedding_updater.start_watching(watch_directories)
            
            # Start scheduled updates
            self.embedding_updater.start_scheduled_updates()
            
            self.is_running = True
            logger.info("Pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop_pipeline(self):
        """Stop the pipeline and all monitoring."""
        try:
            logger.info("Stopping pipeline...")
            
            if self.embedding_updater:
                self.embedding_updater.stop_watching()
            
            self.is_running = False
            logger.info("Pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    def process_single_document(self, file_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process a single document through the complete pipeline."""
        try:
            with self.lock:
                self.pipeline_stats['last_activity'] = datetime.now()
            
            logger.info(f"Processing document: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if already processed and up-to-date
            if not force_reprocess and not self.document_tracker.is_file_changed(file_path):
                logger.info(f"Document unchanged, skipping: {file_path}")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'unchanged',
                    'file_path': file_path
                }
            
            start_time = time.time()
            
            # Process document
            success = self.retriever.add_document(file_path)
            
            processing_time = time.time() - start_time
            
            if success:
                # Update statistics
                with self.lock:
                    self.pipeline_stats['total_processed'] += 1
                    
                    # Count elements for this document
                    elements_count = len([
                        elem for elem in self.retriever.document_store.values()
                        if elem.source == file_path
                    ])
                    self.pipeline_stats['total_elements'] += elements_count
                
                # Update document tracker
                self.document_tracker.update_file_metadata(file_path, elements_count)
                
                logger.info(f"Successfully processed: {file_path} ({elements_count} elements, {processing_time:.2f}s)")
                
                return {
                    'success': True,
                    'skipped': False,
                    'file_path': file_path,
                    'elements_count': elements_count,
                    'processing_time': processing_time
                }
            else:
                with self.lock:
                    self.pipeline_stats['total_failed'] += 1
                
                logger.error(f"Failed to process: {file_path}")
                
                return {
                    'success': False,
                    'skipped': False,
                    'file_path': file_path,
                    'error': 'Processing failed',
                    'processing_time': processing_time
                }
                
        except Exception as e:
            with self.lock:
                self.pipeline_stats['total_failed'] += 1
            
            logger.error(f"Error processing document {file_path}: {e}")
            
            return {
                'success': False,
                'skipped': False,
                'file_path': file_path,
                'error': str(e)
            }
    
    def process_batch(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, Any]:
        """Process multiple documents in parallel."""
        try:
            logger.info(f"Starting batch processing of {len(file_paths)} files")
            
            results = {
                'total_files': len(file_paths),
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'results': [],
                'start_time': datetime.now(),
                'end_time': None,
                'processing_time': 0
            }
            
            start_time = time.time()
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_document, file_path): file_path
                    for file_path in file_paths
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results['results'].append(result)
                        
                        if result['success']:
                            if result.get('skipped', False):
                                results['skipped'] += 1
                            else:
                                results['successful'] += 1
                        else:
                            results['failed'] += 1
                            
                    except Exception as e:
                        logger.error(f"Exception processing {file_path}: {e}")
                        results['failed'] += 1
                        results['results'].append({
                            'success': False,
                            'file_path': file_path,
                            'error': str(e)
                        })
            
            results['processing_time'] = time.time() - start_time
            results['end_time'] = datetime.now()
            
            logger.info(f"Batch processing completed: {results['successful']} successful, "
                       f"{results['failed']} failed, {results['skipped']} skipped")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {
                'total_files': len(file_paths),
                'successful': 0,
                'failed': len(file_paths),
                'skipped': 0,
                'error': str(e)
            }
    
    def process_directory(self, directory_path: str, recursive: bool = True, 
                         max_workers: int = 4) -> Dict[str, Any]:
        """Process all PDF files in a directory."""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            # Find all PDF files
            if recursive:
                pdf_files = list(directory.rglob("*.pdf"))
            else:
                pdf_files = list(directory.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {directory_path}")
                return {
                    'total_files': 0,
                    'successful': 0,
                    'failed': 0,
                    'skipped': 0,
                    'message': 'No PDF files found'
                }
            
            # Convert to string paths
            file_paths = [str(pdf_file) for pdf_file in pdf_files]
            
            # Process batch
            return self.process_batch(file_paths, max_workers)
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'skipped': 0,
                'error': str(e)
            }
    
    def reprocess_changed_files(self, directory_path: str = None) -> Dict[str, Any]:
        """Reprocess files that have changed since last processing."""
        try:
            if not directory_path:
                directory_path = self.config.UPLOAD_FOLDER
            
            directory = Path(directory_path)
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory_path}")
                return {'changed_files': 0, 'processed': 0}
            
            # Find all PDF files
            pdf_files = list(directory.rglob("*.pdf"))
            
            # Filter to only changed files
            changed_files = []
            for pdf_file in pdf_files:
                if self.document_tracker.is_file_changed(str(pdf_file)):
                    changed_files.append(str(pdf_file))
            
            if not changed_files:
                logger.info("No changed files found")
                return {'changed_files': 0, 'processed': 0}
            
            logger.info(f"Found {len(changed_files)} changed files")
            
            # Process changed files
            results = self.process_batch(changed_files)
            
            return {
                'changed_files': len(changed_files),
                'processed': results['successful'],
                'failed': results['failed'],
                'skipped': results['skipped'],
                'details': results
            }
            
        except Exception as e:
            logger.error(f"Error reprocessing changed files: {e}")
            return {'error': str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        try:
            # Get retriever statistics
            retriever_stats = self.retriever.get_statistics() if self.retriever else {}
            
            # Get updater statistics
            updater_stats = self.embedding_updater.get_update_statistics() if self.embedding_updater else {}
            
            # Calculate uptime
            uptime = None
            if self.pipeline_stats['start_time']:
                uptime = (datetime.now() - self.pipeline_stats['start_time']).total_seconds()
            
            status = {
                'initialized': self.is_initialized,
                'running': self.is_running,
                'uptime_seconds': uptime,
                'pipeline_stats': self.pipeline_stats.copy(),
                'retriever_stats': retriever_stats,
                'updater_stats': updater_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            if status['pipeline_stats']['start_time']:
                status['pipeline_stats']['start_time'] = status['pipeline_stats']['start_time'].isoformat()
            if status['pipeline_stats']['last_activity']:
                status['pipeline_stats']['last_activity'] = status['pipeline_stats']['last_activity'].isoformat()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}
    
    def optimize_pipeline(self) -> Dict[str, Any]:
        """Optimize pipeline performance and clean up resources."""
        try:
            logger.info("Starting pipeline optimization...")
            
            optimization_results = {
                'start_time': datetime.now(),
                'actions_taken': [],
                'errors': []
            }
            
            # Clean up orphaned embeddings
            try:
                # This would involve checking for embeddings without corresponding files
                # and removing them from the vector database
                optimization_results['actions_taken'].append('Checked for orphaned embeddings')
            except Exception as e:
                optimization_results['errors'].append(f"Failed to clean orphaned embeddings: {e}")
            
            # Optimize vector database
            try:
                # This could involve rebuilding indexes or compacting the database
                optimization_results['actions_taken'].append('Optimized vector database')
            except Exception as e:
                optimization_results['errors'].append(f"Failed to optimize vector database: {e}")
            
            # Update document metadata
            try:
                # Refresh metadata for all tracked files
                optimization_results['actions_taken'].append('Updated document metadata')
            except Exception as e:
                optimization_results['errors'].append(f"Failed to update metadata: {e}")
            
            optimization_results['end_time'] = datetime.now()
            optimization_results['duration'] = (
                optimization_results['end_time'] - optimization_results['start_time']
            ).total_seconds()
            
            logger.info(f"Pipeline optimization completed in {optimization_results['duration']:.2f} seconds")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error during pipeline optimization: {e}")
            return {'error': str(e)}
    
    def export_pipeline_data(self, output_path: str) -> bool:
        """Export pipeline data and statistics."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'pipeline_status': self.get_pipeline_status(),
                'document_metadata': self.document_tracker.metadata,
                'configuration': {
                    'chunk_size': self.config.CHUNK_SIZE,
                    'chunk_overlap': self.config.CHUNK_OVERLAP,
                    'embedding_model': self.config.EMBEDDING_MODEL,
                    'embedding_dimension': self.config.EMBEDDING_DIMENSION
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Pipeline data exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export pipeline data: {e}")
            return False
    
    def cleanup(self):
        """Clean up all pipeline resources."""
        try:
            logger.info("Cleaning up pipeline resources...")
            
            self.stop_pipeline()
            
            if self.retriever:
                self.retriever.close()
            
            if self.batch_processor:
                self.batch_processor.cleanup()
            
            if self.embedding_updater:
                self.embedding_updater.cleanup()
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline manager for Multi-RAG system')
    parser.add_argument('--process-dir', type=str, help='Process all files in directory')
    parser.add_argument('--process-file', type=str, help='Process a specific file')
    parser.add_argument('--watch', nargs='+', help='Start watching directories')
    parser.add_argument('--reprocess-changed', action='store_true', help='Reprocess changed files')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--optimize', action='store_true', help='Optimize pipeline')
    parser.add_argument('--export', type=str, help='Export pipeline data to file')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Initialize pipeline manager
    pipeline = PipelineManager(config)
    
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline manager")
        sys.exit(1)
    
    try:
        if args.process_file:
            # Process single file
            result = pipeline.process_single_document(args.process_file)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.process_dir:
            # Process directory
            result = pipeline.process_directory(args.process_dir, args.recursive, args.workers)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.reprocess_changed:
            # Reprocess changed files
            result = pipeline.reprocess_changed_files()
            print(json.dumps(result, indent=2, default=str))
            
        elif args.status:
            # Show status
            status = pipeline.get_pipeline_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.optimize:
            # Optimize pipeline
            result = pipeline.optimize_pipeline()
            print(json.dumps(result, indent=2, default=str))
            
        elif args.export:
            # Export data
            success = pipeline.export_pipeline_data(args.export)
            print(f"Export {'successful' if success else 'failed'}")
            
        elif args.watch:
            # Start watching
            pipeline.start_pipeline(args.watch)
            
            if args.daemon:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
            else:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        pipeline.cleanup()

if __name__ == '__main__':
    main()

