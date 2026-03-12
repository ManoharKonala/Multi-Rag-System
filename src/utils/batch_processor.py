#!/usr/bin/env python3
"""
Batch Document Processor for Multi-RAG System

This script processes multiple documents in batch mode, extracting content
and generating embeddings for storage in the vector database.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Config
from core.retriever import MultiVectorRetriever
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processes documents in batch mode with parallel processing."""
    
    def __init__(self, config: Config, max_workers: int = 4):
        self.config = config
        self.max_workers = max_workers
        self.retriever = None
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_elements': 0,
            'processing_time': 0,
            'errors': []
        }
        self.lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize the batch processor."""
        try:
            self.retriever = MultiVectorRetriever(self.config)
            if not self.retriever.initialize():
                logger.error("Failed to initialize retriever")
                return False
            
            logger.info("Batch processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize batch processor: {e}")
            return False
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
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
                return self.stats
            
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            self.stats['total_files'] = len(pdf_files)
            
            start_time = time.time()
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_file, str(pdf_file)): pdf_file
                    for pdf_file in pdf_files
                }
                
                for future in as_completed(future_to_file):
                    pdf_file = future_to_file[future]
                    try:
                        result = future.result()
                        with self.lock:
                            if result['success']:
                                self.stats['processed_files'] += 1
                                self.stats['total_elements'] += result['elements_count']
                                logger.info(f"Successfully processed: {pdf_file.name}")
                            else:
                                self.stats['failed_files'] += 1
                                self.stats['errors'].append({
                                    'file': str(pdf_file),
                                    'error': result['error']
                                })
                                logger.error(f"Failed to process {pdf_file.name}: {result['error']}")
                    except Exception as e:
                        with self.lock:
                            self.stats['failed_files'] += 1
                            self.stats['errors'].append({
                                'file': str(pdf_file),
                                'error': str(e)
                            })
                        logger.error(f"Exception processing {pdf_file.name}: {e}")
            
            self.stats['processing_time'] = time.time() - start_time
            
            # Log final statistics
            self.log_final_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Failed to process directory {directory_path}: {e}")
            return self.stats
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single PDF file."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Add document to retriever
            success = self.retriever.add_document(file_path)
            
            if success:
                # Get element count from document store
                elements_count = len([
                    elem for elem in self.retriever.document_store.values()
                    if elem.source == file_path
                ])
                
                return {
                    'success': True,
                    'elements_count': elements_count,
                    'file_path': file_path
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to add document to retriever',
                    'file_path': file_path
                }
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def process_file_list(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process a list of specific files."""
        try:
            # Validate files exist
            valid_files = []
            for file_path in file_paths:
                if Path(file_path).exists() and file_path.lower().endswith('.pdf'):
                    valid_files.append(file_path)
                else:
                    logger.warning(f"Skipping invalid file: {file_path}")
            
            if not valid_files:
                logger.warning("No valid PDF files to process")
                return self.stats
            
            logger.info(f"Processing {len(valid_files)} files")
            self.stats['total_files'] = len(valid_files)
            
            start_time = time.time()
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_file, file_path): file_path
                    for file_path in valid_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        with self.lock:
                            if result['success']:
                                self.stats['processed_files'] += 1
                                self.stats['total_elements'] += result['elements_count']
                                logger.info(f"Successfully processed: {Path(file_path).name}")
                            else:
                                self.stats['failed_files'] += 1
                                self.stats['errors'].append({
                                    'file': file_path,
                                    'error': result['error']
                                })
                                logger.error(f"Failed to process {Path(file_path).name}: {result['error']}")
                    except Exception as e:
                        with self.lock:
                            self.stats['failed_files'] += 1
                            self.stats['errors'].append({
                                'file': file_path,
                                'error': str(e)
                            })
                        logger.error(f"Exception processing {Path(file_path).name}: {e}")
            
            self.stats['processing_time'] = time.time() - start_time
            
            # Log final statistics
            self.log_final_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Failed to process file list: {e}")
            return self.stats
    
    def log_final_stats(self):
        """Log final processing statistics."""
        logger.info("=" * 50)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Successfully processed: {self.stats['processed_files']}")
        logger.info(f"Failed: {self.stats['failed_files']}")
        logger.info(f"Total elements extracted: {self.stats['total_elements']}")
        logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            if self.stats['processing_time'] > 0:
                throughput = self.stats['total_files'] / self.stats['processing_time']
                logger.info(f"Throughput: {throughput:.2f} files/second")
        
        if self.stats['errors']:
            logger.info(f"\nErrors encountered:")
            for error in self.stats['errors']:
                logger.info(f"  {error['file']}: {error['error']}")
    
    def save_stats_to_file(self, output_path: str):
        """Save processing statistics to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            logger.info(f"Statistics saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.retriever:
            self.retriever.close()

class ProgressReporter:
    """Reports processing progress in real-time."""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed_files = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def update(self, increment: int = 1):
        """Update progress counter."""
        with self.lock:
            self.processed_files += increment
            self.report_progress()
    
    def report_progress(self):
        """Report current progress."""
        if self.total_files == 0:
            return
        
        progress = (self.processed_files / self.total_files) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.processed_files > 0:
            avg_time_per_file = elapsed_time / self.processed_files
            remaining_files = self.total_files - self.processed_files
            eta = remaining_files * avg_time_per_file
            
            logger.info(f"Progress: {self.processed_files}/{self.total_files} ({progress:.1f}%) - ETA: {eta:.1f}s")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Batch process documents for Multi-RAG system')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing PDF files to process')
    parser.add_argument('--files', '-f', nargs='+', help='Specific files to process')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directories recursively')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Number of worker threads (default: 4)')
    parser.add_argument('--output', '-o', type=str, help='Output file for statistics (JSON format)')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before processing')
    
    args = parser.parse_args()
    
    if not args.directory and not args.files:
        parser.error("Either --directory or --files must be specified")
    
    # Initialize configuration
    config = Config()
    
    # Initialize batch processor
    processor = BatchProcessor(config, max_workers=args.workers)
    
    if not processor.initialize():
        logger.error("Failed to initialize batch processor")
        sys.exit(1)
    
    try:
        # Clear existing data if requested
        if args.clear:
            logger.info("Clearing existing data...")
            processor.retriever.clear_all_data()
        
        # Process files
        if args.directory:
            stats = processor.process_directory(args.directory, recursive=args.recursive)
        else:
            stats = processor.process_file_list(args.files)
        
        # Save statistics if output file specified
        if args.output:
            processor.save_stats_to_file(args.output)
        
        # Exit with appropriate code
        if stats['failed_files'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)
    finally:
        processor.cleanup()

if __name__ == '__main__':
    main()

