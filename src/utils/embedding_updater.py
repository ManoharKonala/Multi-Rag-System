#!/usr/bin/env python3
"""
Continuous Embedding Updater for Multi-RAG System

This module provides functionality for continuously updating embeddings
when documents are modified or when embedding models are updated.
"""

import os
import sys
import time
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import schedule

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import Config
from core.retriever import MultiVectorRetriever
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentTracker:
    """Tracks document changes and manages metadata."""
    
    def __init__(self, metadata_file: str = "document_metadata.json"):
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from file."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    
    def save_metadata(self):
        """Save document metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def is_file_changed(self, file_path: str) -> bool:
        """Check if a file has changed since last processing."""
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.metadata.get(file_path, {}).get('hash', '')
        return current_hash != stored_hash
    
    def update_file_metadata(self, file_path: str, elements_count: int = 0):
        """Update metadata for a processed file."""
        self.metadata[file_path] = {
            'hash': self.get_file_hash(file_path),
            'last_processed': datetime.now().isoformat(),
            'elements_count': elements_count,
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        self.save_metadata()
    
    def remove_file_metadata(self, file_path: str):
        """Remove metadata for a deleted file."""
        if file_path in self.metadata:
            del self.metadata[file_path]
            self.save_metadata()
    
    def get_outdated_files(self, directory: str, max_age_days: int = 30) -> List[str]:
        """Get files that haven't been processed recently."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        outdated_files = []
        
        for file_path, metadata in self.metadata.items():
            if not os.path.exists(file_path):
                continue
                
            last_processed = datetime.fromisoformat(metadata.get('last_processed', '1970-01-01'))
            if last_processed < cutoff_date:
                outdated_files.append(file_path)
        
        return outdated_files

class DocumentWatcher(FileSystemEventHandler):
    """Watches for file system changes and triggers updates."""
    
    def __init__(self, updater: 'EmbeddingUpdater'):
        self.updater = updater
        self.pending_updates = set()
        self.update_timer = None
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        if event.src_path.lower().endswith('.pdf'):
            logger.info(f"Detected modification: {event.src_path}")
            self.schedule_update(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
            
        if event.src_path.lower().endswith('.pdf'):
            logger.info(f"Detected new file: {event.src_path}")
            self.schedule_update(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory:
            return
            
        if event.src_path.lower().endswith('.pdf'):
            logger.info(f"Detected deletion: {event.src_path}")
            self.updater.handle_file_deletion(event.src_path)
    
    def schedule_update(self, file_path: str):
        """Schedule an update with debouncing."""
        self.pending_updates.add(file_path)
        
        # Cancel existing timer
        if self.update_timer:
            self.update_timer.cancel()
        
        # Schedule new update after delay
        self.update_timer = threading.Timer(5.0, self.process_pending_updates)
        self.update_timer.start()
    
    def process_pending_updates(self):
        """Process all pending updates."""
        if self.pending_updates:
            files_to_update = list(self.pending_updates)
            self.pending_updates.clear()
            
            for file_path in files_to_update:
                self.updater.update_single_file(file_path)

class EmbeddingUpdater:
    """Manages continuous embedding updates."""
    
    def __init__(self, config: Config):
        self.config = config
        self.retriever = None
        self.tracker = DocumentTracker()
        self.observer = None
        self.running = False
        self.update_thread = None
        
    def initialize(self) -> bool:
        """Initialize the embedding updater."""
        try:
            self.retriever = MultiVectorRetriever(self.config)
            if not self.retriever.initialize():
                logger.error("Failed to initialize retriever")
                return False
            
            logger.info("Embedding updater initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding updater: {e}")
            return False
    
    def start_watching(self, watch_directories: List[str]):
        """Start watching directories for changes."""
        try:
            if self.observer:
                self.stop_watching()
            
            self.observer = Observer()
            watcher = DocumentWatcher(self)
            
            for directory in watch_directories:
                if os.path.exists(directory):
                    self.observer.schedule(watcher, directory, recursive=True)
                    logger.info(f"Watching directory: {directory}")
                else:
                    logger.warning(f"Directory does not exist: {directory}")
            
            self.observer.start()
            self.running = True
            logger.info("File watching started")
            
        except Exception as e:
            logger.error(f"Failed to start watching: {e}")
    
    def stop_watching(self):
        """Stop watching for file changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.running = False
        logger.info("File watching stopped")
    
    def start_scheduled_updates(self):
        """Start scheduled periodic updates."""
        # Schedule daily updates at 2 AM
        schedule.every().day.at("02:00").do(self.run_periodic_update)
        
        # Schedule weekly full reprocessing on Sundays at 3 AM
        schedule.every().sunday.at("03:00").do(self.run_full_reprocessing)
        
        # Start scheduler thread
        self.update_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.update_thread.start()
        
        logger.info("Scheduled updates started")
    
    def run_scheduler(self):
        """Run the scheduled update loop."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def update_single_file(self, file_path: str):
        """Update embeddings for a single file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            # Check if file has actually changed
            if not self.tracker.is_file_changed(file_path):
                logger.info(f"File unchanged, skipping: {file_path}")
                return True
            
            logger.info(f"Updating embeddings for: {file_path}")
            
            # Remove existing embeddings for this file
            self.remove_file_embeddings(file_path)
            
            # Add updated document
            success = self.retriever.add_document(file_path)
            
            if success:
                # Update metadata
                elements_count = len([
                    elem for elem in self.retriever.document_store.values()
                    if elem.source == file_path
                ])
                self.tracker.update_file_metadata(file_path, elements_count)
                logger.info(f"Successfully updated: {file_path}")
                return True
            else:
                logger.error(f"Failed to update: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating file {file_path}: {e}")
            return False
    
    def remove_file_embeddings(self, file_path: str):
        """Remove embeddings for a specific file."""
        try:
            # Find elements from this file
            elements_to_remove = [
                elem_id for elem_id, elem in self.retriever.document_store.items()
                if elem.source == file_path
            ]
            
            # Remove from document store
            for elem_id in elements_to_remove:
                if elem_id in self.retriever.document_store:
                    del self.retriever.document_store[elem_id]
            
            logger.info(f"Removed {len(elements_to_remove)} elements for file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error removing embeddings for {file_path}: {e}")
    
    def handle_file_deletion(self, file_path: str):
        """Handle deletion of a file."""
        try:
            self.remove_file_embeddings(file_path)
            self.tracker.remove_file_metadata(file_path)
            logger.info(f"Handled deletion of: {file_path}")
            
        except Exception as e:
            logger.error(f"Error handling deletion of {file_path}: {e}")
    
    def run_periodic_update(self):
        """Run periodic update of outdated files."""
        try:
            logger.info("Starting periodic update...")
            
            # Get files that haven't been processed recently
            outdated_files = self.tracker.get_outdated_files(
                self.config.UPLOAD_FOLDER, 
                max_age_days=7
            )
            
            if not outdated_files:
                logger.info("No outdated files found")
                return
            
            logger.info(f"Found {len(outdated_files)} outdated files")
            
            # Update outdated files
            updated_count = 0
            for file_path in outdated_files:
                if self.update_single_file(file_path):
                    updated_count += 1
            
            logger.info(f"Periodic update completed: {updated_count}/{len(outdated_files)} files updated")
            
        except Exception as e:
            logger.error(f"Error during periodic update: {e}")
    
    def run_full_reprocessing(self):
        """Run full reprocessing of all documents."""
        try:
            logger.info("Starting full reprocessing...")
            
            # Clear all existing data
            self.retriever.clear_all_data()
            
            # Find all PDF files in upload directory
            upload_dir = Path(self.config.UPLOAD_FOLDER)
            if not upload_dir.exists():
                logger.warning(f"Upload directory does not exist: {upload_dir}")
                return
            
            pdf_files = list(upload_dir.rglob("*.pdf"))
            
            if not pdf_files:
                logger.info("No PDF files found for reprocessing")
                return
            
            logger.info(f"Reprocessing {len(pdf_files)} files")
            
            # Process all files
            processed_count = 0
            for pdf_file in pdf_files:
                if self.update_single_file(str(pdf_file)):
                    processed_count += 1
            
            logger.info(f"Full reprocessing completed: {processed_count}/{len(pdf_files)} files processed")
            
        except Exception as e:
            logger.error(f"Error during full reprocessing: {e}")
    
    def get_update_statistics(self) -> Dict[str, Any]:
        """Get statistics about the update system."""
        try:
            stats = {
                'total_tracked_files': len(self.tracker.metadata),
                'watching_active': self.running,
                'last_update_check': datetime.now().isoformat(),
                'files_by_age': {},
                'total_elements': 0
            }
            
            # Calculate file age distribution
            now = datetime.now()
            age_buckets = {
                'last_day': 0,
                'last_week': 0,
                'last_month': 0,
                'older': 0
            }
            
            for file_path, metadata in self.tracker.metadata.items():
                last_processed = datetime.fromisoformat(metadata.get('last_processed', '1970-01-01'))
                age = now - last_processed
                
                stats['total_elements'] += metadata.get('elements_count', 0)
                
                if age.days <= 1:
                    age_buckets['last_day'] += 1
                elif age.days <= 7:
                    age_buckets['last_week'] += 1
                elif age.days <= 30:
                    age_buckets['last_month'] += 1
                else:
                    age_buckets['older'] += 1
            
            stats['files_by_age'] = age_buckets
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting update statistics: {e}")
            return {}
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_watching()
        if self.retriever:
            self.retriever.close()

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous embedding updater for Multi-RAG system')
    parser.add_argument('--watch', '-w', nargs='+', help='Directories to watch for changes')
    parser.add_argument('--update-file', '-f', type=str, help='Update a specific file')
    parser.add_argument('--periodic', '-p', action='store_true', help='Run periodic update')
    parser.add_argument('--full-reprocess', action='store_true', help='Run full reprocessing')
    parser.add_argument('--stats', action='store_true', help='Show update statistics')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Initialize updater
    updater = EmbeddingUpdater(config)
    
    if not updater.initialize():
        logger.error("Failed to initialize embedding updater")
        sys.exit(1)
    
    try:
        if args.update_file:
            # Update specific file
            success = updater.update_single_file(args.update_file)
            sys.exit(0 if success else 1)
            
        elif args.periodic:
            # Run periodic update
            updater.run_periodic_update()
            
        elif args.full_reprocess:
            # Run full reprocessing
            updater.run_full_reprocessing()
            
        elif args.stats:
            # Show statistics
            stats = updater.get_update_statistics()
            print(json.dumps(stats, indent=2))
            
        elif args.watch:
            # Start watching directories
            updater.start_watching(args.watch)
            
            if args.daemon:
                # Start scheduled updates
                updater.start_scheduled_updates()
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Shutting down...")
            else:
                # Just watch for changes
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
        updater.cleanup()

if __name__ == '__main__':
    main()

