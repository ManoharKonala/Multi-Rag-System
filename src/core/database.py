import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging
from typing import List, Dict, Any, Optional
from config.settings import Config

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """Manages PostgreSQL database operations for structured data storage."""
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        self.connection = None
        
    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.engine = create_engine(self.config.postgres_url)
            self.connection = self.engine.connect()
            logger.info("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Disconnected from PostgreSQL database")
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Create a table from a pandas DataFrame."""
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Successfully created table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            return False
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Insert DataFrame data into existing table."""
        try:
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"Successfully inserted data into table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data into table {table_name}: {e}")
            return False
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute a SQL query and return results as DataFrame."""
        try:
            result = pd.read_sql(query, self.engine)
            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return None
    
    def get_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Retrieve all data from a specific table."""
        try:
            query = f"SELECT * FROM {table_name}"
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to retrieve data from table {table_name}: {e}")
            return None
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                );
            """)
            result = self.connection.execute(query, {"table_name": table_name})
            return result.scalar()
        except Exception as e:
            logger.error(f"Failed to check if table {table_name} exists: {e}")
            return False

class MilvusManager:
    """Manages Milvus vector database operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.collection = None
        self.connected = False
        
    def connect(self):
        """Establish connection to Milvus database."""
        try:
            connections.connect(
                alias="default",
                host=self.config.MILVUS_HOST,
                port=self.config.MILVUS_PORT
            )
            self.connected = True
            logger.info("Successfully connected to Milvus database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def disconnect(self):
        """Close Milvus connection."""
        if self.connected:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus database")
    
    def create_collection(self, collection_name: str = None) -> bool:
        """Create a new collection in Milvus."""
        if not collection_name:
            collection_name = self.config.MILVUS_COLLECTION_NAME
            
        try:
            # Check if collection already exists
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                self.collection = Collection(collection_name)
                return True
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.EMBEDDING_DIMENSION)
            ]
            
            schema = CollectionSchema(fields, f"Multi-RAG collection: {collection_name}")
            self.collection = Collection(collection_name, schema)
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            
            logger.info(f"Successfully created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def load_collection(self, collection_name: str = None):
        """Load collection into memory for search operations."""
        if not collection_name:
            collection_name = self.config.MILVUS_COLLECTION_NAME
            
        try:
            if not self.collection:
                self.collection = Collection(collection_name)
            self.collection.load()
            logger.info(f"Successfully loaded collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {e}")
    
    def insert_vectors(self, data: List[Dict[str, Any]]) -> bool:
        """Insert vector data into the collection."""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            # Prepare data for insertion
            entities = [
                [item["id"] for item in data],
                [item["content"] for item in data],
                [item["content_type"] for item in data],
                [item["source"] for item in data],
                [item["metadata"] for item in data],
                [item["embedding"] for item in data]
            ]
            
            self.collection.insert(entities)
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(data)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False
    
    def search_vectors(self, query_embedding: List[float], top_k: int = None) -> List[Dict]:
        """Search for similar vectors in the collection."""
        if not top_k:
            top_k = self.config.TOP_K
            
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "content", "content_type", "source", "metadata"]
            )
            
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content"),
                        "content_type": hit.entity.get("content_type"),
                        "source": hit.entity.get("source"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete_collection(self, collection_name: str = None):
        """Delete a collection from Milvus."""
        if not collection_name:
            collection_name = self.config.MILVUS_COLLECTION_NAME
            
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")

class DatabaseManager:
    """Unified database manager for both PostgreSQL and Milvus."""
    
    def __init__(self, config: Config):
        self.config = config
        self.postgres = PostgreSQLManager(config)
        self.milvus = MilvusManager(config)
        
    def initialize(self) -> bool:
        """Initialize both database connections."""
        postgres_connected = self.postgres.connect()
        milvus_connected = self.milvus.connect()
        
        if milvus_connected:
            collection_created = self.milvus.create_collection()
            if collection_created:
                self.milvus.load_collection()
        
        return postgres_connected and milvus_connected
    
    def close_connections(self):
        """Close all database connections."""
        self.postgres.disconnect()
        self.milvus.disconnect()
    
    def health_check(self) -> Dict[str, bool]:
        """Check the health of both database connections."""
        return {
            "postgres": self.postgres.connection is not None,
            "milvus": self.milvus.connected
        }

