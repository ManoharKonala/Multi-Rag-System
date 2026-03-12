import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import base64
import io
import logging
from typing import List, Union, Optional, Dict, Any
from config.settings import Config

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for different modalities (text, images)."""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize_models(self):
        """Initialize embedding models."""
        try:
            # Initialize text embedding model
            self.text_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.text_model.to(self.device)
            
            # Initialize CLIP model for image embeddings
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
            logger.info("Successfully initialized embedding models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")
            return False
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text content."""
        try:
            if not self.text_model:
                logger.error("Text model not initialized")
                return None
            
            # Generate embedding
            embedding = self.text_model.encode(text, convert_to_tensor=True)
            
            # Convert to list and ensure correct dimension
            embedding_list = embedding.cpu().numpy().tolist()
            
            # Pad or truncate to match expected dimension
            if len(embedding_list) != self.config.EMBEDDING_DIMENSION:
                if len(embedding_list) > self.config.EMBEDDING_DIMENSION:
                    embedding_list = embedding_list[:self.config.EMBEDDING_DIMENSION]
                else:
                    embedding_list.extend([0.0] * (self.config.EMBEDDING_DIMENSION - len(embedding_list)))
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return None
    
    def generate_image_embedding(self, image_data: Union[str, Image.Image, bytes]) -> Optional[List[float]]:
        """Generate embedding for image content."""
        try:
            if not self.clip_model or not self.clip_processor:
                logger.error("CLIP model not initialized")
                return None
            
            # Handle different image input types
            if isinstance(image_data, str):
                # Assume base64 encoded image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image and generate embedding
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                
            # Normalize and convert to list
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            embedding_list = embedding.cpu().numpy().flatten().tolist()
            
            # Pad or truncate to match expected dimension
            if len(embedding_list) != self.config.EMBEDDING_DIMENSION:
                if len(embedding_list) > self.config.EMBEDDING_DIMENSION:
                    embedding_list = embedding_list[:self.config.EMBEDDING_DIMENSION]
                else:
                    embedding_list.extend([0.0] * (self.config.EMBEDDING_DIMENSION - len(embedding_list)))
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return None
    
    def generate_multimodal_embedding(self, text: str, image: Optional[Image.Image] = None) -> Optional[List[float]]:
        """Generate combined embedding for text and image."""
        try:
            if not self.clip_model or not self.clip_processor:
                logger.error("CLIP model not initialized")
                return None
            
            # Process text and image together
            inputs = self.clip_processor(
                text=[text], 
                images=[image] if image else None, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if image:
                    # Get both text and image features
                    text_features = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    image_features = self.clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                    
                    # Combine features (simple concatenation or averaging)
                    combined_features = (text_features + image_features) / 2
                else:
                    # Only text features
                    combined_features = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            # Normalize and convert to list
            embedding = combined_features / combined_features.norm(dim=-1, keepdim=True)
            embedding_list = embedding.cpu().numpy().flatten().tolist()
            
            # Pad or truncate to match expected dimension
            if len(embedding_list) != self.config.EMBEDDING_DIMENSION:
                if len(embedding_list) > self.config.EMBEDDING_DIMENSION:
                    embedding_list = embedding_list[:self.config.EMBEDDING_DIMENSION]
                else:
                    embedding_list.extend([0.0] * (self.config.EMBEDDING_DIMENSION - len(embedding_list)))
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate multimodal embedding: {e}")
            return None
    
    def batch_generate_text_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts."""
        try:
            if not self.text_model:
                logger.error("Text model not initialized")
                return [None] * len(texts)
            
            # Generate embeddings in batch
            embeddings = self.text_model.encode(texts, convert_to_tensor=True, batch_size=32)
            
            # Convert to list format
            embedding_lists = []
            for embedding in embeddings:
                embedding_list = embedding.cpu().numpy().tolist()
                
                # Pad or truncate to match expected dimension
                if len(embedding_list) != self.config.EMBEDDING_DIMENSION:
                    if len(embedding_list) > self.config.EMBEDDING_DIMENSION:
                        embedding_list = embedding_list[:self.config.EMBEDDING_DIMENSION]
                    else:
                        embedding_list.extend([0.0] * (self.config.EMBEDDING_DIMENSION - len(embedding_list)))
                
                embedding_lists.append(embedding_list)
            
            return embedding_lists
            
        except Exception as e:
            logger.error(f"Failed to generate batch text embeddings: {e}")
            return [None] * len(texts)
    
    def similarity_search(self, query_embedding: List[float], candidate_embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarity between query and candidate embeddings."""
        try:
            query_tensor = torch.tensor(query_embedding).unsqueeze(0)
            candidate_tensor = torch.tensor(candidate_embeddings)
            
            # Calculate cosine similarity
            similarities = torch.cosine_similarity(query_tensor, candidate_tensor, dim=1)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Failed to calculate similarities: {e}")
            return []

class TableEmbeddingGenerator:
    """Specialized embedding generator for table data."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
    
    def generate_table_summary_embedding(self, table_data: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for table by creating a text summary first."""
        try:
            # Create a text summary of the table
            summary = self._create_table_summary(table_data)
            
            # Generate embedding for the summary
            return self.embedding_generator.generate_text_embedding(summary)
            
        except Exception as e:
            logger.error(f"Failed to generate table embedding: {e}")
            return None
    
    def _create_table_summary(self, table_data: Dict[str, Any]) -> str:
        """Create a text summary of table data."""
        try:
            summary_parts = []
            
            # Add table metadata
            if 'title' in table_data:
                summary_parts.append(f"Table: {table_data['title']}")
            
            if 'description' in table_data:
                summary_parts.append(f"Description: {table_data['description']}")
            
            # Add column information
            if 'columns' in table_data:
                columns = table_data['columns']
                summary_parts.append(f"Columns: {', '.join(columns)}")
            
            # Add sample data or statistics
            if 'data' in table_data and table_data['data']:
                data = table_data['data']
                if isinstance(data, list) and len(data) > 0:
                    # Add first few rows as examples
                    sample_rows = data[:3]  # First 3 rows
                    for i, row in enumerate(sample_rows):
                        if isinstance(row, dict):
                            row_summary = ', '.join([f"{k}: {v}" for k, v in row.items()])
                            summary_parts.append(f"Row {i+1}: {row_summary}")
                        elif isinstance(row, list):
                            row_summary = ', '.join([str(val) for val in row])
                            summary_parts.append(f"Row {i+1}: {row_summary}")
            
            # Add statistical information if available
            if 'statistics' in table_data:
                stats = table_data['statistics']
                summary_parts.append(f"Statistics: {stats}")
            
            return ' | '.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to create table summary: {e}")
            return "Table data"

