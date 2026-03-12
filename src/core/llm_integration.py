import openai
import logging
from typing import List, Dict, Any, Optional
import json
from config.settings import Config
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

class LLMIntegrator:
    """Integrates with Large Language Models for response generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        
    def initialize(self) -> bool:
        """Initialize the LLM client."""
        try:
            if not self.config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not provided. Using mock responses.")
                return True
            
            openai.api_key = self.config.OPENAI_API_KEY
            self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            # Test the connection
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                logger.info("Successfully initialized OpenAI client")
                return True
            except Exception as e:
                logger.warning(f"OpenAI API test failed: {e}. Using mock responses.")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return False
    
    def generate_response(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """Generate a response based on query and retrieved documents."""
        try:
            if not self.client or not self.config.OPENAI_API_KEY:
                return self._generate_mock_response(query, retrieval_results)
            
            # Prepare context from retrieval results
            context = self._prepare_context(retrieval_results)
            
            # Create the prompt
            prompt = self._create_rag_prompt(query, context)
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context. Always cite your sources and be accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return self._generate_mock_response(query, retrieval_results)
    
    def generate_image_description(self, image_base64: str) -> str:
        """Generate description for an image using vision model."""
        try:
            if not self.client or not self.config.OPENAI_API_KEY:
                return "Image description not available (OpenAI API key required)"
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail, focusing on any text, charts, graphs, or important visual elements."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate image description: {e}")
            return "Image description not available"
    
    def summarize_table(self, table_data: Dict[str, Any]) -> str:
        """Generate a summary of table data."""
        try:
            if not self.client or not self.config.OPENAI_API_KEY:
                return self._generate_mock_table_summary(table_data)
            
            # Convert table data to text format
            table_text = self._table_to_text(table_data)
            
            prompt = f"""
            Please provide a concise summary of the following table data:
            
            {table_text}
            
            Include:
            1. What the table is about
            2. Key findings or patterns
            3. Important statistics or values
            4. Any notable trends
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Provide clear, concise summaries of tabular data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize table: {e}")
            return self._generate_mock_table_summary(table_data)
    
    def chat(self, message: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Handle conversational chat."""
        try:
            if not self.client or not self.config.OPENAI_API_KEY:
                return self._generate_mock_chat_response(message)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for a multi-modal RAG system. You can help users understand documents, images, and tables."}
            ]
            
            # Add conversation history
            if conversation_history:
                for turn in conversation_history[-10:]:  # Keep last 10 turns
                    messages.append({"role": "user", "content": turn.get("user", "")})
                    messages.append({"role": "assistant", "content": turn.get("assistant", "")})
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate chat response: {e}")
            return self._generate_mock_chat_response(message)
    
    def _prepare_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """Prepare context from retrieval results."""
        context_parts = []
        
        for i, result in enumerate(retrieval_results):
            element = result.element
            
            context_part = f"Source {i+1} (Score: {result.score:.2f}):\n"
            context_part += f"Type: {element.content_type}\n"
            context_part += f"Source: {element.source} (Page {element.page_number})\n"
            
            if element.content_type == 'text':
                context_part += f"Content: {element.content}\n"
            elif element.content_type == 'image':
                description = element.metadata.get('description', 'Image content')
                context_part += f"Image Description: {description}\n"
            elif element.content_type == 'table':
                summary = element.metadata.get('summary', 'Table data')
                context_part += f"Table Summary: {summary}\n"
            
            context_part += f"Relevance: {result.relevance_explanation}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create a RAG prompt for the LLM."""
        prompt = f"""
        Based on the following context from retrieved documents, please answer the user's question.
        
        Context:
        {context}
        
        Question: {query}
        
        Instructions:
        1. Answer based only on the provided context
        2. If the context doesn't contain enough information, say so
        3. Cite specific sources when making claims
        4. Be concise but comprehensive
        5. If there are images or tables mentioned, reference them appropriately
        
        Answer:
        """
        
        return prompt
    
    def _table_to_text(self, table_data: Dict[str, Any]) -> str:
        """Convert table data to text format."""
        try:
            text_parts = []
            
            if 'title' in table_data:
                text_parts.append(f"Table: {table_data['title']}")
            
            if 'columns' in table_data:
                text_parts.append(f"Columns: {', '.join(table_data['columns'])}")
            
            if 'data' in table_data and table_data['data']:
                text_parts.append("Data:")
                for i, row in enumerate(table_data['data'][:5]):  # First 5 rows
                    if isinstance(row, dict):
                        row_text = ', '.join([f"{k}: {v}" for k, v in row.items()])
                        text_parts.append(f"  Row {i+1}: {row_text}")
                
                if len(table_data['data']) > 5:
                    text_parts.append(f"  ... and {len(table_data['data']) - 5} more rows")
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Failed to convert table to text: {e}")
            return "Table data"
    
    def _generate_mock_response(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """Generate a mock response when OpenAI is not available."""
        if not retrieval_results:
            return f"I couldn't find any relevant information to answer your question: '{query}'. Please try uploading some documents first."
        
        response_parts = [
            f"Based on the retrieved documents, here's what I found regarding '{query}':",
            ""
        ]
        
        for i, result in enumerate(retrieval_results[:3]):  # Top 3 results
            element = result.element
            response_parts.append(f"{i+1}. From {element.source} (Page {element.page_number}):")
            
            if element.content_type == 'text':
                content_preview = element.content[:200] + "..." if len(element.content) > 200 else element.content
                response_parts.append(f"   {content_preview}")
            elif element.content_type == 'image':
                description = element.metadata.get('description', 'Image content')
                response_parts.append(f"   Image: {description}")
            elif element.content_type == 'table':
                summary = element.metadata.get('summary', 'Table data')
                response_parts.append(f"   Table: {summary}")
            
            response_parts.append("")
        
        response_parts.append("Note: This is a mock response. Configure OpenAI API key for enhanced responses.")
        
        return '\n'.join(response_parts)
    
    def _generate_mock_table_summary(self, table_data: Dict[str, Any]) -> str:
        """Generate a mock table summary."""
        try:
            num_rows = len(table_data.get('data', []))
            num_cols = len(table_data.get('columns', []))
            columns = table_data.get('columns', [])
            
            summary = f"This table contains {num_rows} rows and {num_cols} columns. "
            if columns:
                summary += f"The columns are: {', '.join(columns)}. "
            
            summary += "Note: Configure OpenAI API key for detailed table analysis."
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate mock table summary: {e}")
            return "Table summary not available"
    
    def _generate_mock_chat_response(self, message: str) -> str:
        """Generate a mock chat response."""
        responses = [
            f"I understand you're asking about: '{message}'. I'm a multi-modal RAG assistant that can help you analyze documents, images, and tables.",
            "To get started, please upload some PDF documents using the upload feature.",
            "Once you have documents uploaded, I can help you search through them and answer questions based on their content.",
            "Note: Configure OpenAI API key for enhanced conversational capabilities."
        ]
        
        return ' '.join(responses)

