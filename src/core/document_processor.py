import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentElement:
    """Represents an element extracted from a document."""
    id: str
    content: str
    content_type: str  # 'text', 'image', 'table'
    source: str
    metadata: Dict[str, Any]
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)

class DocumentProcessor:
    """Processes documents and extracts text, images, and tables."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_pdf(self, file_path: str) -> List[DocumentElement]:
        """Process a PDF file and extract all elements."""
        try:
            doc = fitz.open(file_path)
            elements = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text elements
                text_elements = self._extract_text_from_page(page, file_path, page_num)
                elements.extend(text_elements)
                
                # Extract image elements
                image_elements = self._extract_images_from_page(page, file_path, page_num)
                elements.extend(image_elements)
                
                # Extract table elements
                table_elements = self._extract_tables_from_page(page, file_path, page_num)
                elements.extend(table_elements)
            
            doc.close()
            logger.info(f"Successfully processed PDF: {file_path}, extracted {len(elements)} elements")
            return elements
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            return []
    
    def _extract_text_from_page(self, page, source: str, page_num: int) -> List[DocumentElement]:
        """Extract text content from a page."""
        try:
            text = page.get_text()
            if not text.strip():
                return []
            
            # Clean and chunk the text
            cleaned_text = self._clean_text(text)
            chunks = self._chunk_text(cleaned_text)
            
            elements = []
            for i, chunk in enumerate(chunks):
                element = DocumentElement(
                    id=str(uuid.uuid4()),
                    content=chunk,
                    content_type='text',
                    source=source,
                    metadata={
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'original_length': len(text)
                    },
                    page_number=page_num
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to extract text from page {page_num}: {e}")
            return []
    
    def _extract_images_from_page(self, page, source: str, page_num: int) -> List[DocumentElement]:
        """Extract images from a page."""
        try:
            image_list = page.get_images()
            elements = []
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Convert to base64 for storage
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Get image bbox if available
                        bbox = None
                        try:
                            img_rects = page.get_image_rects(xref)
                            if img_rects:
                                bbox = img_rects[0]
                        except:
                            pass
                        
                        element = DocumentElement(
                            id=str(uuid.uuid4()),
                            content=img_base64,
                            content_type='image',
                            source=source,
                            metadata={
                                'image_index': img_index,
                                'format': 'PNG',
                                'size': image.size,
                                'mode': image.mode
                            },
                            page_number=page_num,
                            bbox=bbox
                        )
                        elements.append(element)
                    
                    pix = None  # Clean up
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to extract images from page {page_num}: {e}")
            return []
    
    def _extract_tables_from_page(self, page, source: str, page_num: int) -> List[DocumentElement]:
        """Extract tables from a page."""
        try:
            # Try to find tables using text analysis
            tables = self._find_tables_in_text(page.get_text())
            elements = []
            
            for table_index, table_data in enumerate(tables):
                # Convert table to structured format
                structured_table = self._structure_table_data(table_data)
                
                element = DocumentElement(
                    id=str(uuid.uuid4()),
                    content=json.dumps(structured_table),
                    content_type='table',
                    source=source,
                    metadata={
                        'table_index': table_index,
                        'rows': len(structured_table.get('data', [])),
                        'columns': len(structured_table.get('columns', [])),
                        'table_type': 'extracted'
                    },
                    page_number=page_num
                )
                elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Failed to extract tables from page {page_num}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + self.chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _find_tables_in_text(self, text: str) -> List[List[List[str]]]:
        """Find table-like structures in text."""
        tables = []
        lines = text.split('\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                    in_table = False
                continue
            
            # Check if line looks like a table row (has multiple columns separated by whitespace or tabs)
            if self._is_table_row(line):
                row = self._parse_table_row(line)
                if row:
                    current_table.append(row)
                    in_table = True
            else:
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                    in_table = False
        
        # Add final table if exists
        if current_table:
            tables.append(current_table)
        
        # Filter out tables that are too small
        return [table for table in tables if len(table) >= 2 and len(table[0]) >= 2]
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Look for multiple columns separated by whitespace or tabs
        parts = re.split(r'\s{2,}|\t+', line.strip())
        return len(parts) >= 2
    
    def _parse_table_row(self, line: str) -> List[str]:
        """Parse a line into table columns."""
        # Split by multiple spaces or tabs
        parts = re.split(r'\s{2,}|\t+', line.strip())
        return [part.strip() for part in parts if part.strip()]
    
    def _structure_table_data(self, table_data: List[List[str]]) -> Dict[str, Any]:
        """Convert raw table data to structured format."""
        if not table_data:
            return {}
        
        # Assume first row is headers
        headers = table_data[0]
        data_rows = table_data[1:]
        
        # Create structured data
        structured_data = []
        for row in data_rows:
            # Pad row if it has fewer columns than headers
            while len(row) < len(headers):
                row.append('')
            
            # Create row dictionary
            row_dict = {}
            for i, header in enumerate(headers):
                if i < len(row):
                    row_dict[header] = row[i]
                else:
                    row_dict[header] = ''
            
            structured_data.append(row_dict)
        
        return {
            'columns': headers,
            'data': structured_data,
            'title': f'Table with {len(headers)} columns and {len(data_rows)} rows',
            'description': f'Extracted table containing: {", ".join(headers)}'
        }

class ImageProcessor:
    """Specialized processor for image analysis and description."""
    
    def __init__(self):
        pass
    
    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """Analyze image and extract metadata."""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract basic metadata
            metadata = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format or 'PNG',
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Analyze image content type
            content_type = self._classify_image_content(image)
            metadata['content_classification'] = content_type
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return {}
    
    def _classify_image_content(self, image: Image.Image) -> str:
        """Classify the type of content in the image."""
        # This is a simplified classification
        # In a real implementation, you might use a trained model
        
        width, height = image.size
        aspect_ratio = width / height
        
        # Simple heuristics
        if aspect_ratio > 2.0:
            return 'chart_or_graph'
        elif 0.5 < aspect_ratio < 2.0:
            return 'general_image'
        else:
            return 'portrait_or_diagram'
    
    def generate_image_description(self, image_data: str) -> str:
        """Generate a text description of the image."""
        # This is a placeholder - in a real implementation,
        # you would use a vision-language model like BLIP or LLaVA
        try:
            metadata = self.analyze_image(image_data)
            size = metadata.get('size', (0, 0))
            content_type = metadata.get('content_classification', 'unknown')
            
            description = f"Image of size {size[0]}x{size[1]} pixels, classified as {content_type}."
            
            return description
            
        except Exception as e:
            logger.error(f"Failed to generate image description: {e}")
            return "Image content"

class TableProcessor:
    """Specialized processor for table analysis and summarization."""
    
    def __init__(self):
        pass
    
    def analyze_table(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze table structure and content."""
        try:
            analysis = {
                'num_columns': len(table_data.get('columns', [])),
                'num_rows': len(table_data.get('data', [])),
                'column_types': {},
                'summary_statistics': {}
            }
            
            # Analyze column types and statistics
            if 'data' in table_data and table_data['data']:
                df = pd.DataFrame(table_data['data'])
                
                for column in df.columns:
                    # Try to infer data type
                    col_data = df[column].dropna()
                    if col_data.empty:
                        analysis['column_types'][column] = 'empty'
                        continue
                    
                    # Check if numeric
                    try:
                        pd.to_numeric(col_data)
                        analysis['column_types'][column] = 'numeric'
                        
                        # Calculate statistics for numeric columns
                        numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                        if not numeric_data.empty:
                            analysis['summary_statistics'][column] = {
                                'mean': float(numeric_data.mean()),
                                'median': float(numeric_data.median()),
                                'min': float(numeric_data.min()),
                                'max': float(numeric_data.max()),
                                'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0.0
                            }
                    except:
                        analysis['column_types'][column] = 'text'
                        
                        # Calculate statistics for text columns
                        analysis['summary_statistics'][column] = {
                            'unique_values': int(col_data.nunique()),
                            'most_common': str(col_data.mode().iloc[0]) if not col_data.mode().empty else '',
                            'avg_length': float(col_data.astype(str).str.len().mean())
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze table: {e}")
            return {}
    
    def generate_table_summary(self, table_data: Dict[str, Any]) -> str:
        """Generate a text summary of the table."""
        try:
            analysis = self.analyze_table(table_data)
            
            summary_parts = []
            
            # Basic structure
            summary_parts.append(f"Table with {analysis['num_rows']} rows and {analysis['num_columns']} columns")
            
            # Column information
            if 'columns' in table_data:
                columns = table_data['columns']
                summary_parts.append(f"Columns: {', '.join(columns)}")
            
            # Data type information
            if analysis['column_types']:
                numeric_cols = [col for col, dtype in analysis['column_types'].items() if dtype == 'numeric']
                text_cols = [col for col, dtype in analysis['column_types'].items() if dtype == 'text']
                
                if numeric_cols:
                    summary_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
                if text_cols:
                    summary_parts.append(f"Text columns: {', '.join(text_cols)}")
            
            # Key statistics
            if analysis['summary_statistics']:
                for col, stats in analysis['summary_statistics'].items():
                    if 'mean' in stats:
                        summary_parts.append(f"{col}: mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
                    elif 'unique_values' in stats:
                        summary_parts.append(f"{col}: {stats['unique_values']} unique values, most common='{stats['most_common']}'")
            
            return '. '.join(summary_parts) + '.'
            
        except Exception as e:
            logger.error(f"Failed to generate table summary: {e}")
            return "Table data"

