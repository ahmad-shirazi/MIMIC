"""
OCR extraction utilities for document analysis
"""
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Dict, Any
import logging

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from .bbox import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR extraction"""
    text: str
    bbox: BoundingBox
    confidence: float
    
    def __post_init__(self):
        """Update bbox confidence with OCR confidence"""
        self.bbox.confidence = self.confidence


class OCRExtractor:
    """OCR extraction engine supporting multiple backends"""
    
    def __init__(self, 
                 engine: str = "paddleocr",
                 confidence_threshold: float = 0.5,
                 languages: List[str] = ['en']):
        """
        Initialize OCR extractor
        
        Args:
            engine: OCR engine ("paddleocr" or "easyocr")
            confidence_threshold: Minimum confidence for text detection
            languages: List of language codes
        """
        self.engine = engine.lower()
        self.confidence_threshold = confidence_threshold
        self.languages = languages
        
        # Initialize OCR engine
        if self.engine == "paddleocr":
            if not PADDLEOCR_AVAILABLE:
                raise ImportError("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en' if 'en' in languages else languages[0],
                show_log=False
            )
        elif self.engine == "easyocr":
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR not available. Install with: pip install easyocr")
            self.ocr = easyocr.Reader(languages, gpu=True)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def extract_text(self, 
                     image: Union[str, np.ndarray, Image.Image]) -> List[OCRResult]:
        """
        Extract text from image using OCR
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            List of OCR results with text, bounding boxes, and confidence
        """
        # Convert image to appropriate format
        if isinstance(image, str):
            # File path
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            # PIL Image
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Numpy array
            img_array = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Perform OCR extraction
        if self.engine == "paddleocr":
            return self._extract_paddleocr(img_array)
        elif self.engine == "easyocr":
            return self._extract_easyocr(img_array)
    
    def _extract_paddleocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using PaddleOCR"""
        try:
            results = self.ocr.ocr(image, cls=True)
            
            ocr_results = []
            for line_result in results[0] if results[0] else []:
                # Extract coordinates and text
                coords = line_result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line_result[1]  # (text, confidence)
                
                text = text_info[0].strip()
                confidence = float(text_info[1])
                
                # Skip if below confidence threshold or empty
                if confidence < self.confidence_threshold or not text:
                    continue
                
                # Convert coordinates to bounding box
                x_coords = [point[0] for point in coords]
                y_coords = [point[1] for point in coords]
                
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                bbox = BoundingBox.from_xyxy(
                    int(x1), int(y1), int(x2), int(y2), confidence
                )
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return []
    
    def _extract_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text using EasyOCR"""
        try:
            results = self.ocr.readtext(image)
            
            ocr_results = []
            for result in results:
                coords = result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = result[1].strip()
                confidence = float(result[2])
                
                # Skip if below confidence threshold or empty
                if confidence < self.confidence_threshold or not text:
                    continue
                
                # Convert coordinates to bounding box
                x_coords = [point[0] for point in coords]
                y_coords = [point[1] for point in coords]
                
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                bbox = BoundingBox.from_xyxy(
                    int(x1), int(y1), int(x2), int(y2), confidence
                )
                
                ocr_results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def extract_with_filtering(self,
                              image: Union[str, np.ndarray, Image.Image],
                              min_text_length: int = 2,
                              max_text_length: int = 200,
                              filter_patterns: Optional[List[str]] = None) -> List[OCRResult]:
        """
        Extract text with additional filtering
        
        Args:
            image: Input image
            min_text_length: Minimum text length to keep
            max_text_length: Maximum text length to keep
            filter_patterns: Regex patterns to filter out (e.g., ['\\d+\\.\\d+'])
            
        Returns:
            Filtered OCR results
        """
        import re
        
        results = self.extract_text(image)
        filtered_results = []
        
        for result in results:
            text = result.text
            
            # Length filtering
            if len(text) < min_text_length or len(text) > max_text_length:
                continue
            
            # Pattern filtering
            if filter_patterns:
                skip = False
                for pattern in filter_patterns:
                    if re.match(pattern, text):
                        skip = True
                        break
                if skip:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def get_text_segments(self, 
                         image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Get OCR results in the format expected by the teacher agent
        
        Returns:
            Dictionary with format: {(text, bbox): confidence}
        """
        results = self.extract_text(image)
        
        segments = {}
        for result in results:
            # Create key as (text, bbox_tuple)
            bbox_tuple = (result.bbox.x, result.bbox.y, result.bbox.w, result.bbox.h)
            key = (result.text, bbox_tuple)
            segments[key] = result.confidence
        
        return segments
    
    def visualize_results(self, 
                         image: Union[str, np.ndarray, Image.Image],
                         results: List[OCRResult],
                         output_path: Optional[str] = None) -> np.ndarray:
        """Visualize OCR results on image"""
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Draw bounding boxes and text
        for result in results:
            bbox = result.bbox
            
            # Draw rectangle
            cv2.rectangle(img, (bbox.x, bbox.y), (bbox.x2, bbox.y2), 
                         (0, 255, 0), 2)
            
            # Add text label
            label = f"{result.text[:20]}... ({result.confidence:.2f})"
            cv2.putText(img, label, (bbox.x, bbox.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img


def run_ocr(image: Union[str, np.ndarray, Image.Image], 
           engine: str = "paddleocr",
           confidence_threshold: float = 0.5) -> List[Tuple[str, BoundingBox]]:
    """
    Convenience function to run OCR extraction
    
    Args:
        image: Input image
        engine: OCR engine to use
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        List of (text, bbox) tuples matching Algorithm 1 format
    """
    extractor = OCRExtractor(engine, confidence_threshold)
    results = extractor.extract_text(image)
    
    # Return in format expected by Algorithm 1: O = RunOCR(I) = {(ti, bi)}
    return [(result.text, result.bbox) for result in results]
