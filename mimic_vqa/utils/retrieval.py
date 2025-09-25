"""
Context retrieval utilities for finding relevant text segments
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import torch

from .bbox import BoundingBox

logger = logging.getLogger(__name__)


@dataclass 
class TextSegment:
    """Text segment with spatial information"""
    text: str
    bbox: BoundingBox
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0


class ContextRetriever:
    """Context retrieval engine using semantic similarity"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 top_k: int = 10,
                 similarity_threshold: float = 0.3,
                 mix_weight: float = 0.7):
        """
        Initialize context retriever
        
        Args:
            model_name: Sentence transformer model name
            top_k: Number of top segments to retrieve
            similarity_threshold: Minimum similarity threshold
            mix_weight: Weight for mixing semantic and spatial similarity
        """
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.mix_weight = mix_weight
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts, 
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            return np.zeros((len(texts), 384))  # Fallback
    
    def compute_semantic_similarity(self, 
                                   query_embedding: np.ndarray,
                                   segment_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and segments"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        segment_norms = segment_embeddings / np.linalg.norm(
            segment_embeddings, axis=1, keepdims=True
        )
        
        # Compute cosine similarity
        similarities = np.dot(segment_norms, query_norm)
        return similarities
    
    def compute_spatial_similarity(self,
                                 query_bbox: Optional[BoundingBox],
                                 segment_bboxes: List[BoundingBox]) -> np.ndarray:
        """
        Compute spatial similarity based on proximity
        Returns higher scores for closer segments
        """
        if query_bbox is None:
            # No spatial reference, return uniform scores
            return np.ones(len(segment_bboxes)) * 0.5
        
        similarities = []
        query_center = query_bbox.center
        
        for bbox in segment_bboxes:
            # Calculate distance and normalize
            distance = bbox.distance_to(query_bbox)
            # Convert distance to similarity (closer = higher score)
            # Use exponential decay
            max_distance = 1000  # Normalize by typical document dimensions
            similarity = np.exp(-distance / max_distance)
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def find_text(self, 
                  query: str,
                  segments: List[TextSegment],
                  query_bbox: Optional[BoundingBox] = None) -> List[TextSegment]:
        """
        Find relevant text segments using semantic and spatial similarity
        
        Implementation of FindText(Q, O) from Algorithm 1
        
        Args:
            query: Question text
            segments: List of text segments from OCR
            query_bbox: Optional spatial reference for the query
            
        Returns:
            Top-k most relevant segments
        """
        if not segments:
            return []
        
        # Encode query
        query_embedding = self.encode_texts([query])[0]
        
        # Encode all segment texts if not already done
        texts_to_encode = []
        indices_to_encode = []
        
        for i, segment in enumerate(segments):
            if segment.embedding is None:
                texts_to_encode.append(segment.text)
                indices_to_encode.append(i)
        
        if texts_to_encode:
            embeddings = self.encode_texts(texts_to_encode)
            for i, idx in enumerate(indices_to_encode):
                segments[idx].embedding = embeddings[i]
        
        # Get all embeddings
        segment_embeddings = np.stack([seg.embedding for seg in segments])
        
        # Compute semantic similarity
        semantic_scores = self.compute_semantic_similarity(
            query_embedding, segment_embeddings
        )
        
        # Compute spatial similarity
        spatial_scores = self.compute_spatial_similarity(
            query_bbox, [seg.bbox for seg in segments]
        )
        
        # Combine scores
        combined_scores = (
            self.mix_weight * semantic_scores + 
            (1 - self.mix_weight) * spatial_scores
        )
        
        # Filter by threshold
        valid_indices = combined_scores >= self.similarity_threshold
        
        if not np.any(valid_indices):
            # If no segments meet threshold, take top segments anyway
            valid_indices = np.ones_like(combined_scores, dtype=bool)
        
        # Get top-k segments
        valid_scores = combined_scores[valid_indices]
        valid_segments = [seg for i, seg in enumerate(segments) if valid_indices[i]]
        
        # Sort by score (descending)
        sorted_indices = np.argsort(valid_scores)[::-1]
        top_segments = [valid_segments[i] for i in sorted_indices[:self.top_k]]
        
        return top_segments
    
    def find_text_from_ocr_results(self,
                                  query: str,
                                  ocr_results: List[Tuple[str, BoundingBox]],
                                  query_bbox: Optional[BoundingBox] = None) -> List[Tuple[str, BoundingBox]]:
        """
        Find text from OCR results in the format expected by Algorithm 1
        
        Args:
            query: Question text
            ocr_results: List of (text, bbox) tuples from RunOCR
            query_bbox: Optional spatial reference
            
        Returns:
            Relevant (text, bbox) tuples
        """
        # Convert to TextSegment objects
        segments = [
            TextSegment(text=text, bbox=bbox)
            for text, bbox in ocr_results
        ]
        
        # Find relevant segments
        relevant_segments = self.find_text(query, segments, query_bbox)
        
        # Convert back to tuple format
        return [(seg.text, seg.bbox) for seg in relevant_segments]
    
    def build_context_string(self, 
                           segments: List[TextSegment],
                           max_length: int = 512) -> str:
        """
        Build context string from relevant segments
        
        Args:
            segments: Relevant text segments
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        if not segments:
            return ""
        
        # Sort segments by reading order (top-to-bottom, left-to-right)
        sorted_segments = sorted(
            segments,
            key=lambda seg: (seg.bbox.y, seg.bbox.x)
        )
        
        context_parts = []
        total_length = 0
        
        for segment in sorted_segments:
            text = segment.text.strip()
            if not text:
                continue
            
            # Check if adding this segment would exceed max length
            if total_length + len(text) + 1 > max_length:
                if total_length == 0:
                    # First segment is too long, truncate it
                    available = max_length - 3  # Account for "..."
                    text = text[:available] + "..."
                    context_parts.append(text)
                break
            
            context_parts.append(text)
            total_length += len(text) + 1  # +1 for space
        
        return " ".join(context_parts)
    
    def get_context_with_bbox_info(self,
                                  segments: List[TextSegment]) -> Dict[str, Any]:
        """
        Get context with bounding box information for grounding
        
        Returns:
            Dictionary with context text and bbox mappings
        """
        if not segments:
            return {"context": "", "text_to_bbox": {}}
        
        # Sort by reading order
        sorted_segments = sorted(
            segments,
            key=lambda seg: (seg.bbox.y, seg.bbox.x)
        )
        
        context_parts = []
        text_to_bbox = {}
        
        for segment in sorted_segments:
            text = segment.text.strip()
            if text:
                context_parts.append(text)
                text_to_bbox[text] = segment.bbox
        
        return {
            "context": " ".join(context_parts),
            "text_to_bbox": text_to_bbox,
            "segments": sorted_segments
        }


def find_text(query: str, 
             ocr_outputs: List[Tuple[str, BoundingBox]],
             top_k: int = 10,
             similarity_threshold: float = 0.3,
             mix_weight: float = 0.7) -> List[Tuple[str, BoundingBox]]:
    """
    Convenience function implementing FindText from Algorithm 1
    
    Args:
        query: Question text Q
        ocr_outputs: OCR outputs O = {(ti, bi)}
        top_k: Number of top segments to retrieve
        similarity_threshold: Minimum similarity threshold  
        mix_weight: Weight for semantic vs spatial similarity
        
    Returns:
        Context C ⊂ O with top-k relevant segments
    """
    retriever = ContextRetriever(
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        mix_weight=mix_weight
    )
    
    return retriever.find_text_from_ocr_results(query, ocr_outputs)
