"""
Answer grounding utilities for mapping textual answers to spatial coordinates
"""
import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging

from .bbox import BoundingBox, merge_bounding_boxes

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result from answer grounding"""
    answer_text: str
    bbox: BoundingBox
    confidence: float
    matched_text: str
    grounding_method: str


class AnswerGrounder:
    """Answer grounding engine using ANLS alignment"""
    
    def __init__(self, 
                 anls_threshold: float = 0.5,
                 min_confidence: float = 0.3):
        """
        Initialize answer grounder
        
        Args:
            anls_threshold: ANLS threshold for confident matches
            min_confidence: Minimum confidence for accepting matches
        """
        self.anls_threshold = anls_threshold
        self.min_confidence = min_confidence
    
    def compute_anls(self, pred: str, target: str) -> float:
        """
        Compute ANLS (Average Normalized Levenshtein Similarity)
        
        ANLS(a,t) = 1 - min(1, Lev(a,t) / max(|a|,|t|))
        
        Args:
            pred: Predicted text
            target: Target text
            
        Returns:
            ANLS score between 0 and 1
        """
        if not pred.strip() or not target.strip():
            return 0.0
        
        pred = pred.strip().lower()
        target = target.strip().lower()
        
        if pred == target:
            return 1.0
        
        # Compute Levenshtein distance
        levenshtein_dist = self._levenshtein_distance(pred, target)
        max_len = max(len(pred), len(target))
        
        if max_len == 0:
            return 1.0
        
        # Compute ANLS
        anls_score = 1.0 - min(1.0, levenshtein_dist / max_len)
        return anls_score
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def ground_answer(self,
                     answer_text: str,
                     ocr_outputs: List[Tuple[str, BoundingBox]]) -> Optional[GroundingResult]:
        """
        Ground textual answer to spatial coordinates using ANLS alignment
        
        Implementation of GroundAnswer(Atext, O) from Algorithm 2
        
        Args:
            answer_text: Textual answer to ground
            ocr_outputs: OCR outputs O = {(ti, bi, confi)}
            
        Returns:
            Grounding result with bounding box and confidence
        """
        if not answer_text.strip() or not ocr_outputs:
            return None
        
        answer_text = answer_text.strip()
        matches = []
        
        # Step 1: Find best matching text segments using ANLS
        for i, (text, bbox) in enumerate(ocr_outputs):
            if not text.strip():
                continue
            
            # Compute ANLS similarity
            anls_score = self.compute_anls(answer_text, text)
            
            # Mark as confident match if above threshold
            is_confident = anls_score > self.anls_threshold
            
            matches.append({
                'index': i,
                'text': text,
                'bbox': bbox,
                'anls_score': anls_score,
                'is_confident': is_confident
            })
        
        if not matches:
            return None
        
        # Step 2: Select confident matches or best matches
        confident_matches = [m for m in matches if m['is_confident']]
        
        if confident_matches:
            # Use confident matches
            selected_matches = confident_matches
        else:
            # No confident matches, select top matches
            matches.sort(key=lambda x: x['anls_score'], reverse=True)
            # Take top matches that are above minimum confidence
            selected_matches = [
                m for m in matches[:3] 
                if m['anls_score'] >= self.min_confidence
            ]
        
        if not selected_matches:
            return None
        
        # Step 3: Aggregate matches to single bounding box
        bboxes = [m['bbox'] for m in selected_matches]
        confidence_scores = [m['anls_score'] for m in selected_matches]
        
        if len(bboxes) == 1:
            # Single match
            final_bbox = bboxes[0]
            final_confidence = confidence_scores[0]
            matched_text = selected_matches[0]['text']
        else:
            # Multiple matches - aggregate
            final_bbox = self._aggregate_bboxes(bboxes, confidence_scores)
            final_confidence = np.mean(confidence_scores)
            matched_text = " | ".join([m['text'] for m in selected_matches])
        
        return GroundingResult(
            answer_text=answer_text,
            bbox=final_bbox,
            confidence=final_confidence,
            matched_text=matched_text,
            grounding_method="anls"
        )
    
    def _aggregate_bboxes(self, 
                         bboxes: List[BoundingBox],
                         confidences: List[float]) -> BoundingBox:
        """
        Aggregate multiple bounding boxes into single box
        Uses weighted average for coordinates based on confidence
        """
        if len(bboxes) == 1:
            return bboxes[0]
        
        # Method 1: Use min/max coordinates (encompasses all)
        min_x = min(bbox.x for bbox in bboxes)
        min_y = min(bbox.y for bbox in bboxes)
        max_x2 = max(bbox.x2 for bbox in bboxes)
        max_y2 = max(bbox.y2 for bbox in bboxes)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences)
        
        return BoundingBox.from_xyxy(min_x, min_y, max_x2, max_y2, avg_confidence)
    
    def ground_answer_with_context_search(self,
                                        answer_text: str,
                                        context_text: str,
                                        text_to_bbox: Dict[str, BoundingBox]) -> Optional[GroundingResult]:
        """
        Ground answer by searching within context text
        
        Args:
            answer_text: Answer to ground
            context_text: Context text from retrieval
            text_to_bbox: Mapping from text segments to bounding boxes
            
        Returns:
            Grounding result if successful
        """
        answer_text = answer_text.strip().lower()
        
        # Strategy 1: Exact substring match
        context_lower = context_text.lower()
        if answer_text in context_lower:
            # Find the text segment containing this answer
            for text, bbox in text_to_bbox.items():
                if answer_text in text.lower():
                    return GroundingResult(
                        answer_text=answer_text,
                        bbox=bbox,
                        confidence=1.0,
                        matched_text=text,
                        grounding_method="exact_match"
                    )
        
        # Strategy 2: ANLS-based matching
        best_match = None
        best_score = 0.0
        
        for text, bbox in text_to_bbox.items():
            anls_score = self.compute_anls(answer_text, text)
            if anls_score > best_score and anls_score >= self.min_confidence:
                best_score = anls_score
                best_match = GroundingResult(
                    answer_text=answer_text,
                    bbox=bbox,
                    confidence=anls_score,
                    matched_text=text,
                    grounding_method="anls_context"
                )
        
        return best_match
    
    def ground_multiple_answers(self,
                              answers: List[str],
                              ocr_outputs: List[Tuple[str, BoundingBox]]) -> List[GroundingResult]:
        """
        Ground multiple answers simultaneously
        
        Args:
            answers: List of answer texts
            ocr_outputs: OCR outputs
            
        Returns:
            List of grounding results
        """
        results = []
        used_indices = set()  # Track used OCR segments
        
        for answer in answers:
            # Filter out already used segments
            available_outputs = [
                (text, bbox) for i, (text, bbox) in enumerate(ocr_outputs)
                if i not in used_indices
            ]
            
            result = self.ground_answer(answer, available_outputs)
            if result:
                results.append(result)
                
                # Mark segments as used (simplified - could be more sophisticated)
                for i, (text, bbox) in enumerate(ocr_outputs):
                    if text == result.matched_text:
                        used_indices.add(i)
                        break
        
        return results
    
    def visualize_grounding(self,
                           image: np.ndarray,
                           grounding_result: GroundingResult,
                           color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Visualize grounding result on image"""
        import cv2
        
        img = image.copy()
        bbox = grounding_result.bbox
        
        # Draw bounding box
        cv2.rectangle(img, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 3)
        
        # Add label
        label = f"Answer: {grounding_result.answer_text} ({grounding_result.confidence:.2f})"
        cv2.putText(img, label, (bbox.x, bbox.y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img


def ground_answer(answer_text: str,
                 ocr_outputs: List[Tuple[str, BoundingBox]],
                 anls_threshold: float = 0.5) -> Tuple[BoundingBox, float]:
    """
    Convenience function implementing GroundAnswer from Algorithm 2
    
    Args:
        answer_text: Textual answer Atext
        ocr_outputs: OCR outputs O
        anls_threshold: ANLS threshold for confident matches
        
    Returns:
        Tuple of (bounding_box, confidence_score) as (BA, score)
    """
    grounder = AnswerGrounder(anls_threshold=anls_threshold)
    result = grounder.ground_answer(answer_text, ocr_outputs)
    
    if result is None:
        # Return empty bbox with zero confidence if no grounding found
        empty_bbox = BoundingBox(0, 0, 0, 0, 0.0)
        return empty_bbox, 0.0
    
    return result.bbox, result.confidence
