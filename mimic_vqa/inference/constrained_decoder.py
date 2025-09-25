"""
Constrained decoding module for spatial grounding
"""
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
import re

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
import numpy as np

from ..config import Config
from ..utils.bbox import BoundingBox
from ..utils.grounding import AnswerGrounder

logger = logging.getLogger(__name__)


class ConstrainedDecoder:
    """
    Constrained decoding for bounding box generation
    
    Implementation of the optional constrained decoding module (gray box in Figure 1).
    Uses OCR coordinates to restrict vocabulary during bbox token generation.
    """
    
    def __init__(self, 
                 config: Config,
                 tokenizer: PreTrainedTokenizer):
        """
        Initialize constrained decoder
        
        Args:
            config: MIMIC-VQA configuration
            tokenizer: Model tokenizer
        """
        self.config = config
        self.tokenizer = tokenizer
        self.coordinate_tolerance = config.inference.coordinate_tolerance  # ±5 pixels
        
        # Answer grounder for finding relevant OCR segments
        self.answer_grounder = AnswerGrounder()
        
        # Cache for coordinate tokens
        self._coordinate_token_cache = {}
        self._build_coordinate_vocabulary()
        
        logger.info("Constrained decoder initialized")
    
    def _build_coordinate_vocabulary(self):
        """Build vocabulary of coordinate tokens"""
        # Get tokens for numbers 0-9999 (covers typical document coordinates)
        self.number_tokens = set()
        self.token_to_number = {}
        
        for num in range(10000):
            token_ids = self.tokenizer.encode(str(num), add_special_tokens=False)
            for token_id in token_ids:
                self.number_tokens.add(token_id)
                token_text = self.tokenizer.decode([token_id])
                if token_text.isdigit():
                    self.token_to_number[token_id] = int(token_text)
        
        logger.info(f"Built coordinate vocabulary with {len(self.number_tokens)} number tokens")
    
    def generate_bbox(self,
                     answer_text: str,
                     ocr_data: List[Tuple[str, BoundingBox]],
                     context: str,
                     use_fast_ocr: bool = True) -> BoundingBox:
        """
        Generate bounding box with constrained decoding
        
        Args:
            answer_text: Generated answer text
            ocr_data: OCR results with bounding boxes
            context: Additional context
            use_fast_ocr: Whether to use fast OCR coordinate building
            
        Returns:
            Generated bounding box with constraints
        """
        # Step 1: Find relevant OCR segments for the answer
        relevant_segments = self._find_relevant_ocr_segments(answer_text, ocr_data)
        
        if not relevant_segments:
            logger.warning("No relevant OCR segments found for answer")
            return BoundingBox(0, 0, 0, 0, 0.0)
        
        # Step 2: Build valid coordinate ranges from OCR
        if use_fast_ocr:
            valid_coordinates = self._build_coordinate_constraints_fast(relevant_segments)
        else:
            valid_coordinates = self._build_coordinate_constraints_precise(relevant_segments, ocr_data)
        
        # Step 3: Generate constrained bounding box
        constrained_bbox = self._generate_constrained_coordinates(
            answer_text, valid_coordinates, relevant_segments
        )
        
        return constrained_bbox
    
    def _find_relevant_ocr_segments(self,
                                  answer_text: str,
                                  ocr_data: List[Tuple[str, BoundingBox]]) -> List[Tuple[str, BoundingBox]]:
        """Find OCR segments relevant to the answer"""
        # Use answer grounder to find matches
        grounding_result = self.answer_grounder.ground_answer(answer_text, ocr_data)
        
        if grounding_result and grounding_result.confidence > 0.3:
            # Find the specific OCR segments that matched
            relevant_segments = []
            for text, bbox in ocr_data:
                anls_score = self.answer_grounder.compute_anls(answer_text, text)
                if anls_score > 0.3:  # Threshold for relevance
                    relevant_segments.append((text, bbox))
            
            return relevant_segments
        
        # Fallback: simple text matching
        answer_lower = answer_text.lower()
        relevant_segments = []
        
        for text, bbox in ocr_data:
            text_lower = text.lower()
            
            # Exact match
            if answer_lower in text_lower or text_lower in answer_lower:
                relevant_segments.append((text, bbox))
            # Partial word match
            elif len(set(answer_lower.split()) & set(text_lower.split())) > 0:
                relevant_segments.append((text, bbox))
        
        return relevant_segments[:5]  # Limit to top 5 for efficiency
    
    def _build_coordinate_constraints_fast(self,
                                         relevant_segments: List[Tuple[str, BoundingBox]]) -> Dict[str, Set[int]]:
        """Build coordinate constraints using fast method"""
        constraints = {
            'x': set(),
            'y': set(), 
            'w': set(),
            'h': set()
        }
        
        for text, bbox in relevant_segments:
            # Add coordinates with tolerance
            x_range = range(
                max(0, bbox.x - self.coordinate_tolerance),
                bbox.x + bbox.w + self.coordinate_tolerance + 1
            )
            y_range = range(
                max(0, bbox.y - self.coordinate_tolerance),
                bbox.y + bbox.h + self.coordinate_tolerance + 1
            )
            
            constraints['x'].update(x_range)
            constraints['y'].update(y_range)
            
            # Width and height constraints (reasonable ranges)
            w_range = range(max(1, bbox.w - self.coordinate_tolerance),
                          bbox.w + self.coordinate_tolerance + 1)
            h_range = range(max(1, bbox.h - self.coordinate_tolerance),
                          bbox.h + self.coordinate_tolerance + 1)
            
            constraints['w'].update(w_range)
            constraints['h'].update(h_range)
        
        return constraints
    
    def _build_coordinate_constraints_precise(self,
                                            relevant_segments: List[Tuple[str, BoundingBox]],
                                            all_ocr_data: List[Tuple[str, BoundingBox]]) -> Dict[str, Set[int]]:
        """Build coordinate constraints using precise OCR-based method"""
        # Implementation of mathematical formulation from the paper:
        # V_bbox^(p) = {str(c) : c ∈ C_valid^(p)}
        
        constraints = {
            'x': set(),
            'y': set(),
            'w': set(), 
            'h': set()
        }
        
        # Extract all valid coordinate values from OCR
        for text, bbox in all_ocr_data:
            # Add exact coordinates
            constraints['x'].add(bbox.x)
            constraints['y'].add(bbox.y)
            constraints['w'].add(bbox.w)
            constraints['h'].add(bbox.h)
            
            # Add coordinate boundaries
            constraints['x'].add(bbox.x + bbox.w)  # Right edge
            constraints['y'].add(bbox.y + bbox.h)  # Bottom edge
        
        # Expand constraints around relevant segments
        for text, bbox in relevant_segments:
            # Add tolerance around relevant segments
            for dx in range(-self.coordinate_tolerance, self.coordinate_tolerance + 1):
                for dy in range(-self.coordinate_tolerance, self.coordinate_tolerance + 1):
                    constraints['x'].add(max(0, bbox.x + dx))
                    constraints['y'].add(max(0, bbox.y + dy))
                    constraints['x'].add(max(0, bbox.x + bbox.w + dx))
                    constraints['y'].add(max(0, bbox.y + bbox.h + dy))
        
        return constraints
    
    def _generate_constrained_coordinates(self,
                                        answer_text: str,
                                        valid_coordinates: Dict[str, Set[int]],
                                        relevant_segments: List[Tuple[str, BoundingBox]]) -> BoundingBox:
        """Generate bounding box coordinates with constraints"""
        
        if not relevant_segments:
            return BoundingBox(0, 0, 0, 0, 0.0)
        
        # Strategy 1: Use weighted average of relevant segments
        if len(relevant_segments) == 1:
            # Single segment - use its coordinates with small adjustment
            _, bbox = relevant_segments[0]
            return BoundingBox(bbox.x, bbox.y, bbox.w, bbox.h, 0.9)
        
        # Multiple segments - find best encompassing box
        x_coords = [bbox.x for _, bbox in relevant_segments]
        y_coords = [bbox.y for _, bbox in relevant_segments]
        x2_coords = [bbox.x + bbox.w for _, bbox in relevant_segments]
        y2_coords = [bbox.y + bbox.h for _, bbox in relevant_segments]
        
        # Find coordinates that encompass relevant segments
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x2 = max(x2_coords)
        max_y2 = max(y2_coords)
        
        # Constrain to valid coordinates
        constrained_x = self._find_closest_valid_coordinate(min_x, valid_coordinates['x'])
        constrained_y = self._find_closest_valid_coordinate(min_y, valid_coordinates['y'])
        
        # Calculate constrained width and height
        target_w = max_x2 - constrained_x
        target_h = max_y2 - constrained_y
        
        constrained_w = self._find_closest_valid_coordinate(target_w, valid_coordinates['w'])
        constrained_h = self._find_closest_valid_coordinate(target_h, valid_coordinates['h'])
        
        # Ensure positive dimensions
        constrained_w = max(1, constrained_w)
        constrained_h = max(1, constrained_h)
        
        confidence = 0.8  # High confidence due to OCR constraints
        
        return BoundingBox(constrained_x, constrained_y, constrained_w, constrained_h, confidence)
    
    def _find_closest_valid_coordinate(self, target: int, valid_coords: Set[int]) -> int:
        """Find closest valid coordinate to target"""
        if target in valid_coords:
            return target
        
        if not valid_coords:
            return max(0, target)  # Fallback
        
        # Find closest valid coordinate
        valid_list = sorted(valid_coords)
        closest = min(valid_list, key=lambda x: abs(x - target))
        
        return closest
    
    def create_vocabulary_mask(self,
                              valid_coordinates: Dict[str, Set[int]],
                              coordinate_position: str) -> torch.Tensor:
        """
        Create vocabulary mask for constrained generation
        
        Implementation of vocabulary masking with constrained softmax
        
        Args:
            valid_coordinates: Valid coordinate sets
            coordinate_position: Which coordinate ('x', 'y', 'w', 'h')
            
        Returns:
            Boolean mask for valid tokens
        """
        vocab_size = len(self.tokenizer)
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        
        if coordinate_position not in valid_coordinates:
            logger.warning(f"Unknown coordinate position: {coordinate_position}")
            return torch.ones(vocab_size, dtype=torch.bool)  # Allow all tokens
        
        valid_coords = valid_coordinates[coordinate_position]
        
        # Mark valid coordinate tokens
        for coord in valid_coords:
            coord_str = str(coord)
            token_ids = self.tokenizer.encode(coord_str, add_special_tokens=False)
            for token_id in token_ids:
                if token_id < vocab_size:
                    mask[token_id] = True
        
        # Always allow special tokens
        special_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
        ]
        
        for token_id in special_tokens:
            if token_id is not None and token_id < vocab_size:
                mask[token_id] = True
        
        # Allow punctuation tokens (commas, brackets, spaces)
        punctuation_tokens = self.tokenizer.encode(", [] ()", add_special_tokens=False)
        for token_id in punctuation_tokens:
            if token_id < vocab_size:
                mask[token_id] = True
        
        return mask
    
    def apply_constrained_generation(self,
                                   logits: torch.Tensor,
                                   valid_coordinates: Dict[str, Set[int]],
                                   current_position: str,
                                   temperature: float = 1.0) -> torch.Tensor:
        """
        Apply constrained generation to logits
        
        Args:
            logits: Model logits
            valid_coordinates: Valid coordinate constraints  
            current_position: Current coordinate being generated
            temperature: Sampling temperature
            
        Returns:
            Constrained probability distribution
        """
        # Create vocabulary mask
        mask = self.create_vocabulary_mask(valid_coordinates, current_position)
        
        # Apply mask (set invalid tokens to very low probability)
        constrained_logits = logits.clone()
        constrained_logits[~mask] = float('-inf')
        
        # Apply temperature scaling
        if temperature != 1.0:
            constrained_logits = constrained_logits / temperature
        
        # Convert to probabilities with softmax
        probabilities = F.softmax(constrained_logits, dim=-1)
        
        return probabilities
    
    def validate_generated_bbox(self,
                               generated_bbox: BoundingBox,
                               valid_coordinates: Dict[str, Set[int]],
                               ocr_data: List[Tuple[str, BoundingBox]]) -> Dict[str, Any]:
        """
        Validate generated bounding box against constraints
        
        Args:
            generated_bbox: Generated bounding box
            valid_coordinates: Valid coordinate constraints
            ocr_data: OCR data for validation
            
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'constraint_violations': [],
            'ocr_overlap_score': 0.0,
            'confidence_adjustment': 1.0
        }
        
        # Check coordinate constraints
        coordinates = {
            'x': generated_bbox.x,
            'y': generated_bbox.y,
            'w': generated_bbox.w,
            'h': generated_bbox.h
        }
        
        for pos, value in coordinates.items():
            if pos in valid_coordinates and value not in valid_coordinates[pos]:
                # Find closest valid coordinate
                closest = self._find_closest_valid_coordinate(value, valid_coordinates[pos])
                validation['constraint_violations'].append({
                    'position': pos,
                    'generated': value,
                    'closest_valid': closest,
                    'distance': abs(value - closest)
                })
        
        # Check overlap with OCR segments
        max_overlap = 0.0
        for text, ocr_bbox in ocr_data:
            overlap = generated_bbox.intersection_over_union(ocr_bbox)
            max_overlap = max(max_overlap, overlap)
        
        validation['ocr_overlap_score'] = max_overlap
        
        # Adjust confidence based on violations
        if validation['constraint_violations']:
            penalty = len(validation['constraint_violations']) * 0.1
            validation['confidence_adjustment'] = max(0.1, 1.0 - penalty)
            validation['is_valid'] = len(validation['constraint_violations']) <= 1  # Allow minor violations
        
        return validation
    
    def get_constraint_statistics(self, 
                                valid_coordinates: Dict[str, Set[int]]) -> Dict[str, Any]:
        """Get statistics about coordinate constraints"""
        stats = {}
        
        for pos, coords in valid_coordinates.items():
            if coords:
                coord_list = sorted(coords)
                stats[pos] = {
                    'count': len(coords),
                    'min': min(coords),
                    'max': max(coords),
                    'range': max(coords) - min(coords),
                    'median': coord_list[len(coord_list) // 2]
                }
            else:
                stats[pos] = {'count': 0, 'min': 0, 'max': 0, 'range': 0, 'median': 0}
        
        return stats
