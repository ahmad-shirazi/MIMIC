"""
Inference pipeline for Phase C - Deployed Student Only
"""
import logging
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import numpy as np

import torch
from PIL import Image

from ..config import Config
from ..student.model import StudentModel
from ..utils.ocr import OCRExtractor, run_ocr
from ..utils.bbox import BoundingBox, format_bbox_string
from .constrained_decoder import ConstrainedDecoder

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline for deployed student model
    
    Implementation of Phase C from Algorithm 1:
    - Student generates reasoning answer & bbox
    - Optional constrained decoding for bbox
    - Single pass inference
    """
    
    def __init__(self, 
                 config: Config,
                 model_path: Optional[str] = None,
                 use_constrained_decoding: bool = True):
        """
        Initialize inference pipeline
        
        Args:
            config: MIMIC-VQA configuration
            model_path: Path to trained student model
            use_constrained_decoding: Whether to use constrained bbox decoding
        """
        self.config = config
        self.model_path = model_path
        self.use_constrained_decoding = use_constrained_decoding
        
        # Components
        self.student_model = None
        self.ocr_extractor = None
        self.constrained_decoder = None
        
        # Performance tracking
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'avg_time_per_inference': 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Inference pipeline initialized")
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        # Initialize OCR extractor
        self.ocr_extractor = OCRExtractor(
            engine=self.config.data.ocr_engine,
            confidence_threshold=self.config.data.ocr_confidence_threshold
        )
        
        # Initialize student model
        self.student_model = StudentModel(self.config, load_pretrained=False)
        
        # Load trained model if path provided
        if self.model_path:
            self.load_model(self.model_path)
        else:
            # Load base model
            self.student_model._load_model()
        
        self.student_model.prepare_for_inference()
        
        # Initialize constrained decoder if enabled
        if self.use_constrained_decoding:
            self.constrained_decoder = ConstrainedDecoder(
                self.config,
                self.student_model.tokenizer
            )
        
        logger.info("Pipeline components initialized")
    
    def load_model(self, model_path: str):
        """Load trained student model"""
        try:
            self.student_model.load_checkpoint(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            # Fall back to base model
            self.student_model._load_model()
            logger.info("Using base model as fallback")
    
    def infer(self, 
              image: Union[str, np.ndarray, Image.Image],
              question: str,
              return_intermediate: bool = False) -> Dict[str, Any]:
        """
        Run end-to-end inference
        
        Implementation of Phase C inference from Algorithm 1
        
        Args:
            image: Input document image
            question: Question about the image
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Inference results with answer and bounding box
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract OCR if constrained decoding is used
            ocr_data = None
            if self.use_constrained_decoding:
                ocr_data = self._extract_ocr_for_constraints(image)
            
            # Step 2: Prepare input for student model
            image_description = self._prepare_image_description(image, ocr_data)
            
            # Step 3: Generate student response
            if self.use_constrained_decoding and ocr_data:
                # Use constrained decoding
                response = self._generate_with_constraints(
                    image_description, question, ocr_data
                )
            else:
                # Standard generation
                response = self.student_model.generate_vqa_response(
                    image_description=image_description,
                    question=question,
                    max_new_tokens=self.config.inference.max_new_tokens
                )
            
            # Step 4: Post-process and validate results
            final_result = self._post_process_result(response, ocr_data)
            
            # Update performance stats
            inference_time = time.time() - start_time
            self._update_stats(inference_time)
            
            # Add metadata
            final_result.update({
                'inference_time': inference_time,
                'used_constrained_decoding': self.use_constrained_decoding,
                'model_path': self.model_path
            })
            
            # Add intermediate results if requested
            if return_intermediate:
                final_result['intermediate'] = {
                    'image_description': image_description,
                    'ocr_data': ocr_data,
                    'raw_response': response.get('raw_response', '')
                }
            
            logger.debug(f"Inference completed in {inference_time:.3f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._create_error_result(str(e))
    
    def _extract_ocr_for_constraints(self, 
                                   image: Union[str, np.ndarray, Image.Image]) -> List[Tuple[str, BoundingBox]]:
        """Extract OCR data for constrained decoding"""
        try:
            ocr_results = run_ocr(
                image=image,
                engine=self.config.data.ocr_engine,
                confidence_threshold=self.config.data.ocr_confidence_threshold
            )
            
            logger.debug(f"Extracted {len(ocr_results)} OCR segments for constraints")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def _prepare_image_description(self, 
                                 image: Union[str, np.ndarray, Image.Image],
                                 ocr_data: Optional[List[Tuple[str, BoundingBox]]] = None) -> str:
        """Prepare image description for model input"""
        
        if ocr_data:
            # Use provided OCR data
            ocr_texts = [text for text, _ in ocr_data[:30]]  # Limit for context length
        else:
            # Extract OCR on demand
            try:
                ocr_results = run_ocr(image, self.config.data.ocr_engine)
                ocr_texts = [text for text, _ in ocr_results[:30]]
            except Exception as e:
                logger.error(f"Failed to extract OCR for description: {e}")
                ocr_texts = []
        
        if ocr_texts:
            description = "Document contains the following text: " + "; ".join(ocr_texts)
        else:
            description = "Document image provided (OCR extraction failed)"
        
        return description
    
    def _generate_with_constraints(self,
                                 image_description: str,
                                 question: str,
                                 ocr_data: List[Tuple[str, BoundingBox]]) -> Dict[str, Any]:
        """Generate response with constrained bbox decoding"""
        
        # First, generate reasoning and answer without constraints
        prompt = f"""Document Content: {image_description}

Question: {question}

Please provide your reasoning and answer. Format:
Thought: [Your reasoning process]
Final Answer: [Your answer]

Response:"""
        
        # Generate reasoning and answer
        reasoning_response = self.student_model.generate_response(
            prompt=prompt,
            max_new_tokens=self.config.inference.max_new_tokens,
            temperature=self.config.inference.temperature
        )
        
        # Parse reasoning and answer
        thought = ""
        answer = ""
        
        if "Thought:" in reasoning_response and "Final Answer:" in reasoning_response:
            parts = reasoning_response.split("Final Answer:")
            thought = parts[0].replace("Thought:", "").strip()
            answer = parts[1].strip()
        else:
            # Fallback parsing
            thought = "Generated response without clear structure"
            answer = reasoning_response.strip()
        
        # Generate constrained bbox
        if self.constrained_decoder:
            constrained_bbox = self.constrained_decoder.generate_bbox(
                answer_text=answer,
                ocr_data=ocr_data,
                context=image_description
            )
        else:
            # Fallback: try to parse from answer or use default
            from ..utils.bbox import parse_bbox_string
            constrained_bbox = parse_bbox_string(answer) or BoundingBox(0, 0, 0, 0, 0.0)
        
        return {
            'thought': thought,
            'answer': answer,
            'bbox': constrained_bbox,
            'confidence': constrained_bbox.confidence,
            'raw_response': reasoning_response
        }
    
    def _post_process_result(self, 
                           response: Dict[str, Any],
                           ocr_data: Optional[List[Tuple[str, BoundingBox]]]) -> Dict[str, Any]:
        """Post-process inference result"""
        
        result = {
            'question_answered': True,
            'reasoning': response.get('thought', ''),
            'answer': response.get('answer', ''),
            'bounding_box': response.get('bbox', BoundingBox(0, 0, 0, 0, 0.0)),
            'confidence': response.get('confidence', 0.0),
            'bbox_coordinates': []  # [x, y, w, h] format
        }
        
        # Format bounding box coordinates
        bbox = result['bounding_box']
        if isinstance(bbox, BoundingBox):
            result['bbox_coordinates'] = [bbox.x, bbox.y, bbox.w, bbox.h]
            result['bbox_string'] = format_bbox_string(bbox, "xywh")
        
        # Validate answer quality
        result['answer_quality'] = self._assess_answer_quality(
            result['answer'], 
            result['reasoning']
        )
        
        # Validate bounding box if OCR data available
        if ocr_data:
            result['bbox_validation'] = self._validate_bbox_with_ocr(
                result['bounding_box'], 
                ocr_data
            )
        
        return result
    
    def _assess_answer_quality(self, answer: str, reasoning: str) -> Dict[str, Any]:
        """Assess quality of generated answer"""
        quality = {
            'is_non_empty': len(answer.strip()) > 0,
            'has_reasoning': len(reasoning.strip()) > 0,
            'answer_length': len(answer.split()),
            'reasoning_length': len(reasoning.split())
        }
        
        # Simple quality score
        score = 0.0
        if quality['is_non_empty']:
            score += 0.4
        if quality['has_reasoning']:
            score += 0.3
        if 3 <= quality['answer_length'] <= 50:  # Reasonable length
            score += 0.2
        if quality['reasoning_length'] >= 5:  # Some reasoning provided
            score += 0.1
        
        quality['overall_score'] = score
        return quality
    
    def _validate_bbox_with_ocr(self, 
                              bbox: BoundingBox,
                              ocr_data: List[Tuple[str, BoundingBox]]) -> Dict[str, Any]:
        """Validate bounding box against OCR coordinates"""
        validation = {
            'is_valid_coordinates': bbox.w > 0 and bbox.h > 0,
            'intersects_with_ocr': False,
            'closest_ocr_distance': float('inf'),
            'overlapping_ocr_segments': []
        }
        
        if not ocr_data:
            return validation
        
        # Check intersections with OCR segments
        for text, ocr_bbox in ocr_data:
            iou = bbox.intersection_over_union(ocr_bbox)
            
            if iou > 0:
                validation['intersects_with_ocr'] = True
                validation['overlapping_ocr_segments'].append({
                    'text': text,
                    'iou': iou,
                    'bbox': [ocr_bbox.x, ocr_bbox.y, ocr_bbox.w, ocr_bbox.h]
                })
            
            # Calculate distance
            distance = bbox.distance_to(ocr_bbox)
            if distance < validation['closest_ocr_distance']:
                validation['closest_ocr_distance'] = distance
        
        return validation
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'question_answered': False,
            'reasoning': f"Error during inference: {error_message}",
            'answer': "Unable to generate answer due to error",
            'bounding_box': BoundingBox(0, 0, 0, 0, 0.0),
            'confidence': 0.0,
            'bbox_coordinates': [0, 0, 0, 0],
            'bbox_string': "[0, 0, 0, 0]",
            'error': error_message,
            'answer_quality': {'overall_score': 0.0},
            'inference_time': 0.0
        }
    
    def _update_stats(self, inference_time: float):
        """Update inference statistics"""
        self.inference_stats['total_inferences'] += 1
        self.inference_stats['total_time'] += inference_time
        self.inference_stats['avg_time_per_inference'] = (
            self.inference_stats['total_time'] / self.inference_stats['total_inferences']
        )
    
    def batch_infer(self, 
                   image_question_pairs: List[Tuple],
                   show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Run inference on batch of image-question pairs
        
        Args:
            image_question_pairs: List of (image, question) tuples
            show_progress: Whether to show progress bar
            
        Returns:
            List of inference results
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            pairs = tqdm(image_question_pairs, desc="Running inference")
        else:
            pairs = image_question_pairs
        
        for image, question in pairs:
            try:
                result = self.infer(image, question)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                results.append(self._create_error_result(str(e)))
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get inference performance statistics"""
        stats = self.inference_stats.copy()
        
        # Add model statistics if available
        if self.student_model:
            stats['model_stats'] = self.student_model.get_model_parameters()
        
        return stats
    
    def benchmark(self, 
                 test_data: List[Tuple],
                 num_runs: int = 5) -> Dict[str, Any]:
        """
        Benchmark inference performance
        
        Args:
            test_data: List of (image, question) pairs for testing
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running benchmark with {len(test_data)} examples, {num_runs} runs")
        
        run_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Run inference on all test data
            for image, question in test_data:
                self.infer(image, question, return_intermediate=False)
            
            run_time = time.time() - start_time
            run_times.append(run_time)
            
            logger.info(f"Run {run + 1}: {run_time:.3f}s")
        
        benchmark_results = {
            'num_examples': len(test_data),
            'num_runs': num_runs,
            'total_inferences': len(test_data) * num_runs,
            'run_times': run_times,
            'avg_run_time': np.mean(run_times),
            'std_run_time': np.std(run_times),
            'avg_time_per_example': np.mean(run_times) / len(test_data),
            'throughput_examples_per_sec': len(test_data) / np.mean(run_times)
        }
        
        logger.info(f"Benchmark completed: {benchmark_results['throughput_examples_per_sec']:.2f} examples/sec")
        return benchmark_results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save inference results to file"""
        import json
        
        # Convert BoundingBox objects to serializable format
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            
            if 'bounding_box' in result and hasattr(result['bounding_box'], '__dict__'):
                serializable_result['bounding_box'] = result['bounding_box'].__dict__
            
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
