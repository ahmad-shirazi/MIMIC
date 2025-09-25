"""
Main teacher agent implementing Phase A - Expert Data Generation
"""
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

import torch
from PIL import Image
import numpy as np

from ..config import Config
from ..utils.ocr import OCRExtractor, run_ocr
from ..utils.retrieval import ContextRetriever, find_text, TextSegment
from ..utils.grounding import AnswerGrounder, ground_answer
from ..utils.formatting import ResponseFormatter, format_teacher_response
from ..utils.bbox import BoundingBox
from .planner import TeacherPlanner
from .qa_model import TeacherQAModel

logger = logging.getLogger(__name__)


@dataclass
class ExpertTrace:
    """Expert trace data for training"""
    image_path: str
    question: str
    answer: str
    teacher_string: str
    ocr_data: List[Tuple[str, BoundingBox]]
    context_data: List[TextSegment]
    grounding_bbox: BoundingBox
    processing_time: float


class TeacherAgent:
    """
    Teacher agent implementing Phase A of MIMIC-VQA framework
    
    Orchestrates the expert data generation pipeline:
    1. OCR extraction 
    2. Context retrieval
    3. Teacher QA
    4. Deterministic grounding
    5. Response formatting
    """
    
    def __init__(self, config: Config):
        """
        Initialize teacher agent
        
        Args:
            config: MIMIC-VQA configuration
        """
        self.config = config
        
        # Initialize components
        self.ocr_extractor = OCRExtractor(
            engine=config.data.ocr_engine,
            confidence_threshold=config.data.ocr_confidence_threshold
        )
        
        self.context_retriever = ContextRetriever(
            top_k=config.data.top_k_segments,
            similarity_threshold=config.data.similarity_threshold,
            mix_weight=0.7  # From Algorithm 1
        )
        
        self.answer_grounder = AnswerGrounder(
            anls_threshold=0.5,  # From Algorithm 2
            min_confidence=0.3
        )
        
        self.response_formatter = ResponseFormatter(format_style="cot")
        
        # Initialize teacher models
        self.planner = TeacherPlanner(config)
        self.qa_model = TeacherQAModel(config)
        
        # Expert dataset storage
        self.expert_dataset = []
        
        logger.info("Teacher agent initialized successfully")
    
    def generate_expert_trace(self,
                            image: Union[str, np.ndarray, Image.Image],
                            question: str) -> ExpertTrace:
        """
        Generate complete expert trace following Algorithm 1
        
        Implementation of Phase 1: Teacher data generation (expert traces)
        
        Args:
            image: Input document image
            question: Question text
            
        Returns:
            Complete expert trace with reasoning string
        """
        start_time = time.time()
        
        try:
            # Step 1: OCR extraction
            logger.debug("Step 1: Running OCR extraction")
            ocr_results = self.run_ocr_step(image)
            
            # Step 2: Context retrieval  
            logger.debug("Step 2: Retrieving relevant context")
            context_segments = self.find_text_step(question, ocr_results)
            
            # Step 3: Teacher answer generation
            logger.debug("Step 3: Generating teacher answer")
            answer_text = self.ask_qa_step(question, context_segments)
            
            # Step 4: Deterministic grounding
            logger.debug("Step 4: Grounding answer to coordinates")
            grounding_bbox, grounding_score = self.ground_answer_step(
                answer_text, ocr_results
            )
            
            # Step 5: Format teacher string
            logger.debug("Step 5: Formatting teacher response")
            teacher_string = self.format_step(
                context_segments, answer_text, grounding_bbox
            )
            
            processing_time = time.time() - start_time
            
            # Create expert trace
            expert_trace = ExpertTrace(
                image_path=image if isinstance(image, str) else "memory_image",
                question=question,
                answer=answer_text,
                teacher_string=teacher_string,
                ocr_data=[(text, bbox) for text, bbox in ocr_results],
                context_data=context_segments,
                grounding_bbox=grounding_bbox,
                processing_time=processing_time
            )
            
            logger.info(f"Expert trace generated in {processing_time:.2f}s")
            return expert_trace
            
        except Exception as e:
            logger.error(f"Failed to generate expert trace: {e}")
            raise
    
    def run_ocr_step(self, image: Union[str, np.ndarray, Image.Image]) -> List[Tuple[str, BoundingBox]]:
        """
        Step 1: OCR extraction
        
        O ← RunOCR(I) = {(ti, bi, confi)}
        
        Args:
            image: Input image I
            
        Returns:
            OCR outputs O as list of (text, bbox) tuples
        """
        try:
            ocr_results = run_ocr(
                image=image,
                engine=self.config.data.ocr_engine,
                confidence_threshold=self.config.data.ocr_confidence_threshold
            )
            
            logger.debug(f"Extracted {len(ocr_results)} text segments")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def find_text_step(self, 
                      question: str,
                      ocr_outputs: List[Tuple[str, BoundingBox]]) -> List[TextSegment]:
        """
        Step 2: Context retrieval
        
        C ← FindText(Q, O, k, α, τ)
        
        Args:
            question: Question text Q
            ocr_outputs: OCR outputs O
            
        Returns:
            Context segments C
        """
        try:
            # Convert to TextSegment objects
            segments = [
                TextSegment(text=text, bbox=bbox)
                for text, bbox in ocr_outputs
            ]
            
            # Find relevant segments
            context_segments = self.context_retriever.find_text(
                query=question,
                segments=segments
            )
            
            logger.debug(f"Retrieved {len(context_segments)} context segments")
            return context_segments
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []
    
    def ask_qa_step(self, 
                   question: str,
                   context: List[TextSegment]) -> str:
        """
        Step 3: Teacher answer generation
        
        Atext ← AskQA(MQA, Q, C)
        
        Args:
            question: Question text Q
            context: Context segments C
            
        Returns:
            Textual answer Atext
        """
        try:
            # Build context string
            context_text = self.context_retriever.build_context_string(
                context, max_length=self.config.data.context_window
            )
            
            # Generate answer using teacher QA model
            answer = self.qa_model.generate_answer(question, context_text)
            
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            return "Unable to determine answer"
    
    def ground_answer_step(self,
                          answer_text: str,
                          ocr_outputs: List[Tuple[str, BoundingBox]]) -> Tuple[BoundingBox, float]:
        """
        Step 4: Deterministic grounding
        
        (BA, score) ← GROUNDANSWER(Atext, O)
        
        Args:
            answer_text: Textual answer Atext
            ocr_outputs: OCR outputs O
            
        Returns:
            Grounding bbox BA and confidence score
        """
        try:
            bbox, score = ground_answer(answer_text, ocr_outputs)
            
            logger.debug(f"Grounded answer to bbox: {bbox} (score: {score:.3f})")
            return bbox, score
            
        except Exception as e:
            logger.error(f"Answer grounding failed: {e}")
            # Return default bbox with zero confidence
            return BoundingBox(0, 0, 0, 0, 0.0), 0.0
    
    def format_step(self,
                   context: List[TextSegment],
                   answer_text: str,
                   answer_bbox: BoundingBox) -> str:
        """
        Step 5: Build teacher string
        
        ST ← FORMAT(C, Atext, BA)
        
        Args:
            context: Context segments C
            answer_text: Textual answer Atext
            answer_bbox: Answer bbox BA
            
        Returns:
            Teacher string ST
        """
        try:
            teacher_string = format_teacher_response(
                context=context,
                answer_text=answer_text,
                answer_bbox=answer_bbox
            )
            
            logger.debug(f"Formatted teacher string: {teacher_string[:100]}...")
            return teacher_string
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return f"Thought: Error in processing; Final Answer: {answer_text}; Location: [0, 0, 0, 0]"
    
    def generate_expert_dataset(self,
                               dataset_items: List[Dict[str, Any]],
                               output_path: Optional[str] = None) -> List[ExpertTrace]:
        """
        Generate complete expert dataset for training
        
        Args:
            dataset_items: List of dicts with 'image', 'question', 'answer' keys
            output_path: Optional path to save expert traces
            
        Returns:
            List of expert traces
        """
        logger.info(f"Generating expert dataset with {len(dataset_items)} items")
        
        expert_traces = []
        
        for i, item in enumerate(dataset_items):
            try:
                logger.info(f"Processing item {i+1}/{len(dataset_items)}")
                
                expert_trace = self.generate_expert_trace(
                    image=item['image'],
                    question=item['question']
                )
                
                expert_traces.append(expert_trace)
                
                # Optional: Validate against ground truth if available
                if 'answer' in item:
                    self._validate_trace(expert_trace, item['answer'])
                
            except Exception as e:
                logger.error(f"Failed to process item {i+1}: {e}")
                continue
        
        # Save expert dataset
        if output_path:
            self.save_expert_dataset(expert_traces, output_path)
        
        logger.info(f"Generated {len(expert_traces)} expert traces")
        return expert_traces
    
    def _validate_trace(self, trace: ExpertTrace, ground_truth: str):
        """Validate expert trace against ground truth"""
        # Simple validation using ANLS
        anls_score = self.answer_grounder.compute_anls(trace.answer, ground_truth)
        if anls_score < 0.5:
            logger.warning(f"Low ANLS score ({anls_score:.3f}) for generated answer")
    
    def save_expert_dataset(self, traces: List[ExpertTrace], output_path: str):
        """Save expert traces to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_traces = []
        for trace in traces:
            serializable_traces.append({
                'image_path': trace.image_path,
                'question': trace.question,
                'answer': trace.answer,
                'teacher_string': trace.teacher_string,
                'processing_time': trace.processing_time,
                'ocr_data': [
                    {
                        'text': text,
                        'bbox': [bbox.x, bbox.y, bbox.w, bbox.h],
                        'confidence': bbox.confidence
                    }
                    for text, bbox in trace.ocr_data
                ],
                'grounding_bbox': {
                    'x': trace.grounding_bbox.x,
                    'y': trace.grounding_bbox.y,
                    'w': trace.grounding_bbox.w,
                    'h': trace.grounding_bbox.h,
                    'confidence': trace.grounding_bbox.confidence
                }
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_traces, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(traces)} expert traces to {output_path}")
    
    def load_expert_dataset(self, dataset_path: str) -> List[ExpertTrace]:
        """Load expert traces from file"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        traces = []
        for item in data:
            # Reconstruct OCR data
            ocr_data = []
            for ocr_item in item['ocr_data']:
                bbox = BoundingBox(
                    x=ocr_item['bbox'][0],
                    y=ocr_item['bbox'][1],
                    w=ocr_item['bbox'][2],
                    h=ocr_item['bbox'][3],
                    confidence=ocr_item['confidence']
                )
                ocr_data.append((ocr_item['text'], bbox))
            
            # Reconstruct grounding bbox
            grounding_data = item['grounding_bbox']
            grounding_bbox = BoundingBox(
                x=grounding_data['x'],
                y=grounding_data['y'],
                w=grounding_data['w'],
                h=grounding_data['h'],
                confidence=grounding_data['confidence']
            )
            
            trace = ExpertTrace(
                image_path=item['image_path'],
                question=item['question'],
                answer=item['answer'],
                teacher_string=item['teacher_string'],
                ocr_data=ocr_data,
                context_data=[],  # Not saved in simplified format
                grounding_bbox=grounding_bbox,
                processing_time=item['processing_time']
            )
            traces.append(trace)
        
        logger.info(f"Loaded {len(traces)} expert traces from {dataset_path}")
        return traces
    
    def get_statistics(self, traces: List[ExpertTrace]) -> Dict[str, Any]:
        """Get statistics about expert traces"""
        if not traces:
            return {}
        
        processing_times = [t.processing_time for t in traces]
        answer_lengths = [len(t.answer.split()) for t in traces]
        
        stats = {
            'total_traces': len(traces),
            'avg_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'avg_answer_length': np.mean(answer_lengths),
            'std_answer_length': np.std(answer_lengths),
            'avg_ocr_segments': np.mean([len(t.ocr_data) for t in traces]),
        }
        
        return stats
