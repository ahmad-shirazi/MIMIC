"""
Teacher planner agent using Llama 4 Scout for orchestration
"""
import logging
from typing import Dict, Any, List, Optional
import json

from ..config import Config

logger = logging.getLogger(__name__)


class TeacherPlanner:
    """
    Teacher planner agent (πT) using Llama 4 Scout
    
    Orchestrates the teacher agent's reasoning process and tool usage.
    In practice, this would interface with the actual Llama 4 Scout model,
    but for this implementation we provide a structured planning framework.
    """
    
    def __init__(self, config: Config):
        """
        Initialize teacher planner
        
        Args:
            config: MIMIC-VQA configuration
        """
        self.config = config
        self.model_name = config.model.teacher_planner  # "llama-4-scout"
        
        # Planning templates
        self.planning_templates = {
            'document_qa': self._get_document_qa_plan(),
            'spatial_grounding': self._get_spatial_grounding_plan(),
            'multi_step_reasoning': self._get_multi_step_plan()
        }
        
        logger.info(f"Teacher planner initialized with {self.model_name}")
    
    def create_execution_plan(self, 
                            question: str,
                            image_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create execution plan for answering a question
        
        Args:
            question: Input question
            image_info: Information about the input image
            
        Returns:
            Structured execution plan
        """
        # Analyze question to determine plan type
        plan_type = self._classify_question_type(question)
        
        # Get base plan template
        base_plan = self.planning_templates.get(plan_type, self.planning_templates['document_qa'])
        
        # Customize plan for specific question
        execution_plan = {
            'question': question,
            'plan_type': plan_type,
            'steps': base_plan['steps'],
            'tools': base_plan['tools'],
            'reasoning_strategy': base_plan['reasoning_strategy'],
            'expected_outputs': base_plan['expected_outputs'],
            'quality_checks': base_plan['quality_checks']
        }
        
        return execution_plan
    
    def _classify_question_type(self, question: str) -> str:
        """
        Classify question to determine appropriate planning strategy
        
        Args:
            question: Input question
            
        Returns:
            Question type classification
        """
        question_lower = question.lower()
        
        # Check for spatial/location questions
        spatial_keywords = ['where', 'location', 'position', 'coordinates', 'find', 'locate']
        if any(keyword in question_lower for keyword in spatial_keywords):
            return 'spatial_grounding'
        
        # Check for multi-step reasoning questions
        reasoning_keywords = ['calculate', 'compute', 'how many', 'total', 'sum', 'compare']
        if any(keyword in question_lower for keyword in reasoning_keywords):
            return 'multi_step_reasoning'
        
        # Default to document QA
        return 'document_qa'
    
    def _get_document_qa_plan(self) -> Dict[str, Any]:
        """Get plan template for document QA questions"""
        return {
            'steps': [
                {
                    'id': 1,
                    'name': 'ocr_extraction',
                    'description': 'Extract text and coordinates from document image',
                    'tool': 'RunOCR',
                    'inputs': ['image'],
                    'outputs': ['ocr_results']
                },
                {
                    'id': 2,
                    'name': 'context_retrieval',
                    'description': 'Find relevant text segments for the question',
                    'tool': 'FindText',
                    'inputs': ['question', 'ocr_results'],
                    'outputs': ['context_segments']
                },
                {
                    'id': 3,
                    'name': 'answer_generation',
                    'description': 'Generate textual answer using context',
                    'tool': 'AskQA',
                    'inputs': ['question', 'context_segments'],
                    'outputs': ['answer_text']
                },
                {
                    'id': 4,
                    'name': 'answer_grounding',
                    'description': 'Ground answer to spatial coordinates',
                    'tool': 'GroundAnswer',
                    'inputs': ['answer_text', 'ocr_results'],
                    'outputs': ['bounding_box', 'confidence_score']
                },
                {
                    'id': 5,
                    'name': 'response_formatting',
                    'description': 'Format final teacher response string',
                    'tool': 'FORMAT',
                    'inputs': ['context_segments', 'answer_text', 'bounding_box'],
                    'outputs': ['teacher_string']
                }
            ],
            'tools': ['RunOCR', 'FindText', 'AskQA', 'GroundAnswer', 'FORMAT'],
            'reasoning_strategy': 'sequential_pipeline',
            'expected_outputs': ['teacher_string'],
            'quality_checks': [
                'ocr_confidence_check',
                'context_relevance_check',
                'answer_grounding_check'
            ]
        }
    
    def _get_spatial_grounding_plan(self) -> Dict[str, Any]:
        """Get plan template for spatial/location questions"""
        return {
            'steps': [
                {
                    'id': 1,
                    'name': 'ocr_extraction',
                    'description': 'Extract text and coordinates from document image',
                    'tool': 'RunOCR',
                    'inputs': ['image'],
                    'outputs': ['ocr_results']
                },
                {
                    'id': 2,
                    'name': 'spatial_analysis',
                    'description': 'Analyze spatial relationships in document',
                    'tool': 'AnalyzeSpatialLayout',
                    'inputs': ['ocr_results'],
                    'outputs': ['spatial_structure']
                },
                {
                    'id': 3,
                    'name': 'target_identification',
                    'description': 'Identify target element for location question',
                    'tool': 'FindText',
                    'inputs': ['question', 'ocr_results'],
                    'outputs': ['target_segments']
                },
                {
                    'id': 4,
                    'name': 'location_reasoning',
                    'description': 'Reason about spatial location of target',
                    'tool': 'AskQA',
                    'inputs': ['question', 'target_segments', 'spatial_structure'],
                    'outputs': ['location_description']
                },
                {
                    'id': 5,
                    'name': 'coordinate_grounding',
                    'description': 'Ground location to precise coordinates',
                    'tool': 'GroundAnswer',
                    'inputs': ['location_description', 'target_segments'],
                    'outputs': ['bounding_box', 'confidence_score']
                },
                {
                    'id': 6,
                    'name': 'response_formatting',
                    'description': 'Format spatial reasoning response',
                    'tool': 'FORMAT',
                    'inputs': ['target_segments', 'location_description', 'bounding_box'],
                    'outputs': ['teacher_string']
                }
            ],
            'tools': ['RunOCR', 'AnalyzeSpatialLayout', 'FindText', 'AskQA', 'GroundAnswer', 'FORMAT'],
            'reasoning_strategy': 'spatial_reasoning',
            'expected_outputs': ['teacher_string'],
            'quality_checks': [
                'ocr_confidence_check',
                'spatial_coherence_check',
                'target_identification_check',
                'coordinate_precision_check'
            ]
        }
    
    def _get_multi_step_plan(self) -> Dict[str, Any]:
        """Get plan template for multi-step reasoning questions"""
        return {
            'steps': [
                {
                    'id': 1,
                    'name': 'ocr_extraction',
                    'description': 'Extract text and coordinates from document image',
                    'tool': 'RunOCR',
                    'inputs': ['image'],
                    'outputs': ['ocr_results']
                },
                {
                    'id': 2,
                    'name': 'question_decomposition',
                    'description': 'Break down complex question into sub-questions',
                    'tool': 'DecomposeQuestion',
                    'inputs': ['question'],
                    'outputs': ['sub_questions']
                },
                {
                    'id': 3,
                    'name': 'iterative_context_retrieval',
                    'description': 'Find context for each sub-question',
                    'tool': 'FindText',
                    'inputs': ['sub_questions', 'ocr_results'],
                    'outputs': ['context_per_subq']
                },
                {
                    'id': 4,
                    'name': 'multi_step_reasoning',
                    'description': 'Perform reasoning for each step',
                    'tool': 'AskQA',
                    'inputs': ['sub_questions', 'context_per_subq'],
                    'outputs': ['intermediate_answers']
                },
                {
                    'id': 5,
                    'name': 'answer_synthesis',
                    'description': 'Combine intermediate answers into final answer',
                    'tool': 'SynthesizeAnswer',
                    'inputs': ['question', 'intermediate_answers'],
                    'outputs': ['final_answer']
                },
                {
                    'id': 6,
                    'name': 'answer_grounding',
                    'description': 'Ground synthesized answer to coordinates',
                    'tool': 'GroundAnswer',
                    'inputs': ['final_answer', 'ocr_results'],
                    'outputs': ['bounding_box', 'confidence_score']
                },
                {
                    'id': 7,
                    'name': 'response_formatting',
                    'description': 'Format multi-step reasoning response',
                    'tool': 'FORMAT',
                    'inputs': ['context_per_subq', 'final_answer', 'bounding_box'],
                    'outputs': ['teacher_string']
                }
            ],
            'tools': ['RunOCR', 'DecomposeQuestion', 'FindText', 'AskQA', 'SynthesizeAnswer', 'GroundAnswer', 'FORMAT'],
            'reasoning_strategy': 'multi_step_reasoning',
            'expected_outputs': ['teacher_string'],
            'quality_checks': [
                'ocr_confidence_check',
                'decomposition_completeness_check',
                'reasoning_coherence_check',
                'synthesis_accuracy_check',
                'final_grounding_check'
            ]
        }
    
    def validate_execution_plan(self, plan: Dict[str, Any]) -> bool:
        """
        Validate that execution plan is well-formed
        
        Args:
            plan: Execution plan to validate
            
        Returns:
            True if plan is valid, False otherwise
        """
        required_keys = ['question', 'plan_type', 'steps', 'tools', 'reasoning_strategy']
        
        # Check required keys
        if not all(key in plan for key in required_keys):
            logger.error("Execution plan missing required keys")
            return False
        
        # Check steps structure
        if not plan['steps']:
            logger.error("Execution plan has no steps")
            return False
        
        for step in plan['steps']:
            step_keys = ['id', 'name', 'tool', 'inputs', 'outputs']
            if not all(key in step for key in step_keys):
                logger.error(f"Step {step.get('id', 'unknown')} missing required keys")
                return False
        
        return True
    
    def get_reasoning_prompt(self, 
                           question: str, 
                           context: str,
                           plan_type: str = 'document_qa') -> str:
        """
        Get reasoning prompt for the teacher QA model
        
        Args:
            question: Input question
            context: Retrieved context
            plan_type: Type of reasoning plan
            
        Returns:
            Formatted reasoning prompt
        """
        if plan_type == 'spatial_grounding':
            return self._get_spatial_reasoning_prompt(question, context)
        elif plan_type == 'multi_step_reasoning':
            return self._get_multi_step_prompt(question, context)
        else:
            return self._get_document_qa_prompt(question, context)
    
    def _get_document_qa_prompt(self, question: str, context: str) -> str:
        """Get prompt for standard document QA"""
        return f"""Given the following context from a document, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
1. Read the context carefully
2. Identify relevant information to answer the question
3. Provide a clear, direct answer
4. If the answer is not in the context, state that clearly

Answer:"""
    
    def _get_spatial_reasoning_prompt(self, question: str, context: str) -> str:
        """Get prompt for spatial reasoning questions"""
        return f"""Given the following text segments from a document with their spatial context, answer the location-based question.

Context (with spatial information):
{context}

Question: {question}

Instructions:
1. Analyze the spatial relationships described in the context
2. Identify the target element being asked about
3. Determine its location based on the spatial information
4. Provide a precise answer about the location

Answer:"""
    
    def _get_multi_step_prompt(self, question: str, context: str) -> str:
        """Get prompt for multi-step reasoning"""
        return f"""Given the following context from a document, solve the multi-step question by breaking it down into logical steps.

Context:
{context}

Question: {question}

Instructions:
1. Break down the question into logical sub-steps
2. For each step, identify the relevant information from the context
3. Perform the necessary reasoning or calculations
4. Combine the results to provide a final answer
5. Show your reasoning process clearly

Answer:"""
    
    def monitor_execution(self, 
                         plan: Dict[str, Any], 
                         execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor plan execution and provide feedback
        
        Args:
            plan: Original execution plan
            execution_results: Results from each step
            
        Returns:
            Execution monitoring report
        """
        report = {
            'plan_type': plan['plan_type'],
            'total_steps': len(plan['steps']),
            'completed_steps': 0,
            'failed_steps': [],
            'quality_scores': {},
            'overall_success': True
        }
        
        # Check each step execution
        for step in plan['steps']:
            step_id = step['id']
            step_name = step['name']
            
            if step_name in execution_results:
                report['completed_steps'] += 1
                
                # Run quality checks for this step
                if 'quality_checks' in plan:
                    quality_score = self._evaluate_step_quality(
                        step, execution_results[step_name]
                    )
                    report['quality_scores'][step_name] = quality_score
            else:
                report['failed_steps'].append(step_name)
                report['overall_success'] = False
        
        return report
    
    def _evaluate_step_quality(self, step: Dict[str, Any], result: Any) -> float:
        """
        Evaluate quality of step execution
        
        Args:
            step: Step definition
            result: Step execution result
            
        Returns:
            Quality score between 0 and 1
        """
        # Simple quality evaluation - could be made more sophisticated
        if result is None:
            return 0.0
        
        if step['name'] == 'ocr_extraction':
            # For OCR, check if we got reasonable number of text segments
            if isinstance(result, list) and len(result) > 0:
                return 0.8
            else:
                return 0.2
        
        elif step['name'] == 'context_retrieval':
            # For context retrieval, check if we got relevant segments
            if isinstance(result, list) and len(result) > 0:
                return 0.8
            else:
                return 0.3
        
        elif step['name'] == 'answer_generation':
            # For answer generation, check if we got a non-empty answer
            if isinstance(result, str) and len(result.strip()) > 0:
                return 0.9
            else:
                return 0.1
        
        # Default quality score
        return 0.7
