"""
Teacher QA model using Google Gemini for answer generation
"""
import logging
import os
from typing import Optional, Dict, Any
import time

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from ..config import Config

logger = logging.getLogger(__name__)


class TeacherQAModel:
    """
    Teacher QA model using Google Gemini-3-27B for answer generation
    
    Implements the AskQA tool from Algorithm 1:
    Atext ← AskQA(MQA, Q, C)
    """
    
    def __init__(self, config: Config):
        """
        Initialize teacher QA model
        
        Args:
            config: MIMIC-VQA configuration
        """
        self.config = config
        self.model_name = config.model.teacher_qa  # "gemini-3-27b"
        
        if not GENAI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        # Configure Gemini API
        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        
        # Initialize model
        try:
            # Use the latest Gemini model (map our config name to actual model)
            actual_model_name = self._map_model_name(self.model_name)
            self.model = genai.GenerativeModel(actual_model_name)
            
            # Generation configuration
            self.generation_config = {
                'temperature': 0.3,  # Lower temperature for more consistent answers
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 1024,
            }
            
            logger.info(f"Teacher QA model initialized: {actual_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def _map_model_name(self, config_name: str) -> str:
        """Map configuration model name to actual Gemini model name"""
        model_mapping = {
            'gemini-3-27b': 'gemini-1.5-pro',  # Use Gemini 1.5 Pro as closest equivalent
            'gemini-3-9b': 'gemini-1.5-flash',  # Use Gemini 1.5 Flash for smaller model
            'gemini-1.5-pro': 'gemini-1.5-pro',
            'gemini-1.5-flash': 'gemini-1.5-flash'
        }
        
        return model_mapping.get(config_name, 'gemini-1.5-pro')
    
    def generate_answer(self, 
                       question: str, 
                       context: str,
                       max_retries: int = 3) -> str:
        """
        Generate answer for question given context
        
        Implementation of AskQA(MQA, Q, C) from Algorithm 1
        
        Args:
            question: Question text Q
            context: Retrieved context C
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated answer text Atext
        """
        # Build prompt
        prompt = self._build_qa_prompt(question, context)
        
        # Generate answer with retries
        for attempt in range(max_retries):
            try:
                logger.debug(f"Generating answer (attempt {attempt + 1}/{max_retries})")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                
                if response.text:
                    answer = self._post_process_answer(response.text)
                    logger.debug(f"Generated answer: {answer[:100]}...")
                    return answer
                else:
                    logger.warning("Empty response from Gemini model")
                    
            except Exception as e:
                logger.error(f"Answer generation failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        # Fallback answer
        logger.error("All answer generation attempts failed")
        return "Unable to determine answer from the provided context."
    
    def _build_qa_prompt(self, question: str, context: str) -> str:
        """
        Build QA prompt for document understanding
        
        Args:
            question: Question to answer
            context: Document context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are an expert document analyst. Your task is to answer questions about documents by carefully analyzing the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Read the context carefully and identify all relevant information
2. Answer the question directly and precisely based only on the information in the context
3. If the answer requires specific values, names, or locations, extract them exactly as they appear
4. Keep your answer concise and focused on the question asked
5. If the information is not available in the context, state "The information is not available in the provided context"

ANSWER:"""
        
        return prompt
    
    def _post_process_answer(self, raw_answer: str) -> str:
        """
        Post-process generated answer
        
        Args:
            raw_answer: Raw answer from model
            
        Returns:
            Cleaned and processed answer
        """
        # Clean up the answer
        answer = raw_answer.strip()
        
        # Remove common prefixes that models sometimes add
        prefixes_to_remove = [
            "Answer:",
            "The answer is:",
            "Based on the context,",
            "According to the document,",
            "ANSWER:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Ensure answer is not empty
        if not answer:
            answer = "Unable to determine answer"
        
        return answer
    
    def generate_answer_with_reasoning(self, 
                                     question: str, 
                                     context: str) -> Dict[str, str]:
        """
        Generate answer with explicit reasoning chain
        
        Args:
            question: Question text
            context: Retrieved context
            
        Returns:
            Dictionary with 'answer' and 'reasoning' keys
        """
        reasoning_prompt = f"""You are an expert document analyst. Your task is to answer questions about documents by providing both the answer and your reasoning process.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. First, analyze the context and explain your reasoning step by step
2. Then, provide your final answer
3. Format your response as follows:

REASONING:
[Your step-by-step reasoning here]

ANSWER:
[Your final answer here]

Please provide your response:"""
        
        try:
            response = self.model.generate_content(
                reasoning_prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                return self._parse_reasoning_response(response.text)
            
        except Exception as e:
            logger.error(f"Answer generation with reasoning failed: {e}")
        
        # Fallback
        return {
            'answer': self.generate_answer(question, context),
            'reasoning': "Generated answer without explicit reasoning due to processing error."
        }
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, str]:
        """Parse reasoning response into components"""
        response = response.strip()
        
        # Try to split by REASONING: and ANSWER: markers
        reasoning = ""
        answer = ""
        
        if "REASONING:" in response and "ANSWER:" in response:
            parts = response.split("ANSWER:")
            if len(parts) >= 2:
                reasoning_part = parts[0].replace("REASONING:", "").strip()
                answer_part = parts[1].strip()
                
                reasoning = reasoning_part
                answer = self._post_process_answer(answer_part)
        else:
            # Fallback: treat entire response as answer
            answer = self._post_process_answer(response)
            reasoning = "Reasoning not explicitly separated in response."
        
        return {
            'answer': answer,
            'reasoning': reasoning
        }
    
    def batch_generate_answers(self, 
                              questions_contexts: list) -> list:
        """
        Generate answers for multiple question-context pairs
        
        Args:
            questions_contexts: List of (question, context) tuples
            
        Returns:
            List of generated answers
        """
        answers = []
        
        for i, (question, context) in enumerate(questions_contexts):
            logger.info(f"Processing question {i+1}/{len(questions_contexts)}")
            
            try:
                answer = self.generate_answer(question, context)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Failed to generate answer for question {i+1}: {e}")
                answers.append("Error generating answer")
        
        return answers
    
    def evaluate_answer_quality(self, 
                               question: str,
                               context: str, 
                               generated_answer: str,
                               ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate quality of generated answer
        
        Args:
            question: Original question
            context: Context used for generation
            generated_answer: Generated answer
            ground_truth: Optional ground truth answer
            
        Returns:
            Quality evaluation metrics
        """
        evaluation = {
            'answer_length': len(generated_answer.split()),
            'is_non_empty': len(generated_answer.strip()) > 0,
            'contains_context_info': self._check_context_grounding(context, generated_answer)
        }
        
        if ground_truth:
            # Simple similarity metrics if ground truth available
            from ..utils.grounding import AnswerGrounder
            grounder = AnswerGrounder()
            
            anls_score = grounder.compute_anls(generated_answer, ground_truth)
            evaluation['anls_score'] = anls_score
            evaluation['exact_match'] = generated_answer.strip().lower() == ground_truth.strip().lower()
        
        return evaluation
    
    def _check_context_grounding(self, context: str, answer: str) -> bool:
        """Check if answer is grounded in context"""
        if not context or not answer:
            return False
        
        # Simple heuristic: check if answer contains words from context
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        # Check for overlap (excluding very common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        context_content = context_words - common_words
        answer_content = answer_words - common_words
        
        overlap = len(answer_content & context_content)
        return overlap > 0
