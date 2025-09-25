"""
Student model implementation for Phase B training and Phase C inference
"""
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

from ..config import Config
from ..utils.bbox import BoundingBox, parse_bbox_string

logger = logging.getLogger(__name__)


class StudentModel(nn.Module):
    """
    Student VLM model for end-to-end document VQA
    
    Based on Gemma architecture, pruned from 27B to 9B parameters.
    Implements the student model πS with parameters θS from Algorithm 1.
    """
    
    def __init__(self, config: Config, load_pretrained: bool = True):
        """
        Initialize student model
        
        Args:
            config: MIMIC-VQA configuration
            load_pretrained: Whether to load pretrained weights
        """
        super().__init__()
        self.config = config
        self.model_name = config.model.student_base  # Start with base Gemma-27B
        self.target_model = config.model.student_target  # Target Gemma-9B
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.lora_config = None
        
        # Training state
        self.is_pruned = False
        self.current_sparsity = 0.0
        
        if load_pretrained:
            self._load_model()
            
        logger.info(f"Student model initialized: {self.model_name}")
    
    def _load_model(self):
        """Load pretrained model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.base_model_path,
                trust_remote_code=True,
                padding_side="left"  # For generation
            )
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if specified
            quantization_config = None
            if self.config.model.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.model.use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.config.training.mixed_precision == "fp16" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.config.training.num_gpus > 1 else None
            )
            
            # Setup LoRA for efficient fine-tuning
            self._setup_lora()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_lora(self):
        """Setup LoRA configuration for efficient training"""
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Low-rank dimension
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none"
        )
        
        # Apply LoRA to model
        if not self.config.model.use_4bit and not self.config.model.use_8bit:
            self.model = get_peft_model(self.model, self.lora_config)
            logger.info("LoRA configuration applied")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training
            
        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }
    
    def generate_response(self,
                         prompt: str,
                         max_new_tokens: int = 512,
                         temperature: float = 0.7,
                         do_sample: bool = True,
                         **kwargs) -> str:
        """
        Generate response for given prompt
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_sequence_length
        ).to(self.model.device)
        
        # Generation configuration
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=kwargs.get('top_p', 0.9),
            top_k=kwargs.get('top_k', 50),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=kwargs.get('repetition_penalty', 1.1)
        )
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response (excluding input)
            input_length = inputs['input_ids'].shape[1]
            response_ids = outputs[0][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Error generating response"
    
    def generate_vqa_response(self,
                             image_description: str,
                             question: str,
                             max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        Generate VQA response in the student format
        
        Implementation for Phase C inference
        
        Args:
            image_description: Description or OCR text of the image
            question: Question about the image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with reasoning, answer, and bounding box
        """
        # Format prompt for VQA task
        prompt = self._format_vqa_prompt(image_description, question)
        
        # Generate response
        response = self.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=self.config.inference.temperature,
            do_sample=self.config.inference.do_sample
        )
        
        # Parse response into components
        parsed_response = self._parse_vqa_response(response)
        
        return parsed_response
    
    def _format_vqa_prompt(self, image_description: str, question: str) -> str:
        """
        Format prompt for VQA task following teacher format
        
        Args:
            image_description: Image content description
            question: Question to answer
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Document Image Content:
{image_description}

Question: {question}

Please provide your response in the following format:
Thought: [Your reasoning process]
Final Answer: [Your answer]
Location: [x, y, w, h]

Response:"""
        
        return prompt
    
    def _parse_vqa_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VQA response into structured format
        
        Args:
            response: Generated response string
            
        Returns:
            Parsed response components
        """
        result = {
            'thought': '',
            'answer': '',
            'bbox': BoundingBox(0, 0, 0, 0, 0.0),
            'confidence': 0.0,
            'raw_response': response
        }
        
        try:
            # Extract thought/reasoning
            if 'Thought:' in response:
                thought_part = response.split('Thought:')[1]
                if 'Final Answer:' in thought_part:
                    result['thought'] = thought_part.split('Final Answer:')[0].strip()
                else:
                    result['thought'] = thought_part.strip()
            
            # Extract final answer
            if 'Final Answer:' in response:
                answer_part = response.split('Final Answer:')[1]
                if 'Location:' in answer_part:
                    result['answer'] = answer_part.split('Location:')[0].strip()
                else:
                    result['answer'] = answer_part.strip()
            
            # Extract location/bounding box
            if 'Location:' in response:
                location_part = response.split('Location:')[1].strip()
                bbox = parse_bbox_string(location_part)
                if bbox:
                    result['bbox'] = bbox
                    result['confidence'] = bbox.confidence
            
        except Exception as e:
            logger.error(f"Failed to parse VQA response: {e}")
            # Fallback: use entire response as answer
            result['answer'] = response.strip()
        
        return result
    
    def compute_cross_entropy_loss(self,
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  teacher_sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for teacher sequence
        
        Implementation of L(θS) from Algorithm 1:
        L(θS) = -∑(k=1 to |ST|) log P(ST,k | ST,<k, I, Q; θS)
        
        Args:
            input_ids: Input token IDs (I, Q)
            attention_mask: Attention mask
            teacher_sequence: Teacher sequence tokens ST
            
        Returns:
            Cross-entropy loss
        """
        # Concatenate input and teacher sequence
        full_sequence = torch.cat([input_ids, teacher_sequence], dim=1)
        full_attention = torch.cat([
            attention_mask, 
            torch.ones_like(teacher_sequence)
        ], dim=1)
        
        # Shift for causal LM loss
        labels = full_sequence.clone()
        labels[:, :input_ids.shape[1]] = -100  # Ignore input tokens in loss
        
        # Forward pass
        outputs = self.forward(
            input_ids=full_sequence[:, :-1],  # All but last token as input
            attention_mask=full_attention[:, :-1],
            labels=labels[:, 1:]  # All but first token as labels
        )
        
        return outputs['loss']
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameter statistics"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': trainable_params / total_params * 100,
            'is_pruned': self.is_pruned,
            'current_sparsity': self.current_sparsity,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def save_model(self, output_dir: str, save_tokenizer: bool = True):
        """Save model checkpoint"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        
        if save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            if hasattr(self.model, 'load_state_dict'):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(checkpoint)
                logger.info(f"Checkpoint loaded from {checkpoint_path}")
            else:
                # For PEFT models
                self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                logger.info(f"Model loaded from {checkpoint_path}")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        if self.model is not None:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def prepare_for_training(self):
        """Prepare model for training"""
        if self.model is not None:
            self.model.train()
            
            # Enable gradient checkpointing if configured
            if self.config.training.gradient_checkpointing:
                self.enable_gradient_checkpointing()
            
            logger.info("Model prepared for training")
    
    def prepare_for_inference(self):
        """Prepare model for inference"""
        if self.model is not None:
            self.model.eval()
            logger.info("Model prepared for inference")
