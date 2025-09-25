"""
Student trainer implementing Phase B - Student Training
"""
import logging
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_scheduler
import wandb
from tqdm import tqdm

from ..config import Config
from ..teacher.agent import ExpertTrace
from .model import StudentModel
from .pruning import IterativeMagnitudePruner

logger = logging.getLogger(__name__)


class ExpertTraceDataset(Dataset):
    """Dataset for expert traces from teacher agent"""
    
    def __init__(self, 
                 expert_traces: List[ExpertTrace],
                 tokenizer,
                 max_length: int = 2048):
        """
        Initialize expert trace dataset
        
        Args:
            expert_traces: List of expert traces from teacher
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.expert_traces = expert_traces
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.expert_traces)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training example"""
        trace = self.expert_traces[idx]
        
        # Format input (image context + question)
        input_text = self._format_input(trace)
        
        # Teacher string is the target
        target_text = trace.teacher_string
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length // 2,  # Leave space for target
            padding=False,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding=False,
            return_tensors="pt"
        )
        
        # Combine input and target
        input_ids = torch.cat([
            input_encoding["input_ids"].squeeze(0),
            target_encoding["input_ids"].squeeze(0)
        ])
        
        attention_mask = torch.ones_like(input_ids)
        
        # Labels: -100 for input tokens, target tokens for output
        labels = input_ids.clone()
        labels[:len(input_encoding["input_ids"].squeeze(0))] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _format_input(self, trace: ExpertTrace) -> str:
        """Format input from expert trace"""
        # Create OCR text summary
        ocr_texts = [text for text, _ in trace.ocr_data[:20]]  # Limit for length
        ocr_summary = "; ".join(ocr_texts) if ocr_texts else "No OCR text available"
        
        input_text = f"""Document Content: {ocr_summary}

Question: {trace.question}

Please provide your response in the format:
Thought: [reasoning]
Final Answer: [answer]
Location: [x, y, w, h]

Response: """
        
        return input_text


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching"""
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        labels = item['labels']
        
        # Pad to max length
        padding_length = max_len - len(input_ids)
        
        padded_input_ids = torch.cat([
            torch.full((padding_length,), 0),  # Pad token ID
            input_ids
        ])
        
        padded_attention_mask = torch.cat([
            torch.zeros(padding_length),
            attention_mask
        ])
        
        padded_labels = torch.cat([
            torch.full((padding_length,), -100),
            labels
        ])
        
        padded_batch['input_ids'].append(padded_input_ids)
        padded_batch['attention_mask'].append(padded_attention_mask)
        padded_batch['labels'].append(padded_labels)
    
    # Stack into tensors
    return {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels'])
    }


class StudentTrainer:
    """
    Student trainer implementing Phase B of MIMIC-VQA
    
    Implements Algorithm 1 lines 16-27:
    - Iterative magnitude pruning
    - Supervised imitation on teacher strings
    - Cross-entropy loss optimization
    """
    
    def __init__(self, config: Config):
        """
        Initialize student trainer
        
        Args:
            config: MIMIC-VQA configuration
        """
        self.config = config
        
        # Initialize components
        self.model = None
        self.pruner = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
        
        logger.info("Student trainer initialized")
    
    def setup_logging(self):
        """Setup training logging"""
        if self.config.wandb_api_key:
            wandb.init(
                project="mimic-vqa",
                name=f"student-training-{int(time.time())}",
                config=self.config.__dict__
            )
    
    def setup_model(self) -> StudentModel:
        """Initialize student model"""
        self.model = StudentModel(self.config, load_pretrained=True)
        self.model.prepare_for_training()
        
        logger.info(f"Model parameters: {self.model.get_model_parameters()}")
        return self.model
    
    def setup_pruning(self, total_steps: int) -> IterativeMagnitudePruner:
        """Initialize pruning component"""
        self.pruner = IterativeMagnitudePruner(
            config=self.config,
            target_sparsity=self.config.training.target_sparsity,
            pruning_frequency=self.config.training.pruning_frequency
        )
        
        self.pruner.initialize_pruning(self.model.model, total_steps)
        return self.pruner
    
    def setup_optimizer_and_scheduler(self, total_steps: int):
        """Setup optimizer and learning rate scheduler"""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Setup scheduler
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer and scheduler setup for {total_steps} steps")
    
    def train(self, 
              expert_traces: List[ExpertTrace],
              validation_traces: Optional[List[ExpertTrace]] = None) -> Dict[str, Any]:
        """
        Train student model on expert traces
        
        Implementation of Phase B from Algorithm 1
        
        Args:
            expert_traces: Expert traces from teacher (Dexpert)
            validation_traces: Optional validation traces
            
        Returns:
            Training results and statistics
        """
        logger.info(f"Starting student training with {len(expert_traces)} expert traces")
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Create dataset and dataloader
        train_dataset = ExpertTraceDataset(
            expert_traces, 
            self.model.tokenizer,
            self.config.data.max_sequence_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Avoid multiprocessing issues with tokenizers
        )
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * self.config.training.num_epochs
        
        # Setup training components
        self.setup_optimizer_and_scheduler(total_steps)
        self.setup_pruning(total_steps)
        
        # Training loop
        training_stats = self._training_loop(
            train_dataloader, 
            validation_traces,
            total_steps
        )
        
        # Final model statistics
        final_stats = {
            'training_stats': training_stats,
            'model_stats': self.model.get_model_parameters(),
            'pruning_stats': self.pruner.get_sparsity_stats(self.model.model),
            'final_loss': self.best_loss
        }
        
        logger.info("Student training completed")
        return final_stats
    
    def _training_loop(self, 
                      dataloader: DataLoader,
                      validation_traces: Optional[List[ExpertTrace]],
                      total_steps: int) -> Dict[str, Any]:
        """Main training loop"""
        
        training_losses = []
        validation_losses = []
        
        # Training epochs
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training phase
            epoch_loss = self._train_epoch(dataloader)
            training_losses.append(epoch_loss)
            
            # Validation phase
            if validation_traces:
                val_loss = self._validate_epoch(validation_traces)
                validation_losses.append(val_loss)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint("best_model")
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch}")
            
            # Early stopping check
            if self._should_early_stop(validation_losses):
                logger.info("Early stopping triggered")
                break
        
        return {
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'total_steps': self.global_step,
            'epochs_completed': self.current_epoch + 1
        }
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.model.train()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Pruning step
            pruned = self.pruner.step(self.model.model, self.global_step)
            if pruned:
                # Apply masks after pruning
                self.pruner.apply_masks(self.model.model)
            
            # Update statistics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                'sparsity': f"{self.pruner.current_sparsity:.3f}"
            })
            
            # Logging
            if self.global_step % 10 == 0:
                self._log_training_step(loss.item(), batch_idx, num_batches)
        
        epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {self.current_epoch + 1} training loss: {epoch_loss:.4f}")
        
        return epoch_loss
    
    def _validate_epoch(self, validation_traces: List[ExpertTrace]) -> float:
        """Validation for one epoch"""
        if not validation_traces:
            return float('inf')
        
        self.model.model.eval()
        
        val_dataset = ExpertTraceDataset(
            validation_traces,
            self.model.tokenizer,
            self.config.data.max_sequence_length
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        total_loss = 0.0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs['loss'].item()
        
        val_loss = total_loss / num_batches
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        return val_loss
    
    def _log_training_step(self, loss: float, batch_idx: int, num_batches: int):
        """Log training step metrics"""
        if wandb.run is not None:
            wandb.log({
                'train_loss': loss,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'sparsity': self.pruner.current_sparsity,
                'epoch': self.current_epoch,
                'step': self.global_step
            })
    
    def _should_early_stop(self, validation_losses: List[float]) -> bool:
        """Check if early stopping should be triggered"""
        if len(validation_losses) < 3:
            return False
        
        # Stop if validation loss hasn't improved for 2 epochs
        recent_losses = validation_losses[-3:]
        return all(loss >= self.best_loss for loss in recent_losses)
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir) / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(checkpoint_dir))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Save pruning masks
        self.pruner.save_masks(str(checkpoint_dir / "pruning_masks.pt"))
        
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        self.model.load_checkpoint(str(checkpoint_dir))
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path)
            self.global_step = training_state['global_step']
            self.current_epoch = training_state['current_epoch']
            self.best_loss = training_state['best_loss']
            
            if self.optimizer and 'optimizer_state' in training_state:
                self.optimizer.load_state_dict(training_state['optimizer_state'])
            
            if self.scheduler and 'scheduler_state' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state'])
        
        # Load pruning masks
        pruning_masks_path = checkpoint_dir / "pruning_masks.pt"
        if pruning_masks_path.exists() and self.pruner:
            self.pruner.load_masks(str(pruning_masks_path))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def evaluate_model(self, test_traces: List[ExpertTrace]) -> Dict[str, Any]:
        """Evaluate trained model on test set"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.prepare_for_inference()
        
        results = {
            'num_examples': len(test_traces),
            'predictions': [],
            'metrics': {}
        }
        
        total_loss = 0.0
        correct_predictions = 0
        
        for trace in tqdm(test_traces, desc="Evaluating"):
            try:
                # Generate prediction
                ocr_text = "; ".join([text for text, _ in trace.ocr_data[:10]])
                prediction = self.model.generate_vqa_response(
                    image_description=ocr_text,
                    question=trace.question
                )
                
                # Compare with ground truth
                is_correct = self._compare_answers(
                    prediction['answer'], 
                    trace.answer
                )
                
                if is_correct:
                    correct_predictions += 1
                
                results['predictions'].append({
                    'question': trace.question,
                    'predicted_answer': prediction['answer'],
                    'ground_truth': trace.answer,
                    'is_correct': is_correct,
                    'predicted_bbox': prediction['bbox'].__dict__,
                    'ground_truth_bbox': trace.grounding_bbox.__dict__
                })
                
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_traces) if test_traces else 0.0
        
        results['metrics'] = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'model_stats': self.model.get_model_parameters()
        }
        
        logger.info(f"Evaluation completed: {accuracy:.3f} accuracy")
        return results
    
    def _compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """Simple answer comparison (can be made more sophisticated)"""
        from ..utils.grounding import AnswerGrounder
        grounder = AnswerGrounder()
        anls_score = grounder.compute_anls(predicted, ground_truth)
        return anls_score > 0.5
