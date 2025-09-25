"""
Iterative magnitude pruning for student model compression
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy

from ..config import Config

logger = logging.getLogger(__name__)


class IterativeMagnitudePruner:
    """
    Iterative magnitude pruning implementation for compressing Gemma-27B to 9B
    
    Implementation of "Iterative magnitude pruning to ~9B params" from Algorithm 1
    This achieves the target sparsity of ~65% to reduce from 27B to 9B parameters
    """
    
    def __init__(self, 
                 config: Config,
                 target_sparsity: float = 0.65,
                 pruning_frequency: int = 100):
        """
        Initialize iterative magnitude pruner
        
        Args:
            config: MIMIC-VQA configuration
            target_sparsity: Target sparsity level (0.65 = 65% reduction)
            pruning_frequency: Steps between pruning iterations
        """
        self.config = config
        self.target_sparsity = target_sparsity
        self.pruning_frequency = pruning_frequency
        
        # Pruning state
        self.current_sparsity = 0.0
        self.pruning_step = 0
        self.total_pruning_steps = 0
        
        # Masks for pruned parameters
        self.masks = {}
        self.original_weights = {}
        
        # Pruning schedule
        self.sparsity_schedule = []
        
        logger.info(f"Iterative magnitude pruner initialized (target: {target_sparsity:.1%})")
    
    def initialize_pruning(self, model: nn.Module, total_training_steps: int):
        """
        Initialize pruning schedule and masks
        
        Args:
            model: Model to be pruned
            total_training_steps: Total number of training steps
        """
        self.total_pruning_steps = total_training_steps // self.pruning_frequency
        
        # Create sparsity schedule (gradual increase)
        self._create_sparsity_schedule()
        
        # Initialize masks and store original weights
        self._initialize_masks(model)
        
        logger.info(f"Pruning initialized: {self.total_pruning_steps} pruning steps")
    
    def _create_sparsity_schedule(self):
        """
        Create gradual sparsity schedule
        
        Uses polynomial schedule to gradually increase sparsity
        """
        if self.total_pruning_steps == 0:
            self.sparsity_schedule = [self.target_sparsity]
            return
        
        # Polynomial schedule (cubic by default)
        for step in range(self.total_pruning_steps):
            progress = step / max(1, self.total_pruning_steps - 1)
            # Cubic schedule: slow start, accelerated middle, slower end
            sparsity = self.target_sparsity * (progress ** 3)
            self.sparsity_schedule.append(sparsity)
        
        # Ensure final sparsity matches target
        self.sparsity_schedule[-1] = self.target_sparsity
        
        logger.info(f"Sparsity schedule: {self.sparsity_schedule[0]:.3f} -> {self.sparsity_schedule[-1]:.3f}")
    
    def _initialize_masks(self, model: nn.Module):
        """Initialize pruning masks for all parameters"""
        for name, param in model.named_parameters():
            if self._should_prune_parameter(name, param):
                # Initialize mask as all ones (no pruning initially)
                self.masks[name] = torch.ones_like(param.data)
                # Store original weights for recovery if needed
                self.original_weights[name] = param.data.clone()
                
        logger.info(f"Initialized masks for {len(self.masks)} parameters")
    
    def _should_prune_parameter(self, name: str, param: torch.Tensor) -> bool:
        """
        Determine if parameter should be pruned
        
        Args:
            name: Parameter name
            param: Parameter tensor
            
        Returns:
            True if parameter should be pruned
        """
        # Don't prune biases, layer norms, or embeddings
        skip_patterns = [
            'bias', 'norm', 'embedding', 'embed', 
            'layernorm', 'layer_norm', 'ln'
        ]
        
        if any(pattern in name.lower() for pattern in skip_patterns):
            return False
        
        # Only prune parameters with sufficient size
        if param.numel() < 1000:  # Skip very small parameters
            return False
        
        # Only prune 2D parameters (weight matrices)
        if len(param.shape) < 2:
            return False
        
        return True
    
    def step(self, model: nn.Module, step_num: int) -> bool:
        """
        Perform pruning step if needed
        
        Args:
            model: Model to prune
            step_num: Current training step
            
        Returns:
            True if pruning was performed
        """
        if step_num % self.pruning_frequency != 0:
            return False
        
        pruning_iteration = step_num // self.pruning_frequency
        
        if pruning_iteration >= len(self.sparsity_schedule):
            return False
        
        target_sparsity = self.sparsity_schedule[pruning_iteration]
        
        # Perform pruning
        self._prune_model(model, target_sparsity)
        
        self.current_sparsity = target_sparsity
        self.pruning_step += 1
        
        logger.info(f"Pruning step {self.pruning_step}: sparsity = {target_sparsity:.3f}")
        return True
    
    def _prune_model(self, model: nn.Module, target_sparsity: float):
        """
        Prune model to achieve target sparsity using magnitude-based pruning
        
        Args:
            model: Model to prune
            target_sparsity: Target sparsity level
        """
        # Collect all prunable parameters and their magnitudes
        all_weights = []
        param_info = []
        
        for name, param in model.named_parameters():
            if name in self.masks:
                weights = param.data.abs().flatten()
                all_weights.append(weights)
                param_info.append((name, param, len(weights)))
        
        if not all_weights:
            logger.warning("No parameters to prune")
            return
        
        # Concatenate all weights
        all_weights = torch.cat(all_weights)
        
        # Find threshold for target sparsity
        num_params = len(all_weights)
        num_to_prune = int(target_sparsity * num_params)
        
        if num_to_prune > 0:
            # Get threshold (k-th smallest magnitude)
            threshold, _ = torch.kthvalue(all_weights, num_to_prune)
            threshold = threshold.item()
        else:
            threshold = 0.0
        
        # Apply pruning masks
        total_params = 0
        pruned_params = 0
        
        for name, param, param_size in param_info:
            # Create mask based on magnitude threshold
            mask = (param.data.abs() > threshold).float()
            
            # Handle ties by randomly pruning some parameters at threshold
            if threshold > 0:
                at_threshold = (param.data.abs() == threshold)
                if at_threshold.sum() > 0:
                    # Randomly prune some parameters at threshold
                    random_mask = torch.rand_like(param.data) > 0.5
                    mask = mask + (at_threshold.float() * random_mask.float())
                    mask = torch.clamp(mask, 0, 1)
            
            # Update mask
            self.masks[name] = mask
            
            # Apply mask to parameter
            param.data.mul_(mask)
            
            # Update statistics
            total_params += param.numel()
            pruned_params += (mask == 0).sum().item()
        
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Applied pruning: {actual_sparsity:.3f} sparsity "
                   f"({pruned_params}/{total_params} params)")
    
    def apply_masks(self, model: nn.Module):
        """Apply pruning masks to model parameters"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data.mul_(self.masks[name])
    
    def get_sparsity_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get sparsity statistics for the model
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with sparsity statistics
        """
        stats = {
            'total_params': 0,
            'pruned_params': 0,
            'per_layer_sparsity': {},
            'overall_sparsity': 0.0,
            'pruning_step': self.pruning_step,
            'target_sparsity': self.target_sparsity
        }
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            stats['total_params'] += param_count
            
            if name in self.masks:
                mask = self.masks[name]
                pruned_count = (mask == 0).sum().item()
                layer_sparsity = pruned_count / param_count
                
                stats['pruned_params'] += pruned_count
                stats['per_layer_sparsity'][name] = {
                    'sparsity': layer_sparsity,
                    'pruned': pruned_count,
                    'total': param_count
                }
        
        if stats['total_params'] > 0:
            stats['overall_sparsity'] = stats['pruned_params'] / stats['total_params']
        
        return stats
    
    def save_masks(self, filepath: str):
        """Save pruning masks to file"""
        torch.save({
            'masks': self.masks,
            'current_sparsity': self.current_sparsity,
            'pruning_step': self.pruning_step,
            'target_sparsity': self.target_sparsity
        }, filepath)
        logger.info(f"Pruning masks saved to {filepath}")
    
    def load_masks(self, filepath: str):
        """Load pruning masks from file"""
        checkpoint = torch.load(filepath)
        self.masks = checkpoint['masks']
        self.current_sparsity = checkpoint['current_sparsity']
        self.pruning_step = checkpoint.get('pruning_step', 0)
        self.target_sparsity = checkpoint.get('target_sparsity', self.target_sparsity)
        
        logger.info(f"Pruning masks loaded from {filepath}")
    
    def recover_weights(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Recover original weights for specified layers (useful for fine-tuning)
        
        Args:
            model: Model to recover weights for
            layer_names: Specific layer names to recover (None for all)
        """
        recovered = 0
        
        for name, param in model.named_parameters():
            if name in self.original_weights:
                if layer_names is None or name in layer_names:
                    param.data.copy_(self.original_weights[name])
                    # Reset mask
                    if name in self.masks:
                        self.masks[name] = torch.ones_like(param.data)
                    recovered += 1
        
        logger.info(f"Recovered weights for {recovered} parameters")
    
    def get_model_size_reduction(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size reduction achieved by pruning
        
        Args:
            model: Pruned model
            
        Returns:
            Size reduction statistics
        """
        stats = self.get_sparsity_stats(model)
        
        original_size = stats['total_params']
        effective_size = stats['total_params'] - stats['pruned_params']
        
        reduction_ratio = stats['pruned_params'] / original_size if original_size > 0 else 0
        compression_ratio = original_size / effective_size if effective_size > 0 else float('inf')
        
        return {
            'original_params': original_size,
            'effective_params': effective_size,
            'pruned_params': stats['pruned_params'],
            'reduction_ratio': reduction_ratio,
            'compression_ratio': compression_ratio,
            'size_mb_original': original_size * 4 / (1024 * 1024),  # Assuming float32
            'size_mb_effective': effective_size * 4 / (1024 * 1024)
        }
    
    def create_structured_masks(self, model: nn.Module, block_size: int = 4) -> Dict[str, torch.Tensor]:
        """
        Create structured pruning masks (block pruning)
        
        Args:
            model: Model to create masks for
            block_size: Size of blocks for structured pruning
            
        Returns:
            Structured pruning masks
        """
        structured_masks = {}
        
        for name, param in model.named_parameters():
            if name in self.masks:
                original_mask = self.masks[name]
                
                if len(param.shape) == 2:  # Weight matrix
                    # Reshape to blocks
                    h, w = param.shape
                    h_blocks = h // block_size
                    w_blocks = w // block_size
                    
                    if h_blocks > 0 and w_blocks > 0:
                        # Reshape and check block sparsity
                        reshaped = original_mask[:h_blocks*block_size, :w_blocks*block_size]
                        blocks = reshaped.view(h_blocks, block_size, w_blocks, block_size)
                        
                        # Block is pruned if more than half of its elements are pruned
                        block_masks = (blocks.sum(dim=(1, 3)) > (block_size * block_size * 0.5)).float()
                        
                        # Expand back to original shape
                        structured_mask = block_masks.unsqueeze(1).unsqueeze(3).repeat(1, block_size, 1, block_size)
                        structured_mask = structured_mask.view(h_blocks*block_size, w_blocks*block_size)
                        
                        # Pad if necessary
                        if structured_mask.shape != param.shape:
                            padded_mask = torch.ones_like(param)
                            padded_mask[:structured_mask.shape[0], :structured_mask.shape[1]] = structured_mask
                            structured_mask = padded_mask
                        
                        structured_masks[name] = structured_mask
                    else:
                        structured_masks[name] = original_mask
                else:
                    structured_masks[name] = original_mask
        
        return structured_masks
