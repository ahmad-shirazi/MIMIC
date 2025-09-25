"""
Configuration management for MIMIC-VQA framework
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    # Teacher models
    teacher_planner: str = "llama-4-scout"  # Llama 4 Scout for planning
    teacher_qa: str = "gemini-3-27b"        # Gemma-3-27B for QA
    
    # Student models  
    student_base: str = "gemini-3-27b"      # Base Gemma-3-27B
    student_target: str = "gemini-3-9b"     # Pruned Gemma-3-9B
    
    # Model paths
    base_model_path: str = "google/gemma-2-27b-it"
    student_model_path: str = "google/gemma-2-9b-it"
    
    # Quantization settings
    use_4bit: bool = True
    use_8bit: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # General training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Pruning parameters
    target_sparsity: float = 0.65  # 65% reduction from 27B to 9B
    pruning_frequency: int = 100   # Steps between pruning
    
    # Distillation parameters
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    
    # Hardware
    num_gpus: int = 4
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Dataset parameters
    max_sequence_length: int = 2048
    max_bbox_tokens: int = 100
    
    # OCR parameters
    ocr_engine: str = "paddleocr"  # or "easyocr"
    ocr_confidence_threshold: float = 0.5
    
    # Context retrieval
    top_k_segments: int = 10
    similarity_threshold: float = 0.3
    context_window: int = 512
    
    # Data paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    expert_data_path: str = "data/expert_traces"


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    # Generation parameters
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Constrained decoding
    use_constrained_decoding: bool = True
    coordinate_tolerance: int = 5  # ±5 pixels
    
    # Output format
    return_reasoning: bool = True
    return_confidence: bool = True


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Environment
    device: str = "cuda"
    seed: int = 42
    log_level: str = "INFO"
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # API keys (loaded from environment)
    gemini_api_key: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load environment variables"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.wandb_api_key = os.getenv("WANDB_API_KEY")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'training', 'data', 'inference']}
        )
    
    def to_json(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'inference': self.inference.__dict__,
            'device': self.device,
            'seed': self.seed,
            'log_level': self.log_level,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
