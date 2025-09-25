#!/usr/bin/env python3
"""
Quick start example for MIMIC-VQA framework
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mimic_vqa import Config, TeacherAgent, StudentTrainer, InferencePipeline
from mimic_vqa.config import get_default_config

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = [
        {
            "image": "sample_document.jpg",  # Replace with actual image path
            "question": "What is the department name?",
            "answer": "COMPUTER SCIENCE"  # Optional ground truth
        },
        {
            "image": "sample_document.jpg",
            "question": "What is the total amount?",
            "answer": "$1,234.56"
        }
    ]
    return sample_data


def example_phase_a_teacher_data_generation():
    """Example: Phase A - Teacher Expert Data Generation"""
    print("=== Phase A: Teacher Expert Data Generation ===")
    
    # Initialize configuration
    config = get_default_config()
    
    # Initialize teacher agent
    teacher = TeacherAgent(config)
    
    # Create sample data
    sample_data = create_sample_data()
    
    print(f"Generating expert traces for {len(sample_data)} samples...")
    
    # Generate expert traces
    try:
        expert_traces = teacher.generate_expert_dataset(
            sample_data,
            output_path="expert_traces_sample.json"
        )
        
        print(f"✓ Generated {len(expert_traces)} expert traces")
        
        # Show statistics
        stats = teacher.get_statistics(expert_traces)
        print(f"  - Average processing time: {stats['avg_processing_time']:.3f}s")
        print(f"  - Average answer length: {stats['avg_answer_length']:.1f} words")
        
        return expert_traces
        
    except Exception as e:
        print(f"✗ Phase A failed: {e}")
        return []


def example_phase_b_student_training(expert_traces):
    """Example: Phase B - Student Training"""
    print("\n=== Phase B: Student Training ===")
    
    if not expert_traces:
        print("No expert traces available for training")
        return None
    
    # Initialize configuration
    config = get_default_config()
    
    # Adjust for quick demo
    config.training.num_epochs = 1
    config.training.batch_size = 1
    
    # Initialize trainer
    trainer = StudentTrainer(config)
    
    print(f"Training student model on {len(expert_traces)} expert traces...")
    
    try:
        # Split data (using same data for train/val in this demo)
        train_traces = expert_traces
        val_traces = expert_traces[:1]  # Just one for demo
        
        # Train student model
        results = trainer.train(train_traces, val_traces)
        
        print("✓ Student training completed")
        print(f"  - Final loss: {results['final_loss']:.4f}")
        print(f"  - Model sparsity: {results['pruning_stats']['overall_sparsity']:.3f}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Phase B failed: {e}")
        return None


def example_phase_c_inference():
    """Example: Phase C - Student Inference"""
    print("\n=== Phase C: Student Inference ===")
    
    # Initialize configuration
    config = get_default_config()
    
    # Initialize inference pipeline (will use base model if no trained model available)
    pipeline = InferencePipeline(
        config,
        model_path=None,  # Use base model for demo
        use_constrained_decoding=True
    )
    
    # Example inference
    sample_image = "sample_document.jpg"  # Replace with actual image path
    sample_question = "What is the department name?"
    
    print(f"Running inference...")
    print(f"Question: {sample_question}")
    
    try:
        # Check if sample image exists
        if not os.path.exists(sample_image):
            print(f"⚠ Sample image not found: {sample_image}")
            print("Please provide a valid image path in the sample_data")
            return
        
        result = pipeline.infer(
            image=sample_image,
            question=sample_question,
            return_intermediate=True
        )
        
        print("✓ Inference completed")
        print(f"  - Answer: {result['answer']}")
        print(f"  - Reasoning: {result['reasoning'][:100]}...")
        print(f"  - Bounding Box: {result['bbox_string']}")
        print(f"  - Confidence: {result['confidence']:.3f}")
        print(f"  - Inference Time: {result['inference_time']:.3f}s")
        
    except Exception as e:
        print(f"✗ Phase C failed: {e}")


def main():
    """Run complete MIMIC-VQA pipeline demo"""
    print("MIMIC-VQA Framework Quick Start Demo")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠ Warning: GEMINI_API_KEY not set in environment")
        print("Please set your Gemini API key for full functionality")
    
    # Phase A: Generate expert data
    expert_traces = example_phase_a_teacher_data_generation()
    
    # Phase B: Train student model (optional for demo)
    # trainer = example_phase_b_student_training(expert_traces)
    
    # Phase C: Run inference
    example_phase_c_inference()
    
    print("\n" + "=" * 50)
    print("Quick start demo completed!")
    print("\nNext steps:")
    print("1. Prepare your dataset with real images and questions")
    print("2. Set up your API keys in .env file")
    print("3. Run full training with: python main.py --phase B ...")
    print("4. Deploy trained model for inference")


if __name__ == "__main__":
    main()
