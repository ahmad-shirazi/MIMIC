"""
Main entry point for MIMIC-VQA framework
"""
import argparse
import logging
import sys
from pathlib import Path

from mimic_vqa.config import Config, get_default_config
from mimic_vqa.teacher.agent import TeacherAgent
from mimic_vqa.student.trainer import StudentTrainer
from mimic_vqa.inference.pipeline import InferencePipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def phase_a_generate_expert_data(config: Config, args):
    """Run Phase A - Teacher Expert Data Generation"""
    logger.info("Starting Phase A: Teacher Expert Data Generation")
    
    # Initialize teacher agent
    teacher = TeacherAgent(config)
    
    # Load dataset
    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    import json
    with open(args.dataset_path, 'r') as f:
        dataset_items = json.load(f)
    
    logger.info(f"Loaded {len(dataset_items)} dataset items")
    
    # Generate expert traces
    expert_traces = teacher.generate_expert_dataset(
        dataset_items,
        output_path=args.output_path or "expert_traces.json"
    )
    
    # Print statistics
    stats = teacher.get_statistics(expert_traces)
    logger.info(f"Generated {stats['total_traces']} expert traces")
    logger.info(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    logger.info(f"Average answer length: {stats['avg_answer_length']:.1f} words")


def phase_b_train_student(config: Config, args):
    """Run Phase B - Student Training"""
    logger.info("Starting Phase B: Student Training")
    
    # Initialize trainer
    trainer = StudentTrainer(config)
    
    # Load expert traces
    if not Path(args.expert_traces_path).exists():
        raise FileNotFoundError(f"Expert traces not found: {args.expert_traces_path}")
    
    # Load traces using teacher agent (for convenience)
    from mimic_vqa.teacher.agent import TeacherAgent
    temp_teacher = TeacherAgent(config)
    expert_traces = temp_teacher.load_expert_dataset(args.expert_traces_path)
    
    # Split into train/validation if validation path not provided
    if args.validation_traces_path:
        validation_traces = temp_teacher.load_expert_dataset(args.validation_traces_path)
    else:
        # Use 10% for validation
        split_idx = int(0.9 * len(expert_traces))
        validation_traces = expert_traces[split_idx:]
        expert_traces = expert_traces[:split_idx]
    
    logger.info(f"Training on {len(expert_traces)} traces")
    logger.info(f"Validation on {len(validation_traces)} traces")
    
    # Train student model
    results = trainer.train(expert_traces, validation_traces)
    
    # Print results
    logger.info(f"Training completed in {results['training_stats']['epochs_completed']} epochs")
    logger.info(f"Final training loss: {results['training_stats']['training_losses'][-1]:.4f}")
    logger.info(f"Best validation loss: {results['final_loss']:.4f}")
    logger.info(f"Model parameters: {results['model_stats']['total_parameters']:,}")
    logger.info(f"Sparsity achieved: {results['pruning_stats']['overall_sparsity']:.3f}")


def phase_c_inference(config: Config, args):
    """Run Phase C - Student Inference"""
    logger.info("Starting Phase C: Student Inference")
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(
        config, 
        model_path=args.model_path,
        use_constrained_decoding=args.use_constrained_decoding
    )
    
    # Single image inference
    if args.image_path and args.question:
        logger.info(f"Running inference on: {args.image_path}")
        logger.info(f"Question: {args.question}")
        
        result = pipeline.infer(
            image=args.image_path,
            question=args.question,
            return_intermediate=args.verbose
        )
        
        # Print results
        print(f"\nAnswer: {result['answer']}")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Bounding Box: {result['bbox_string']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Inference Time: {result['inference_time']:.3f}s")
        
        if args.verbose and 'intermediate' in result:
            print("\nIntermediate Results:")
            print(f"Image Description: {result['intermediate']['image_description'][:200]}...")
    
    # Batch inference
    elif args.batch_file:
        import json
        with open(args.batch_file, 'r') as f:
            batch_data = json.load(f)
        
        logger.info(f"Running batch inference on {len(batch_data)} examples")
        
        # Convert to (image, question) pairs
        image_question_pairs = [(item['image'], item['question']) for item in batch_data]
        
        results = pipeline.batch_infer(image_question_pairs, show_progress=True)
        
        # Save results
        output_path = args.output_path or "inference_results.json"
        pipeline.save_results(results, output_path)
        
        # Print summary
        successful = sum(1 for r in results if r['question_answered'])
        logger.info(f"Batch inference completed: {successful}/{len(results)} successful")
        logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MIMIC-VQA Framework")
    
    # Global arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--phase", choices=['A', 'B', 'C'], required=True,
                       help="Phase to run: A (Teacher Data Generation), B (Student Training), C (Inference)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Phase A arguments
    parser.add_argument("--dataset-path", type=str, help="Path to input dataset (Phase A)")
    
    # Phase B arguments  
    parser.add_argument("--expert-traces-path", type=str, help="Path to expert traces (Phase B)")
    parser.add_argument("--validation-traces-path", type=str, help="Path to validation traces (Phase B)")
    
    # Phase C arguments
    parser.add_argument("--model-path", type=str, help="Path to trained model (Phase C)")
    parser.add_argument("--image-path", type=str, help="Path to input image (Phase C)")
    parser.add_argument("--question", type=str, help="Question about the image (Phase C)")
    parser.add_argument("--batch-file", type=str, help="Path to batch inference file (Phase C)")
    parser.add_argument("--use-constrained-decoding", action="store_true", 
                       help="Use constrained decoding for bbox generation")
    
    # Output arguments
    parser.add_argument("--output-path", type=str, help="Path for output files")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = Config.from_json(args.config)
        else:
            config = get_default_config()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Run appropriate phase
        if args.phase == 'A':
            if not args.dataset_path:
                parser.error("Phase A requires --dataset-path")
            phase_a_generate_expert_data(config, args)
            
        elif args.phase == 'B':
            if not args.expert_traces_path:
                parser.error("Phase B requires --expert-traces-path")
            phase_b_train_student(config, args)
            
        elif args.phase == 'C':
            if not args.model_path:
                parser.error("Phase C requires --model-path")
            if not (args.image_path and args.question) and not args.batch_file:
                parser.error("Phase C requires either (--image-path and --question) or --batch-file")
            phase_c_inference(config, args)
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
