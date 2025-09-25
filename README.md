# MIMIC-VQA: Teacher-Student Knowledge Distillation Framework

A comprehensive implementation of the MIMIC-VQA framework for Visual Question Answering on documents, featuring teacher-student knowledge distillation with iterative magnitude pruning.

## Overview

MIMIC-VQA is a two-phase framework that transfers expert reasoning from a powerful "teacher" agent into an efficient, end-to-end "student" model:

- **Phase A (Teacher Expert Data Generation)**: A teacher agent decomposes document VQA into modular steps: OCR extraction, context retrieval, teacher QA, deterministic grounding, and response formatting.
- **Phase B (Student Training)**: A student VLM (Gemma-3-27B → 9B) is trained via cross-entropy loss on teacher traces with iterative magnitude pruning.
- **Phase C (Inference)**: The deployed student generates chain-of-thought, final answer, and spatial coordinates in one pass.

## Architecture

```
Phase A - Teacher Expert Data Generation (Yellow)
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ OCR Extract │ -> │ FindText QA  │ -> │ GroundAnswer│ -> │ Format String│
│ tokens &    │    │ over context │    │ maps answer │    │ with reasoning│
│ bboxes      │    │ returns text │    │ text to bbox│    │ and location │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘

Phase B - Student Training (Blue)  
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Base VLM    │ -> │ Iterative    │ -> │ Student VLM │ -> │ Cross-entropy│
│ Gemma 3-27B │    │ magnitude    │    │ Gemma 3-9B  │    │ on teacher   │
│             │    │ pruning      │    │             │    │ strings      │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘

Phase C - Inference (Green)
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Student     │ -> │ Parse outputs│ -> │ Answer +    │
│ generates   │    │ reasoning &  │    │ BBox coords │
│ reasoning   │    │ final answer │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Features

- **Modular Teacher Pipeline**: OCR extraction, semantic context retrieval, answer generation, and spatial grounding
- **Iterative Magnitude Pruning**: Systematic compression from 27B to 9B parameters
- **Constrained Decoding**: OCR-guided bounding box generation for improved spatial accuracy
- **End-to-End Training**: Cross-entropy loss on teacher reasoning traces
- **Multiple OCR Backends**: Support for PaddleOCR and EasyOCR
- **Flexible Configuration**: JSON-based configuration system
- **Performance Monitoring**: Comprehensive logging and metrics

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd mimic-code
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env_example.txt .env
# Edit .env with your API keys
```

Required environment variables:
- `GEMINI_API_KEY`: Google Gemini API key for teacher QA model
- `WANDB_API_KEY`: Weights & Biases API key (optional, for training monitoring)

## Quick Start

### 1. Run the Demo

```bash
python examples/quick_start.py
```

This demonstrates all three phases with sample data.

### 2. Phase A: Generate Expert Data

Prepare your dataset in JSON format:
```json
[
  {
    "image": "path/to/document1.jpg",
    "question": "What is the department name?",
    "answer": "Computer Science"
  },
  {
    "image": "path/to/document2.jpg", 
    "question": "What is the total amount?",
    "answer": "$1,234.56"
  }
]
```

Generate expert traces:
```bash
python main.py --phase A --dataset-path data/dataset.json --output-path expert_traces.json
```

### 3. Phase B: Train Student Model

```bash
python main.py --phase B \
  --expert-traces-path expert_traces.json \
  --validation-traces-path validation_traces.json
```

### 4. Phase C: Run Inference

Single image inference:
```bash
python main.py --phase C \
  --model-path checkpoints/best_model \
  --image-path sample.jpg \
  --question "What is the department name?" \
  --use-constrained-decoding
```

Batch inference:
```bash
python main.py --phase C \
  --model-path checkpoints/best_model \
  --batch-file batch_questions.json \
  --output-path results.json
```

## Configuration

The framework uses a hierarchical configuration system. Create custom configs:

```json
{
  "model": {
    "teacher_qa": "gemini-1.5-pro",
    "student_base": "google/gemma-2-27b-it",
    "use_4bit": true
  },
  "training": {
    "num_epochs": 3,
    "batch_size": 4,
    "target_sparsity": 0.65,
    "pruning_frequency": 100
  },
  "inference": {
    "use_constrained_decoding": true,
    "temperature": 0.7
  }
}
```

Use with: `python main.py --config custom_config.json --phase A ...`

## API Usage

### Teacher Agent

```python
from mimic_vqa import Config, TeacherAgent

config = Config()
teacher = TeacherAgent(config)

# Generate single expert trace
trace = teacher.generate_expert_trace(
    image="document.jpg",
    question="What is the total amount?"
)

print(f"Teacher reasoning: {trace.teacher_string}")
```

### Student Training

```python
from mimic_vqa import StudentTrainer

trainer = StudentTrainer(config)
results = trainer.train(expert_traces, validation_traces)

print(f"Final sparsity: {results['pruning_stats']['overall_sparsity']:.3f}")
```

### Inference Pipeline

```python
from mimic_vqa import InferencePipeline

pipeline = InferencePipeline(
    config, 
    model_path="trained_model/",
    use_constrained_decoding=True
)

result = pipeline.infer(
    image="document.jpg",
    question="What is the department name?"
)

print(f"Answer: {result['answer']}")
print(f"Location: {result['bbox_coordinates']}")
```

## Model Architecture

### Teacher Components

- **Planner**: Llama 4 Scout for orchestrating reasoning steps
- **QA Model**: Gemini-3-27B for answer generation
- **OCR Engine**: PaddleOCR/EasyOCR for text extraction
- **Context Retriever**: Sentence-BERT for semantic similarity
- **Answer Grounder**: ANLS-based spatial grounding

### Student Model

- **Base Model**: Gemma-3-27B (27 billion parameters)
- **Target Model**: Gemma-3-9B (9 billion parameters, 65% sparsity)
- **Training**: Cross-entropy loss on teacher sequences
- **Pruning**: Iterative magnitude pruning during training
- **Output**: Chain-of-thought + answer + bounding box

## Performance

### Model Compression
- **Original**: 27B parameters
- **Compressed**: 9B parameters (65% reduction)
- **Method**: Iterative magnitude pruning
- **Performance**: Maintains reasoning capability with 3x size reduction

### Inference Speed
- **End-to-end**: Single forward pass
- **Constrained Decoding**: OCR-guided bbox generation
- **Throughput**: ~X examples/second (hardware dependent)

## Advanced Features

### Constrained Decoding

Enable OCR-guided bounding box generation:

```python
pipeline = InferencePipeline(
    config,
    use_constrained_decoding=True
)

result = pipeline.infer(image, question)
# Bounding box coordinates are constrained to valid OCR regions
```

### Custom OCR Engines

```python
from mimic_vqa.utils import OCRExtractor

# Use EasyOCR
ocr = OCRExtractor(engine="easyocr", confidence_threshold=0.7)

# Use PaddleOCR  
ocr = OCRExtractor(engine="paddleocr", confidence_threshold=0.5)
```

### Pruning Strategies

```python
from mimic_vqa.student import IterativeMagnitudePruner

pruner = IterativeMagnitudePruner(
    config,
    target_sparsity=0.70,  # 70% parameter reduction
    pruning_frequency=50   # Prune every 50 steps
)
```

## Evaluation

### Built-in Metrics

The framework includes evaluation on standard datasets:
- DocVQA
- FUNSD  
- CORD
- SROIE

### Custom Evaluation

```python
# Evaluate trained model
results = trainer.evaluate_model(test_traces)
print(f"Accuracy: {results['metrics']['accuracy']:.3f}")

# Benchmark inference speed
benchmark_results = pipeline.benchmark(test_data, num_runs=5)
print(f"Throughput: {benchmark_results['throughput_examples_per_sec']:.2f} examples/sec")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `config.training.batch_size = 1`
   - Enable gradient checkpointing: `config.training.gradient_checkpointing = True`
   - Use 4-bit quantization: `config.model.use_4bit = True`

2. **OCR extraction fails**
   - Install OCR dependencies: `pip install paddlepaddle paddleocr`
   - Try different engine: `config.data.ocr_engine = "easyocr"`
   - Lower confidence threshold: `config.data.ocr_confidence_threshold = 0.3`

3. **Gemini API errors**
   - Check API key: `echo $GEMINI_API_KEY`
   - Verify API quotas and limits
   - Add retry logic for rate limiting

### Performance Optimization

1. **Training Speed**
   - Use multiple GPUs: `config.training.num_gpus = 4`
   - Mixed precision: `config.training.mixed_precision = "fp16"`
   - Gradient checkpointing: `config.training.gradient_checkpointing = True`

2. **Inference Speed**
   - Disable constrained decoding for speed: `use_constrained_decoding=False`
   - Use smaller batch sizes
   - Quantize model: `config.model.use_4bit = True`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mimic-vqa-2024,
  title={MIMIC-VQA: Teacher-Student Knowledge Distillation Framework for Visual Question Answering on Documents},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Acknowledgments

- Google Gemini for the base language models
- PaddlePaddle and EasyOCR teams for OCR engines
- Hugging Face for model infrastructure
- The document VQA research community

## Support

For issues and questions:
- Open a GitHub issue
- Check the documentation
- Review example scripts in `examples/`

---

**MIMIC-VQA Framework** - Efficient Document Visual Question Answering through Teacher-Student Knowledge Distillation
