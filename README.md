# ArabicSense Benchmark

ArabicSense is a benchmark for evaluating commonsense reasoning in Arabic using Large Language Models (LLMs). It includes three distinct tasks to test world-knowledge and reasoning abilities.

## Tasks

1. **Commonsense Validation**:
   - Given two sentences, identify which one does not make sense.
   
2. **Commonsense Explanation (Multiple Choice)**:
   - After identifying a nonsensical sentence, select the correct reason from three options explaining why it contradicts commonsense.
   
3. **Commonsense Explanation (Generation)**:
   - Generate a natural language explanation for why a statement is nonsensical.

## Dataset
The dataset is derived from Arabic Wikipedia and processed with GPT-4 to generate task-specific examples. It has been validated by human annotators for quality and relevance.

## Models
The project evaluates:
- Pre-trained Arabic BERT models (e.g., AraBERT, MarBERT, CamelBERT).
- State-of-the-art causal LLMs (e.g., Mistral-7b, LLaMA-3, Gemma).

Fine-tuning the models on the ArabicSense dataset shows significant improvements in performance, emphasizing the value of task-specific training.

## Code Structure

- `Task1.py`: Implements Tasks A (Commonsense Validation) and B (Commonsense Explanation - Multiple Choice).
- `Task2.py`: Implements Task C (Commonsense Explanation - Generation).

## Results
- Fine-tuned AraBERT v2 achieved 87% F1 on Task A and 83% F1 on Task B.
- LLaMA-3 achieved 77.3% BERTScore F1 for Task C generation.

## Usage

1. Clone the repository.
2. Install required dependencies (`requirements.txt`).
3. Run the scripts for each task:
   ```bash
   python Task1.py
   python Task2.py

## Citation

@article{ArabicSense2025,
  title={ArabicSense: A Benchmark for Evaluating Commonsense Reasoning in Arabic},
  author={Anonymous},
  journal={COLING},
  year={2025}
}
