"""
Exercise 3: Your Own Dataset
=============================

Goal: Train RNN on your own text and analyze what it learns.

Your Task:
- Choose your own dataset
- Prepare and clean the data
- Train the model
- Analyze learned patterns
- Experiment with hyperparameters

Learning Objectives:
1. Data preparation techniques
2. Understanding domain-specific patterns
3. Hyperparameter tuning
4. Overfitting vs underfitting detection

Time: 30-60 minutes
Difficulty: Medium ⏱️⏱️
"""

import numpy as np
import sys
sys.path.append('..')

from implementation import CharRNN
from train_minimal import train


def prepare_dataset(file_path):
    """
    Load and prepare your custom dataset.
    
    Args:
        file_path: Path to your text file
    
    Returns:
        text: Cleaned text string
        metadata: Dictionary with statistics
    
    TODO 1: Implement data loading and cleaning
    """
    print(f"Loading data from {file_path}...")
    
    # TODO: Load file
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     text = f.read()
    
    # TODO: Clean data (optional)
    # - Remove unwanted characters
    # - Normalize whitespace
    # - Handle encoding issues
    
    # TODO: Compute statistics
    metadata = {
        'length': 0,  # len(text)
        'vocab_size': 0,  # len(set(text))
        'unique_chars': [],  # sorted(set(text))
    }
    
    return None, metadata


def analyze_dataset(text):
    """
    Analyze characteristics of your dataset.
    
    TODO 2: Compute and print these statistics:
    - Total characters
    - Vocabulary size
    - Most common characters
    - Average word length (if text has words)
    - Patterns or structure
    """
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # TODO: Your analysis here
    print("\nTODO: Add statistics")


def train_on_custom_data(text, hidden_size=128, epochs=50):
    """
    Train RNN on your custom dataset.
    
    TODO 3: Implement training pipeline
    """
    print("\nTraining on custom dataset...")
    
    # TODO: Create vocabulary mappings
    # chars = sorted(list(set(text)))
    # char_to_idx = {ch: i for i, ch in enumerate(chars)}
    # idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # TODO: Create model
    # vocab_size = len(chars)
    # rnn = CharRNN(vocab_size, hidden_size)
    
    # TODO: Train
    # Use train_minimal.py's train function
    
    pass


def hyperparameter_experiments(text):
    """
    Experiment with different hyperparameters.
    
    TODO 4: Try these combinations and compare:
    
    1. Hidden size: 50, 100, 200
    2. Learning rate: 0.001, 0.01, 0.1
    3. Sequence length: 10, 25, 50
    
    For each, record:
    - Final loss
    - Training time
    - Sample quality
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER EXPERIMENTS")
    print("=" * 60)
    
    # TODO: Run experiments
    results = []
    
    # Example structure:
    # for hidden_size in [50, 100, 200]:
    #     for lr in [0.001, 0.01, 0.1]:
    #         # Train and record results
    #         pass
    
    # TODO: Print comparison table
    print("\nResults:")
    print("Hidden | LR    | Loss  | Quality | Time")
    print("-" * 50)
    # Print your results
    

def analyze_learned_patterns(rnn, idx_to_char):
    """
    Analyze what patterns the RNN learned.
    
    TODO 5: Answer these questions:
    
    1. What structure did it learn?
       - Words, phrases, punctuation patterns?
       - Domain-specific terminology?
    
    2. What mistakes does it make?
       - Grammar errors?
       - Made-up words?
       - Inconsistent style?
    
    3. How long until coherent output?
       - After how many epochs?
       - What changed in the samples?
    
    4. Does it capture the "style"?
       - Compare real vs generated
       - What's similar/different?
    """
    print("\n" + "=" * 60)
    print("LEARNED PATTERNS ANALYSIS")
    print("=" * 60)
    
    # TODO: Generate samples
    # samples = [rnn.sample(...) for _ in range(5)]
    
    # TODO: Analyze and answer questions
    print("\nYour analysis here...")


def compare_datasets():
    """
    Compare results across different datasets.
    
    TODO 6 (BONUS): Train on 3 different datasets and compare:
    - Code (Python, JavaScript, etc.)
    - Literature (Shakespeare, modern novels)
    - Social media (tweets, Reddit)
    
    How do results differ?
    """
    print("\n" + "=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)
    
    print("\nTODO: Compare 3 datasets")


if __name__ == "__main__":
    print(__doc__)
    
    print("\nSuggested datasets:")
    print("  📚 Books: Shakespeare, Jane Austen, Harry Potter")
    print("  💻 Code: Your GitHub repos, Linux kernel")
    print("  🎵 Lyrics: Your favorite artist")
    print("  📱 Social: Your Twitter archive")
    print("  📰 News: Wikipedia, news articles")
    
    # TODO 7: Specify your dataset path
    dataset_path = "your_dataset.txt"  # CHANGE THIS
    
    # Prepare data
    # text, metadata = prepare_dataset(dataset_path)
    
    # Analyze dataset
    # analyze_dataset(text)
    
    # Train model
    # train_on_custom_data(text)
    
    # Hyperparameter experiments
    # hyperparameter_experiments(text)
    
    # Analyze patterns
    # analyze_learned_patterns(rnn, idx_to_char)
    
    # Bonus: Compare datasets
    # compare_datasets()
    
    print("\n✅ Exercise 3 complete!")
    print("Compare your results with solutions/exercise_03_solution.md")
