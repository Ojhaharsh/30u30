"""
Exercise 2: Temperature Sampling
=================================

Goal: Understand how temperature affects text generation quality.

Your Task:
- Train an RNN on text
- Generate samples with different temperatures
- Analyze the trade-offs
- Write a short report

Learning Objectives:
1. How temperature controls randomness
2. Trade-off between creativity and coherence
3. When to use high vs low temperature
4. Mathematical effect on softmax

Time: 15-30 minutes
Difficulty: Quick ⏱️
"""

import numpy as np
import sys
sys.path.append('..')

from implementation import CharRNN
from train_minimal import load_data


def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax with temperature scaling.
    
    Args:
        logits: Raw network outputs
        temperature: Controls randomness (higher = more random)
    
    Returns:
        probabilities: Softmax probabilities
    
    TODO 1: Implement temperature-scaled softmax
    Hint: p = exp(logits/T) / sum(exp(logits/T))
    """
    # Your code here
    pass


def generate_with_temperature(rnn, idx_to_char, seed_char, length, temperature):
    """
    Generate text with specific temperature.
    
    Args:
        rnn: Trained CharRNN model
        idx_to_char: Index to character mapping
        seed_char: Starting character
        length: Number of characters to generate
        temperature: Sampling temperature
    
    Returns:
        generated_text: String of generated text
    
    TODO 2: Implement temperature-based generation
    """
    # Your code here
    pass


def experiment_with_temperatures():
    """
    Run temperature sampling experiment.
    
    TODO 3: Complete this experiment
    """
    # Load data
    print("Loading data...")
    # TODO: Load your training data
    
    # Train model
    print("Training model...")
    # TODO: Train RNN (or load pre-trained)
    
    # Try different temperatures
    temperatures = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    print("\n" + "=" * 60)
    print("TEMPERATURE SAMPLING EXPERIMENTS")
    print("=" * 60)
    
    for temp in temperatures:
        # TODO 4: Generate sample with this temperature
        # sample = generate_with_temperature(...)
        
        print(f"\nTemperature = {temp}")
        print("-" * 60)
        # print(sample[:200])
        pass


def analyze_temperature_effects():
    """
    Analyze what happens at different temperatures.
    
    TODO 5: Answer these questions in your analysis:
    
    1. What happens at very low temperature (0.2)?
       - Is the text coherent?
       - Is it repetitive?
       - Why does this happen mathematically?
    
    2. What happens at medium temperature (1.0)?
       - Quality of generated text?
       - Balance of creativity vs coherence?
    
    3. What happens at high temperature (2.0)?
       - Is the text still readable?
       - Does it make sense?
       - Why does it become random?
    
    4. What is the "sweet spot" temperature?
       - For your specific dataset
       - Why this value?
    
    5. Mathematical insight:
       - How does T affect softmax distribution?
       - Plot: probability distribution at different temps
    """
    print("\n" + "=" * 60)
    print("TEMPERATURE ANALYSIS")
    print("=" * 60)
    
    print("\nYour analysis here...")
    print("\n1. Low temperature (0.2):")
    print("   - TODO: Your observations")
    
    print("\n2. Medium temperature (1.0):")
    print("   - TODO: Your observations")
    
    print("\n3. High temperature (2.0):")
    print("   - TODO: Your observations")
    
    print("\n4. Sweet spot:")
    print("   - TODO: Your recommendation")
    
    print("\n5. Mathematical insight:")
    print("   - TODO: Explain softmax behavior")


def visualize_temperature_effect():
    """
    Visualize how temperature affects probability distribution.
    
    TODO 6 (BONUS): Create visualization
    """
    import matplotlib.pyplot as plt
    
    # Example logits
    logits = np.array([2.0, 1.0, 0.5, 0.1])
    temperatures = [0.5, 1.0, 2.0]
    
    # TODO: Plot softmax distribution at each temperature
    # This should show how temperature "flattens" the distribution
    
    pass


if __name__ == "__main__":
    print(__doc__)
    
    # Run experiments
    experiment_with_temperatures()
    
    # Analyze results
    analyze_temperature_effects()
    
    # Visualize (bonus)
    visualize_temperature_effect()
    
    print("\n✅ Exercise 2 complete!")
    print("Compare your analysis with solutions/exercise_02_solution.md")
