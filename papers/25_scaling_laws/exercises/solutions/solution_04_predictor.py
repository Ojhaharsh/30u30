"""
Day 25 Solution 4: GPT-3 Performance Predictor

This solution uses the power-law formula to extrapolate performance.
"""

import math

def predict_loss(n_params):
    """
    L(N) = 1.69 * N^-0.076
    """
    return 1.69 * math.pow(n_params, -0.076)

if __name__ == "__main__":
    n_gpt3 = 175e9
    
    predicted_l = predict_loss(n_gpt3)
    
    print(f"Target Model: GPT-3 (175 Billion Parameters)")
    print(f"Predicted Loss: {predicted_l:.4f}")
    
    # Context
    print("\nNote: This predictability is why OpenAI felt confident building GPT-3.")
    print("By fitting a curve to models that cost $100, they could predict the")
    print("performance of a model that cost $10,000,000.")
