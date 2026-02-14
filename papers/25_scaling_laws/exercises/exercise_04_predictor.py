"""
Day 25 Exercise 4: GPT-3 Performance Predictor

In this exercise, you will act as an OpenAI researcher in early 2020.
You have trained several small models (1M to 100M params) and have a fitted
scaling law. Your task is to predict the loss of a 175B parameter model.

Scaling Law: L(N) = 1.69 * N^-0.076

Instructions:
1. Implement the predict_loss function.
2. Calculate the cross-entropy loss for N = 175 * 10^9.
"""

def predict_loss(n_params):
    """
    TODO: Use the law L(N) = 1.69 * n_params^-0.076 to predict loss.
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    n_gpt3 = 175e9
    
    predicted_l = predict_loss(n_gpt3)
    
    if predicted_l is not None:
        print(f"Target Model: GPT-3 (175 Billion Parameters)")
        print(f"Predicted Loss: {predicted_l:.4f}")
        
        # In reality, GPT-3 achieved approx 1.7 - 2.0 based on the dataset
        if 1.5 < predicted_l < 2.5:
            print("[OK] Your prediction is in the ballpark of the scaling law.")
    else:
        print("[FAIL] predict_loss not implemented.")
