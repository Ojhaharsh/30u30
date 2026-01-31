"""
Exercise 4: Monte Carlo Dropout for Uncertainty

Goal: Use dropout at inference time for uncertainty estimation.

Time: 60 minutes
Difficulty: Hard ⏱️⏱️⏱️⏱️
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mc_dropout_inference(model, x, n_samples=100):
    """
    Run multiple forward passes with dropout enabled for uncertainty estimation.
    
    Args:
        model: Model with dropout layers
        x: Input sample(s)
        n_samples: Number of forward passes
        
    Returns:
        mean_prediction: Average prediction across samples
        std_prediction: Standard deviation (uncertainty)
        
    TODO: Implement this function:
    1. Set model to training mode (keep dropout active)
    2. Run n_samples forward passes
    3. Collect all predictions
    4. Compute mean and standard deviation
    """
    # TODO: Implement MC Dropout inference
    # model.train()  # Keep dropout ON!
    
    # predictions = []
    # for _ in range(n_samples):
    #     pred = model.forward(x)
    #     predictions.append(pred)
    
    # predictions = np.stack(predictions, axis=0)
    # mean_prediction = predictions.mean(axis=0)
    # std_prediction = predictions.std(axis=0)
    
    # return mean_prediction, std_prediction
    
    raise NotImplementedError("Implement MC Dropout inference!")


def analyze_uncertainty():
    """
    Analyze the relationship between uncertainty and prediction errors.
    
    Key questions to answer:
    1. Are wrong predictions more uncertain?
    2. Can we reject uncertain predictions to improve accuracy?
    3. How many forward passes are needed for stable estimates?
    """
    from train_minimal import load_mnist, DropoutMLP, SGD, train_epoch, evaluate
    
    print("=" * 60)
    print("EXERCISE 4: MONTE CARLO DROPOUT FOR UNCERTAINTY")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, y_train = X_train[:5000], y_train[:5000]
    
    # Train a model with dropout
    print("\nTraining model with dropout...")
    model = DropoutMLP(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        dropout_p=0.5
    )
    
    optimizer = SGD(model.get_params(), lr=0.01, momentum=0.9)
    
    for epoch in range(10):
        train_epoch(model, X_train, y_train, optimizer, batch_size=64)
        if (epoch + 1) % 5 == 0:
            train_acc, _ = evaluate(model, X_train, y_train)
            print(f"  Epoch {epoch+1}: Train acc = {train_acc:.3f}")
    
    print("\nRunning MC Dropout for uncertainty estimation...")
    
    # TODO 1: Run MC Dropout on test samples
    n_test = 100  # Use first 100 test samples
    n_forward_passes = 50
    
    # TODO 2: For each test sample, compute:
    # - Mean prediction (average across forward passes)
    # - Uncertainty (std across forward passes)
    
    # TODO 3: Analyze results
    # - Separate correct vs incorrect predictions
    # - Compare their uncertainties
    # - Plot uncertainty distribution for correct vs incorrect
    
    # TODO 4: Rejection experiment
    # - Sort predictions by uncertainty
    # - Reject top 10%, 20%, 30% most uncertain
    # - Compute accuracy on remaining
    # - Does rejecting uncertain predictions improve accuracy?
    
    # TODO 5: Stability analysis
    # - Run with n_samples = 10, 20, 50, 100, 200
    # - How do estimates change?
    # - When do they stabilize?
    
    print("\nExercise not yet implemented!")
    print("Fill in the TODOs above and run again.")


if __name__ == "__main__":
    analyze_uncertainty()
