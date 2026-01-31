"""
Exercise 2: Dropout Rate Exploration

Goal: Train MNIST classifiers with different dropout rates and find the optimal value.

Time: 45-60 minutes
Difficulty: Medium ⏱️⏱️⏱️
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sweep_dropout_rates():
    """
    Train models with different dropout rates and compare performance.
    
    Your tasks:
    1. Load MNIST dataset
    2. Train models with p ∈ {0.0, 0.3, 0.5, 0.7, 0.9}
    3. Record train accuracy, test accuracy, and the gap
    4. Identify the optimal dropout rate
    5. Plot results
    """
    from train_minimal import load_mnist, DropoutMLP, SGD, train_epoch, evaluate
    
    print("=" * 60)
    print("EXERCISE 2: DROPOUT RATE EXPLORATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use subset for faster training
    X_train, y_train = X_train[:5000], y_train[:5000]
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Dropout rates to test (keep probability = 1 - drop_probability)
    # keep_prob=0.0 means drop everything (broken)
    # keep_prob=1.0 means keep everything (no dropout)
    dropout_rates = [1.0, 0.7, 0.5, 0.3, 0.1]  # Keep probabilities
    
    results = {}
    
    for keep_prob in dropout_rates:
        print(f"\n--- Training with keep_prob = {keep_prob} ---")
        
        # TODO 1: Create model with this dropout rate
        # model = DropoutMLP(
        #     input_size=784,
        #     hidden_sizes=[512, 256],
        #     output_size=10,
        #     dropout_p=keep_prob,
        #     input_dropout_p=1.0 if keep_prob == 1.0 else 0.9
        # )
        
        # TODO 2: Create optimizer
        # optimizer = SGD(model.get_params(), lr=0.01, momentum=0.9)
        
        # TODO 3: Train for 20 epochs
        # for epoch in range(20):
        #     train_epoch(model, X_train, y_train, optimizer, batch_size=64)
        
        # TODO 4: Evaluate
        # train_acc, _ = evaluate(model, X_train, y_train)
        # test_acc, _ = evaluate(model, X_test, y_test)
        
        # TODO 5: Store results
        # results[keep_prob] = {
        #     'train_acc': train_acc,
        #     'test_acc': test_acc,
        #     'gap': train_acc - test_acc
        # }
        
        # print(f"  Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {train_acc - test_acc:.3f}")
        
        pass
    
    # TODO 6: Plot results
    # - X-axis: keep probability
    # - Y-axis: accuracy (train and test)
    # - Highlight the optimal dropout rate
    
    # TODO 7: Answer these questions:
    # 1. Which dropout rate gives the best test accuracy?
    # 2. Which dropout rate has the smallest train-test gap?
    # 3. What happens when dropout is too high (keep_prob=0.1)?
    # 4. What happens with no dropout (keep_prob=1.0)?
    
    print("\nExercise not yet implemented!")
    print("Fill in the TODOs above and run again.")


if __name__ == "__main__":
    sweep_dropout_rates()
