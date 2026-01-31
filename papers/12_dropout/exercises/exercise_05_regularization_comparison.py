"""
Exercise 5: Regularization Comparison

Goal: Compare dropout with other regularization techniques.

Time: 90 minutes
Difficulty: Hard ⏱️⏱️⏱️⏱️⏱️
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_with_regularization(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    regularization: str,
    epochs: int = 30,
    **kwargs
) -> Dict:
    """
    Train a model with specified regularization technique.
    
    Args:
        regularization: One of 'none', 'l2', 'dropout', 'early_stopping', 'combined'
        
    Returns:
        Dictionary with train_acc, test_acc, history
        
    TODO: Implement each regularization technique
    """
    from train_minimal import DropoutMLP, SGD, train_epoch, evaluate
    
    history = {'train_acc': [], 'test_acc': [], 'train_loss': []}
    
    if regularization == 'none':
        # TODO: Train without any regularization
        # - dropout_p = 1.0 (keep everything)
        # - weight_decay = 0
        # - Train for all epochs
        pass
    
    elif regularization == 'l2':
        # TODO: Train with L2 regularization (weight decay)
        # - dropout_p = 1.0
        # - Use weight_decay from kwargs (default 1e-4)
        pass
    
    elif regularization == 'dropout':
        # TODO: Train with dropout only
        # - dropout_p from kwargs (default 0.5)
        # - weight_decay = 0
        pass
    
    elif regularization == 'early_stopping':
        # TODO: Train with early stopping
        # - Monitor validation loss
        # - Stop if no improvement for 'patience' epochs
        # - patience from kwargs (default 5)
        pass
    
    elif regularization == 'combined':
        # TODO: Train with dropout + L2 + early stopping
        # Best of all worlds!
        pass
    
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    
    # Placeholder results
    return {
        'train_acc': 0.0,
        'test_acc': 0.0,
        'gap': 0.0,
        'history': history
    }


def run_comparison():
    """Compare all regularization techniques on the same dataset."""
    from train_minimal import load_mnist
    
    print("=" * 60)
    print("EXERCISE 5: REGULARIZATION COMPARISON")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use subset for faster experiments
    X_train, y_train = X_train[:5000], y_train[:5000]
    print(f"Training samples: {len(X_train)}")
    
    # Define experiments
    techniques = [
        ('none', {}),
        ('l2', {'weight_decay': 1e-4}),
        ('l2', {'weight_decay': 1e-3}),
        ('dropout', {'dropout_p': 0.3}),
        ('dropout', {'dropout_p': 0.5}),
        ('dropout', {'dropout_p': 0.7}),
        ('early_stopping', {'patience': 5}),
        ('combined', {'dropout_p': 0.5, 'weight_decay': 1e-4, 'patience': 5}),
    ]
    
    results = {}
    
    for name, kwargs in techniques:
        key = f"{name}"
        if kwargs:
            key += f"({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        
        print(f"\n--- Training with {key} ---")
        
        # TODO: Run training
        # results[key] = train_with_regularization(
        #     X_train, y_train, X_test, y_test,
        #     regularization=name,
        #     epochs=30,
        #     **kwargs
        # )
        # print(f"  Train: {results[key]['train_acc']:.3f}")
        # print(f"  Test:  {results[key]['test_acc']:.3f}")
        # print(f"  Gap:   {results[key]['gap']:.3f}")
        
        pass
    
    # TODO: Create comparison summary
    # 1. Table showing all results
    # 2. Bar chart comparing test accuracies
    # 3. Training curves comparison
    # 4. Gap (overfitting) comparison
    
    # TODO: Answer these questions:
    # 1. Which single technique works best?
    # 2. Does combining techniques help?
    # 3. Is there such a thing as too much regularization?
    # 4. When would you choose dropout over L2 or vice versa?
    
    print("\n" + "=" * 60)
    print("SUMMARY (TODO: Implement comparison)")
    print("=" * 60)
    
    print("\nExercise not yet implemented!")
    print("Fill in the TODOs above and run again.")


if __name__ == "__main__":
    run_comparison()
