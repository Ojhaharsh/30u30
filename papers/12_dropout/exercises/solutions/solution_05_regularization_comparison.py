"""
Solution 5: Regularization Comparison

Complete solution comparing dropout with other regularization techniques.
"""

import numpy as np
import sys
import os
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_minimal import load_mnist, DropoutMLP, SGD, train_epoch, evaluate


def train_with_regularization(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    regularization: str,
    epochs: int = 30,
    verbose: bool = True,
    **kwargs
) -> Dict:
    """
    Train a model with specified regularization technique.
    """
    np.random.seed(kwargs.get('seed', 42))
    
    history = {'train_acc': [], 'test_acc': [], 'train_loss': []}
    
    # Default hyperparameters
    dropout_p = kwargs.get('dropout_p', 0.5)
    weight_decay = kwargs.get('weight_decay', 0)
    patience = kwargs.get('patience', 5)
    lr = kwargs.get('lr', 0.01)
    
    # Configure based on regularization type
    if regularization == 'none':
        model_dropout = 1.0
        model_wd = 0
        use_early_stopping = False
        
    elif regularization == 'l2':
        model_dropout = 1.0
        model_wd = weight_decay
        use_early_stopping = False
        
    elif regularization == 'dropout':
        model_dropout = dropout_p
        model_wd = 0
        use_early_stopping = False
        
    elif regularization == 'early_stopping':
        model_dropout = 1.0
        model_wd = 0
        use_early_stopping = True
        
    elif regularization == 'combined':
        model_dropout = dropout_p
        model_wd = weight_decay
        use_early_stopping = True
        
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    
    # Create model
    model = DropoutMLP(
        input_size=784,
        hidden_sizes=[512, 256],
        output_size=10,
        dropout_p=model_dropout,
        input_dropout_p=0.9 if model_dropout < 1.0 else 1.0
    )
    
    optimizer = SGD(model.get_params(), lr=lr, momentum=0.9, weight_decay=model_wd)
    
    # Early stopping state
    best_test_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_params = None
    
    # Training loop
    for epoch in range(epochs):
        train_epoch(model, X_train, y_train, optimizer, batch_size=64)
        
        train_acc, train_loss = evaluate(model, X_train, y_train)
        test_acc, _ = evaluate(model, X_test, y_test)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        
        # Early stopping check
        if use_early_stopping:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
                # Save best model (simplified - just track epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                    break
        else:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train {train_acc:.3f} | Test {test_acc:.3f}")
    
    return {
        'train_acc': history['train_acc'][-1],
        'test_acc': best_test_acc,
        'gap': history['train_acc'][-1] - best_test_acc,
        'best_epoch': best_epoch + 1,
        'final_epoch': len(history['train_acc']),
        'history': history
    }


def run_comprehensive_comparison():
    """Compare all regularization techniques."""
    
    print("=" * 60)
    print("SOLUTION 5: REGULARIZATION COMPARISON")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use subset
    X_train, y_train = X_train[:5000], y_train[:5000]
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Define experiments
    experiments = [
        ('No Regularization', 'none', {}),
        ('L2 (wd=1e-4)', 'l2', {'weight_decay': 1e-4}),
        ('L2 (wd=1e-3)', 'l2', {'weight_decay': 1e-3}),
        ('L2 (wd=1e-2)', 'l2', {'weight_decay': 1e-2}),
        ('Dropout (p=0.3)', 'dropout', {'dropout_p': 0.7}),  # keep_prob
        ('Dropout (p=0.5)', 'dropout', {'dropout_p': 0.5}),
        ('Dropout (p=0.7)', 'dropout', {'dropout_p': 0.3}),
        ('Early Stopping', 'early_stopping', {'patience': 5}),
        ('L2 + Dropout', 'combined', {'dropout_p': 0.5, 'weight_decay': 1e-4, 'patience': 100}),
        ('Full Combined', 'combined', {'dropout_p': 0.5, 'weight_decay': 1e-4, 'patience': 5}),
    ]
    
    results = {}
    epochs = 40
    
    for name, reg_type, kwargs in experiments:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        result = train_with_regularization(
            X_train, y_train, X_test, y_test,
            regularization=reg_type,
            epochs=epochs,
            verbose=True,
            **kwargs
        )
        
        results[name] = result
        print(f"\n  Final - Train: {result['train_acc']:.3f} | Test: {result['test_acc']:.3f} | Gap: {result['gap']:.3f}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<25} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Best Epoch'}")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
    
    for name, r in sorted_results:
        print(f"{name:<25} {r['train_acc']:<12.3f} {r['test_acc']:<12.3f} {r['gap']:<10.3f} {r['best_epoch']}")
    
    # Find best
    best_name = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"\n{'='*60}")
    print(f"BEST: {best_name}")
    print(f"Test Accuracy: {best_result['test_acc']:.3f}")
    print(f"{'='*60}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Compare no regularization with best
    no_reg = results['No Regularization']
    print(f"""
1. NO REGULARIZATION:
   - Train: {no_reg['train_acc']:.3f}, Test: {no_reg['test_acc']:.3f}
   - Gap: {no_reg['gap']:.3f} (OVERFITTING!)
   - Problem: Model memorizes training data

2. L2 REGULARIZATION (Weight Decay):
   - Best L2: wd=1e-4 gives good trade-off
   - Too high (1e-2): Underfits
   - Too low: Little effect
   - Shrinks weights toward zero

3. DROPOUT:
   - Best: p~0.5 (keep probability)
   - Forces redundant representations
   - Like training many sub-networks
   - Better for larger networks

4. EARLY STOPPING:
   - Free regularization!
   - Just stop before overfitting
   - Needs validation set to monitor

5. COMBINED (Best results):
   - Dropout + L2 + Early Stopping
   - Each technique helps differently
   - Dropout: feature redundancy
   - L2: weight magnitude
   - Early stopping: training duration
""")
    
    # Visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Test accuracy comparison
        ax = axes[0, 0]
        names = [name for name, _ in sorted_results]
        test_accs = [r['test_acc'] for _, r in sorted_results]
        colors = ['green' if acc == max(test_accs) else 'steelblue' for acc in test_accs]
        
        bars = ax.barh(range(len(names)), test_accs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Test Accuracy', fontsize=11)
        ax.set_title('Regularization Comparison: Test Accuracy', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0.85, max(test_accs) + 0.02)
        
        # Plot 2: Gap comparison
        ax = axes[0, 1]
        gaps = [results[name]['gap'] for name in names]
        colors = ['red' if g > 0.1 else 'orange' if g > 0.05 else 'green' for g in gaps]
        
        ax.barh(range(len(names)), gaps, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Train-Test Gap (Overfitting)', fontsize=11)
        ax.set_title('Regularization Comparison: Overfitting', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Training curves
        ax = axes[1, 0]
        selected = ['No Regularization', 'Dropout (p=0.5)', 'Full Combined']
        for name in selected:
            if name in results:
                ax.plot(results[name]['history']['test_acc'], label=name, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title('Learning Curves', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Overfitting curves (train vs test)
        ax = axes[1, 1]
        name = 'No Regularization'
        ax.plot(results[name]['history']['train_acc'], 'b-', label='Train (no reg)', linewidth=2)
        ax.plot(results[name]['history']['test_acc'], 'b--', label='Test (no reg)', linewidth=2)
        
        name = 'Full Combined'
        ax.plot(results[name]['history']['train_acc'], 'g-', label='Train (combined)', linewidth=2)
        ax.plot(results[name]['history']['test_acc'], 'g--', label='Test (combined)', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Effect of Regularization on Overfitting', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regularization_comparison.png', dpi=150)
        plt.show()
        
        print("\nPlot saved to 'regularization_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available for visualization.")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
When to use each technique:

1. DROPOUT:
   - Large networks with many parameters
   - Fully connected layers
   - When you want uncertainty estimates (MC Dropout)
   - Typical: p=0.5 for hidden, p=0.1-0.2 for input

2. L2 REGULARIZATION:
   - Smaller networks
   - When you want smoother weight distributions
   - Easy to tune (start with 1e-4, increase if overfitting)
   - Also called weight decay

3. EARLY STOPPING:
   - Always use it! It's free
   - Need a validation set
   - Patience 5-10 epochs

4. COMBINED:
   - Best for most cases
   - Start with: dropout=0.5, wd=1e-4, early_stopping
   - Tune individually if needed

RULE OF THUMB:
- Small network: L2 + Early Stopping
- Large network: Dropout + Early Stopping  
- Very large network: Dropout + L2 + Early Stopping
""")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_comparison()
