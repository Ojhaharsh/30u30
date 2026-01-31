"""
Solution 2: Dropout Rate Exploration

Complete solution showing optimal dropout rate discovery.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_minimal import load_mnist, DropoutMLP, SGD, train_epoch, evaluate


def sweep_dropout_rates():
    """Train models with different dropout rates and find optimal."""
    
    print("=" * 60)
    print("SOLUTION 2: DROPOUT RATE EXPLORATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use subset for faster training
    X_train, y_train = X_train[:5000], y_train[:5000]
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Dropout rates (keep probability)
    dropout_rates = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    
    results = {}
    epochs = 25
    
    for keep_prob in dropout_rates:
        print(f"\n{'='*40}")
        print(f"Training with keep_prob = {keep_prob}")
        print(f"{'='*40}")
        
        np.random.seed(42)  # Reproducible results
        
        # Create model
        model = DropoutMLP(
            input_size=784,
            hidden_sizes=[512, 256],
            output_size=10,
            dropout_p=keep_prob,
            input_dropout_p=1.0 if keep_prob == 1.0 else 0.9
        )
        
        # Create optimizer (higher LR for heavy dropout)
        lr = 0.01 if keep_prob >= 0.5 else 0.02
        optimizer = SGD(model.get_params(), lr=lr, momentum=0.9, weight_decay=1e-5)
        
        # Training loop
        train_history = []
        test_history = []
        
        for epoch in range(epochs):
            train_epoch(model, X_train, y_train, optimizer, batch_size=64)
            
            train_acc, _ = evaluate(model, X_train, y_train)
            test_acc, _ = evaluate(model, X_test, y_test)
            
            train_history.append(train_acc)
            test_history.append(test_acc)
            
            if (epoch + 1) % 5 == 0:
                gap = train_acc - test_acc
                print(f"  Epoch {epoch+1:2d}: Train {train_acc:.3f} | Test {test_acc:.3f} | Gap {gap:.3f}")
        
        results[keep_prob] = {
            'train_acc': train_history[-1],
            'test_acc': test_history[-1],
            'gap': train_history[-1] - test_history[-1],
            'train_history': train_history,
            'test_history': test_history,
            'best_test_epoch': np.argmax(test_history) + 1
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Keep Prob':<12} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Best Epoch'}")
    print("-" * 58)
    
    for p in dropout_rates:
        r = results[p]
        print(f"{p:<12.1f} {r['train_acc']:<12.3f} {r['test_acc']:<12.3f} {r['gap']:<10.3f} {r['best_test_epoch']}")
    
    # Find optimal
    best_p = max(results.keys(), key=lambda p: results[p]['test_acc'])
    print(f"\nOptimal keep_prob: {best_p} (Test acc: {results[best_p]['test_acc']:.3f})")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print("\n1. No Dropout (keep_prob=1.0):")
    r = results[1.0]
    print(f"   Train: {r['train_acc']:.3f}, Test: {r['test_acc']:.3f}, Gap: {r['gap']:.3f}")
    print("   Problem: Overfitting! Large train-test gap.")
    
    print("\n2. Optimal Dropout (keep_prob~0.5-0.7):")
    print(f"   Best generalization with moderate dropout.")
    print(f"   The gap between train and test is smaller.")
    
    print("\n3. Heavy Dropout (keep_prob=0.1):")
    r = results[0.1]
    print(f"   Train: {r['train_acc']:.3f}, Test: {r['test_acc']:.3f}, Gap: {r['gap']:.3f}")
    print("   Problem: Underfitting! Too much regularization.")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Test accuracy vs keep_prob
        ax = axes[0]
        probs = sorted(results.keys())
        test_accs = [results[p]['test_acc'] for p in probs]
        train_accs = [results[p]['train_acc'] for p in probs]
        
        ax.plot(probs, train_accs, 'b-o', label='Train', linewidth=2, markersize=8)
        ax.plot(probs, test_accs, 'r-s', label='Test', linewidth=2, markersize=8)
        ax.axvline(best_p, color='g', linestyle='--', label=f'Optimal: {best_p}')
        ax.set_xlabel('Keep Probability', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Dropout Rate', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gap vs keep_prob
        ax = axes[1]
        gaps = [results[p]['gap'] for p in probs]
        ax.bar(range(len(probs)), gaps, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels([f'{p:.1f}' for p in probs])
        ax.set_xlabel('Keep Probability', fontsize=12)
        ax.set_ylabel('Train-Test Gap', fontsize=12)
        ax.set_title('Overfitting (Gap) vs Dropout', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Training curves for select rates
        ax = axes[2]
        for p in [1.0, 0.5, 0.1]:
            ax.plot(results[p]['test_history'], label=f'p={p}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Learning Curves', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dropout_rate_sweep.png', dpi=150)
        plt.show()
        print("\nPlot saved to 'dropout_rate_sweep.png'")
        
    except ImportError:
        print("\nMatplotlib not available for plotting.")
    
    return results


if __name__ == "__main__":
    results = sweep_dropout_rates()
