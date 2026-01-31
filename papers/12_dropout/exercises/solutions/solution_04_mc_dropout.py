"""
Solution 4: Monte Carlo Dropout for Uncertainty

Complete solution demonstrating uncertainty estimation with MC Dropout.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_minimal import load_mnist, DropoutMLP, SGD, train_epoch, evaluate


def mc_dropout_inference(model, x, n_samples=100):
    """
    Run multiple forward passes with dropout enabled.
    
    Args:
        model: Model with dropout layers
        x: Input sample(s)
        n_samples: Number of forward passes
        
    Returns:
        mean_prediction: Average prediction
        std_prediction: Standard deviation (uncertainty)
    """
    # Keep dropout ON during inference!
    model.train()
    
    predictions = []
    for _ in range(n_samples):
        pred = model.forward(x)
        predictions.append(pred)
    
    predictions = np.stack(predictions, axis=0)  # (n_samples, batch, classes)
    
    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)
    
    return mean_prediction, std_prediction


def compute_uncertainty_metrics(std_prediction, mean_prediction):
    """Compute various uncertainty metrics."""
    # Total uncertainty (sum of class variances)
    total_variance = std_prediction.sum(axis=-1)
    
    # Predictive entropy
    eps = 1e-10
    probs = np.clip(mean_prediction, eps, 1 - eps)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    
    # Confidence (max probability)
    confidence = mean_prediction.max(axis=-1)
    
    return {
        'variance': total_variance,
        'entropy': entropy,
        'confidence': confidence
    }


def run_mc_dropout_analysis():
    """Full MC Dropout uncertainty analysis."""
    
    print("=" * 60)
    print("SOLUTION 4: MC DROPOUT FOR UNCERTAINTY")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    X_train, y_train = X_train[:5000], y_train[:5000]
    
    # Train model
    print("\nTraining model with dropout...")
    np.random.seed(42)
    
    model = DropoutMLP(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        dropout_p=0.5
    )
    
    optimizer = SGD(model.get_params(), lr=0.01, momentum=0.9)
    
    for epoch in range(15):
        train_epoch(model, X_train, y_train, optimizer, batch_size=64)
        if (epoch + 1) % 5 == 0:
            train_acc, _ = evaluate(model, X_train, y_train)
            test_acc, _ = evaluate(model, X_test, y_test)
            print(f"  Epoch {epoch+1}: Train {train_acc:.3f} | Test {test_acc:.3f}")
    
    # Standard evaluation
    model.eval()
    test_acc, _ = evaluate(model, X_test, y_test)
    print(f"\nStandard test accuracy: {test_acc:.3f}")
    
    # MC Dropout analysis
    print("\n" + "=" * 40)
    print("MC DROPOUT UNCERTAINTY ANALYSIS")
    print("=" * 40)
    
    n_samples_test = 500
    n_forward_passes = 50
    
    print(f"\nAnalyzing {n_samples_test} test samples with {n_forward_passes} forward passes each...")
    
    # Get predictions with uncertainty
    X_subset = X_test[:n_samples_test]
    y_subset = y_test[:n_samples_test]
    
    mean_pred, std_pred = mc_dropout_inference(model, X_subset, n_forward_passes)
    
    # Get predictions and check correctness
    predicted_classes = mean_pred.argmax(axis=1)
    correct_mask = (predicted_classes == y_subset)
    
    print(f"\nMC Dropout accuracy: {correct_mask.mean():.3f}")
    
    # Compute uncertainty metrics
    metrics = compute_uncertainty_metrics(std_pred, mean_pred)
    
    # Analysis 1: Uncertainty for correct vs incorrect
    print("\n--- Uncertainty: Correct vs Incorrect ---")
    
    correct_uncertainty = metrics['variance'][correct_mask]
    incorrect_uncertainty = metrics['variance'][~correct_mask]
    
    print(f"Correct predictions:   mean uncertainty = {correct_uncertainty.mean():.4f}")
    print(f"Incorrect predictions: mean uncertainty = {incorrect_uncertainty.mean():.4f}")
    print(f"Ratio: {incorrect_uncertainty.mean() / correct_uncertainty.mean():.2f}x higher for incorrect!")
    
    # Analysis 2: Rejection experiment
    print("\n--- Rejection Experiment ---")
    print("Rejecting most uncertain predictions:")
    
    rejection_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Sort by uncertainty
    uncertainty_order = np.argsort(metrics['variance'])  # Low to high uncertainty
    
    for reject_rate in rejection_rates:
        n_keep = int(n_samples_test * (1 - reject_rate))
        keep_indices = uncertainty_order[:n_keep]  # Keep least uncertain
        
        kept_accuracy = correct_mask[keep_indices].mean()
        print(f"  Reject {100*reject_rate:4.0f}%: Accuracy on remaining = {kept_accuracy:.3f}")
    
    # Analysis 3: Stability with n_samples
    print("\n--- Stability Analysis ---")
    print("How many forward passes are needed?")
    
    test_sample = X_test[0:1]  # Single sample
    
    n_values = [5, 10, 20, 50, 100, 200]
    means = []
    stds = []
    
    for n in n_values:
        mean, std = mc_dropout_inference(model, test_sample, n)
        means.append(mean[0])
        stds.append(std[0].sum())
    
    print(f"\n{'N Samples':<12} {'Predicted Class':<18} {'Total Std'}")
    print("-" * 45)
    for i, n in enumerate(n_values):
        print(f"{n:<12} {means[i].argmax():<18} {stds[i]:.4f}")
    
    # Visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Uncertainty distributions
        ax = axes[0, 0]
        ax.hist(correct_uncertainty, bins=30, alpha=0.7, label='Correct', color='green')
        ax.hist(incorrect_uncertainty, bins=30, alpha=0.7, label='Incorrect', color='red')
        ax.set_xlabel('Uncertainty (Total Variance)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Uncertainty Distribution: Correct vs Incorrect', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Rejection curve
        ax = axes[0, 1]
        accuracies = []
        for reject_rate in np.linspace(0, 0.7, 20):
            n_keep = int(n_samples_test * (1 - reject_rate))
            keep_indices = uncertainty_order[:n_keep]
            acc = correct_mask[keep_indices].mean()
            accuracies.append(acc)
        
        ax.plot(np.linspace(0, 70, 20), accuracies, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Rejection Rate (%)', fontsize=11)
        ax.set_ylabel('Accuracy on Remaining', fontsize=11)
        ax.set_title('Accuracy vs Rejection Rate', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(test_acc, color='r', linestyle='--', label='No rejection')
        ax.legend()
        
        # Plot 3: Confidence vs accuracy
        ax = axes[1, 0]
        confidence = metrics['confidence']
        
        # Bin by confidence
        bins = np.linspace(confidence.min(), confidence.max(), 11)
        bin_accs = []
        bin_centers = []
        
        for i in range(len(bins) - 1):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(correct_mask[mask].mean())
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        ax.plot(bin_centers, bin_accs, 'go-', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Actual Accuracy', fontsize=11)
        ax.set_title('Calibration: Confidence vs Accuracy', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Show uncertain samples
        ax = axes[1, 1]
        ax.text(0.5, 0.9, 'MC Dropout Key Insights', ha='center', fontsize=14, 
                weight='bold', transform=ax.transAxes)
        
        insights = """
1. Wrong predictions are MORE UNCERTAIN
   (Higher variance across forward passes)

2. Rejecting uncertain samples IMPROVES accuracy
   (Quality vs quantity tradeoff)

3. ~50 forward passes is usually enough
   (Estimates stabilize after that)

4. MC Dropout provides FREE uncertainty
   (Just keep dropout on at test time!)

5. Great for safety-critical applications
   (Reject when model is uncertain)
"""
        ax.text(0.1, 0.75, insights, fontsize=10, 
                transform=ax.transAxes, verticalalignment='top',
                fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('mc_dropout_analysis.png', dpi=150)
        plt.show()
        
        print("\nPlot saved to 'mc_dropout_analysis.png'")
        
    except ImportError:
        print("\nMatplotlib not available for visualization.")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
MC Dropout provides uncertainty estimates by:
1. Keeping dropout ON at inference time
2. Running multiple forward passes
3. Computing mean (prediction) and variance (uncertainty)

Key findings:
- Incorrect predictions have {incorrect_uncertainty.mean() / correct_uncertainty.mean():.1f}x higher uncertainty
- Rejecting 30% most uncertain: accuracy improves from {test_acc:.3f} to ~{correct_mask[uncertainty_order[:int(0.7*n_samples_test)]].mean():.3f}
- Use ~50 forward passes for stable estimates
""")


if __name__ == "__main__":
    run_mc_dropout_analysis()
