"""
Solution to Exercise 4: Loss Visualization & Diagnostics
========================================================

Complete implementation of training diagnostics and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from implementation import CharRNN


def train_with_logging(rnn, data, char_to_idx, epochs=100, seq_length=25, learning_rate=0.01):
    """Train RNN and log detailed metrics."""
    losses = []
    epoch_times = []
    gradient_norms = []
    sample_perplexities = []
    
    import time
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = 0
        grad_norm_sum = 0
        
        # Reset hidden state at start of epoch
        h = np.zeros(rnn.hidden_size)
        
        # Process data in sequences
        for i in range(0, len(data) - seq_length, seq_length):
            # Get sequence
            inputs = [char_to_idx[ch] for ch in data[i:i+seq_length]]
            targets = [char_to_idx[ch] for ch in data[i+1:i+seq_length+1]]
            
            # Forward pass
            loss, h = rnn.forward(inputs, targets, h)
            total_loss += loss
            
            # Backward pass
            grads = rnn.backward()
            
            # Update parameters
            for param_name in ['Wxh', 'Whh', 'Why', 'bh', 'by']:
                param = getattr(rnn, param_name)
                grad = grads['d' + param_name]
                param -= learning_rate * grad
            
            # Track gradient norms
            total_grad_norm = sum(np.sum(grad**2) for grad in grads.values())
            grad_norm_sum += np.sqrt(total_grad_norm)
            
            num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        epoch_times.append(time.time() - start_time)
        gradient_norms.append(grad_norm_sum / num_batches)
        sample_perplexities.append(np.exp(avg_loss))
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Perplexity: {np.exp(avg_loss):.2f}")
            print(f"  Grad norm: {gradient_norms[-1]:.4f}")
            print(f"  Time: {epoch_times[-1]:.2f}s")
    
    return {
        'losses': losses,
        'epoch_times': epoch_times,
        'gradient_norms': gradient_norms,
        'perplexities': sample_perplexities
    }


def plot_loss_curve(losses, smoothing_window=5, title="Training Loss"):
    """Plot loss with optional smoothing."""
    plt.figure(figsize=(12, 6))
    
    # Plot raw loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, alpha=0.3, label='Raw Loss', color='blue')
    
    # Plot smoothed loss
    if smoothing_window > 1:
        smoothed = np.convolve(losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
        plt.plot(range(smoothing_window-1, len(losses)), smoothed, 
                label=f'Smoothed (window={smoothing_window})', color='red', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot log-scale view
    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.6, color='blue')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(title + ' (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()


def identify_training_patterns(losses):
    """Identify common training patterns in loss curve."""
    patterns = []
    
    # 1. Check for smooth decrease
    early_loss = np.mean(losses[:10])
    late_loss = np.mean(losses[-10:])
    improvement = (early_loss - late_loss) / early_loss
    
    if improvement > 0.5:
        patterns.append({
            'type': 'smooth_decrease',
            'description': 'Loss is decreasing smoothly - healthy training',
            'severity': 'good'
        })
    
    # 2. Check for plateau
    if len(losses) > 50:
        last_50 = losses[-50:]
        std_dev = np.std(last_50)
        mean_loss = np.mean(last_50)
        
        if std_dev < 0.01 * mean_loss:
            patterns.append({
                'type': 'plateau',
                'description': f'Loss plateaued at {mean_loss:.4f}',
                'severity': 'warning'
            })
    
    # 3. Check for spikes
    diffs = np.diff(losses)
    large_increases = np.where(diffs > 0.5)[0]
    
    if len(large_increases) > 0:
        patterns.append({
            'type': 'spikes',
            'description': f'Found {len(large_increases)} loss spikes',
            'severity': 'bad',
            'epochs': large_increases.tolist()
        })
    
    # 4. Check for oscillation
    if len(losses) > 20:
        last_20 = losses[-20:]
        # Check if alternating up/down
        sign_changes = np.sum(np.diff(np.sign(np.diff(last_20))) != 0)
        
        if sign_changes > 12:  # More than 60% are sign changes
            patterns.append({
                'type': 'oscillation',
                'description': 'Loss is oscillating - learning rate might be too high',
                'severity': 'warning'
            })
    
    return patterns


def diagnose_problems(training_log):
    """Diagnose common training problems."""
    losses = training_log['losses']
    grad_norms = training_log['gradient_norms']
    
    diagnoses = []
    
    # 1. Gradient explosion
    if max(grad_norms) > 10.0:
        diagnoses.append({
            'problem': 'gradient_explosion',
            'description': f'Gradient norm reached {max(grad_norms):.2f}',
            'fix': 'Add gradient clipping (clip at 5.0)'
        })
    
    # 2. Learning rate too high
    if len(losses) > 10:
        early_volatility = np.std(losses[:10])
        if early_volatility > np.mean(losses[:10]):
            diagnoses.append({
                'problem': 'high_learning_rate',
                'description': 'Loss is very volatile early on',
                'fix': 'Reduce learning rate by 10x'
            })
    
    # 3. Learning rate too low
    if len(losses) > 50:
        recent_change = abs(losses[-1] - losses[-20]) / losses[-20]
        if recent_change < 0.01:
            diagnoses.append({
                'problem': 'low_learning_rate',
                'description': 'Loss barely changing',
                'fix': 'Increase learning rate by 2-5x'
            })
    
    # 4. Insufficient capacity
    final_loss = losses[-1]
    if final_loss > 1.5:
        diagnoses.append({
            'problem': 'insufficient_capacity',
            'description': f'Final loss {final_loss:.2f} is high',
            'fix': 'Increase hidden_size (try 128 or 256)'
        })
    
    # 5. Possible overfitting (if loss too low)
    if final_loss < 0.5:
        diagnoses.append({
            'problem': 'possible_overfitting',
            'description': f'Final loss {final_loss:.2f} is very low',
            'fix': 'Use validation set; add regularization'
        })
    
    return diagnoses


def suggest_fixes(diagnoses):
    """Suggest fixes based on diagnoses."""
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    
    if not diagnoses:
        print("No obvious problems detected!")
        print("   Training appears healthy.")
        return
    
    print(f"\nFound {len(diagnoses)} potential issues:\n")
    
    for i, diag in enumerate(diagnoses, 1):
        print(f"{i}. {diag['problem'].upper()}")
        print(f"   Issue: {diag['description']}")
        print(f"   Fix: {diag['fix']}\n")


def create_diagnostic_dashboard(training_log):
    """Create comprehensive 4-panel diagnostic dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    losses = training_log['losses']
    grad_norms = training_log['gradient_norms']
    perplexities = training_log['perplexities']
    
    # Panel 1: Loss curve
    axes[0, 0].plot(losses, color='blue', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: Gradient norms
    axes[0, 1].plot(grad_norms, color='red', linewidth=2)
    axes[0, 1].axhline(y=5.0, color='orange', linestyle='--', label='Clip threshold')
    axes[0, 1].set_title('Gradient Norms', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Perplexity
    axes[1, 0].plot(perplexities, color='green', linewidth=2)
    axes[1, 0].set_title('Perplexity (exp(loss))', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Loss distribution (histogram)
    axes[1, 1].hist(losses, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(losses), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(losses):.3f}')
    axes[1, 1].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Diagnostic Dashboard', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*60)
    print("TRAINING DIAGNOSTICS TOOLKIT")
    print("="*60)
    
    print("\nThis solution provides tools for:")
    print("  1. Detailed training logging")
    print("  2. Loss curve visualization with smoothing")
    print("  3. Pattern identification (plateau, spikes, etc.)")
    print("  4. Problem diagnosis (gradient explosion, etc.)")
    print("  5. Automated fix suggestions")
    print("  6. Comprehensive diagnostic dashboard")
    
    print("\n" + "-"*60)
    print("COMMON TRAINING PROBLEMS & FIXES")
    print("-"*60)
    
    problems = {
        'Gradient Explosion': 'Add gradient clipping (clip_norm=5.0)',
        'Loss Spikes': 'Reduce learning rate by 10x',
        'Plateau Too Early': 'Increase hidden_size or learning rate',
        'Oscillating Loss': 'Reduce learning rate',
        'Very Slow Learning': 'Increase learning rate',
        'High Final Loss': 'Increase model capacity',
        'Extremely Low Loss': 'Check for overfitting'
    }
    
    for problem, fix in problems.items():
        print(f"\n• {problem}")
        print(f"  → {fix}")
    
    print("\n" + "="*60)
    print("Use these tools to debug your RNN training!")
    print("="*60)
