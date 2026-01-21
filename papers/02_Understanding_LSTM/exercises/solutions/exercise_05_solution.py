"""
Solution to Exercise 5: LSTM vs GRU
===================================

Complete implementation of GRU and detailed comparison with LSTM.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')

from lstm_implementation import LSTM


# ============================================================================
# GRU IMPLEMENTATION
# ============================================================================

class GRU:
    """
    Gated Recurrent Unit (GRU) implementation.
    
    GRU is a simplified variant of LSTM with 2 gates instead of 3:
    - Reset gate (r): controls how much past information to forget
    - Update gate (z): controls how much new information to accept
    
    Key difference from LSTM:
    - No separate cell state (only hidden state)
    - Fewer parameters (faster training)
    - Often performs similarly to LSTM
    
    Equations:
        z_t = σ(W_z [h_{t-1}, x_t])      # Update gate
        r_t = σ(W_r [h_{t-1}, x_t])      # Reset gate
        h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])  # Candidate hidden state
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # New hidden state
    """
    
    def __init__(self, vocab_size, hidden_size, output_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize parameters
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        # Update gate parameters
        self.Wz = np.random.randn(hidden_size, input_size) * scale
        self.bz = np.zeros((hidden_size, 1))
        
        # Reset gate parameters
        self.Wr = np.random.randn(hidden_size, input_size) * scale
        self.br = np.zeros((hidden_size, 1))
        
        # Candidate hidden state parameters
        self.Wh = np.random.randn(hidden_size, input_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        
        # Output layer parameters
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, targets, h_prev):
        """
        Forward pass through GRU.
        
        Args:
            inputs: List of input indices
            targets: List of target indices
            h_prev: Previous hidden state
            
        Returns:
            loss: Cross-entropy loss
            h: Final hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        zs, rs, h_tildes = {}, {}, {}  # Store gates and candidates
        
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            # One-hot encode input
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            # Concatenate previous hidden state and input
            combined = np.vstack([hs[t-1], xs[t]])
            
            # Update gate: z_t = σ(W_z [h_{t-1}, x_t])
            zs[t] = sigmoid(np.dot(self.Wz, combined) + self.bz)
            
            # Reset gate: r_t = σ(W_r [h_{t-1}, x_t])
            rs[t] = sigmoid(np.dot(self.Wr, combined) + self.br)
            
            # Candidate hidden state: h̃_t = tanh(W [r_t ⊙ h_{t-1}, x_t])
            reset_combined = np.vstack([rs[t] * hs[t-1], xs[t]])
            h_tildes[t] = np.tanh(np.dot(self.Wh, reset_combined) + self.bh)
            
            # New hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
            hs[t] = (1 - zs[t]) * hs[t-1] + zs[t] * h_tildes[t]
            
            # Output layer
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            
            # Softmax
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
            # Cross-entropy loss
            loss += -np.log(ps[t][target, 0])
        
        # Store intermediate values for backward pass
        self.cache = (xs, hs, zs, rs, h_tildes, ys, ps, targets)
        
        return loss, hs[len(inputs)-1]
    
    def backward(self):
        """
        Backward pass through GRU (BPTT).
        
        Returns dictionary of gradients.
        """
        xs, hs, zs, rs, h_tildes, ys, ps, targets = self.cache
        
        # Initialize gradients
        dWz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.bz)
        dWr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.br)
        dWh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.bh)
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
        
        dh_next = np.zeros_like(hs[0])
        
        # Backward pass through time
        for t in reversed(range(len(targets))):
            # Output layer gradient
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            
            # Hidden state gradient
            dh = np.dot(self.Why.T, dy) + dh_next
            
            # Candidate hidden state gradient
            dh_tilde = dh * zs[t]
            dh_tilde_raw = dh_tilde * (1 - h_tildes[t]**2)  # tanh derivative
            
            combined_reset = np.vstack([rs[t] * hs[t-1], xs[t]])
            dWh += np.dot(dh_tilde_raw, combined_reset.T)
            dbh += dh_tilde_raw
            
            # Reset gate gradient
            dr = np.dot(self.Wh.T, dh_tilde_raw)[:self.hidden_size] * hs[t-1]
            dr_raw = dr * rs[t] * (1 - rs[t])  # sigmoid derivative
            
            combined = np.vstack([hs[t-1], xs[t]])
            dWr += np.dot(dr_raw, combined.T)
            dbr += dr_raw
            
            # Update gate gradient
            dz = dh * (h_tildes[t] - hs[t-1])
            dz_raw = dz * zs[t] * (1 - zs[t])  # sigmoid derivative
            
            dWz += np.dot(dz_raw, combined.T)
            dbz += dz_raw
            
            # Gradient for next timestep
            dh_next = dh * (1 - zs[t])
            dh_next += np.dot(self.Wz.T, dz_raw)[:self.hidden_size]
            dh_next += np.dot(self.Wr.T, dr_raw)[:self.hidden_size]
            dh_next += np.dot(self.Wh.T, dh_tilde_raw)[:self.hidden_size] * rs[t]
        
        # Clip gradients
        for dparam in [dWz, dWr, dWh, dWhy, dbz, dbr, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return {
            'dWz': dWz, 'dbz': dbz,
            'dWr': dWr, 'dbr': dbr,
            'dWh': dWh, 'dbh': dbh,
            'dWhy': dWhy, 'dby': dby
        }
    
    def parameter_count(self):
        """Count total number of parameters."""
        count = 0
        for param in [self.Wz, self.bz, self.Wr, self.br, 
                     self.Wh, self.bh, self.Why, self.by]:
            count += param.size
        return count


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ============================================================================
# COMPARISON STUDY
# ============================================================================

def compare_lstm_gru(data, char_to_idx, hidden_size=128, seq_length=25,
                    num_iterations=1000, learning_rate=0.01):
    """
    Train both LSTM and GRU on same data and compare.
    
    Metrics to compare:
    1. Training loss convergence
    2. Training speed (time per iteration)
    3. Parameter count
    4. Memory usage
    5. Generation quality
    """
    print("="*70)
    print("LSTM vs GRU COMPARISON")
    print("="*70)
    
    vocab_size = len(char_to_idx)
    
    # Initialize models
    lstm = LSTM(vocab_size, hidden_size, vocab_size)
    gru = GRU(vocab_size, hidden_size, vocab_size)
    
    # Compare parameter counts
    print("\n[1] PARAMETER COUNT")
    print("-" * 70)
    print(f"LSTM parameters: {lstm.parameter_count():,}")
    print(f"GRU parameters:  {gru.parameter_count():,}")
    print(f"Difference:      {lstm.parameter_count() - gru.parameter_count():,}")
    print(f"GRU has {(1 - gru.parameter_count()/lstm.parameter_count())*100:.1f}% fewer parameters")
    
    # Training comparison
    print("\n[2] TRAINING COMPARISON")
    print("-" * 70)
    
    results = {
        'LSTM': {'losses': [], 'times': []},
        'GRU': {'losses': [], 'times': []}
    }
    
    # Train LSTM
    print("\nTraining LSTM...")
    lstm_start = time.time()
    
    for iteration in range(num_iterations):
        # Sample batch
        p = np.random.randint(0, len(data) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
        
        # Forward pass
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))
        loss, h, c = lstm.forward(inputs, targets, h, c)
        
        # Backward pass
        grads = lstm.backward()
        
        # Update (simplified - should use proper optimizer)
        for param_name in ['Wf', 'Wi', 'Wc', 'Wo', 'Why', 
                          'bf', 'bi', 'bc', 'bo', 'by']:
            param = getattr(lstm, param_name)
            grad = grads['d' + param_name]
            param -= learning_rate * grad
        
        results['LSTM']['losses'].append(loss)
        
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1}/{num_iterations}, Loss: {loss:.4f}")
    
    lstm_time = time.time() - lstm_start
    results['LSTM']['times'].append(lstm_time)
    print(f"LSTM training time: {lstm_time:.2f}s")
    
    # Train GRU
    print("\nTraining GRU...")
    gru_start = time.time()
    
    for iteration in range(num_iterations):
        # Sample batch
        p = np.random.randint(0, len(data) - seq_length - 1)
        inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]
        
        # Forward pass
        h = np.zeros((hidden_size, 1))
        loss, h = gru.forward(inputs, targets, h)
        
        # Backward pass
        grads = gru.backward()
        
        # Update
        for param_name in ['Wz', 'Wr', 'Wh', 'Why', 'bz', 'br', 'bh', 'by']:
            param = getattr(gru, param_name)
            grad = grads['d' + param_name]
            param -= learning_rate * grad
        
        results['GRU']['losses'].append(loss)
        
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1}/{num_iterations}, Loss: {loss:.4f}")
    
    gru_time = time.time() - gru_start
    results['GRU']['times'].append(gru_time)
    print(f"GRU training time: {gru_time:.2f}s")
    
    return results, lstm, gru


def analyze_comparison(results):
    """Analyze and report comparison results."""
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)
    
    lstm_losses = results['LSTM']['losses']
    gru_losses = results['GRU']['losses']
    
    print("\n[1] FINAL PERFORMANCE")
    print("-" * 70)
    print(f"LSTM final loss: {np.mean(lstm_losses[-100:]):.4f}")
    print(f"GRU final loss:  {np.mean(gru_losses[-100:]):.4f}")
    
    diff = np.mean(lstm_losses[-100:]) - np.mean(gru_losses[-100:])
    if abs(diff) < 0.1:
        print("→ Similar performance!")
    elif diff > 0:
        print(f"→ GRU performs better by {diff:.4f}")
    else:
        print(f"→ LSTM performs better by {-diff:.4f}")
    
    print("\n[2] TRAINING SPEED")
    print("-" * 70)
    lstm_time = results['LSTM']['times'][0]
    gru_time = results['GRU']['times'][0]
    
    print(f"LSTM: {lstm_time:.2f}s")
    print(f"GRU:  {gru_time:.2f}s")
    print(f"Speedup: {lstm_time/gru_time:.2f}x")
    
    if gru_time < lstm_time:
        print("→ GRU is faster!")
    
    print("\n[3] CONVERGENCE SPEED")
    print("-" * 70)
    
    # Find iteration where loss < threshold
    threshold = 1.5
    lstm_converge = next((i for i, loss in enumerate(lstm_losses) if loss < threshold), None)
    gru_converge = next((i for i, loss in enumerate(gru_losses) if loss < threshold), None)
    
    if lstm_converge and gru_converge:
        print(f"LSTM converged at iteration: {lstm_converge}")
        print(f"GRU converged at iteration:  {gru_converge}")
        if gru_converge < lstm_converge:
            print("→ GRU converges faster!")
        else:
            print("→ LSTM converges faster!")
    
    print("\n[4] SUMMARY")
    print("-" * 70)
    print("✓ GRU typically:")
    print("  - Fewer parameters (25-30% less)")
    print("  - Faster training (10-20% faster)")
    print("  - Similar final performance")
    print("  - Simpler architecture (easier to understand)")
    
    print("\n✓ LSTM typically:")
    print("  - More expressive (separate cell state)")
    print("  - Better for very long sequences")
    print("  - More control (3 gates vs 2)")
    print("  - Established track record")


def visualize_comparison(results, save_path='lstm_vs_gru.png'):
    """Create comprehensive comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    lstm_losses = results['LSTM']['losses']
    gru_losses = results['GRU']['losses']
    
    # Panel 1: Loss curves
    ax = axes[0, 0]
    window = 50
    lstm_smoothed = np.convolve(lstm_losses, np.ones(window)/window, mode='valid')
    gru_smoothed = np.convolve(gru_losses, np.ones(window)/window, mode='valid')
    
    ax.plot(lstm_smoothed, label='LSTM', color='blue', linewidth=2)
    ax.plot(gru_smoothed, label='GRU', color='red', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (smoothed)', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Direct comparison (scatter)
    ax = axes[0, 1]
    ax.scatter(lstm_losses[::10], gru_losses[::10], alpha=0.5, s=20)
    ax.plot([min(lstm_losses), max(lstm_losses)], 
           [min(lstm_losses), max(lstm_losses)],
           'k--', label='Equal performance', linewidth=2)
    ax.set_xlabel('LSTM Loss', fontsize=12)
    ax.set_ylabel('GRU Loss', fontsize=12)
    ax.set_title('Loss Correlation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Parameter count comparison
    ax = axes[1, 0]
    # Placeholder - would need actual model instances
    categories = ['Update/Forget', 'Input/Reset', 'Cell/Hidden', 'Output']
    lstm_params = [25, 25, 25, 25]  # Simplified
    gru_params = [20, 20, 20, 20]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, lstm_params, width, label='LSTM', color='blue', alpha=0.7)
    ax.bar(x + width/2, gru_params, width, label='GRU', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Parameters (%)', fontsize=12)
    ax.set_title('Parameter Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = [
        ['Metric', 'LSTM', 'GRU', 'Winner'],
        ['', '', '', ''],
        ['Parameters', 'More', 'Less', 'GRU'],
        ['Training Speed', 'Slower', 'Faster', 'GRU'],
        ['Performance', 'Similar', 'Similar', 'Tie'],
        ['Complexity', 'Higher', 'Lower', 'GRU'],
        ['Long Sequences', 'Better', 'Good', 'LSTM'],
        ['Memory Usage', 'Higher', 'Lower', 'GRU']
    ]
    
    table = ax.table(cellText=summary, cellLoc='left', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("LSTM vs GRU: COMPREHENSIVE COMPARISON")
    print("="*70)
    
    print("\nThis solution provides:")
    print("  1. Full GRU implementation from scratch")
    print("  2. Side-by-side training comparison")
    print("  3. Parameter count analysis")
    print("  4. Training speed benchmarks")
    print("  5. Performance comparison")
    print("  6. Comprehensive visualizations")
    
    print("\n" + "-"*70)
    print("KEY DIFFERENCES")
    print("-"*70)
    
    print("\nLSTM:")
    print("  • 3 gates (forget, input, output)")
    print("  • Separate cell state and hidden state")
    print("  • More parameters")
    print("  • Explicit memory management")
    
    print("\nGRU:")
    print("  • 2 gates (update, reset)")
    print("  • Single hidden state (no cell state)")
    print("  • Fewer parameters (25-30% less)")
    print("  • Implicit memory through gating")
    
    print("\n" + "-"*70)
    print("WHEN TO USE EACH")
    print("-"*70)
    
    print("\n✓ Use LSTM when:")
    print("  - Very long sequences (>100 steps)")
    print("  - Need precise memory control")
    print("  - Have lots of training data")
    print("  - Computational cost not critical")
    
    print("\n✓ Use GRU when:")
    print("  - Medium-length sequences")
    print("  - Want faster training")
    print("  - Limited training data")
    print("  - Need simpler architecture")
    print("  - Memory/compute constrained")
    
    print("\n" + "="*70)
    print("In practice: Try both and see what works!")
    print("="*70)
