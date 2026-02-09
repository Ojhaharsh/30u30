"""
Day 20: Relational RNNs - Minimal Training Script
=============================================================================

A complete, runnable demonstration of Relational Memory Cores (RMC).
Run this to see RMC outperform (or match) LSTM on relational tasks!

Usage:
    python train_minimal.py                             # Train both RMC and LSTM
    python train_minimal.py --model rmc --visualize     # Train RMC and show attention
    python train_minimal.py --seq_len 50                # Try harder long sequences

What this script does:
1. Generates "N-th Farthest" data (geometric reasoning task)
2. Trains an RMC and an LSTM baseline side-by-side
3. Visualizes the RMC's "thought process" (attention weights)
4. Saves comparison plots

The key insight: RMC uses *explicit* attention to reason about the past,
while LSTM relies on *implicit* gradient flow (which vanishes).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from implementation import RelationalRNN, StandardLSTM
from visualization import plot_attention_heatmap, plot_comparison

def generate_nth_farthest_batch(batch_size, seq_len, vector_dim, n_index=0):
    """
    Generates a batch for the N-th Farthest task.
    """
    inputs = torch.rand(batch_size, seq_len, vector_dim) * 2 - 1
    targets = []
    
    for i in range(batch_size):
        seq = inputs[i]
        ref_vec = seq[n_index]
        dists = torch.norm(seq - ref_vec, dim=1)
        farthest_idx = torch.argmax(dists)
        target_vec = seq[farthest_idx]
        targets.append(target_vec)
        
    targets = torch.stack(targets)
    return inputs, targets

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model_name, model, args, inputs_targets_gen):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    losses = []
    
    print(f"\nTraining {model_name}...")
    print(f"Parameters: {count_parameters(model):,}")
    
    model.train()
    for step in range(args.epochs):
        inputs, targets = inputs_targets_gen(args.batch_size, args.seq_len, args.dim)
        
        optimizer.zero_grad()
        if model_name == "RMC":
            output = model(inputs) # RMC default return
        else:
            output = model(inputs) # LSTM default return
            
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"[{model_name}] Step {step}: Loss = {loss.item():.6f}")
            
    return losses

def main():
    parser = argparse.ArgumentParser(description="Day 20: RMC vs LSTM Comparison")
    parser.add_argument('--seq_len', type=int, default=15, help='Sequence length')
    parser.add_argument('--dim', type=int, default=16, help='Vector dimension')
    parser.add_argument('--slots', type=int, default=4, help='RMC memory slots')
    parser.add_argument('--mem_size', type=int, default=32, help='RMC memory slot size')
    parser.add_argument('--heads', type=int, default=4, help='RMC attention heads')
    parser.add_argument('--lstm_hidden', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--epochs', type=int, default=2000, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', type=str, choices=['rmc', 'lstm', 'both'], default='both')
    parser.add_argument('--visualize', action='store_true', help='Visualize attention (RMC only)')
    parser.add_argument('--save_plot', action='store_true', help='Save loss comparison plot')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Task Generator
    gen_data = generate_nth_farthest_batch
    
    results = {}
    
    # 1. Train RMC
    if args.model in ['rmc', 'both']:
        rmc = RelationalRNN(args.dim, args.slots, args.mem_size, args.dim, args.heads)
        rmc_losses = train_model("RMC", rmc, args, gen_data)
        results['rmc'] = (rmc, rmc_losses)
        
        # Verify
        rmc.eval()
        inputs, targets = gen_data(100, args.seq_len, args.dim)
        with torch.no_grad():
            out = rmc(inputs)
            val_loss = nn.MSELoss()(out, targets).item()
        print(f"RMC Final Validation Loss: {val_loss:.6f}")

    # 2. Train LSTM
    if args.model in ['lstm', 'both']:
        lstm = StandardLSTM(args.dim, args.lstm_hidden, args.dim)
        lstm_losses = train_model("LSTM", lstm, args, gen_data)
        results['lstm'] = (lstm, lstm_losses)
        
        # Verify
        lstm.eval()
        inputs, targets = gen_data(100, args.seq_len, args.dim)
        with torch.no_grad():
            out = lstm(inputs)
            val_loss = nn.MSELoss()(out, targets).item()
        print(f"LSTM Final Validation Loss: {val_loss:.6f}")

    # 3. Compare & Plot
    if args.model == 'both' and args.save_plot:
        plot_path = os.path.join(args.output_dir, "loss_comparison.png")
        plot_comparison(results['rmc'][1], results['lstm'][1], plot_path)
        
    # 4. Visualize Attention (RMC only)
    if args.visualize and 'rmc' in results:
        print("\nVisualizing Attention...")
        rmc = results['rmc'][0]
        rmc.eval()
        inputs, _ = gen_data(1, args.seq_len, args.dim)
        
        # Forward with attention return
        _, attn_weights = rmc(inputs, return_attention=True)
        # attn_weights: (batch, seq, heads, slots, slots)
        
        # Take first item in batch: (seq, heads, slots, slots)
        attn_sample = attn_weights[0]
        
        plot_path = os.path.join(args.output_dir, "attention_heatmap.png")
        plot_attention_heatmap(attn_sample, plot_path)

if __name__ == "__main__":
    main()
