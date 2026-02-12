"""
train_minimal.py - GPipe Benchmarking Script

This script demonstrates training a deep network using GPipe's 
pipeline parallelism and micro-batching. It compares memory 
usage and speed across different configurations.

Usage:
    python train_minimal.py --partitions 4 --micro-batches 8 --layers 40
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import os
from implementation import GPipe, get_peak_memory, summarize_results

def parse_args():
    parser = argparse.ArgumentParser(description="GPipe Gold Standard Benchmark")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--micro-batches', type=int, default=4)
    parser.add_argument('--partitions', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--no-checkpoint', action='store_true', help="Disable activation checkpointing")
    return parser.parse_args()

def create_model(hidden_dim, n_layers):
    """Helper to create a deep sequential model."""
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Log Header
    print("\n" + "+" + "-"*58 + "+")
    print(f"|{'GPIPE GOLD STANDARD BENCHMARK':^58}|")
    print("+" + "-"*58 + "+")
    print(f"  Device: {str(device).upper()}")
    print(f"  Model:  {args.layers} Layers, {args.hidden_dim} Hidden Dim")
    print(f"  Setup:  {args.partitions} Partitions, {args.micro_batches} Micro-batches")
    print("-" * 60)

    # 1. Setup Data
    # We use synthetic data for throughput benchmarking
    x = torch.randn(args.batch_size, args.hidden_dim).to(device)
    y = torch.randn(args.batch_size, args.hidden_dim).to(device)

    # 2. Setup Model
    raw_model = create_model(args.hidden_dim, args.layers)
    model = GPipe(
        raw_model, 
        n_partitions=args.partitions, 
        n_microbatches=args.micro_batches, 
        use_checkpoint=not args.no_checkpoint
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. Training Loop
    print("\nStarting Training Step...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        optimizer.zero_grad()
        
        # The main GPipe forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Synchronous backward pass
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch}/{args.epochs} | Loss: {loss.item():.6f} | Time: {elapsed:.4f}s")

    total_time = time.time() - start_time
    peak_mem = get_peak_memory()

    # 4. Results Reporting
    config = {
        'layers': args.layers,
        'partitions': args.partitions,
        'micro_batches': args.micro_batches,
        'use_checkpoint': not args.no_checkpoint
    }
    stats = {
        'total_time': total_time,
        'peak_mem': peak_mem
    }
    
    print("\n" + summarize_results(config, stats))

if __name__ == "__main__":
    main()
