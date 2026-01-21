"""
Minimal LSTM Training Script
============================

Train a character-level LSTM on any text file.

Usage:
    # Train a model
    python train_minimal.py --data data/input.txt --epochs 200
    
    # Generate from trained model
    python train_minimal.py --generate --checkpoint lstm_model.pkl --length 500
    
    # Custom hyperparameters
    python train_minimal.py --data data/input.txt --hidden-size 128 --seq-length 50 --lr 0.001
"""

import numpy as np
import argparse
from implementation import LSTM


def load_data(filepath):
    """
    Load text data and create character mappings.
    
    Args:
        filepath: Path to text file
        
    Returns:
        data: Raw text string
        chars: Sorted list of unique characters
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        print("Please provide a valid text file with --data")
        exit(1)
    
    # Get unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    print(f"Data loaded successfully!")
    print(f"  - Total characters: {len(data):,}")
    print(f"  - Unique characters: {vocab_size}")
    print(f"  - Character set: {''.join(chars[:50])}" + ("..." if len(chars) > 50 else ""))
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return data, chars, char_to_idx, idx_to_char


def train(args):
    """
    Train the LSTM model.
    
    Args:
        args: Command-line arguments
    """
    # Load data
    data, chars, char_to_idx, idx_to_char = load_data(args.data)
    vocab_size = len(chars)
    
    # Convert text to indices
    data_indices = [char_to_idx[ch] for ch in data]
    data_size = len(data_indices)
    
    # Initialize LSTM
    print(f"\nInitializing LSTM...")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Sequence length: {args.seq_length}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Parameters: ~{4 * args.hidden_size * (args.hidden_size + vocab_size) + vocab_size * args.hidden_size:,}")
    
    lstm = LSTM(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        seq_length=args.seq_length,
        learning_rate=args.lr
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    # Initialize states
    h_prev = np.zeros((args.hidden_size, 1))
    C_prev = np.zeros((args.hidden_size, 1))
    
    # For loss tracking
    smooth_loss = -np.log(1.0/vocab_size) * args.seq_length  # Initial loss estimate
    losses = []
    
    # Data pointer
    p = 0
    
    # Calculate total iterations
    sequences_per_epoch = data_size // args.seq_length
    total_iterations = args.epochs * sequences_per_epoch
    
    iteration = 0
    epoch = 0
    
    while epoch < args.epochs:
        # Reset at end of data or start of training
        if p + args.seq_length + 1 >= data_size or iteration == 0:
            h_prev = np.zeros((args.hidden_size, 1))
            C_prev = np.zeros((args.hidden_size, 1))
            p = 0
            if iteration > 0:
                epoch += 1
                print(f"\n--- Epoch {epoch}/{args.epochs} complete ---")
        
        # Get batch
        inputs = data_indices[p:p+args.seq_length]
        targets = data_indices[p+1:p+args.seq_length+1]
        
        # Skip if we don't have enough data
        if len(inputs) < args.seq_length:
            p = 0
            continue
        
        # Forward pass
        loss, h_prev, C_prev, cache = lstm.forward(inputs, targets, h_prev, C_prev)
        
        # Update smooth loss (exponential moving average)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        losses.append(smooth_loss)
        
        # Backward pass
        grads = lstm.backward(inputs, targets, cache)
        
        # Update weights
        lstm.update_weights(grads)
        
        # Move data pointer
        p += args.seq_length
        iteration += 1
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iter {iteration}/{total_iterations} | Epoch {epoch+1}/{args.epochs} | Loss: {smooth_loss:.4f}")
            
            # Generate sample text
            if iteration % 500 == 0:
                print("\n" + "-" * 60)
                print("Sample generation:")
                sample_h = np.copy(h_prev)
                sample_C = np.copy(C_prev)
                sample_indices = lstm.sample(sample_h, sample_C, inputs[0], 200, temperature=0.8)
                sample_text = ''.join([idx_to_char[i] for i in sample_indices])
                print(sample_text)
                print("-" * 60 + "\n")
    
    # Training complete
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Final loss: {smooth_loss:.4f}")
    print(f"Total iterations: {iteration}")
    
    # Save model
    lstm.save(args.checkpoint)
    print(f"\nModel saved to: {args.checkpoint}")
    
    # Final generation
    print(f"\n{'='*60}")
    print("Final sample generation (temperature=0.5):")
    print(f"{'='*60}\n")
    
    h = np.zeros((args.hidden_size, 1))
    C = np.zeros((args.hidden_size, 1))
    sample_indices = lstm.sample(h, C, char_to_idx[chars[0]], 500, temperature=0.5)
    sample_text = ''.join([idx_to_char[i] for i in sample_indices])
    print(sample_text)
    
    print(f"\n{'='*60}")
    print("Try different temperatures:")
    print("  python train_minimal.py --generate --checkpoint lstm_model.pkl --temperature 0.3")
    print("  python train_minimal.py --generate --checkpoint lstm_model.pkl --temperature 1.0")
    print("  python train_minimal.py --generate --checkpoint lstm_model.pkl --temperature 1.5")


def generate(args):
    """
    Generate text from a trained model.
    
    Args:
        args: Command-line arguments
    """
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    lstm = LSTM.load(args.checkpoint)
    
    # We need the character mappings - they should be saved with model
    # For now, we'll regenerate from data file
    if args.data is None:
        print("Error: --data required for generation (to get character mappings)")
        exit(1)
    
    _, chars, char_to_idx, idx_to_char = load_data(args.data)
    
    print(f"\nGenerating {args.length} characters...")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: '{args.seed}'\n")
    print("=" * 60)
    
    # Initialize states
    h = np.zeros((lstm.hidden_size, 1))
    C = np.zeros((lstm.hidden_size, 1))
    
    # Get seed index
    if args.seed in char_to_idx:
        seed_idx = char_to_idx[args.seed]
    else:
        print(f"Warning: Seed character '{args.seed}' not in vocabulary. Using first character.")
        seed_idx = 0
    
    # Generate
    sample_indices = lstm.sample(h, C, seed_idx, args.length, temperature=args.temperature)
    sample_text = ''.join([idx_to_char[i] for i in sample_indices])
    
    print(sample_text)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train or generate from character-level LSTM')
    
    # Mode
    parser.add_argument('--generate', action='store_true', 
                       help='Generate mode (default: train)')
    
    # Data
    parser.add_argument('--data', type=str, default='data/input.txt',
                       help='Path to training data (default: data/input.txt)')
    
    # Training hyperparameters
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Size of hidden state (default: 128)')
    parser.add_argument('--seq-length', type=int, default=50,
                       help='Sequence length for BPTT (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    
    # Generation parameters
    parser.add_argument('--checkpoint', type=str, default='lstm_model.pkl',
                       help='Path to save/load model (default: lstm_model.pkl)')
    parser.add_argument('--length', type=int, default=500,
                       help='Number of characters to generate (default: 500)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--seed', type=str, default='T',
                       help='Seed character for generation (default: T)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("LSTM Character-Level Language Model")
    print("=" * 60 + "\n")
    
    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
