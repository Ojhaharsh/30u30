"""
Minimal Training Script for Character-Level RNN
Train on your own text and generate samples!

Usage:
    python train_minimal.py --data path/to/text.txt --epochs 100
    python train_minimal.py --generate --checkpoint model.pkl
"""

import numpy as np
import argparse
import pickle
import time
from implementation import CharRNN, prepare_data


def train(model, data, char_to_idx, idx_to_char, epochs=100, print_every=10):
    """
    Train the RNN on data.
    
    Parameters:
    -----------
    model : CharRNN
        The model to train
    data : list of int
        Training data as character indices
    char_to_idx, idx_to_char : dict
        Character mappings
    epochs : int
        Number of passes through the data
    print_every : int
        Print sample every N iterations
    """
    n = 0  # Current position in data
    p = 0  # Iteration counter
    hprev = np.zeros((model.hidden_size, 1))  # Initial hidden state
    
    smooth_loss = -np.log(1.0 / model.vocab_size) * model.seq_length  # Expected loss at start
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Initial expected loss: {smooth_loss:.2f}")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Reset position at start of each epoch
        if n + model.seq_length + 1 >= len(data):
            n = 0
            hprev = np.zeros((model.hidden_size, 1))  # Reset hidden state
        
        # Prepare input and target sequences
        inputs = data[n:n + model.seq_length]
        targets = data[n + 1:n + model.seq_length + 1]
        
        # Sample from model occasionally
        if p % print_every == 0:
            # Generate sample text
            sample_idx = model.sample(hprev, inputs[0], 200, temperature=0.8)
            sample_text = ''.join([idx_to_char[idx] for idx in sample_idx])
            
            elapsed = time.time() - start_time
            print(f"\n--- Iteration {p} (Epoch {epoch}) | Loss: {smooth_loss:.2f} | Time: {elapsed:.1f}s ---")
            print(f"Sample:\n{sample_text}\n")
        
        # Forward pass
        xs, hs, ys, ps = model.forward(inputs, hprev)
        
        # Compute loss
        loss = model.loss(ps, targets)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001  # Exponential moving average
        
        # Backward pass (compute gradients)
        gradients = model.backward(xs, hs, ps, targets)
        
        # Update parameters
        model.update_parameters(gradients)
        
        # Update hidden state (detach from graph to prevent backprop through entire history)
        hprev = hs[len(inputs) - 1]
        
        # Move to next sequence
        n += model.seq_length
        p += 1
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete! Total time: {total_time:.1f}s")
    print(f"Final loss: {smooth_loss:.2f}")
    
    return model, hprev


def generate(model, idx_to_char, seed_text, n_chars=500, temperature=0.8):
    """
    Generate text from trained model.
    
    Parameters:
    -----------
    model : CharRNN
        Trained model
    idx_to_char : dict
        Index to character mapping
    seed_text : str
        Starting text
    n_chars : int
        Number of characters to generate
    temperature : float
        Sampling temperature (0.5 = conservative, 1.5 = creative)
    """
    print(f"\nGenerating {n_chars} characters with temperature {temperature}...")
    print(f"Seed: '{seed_text}'")
    print("=" * 60)
    
    # Convert seed to indices
    char_to_idx = {ch: i for i, ch in idx_to_char.items()}
    seed_indices = [char_to_idx[ch] for ch in seed_text if ch in char_to_idx]
    
    if not seed_indices:
        print("Error: No valid characters in seed text")
        return
    
    # Initialize hidden state
    h = np.zeros((model.hidden_size, 1))
    
    # Warm up hidden state with seed
    for idx in seed_indices[:-1]:
        x = np.zeros((model.vocab_size, 1))
        x[idx] = 1
        h = np.tanh(np.dot(model.Wxh, x) + np.dot(model.Whh, h) + model.bh)
    
    # Generate
    sample_idx = model.sample(h, seed_indices[-1], n_chars, temperature)
    generated = ''.join([idx_to_char[idx] for idx in sample_idx])
    
    print(f"{seed_text}{generated}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train a character-level RNN')
    parser.add_argument('--data', type=str, default='data/tiny_shakespeare.txt',
                        help='Path to text file')
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='Size of hidden layer')
    parser.add_argument('--learning-rate', type=float, default=1e-1,
                        help='Learning rate')
    parser.add_argument('--seq-length', type=int, default=25,
                        help='Sequence length for BPTT')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--checkpoint', type=str, default='model.pkl',
                        help='Path to save/load model')
    parser.add_argument('--generate', action='store_true',
                        help='Generate text from saved model')
    parser.add_argument('--seed', type=str, default='The ',
                        help='Seed text for generation')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--gen-length', type=int, default=500,
                        help='Number of characters to generate')
    
    args = parser.parse_args()
    
    if args.generate:
        # Load model and generate
        print("Loading model from", args.checkpoint)
        with open(args.checkpoint, 'rb') as f:
            model, char_to_idx, idx_to_char = pickle.load(f)
        
        generate(model, idx_to_char, args.seed, args.gen_length, args.temperature)
    
    else:
        # Load data
        print(f"Loading data from {args.data}...")
        try:
            with open(args.data, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: Could not find {args.data}")
            print("Creating sample data...")
            text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.""" * 10
        
        print(f"Data loaded: {len(text)} characters")
        
        # Prepare data
        char_to_idx, idx_to_char, data = prepare_data(text)
        print(f"Vocabulary: {len(char_to_idx)} unique characters")
        print(f"Sample: {text[:100]}...")
        
        # Initialize model
        model = CharRNN(
            vocab_size=len(char_to_idx),
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            seq_length=args.seq_length
        )
        
        # Train
        model, hprev = train(model, data, char_to_idx, idx_to_char, 
                            epochs=args.epochs, print_every=10)
        
        # Save model
        print(f"\nSaving model to {args.checkpoint}...")
        with open(args.checkpoint, 'wb') as f:
            pickle.dump((model, char_to_idx, idx_to_char), f)
        print("Model saved!")
        
        # Generate sample
        print("\n" + "=" * 60)
        print("Final sample generation:")
        generate(model, idx_to_char, args.seed, args.gen_length, args.temperature)


if __name__ == "__main__":
    main()
