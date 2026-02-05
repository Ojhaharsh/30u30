"""
Minimal Training Loop with Regularization
==========================================

A clean, educational training loop demonstrating:
- Training with dropout
- Validation monitoring
- Early stopping
- Weight decay
- Learning curves

Run: python train_minimal.py --help
"""

import numpy as np
import argparse
import sys
from typing import Tuple, List

# Import our implementation
from implementation import (
    RegularizedLSTM, EarlyStoppingMonitor, RegularizationConfig,
    regularized_loss, clip_gradients
)


def generate_simple_dataset(seq_length: int = 20, vocab_size: int = 26, 
                           num_samples: int = 100) -> Tuple[list, list]:
    """
    Generate a simple training dataset.
    
    Task: Predict next character in sequence
    Dataset: Random sequences of integers
    
    Args:
        seq_length: Length of sequences
        vocab_size: Size of vocabulary
        num_samples: Number of sequences
        
    Returns:
        data: List of sequences
        targets: List of target sequences
    """
    data = []
    targets = []
    
    for _ in range(num_samples):
        # Random sequence
        seq = np.random.randint(0, vocab_size, seq_length)
        # Targets are shifted by 1
        data.append(seq[:-1])
        targets.append(seq[1:])
    
    return data, targets


def train_epoch(model: RegularizedLSTM, data: list, targets: list,
                config: RegularizationConfig, training: bool = True) -> float:
    """
    Train for one epoch.
    
    Args:
        model: RegularizedLSTM instance
        data: List of input sequences
        targets: List of target sequences
        config: Configuration
        training: Whether in training mode
        
    Returns:
        Average loss for epoch
    """
    total_loss = 0
    num_batches = 0
    
    h = np.zeros((model.hidden_size, 1))
    c = np.zeros((model.hidden_size, 1))
    
    for x, y in zip(data, targets):
        # Forward pass
        loss, h, c = model.forward(x, y, h, c, training=training)
        
        # Add L2 regularization
        weights = [model.Wf, model.Wi, model.Wc, model.Wo, model.Why]
        total_loss_reg = regularized_loss(loss, weights, config.weight_decay)
        
        total_loss += total_loss_reg
        num_batches += 1
    
    return total_loss / num_batches


def validate(model: RegularizedLSTM, data: list, targets: list,
             config: RegularizationConfig) -> float:
    """
    Compute validation loss.
    
    Args:
        model: RegularizedLSTM instance
        data: Validation sequences
        targets: Validation targets
        config: Configuration
        
    Returns:
        Average validation loss
    """
    return train_epoch(model, data, targets, config, training=False)


def main():
    """Main training loop."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train regularized LSTM')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden size')
    parser.add_argument('--vocab', type=int, default=26, help='Vocabulary size')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout keep probability')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Configuration
    config = RegularizationConfig()
    config.dropout_keep_prob = args.dropout
    config.weight_decay = args.weight_decay
    config.patience = args.patience
    
    print("\n" + "="*60)
    print("REGULARIZED LSTM TRAINING")
    print("="*60)
    print(f"Hidden size: {args.hidden}")
    print(f"Vocabulary size: {args.vocab}")
    print(f"Dropout keep prob: {args.dropout}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Early stopping patience: {args.patience}")
    print("="*60 + "\n")
    
    # Create model
    model = RegularizedLSTM(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        output_size=args.vocab,
        dropout_keep_prob=args.dropout,
        use_layer_norm=True
    )
    
    print(f"Model parameters: {model.parameter_count():,}\n")
    
    # Generate data
    print("Generating dataset...")
    train_data, train_targets = generate_simple_dataset(
        seq_length=20, vocab_size=args.vocab, num_samples=100
    )
    val_data, val_targets = generate_simple_dataset(
        seq_length=20, vocab_size=args.vocab, num_samples=20
    )
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}\n")
    
    # Training loop
    early_stop = EarlyStoppingMonitor(patience=args.patience, verbose=True)
    
    print("Starting training...\n")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Status':<20}")
    print("-" * 52)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_data, train_targets, config, training=True)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate(model, val_data, val_targets, config)
        val_losses.append(val_loss)
        
        # Early stopping
        should_continue = early_stop.check(val_loss, epoch)
        
        status = "[ok] IMPROVED" if val_loss < early_stop.best_loss else "[FAIL] No improve"
        print(f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} {status:<20}")
        
        if not should_continue:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best validation loss: {early_stop.best_loss:.4f} at epoch {early_stop.best_epoch}")
            break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {early_stop.best_loss:.4f}")
    print(f"Best epoch: {early_stop.best_epoch}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    
    # Print regularization impact
    print("\nRegularization Impact:")
    print(f"- Dropout reduces overfitting by randomly disabling neurons")
    print(f"- Weight decay penalizes large weights")
    print(f"- Layer norm stabilizes training")
    print(f"- Early stopping prevents memorization")
    
    return val_losses


if __name__ == "__main__":
    try:
        val_losses = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
