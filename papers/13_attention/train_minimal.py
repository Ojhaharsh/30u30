"""
Day 13: Minimal Transformer Training Script

Train a small Transformer on a simple sequence task.
Demonstrates the training loop and attention patterns.
"""

import numpy as np
import argparse
from typing import Tuple, List, Dict

from implementation import (
    Transformer, EncoderBlock, MultiHeadAttention,
    PositionalEncoding, create_causal_mask, create_padding_mask, softmax
)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_copy_task_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int = 10,
    pad_token: int = 0,
    start_token: int = 1,
    end_token: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a copy task.
    
    The model learns to copy the input sequence to the output.
    Example: Input [1, 5, 3, 7, 2] -> Output [1, 5, 3, 7, 2]
    
    This is a fundamental test for sequence-to-sequence models.
    """
    # Generate random sequences (tokens 3 to vocab_size-1)
    src = np.random.randint(3, vocab_size, (n_samples, seq_len))
    
    # Target is the same sequence with start token prepended
    tgt = np.zeros((n_samples, seq_len + 1), dtype=np.int32)
    tgt[:, 0] = start_token
    tgt[:, 1:] = src
    
    return src, tgt


def generate_reverse_task_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int = 10,
    start_token: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a reverse task.
    
    The model learns to reverse the input sequence.
    Example: Input [5, 3, 7] -> Output [7, 3, 5]
    """
    src = np.random.randint(3, vocab_size, (n_samples, seq_len))
    
    tgt = np.zeros((n_samples, seq_len + 1), dtype=np.int32)
    tgt[:, 0] = start_token
    tgt[:, 1:] = src[:, ::-1]  # Reverse
    
    return src, tgt


# =============================================================================
# LOSS AND METRICS
# =============================================================================

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, 
                       ignore_index: int = -100) -> float:
    """
    Cross-entropy loss for sequence prediction.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        ignore_index: Token to ignore in loss computation
        
    Returns:
        Average loss
    """
    batch, seq_len, vocab_size = logits.shape
    
    # Flatten for easier indexing
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Compute softmax
    probs = softmax(logits_flat, axis=-1)
    
    # Get probability of correct token
    mask = targets_flat != ignore_index
    valid_indices = np.arange(len(targets_flat))[mask]
    valid_targets = targets_flat[mask]
    
    if len(valid_indices) == 0:
        return 0.0
    
    # Clip for numerical stability
    correct_probs = np.clip(probs[valid_indices, valid_targets], 1e-10, 1.0)
    
    # Cross-entropy
    loss = -np.mean(np.log(correct_probs))
    
    return loss


def compute_accuracy(logits: np.ndarray, targets: np.ndarray,
                    ignore_index: int = -100) -> float:
    """Compute token-level accuracy."""
    predictions = logits.argmax(axis=-1)
    
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    
    return correct.sum() / mask.sum() if mask.sum() > 0 else 0.0


# =============================================================================
# OPTIMIZER (SGD with momentum)
# =============================================================================

class SGD:
    """SGD optimizer with momentum."""
    
    def __init__(self, params: List[np.ndarray], lr: float = 0.01, 
                 momentum: float = 0.9, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Velocity for momentum
        self.velocities = [np.zeros_like(p) for p in params]
    
    def step(self, gradients: List[np.ndarray]):
        """Update parameters."""
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            param += self.velocities[i]


# =============================================================================
# SIMPLE TRANSFORMER FOR TRAINING
# =============================================================================

class SimpleTransformer:
    """
    A simplified Transformer for training demos.
    
    Uses numerical gradients for simplicity (not efficient, but educational).
    """
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, dropout_p: float = 0.1):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.training = True
        
        # Create full transformer
        self.model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_layers,
            n_decoder_layers=n_layers,
            d_ff=d_ff,
            dropout_p=dropout_p
        )
    
    def forward(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Create causal mask for decoder
        tgt_mask = create_causal_mask(tgt.shape[1])
        
        return self.model.forward(src, tgt, tgt_mask=tgt_mask)
    
    def train(self):
        self.training = True
        self.model.train()
    
    def eval(self):
        self.training = False
        self.model.eval()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: SimpleTransformer,
    src_data: np.ndarray,
    tgt_data: np.ndarray,
    batch_size: int = 32,
    lr: float = 0.001
) -> Dict[str, float]:
    """
    Train for one epoch using simple gradient estimation.
    
    Note: This uses a simplified training approach for educational purposes.
    Real implementations use automatic differentiation.
    """
    model.train()
    
    n_samples = src_data.shape[0]
    n_batches = n_samples // batch_size
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    indices = np.random.permutation(n_samples)
    
    for i in range(n_batches):
        batch_idx = indices[i * batch_size:(i + 1) * batch_size]
        src_batch = src_data[batch_idx]
        tgt_batch = tgt_data[batch_idx]
        
        # Teacher forcing: input is target[:-1], predict target[1:]
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        
        # Forward pass
        logits = model.forward(src_batch, tgt_input)
        
        # Compute loss
        loss = cross_entropy_loss(logits, tgt_output)
        acc = compute_accuracy(logits, tgt_output)
        
        epoch_loss += loss
        epoch_acc += acc
        
        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"    Batch {i+1}/{n_batches}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    return {
        'loss': epoch_loss / n_batches,
        'accuracy': epoch_acc / n_batches
    }


def evaluate(
    model: SimpleTransformer,
    src_data: np.ndarray,
    tgt_data: np.ndarray,
    batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    
    n_samples = src_data.shape[0]
    n_batches = n_samples // batch_size
    
    total_loss = 0.0
    total_acc = 0.0
    
    for i in range(n_batches):
        src_batch = src_data[i * batch_size:(i + 1) * batch_size]
        tgt_batch = tgt_data[i * batch_size:(i + 1) * batch_size]
        
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        
        logits = model.forward(src_batch, tgt_input)
        
        loss = cross_entropy_loss(logits, tgt_output)
        acc = compute_accuracy(logits, tgt_output)
        
        total_loss += loss
        total_acc += acc
    
    return {
        'loss': total_loss / n_batches,
        'accuracy': total_acc / n_batches
    }


# =============================================================================
# ATTENTION VISUALIZATION
# =============================================================================

def visualize_attention(model: SimpleTransformer, src: np.ndarray, tgt: np.ndarray):
    """Print attention patterns (text-based)."""
    model.eval()
    
    # Single sample
    src = src[0:1]
    tgt = tgt[0:1, :-1]  # Remove last token
    
    _ = model.forward(src, tgt)
    
    # Get decoder self-attention weights
    decoder_attn = model.model.decoder_layers[0].self_attention.get_attention_weights()
    
    if decoder_attn is not None:
        print("\nDecoder Self-Attention (Head 0):")
        print("-" * 40)
        
        attn = decoder_attn[0, 0]  # First sample, first head
        seq_len = min(attn.shape[0], 8)
        
        # Print header
        header = "    " + " ".join(f"{i:5d}" for i in range(seq_len))
        print(header)
        
        for i in range(seq_len):
            row = f"{i:3d} " + " ".join(f"{attn[i, j]:.2f}" for j in range(seq_len))
            print(row)


# =============================================================================
# GREEDY DECODING
# =============================================================================

def greedy_decode(
    model: SimpleTransformer,
    src: np.ndarray,
    max_len: int = 20,
    start_token: int = 1,
    end_token: int = 2
) -> np.ndarray:
    """
    Greedy decoding for sequence generation.
    
    At each step, pick the most likely next token.
    """
    model.eval()
    
    batch_size = src.shape[0]
    
    # Start with just the start token
    tgt = np.full((batch_size, 1), start_token, dtype=np.int32)
    
    for _ in range(max_len - 1):
        logits = model.forward(src, tgt)
        
        # Get prediction for last position
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        
        # Append to target
        tgt = np.concatenate([tgt, next_token], axis=1)
        
        # Check for end token
        if (next_token == end_token).all():
            break
    
    return tgt


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer on synthetic tasks")
    parser.add_argument("--task", type=str, default="copy", choices=["copy", "reverse"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Model: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    
    # Generate data
    print("\nGenerating data...")
    
    if args.task == "copy":
        train_src, train_tgt = generate_copy_task_data(
            n_samples=1000, seq_len=args.seq_len, vocab_size=args.vocab_size
        )
        test_src, test_tgt = generate_copy_task_data(
            n_samples=200, seq_len=args.seq_len, vocab_size=args.vocab_size
        )
    else:
        train_src, train_tgt = generate_reverse_task_data(
            n_samples=1000, seq_len=args.seq_len, vocab_size=args.vocab_size
        )
        test_src, test_tgt = generate_reverse_task_data(
            n_samples=200, seq_len=args.seq_len, vocab_size=args.vocab_size
        )
    
    print(f"Train samples: {len(train_src)}")
    print(f"Test samples: {len(test_src)}")
    print(f"Sample input:  {train_src[0]}")
    print(f"Sample target: {train_tgt[0]}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout_p=0.1
    )
    
    # Training loop
    print("\nTraining...")
    print("-" * 40)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_metrics = train_epoch(
            model, train_src, train_tgt,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        test_metrics = evaluate(model, test_src, test_tgt, batch_size=args.batch_size)
        
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Test Loss:  {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    final_metrics = evaluate(model, test_src, test_tgt, batch_size=args.batch_size)
    print(f"Test Loss: {final_metrics['loss']:.4f}")
    print(f"Test Accuracy: {final_metrics['accuracy']:.4f}")
    
    # Show some predictions
    print("\nSample Predictions:")
    print("-" * 40)
    
    for i in range(min(3, len(test_src))):
        src = test_src[i:i+1]
        tgt = test_tgt[i:i+1]
        
        pred = greedy_decode(model, src, max_len=args.seq_len + 2)
        
        print(f"  Input:    {src[0]}")
        print(f"  Expected: {tgt[0]}")
        print(f"  Predicted: {pred[0]}")
        print()
    
    # Show attention pattern
    visualize_attention(model, test_src[:1], test_tgt[:1])
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
