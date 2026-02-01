"""
Solution 5: Full Transformer

Complete encoder-decoder Transformer implementation.
"""

import numpy as np
from typing import Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create mask preventing attention to future positions."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask[np.newaxis, np.newaxis, :, :]


# ============= Components =============

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
    
    def forward(self, query, key, value, mask=None):
        batch, seq_q, _ = query.shape
        seq_k = key.shape[1]
        
        Q = (query @ self.W_Q).reshape(batch, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = (key @ self.W_K).reshape(batch, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = (value @ self.W_V).reshape(batch, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        attn = softmax(scores, axis=-1) @ V
        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_q, self.d_model)
        return attn @ self.W_O


class FeedForward:
    def __init__(self, d_model, d_ff):
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = pe
    
    def forward(self, x):
        return x + self.pe[:x.shape[1]]


class EncoderBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.norm1.forward(x + self.attn.forward(x, x, x, mask))
        x = self.norm2.forward(x + self.ff.forward(x))
        return x


class DecoderBlock:
    """
    Transformer Decoder Block.
    
    Key differences from encoder:
    1. MASKED self-attention (can't see future)
    2. CROSS-attention to encoder output
    
    Structure:
        x -> Masked Self-Attn -> Add & Norm
          -> Cross-Attn -> Add & Norm  
          -> FFN -> Add & Norm -> output
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: np.ndarray = None, 
                cross_mask: np.ndarray = None) -> np.ndarray:
        # Masked self-attention: each position only attends to previous positions
        x = self.norm1.forward(x + self.self_attn.forward(x, x, x, self_mask))
        
        # Cross-attention: decoder queries attend to encoder keys/values
        # This is how the decoder "reads" the input
        x = self.norm2.forward(x + self.cross_attn.forward(x, encoder_output, encoder_output, cross_mask))
        
        # Feed-forward
        x = self.norm3.forward(x + self.ff.forward(x))
        
        return x


class Transformer:
    """
    Complete Transformer Model.
    
    Architecture:
        Encoder: Embed -> PE -> N x EncoderBlock
        Decoder: Embed -> PE -> N x DecoderBlock -> Linear
    
    Key insight: The encoder processes input once, then the decoder
    generates output tokens one at a time, attending to the encoder.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256
    ):
        self.d_model = d_model
        
        # Embeddings
        self.src_embed = np.random.randn(src_vocab_size, d_model) * 0.02
        self.tgt_embed = np.random.randn(tgt_vocab_size, d_model) * 0.02
        self.pos_enc = PositionalEncoding(d_model)
        
        # Encoder and decoder stacks
        self.encoders = [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.decoders = [DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, tgt_vocab_size) * 0.02
    
    def encode(self, src: np.ndarray) -> np.ndarray:
        """Encode source sequence."""
        # Scale embeddings by sqrt(d_model) as per paper
        x = self.src_embed[src] * np.sqrt(self.d_model)
        x = self.pos_enc.forward(x)
        
        for encoder in self.encoders:
            x = encoder.forward(x)
        
        return x
    
    def decode(self, tgt: np.ndarray, encoder_output: np.ndarray) -> np.ndarray:
        """Decode target sequence given encoder output."""
        x = self.tgt_embed[tgt] * np.sqrt(self.d_model)
        x = self.pos_enc.forward(x)
        
        # Causal mask prevents attending to future tokens
        tgt_len = tgt.shape[1]
        causal_mask = create_causal_mask(tgt_len)
        
        for decoder in self.decoders:
            x = decoder.forward(x, encoder_output, self_mask=causal_mask)
        
        return x
    
    def forward(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """
        Full forward pass.
        
        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            
        Returns:
            Logits (batch, tgt_len, tgt_vocab_size)
        """
        # Step 1: Encode source sequence
        encoder_output = self.encode(src)
        
        # Step 2: Decode target sequence
        decoder_output = self.decode(tgt, encoder_output)
        
        # Step 3: Project to vocabulary
        logits = decoder_output @ self.output_proj
        
        return logits


# =============================================================================
# TESTS
# =============================================================================

def test_causal_mask():
    """Test causal mask creation."""
    mask = create_causal_mask(4)
    
    expected = np.array([
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False]
    ])
    
    assert np.array_equal(mask[0, 0], expected)
    print("[PASS] Causal mask test passed!")


def test_encoder():
    """Test encoder."""
    transformer = Transformer(100, 100, d_model=32, n_heads=4, n_layers=2)
    src = np.random.randint(0, 100, (2, 8))
    
    encoder_output = transformer.encode(src)
    
    assert encoder_output.shape == (2, 8, 32)
    print("[PASS] Encoder test passed!")


def test_decoder():
    """Test decoder."""
    transformer = Transformer(100, 100, d_model=32, n_heads=4, n_layers=2)
    
    encoder_output = np.random.randn(2, 8, 32)
    tgt = np.random.randint(0, 100, (2, 6))
    
    decoder_output = transformer.decode(tgt, encoder_output)
    
    assert decoder_output.shape == (2, 6, 32)
    print("[PASS] Decoder test passed!")


def test_full_forward():
    """Test full forward pass."""
    transformer = Transformer(100, 100, d_model=32, n_heads=4, n_layers=2)
    
    src = np.random.randint(0, 100, (2, 8))
    tgt = np.random.randint(0, 100, (2, 6))
    
    logits = transformer.forward(src, tgt)
    
    assert logits.shape == (2, 6, 100)
    print("[PASS] Full forward test passed!")


def test_prediction():
    """Test that we can make predictions."""
    np.random.seed(42)
    
    transformer = Transformer(20, 20, d_model=32, n_heads=4, n_layers=2)
    
    src = np.array([[1, 2, 3, 4, 5]])
    tgt = np.array([[1]])
    
    logits = transformer.forward(src, tgt)
    prediction = logits.argmax(axis=-1)
    
    assert prediction.shape == (1, 1)
    print(f"[PASS] Prediction test passed!")


def test_copy_task():
    """Test on simple copy task."""
    np.random.seed(42)
    
    transformer = Transformer(10, 10, d_model=32, n_heads=2, n_layers=1)
    
    src = np.array([[1, 2, 3, 4, 5]])
    tgt_input = np.array([[0, 1, 2, 3, 4]])
    
    logits = transformer.forward(src, tgt_input)
    
    assert logits.shape == (1, 5, 10)
    print("[PASS] Copy task test passed!")


def demonstrate_transformer():
    """Demonstrate the full Transformer."""
    print("\n" + "=" * 50)
    print("FULL TRANSFORMER DEMONSTRATION")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Create a small Transformer
    transformer = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128
    )
    
    print("\nConfiguration:")
    print(f"  src_vocab_size: 20")
    print(f"  tgt_vocab_size: 20")
    print(f"  d_model: 32")
    print(f"  n_heads: 4")
    print(f"  n_layers: 2")
    print(f"  d_ff: 128")
    
    # Example: translation-like task
    src = np.array([[1, 5, 8, 3, 2]])  # Source sentence tokens
    tgt = np.array([[0]])  # Start with just start token
    
    print(f"\nSource: {src[0]}")
    print(f"Initial target: {tgt[0]}")
    
    # Greedy decoding
    print("\nGreedy decoding:")
    for step in range(5):
        logits = transformer.forward(src, tgt)
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        tgt = np.concatenate([tgt, next_token], axis=1)
        print(f"  Step {step + 1}: predicted token {next_token[0, 0]}, sequence: {tgt[0]}")
    
    print(f"\nFinal output: {tgt[0]}")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("SOLUTION 5: FULL TRANSFORMER")
    print("=" * 50)
    
    test_causal_mask()
    test_encoder()
    test_decoder()
    test_full_forward()
    test_prediction()
    test_copy_task()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
    
    demonstrate_transformer()
    
    print("\n" + "=" * 50)
    print("CONGRATULATIONS!")
    print("You've built a complete Transformer from scratch!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
