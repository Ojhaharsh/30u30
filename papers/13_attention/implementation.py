"""
Day 13: Attention Is All You Need - Implementation

A complete, from-scratch implementation of the Transformer architecture.
All core components with detailed explanations.

Paper: https://arxiv.org/abs/1706.03762

Components implemented:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Feed-Forward Network
5. Encoder Block
6. Decoder Block
7. Full Transformer
"""

import numpy as np
from typing import Optional, Tuple, List


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (used in modern transformers)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
               eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization.
    
    Normalizes across the last dimension (features).
    
    Args:
        x: Input of shape (..., features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized output of same shape
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding:
    """
    Sinusoidal Positional Encoding.
    
    Adds position information using sine and cosine functions of
    different frequencies. This allows the model to attend to
    relative positions, as PE(pos+k) can be represented as a
    linear function of PE(pos).
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to precompute
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.training = True
        
        # Precompute positional encodings
        self.pe = self._create_encoding(max_len, d_model)
    
    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create the sinusoidal encoding matrix."""
        pe = np.zeros((max_len, d_model))
        
        # Position indices: [0, 1, 2, ..., max_len-1]
        position = np.arange(max_len)[:, np.newaxis]
        
        # Division term: 10000^(2i/d_model) using exp-log for stability
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch, seq_len, d_model)
            
        Returns:
            Embeddings + positional encoding
        """
        seq_len = x.shape[1]
        
        # Add positional encoding (broadcasts over batch)
        output = x + self.pe[:seq_len]
        
        # Apply dropout during training
        if self.training and self.dropout_p > 0:
            mask = (np.random.rand(*output.shape) > self.dropout_p).astype(np.float32)
            output = output * mask / (1 - self.dropout_p)
        
        return output
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# SCALED DOT-PRODUCT ATTENTION
# =============================================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.
    
    The core attention mechanism:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Q (Query): What am I looking for?
    K (Key): What do I contain?
    V (Value): What do I give when matched?
    
    The scaling by sqrt(d_k) prevents the dot products from growing
    too large, which would push softmax into regions with tiny gradients.
    """
    
    def __init__(self, dropout_p: float = 0.1):
        self.dropout_p = dropout_p
        self.training = True
        self.attention_weights = None  # Cache for visualization
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries of shape (batch, heads, seq_q, d_k)
            K: Keys of shape (batch, heads, seq_k, d_k)
            V: Values of shape (batch, heads, seq_k, d_v)
            mask: Optional boolean mask, True = mask out (batch, 1, seq_q, seq_k)
            
        Returns:
            Attention output of shape (batch, heads, seq_q, d_v)
        """
        d_k = K.shape[-1]
        
        # Step 1: Compute attention scores
        # Q @ K^T -> (batch, heads, seq_q, seq_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Step 2: Scale by sqrt(d_k)
        scores = scores / np.sqrt(d_k)
        
        # Step 3: Apply mask (set masked positions to -inf)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        # Step 4: Softmax to get attention weights
        attention_weights = softmax(scores, axis=-1)
        
        # Cache for visualization
        self.attention_weights = attention_weights
        
        # Step 5: Apply dropout to attention weights
        if self.training and self.dropout_p > 0:
            dropout_mask = (np.random.rand(*attention_weights.shape) > self.dropout_p)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_p)
        
        # Step 6: Multiply by values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, d_v)
        # -> (batch, heads, seq_q, d_v)
        output = np.matmul(attention_weights, V)
        
        return output
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Instead of computing one attention function, we project Q, K, V
    into multiple subspaces (heads) and compute attention in parallel.
    This allows the model to learn different types of relationships.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout_p: Dropout probability
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout_p = dropout_p
        self.training = True
        
        # Initialize projection weights
        # Each head gets d_k dimensions
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout_p)
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute multi-head attention.
        
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional mask (batch, 1, seq_q, seq_k)
            
        Returns:
            Output of shape (batch, seq_q, d_model)
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]
        
        # Step 1: Linear projections
        Q = query @ self.W_Q  # (batch, seq_q, d_model)
        K = key @ self.W_K    # (batch, seq_k, d_model)
        V = value @ self.W_V  # (batch, seq_k, d_model)
        
        # Step 2: Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        Q = Q.reshape(batch_size, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Step 3: Apply attention
        self.attention.training = self.training
        attn_output = self.attention.forward(Q, K, V, mask)
        # Shape: (batch, n_heads, seq_q, d_k)
        
        # Step 4: Concatenate heads
        # (batch, n_heads, seq_q, d_k) -> (batch, seq_q, n_heads, d_k) -> (batch, seq_q, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.d_model)
        
        # Step 5: Final linear projection
        output = attn_output @ self.W_O
        
        return output
    
    def get_attention_weights(self) -> np.ndarray:
        """Get the last computed attention weights."""
        return self.attention.attention_weights
    
    def train(self):
        self.training = True
        self.attention.training = True
    
    def eval(self):
        self.training = False
        self.attention.training = False


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    
    Applied independently to each position.
    Typically d_ff = 4 * d_model.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (usually 4 * d_model)
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.training = True
        
        # Initialize weights
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input of shape (..., d_model)
            
        Returns:
            Output of shape (..., d_model)
        """
        # First linear + ReLU
        hidden = relu(x @ self.W1 + self.b1)
        
        # Dropout
        if self.training and self.dropout_p > 0:
            mask = (np.random.rand(*hidden.shape) > self.dropout_p).astype(np.float32)
            hidden = hidden * mask / (1 - self.dropout_p)
        
        # Second linear
        output = hidden @ self.W2 + self.b2
        
        return output
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# LAYER NORMALIZATION MODULE
# =============================================================================

class LayerNorm:
    """Layer Normalization module with learnable parameters."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        return layer_norm(x, self.gamma, self.beta, self.eps)


# =============================================================================
# ENCODER BLOCK
# =============================================================================

class EncoderBlock:
    """
    Transformer Encoder Block.
    
    Structure:
        x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
        
    Uses pre-norm in modern implementations (norm before attention).
    Original paper uses post-norm (norm after residual).
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.training = True
        
        # Components
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional padding mask
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.self_attention.forward(x, x, x, mask)
        
        # Dropout on attention output
        if self.training and self.dropout_p > 0:
            dropout_mask = (np.random.rand(*attn_output.shape) > self.dropout_p)
            attn_output = attn_output * dropout_mask / (1 - self.dropout_p)
        
        x = self.norm1.forward(x + attn_output)  # Add & Norm
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        
        # Dropout on FFN output
        if self.training and self.dropout_p > 0:
            dropout_mask = (np.random.rand(*ff_output.shape) > self.dropout_p)
            ff_output = ff_output * dropout_mask / (1 - self.dropout_p)
        
        x = self.norm2.forward(x + ff_output)  # Add & Norm
        
        return x
    
    def train(self):
        self.training = True
        self.self_attention.train()
        self.feed_forward.train()
    
    def eval(self):
        self.training = False
        self.self_attention.eval()
        self.feed_forward.eval()


# =============================================================================
# DECODER BLOCK
# =============================================================================

class DecoderBlock:
    """
    Transformer Decoder Block.
    
    Structure:
        x -> Masked Self-Attention -> Add & Norm 
          -> Cross-Attention -> Add & Norm 
          -> FFN -> Add & Norm -> output
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.training = True
        
        # Components
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through decoder block.
        
        Args:
            x: Decoder input (batch, seq_tgt, d_model)
            encoder_output: Encoder output (batch, seq_src, d_model)
            self_mask: Causal mask for self-attention
            cross_mask: Padding mask for cross-attention
            
        Returns:
            Output of shape (batch, seq_tgt, d_model)
        """
        # Masked self-attention
        self_attn = self.self_attention.forward(x, x, x, self_mask)
        if self.training and self.dropout_p > 0:
            mask = (np.random.rand(*self_attn.shape) > self.dropout_p)
            self_attn = self_attn * mask / (1 - self.dropout_p)
        x = self.norm1.forward(x + self_attn)
        
        # Cross-attention (attend to encoder output)
        cross_attn = self.cross_attention.forward(x, encoder_output, encoder_output, cross_mask)
        if self.training and self.dropout_p > 0:
            mask = (np.random.rand(*cross_attn.shape) > self.dropout_p)
            cross_attn = cross_attn * mask / (1 - self.dropout_p)
        x = self.norm2.forward(x + cross_attn)
        
        # Feed-forward
        ff_output = self.feed_forward.forward(x)
        if self.training and self.dropout_p > 0:
            mask = (np.random.rand(*ff_output.shape) > self.dropout_p)
            ff_output = ff_output * mask / (1 - self.dropout_p)
        x = self.norm3.forward(x + ff_output)
        
        return x
    
    def train(self):
        self.training = True
        self.self_attention.train()
        self.cross_attention.train()
        self.feed_forward.train()
    
    def eval(self):
        self.training = False
        self.self_attention.eval()
        self.cross_attention.eval()
        self.feed_forward.eval()


# =============================================================================
# TOKEN EMBEDDING
# =============================================================================

class TokenEmbedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Look up embeddings.
        
        Args:
            x: Token indices (batch, seq_len)
            
        Returns:
            Embeddings (batch, seq_len, d_model)
        """
        return self.embedding[x]


# =============================================================================
# FULL TRANSFORMER
# =============================================================================

class Transformer:
    """
    Complete Transformer Model (Encoder-Decoder).
    
    For sequence-to-sequence tasks like translation.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout_p: float = 0.1,
        max_len: int = 5000
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder blocks
            n_decoder_layers: Number of decoder blocks
            d_ff: Feed-forward hidden dimension
            dropout_p: Dropout probability
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.training = True
        
        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_p)
        
        # Encoder stack
        self.encoder_layers = [
            EncoderBlock(d_model, n_heads, d_ff, dropout_p)
            for _ in range(n_encoder_layers)
        ]
        
        # Decoder stack
        self.decoder_layers = [
            DecoderBlock(d_model, n_heads, d_ff, dropout_p)
            for _ in range(n_decoder_layers)
        ]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, tgt_vocab_size) * 0.02
    
    def encode(self, src: np.ndarray, src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode source sequence.
        
        Args:
            src: Source tokens (batch, src_len)
            src_mask: Optional padding mask
            
        Returns:
            Encoder output (batch, src_len, d_model)
        """
        # Embed and add positional encoding
        x = self.src_embedding.forward(src) * np.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer.forward(x, src_mask)
        
        return x
    
    def decode(self, tgt: np.ndarray, encoder_output: np.ndarray,
               tgt_mask: Optional[np.ndarray] = None,
               cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode target sequence given encoder output.
        
        Args:
            tgt: Target tokens (batch, tgt_len)
            encoder_output: From encoder (batch, src_len, d_model)
            tgt_mask: Causal + padding mask for target
            cross_mask: Padding mask for encoder output
            
        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        # Embed and add positional encoding
        x = self.tgt_embedding.forward(tgt) * np.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, tgt_mask, cross_mask)
        
        return x
    
    def forward(self, src: np.ndarray, tgt: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full forward pass.
        
        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
            
        Returns:
            Logits (batch, tgt_len, tgt_vocab_size)
        """
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        logits = decoder_output @ self.output_projection
        
        return logits
    
    def train(self):
        self.training = True
        self.positional_encoding.train()
        for layer in self.encoder_layers:
            layer.train()
        for layer in self.decoder_layers:
            layer.train()
    
    def eval(self):
        self.training = False
        self.positional_encoding.eval()
        for layer in self.encoder_layers:
            layer.eval()
        for layer in self.decoder_layers:
            layer.eval()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask for decoder self-attention.
    Prevents attending to future positions.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Boolean mask (1, 1, seq_len, seq_len), True = masked
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask[np.newaxis, np.newaxis, :, :]  # Add batch and head dims


def create_padding_mask(seq: np.ndarray, pad_token: int = 0) -> np.ndarray:
    """
    Create padding mask.
    
    Args:
        seq: Token sequence (batch, seq_len)
        pad_token: Padding token ID
        
    Returns:
        Boolean mask (batch, 1, 1, seq_len), True = masked
    """
    mask = (seq == pad_token)
    return mask[:, np.newaxis, np.newaxis, :]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFORMER IMPLEMENTATION DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # ---------------------------------
    # 1. Scaled Dot-Product Attention
    # ---------------------------------
    print("\n1. Scaled Dot-Product Attention:")
    print("-" * 40)
    
    batch, heads, seq_len, d_k = 2, 4, 5, 64
    Q = np.random.randn(batch, heads, seq_len, d_k)
    K = np.random.randn(batch, heads, seq_len, d_k)
    V = np.random.randn(batch, heads, seq_len, d_k)
    
    attention = ScaledDotProductAttention(dropout_p=0.0)
    attention.eval()
    output = attention.forward(Q, K, V)
    
    print(f"  Q, K, V shapes: {Q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attention.attention_weights.shape}")
    print(f"  Attention weights sum (per query): {attention.attention_weights[0, 0].sum(axis=-1)}")
    
    # ---------------------------------
    # 2. Multi-Head Attention
    # ---------------------------------
    print("\n2. Multi-Head Attention:")
    print("-" * 40)
    
    batch, seq_len, d_model, n_heads = 2, 10, 128, 8
    x = np.random.randn(batch, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, n_heads, dropout_p=0.0)
    mha.eval()
    output = mha.forward(x, x, x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Dimension per head: {d_model // n_heads}")
    
    # ---------------------------------
    # 3. Positional Encoding
    # ---------------------------------
    print("\n3. Positional Encoding:")
    print("-" * 40)
    
    pe = PositionalEncoding(d_model=128, max_len=100)
    embeddings = np.random.randn(2, 20, 128)
    encoded = pe.forward(embeddings)
    
    print(f"  Input shape: {embeddings.shape}")
    print(f"  Output shape: {encoded.shape}")
    print(f"  PE matrix shape: {pe.pe.shape}")
    print(f"  Sample PE values (pos 0): {pe.pe[0, :4]}")
    print(f"  Sample PE values (pos 1): {pe.pe[1, :4]}")
    
    # ---------------------------------
    # 4. Encoder Block
    # ---------------------------------
    print("\n4. Encoder Block:")
    print("-" * 40)
    
    d_model, n_heads, d_ff = 128, 8, 512
    encoder = EncoderBlock(d_model, n_heads, d_ff, dropout_p=0.0)
    encoder.eval()
    
    x = np.random.randn(2, 10, d_model)
    output = encoder.forward(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}")
    
    # ---------------------------------
    # 5. Full Transformer
    # ---------------------------------
    print("\n5. Full Transformer:")
    print("-" * 40)
    
    transformer = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout_p=0.0
    )
    transformer.eval()
    
    src = np.random.randint(0, 1000, (2, 8))  # Source sequence
    tgt = np.random.randint(0, 1000, (2, 6))  # Target sequence
    
    # Create causal mask for decoder
    tgt_mask = create_causal_mask(tgt.shape[1])
    
    logits = transformer.forward(src, tgt, tgt_mask=tgt_mask)
    
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Predicted tokens: {logits.argmax(axis=-1)[0]}")
    
    # ---------------------------------
    # 6. Attention Visualization
    # ---------------------------------
    print("\n6. Attention Weights Sample:")
    print("-" * 40)
    
    weights = transformer.decoder_layers[0].self_attention.get_attention_weights()
    if weights is not None:
        print(f"  Attention weights shape: {weights.shape}")
        print(f"  (batch, heads, query_pos, key_pos)")
        print(f"  Sample attention pattern (head 0, first 4x4):")
        sample = weights[0, 0, :4, :4]
        for row in sample:
            print(f"    [{', '.join(f'{v:.2f}' for v in row)}]")
    
    print("\n" + "=" * 60)
    print("Demo complete! Explore the components to learn more.")
    print("=" * 60)
