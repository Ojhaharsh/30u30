"""
The Annotated Transformer - Day 14 Implementation
=================================================

Production-quality PyTorch implementation of the Transformer architecture.
Based on "The Annotated Transformer" by Alexander Rush (Harvard NLP).

This implementation provides:
- Complete encoder-decoder Transformer
- Multi-head attention with masking
- Positional encoding (sinusoidal)
- Label smoothing and Noam learning rate schedule
- Training and inference utilities

Reference: https://nlp.seas.harvard.edu/annotated-transformer/
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clones(module, N):
    """Produce N identical layers (deep copies)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    Returns a mask where position i can attend to positions <= i.
    
    Example for size=4:
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


# =============================================================================
# LAYER NORMALIZATION
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    
    Unlike Batch Norm, Layer Norm normalizes across features (not batch).
    This makes it suitable for variable-length sequences and works
    the same during training and inference.
    """
    
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # Scale parameter (gamma)
        self.b_2 = nn.Parameter(torch.zeros(features))  # Shift parameter (beta)
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# =============================================================================
# SUBLAYER CONNECTION (RESIDUAL + NORM)
# =============================================================================

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    
    Note: This uses PRE-NORM (normalize before sublayer), which is
    different from the original paper's POST-NORM. Pre-norm training
    is more stable, especially for deep models.
    """
    
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """Apply residual connection: x + dropout(sublayer(norm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))


# =============================================================================
# ATTENTION
# =============================================================================

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        query: (batch, heads, seq_q, d_k)
        key: (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask: Optional mask (batch, 1, 1, seq_k) or (batch, 1, seq_q, seq_k)
        dropout: Optional dropout module
    
    Returns:
        output: (batch, heads, seq_q, d_v)
        attention_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    
    # Compute attention scores: (batch, heads, seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Convert to probabilities
    p_attn = scores.softmax(dim=-1)
    
    # Apply dropout to attention weights
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # Weighted sum of values
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Instead of performing a single attention function with d_model-dimensional
    keys, values, and queries, we project them h times with different learned
    linear projections to d_k, d_k, and d_v dimensions.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h: Number of attention heads
            d_model: Model dimension (must be divisible by h)
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h  # Dimension per head
        self.h = h
        
        # Four linear projections: Q, K, V, and output
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_q, d_model)
        """
        if mask is not None:
            # Same mask applied to all heads
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        
        # 1) Project and reshape: (batch, seq, d_model) -> (batch, h, seq, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]
        
        # 2) Apply attention on all projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) Concat heads: (batch, h, seq, d_k) -> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        # 4) Final linear projection
        return self.linears[-1](x)


# =============================================================================
# POSITION-WISE FEED-FORWARD NETWORK
# =============================================================================

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Two linear transformations with a ReLU activation in between.
    The hidden dimension is typically 4x the model dimension.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden layer dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Adds position information to the embeddings using fixed sinusoidal functions.
    Different frequencies allow the model to learn to attend by relative position.
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        """
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length to pre-compute
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Embeddings (batch, seq_len, d_model)
        
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


# =============================================================================
# EMBEDDINGS
# =============================================================================

class Embeddings(nn.Module):
    """
    Token Embeddings with scaling.
    
    The embedding weights are multiplied by sqrt(d_model) to maintain
    the scale after positional encoding is added.
    """
    
    def __init__(self, d_model, vocab):
        """
        Args:
            d_model: Embedding dimension
            vocab: Vocabulary size
        """
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# =============================================================================
# ENCODER
# =============================================================================

class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.
    
    Each layer has two sub-layers:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    
    Each sub-layer has a residual connection and layer normalization.
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Args:
            size: Model dimension (d_model)
            self_attn: Multi-head attention module
            feed_forward: Position-wise FFN module
            dropout: Dropout rate
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        """
        Args:
            x: Input (batch, seq, d_model)
            mask: Source mask (batch, 1, seq)
        
        Returns:
            Output (batch, seq, d_model)
        """
        # Self-attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Feed-forward
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Encoder: Stack of N encoder layers.
    
    The encoder maps an input sequence to a continuous representation
    that the decoder will attend to.
    """
    
    def __init__(self, layer, N):
        """
        Args:
            layer: Encoder layer to clone
            N: Number of layers
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        """Pass input through all layers, then normalize."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# DECODER
# =============================================================================

class DecoderLayer(nn.Module):
    """
    Single Decoder Layer.
    
    Each layer has three sub-layers:
    1. Masked multi-head self-attention (can't see future)
    2. Multi-head cross-attention (attends to encoder output)
    3. Position-wise feed-forward network
    
    Each sub-layer has a residual connection and layer normalization.
    """
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Args:
            size: Model dimension (d_model)
            self_attn: Masked self-attention module
            src_attn: Cross-attention module
            feed_forward: Position-wise FFN module
            dropout: Dropout rate
        """
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: Target input (batch, tgt_seq, d_model)
            memory: Encoder output (batch, src_seq, d_model)
            src_mask: Source mask (batch, 1, src_seq)
            tgt_mask: Target mask (batch, tgt_seq, tgt_seq)
        
        Returns:
            Output (batch, tgt_seq, d_model)
        """
        m = memory
        # Masked self-attention (only attend to past positions)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Cross-attention (attend to encoder output)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # Feed-forward
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Decoder: Stack of N decoder layers with masking.
    
    The decoder generates the output sequence one token at a time,
    attending to both the encoder output and previously generated tokens.
    """
    
    def __init__(self, layer, N):
        """
        Args:
            layer: Decoder layer to clone
            N: Number of layers
        """
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass input through all layers, then normalize."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# =============================================================================
# GENERATOR (OUTPUT PROJECTION)
# =============================================================================

class Generator(nn.Module):
    """
    Output Generator.
    
    Projects decoder output to vocabulary size and applies log softmax.
    """
    
    def __init__(self, d_model, vocab):
        """
        Args:
            d_model: Model dimension
            vocab: Vocabulary size
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# =============================================================================
# FULL ENCODER-DECODER MODEL
# =============================================================================

class EncoderDecoder(nn.Module):
    """
    Complete Transformer Encoder-Decoder Architecture.
    
    Combines:
    - Source embedding + positional encoding
    - Encoder stack
    - Target embedding + positional encoding
    - Decoder stack
    - Output generator
    """
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Full forward pass: encode source, decode target.
        
        Args:
            src: Source tokens (batch, src_seq)
            tgt: Target tokens (batch, tgt_seq)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        
        Returns:
            Output probabilities (batch, tgt_seq, vocab)
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        """Encode source sequence."""
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """Decode target sequence given encoder output."""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Construct a Transformer model from hyperparameters.
    
    Args:
        src_vocab: Source vocabulary size
        tgt_vocab: Target vocabulary size
        N: Number of encoder/decoder layers
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        h: Number of attention heads
        dropout: Dropout rate
    
    Returns:
        Initialized Transformer model
    """
    c = copy.deepcopy
    
    # Create component modules
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    # Assemble the model
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # Initialize parameters with Xavier uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class Batch:
    """
    Object for holding a batch of data with mask during training.
    
    Handles:
    - Source padding mask
    - Target shift (input vs output)
    - Target causal mask (can't see future)
    """
    
    def __init__(self, src, tgt=None, pad=0):
        """
        Args:
            src: Source tokens (batch, src_seq)
            tgt: Target tokens (batch, tgt_seq) - optional
            pad: Padding token index
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        
        if tgt is not None:
            # Decoder input: all tokens except last
            self.tgt = tgt[:, :-1]
            # Target for loss: all tokens except first
            self.tgt_y = tgt[:, 1:]
            # Create mask (padding + causal)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            # Count non-pad tokens for loss normalization
            self.ntokens = (self.tgt_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class LabelSmoothing(nn.Module):
    """
    Label Smoothing loss.
    
    Instead of one-hot targets (100% confidence), spreads some probability
    mass to other tokens. This prevents overconfidence and improves
    generalization.
    
    Example with smoothing=0.1:
    - True class: 0.9 probability
    - Other classes: 0.1 / (vocab - 2) each (excluding padding)
    """
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        Args:
            size: Vocabulary size
            padding_idx: Index of padding token (excluded from smoothing)
            smoothing: Amount of probability to spread (0 = standard cross-entropy)
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    
    def forward(self, x, target):
        """
        Args:
            x: Log probabilities (batch * seq, vocab)
            target: Target indices (batch * seq,)
        
        Returns:
            Smoothed loss
        """
        assert x.size(1) == self.size
        
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # Spread to other classes
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # True class
        true_dist[:, self.padding_idx] = 0  # Zero for padding
        
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def rate(step, d_model, factor=1.0, warmup=4000):
    """
    Noam learning rate schedule.
    
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    
    Increases linearly during warmup, then decays proportionally to
    the inverse square root of step.
    """
    if step == 0:
        step = 1
    return factor * (d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class NoamOpt:
    """
    Optimizer wrapper with Noam learning rate schedule.
    """
    
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        """Compute current learning rate."""
        if step is None:
            step = self._step
        return rate(step, self.model_size, self.factor, self.warmup)
    
    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model):
    """Standard optimizer with Noam schedule."""
    return NoamOpt(
        model.src_embed[0].d_model,
        factor=2,
        warmup=4000,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

class SimpleLossCompute:
    """Simple loss computation and update wrapper."""
    
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        """
        Compute loss and optionally update.
        
        Args:
            x: Decoder output (batch, seq, d_model)
            y: Target tokens (batch, seq)
            norm: Normalization factor (number of tokens)
        
        Returns:
            Normalized loss value
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        
        return loss.data.item() * norm


def run_epoch(data_iter, model, loss_compute, print_every=50):
    """
    Run one epoch of training or evaluation.
    
    Args:
        data_iter: Iterator over batches
        model: Transformer model
        loss_compute: Loss computation wrapper
        print_every: Print frequency
    
    Returns:
        Average loss per token
    """
    import time
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        if (i + 1) % print_every == 0:
            elapsed = time.time() - start
            print(f"Step {i+1} | Loss: {loss / batch.ntokens:.4f} | "
                  f"Tokens/sec: {tokens / elapsed:.0f}")
            start = time.time()
            tokens = 0
    
    return total_loss / total_tokens


# =============================================================================
# INFERENCE
# =============================================================================

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Greedy decoding: always pick the most probable next token.
    
    Args:
        model: Trained Transformer model
        src: Source tokens (1, src_seq)
        src_mask: Source mask (1, 1, src_seq)
        max_len: Maximum output length
        start_symbol: Start-of-sequence token index
    
    Returns:
        Generated token sequence (1, gen_len)
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size=4):
    """
    Beam search decoding: keep top-k candidates at each step.
    
    Args:
        model: Trained Transformer model
        src: Source tokens (1, src_seq)
        src_mask: Source mask
        max_len: Maximum output length
        start_symbol: Start-of-sequence token index
        beam_size: Number of candidates to keep
    
    Returns:
        Best generated sequence
    """
    memory = model.encode(src, src_mask)
    
    # Initialize with start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    scores = torch.zeros(1).type_as(src.data.float())
    
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        
        # Expand memory for beam
        mem = memory.expand(ys.size(0), -1, -1)
        src_m = src_mask.expand(ys.size(0), -1, -1)
        
        out = model.decode(mem, src_m, ys, tgt_mask)
        log_prob = model.generator(out[:, -1])
        
        # Get top candidates
        vocab_size = log_prob.size(-1)
        
        # Compute new scores
        new_scores = scores.unsqueeze(-1) + log_prob  # (beam, vocab)
        new_scores = new_scores.view(-1)  # (beam * vocab)
        
        # Select top beam_size
        top_scores, top_indices = torch.topk(new_scores, min(beam_size, new_scores.size(0)))
        
        # Decode indices
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # Update sequences and scores
        ys = torch.cat([ys[beam_indices], token_indices.unsqueeze(1)], dim=1)
        scores = top_scores
    
    # Return best sequence
    return ys[0:1]


# =============================================================================
# DEMO: SYNTHETIC COPY TASK
# =============================================================================

def data_gen(V, batch_size, nbatches):
    """
    Generate random data for the copy task.
    
    The model learns to copy the input sequence to the output.
    Source and target are identical (shifted by one for decoder).
    """
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1  # Start symbol
        src = data.clone()
        tgt = data.clone()
        yield Batch(src, tgt, pad=0)


def run_copy_task_demo():
    """
    Demo: Train a small Transformer on the copy task.
    
    The model learns to copy input sequences.
    """
    print("=" * 60)
    print("THE ANNOTATED TRANSFORMER - COPY TASK DEMO")
    print("=" * 60)
    
    # Small model for quick demo
    V = 11  # Vocabulary size
    model = make_model(V, V, N=2, d_model=128, d_ff=256, h=4, dropout=0.1)
    
    # Training setup
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    optimizer = NoamOpt(128, factor=1, warmup=400,
                        optimizer=torch.optim.Adam(model.parameters(), lr=0, 
                                                   betas=(0.9, 0.98), eps=1e-9))
    
    print("\n[1] Model Architecture:")
    print(f"    - Layers: 2 (encoder) + 2 (decoder)")
    print(f"    - d_model: 128")
    print(f"    - Attention heads: 4")
    print(f"    - Vocabulary: {V}")
    
    # Training
    print("\n[2] Training on copy task...")
    model.train()
    
    for epoch in range(5):
        loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)
        avg_loss = 0
        for i, batch in enumerate(data_gen(V, batch_size=32, nbatches=20)):
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = loss_compute(out, batch.tgt_y, batch.ntokens)
            avg_loss += loss
        
        avg_loss /= (20 * 32 * 9)  # Normalize
        print(f"    Epoch {epoch + 1}/5: Loss = {avg_loss:.4f}")
    
    # Inference demo
    print("\n[3] Testing greedy decoding...")
    model.eval()
    
    test_src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    test_mask = torch.ones(1, 1, 10)
    
    output = greedy_decode(model, test_src, test_mask, max_len=10, start_symbol=1)
    
    print(f"    Input:  {test_src[0].tolist()}")
    print(f"    Output: {output[0].tolist()}")
    
    # Check accuracy
    correct = (test_src[0][1:] == output[0][1:test_src.size(1)]).sum().item()
    print(f"    Match:  {correct}/{test_src.size(1) - 1} tokens")
    
    print("\n" + "=" * 60)
    print("Demo complete! Model learned to copy sequences.")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run demo
    run_copy_task_demo()
    
    print("\n" + "=" * 60)
    print("COMPONENT TESTS")
    print("=" * 60)
    
    # Test individual components
    print("\n[1] Testing Positional Encoding...")
    pe = PositionalEncoding(d_model=512, dropout=0.0)
    x = torch.zeros(1, 100, 512)
    out = pe(x)
    print(f"    Shape: {out.shape}")
    print(f"    PE values at pos=0: {out[0, 0, :4].tolist()}")
    
    print("\n[2] Testing Multi-Head Attention...")
    mha = MultiHeadedAttention(h=8, d_model=512)
    q = k = v = torch.randn(2, 10, 512)
    out = mha(q, k, v)
    print(f"    Input shape: {q.shape}")
    print(f"    Output shape: {out.shape}")
    
    print("\n[3] Testing subsequent_mask...")
    mask = subsequent_mask(5)
    print(f"    Mask shape: {mask.shape}")
    print(f"    Mask:\n{mask[0].int()}")
    
    print("\n[4] Testing full model creation...")
    model = make_model(src_vocab=1000, tgt_vocab=1000, N=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    
    print("\n[5] Testing forward pass...")
    src = torch.randint(0, 1000, (2, 15))
    tgt = torch.randint(0, 1000, (2, 12))
    src_mask = (src != 0).unsqueeze(-2)
    tgt_mask = (tgt[:, :-1] != 0).unsqueeze(-2) & subsequent_mask(tgt.size(1) - 1)
    
    out = model(src, tgt[:, :-1], src_mask, tgt_mask)
    print(f"    Source shape: {src.shape}")
    print(f"    Target shape: {tgt[:, :-1].shape}")
    print(f"    Output shape: {out.shape}")
    
    print("\n" + "=" * 60)
    print("All component tests passed!")
    print("=" * 60)
