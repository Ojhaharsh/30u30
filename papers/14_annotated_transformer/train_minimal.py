"""
Minimal Training Script for The Annotated Transformer
======================================================

A simple, self-contained script to train a Transformer on the copy task.
Run this to verify your implementation works!

Usage:
    python train_minimal.py
"""

import math
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MODEL COMPONENTS (Minimal, self-contained)
# =============================================================================

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1) == 0
    return mask


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(vocab, N=2, d_model=128, d_ff=256, h=4, dropout=0.1):
    """Create a small Transformer for the copy task."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class Batch:
    """Simple batch holder."""
    def __init__(self, src, tgt, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = self.make_mask(self.tgt, pad)
        self.ntokens = (self.tgt_y != pad).sum().item()
    
    @staticmethod
    def make_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def data_gen(vocab, batch_size, nbatches, seq_len=10):
    """Generate random data for the copy task."""
    for _ in range(nbatches):
        data = torch.randint(1, vocab, size=(batch_size, seq_len))
        data[:, 0] = 1  # Start token
        yield Batch(data.clone(), data.clone())


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Simple greedy decoding."""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    
    for _ in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return ys


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train():
    """Train a small Transformer on the copy task."""
    print("=" * 60)
    print("MINIMAL TRANSFORMER TRAINING")
    print("=" * 60)
    
    # Hyperparameters
    VOCAB = 11
    BATCH_SIZE = 32
    EPOCHS = 10
    BATCHES_PER_EPOCH = 20
    SEQ_LEN = 10
    
    # Create model
    model = make_model(VOCAB)
    print(f"\n[1] Model created")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    print(f"\n[2] Training on copy task...")
    print(f"    Epochs: {EPOCHS}")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Sequence length: {SEQ_LEN}")
    print()
    
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_tokens = 0
        
        for batch in data_gen(VOCAB, BATCH_SIZE, BATCHES_PER_EPOCH, SEQ_LEN):
            optimizer.zero_grad()
            
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            
            loss = criterion(
                out.reshape(-1, VOCAB),
                batch.tgt_y.reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.ntokens
            total_tokens += batch.ntokens
        
        avg_loss = total_loss / total_tokens
        print(f"    Epoch {epoch + 1:2d}/{EPOCHS}: Loss = {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n    Training completed in {elapsed:.1f}s")
    
    # Test
    print(f"\n[3] Testing...")
    model.eval()
    
    test_src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    test_mask = torch.ones(1, 1, 10)
    
    with torch.no_grad():
        output = greedy_decode(model, test_src, test_mask, max_len=10, start_symbol=1)
    
    print(f"    Input:  {test_src[0].tolist()}")
    print(f"    Output: {output[0].tolist()}")
    
    # Check accuracy
    match = (test_src[0] == output[0][:10]).sum().item()
    print(f"    Match:  {match}/10 tokens ({100 * match / 10:.0f}%)")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    train()
