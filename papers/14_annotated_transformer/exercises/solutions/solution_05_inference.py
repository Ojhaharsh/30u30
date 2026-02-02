"""
Solution 5: Inference and Decoding
===================================

Complete solution for Exercise 5.
"""

import math
import copy
import torch
import torch.nn as nn


# =============================================================================
# MODEL COMPONENTS (Same as in exercise)
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


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears[:3], (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        import torch.nn.functional as F
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
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
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
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
        import torch.nn.functional as F
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


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# =============================================================================
# DECODING FUNCTIONS
# =============================================================================

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Greedy decoding: always pick the most probable next token.
    """
    # Step 1: Encode source (do this ONCE, reuse for all steps)
    memory = model.encode(src, src_mask)
    
    # Step 2: Initialize output with start symbol
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    # Step 3: Generate tokens one by one
    for i in range(max_len - 1):
        # Create causal mask for current output
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        
        # Decode
        out = model.decode(memory, src_mask, ys, tgt_mask)
        
        # Project last position to vocabulary (log probs)
        prob = model.generator(out[:, -1])
        
        # Pick most probable token
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        
        # Append to output
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    
    return ys


def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_size=4, end_symbol=None):
    """
    Beam search decoding: keep top-k candidates at each step.
    """
    # Step 1: Encode source
    memory = model.encode(src, src_mask)
    
    # Step 2: Initialize
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    scores = torch.zeros(1).type_as(src.data.float())
    
    # Step 3: Main loop
    for step in range(max_len - 1):
        # Create mask for current hypotheses
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        
        # Expand memory and mask for beam
        curr_beam = ys.size(0)
        mem = memory.expand(curr_beam, -1, -1)
        src_m = src_mask.expand(curr_beam, -1, -1)
        
        # Decode all hypotheses
        out = model.decode(mem, src_m, ys, tgt_mask)
        log_prob = model.generator(out[:, -1])  # (beam, vocab)
        
        vocab_size = log_prob.size(-1)
        
        # Compute new scores: current_score + new_token_score
        new_scores = scores.unsqueeze(-1) + log_prob  # (beam, vocab)
        new_scores = new_scores.view(-1)  # (beam * vocab)
        
        # Select top beam_size candidates
        top_scores, top_indices = torch.topk(new_scores, beam_size)
        
        # Decode which hypothesis and which token
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # Update hypotheses
        ys = torch.cat([ys[beam_indices], token_indices.unsqueeze(1)], dim=1)
        scores = top_scores
        
        # Optional: Early stopping if all beams end with end_symbol
        if end_symbol is not None:
            if (ys[:, -1] == end_symbol).all():
                break
    
    # Step 4: Return best hypothesis
    best_idx = scores.argmax()
    return ys[best_idx:best_idx+1]


# =============================================================================
# TESTS
# =============================================================================

def test_greedy_decode():
    """Test greedy decoding."""
    print("Testing greedy_decode...")
    
    vocab_size = 11
    model = make_model(vocab_size, vocab_size, N=2, d_model=64, d_ff=128, h=2)
    model.eval()
    
    src = torch.LongTensor([[1, 2, 3, 4, 5]])
    src_mask = torch.ones(1, 1, 5)
    
    with torch.no_grad():
        output = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
    
    assert output.dim() == 2
    assert output.size(0) == 1
    assert output.size(1) <= 10
    assert output[0, 0] == 1
    
    print(f"✓ Input: {src[0].tolist()}")
    print(f"✓ Output: {output[0].tolist()}")
    print("✓ Greedy decode test passed!")
    return True


def test_beam_search_decode():
    """Test beam search decoding."""
    print("\nTesting beam_search_decode...")
    
    vocab_size = 11
    model = make_model(vocab_size, vocab_size, N=2, d_model=64, d_ff=128, h=2)
    model.eval()
    
    src = torch.LongTensor([[1, 2, 3, 4, 5]])
    src_mask = torch.ones(1, 1, 5)
    
    with torch.no_grad():
        output = beam_search_decode(
            model, src, src_mask, 
            max_len=10, start_symbol=1, beam_size=3
        )
    
    assert output.dim() == 2
    assert output.size(0) == 1
    assert output[0, 0] == 1
    
    print(f"✓ Input: {src[0].tolist()}")
    print(f"✓ Output (beam=3): {output[0].tolist()}")
    print("✓ Beam search test passed!")
    return True


def test_greedy_vs_beam():
    """Compare greedy and beam search."""
    print("\nComparing greedy vs beam search...")
    
    vocab_size = 20
    model = make_model(vocab_size, vocab_size, N=2, d_model=64, d_ff=128, h=2)
    model.eval()
    
    torch.manual_seed(42)
    
    for i in range(3):
        src = torch.randint(1, vocab_size, (1, 8))
        src_mask = torch.ones(1, 1, 8)
        
        with torch.no_grad():
            greedy_out = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1)
            beam_out = beam_search_decode(
                model, src, src_mask, 
                max_len=10, start_symbol=1, beam_size=4
            )
        
        print(f"\n  Example {i+1}:")
        print(f"    Input:  {src[0].tolist()}")
        print(f"    Greedy: {greedy_out[0].tolist()}")
        print(f"    Beam-4: {beam_out[0].tolist()}")
    
    print("\n✓ Comparison complete!")
    return True


def test_trained_copy_task():
    """Test decoding on a trained copy task model."""
    print("\nTesting on trained copy model...")
    
    vocab_size = 11
    model = make_model(vocab_size, vocab_size, N=2, d_model=64, d_ff=128, h=2)
    
    print("  Training on copy task...")
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(30):
        data = torch.randint(1, vocab_size, (32, 8))
        data[:, 0] = 1
        
        src = data.clone()
        tgt = data.clone()
        
        src_mask = torch.ones(32, 1, 8)
        tgt_mask = subsequent_mask(7).expand(32, -1, -1)
        
        optimizer.zero_grad()
        out = model(src, tgt[:, :-1], src_mask, tgt_mask)
        loss = criterion(
            out.reshape(-1, vocab_size),
            tgt[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
    
    print(f"  Final loss: {loss.item():.4f}")
    
    model.eval()
    test_src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    test_mask = torch.ones(1, 1, 8)
    
    with torch.no_grad():
        greedy_out = greedy_decode(model, test_src, test_mask, max_len=8, start_symbol=1)
        beam_out = beam_search_decode(
            model, test_src, test_mask, 
            max_len=8, start_symbol=1, beam_size=4
        )
    
    print(f"\n  Input:  {test_src[0].tolist()}")
    print(f"  Greedy: {greedy_out[0].tolist()}")
    print(f"  Beam-4: {beam_out[0].tolist()}")
    
    greedy_match = (test_src[0][1:] == greedy_out[0][1:test_src.size(1)]).sum().item()
    beam_match = (test_src[0][1:] == beam_out[0][1:test_src.size(1)]).sum().item()
    
    print(f"\n  Greedy match: {greedy_match}/{test_src.size(1)-1}")
    print(f"  Beam match:   {beam_match}/{test_src.size(1)-1}")
    
    print("✓ Trained model test complete!")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SOLUTION 5: INFERENCE AND DECODING")
    print("=" * 60)
    
    test_greedy_decode()
    test_beam_search_decode()
    test_greedy_vs_beam()
    test_trained_copy_task()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 Congratulations! You've completed Day 14!")


if __name__ == "__main__":
    run_all_tests()
