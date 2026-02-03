# Day 15: Neural Machine Translation by Jointly Learning to Align and Translate

> *"Neural Machine Translation by Jointly Learning to Align and Translate"* - Bahdanau, Cho, Bengio (2014)

**📖 Original Paper:** https://arxiv.org/abs/1409.0473

**⏱️ Time to Complete:** 4-5 hours

**🎯 What You'll Learn:**
- The original attention mechanism (before Transformers existed!)
- Why vanilla Seq2Seq fails on long sequences
- How "alignment" solves the bottleneck problem
- Implementing additive attention from scratch
- The bridge from RNNs to modern Transformers

---

## 🧠 The Big Idea

**In one sentence:** Instead of compressing an entire sentence into one fixed vector, let the decoder "look back" at different parts of the input at each step.

### The Bottleneck Problem

Before Bahdanau, Seq2Seq models looked like this:

```
Encoder:  "The cat sat on the mat" → [single fixed vector] → Decoder: "Le chat..."
```

**The problem:** That single vector must encode EVERYTHING. For long sentences, information gets lost—like trying to describe a whole movie in one tweet.

### The Bahdanau Solution

```
                    ┌─────────────────────────────────────┐
                    │     Which input words matter NOW?   │
                    └─────────────────────────────────────┘
                                      ↓
Encoder:  "The cat sat on the mat" → [h₁, h₂, h₃, h₄, h₅, h₆]
                                          ↑   ↑   ↑
                                    Attention weights (different each step!)
                                          ↓   ↓   ↓
                                      Context vector (weighted sum)
                                              ↓
Decoder:                            "Le" → "chat" → "assis" → ...
```

**The key insight:** Different output words need different input words. "chat" should focus on "cat", not "mat".

---

## 🎯 The Attention Mechanism (v1.0)

### 🎭 Fun Analogies to Understand Attention

**Analogy 1: The Spotlight on Stage 🔦**

Imagine you're at a theater watching a play:
- **Without attention:** The whole stage is lit equally—you see everything but focus on nothing
- **With attention:** A spotlight follows the important actor in each scene

The decoder is like a spotlight operator who knows exactly which part of the input "stage" to illuminate for each output word!

---

**Analogy 2: The Open-Book Exam 📖**

Think about taking an exam:
- **Without attention (Seq2Seq):** You read the textbook once, close it, then answer ALL questions from memory. Good luck with a 500-page book! 😱
- **With attention:** It's an open-book exam! For each question, you can flip back and look at the relevant pages. Much easier!

The attention weights are literally telling you: "For this question, pages 42-45 are most helpful."

---

**Analogy 3: The Google Translator at a Party 🎉**

Imagine you're translating for someone at an international party:
- **Without attention:** Your friend whispers a 10-minute story in French, THEN you try to repeat it in English. By the end, you've forgotten the beginning!
- **With attention:** After each sentence your friend says, you glance back at your notes: "Wait, who was Marie again? Oh right, the cousin from Paris!" 

---

### 🚶 Step-by-Step: Translating "The black cat sat on the mat"

Let's walk through a real translation to see attention in action!

**Source:** "The black cat sat on the mat" (English)
**Target:** "Le chat noir était assis sur le tapis" (French)

```
Step 1: Decoder wants to generate "Le" (The)
┌────────────────────────────────────────────────────────┐
│  The   black   cat    sat    on    the    mat         │
│  0.6   0.1    0.1    0.05   0.05  0.05   0.05         │
│  ^^^                                                   │
│  Spotlight shines on "The" → Output: "Le"             │
└────────────────────────────────────────────────────────┘

Step 2: Decoder wants to generate "chat" (cat)
┌────────────────────────────────────────────────────────┐
│  The   black   cat    sat    on    the    mat         │
│  0.05  0.15   0.65   0.05   0.05  0.02   0.03         │
│               ^^^^                                     │
│  Spotlight jumps to "cat" → Output: "chat"            │
└────────────────────────────────────────────────────────┘

Step 3: Decoder wants to generate "noir" (black)
┌────────────────────────────────────────────────────────┐
│  The   black   cat    sat    on    the    mat         │
│  0.05  0.70   0.15   0.03   0.02  0.02   0.03         │
│        ^^^^^                                           │
│  Spotlight swings to "black" → Output: "noir"         │
│  (Notice: French puts adjective AFTER noun!)          │
└────────────────────────────────────────────────────────┘

Step 4: Decoder wants to generate "assis" (sat)
┌────────────────────────────────────────────────────────┐
│  The   black   cat    sat    on    the    mat         │
│  0.02  0.05   0.10   0.70   0.08  0.02   0.03         │
│                      ^^^^                              │
│  Spotlight on "sat" → Output: "était assis"           │
└────────────────────────────────────────────────────────┘
```

**🔑 Key Insight:** Notice how the attention "jumps around" based on what's needed, not just left-to-right! This is the magic of alignment.

---

### The Math (Additive Attention)

For each decoder step $t$:

**Step 1: Compute alignment scores**
$$e_{t,i} = v^T \tanh(W_s \cdot s_{t-1} + W_h \cdot h_i)$$

Where:
- $s_{t-1}$ = previous decoder hidden state
- $h_i$ = encoder hidden state at position $i$
- $W_s, W_h, v$ = learnable parameters

**Step 2: Normalize to get attention weights**
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

**Step 3: Compute context vector**
$$c_t = \sum_i \alpha_{t,i} \cdot h_i$$

**Step 4: Generate output**
$$s_t = \text{GRU}(s_{t-1}, [y_{t-1}; c_t])$$
$$y_t = \text{softmax}(W_o \cdot s_t)$$

---

## 🔄 Why "Additive" Attention?

There are two main attention variants:

| Type | Formula | Paper |
|------|---------|-------|
| **Additive** (Bahdanau) | $v^T \tanh(W_s s + W_h h)$ | This paper (2014) |
| **Multiplicative** (Luong) | $s^T W h$ or $s^T h$ | Luong et al. (2015) |
| **Scaled Dot-Product** | $\frac{QK^T}{\sqrt{d_k}}$ | Transformer (2017) |

Bahdanau uses **additive** because:
1. More expressive (non-linear combination)
2. Works well with different-sized vectors
3. Was state-of-the-art at the time

Later, scaled dot-product became standard (faster on GPUs).

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER (Bidirectional GRU)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    x₁ ──→ [h₁→] ──→ [h₂→] ──→ [h₃→] ──→ [h₄→]             │
│           [h₁←] ←── [h₂←] ←── [h₃←] ←── [h₄←] ←── x₄      │
│              ↓         ↓         ↓         ↓               │
│           [h₁]      [h₂]      [h₃]      [h₄]              │
│           (concat forward + backward)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ATTENTION MECHANISM                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   s_{t-1} ──→ [Alignment Model] ←── h₁, h₂, h₃, h₄        │
│                      ↓                                      │
│               α₁, α₂, α₃, α₄  (attention weights)          │
│                      ↓                                      │
│               c_t = Σ αᵢ · hᵢ  (context vector)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DECODER (GRU)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [y_{t-1}; c_t] ──→ GRU ──→ s_t ──→ softmax ──→ y_t      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Bidirectional Encoder?

The encoder reads the input **both ways**:
- Forward: "The cat sat" (left-to-right context)
- Backward: "sat cat The" (right-to-left context)

This way, each $h_i$ contains information about the ENTIRE sentence, not just words before it.

---

## 💻 Implementation

### Complete Bahdanau Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Additive attention mechanism from Bahdanau et al. (2014).
    
    Computes: e = v^T * tanh(W_s * s + W_h * h)
    """
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super().__init__()
        key_size = key_size or hidden_size
        query_size = query_size or hidden_size
        
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, query, keys, mask=None):
        """
        Args:
            query: Decoder hidden state [batch, query_size]
            keys: Encoder outputs [batch, seq_len, key_size]
            mask: Boolean mask [batch, seq_len] (True = ignore)
            
        Returns:
            context: Weighted sum of keys [batch, key_size]
            attention_weights: [batch, seq_len]
        """
        # Project query and keys to same dimension
        # query: [batch, hidden] -> [batch, 1, hidden]
        query = self.query_layer(query).unsqueeze(1)
        
        # keys: [batch, seq_len, key_size] -> [batch, seq_len, hidden]
        keys_proj = self.key_layer(keys)
        
        # Compute alignment scores
        # [batch, seq_len, hidden]
        scores = torch.tanh(query + keys_proj)
        
        # [batch, seq_len, 1] -> [batch, seq_len]
        scores = self.energy_layer(scores).squeeze(-1)
        
        # Apply mask (set masked positions to -inf before softmax)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Normalize to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector
        # [batch, 1, seq_len] @ [batch, seq_len, key_size] -> [batch, key_size]
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attention_weights
```

### Encoder with Bidirectional GRU

```python
class Encoder(nn.Module):
    """Bidirectional GRU encoder."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(
            embed_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lengths):
        """
        Args:
            src: Source token ids [batch, src_len]
            src_lengths: Lengths of each sequence [batch]
            
        Returns:
            outputs: [batch, src_len, 2*hidden_size]
            hidden: [num_layers, batch, hidden_size]
        """
        embedded = self.dropout(self.embedding(src))
        
        # Pack for efficiency with variable lengths
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, hidden = self.gru(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Combine bidirectional hidden states
        # hidden: [2*num_layers, batch, hidden] -> [num_layers, batch, hidden]
        hidden = self._combine_bidirectional(hidden)
        
        return outputs, hidden
    
    def _combine_bidirectional(self, hidden):
        """Combine forward and backward hidden states."""
        # hidden: [num_layers*2, batch, hidden]
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, -1, hidden.size(-1))
        # Sum forward and backward
        hidden = hidden.sum(dim=1)
        return hidden
```

### Decoder with Attention

```python
class AttentionDecoder(nn.Module):
    """GRU decoder with Bahdanau attention."""
    
    def __init__(self, vocab_size, embed_size, hidden_size, 
                 encoder_hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Attention over encoder outputs (bidirectional = 2x hidden)
        self.attention = BahdanauAttention(
            hidden_size=hidden_size,
            key_size=encoder_hidden_size * 2,  # Bidirectional
            query_size=hidden_size
        )
        
        # GRU input: embedding + context
        self.gru = nn.GRU(
            embed_size + encoder_hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_layer = nn.Linear(
            hidden_size + encoder_hidden_size * 2 + embed_size,
            vocab_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward_step(self, prev_token, hidden, encoder_outputs, src_mask=None):
        """Single decoding step."""
        # Embed previous token
        embedded = self.dropout(self.embedding(prev_token))  # [batch, embed]
        
        # Compute attention
        query = hidden[-1]  # Last layer hidden state
        context, attn_weights = self.attention(query, encoder_outputs, src_mask)
        
        # GRU input: [embedding; context]
        gru_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        
        # GRU step
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)  # [batch, hidden]
        
        # Output: combine all information
        output = torch.cat([output, context, embedded], dim=-1)
        output = self.output_layer(output)  # [batch, vocab_size]
        
        return output, hidden, attn_weights
    
    def forward(self, trg, hidden, encoder_outputs, src_mask=None):
        """
        Full sequence decoding (teacher forcing).
        
        Args:
            trg: Target tokens [batch, trg_len]
            hidden: Initial decoder hidden state
            encoder_outputs: [batch, src_len, enc_hidden*2]
            src_mask: [batch, src_len]
            
        Returns:
            outputs: [batch, trg_len, vocab_size]
            attentions: [batch, trg_len, src_len]
        """
        batch_size, trg_len = trg.size()
        
        outputs = []
        attentions = []
        
        # First input is <sos> token
        prev_token = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, attn = self.forward_step(
                prev_token, hidden, encoder_outputs, src_mask
            )
            outputs.append(output)
            attentions.append(attn)
            
            # Teacher forcing: use ground truth
            prev_token = trg[:, t]
        
        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)
        
        return outputs, attentions
```

### Complete Seq2Seq Model

```python
class Seq2SeqAttention(nn.Module):
    """Complete Seq2Seq model with Bahdanau attention."""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, src_lengths, trg, src_mask=None):
        """
        Args:
            src: Source tokens [batch, src_len]
            src_lengths: Source lengths [batch]
            trg: Target tokens [batch, trg_len]
            src_mask: Padding mask [batch, src_len]
            
        Returns:
            outputs: [batch, trg_len-1, vocab_size]
            attentions: [batch, trg_len-1, src_len]
        """
        # Encode source
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Decode with attention
        outputs, attentions = self.decoder(trg, hidden, encoder_outputs, src_mask)
        
        return outputs, attentions
    
    def translate(self, src, src_lengths, max_len=50, sos_idx=2, eos_idx=3):
        """Greedy decoding for inference."""
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            
            # Start with <sos>
            prev_token = torch.full((batch_size,), sos_idx, device=device)
            
            translations = []
            attentions = []
            
            for _ in range(max_len):
                output, hidden, attn = self.decoder.forward_step(
                    prev_token, hidden, encoder_outputs
                )
                
                # Greedy selection
                prev_token = output.argmax(dim=-1)
                translations.append(prev_token)
                attentions.append(attn)
                
                # Stop if all sequences produced <eos>
                if (prev_token == eos_idx).all():
                    break
            
            translations = torch.stack(translations, dim=1)
            attentions = torch.stack(attentions, dim=1)
            
        return translations, attentions
```

---

## 📊 Visualizing Attention

One of the most beautiful things about attention is that it's **interpretable**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention, src_tokens, trg_tokens, figsize=(10, 10)):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention: [trg_len, src_len] attention weights
        src_tokens: List of source tokens
        trg_tokens: List of target tokens
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        attention.cpu().numpy(),
        xticklabels=src_tokens,
        yticklabels=trg_tokens,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_xlabel('Source (Input)')
    ax.set_ylabel('Target (Output)')
    ax.set_title('Bahdanau Attention Weights')
    
    plt.tight_layout()
    return fig

# Example usage:
# plot_attention(attentions[0], ["The", "cat", "sat"], ["Le", "chat", "assis"])
```

What you should see:
- Diagonal-ish pattern (word order often similar)
- "Le" attends to "The"
- "chat" attends strongly to "cat"
- Some words attend to multiple source words

---

## 🔬 The Historical Context

### Timeline of Attention

```
2014: Bahdanau Attention (THIS PAPER)
      ↓ "Attention solves the bottleneck!"
      
2015: Luong Attention (simpler, faster variants)
      ↓ "Dot-product works too!"
      
2016: Attention everywhere (image captioning, speech, etc.)
      ↓ "Attention is a general mechanism!"
      
2017: Transformer (Attention Is All You Need)
      ↓ "We don't even need RNNs!"
      
2018-now: GPT, BERT, and the modern era
```

Bahdanau attention is the **grandfather** of all modern attention mechanisms. Without this paper, there would be no ChatGPT.

---

## 🎓 Key Takeaways

### What Made This Paper Revolutionary

1. **Solved a real problem**: Long sequences were impossible before this
2. **Interpretable**: You can SEE what the model focuses on
3. **Elegant**: Simple idea, huge impact
4. **General**: Works for any sequence-to-sequence task

### The Core Insight

> "Don't force the entire input through a bottleneck. Let the decoder dynamically access what it needs."

This is the fundamental idea behind ALL modern attention mechanisms, including the Transformer's self-attention.

---

## 🔗 Connection to the Transformer

| Bahdanau (2014) | Transformer (2017) |
|-----------------|-------------------|
| RNN encoder | Self-attention encoder |
| RNN decoder | Self-attention decoder |
| Additive attention | Scaled dot-product attention |
| Sequential processing | Parallel processing |
| ~1 attention per step | Multi-head attention |

The Transformer took Bahdanau's insight ("attention is useful") and asked: "What if we use attention for EVERYTHING?"

---

## 📁 Files in This Directory

```
15_bahdanau_attention/
├── README.md                 # This file
├── implementation.py         # Complete PyTorch implementation
├── train.py                  # Training script
├── exercises/
│   ├── exercise_1.py        # Implement basic additive attention
│   ├── exercise_2.py        # Build the encoder
│   ├── exercise_3.py        # Build the attention decoder
│   ├── exercise_4.py        # Train on a toy dataset
│   └── exercise_5.py        # Visualize attention patterns
├── solutions/
│   ├── solution_1.py
│   ├── solution_2.py
│   ├── solution_3.py
│   ├── solution_4.py
│   └── solution_5.py
└── requirements.txt
```

---

## 🚀 Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run training on toy data
python train.py

# Complete exercises
cd exercises
python exercise_1.py  # Start here!
```

---

## 📚 References

- [Original Paper](https://arxiv.org/abs/1409.0473) - Bahdanau et al. (2014)
- [Luong Attention](https://arxiv.org/abs/1508.04025) - Effective Approaches to Attention-based NMT
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The Transformer
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar's visual guide

---

*"Attention is all you need... but first, you needed Bahdanau to invent it."* 🎯
