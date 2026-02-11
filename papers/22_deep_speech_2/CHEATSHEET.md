# Day 22 Cheat Sheet: Deep Speech 2

## The Big Idea (30 seconds)

A single neural network (conv + bidirectional GRU + CTC) replaces the entire traditional ASR pipeline. Trained end-to-end on spectrograms and transcripts — no phoneme labels, no HMMs, no pronunciation dictionaries. Same architecture works for English and Mandarin. Deeper models (5-7 RNN layers) beat wider ones. More data helps log-linearly.

**The one picture to remember:**

```
                    ┌─────────────────────────────────────────────┐
                    │         Deep Speech 2 Pipeline              │
                    └─────────────────────────────────────────────┘

  Audio Waveform    Spectrogram      Conv       BiGRU x5-7     FC + CTC
  ~~~~~~~~~~~~    ┌───────────┐   ┌───────┐   ┌───────────┐   ┌───────┐
  ∿∿∿∿∿∿∿∿∿∿∿ -> │ freq      │-> │ 2D    │-> │ ←─────────│-> │ chars │-> "hello"
  (raw audio)     │ x time    │   │ conv  │   │ ─────────→│   │ probs │
                  │ (20ms win)│   │ x 1-3 │   │ +BatchNorm│   │       │
                  └───────────┘   └───────┘   └───────────┘   └───────┘
                                                                  │
                                                             CTC collapse:
                                                         -hh-ee-ll-l-oo-
                                                              -> "hello"
```

---

## Quick Start

```bash
cd papers/22_deep_speech_2

# Check environment
python setup.py

# Train on synthetic data
python train_minimal.py --epochs 5

# With more layers (closer to paper's best model)
python train_minimal.py --epochs 10 --n-rnn 5 --rnn-hidden 512

# Disable SortaGrad to see the difference
python train_minimal.py --epochs 5 --no-sortagrad

# Generate visualizations
python visualization.py
```

---

## Architecture at a Glance

| Layer | What It Does | Paper Reference |
|-------|-------------|-----------------|
| Spectrogram | 20ms windows, 10ms hop, log power | Section 3.1 |
| Conv2D (1-3) | Extract spectral-temporal patterns | Section 3.1 |
| Bidirectional GRU (1-7) | Sequence modeling with BatchNorm | Sections 3.1, 3.2 |
| Clipped ReLU | min(max(x,0), 20) | Section 3.1 |
| FC | Project to vocab_size | Section 3.1 |
| Softmax + CTC | Character probabilities, alignment-free loss | Section 3.1, 3.3 |

```
Audio -> Spectrogram -> [Conv->BN->ReLU] x N -> [BiGRU->BN->ReLU] x M -> FC -> Softmax -> CTC
```

---

## Key Hyperparameters

| Parameter | Paper Value | Our Default | What It Does | Tips |
|-----------|-----------|-------------|-------------|------|
| n_conv | 2-3 | 2 | Convolutional layers | 2 is good for most cases |
| n_rnn | 5-7 | 3 | Recurrent layers | More depth = better WER (Section 3.5) |
| rnn_hidden | 1024-2048 | 256 | Hidden size per direction | Scale with data size |
| rnn_type | GRU | GRU | Cell type | GRU trains faster than simple RNN (Section 3.1) |
| window_ms | 20 | 20 | Spectrogram window | Paper standard |
| stride_ms | 10 | 10 | Spectrogram hop | Paper standard |
| clip_value | 20 | 20 | ReLU clipping threshold | From paper (Section 3.1) |
| learning_rate | annealed | 3e-4 (Adam) | Optimizer LR | Paper uses SGD+momentum; Adam converges faster on small data |
| batch_size | varies | 8 | Batch size | Larger = more stable gradients |
| SortaGrad | first epoch | first epoch | Sort by length in epoch 0 | Stabilizes early training (Section 3.1) |

---

## Common Issues and Fixes

### CTC Loss Explodes or Returns Inf

```python
# Problem: output sequences shorter than target sequences after conv downsampling
# Fix: check that output_lengths > target_lengths for every sample

loss = F.ctc_loss(
    log_probs, targets, output_lengths, target_lengths,
    blank=0, reduction='mean',
    zero_infinity=True  # clamp inf losses to 0
)
```

### Model Outputs Only Blanks

```python
# Problem: model collapses to always predicting blank
# Fix 1: reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fix 2: use SortaGrad (start with short, easy examples)
# Fix 3: check gradient clipping is not too aggressive
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400)
```

### Spectrogram Has Wrong Shape

```python
# Problem: n_freq depends on n_fft
# n_fft=512 -> n_freq=257 (n_fft // 2 + 1)
# n_fft=256 -> n_freq=129
spec = compute_spectrogram(audio, n_fft=512)
print(f"Spectrogram shape: {spec.shape}")  # (n_freq, n_time)
```

### Variable-Length Sequences Cause Errors

```python
# Problem: RNN and CTC need to know actual lengths
# Fix: always pass feature_lengths through the model
log_probs, output_lengths = model(features, feature_lengths)
# output_lengths accounts for downsampling by conv layers
```

### Greedy Decode Returns Garbage

```python
# Problem: model is undertrained or wrong character encoding
# Fix: verify the encoder roundtrips correctly
encoder = CharEncoder('english')
text = "hello"
encoded = encoder.encode(text)
decoded = encoder.decode(encoded)
assert decoded == text, f"Roundtrip failed: {decoded}"
```

---

## The Math (Copy-Paste Ready)

### Clipped ReLU (Section 3.1)

```python
def clipped_relu(x, clip=20.0):
    """sigma(x) = min(max(x, 0), 20)"""
    return torch.clamp(x, min=0.0, max=clip)
```

### Spectrogram

```python
import torch

def log_spectrogram(audio, n_fft=512, hop=160, win=320):
    """Log power spectrogram, Section 3.1."""
    window = torch.hann_window(win, device=audio.device)
    spec = torch.stft(audio, n_fft, hop, win, window, return_complex=True)
    log_power = torch.log(spec.abs().pow(2) + 1e-10)
    return (log_power - log_power.mean()) / (log_power.std() + 1e-10)
```

### CTC Greedy Decode

```python
def ctc_greedy_decode(log_probs, blank_idx=0):
    """Collapse repeated characters and remove blanks."""
    argmax = log_probs.argmax(dim=-1)  # (time,)
    result = []
    prev = None
    for idx in argmax.tolist():
        if idx != prev:
            if idx != blank_idx:
                result.append(idx)
        prev = idx
    return result
```

### Word Error Rate

```python
def wer(ref_words, hyp_words):
    """Levenshtein distance at word level / number of reference words."""
    # Standard dynamic programming edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / max(n, 1)
```

---

## Sequence-wise BatchNorm (Section 3.2)

```python
# Standard BatchNorm: stats per feature, per timestep
# Sequence-wise BatchNorm: stats per feature, over (batch * time)

def seq_batchnorm(x, weight, bias, eps=1e-5):
    """x: (batch, time, features)"""
    flat = x.reshape(-1, x.size(-1))  # (batch*time, features)
    mean = flat.mean(dim=0)
    var = flat.var(dim=0, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias
```

---

## Debugging Checklist

- [ ] Spectrogram shape is (n_freq, n_time), not transposed
- [ ] output_lengths > target_lengths for every sample in the batch
- [ ] CTC blank is index 0
- [ ] Log softmax is applied before CTC loss (not raw logits)
- [ ] Gradient clipping is active (max_norm ~400)
- [ ] SortaGrad is ON for epoch 0
- [ ] BatchNorm is sequence-wise, not per-timestep
- [ ] Character encoder roundtrips correctly: encode(decode(x)) == x
- [ ] Bidirectional RNN output has size hidden_dim * 2

---

## Paper Results Quick Reference

| Metric | Value | Source |
|--------|-------|--------|
| English WER (regular, 12K hr) | 8.46% | Table 4 |
| English WER (noisy, 12K hr) | 13.59% | Table 4 |
| Best architecture | 2 conv + 7 biGRU | Section 3.5 |
| Mandarin: beat humans on | 3/4 test sets | Section 3.7 |
| Training speedup | 7x over DS1 | Section 1 |
| English alphabet | 29 symbols | Section 3.1 |

---

## Next: Day 23 — Variational Lossy Autoencoder
