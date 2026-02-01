# Day 13: Data Directory

This directory is for datasets and model checkpoints used in Day 13 exercises.

## Directory Structure

```
data/
├── README.md           # This file
├── wmt/               # WMT translation dataset (optional)
│   ├── train.en
│   ├── train.de
│   ├── val.en
│   └── val.de
├── iwslt/             # IWSLT smaller translation dataset
│   └── ...
└── checkpoints/       # Trained model weights
    └── ...
```

## Datasets

### For Basic Demo (No Download Needed)

The training scripts generate synthetic data:
- **Copy Task**: Learn to copy input sequence to output
- **Reverse Task**: Learn to reverse input sequence

These are perfect for testing the implementation!

### For Real Translation (Optional)

#### WMT Dataset (Large)
```bash
# English-German
wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz
tar -xzf training-parallel-nc-v9.tgz
```

#### IWSLT Dataset (Smaller)
```bash
# Recommended for quick experiments
pip install torchtext
# Use torchtext.datasets.IWSLT2016 or IWSLT2017
```

#### Multi30k (Very Small)
```python
# Great for learning - ~30k sentence pairs
# Images with captions in English and German
# pip install torchtext
from torchtext.datasets import Multi30k
```

## Tokenization

For real text data, you'll need a tokenizer:

```python
# Option 1: Simple whitespace splitting (demo only)
tokens = text.lower().split()

# Option 2: Subword tokenization (recommended)
pip install sentencepiece

# Option 3: Hugging Face tokenizers
pip install tokenizers
```

## Data Format

Expected format for custom data:

```
# source.txt
The cat sat on the mat
Hello world

# target.txt  
Die Katze sass auf der Matte
Hallo Welt
```

## Synthetic Data Generation

For learning, the synthetic tasks are best:

```python
from train_minimal import generate_copy_task_data, generate_reverse_task_data

# Copy task: input → same output
src, tgt = generate_copy_task_data(n_samples=1000, seq_len=10)

# Reverse task: input → reversed output
src, tgt = generate_reverse_task_data(n_samples=1000, seq_len=10)
```

## Model Checkpoints

After training, save your models:

```python
import numpy as np

# Save
np.savez('checkpoints/transformer_v1.npz',
         src_embedding=model.src_embedding.embedding,
         tgt_embedding=model.tgt_embedding.embedding,
         # ... other weights
         )

# Load
checkpoint = np.load('checkpoints/transformer_v1.npz')
model.src_embedding.embedding = checkpoint['src_embedding']
# ... restore other weights
```

## Quick Start

```bash
# Just run with synthetic data - no downloads needed!
cd papers/13_attention
python train_minimal.py --task copy --epochs 10
```
