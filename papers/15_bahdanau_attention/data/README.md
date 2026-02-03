# Data Directory

This folder contains data for the Bahdanau Attention experiments.

## Toy Dataset: Sequence Reversal

For learning attention, we use a synthetic **sequence reversal** task:

```
Input:  [5, 3, 8, 2, 1]
Output: [1, 2, 8, 3, 5]
```

### Why This Task?

1. **Easy to verify** - You can check correctness instantly
2. **Clear attention pattern** - Should form a reversed diagonal
3. **No external data needed** - Generated on-the-fly
4. **Fast training** - Converges in ~30 epochs

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| PAD   | 0   | Padding for batching |
| SOS   | 1   | Start of sequence |
| EOS   | 2   | End of sequence |
| Data  | 3+  | Actual sequence values |

## Real-World Datasets

For actual machine translation, consider:

### Small (For Learning)
- **Multi30k** - 30k English-German sentence pairs
- **Tatoeba** - Community-translated sentences
- **IWSLT** - TED talk translations

### Large (For Research)
- **WMT** - Workshop on Machine Translation datasets
- **Europarl** - European Parliament proceedings
- **OpenSubtitles** - Movie subtitles

## Generating Data

The training script generates data automatically:

```python
from train import ReversalDataset

# Create dataset
dataset = ReversalDataset(
    num_samples=5000,
    min_len=4,
    max_len=10,
    vocab_size=50
)

# Get a sample
src, trg = dataset[0]
print(f"Source: {src.tolist()}")
print(f"Target: {trg.tolist()}")
```

## Data Format

Each sample is a tuple of:
- `src`: Source sequence tensor (without special tokens)
- `trg`: Target sequence tensor (with SOS and EOS)

The collate function handles:
- Padding to equal lengths
- Creating source length tensor for packing
