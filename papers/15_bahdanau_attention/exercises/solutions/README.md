# Solutions

Complete solutions for all exercises. **Try the exercises first.**

## Files

| Solution | Exercise | Description |
|----------|----------|-------------|
| `solution_1.py` | Additive Attention | Complete `AdditiveAttention` class |
| `solution_2.py` | Bidirectional Encoder | Complete `BidirectionalEncoder` class |
| `solution_3.py` | Attention Decoder | Complete `AttentionDecoder` class |
| `solution_4.py` | Training Loop | Full training script that achieves >90% |
| `solution_5.py` | Visualization | Attention heatmaps and analysis |

## Running Solutions

```bash
# Test individual components
python solution_1.py  # Tests attention
python solution_2.py  # Tests encoder
python solution_3.py  # Tests decoder

# Full training
python solution_4.py  # Trains and evaluates

# Visualization
python solution_5.py --demo  # Demo with synthetic data
```

## Expected Results

After running `solution_4.py`:

```
Device: cuda (or cpu)
Creating datasets...
Creating model...
Parameters: ~150,000

Training...
==================================================
Epoch  5 | Train: 1.2345 | Val: 1.1234 | Acc: 45%
Epoch 10 | Train: 0.5678 | Val: 0.4567 | Acc: 72%
Epoch 15 | Train: 0.2345 | Val: 0.2123 | Acc: 85%
Epoch 20 | Train: 0.1234 | Val: 0.1345 | Acc: 91%
Epoch 25 | Train: 0.0789 | Val: 0.0912 | Acc: 94%
Epoch 30 | Train: 0.0567 | Val: 0.0678 | Acc: 96%
==================================================

Final Accuracy: 96%
Success: Model achieves >90% accuracy.

Examples:
  Input:  [5, 3, 8, 2, 1]
  Output: [1, 2, 8, 3, 5]
  Target: [1, 2, 8, 3, 5]
```

## Key Implementation Details

### Solution 1: Attention
- Uses `nn.Linear` without bias for cleaner math
- Xavier initialization for stable training
- Proper masking with `-inf` before softmax

### Solution 2: Encoder
- Packs sequences for efficient RNN processing
- Projects bidirectional output to single hidden size
- Handles variable-length sequences correctly

### Solution 3: Decoder
- Attention computed BEFORE GRU step (Bahdanau style)
- Combines hidden + context + embedding for output
- Supports both step-by-step and full sequence modes

### Solution 4: Training
- CrossEntropyLoss with `ignore_index=0` for padding
- Gradient clipping with `max_norm=1.0`
- Greedy decoding for inference

### Solution 5: Visualization
- Uses matplotlib for heatmaps
- Computes attention entropy (lower = more focused)
- Checks for reversed diagonal pattern

## Learning Points

After completing all solutions, you should understand:

1. **Why attention works** - Dynamic focus on relevant input parts
2. **The alignment problem** - Matching output positions to input positions
3. **Bidirectional encoding** - Context from both directions
4. **Teacher forcing** - Using ground truth during training
5. **Attention visualization** - Interpreting what the model learned

## Next Steps

Once you've mastered Bahdanau attention:

1. **Try different tasks** - Sorting, copying, arithmetic
2. **Scale up** - Real translation datasets
3. **Compare** - Implement Luong attention (dot-product)
4. **Move forward** - Study the Transformer (self-attention)

That completes Day 15.
