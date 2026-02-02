# Exercise Solutions

Complete solutions for all Day 14 exercises.

## Files

| Solution | Exercise | Description |
|----------|----------|-------------|
| `solution_01_attention.py` | Attention with Masking | Scaled dot-product attention + subsequent mask |
| `solution_02_multihead.py` | Multi-Head Attention | Complete nn.Module with view/transpose |
| `solution_03_encoder.py` | Encoder Stack | LayerNorm, FFN, EncoderLayer, Encoder |
| `solution_04_training.py` | Training Pipeline | Batch, LabelSmoothing, NoamOpt |
| `solution_05_inference.py` | Inference | Greedy and beam search decoding |

## Usage

```bash
# Run any solution to verify it works
python solution_01_attention.py

# Compare with your exercise
diff exercise_01_attention.py solutions/solution_01_attention.py
```

## Tips

1. **Try the exercise first** - Only look at solutions when stuck
2. **Compare approaches** - Your solution may differ but still be correct
3. **Run the tests** - Each solution includes its own test suite
