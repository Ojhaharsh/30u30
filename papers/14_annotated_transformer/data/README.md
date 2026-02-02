# Day 14: Data Directory

This directory is for datasets used in Day 14 exercises.

## Synthetic Data

For the copy/reverse tasks, data is generated programmatically:
- Copy task: Input sequence copied to output
- Reverse task: Input sequence reversed
- Sort task: Input sequence sorted

## Real Translation Data (Optional)

For actual machine translation training:

### Multi30k (Small, Good for Learning)
- English-German parallel corpus
- ~30k sentence pairs
- Download: https://github.com/multi30k/dataset

### WMT Datasets (Large, Research-Grade)
- WMT14 English-German: http://www.statmt.org/wmt14/
- WMT14 English-French: http://www.statmt.org/wmt14/

### IWSLT (Medium)
- TED talks parallel corpus
- Good for intermediate experiments
- https://wit3.fbk.eu/

## Data Format

Expected format for training:
```
# source.txt (one sentence per line)
The cat sat on the mat.
Hello world.

# target.txt (parallel translations)
Le chat était assis sur le tapis.
Bonjour le monde.
```

## Note

Large datasets are not included in the repository.
Use the synthetic data generators for quick experiments.
