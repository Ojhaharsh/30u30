# NTM Copy Task Data

The data for the Neural Turing Machine (NTM) experiments is generated procedurally. In the [train.py](../train.py) script, the `generate_copy_task` function creates random bit sequences for the model to learn to copy.

## Sequence Format
For a given width $W$ (default 8) and sequence length $L$:
1. **Input Phase:**
   - $L$ vectors of bits $\{0, 1\}^W$.
   - A delimiter vector (e.g., all 0s with a 1 at index $W+1$).
2. **Output Phase:**
   - $L$ vectors of 0s (model is expected to output the copied sequence here).

## Directory Purpose
This directory is reserved for saving trained weights or any extracted features if you wish to run larger-scale experiments beyond the provided training script.
