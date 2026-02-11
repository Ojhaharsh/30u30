# Exercises: Deep Speech 2

Five exercises that build up the Deep Speech 2 pipeline from individual components to a complete end-to-end speech recognizer.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | Spectrogram feature extraction | Easy (2/5) | 30 min |
| 2 | CTC loss and greedy decoding | Medium (3/5) | 45 min |
| 3 | RNN with sequence-wise BatchNorm | Medium (3/5) | 45 min |
| 4 | SortaGrad curriculum learning | Easy (2/5) | 20 min |
| 5 | End-to-end DS2 pipeline | Hard (4/5) | 60 min |

## Exercise 1: Spectrogram Feature Extraction

Build the function that converts raw audio waveforms into log power spectrograms — the input representation for DS2. You will implement windowing, STFT, power computation, and normalization.

**What you will learn:** How audio becomes a 2D representation suitable for convolutional processing.

## Exercise 2: CTC Loss and Greedy Decoding

Implement CTC greedy decoding (collapse repeated characters, remove blanks) and wire up PyTorch's CTC loss for training. You will also build a character encoder that maps between text and integer indices.

**What you will learn:** How CTC handles the alignment problem without frame-level labels.

## Exercise 3: RNN with Sequence-wise BatchNorm

Build a bidirectional GRU layer with sequence-wise batch normalization (Section 3.2). You will implement the normalization where stats are computed over (batch x time), not per-timestep.

**What you will learn:** Why standard BatchNorm fails for variable-length sequences and what the paper does instead.

## Exercise 4: SortaGrad Curriculum Learning

Implement the SortaGrad strategy: sort training samples by utterance length in the first epoch, then use random order for subsequent epochs. Compare training stability with and without SortaGrad.

**What you will learn:** How curriculum learning stabilizes early training for sequence models.

## Exercise 5: End-to-End DS2 Pipeline

Wire together the spectrogram, convolutional front-end, recurrent layers, and CTC decoder into a complete Deep Speech 2 model. Train it on synthetic data and evaluate with WER.

**What you will learn:** How all the components fit together into a working speech recognizer.

## How to Use

1. Read the exercise file — each has detailed instructions in the docstrings
2. Find the TODO sections — these are what you implement
3. Run the test function at the bottom of each file
4. Check solutions — compare with `solutions/solution_X.py`

## Tips

- Start with Exercise 1 (spectrogram) — it is standalone and does not depend on the others.
- Exercise 5 builds on concepts from 1-4, so do those first.
- If CTC loss returns inf, check that output_lengths > target_lengths.
- Use `python setup.py` to verify your environment before starting.

## Common Issues

- **torch.stft requires return_complex=True** in recent PyTorch versions.
- **Variable-length sequences** need explicit length tracking through the pipeline.
- **CTC blank index** must be 0 (PyTorch convention).
