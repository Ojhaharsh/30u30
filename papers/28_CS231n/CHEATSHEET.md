# Day 28 Cheat Sheet: CNNs for Visual Recognition

Quick reference for building and debugging convolutional neural networks.

## The Big Idea (30 seconds)

CNNs exploit spatial structure in images by using small, shared filters instead of full connectivity. Three ideas make this work: local receptive fields (each neuron sees a small patch), parameter sharing (same filter applied everywhere), and pooling (progressive spatial downsampling). Stack these into deep hierarchies and CNNs learn to detect edges, then textures, then objects.

---

## Quick Start

```bash
python setup.py                          # Check environment
python train_minimal.py --epochs 5       # Train on CIFAR-10
python visualization.py                  # Generate filter/activation visualizations
```

---

## Output Size Formula

For CONV layers:
```
output_size = (W - F + 2P) / S + 1

W = input size
F = filter size
S = stride
P = zero-padding
```

For POOL layers:
```
output_size = (W - F) / S + 1
```

### Quick Examples

| Input | Filter | Stride | Padding | Output |
|-------|--------|--------|---------|--------|
| 32 | 3 | 1 | 1 | 32 |
| 32 | 5 | 1 | 2 | 32 |
| 32 | 3 | 2 | 1 | 16 |
| 224 | 7 | 2 | 3 | 112 |
| 224 | 11 | 4 | 0 | 55 |
| 56 | 2 (pool) | 2 | 0 | 28 |

---

## Key Hyperparameters

| Parameter | Typical Values | What It Does | Tips |
|-----------|---------------|-------------|------|
| Filter size (F) | 3x3, 5x5 (first layer: 7x7 or 11x11) | Receptive field of each neuron | Prefer 3x3 — stack multiple for larger RF |
| Stride (S) | 1 (CONV), 2 (POOL or first CONV) | Step size when sliding | Stride 1 preserves spatial size |
| Padding (P) | (F-1)/2 for "same" output | Zeros added to borders | Use to preserve spatial dimensions |
| Num filters (K) | 32, 64, 128, 256, 512 | Depth of output volume | Double when spatial dims halve |
| Pool size | 2x2, stride 2 | Downsampling | Discards 75% of activations |

---

## Parameter Count Quick Reference

Per CONV layer: `(F * F * D_in + 1) * K`
- F = filter size, D_in = input depth, K = num filters

| Layer | Input Depth | Filters | Filter Size | Parameters |
|-------|-------------|---------|-------------|-----------|
| CONV1 | 3 (RGB) | 64 | 3x3 | (3x3x3+1)x64 = 1,792 |
| CONV2 | 64 | 128 | 3x3 | (3x3x64+1)x128 = 73,856 |
| CONV3 | 128 | 256 | 3x3 | (3x3x128+1)x256 = 295,168 |
| FC | 7x7x512 | 4096 | — | 7x7x512x4096 = 102,760,448 |

Note: FC layers dominate parameter count. VGGNet's first FC = 74% of total params.

---

## Architecture Cheat Sheet

```
Standard pattern:
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

Common stacks:
  Simple:     CONV -> RELU -> POOL (repeat)
  VGG-style:  CONV -> RELU -> CONV -> RELU -> POOL (repeat)
  Modern:     CONV -> BN -> RELU (repeat, use stride for downsampling)
```

| Architecture | Params | Layers | Key Feature |
|-------------|--------|--------|------------|
| LeNet | ~60K | 5 | First CNN |
| AlexNet | 60M | 8 | GPUs + ReLU + Dropout |
| VGGNet | 138M | 16 | All 3x3 convs |
| GoogLeNet | 4M | 22 | Inception module |
| ResNet-50 | 25.6M | 50 | Skip connections |
| ResNet-152 | 60.2M | 152 | Deeper skip connections |

---

## Common Issues and Fixes

### Loss explodes
- Reduce learning rate (try 10x smaller)
- Add gradient clipping
- Check input normalization (should be zero-mean, unit variance)

### Loss doesn't decrease
- Learning rate too low (try 10x larger)
- Check data pipeline (are labels shuffled correctly?)
- Verify output size formula — mismatched dimensions crash silently in some frameworks

### Overfitting (train acc high, val acc low)
- Add dropout before FC layers
- Add data augmentation (flips, crops, color jitter)
- Reduce model size or add weight decay

### OOM (out of memory)
- Reduce batch size
- Use smaller input size
- Replace FC layers with global average pooling
- Use mixed precision (float16)

---

## Debugging Checklist

- [ ] Output sizes work out (no fractional dimensions from (W-F+2P)/S+1)
- [ ] Input data normalized (zero mean, unit variance per channel)
- [ ] Labels are correct (visualize a few images with their labels)
- [ ] Learning rate is reasonable (start with 0.01 for SGD, 0.001 for Adam)
- [ ] Loss decreases on the first few batches
- [ ] Gradients are not all zeros (check with .grad)
- [ ] Batch size fits in GPU memory
- [ ] Random seed set for reproducibility

---

## Pro Tips

1. **Start with a proven architecture.** Don't design your own CNN from scratch. Use ResNet or EfficientNet and fine-tune. CS231n's Karpathy: "Don't be a hero."
2. **Overfit a single batch first.** Before training on the full dataset, verify your model can memorize 1-2 batches. If it can't, there's a bug.
3. **3x3 filters are almost always right.** Only use larger filters (5x5, 7x7) on the very first layer, if at all.
4. **Double depth when halving spatial size.** Going from [56x56x128] to [28x28x256] keeps the total "information capacity" roughly constant.
5. **Watch memory, not just params.** VGGNet has 138M params but uses 93 MB per image. Early layers are memory-heavy, late layers are param-heavy.
6. **Padding = (F-1)/2 preserves spatial size.** With stride 1, this means CONV layers don't shrink your feature maps. Only POOL layers downsample.
7. **Batch norm helps everything.** Put it after CONV and before ReLU. It stabilizes training and acts as regularization.
8. **Data augmentation is free regularization.** Random crops, horizontal flips, and color jitter significantly reduce overfitting.

---

## File Reference

| File | Use It For |
|------|-----------|
| `implementation.py` | Understanding conv/pool/FC forward pass internals |
| `train_minimal.py` | Quick CNN training on CIFAR-10 |
| `visualization.py` | Visualizing filters and activation maps |
| `exercises/` | Building CNN components from scratch |

---

## Next: Day 29

Ready for reinforcement learning? See [Day 29 — Proximal Policy Optimization](../29_PPO/).
