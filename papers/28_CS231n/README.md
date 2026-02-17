# Day 28: CS231n — CNNs for Visual Recognition

> Stanford CS231n Course Notes — Andrej Karpathy, Fei-Fei Li (2015-2024)

**Original Course:** https://cs231n.github.io/

**Time:** 4-6 hours
**Prerequisites:** Day 1 (RNNs), Day 8 (AlexNet helpful), basic linear algebra
**Code:** NumPy + Matplotlib (PyTorch optional for exercises)

---

## What This Course Is Actually About

CS231n is Stanford's deep learning course for computer vision. The course notes — written primarily by Andrej Karpathy — became the de facto introduction to convolutional neural networks for a generation of ML practitioners. Ilya Sutskever included it in his 30-paper reading list as "the Stanford course that solidifies everything."

The course covers image classification, linear classifiers, optimization, backpropagation, neural network architecture, and convolutional neural networks. Our Day 28 focuses on the **CNN module** (https://cs231n.github.io/convolutional-networks/), which is the core technical contribution: how to build networks that exploit the spatial structure of images.

This matters because CNNs are the backbone of essentially all modern computer vision. Every image classifier, object detector, and image generator uses convolutional layers. Understanding the mechanics — local connectivity, parameter sharing, spatial hierarchies — is prerequisite for understanding anything in vision ML.

---

## The Core Idea

Regular neural networks don't scale to images. A single fully-connected neuron looking at a 200x200x3 image would need 120,000 weights — and you'd need many such neurons. ConvNets solve this with three ideas:

1. **Local connectivity** — each neuron sees only a small region (the "receptive field"), not the entire image.
2. **Parameter sharing** — every neuron in a "depth slice" uses the same weights (same filter). If an edge detector is useful at position (10, 10), it's useful at (50, 50) too.
3. **3D volumes** — layers operate on width x height x depth volumes, not flat vectors.

```
Regular NN (fully connected):            ConvNet (local connectivity):

  Every neuron connected to                Each neuron sees a small
  ALL input pixels                         local patch (e.g., 3x3)

  Input: 32x32x3 = 3072 weights           Input: 3x3x3 = 27 weights
  per neuron                               per neuron (shared across
                                           all spatial positions)
```

The result: ConvNets can be much deeper (more layers) with far fewer parameters, because each layer only looks locally and shares weights across space.

---

## What CS231n Actually Covers

### Convolutional Layer

The CONV layer is the core building block. Its parameters are a set of learnable filters, each small spatially (e.g., 3x3 or 5x5) but extending through the full depth of the input. During the forward pass, each filter slides across the input and computes dot products at every spatial position, producing a 2D activation map. Multiple filters produce multiple maps, stacked along the depth dimension.

The output size is governed by a formula. Given input size W, filter size F, stride S, and padding P:

$$\text{Output size} = (W - F + 2P) / S + 1$$

For example: input 32x32, filter 5x5, stride 1, padding 2 gives output 32x32 (spatial size preserved).

**Parameter sharing** reduces the parameter count dramatically. Without sharing, AL's first CONV layer (55x55x96 neurons, each with 11x11x3=363 weights) would have 105 million parameters. With sharing: 96 filters x 363 weights = 34,944 parameters.

### Pooling Layer

Pooling reduces spatial dimensions. The most common form: 2x2 max pooling with stride 2, which halves width and height (discarding 75% of activations). Pooling has no learnable parameters — it computes a fixed function.

The output size formula for pooling:

$$W_2 = (W_1 - F) / S + 1$$

The trend is moving away from pooling toward using stride > 1 in CONV layers instead (as noted in the CS231n notes and explored by Springenberg et al., "Striving for Simplicity," 2014).

### Fully-Connected Layer

Identical to regular neural network layers. Neurons connect to all activations in the previous volume. Typically appear at the end of the network for classification. CS231n notes that any FC layer can be converted to an equivalent CONV layer.

### The im2col Implementation Trick

CS231n describes the standard implementation: stretch local input regions into columns (im2col), then compute all convolutions as a single large matrix multiply. For a 227x227x3 input with 11x11x3 filters at stride 4: im2col produces a [363 x 3025] matrix, filter weights form a [96 x 363] matrix, and np.dot gives [96 x 3025] — the full output.

### Layer Sizing Patterns

The CS231n notes give concrete rules of thumb:
- **Input:** divisible by 2 many times (32, 64, 224, etc.)
- **CONV:** use small filters (3x3 or 5x5), stride 1, padding (F-1)/2 to preserve spatial size
- **POOL:** 2x2 with stride 2 (most common) or 3x3 with stride 2
- **General pattern:** `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

Three stacked 3x3 CONV layers have the same effective receptive field as one 7x7 layer, but with more non-linearities and fewer parameters (27C^2 vs 49C^2).

### Architecture Case Studies

The notes walk through the evolution of ConvNet architectures:

| Architecture | Year | Top-5 Error | Key Innovation |
|------------|------|------------|----------------|
| **LeNet** | 1998 | — | First successful CNN (digit recognition) |
| **AlexNet** | 2012 | 16.4% | Deep CNNs on GPUs; ReLU, Dropout |
| **ZFNet** | 2013 | 14.8% | Tuned AlexNet hyperparameters |
| **GoogLeNet** | 2014 | 6.7% | Inception module; 4M params (vs AlexNet's 60M) |
| **VGGNet** | 2014 | 7.3% | Depth matters; homogeneous 3x3 convs; 138M params |
| **ResNet** | 2015 | 3.6% | Skip connections; batch norm; 152 layers |

Note: error rates are from CS231n's cited results. See further reading for the original papers.

---

## The Architecture

### VGGNet Case Study (from CS231n)

CS231n breaks down VGGNet-16 layer by layer. The architecture uses only 3x3 convolutions with stride 1 and pad 1, and 2x2 max pooling with stride 2:

```
INPUT:     [224x224x3]    memory:  150K    weights: 0
CONV3-64:  [224x224x64]   memory:  3.2M    weights: 1,728
CONV3-64:  [224x224x64]   memory:  3.2M    weights: 36,864
POOL2:     [112x112x64]   memory:  800K    weights: 0
CONV3-128: [112x112x128]  memory:  1.6M    weights: 73,728
CONV3-128: [112x112x128]  memory:  1.6M    weights: 147,456
POOL2:     [56x56x128]    memory:  400K    weights: 0
CONV3-256: [56x56x256]    memory:  800K    weights: 294,912
CONV3-256: [56x56x256]    memory:  800K    weights: 589,824
CONV3-256: [56x56x256]    memory:  800K    weights: 589,824
POOL2:     [28x28x256]    memory:  200K    weights: 0
CONV3-512: [28x28x512]    memory:  400K    weights: 1,179,648
CONV3-512: [28x28x512]    memory:  400K    weights: 2,359,296
CONV3-512: [28x28x512]    memory:  400K    weights: 2,359,296
POOL2:     [14x14x512]    memory:  100K    weights: 0
CONV3-512: [14x14x512]    memory:  100K    weights: 2,359,296
CONV3-512: [14x14x512]    memory:  100K    weights: 2,359,296
CONV3-512: [14x14x512]    memory:  100K    weights: 2,359,296
POOL2:     [7x7x512]      memory:  25K     weights: 0
FC:        [1x1x4096]     memory:  4K      weights: 102,760,448
FC:        [1x1x4096]     memory:  4K      weights: 16,777,216
FC:        [1x1x1000]     memory:  1K      weights: 4,096,000

TOTAL memory: ~93 MB per image (forward only; ~186 MB with gradients)
TOTAL params: 138 million
```

Key observation from this breakdown: most memory is in the early CONV layers, but most parameters are in the FC layers (the first FC alone has 102M of the 138M total).

---

## Implementation Notes

### Initialization
- Xavier/Glorot initialization works well for tanh activations.
- He initialization (variance = 2/fan_in) is preferred with ReLU — this became standard after Kaiming He's 2015 paper.

### The im2col Trick (Practical)
The naive nested-loop convolution is O(output_H x output_W x K x F x F x D). The im2col approach reshapes the problem into a single matrix multiply, which leverages optimized BLAS routines. The tradeoff is higher memory usage (input values are duplicated across columns).

### Batch Normalization
Not in the original CS231n CNN notes, but covered in the course. Normalizes activations before each layer, which stabilizes training and allows higher learning rates. Now standard in most deep CNNs.

[Our Addition: The CS231n notes were written during a period when batch norm was still new. Modern practice treats it as essential, not optional.]

### Memory Considerations
VGGNet uses ~93 MB per image in the forward pass alone. With gradients, that doubles. Mini-batch of 64 images = ~12 GB. This is why GPU memory is often the bottleneck, not computation.

---

## What to Build

### Quick Start

```bash
# Verify environment
python setup.py

# Run the CNN implementation on CIFAR-10
python train_minimal.py --epochs 5 --lr 0.01

# Generate visualizations
python visualization.py
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|---------------------------|
| 1 | Implement conv forward pass (naive) | Understanding what convolution actually computes |
| 2 | Implement max pooling forward + backward | Understanding spatial downsampling |
| 3 | Compute output dimensions for given architectures | Fluency with the output size formula |
| 4 | Count parameters in VGGNet-like architectures | Understanding where parameters live |
| 5 | Visualize learned filters and activation maps | Seeing what CNNs actually learn |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **Local connectivity + parameter sharing = scalability** — CNNs can process large images because each neuron sees only a small patch, and filters are shared across all positions. This is the fundamental insight.
2. **The output size formula matters in practice** — (W - F + 2P)/S + 1 governs every layer's spatial dimensions. Getting this wrong means your architecture won't compile.
3. **Depth beats width** — VGGNet showed that deeper networks with small (3x3) filters outperform shallower networks with large filters, using fewer parameters for the same receptive field. Three stacked 3x3 layers = one 7x7 layer's receptive field, but with 27C^2 vs 49C^2 parameters.
4. **Most parameters live in FC layers, most memory in CONV layers** — VGGNet's first FC layer alone holds 74% of all parameters. This motivated later architectures (GoogLeNet, ResNet) to eliminate FC layers entirely.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `README.md` | This file — overview and guide |
| `paper_notes.md` | ELI5 + detailed notes on CS231n CNN module |
| `CHEATSHEET.md` | Quick reference for CNN hyperparameters and formulas |
| `implementation.py` | CNN from scratch in NumPy (conv, pool, FC layers) |
| `train_minimal.py` | Train a simple CNN on CIFAR-10 |
| `visualization.py` | Visualize filters, activation maps, feature hierarchies |
| `setup.py` | Verify your environment |
| `requirements.txt` | Python dependencies |
| `data/` | CIFAR-10 data directory |
| `exercises/` | 5 exercises to build CNN components from scratch |

---

## Further Reading

- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) — The full CNN module this day is based on
- [Krizhevsky et al. (2012) — ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — AlexNet (Day 8)
- [Simonyan and Zisserman (2014) — Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556) — VGGNet
- [He et al. (2015) — Deep Residual Learning](https://arxiv.org/abs/1512.03385) — ResNet (Day 9)
- [Springenberg et al. (2014) — Striving for Simplicity](https://arxiv.org/abs/1412.6806) — All-convolutional nets (no pooling)
- [CS231n: Full Course Notes](https://cs231n.github.io/) — Classification, optimization, backpropagation, and more

---

**Previous:** [Day 27 — Machine Super Intelligence](../27_MSI/)  
**Next:** [Day 29 — Proximal Policy Optimization](../29_PPO/)
