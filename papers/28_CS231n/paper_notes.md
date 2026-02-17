# Paper Notes: CS231n — Convolutional Neural Networks for Visual Recognition

> Notes on Stanford CS231n Course Notes (Karpathy, Fei-Fei Li)

---

## ELI5 (Explain Like I'm 5)

### The Flashlight on a Photograph

Imagine you have a huge photograph and a small flashlight. Instead of looking at the entire photo at once (which would overwhelm you), you shine the flashlight on a small patch — say 3x3 inches — and describe what you see: "edge going left-to-right," "blue color," "corner." Then you move the flashlight one inch to the right and look again. You sweep the entire photo this way, building up a map of "things I found."

Now imagine you have 64 different colored flashlights, each trained to spot a different pattern — one spots vertical edges, another spots horizontal edges, another spots curves. You sweep all 64 across the photo and get 64 separate maps. Stack those maps up, and you've built a richer description of the image.

That's what a convolutional layer does. The flashlight is a "filter" (a small grid of learnable weights). Sliding it across the image is "convolution." The map it produces is an "activation map." Multiple filters = multiple maps = a deeper understanding of the image.

> **Note:** This analogy is ours, not from the CS231n notes. The notes use a "neuron connected to a local region" framing instead.

---

## What the Course Notes Actually Cover

The CS231n CNN module (https://cs231n.github.io/convolutional-networks/) covers the following:

### Why Regular Neural Networks Fail on Images

The notes open with a scaling argument. A regular (fully connected) neural network would need one weight per pixel per neuron. For a 200x200x3 image, that's 120,000 weights per neuron — and the first hidden layer alone would have millions of parameters. This "full connectivity is wasteful" and leads to overfitting.

ConvNets solve this by arranging neurons in 3D volumes (width, height, depth) and connecting each neuron to only a small local region.

### The Convolutional Layer

The core building block. The notes explain:

**Filters and activation maps.** Each filter is small spatially (e.g., 5x5) but extends through the full input depth (e.g., 5x5x3 for RGB). The filter slides across the input, computing dot products at every position, producing a 2D activation map. Multiple filters produce multiple maps, stacked along the depth axis.

**Spatial arrangement.** Three hyperparameters control the output:
- **Depth (K):** number of filters
- **Stride (S):** how far the filter moves between positions
- **Padding (P):** zeros added around the input border

The output size formula: (W - F + 2P) / S + 1

**Local connectivity.** Each neuron connects to a local region (the "receptive field"), not the whole input. Connections are local in 2D (width, height) but full along depth.

**Parameter sharing.** All neurons in a given depth slice use the same weights. This is the key insight: if a feature detector is useful at one location, it should be useful everywhere. Without sharing, AlexNet's first CONV layer would have 105 million parameters. With sharing: 34,944.

**im2col.** The notes describe the standard implementation trick: reshape local input patches into columns, then compute all convolutions as a single matrix multiply. This leverages optimized BLAS routines at the cost of higher memory usage.

### The Pooling Layer

Progressively reduces spatial dimensions. The notes describe max pooling (most common: 2x2 with stride 2, discards 75% of activations) and note that average pooling has "fallen out of favor." The notes also mention the trend toward eliminating pooling entirely, citing Springenberg et al. (2014).

No learnable parameters. Output formula: (W - F) / S + 1.

### Fully-Connected Layers

Identical to regular neural networks. The notes make an important observation: any FC layer can be expressed as an equivalent CONV layer (set filter size equal to input spatial size). This is useful for running a trained network on larger images.

### Normalization Layers

The notes mention local response normalization but state it has "fallen out of favor" and its "contribution has been shown to be minimal, if any."

### ConvNet Architectures and Layer Patterns

The common stacking pattern:

```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
```

The notes recommend small filters (3x3), stride 1 with (F-1)/2 padding for CONV, and 2x2/stride-2 for POOL. They explain why: three stacked 3x3 CONV layers have the same receptive field as one 7x7 layer, but with more non-linearities and fewer parameters (27C^2 vs 49C^2).

### Case Studies

The notes walk through the evolution of CNN architectures from LeNet (1998) through ResNet (2015). Key entries:

**AlexNet (2012):** Five CONV layers, three FC layers, 60M parameters. Used ReLU, dropout, data augmentation, GPU training. Won ILSVRC 2012 with 16.4% top-5 error (vs. 26% for the runner-up). Similar to LeNet but deeper and bigger.

**VGGNet (2014):** 16 CONV/FC layers, all 3x3 convolutions, 138M parameters. Showed that depth is critical. Homogeneous architecture. The notes provide a full layer-by-layer breakdown showing memory and parameter counts for each layer.

**GoogLeNet (2014):** Inception module, 4M parameters (vs. AlexNet's 60M), no FC layers (uses average pooling instead). Won ILSVRC 2014 with 6.7%.

**ResNet (2015):** Skip connections, batch normalization, up to 152 layers. Won ILSVRC 2015 with 3.6%. The notes call it "by far state of the art" and "the default choice for using ConvNets in practice."

---

## The Math

### Conv Layer Output Size (from CS231n)

Given:
- Input volume: $W_1 \times H_1 \times D_1$
- Number of filters: $K$
- Filter size: $F$
- Stride: $S$
- Zero padding: $P$

Output volume: $W_2 \times H_2 \times D_2$

$$W_2 = \frac{W_1 - F + 2P}{S} + 1$$
$$H_2 = \frac{H_1 - F + 2P}{S} + 1$$
$$D_2 = K$$

Parameters per filter: $F \cdot F \cdot D_1 + 1$ (weights + bias)
Total parameters: $(F \cdot F \cdot D_1 + 1) \cdot K$

### Pooling Output Size

Given:
- Input: $W_1 \times H_1 \times D_1$
- Pool size: $F$, Stride: $S$

$$W_2 = \frac{W_1 - F}{S} + 1$$
$$H_2 = \frac{H_1 - F}{S} + 1$$
$$D_2 = D_1$$

### Receptive Field Growth

For $n$ stacked CONV layers with filter size $F$ and stride 1:

$$\text{Effective receptive field} = n \cdot (F - 1) + 1$$

So three 3x3 layers → effective receptive field of 7. Three layers, 27C^2 parameters. One 7x7 layer, 49C^2 parameters. Same receptive field, fewer parameters, more non-linearities.

---

## What CS231n Gets Right

- The output size formula and spatial arrangement explanation is the clearest treatment available. The constraint-checking (must produce integer output) is practical and often skipped elsewhere.
- The im2col description bridges the gap between mathematical definition and actual implementation.
- Layer sizing patterns (3x3 convs, stride 1, pad to preserve size) are presented as concrete rules of thumb, not just theory.
- The VGGNet case study with per-layer memory and parameter counts is genuinely useful for building intuition about where resources go.
- The case studies section traces the evolution from LeNet to ResNet with specific numbers, not just hand-waving.

## What the Course Notes Don't Cover

- **Training details:** The CNN module doesn't discuss optimization, learning rate schedules, or data augmentation (these are in separate CS231n modules).
- **Batch normalization:** Not in the CNN module itself, though covered elsewhere in the course. Now considered essential for deep CNNs.
- **Modern architectures (post-2016):** No EfficientNet, Vision Transformers, or depthwise separable convolutions. The notes were written when ResNet was state of the art.
- **Transfer learning details:** Mentioned briefly but the full treatment is in a separate CS231n module.
- **Depthwise separable convolutions:** The key innovation in MobileNets (Howard et al., 2017) that reduces parameters by factoring standard convolution into depthwise and pointwise components.

---

## Going Beyond the Course Notes (Our Retrospective)

[Our Addition: This section is our commentary, not from CS231n.]

CS231n's CNN module was written during the "golden age" of ConvNets (2014-2016), when VGGNet and ResNet were state of the art. Several developments since then are worth noting:

**Depthwise separable convolutions** (MobileNet, 2017) factored standard CONV into depthwise and pointwise components, reducing computation by 8-9x. This made CNNs viable on mobile devices.

**Vision Transformers** (ViT, Dosovitskiy et al., 2020) showed that you can process images as sequences of patches using self-attention, without convolutions at all. For large datasets, ViTs match or beat CNNs.

**ConvNeXt** (Liu et al., 2022) showed that with modern training techniques (larger kernels, layer norm, GELU), pure ConvNets can match ViTs. The "convolutions vs. attention" debate is ongoing.

The fundamental CS231n concepts — local connectivity, weight sharing, spatial hierarchies — remain correct and foundational regardless of which architecture wins.

---

## Questions Worth Thinking About

1. Why does parameter sharing work? Under what conditions would you NOT want to share parameters spatially? (Hint: CS231n mentions "locally connected layers" for face-centered images.)
2. If three 3x3 layers = one 7x7 layer in receptive field, why not just stack more layers indefinitely? What limits depth? (Think about vanishing gradients — and how ResNet solved this.)
3. The VGGNet breakdown shows that 74% of parameters are in the first FC layer. GoogLeNet and ResNet removed FC layers entirely. What does this tell you about where the "knowledge" lives in a CNN?
4. The trend is moving away from max pooling toward stride > 1 in CONV layers. What's the difference? Why might learned downsampling be better than fixed max operations?

---

**Previous:** [Day 27 — Machine Super Intelligence](../27_MSI/)
**Next:** [Day 29 — Proximal Policy Optimization](../29_PPO/)
