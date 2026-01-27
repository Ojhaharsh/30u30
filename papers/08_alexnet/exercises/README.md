# Exercises: AlexNet - The Deep Learning Revolution

5 hands-on exercises to master convolutional neural networks. Work through them in order - each builds on the previous!

---

## Exercise 1: Build AlexNet from Scratch ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_01_build_alexnet.py`

### Goal
Implement AlexNet's architecture from scratch using PyTorch.

### What You'll Learn
- Convolutional layer mechanics
- Pooling strategies
- Fully connected layers
- Network depth design

### Tasks
1. Implement Conv1: 96 filters, 11×11, stride 4
2. Implement Conv2-5 with proper padding/stride
3. Add max pooling layers
4. Implement fully connected layers with dropout
5. Add ReLU activations throughout
6. Test on sample images

### Success Criteria
- All layer shapes match AlexNet paper
- Forward pass works with 224×224 input
- Output is 1000-dimensional

### Hints
- Use `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`
- Check output shapes at each layer
- Total params should be ~61 million

---

## Exercise 2: Ablation Study ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_02_ablation_study.py`

### Goal
Remove AlexNet innovations one at a time and measure impact.

### What You'll Learn
- Which innovations matter most
- Trade-offs in architecture design
- Scientific methodology for DL research

### Tasks
1. Create baseline AlexNet (all innovations)
2. Remove ReLU → use sigmoid
3. Remove dropout
4. Remove data augmentation
5. Train each variant on CIFAR-10
6. Compare performance

### Success Criteria
- All variants train without crashes
- Clear comparison table
- Identify most important innovation

### Hints
- Use CIFAR-10 (faster than ImageNet)
- Same hyperparameters for fair comparison
- Track both train and val accuracy

---

## Exercise 3: Feature Visualization ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_03_feature_viz.py`

### Goal
Visualize what AlexNet learns at each layer.

### What You'll Learn
- What early layers detect (edges, colors)
- What later layers detect (textures, objects)
- Feature hierarchy in CNNs

### Tasks
1. Load pretrained AlexNet
2. Visualize conv1 filters (96 filters)
3. Visualize feature maps for sample images
4. Compare activations for different object categories
5. Find maximally activating images for specific neurons

### Success Criteria
- Conv1 shows Gabor-like filters
- Feature maps show meaningful patterns
- Can interpret what neurons detect

### Hints
- `torchvision.models.alexnet(pretrained=True)`
- Use hooks to capture activations
- Normalize visualizations for visibility

---

## Exercise 4: Transfer Learning ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_04_transfer_learning.py`

### Goal
Fine-tune pretrained AlexNet on a new dataset.

### What You'll Learn
- Transfer learning workflow
- Feature extraction vs fine-tuning
- How to adapt to new domains

### Tasks
1. Load pretrained AlexNet
2. Choose new dataset (flowers, pets, food)
3. Freeze convolutional layers
4. Replace classifier for new classes
5. Fine-tune and evaluate
6. Compare: random init vs pretrained

### Success Criteria
- Model converges on new dataset
- Pretrained >> Random initialization
- Document accuracy improvement

### Hints
- Freeze layers: `param.requires_grad = False`
- Lower learning rate for fine-tuning
- Use smaller dataset (faster iteration)

---

## Exercise 5: Modern Improvements ⏱️⏱️⏱️⏱️
**Difficulty**: Very Hard  
**Time**: 3-4 hours  
**File**: `exercise_05_modern_improvements.py`

### Goal
Add modern techniques to AlexNet and measure improvement.

### What You'll Learn
- Batch normalization
- Better initialization (Kaiming/He)
- Modern optimizers (Adam)
- Learning rate scheduling

### Tasks
1. Add batch normalization after each conv
2. Use Kaiming initialization
3. Switch to AdamW optimizer
4. Add cosine learning rate schedule
5. Add modern augmentations (AutoAugment)
6. Compare classic vs modern AlexNet

### Success Criteria
- Modern version trains faster
- Higher final accuracy
- Clearer training curves

### Hints
- BatchNorm goes before ReLU (or after, debate exists)
- Start with one improvement, add incrementally
- Monitor training stability

---

## Bonus Challenges 🌟

### Challenge 1: Mini AlexNet
Design a smaller AlexNet for CIFAR-10 (32×32 images).

### Challenge 2: Attention AlexNet
Add attention mechanisms to AlexNet layers.

### Challenge 3: Pruning
Remove 50% of weights and maintain accuracy.

### Challenge 4: Quantization
Convert AlexNet to INT8 for faster inference.

### Challenge 5: Architecture Search
Use hyperparameter search to find better filter counts.

---

## Tips for Success

### General Advice
- **GPU is helpful**: Training is faster with CUDA
- **Use pretrained weights**: Saves hours of training
- **Visualize often**: Plot losses, activations, filters
- **Debug with small data**: Test on 1 batch first

### Debugging Checklist
1. Check input shapes (should be [B, 3, 224, 224])
2. Check for NaN in outputs
3. Verify gradients are flowing
4. Monitor GPU memory usage

### Common Mistakes
- Wrong input size (must be 224×224)
- Missing normalization (ImageNet mean/std)
- Learning rate too high for fine-tuning
- Not using eval() mode for inference

---

## Solutions

Complete solutions are available in `solutions/` folder:
- `solution_01_build_alexnet.py`
- `solution_02_ablation_study.py`
- `solution_03_feature_viz.py`
- `solution_04_transfer_learning.py`
- `solution_05_modern_improvements.py`

**Recommendation**: Try solving on your own first!

---

## Estimated Time

| Exercise | Difficulty | Time       | Prerequisites |
|----------|-----------|------------|---------------|
| 1        | Hard      | 2-3 hours  | None          |
| 2        | Medium    | 1-2 hours  | Exercise 1    |
| 3        | Medium    | 1-2 hours  | Pretrained    |
| 4        | Hard      | 2-3 hours  | Pretrained    |
| 5        | V. Hard   | 3-4 hours  | Exercise 1    |

**Total**: 9-14 hours for all exercises

---

## Learning Path

```
Day 8 README → Paper Notes → Implementation
                                    ↓
                            Exercise 1
                          (Build AlexNet)
                                    ↓
                ┌───────────────────┼───────────────────┐
                ↓                   ↓                   ↓
          Exercise 2           Exercise 3         Exercise 4
         (Ablation)          (Feature Viz)    (Transfer Learn)
                └───────────────────┼───────────────────┘
                                    ↓
                              Exercise 5
                          (Modern Improvements)
                                    ↓
                               Day 9! 🚀
```

---

Good luck! AlexNet started the deep learning revolution - now it's your turn to understand it deeply! 🏆
