# Exercises: ResNet - Deep Residual Learning

5 hands-on exercises to master residual networks. Work through them in order!

---

## Exercise 1: Build ResNet-18 ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_01_build_resnet18.py`

### Goal
Implement ResNet-18 with skip connections from scratch.

### What You'll Learn
- Residual block architecture
- Skip connections (identity mappings)
- Batch normalization placement

### Tasks
1. Implement BasicBlock with skip connection
2. Stack blocks into ResNet-18
3. Handle dimension mismatches with 1x1 convolutions
4. Test on CIFAR-10

---

## Exercise 2: Skip Connection Ablation ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_02_ablation_skip.py`

### Goal
Remove skip connections and observe training degradation.

### What You'll Learn
- Why skip connections enable deep training
- Vanishing gradient problem
- Degradation problem in deep networks

---

## Exercise 3: Gradient Flow Analysis ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_03_gradient_flow.py`

### Goal
Visualize how gradients flow through ResNet vs plain network.

### What You'll Learn
- Gradient magnitude at each layer
- How skip connections preserve gradients
- Training dynamics of deep networks

---

## Exercise 4: Depth Scaling ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_04_depth_scaling.py`

### Goal
Train ResNets of various depths and compare performance.

### What You'll Learn
- ResNet-18, -34, -50, -101 comparison
- Diminishing returns of depth
- Computation vs accuracy trade-offs

---

## Exercise 5: Transfer Learning ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_05_transfer_learning.py`

### Goal
Fine-tune pretrained ResNet on custom dataset.

### What You'll Learn
- Feature extraction with ResNet
- Fine-tuning strategies
- Domain adaptation

---

## Solutions

Complete solutions in `solutions/` folder.

---

## Estimated Time

| Exercise | Difficulty | Time |
|----------|-----------|------|
| 1 | Hard | 2-3 hours |
| 2 | Medium | 1-2 hours |
| 3 | Hard | 2-3 hours |
| 4 | Hard | 2-3 hours |
| 5 | Medium | 1-2 hours |

**Total**: 8-13 hours

---

Good luck! Skip connections changed everything! 🚀
