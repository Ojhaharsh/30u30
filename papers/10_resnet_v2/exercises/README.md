# Exercises: ResNet V2 - Identity Mappings

5 hands-on exercises on pre-activation residual networks.

---

## Exercise 1: Pre vs Post Activation ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_01_pre_vs_post.py`

### Goal
Compare pre-activation and post-activation residual blocks.

### What You'll Learn
- BN-ReLU-Conv vs Conv-BN-ReLU ordering
- Information flow in residual paths
- Why pre-activation enables deeper training

---

## Exercise 2: Ablation on Activation Placement ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_02_ablation_placement.py`

### Goal
Test all 5 placement variants from the paper.

### What You'll Learn
- Which placement matters most
- Impact on training stability
- Effect on final accuracy

---

## Exercise 3: Ultra-Deep Training ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_03_ultra_deep.py`

### Goal
Train 100+ layer networks successfully.

### What You'll Learn
- Scaling to extreme depth
- When depth helps vs hurts
- Practical limits of residual networks

---

## Exercise 4: Build ResNet-1001 ⏱️⏱️⏱️⏱️
**Difficulty**: Very Hard  
**Time**: 3-4 hours  
**File**: `exercise_04_resnet_1001.py`

### Goal
Implement and train a 1001-layer ResNet.

### What You'll Learn
- Extreme depth architectures
- Memory optimization
- Gradient checkpointing

---

## Exercise 5: Transfer Learning ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_05_transfer_learning.py`

### Goal
Compare transfer learning: ResNet V1 vs V2.

---

## Solutions

Complete solutions in `solutions/` folder.

---

Good luck! Pre-activation unlocks true deep learning! 🚀
