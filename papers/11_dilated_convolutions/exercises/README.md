# Exercises: Dilated Convolutions - Multi-Scale Context

5 hands-on exercises on dilated/atrous convolutions for dense prediction.

---

## Exercise 1: Understand Receptive Fields ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_01_receptive_field.py`

### Goal
Visualize how dilation expands receptive fields without losing resolution.

### What You'll Learn
- Receptive field calculation
- Dilation rate effects
- Resolution preservation

### Tasks
1. Calculate receptive field for various dilation rates
2. Visualize receptive field growth
3. Compare dilated vs standard convolutions

---

## Exercise 2: Build Context Module ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_02_context_module.py`

### Goal
Implement a multi-scale context aggregation module.

### What You'll Learn
- Parallel dilated convolutions
- Feature pyramid at single scale
- Context aggregation strategies

---

## Exercise 3: Implement ASPP ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_03_aspp.py`

### Goal
Build Atrous Spatial Pyramid Pooling (DeepLab).

### What You'll Learn
- Multi-scale feature extraction
- Parallel branches with different dilation
- Feature fusion strategies

---

## Exercise 4: Architecture Ablation ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_04_ablation.py`

### Goal
Compare different dilation patterns and architectures.

### What You'll Learn
- Optimal dilation rate sequences
- Gridding artifact problem
- HDC (Hybrid Dilated Convolution)

---

## Exercise 5: WaveNet Audio ⏱️⏱️⏱️⏱️
**Difficulty**: Very Hard  
**Time**: 3-4 hours  
**File**: `exercise_05_wavenet_audio.py`

### Goal
Apply dilated convolutions to audio generation.

### What You'll Learn
- 1D dilated convolutions
- Exponentially growing receptive field
- Autoregressive models

---

## Solutions

Complete solutions in `solutions/` folder.

---

## Estimated Time

| Exercise | Difficulty | Time |
|----------|-----------|------|
| 1 | Medium | 1-2 hours |
| 2 | Hard | 2-3 hours |
| 3 | Hard | 2-3 hours |
| 4 | Medium | 1-2 hours |
| 5 | V. Hard | 3-4 hours |

**Total**: 10-14 hours

---

Good luck! Dilated convolutions see more with less! 🔍
