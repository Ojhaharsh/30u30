# Solutions: Dropout Exercises

Complete solutions for Day 12 dropout exercises.

## Key Concepts

### Inverted Dropout Formula
```python
# Training
output = (input * mask) / keep_prob

# Inference  
output = input  # No change needed!
```

### Why Inverted Dropout?
1. **Simpler inference**: No scaling needed at test time
2. **Safer**: Can't forget to scale
3. **Standard**: Used by PyTorch, TensorFlow, etc.

### Optimal Dropout Rates
- **Input layer**: 0.8-0.9 (light dropout)
- **Hidden layers**: 0.5 (standard)
- **Output layer**: 1.0 (no dropout!)

### Spatial Dropout (Dropout2D)
For CNNs, drop entire channels instead of individual pixels:
- Nearby pixels are correlated
- Dropping one pixel doesn't help much
- Dropping a channel = removing a feature

### MC Dropout for Uncertainty
1. Keep dropout ON at inference
2. Run N forward passes
3. Mean = prediction
4. Variance = uncertainty

---

## Solution Summaries

### Solution 1: Basic Dropout
- Generate Bernoulli mask with `np.random.rand() < p`
- Scale by `1/p` during training
- Apply same mask in backward pass

### Solution 2: Dropout Rate Sweep
- Optimal rate usually around 0.5
- Too high = underfitting (can't learn)
- Too low = overfitting (gap increases)

### Solution 3: Spatial Dropout
- Mask shape: (batch, channels, 1, 1)
- Broadcast to full spatial dimensions
- Better regularization for CNNs

### Solution 4: MC Dropout
- Uncertainty correlates with errors
- Rejecting uncertain samples improves accuracy
- ~50-100 samples needed for stability

### Solution 5: Regularization Comparison
- Dropout best for large networks
- L2 good for smaller networks
- Combined often works best
- Early stopping almost always helps

---

Run solutions with: `python solution_01_build_dropout.py`
