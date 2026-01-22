# Exercises: RNN Regularization

5 hands-on exercises to master regularization techniques. Work through them in order—each builds on the previous!

---

## Exercise 1: Implement Dropout ⏱️⏱️
**Difficulty:** Medium  
**Time:** 30-45 minutes  
**File:** `exercise_01_dropout.py`

### Goal
Implement a dropout layer with forward and backward passes from scratch.

### What You'll Learn
- How dropout randomly disables neurons
- Why we scale outputs by 1/keep_prob
- Difference between training and inference modes
- How gradients flow through dropout

### Tasks
1. Implement `dropout_forward(x, keep_prob, training)`:
   - Generate binary mask: `mask = np.random.binomial(1, keep_prob, x.shape)`
   - Scale outputs: `output = x * mask / keep_prob`
   - Return (output, mask) for backward pass
   
2. Implement `dropout_backward(dout, mask)`:
   - Gradient only flows through active neurons
   - Return `dout * mask`

3. Handle edge cases:
   - `training=False`: return input unchanged
   - `keep_prob=1.0`: no dropout (return input)

### Success Criteria
- Forward pass produces correct sparsity (verify ~20% zeros with keep_prob=0.8)
- Output expectation is unchanged (mean stays same)
- Backward pass only propagates through active neurons
- All test cases pass

### Hints
- Remember: mask is already scaled by 1/keep_prob in forward
- Use `np.random.binomial(1, keep_prob, size)` for mask
- Test with known seeds for reproducibility
- Common mistake: applying dropout at test time!

---

## Exercise 2: Layer Normalization ⏱️⏱️⏱️
**Difficulty:** Hard  
**Time:** 45-60 minutes  
**File:** `exercise_02_normalization.py`

### Goal
Implement layer normalization with learnable parameters (gamma, beta).

### What You'll Learn
- How normalization stabilizes training
- Computing statistics per sample (not per batch)
- Learnable scale and shift parameters
- Backward pass through normalization

### Tasks
1. Implement `layer_norm_forward(x, gamma, beta, eps)`:
   - Compute mean: `μ = mean(x, axis=-1)`
   - Compute variance: `σ² = var(x, axis=-1)`
   - Normalize: `x̂ = (x - μ) / sqrt(σ² + ε)`
   - Scale and shift: `y = γ * x̂ + β`
   - Cache values for backward pass

2. Implement `layer_norm_backward(dout, cache)`:
   - Compute `dgamma = sum(dout * x_hat)`
   - Compute `dbeta = sum(dout)`
   - Compute `dx` using chain rule (tricky!)

3. Use `eps=1e-5` for numerical stability

### Success Criteria
- Forward output has zero mean, unit variance (per sample)
- Backward gradients match numerical gradient check
- Learnable parameters (gamma, beta) update correctly
- All test cases pass

### Hints
- Use `keepdims=True` when computing mean/var
- Layer norm normalizes across features, NOT across batch
- For backward, the derivative is complex—see solution if stuck
- Test with gamma=1, beta=0 first (identity transform)

---

## Exercise 3: Weight Decay (L2 Regularization) ⏱️
**Difficulty:** Easy  
**Time:** 20-30 minutes  
**File:** `exercise_03_weight_decay.py`

### Goal
Implement L2 regularization that penalizes large weights.

### What You'll Learn
- Why simpler models generalize better (Occam's Razor)
- How weight decay pushes weights toward zero
- Adding regularization to the loss
- Gradient of L2 penalty

### Tasks
1. Implement `compute_l2_penalty(weights_list, weight_decay)`:
   - Sum squared weights: `Σ w²`
   - Multiply by coefficient: `λ/2 * Σ w²`
   - Return scalar penalty

2. Implement `regularized_loss(model_loss, weights, weight_decay)`:
   - Combine: `total = model_loss + l2_penalty`

3. Implement gradient contribution:
   - Weight decay adds `λ * W` to gradient

### Success Criteria
- Penalty increases with larger weights
- Penalty increases with higher weight_decay
- Regularized loss = original loss when weight_decay=0
- Gradient includes regularization term

### Hints
- Don't forget the 1/2 factor: `0.5 * λ * Σw²`
- Weight decay should NOT apply to biases
- Start with weight_decay=0.0001 (very small)
- Test: penalty should be proportional to weight magnitude

---

## Exercise 4: Early Stopping ⏱️⏱️
**Difficulty:** Medium  
**Time:** 30-40 minutes  
**File:** `exercise_04_early_stopping.py`

### Goal
Implement an early stopping monitor that tracks validation loss and stops training when improvement plateaus.

### What You'll Learn
- Why we stop before training loss is minimized
- The patience mechanism
- Saving and restoring best model state
- Detecting overfitting in real-time

### Tasks
1. Implement `EarlyStoppingMonitor.__init__`:
   - Store patience value
   - Initialize `best_loss = infinity`
   - Initialize `no_improve_count = 0`
   - Initialize `best_model_state = None`

2. Implement `check(val_loss, epoch, model_state)`:
   - If `val_loss < best_loss`:
     - Update best_loss, best_epoch
     - Reset counter to 0
     - Save model_state
   - Else:
     - Increment counter
   - Return `True` to continue, `False` to stop

3. Implement `should_stop()`:
   - Return `no_improve_count >= patience`

### Success Criteria
- Monitor correctly tracks best loss
- Counter resets on improvement
- Counter increments on plateau
- Stops after `patience` consecutive non-improvements
- Best model state is saved correctly

### Hints
- Use `float('inf')` for initial best_loss
- Common mistake: not resetting counter on improvement
- Test with artificial loss sequences to verify behavior
- Patience=3 means stop after 3 consecutive bad epochs

---

## Exercise 5: Full Pipeline ⏱️⏱️⏱️⏱️
**Difficulty:** Very Hard  
**Time:** 60-90 minutes  
**File:** `exercise_05_full_pipeline.py`

### Goal
Combine all four regularization techniques in a complete training loop.

### What You'll Learn
- How regularization techniques work together
- Training vs validation mode switching
- Managing hidden states in RNNs
- Real-world training workflow

### Tasks
1. Implement `train_one_epoch()`:
   - Set `training=True`
   - Apply dropout to hidden states
   - Add L2 penalty to loss
   - Compute average loss across sequences

2. Implement `evaluate()`:
   - Set `training=False` (no dropout!)
   - Same computation but no weight decay in loss
   - Return validation loss

3. Implement `train_with_regularization()`:
   - Initialize early stopping monitor
   - Loop through epochs:
     - Train one epoch
     - Evaluate on validation
     - Check early stopping
     - Print progress
   - Return training history

4. Track metrics:
   - Training loss per epoch
   - Validation loss per epoch
   - Best epoch and best val loss
   - Stop reason (early_stop or max_epochs)

### Success Criteria
- Training loss decreases over epochs
- Validation loss tracked separately
- Early stopping triggers correctly
- Model state saved at best epoch
- All techniques work together

### Hints
- Don't forget `training=False` during validation!
- Reset hidden states between sequences
- Use exercises 1-4 as building blocks
- Start with simple mock model for testing

---

## 🚀 How to Run Exercises

### Run a Single Exercise
```bash
cd exercises
python exercise_01_dropout.py
```

### Expected Output
```
============================================================
Exercise 1: Dropout Layer
============================================================

Testing dropout_forward...
  ✓ Test 1: No dropout mode works
  ✓ Test 2: Shape preservation works
  ✓ Test 3: Dropout rate correct (~20%, got 19.8%)
  ✓ Test 4: Output scaling correct (mean=2.00)
✓ All dropout_forward tests passed!

Testing dropout_backward...
  ✓ Test 1: Gradient shape correct
  ✓ Test 2: Gradient only flows through active neurons
✓ All dropout_backward tests passed!

🎉 All tests passed! You've mastered dropout!
```

### Compare with Solution
```bash
python solutions/01_dropout.py
```

---

## 💡 Tips for Success

### General Tips
1. **Read the docstring first** — it explains exactly what to implement
2. **Check test cases** — they show expected behavior
3. **Start simple** — get the basic case working before edge cases
4. **Print intermediate values** — helps debug shape/value issues

### When Stuck
1. Re-read the relevant section in [../README.md](../README.md)
2. Check [../CHEATSHEET.md](../CHEATSHEET.md) for formulas
3. Look at the solution for just the function you need
4. Understanding > completing — take your time!

### Common Mistakes by Exercise

**Exercise 1 (Dropout):**
- ❌ Applying dropout at test time
- ❌ Forgetting to scale by 1/keep_prob
- ✅ Check: output mean should equal input mean

**Exercise 2 (Layer Norm):**
- ❌ Using wrong axis for statistics (should be -1)
- ❌ Forgetting epsilon for numerical stability
- ✅ Check: output should have mean≈0, var≈1

**Exercise 3 (Weight Decay):**
- ❌ Missing the 0.5 factor in penalty
- ❌ Applying to biases (should only apply to weights)
- ✅ Check: penalty should be 0 when weights are 0

**Exercise 4 (Early Stopping):**
- ❌ Not resetting counter on improvement
- ❌ Using <= instead of < for comparison
- ✅ Check: counter should be 0 after improvement

**Exercise 5 (Full Pipeline):**
- ❌ Dropout during validation (should be off!)
- ❌ Not resetting hidden states between sequences
- ✅ Check: val_loss should be evaluated without dropout

---

## 📊 Exercise Summary

| # | Exercise | Difficulty | Time | Key Concept |
|---|----------|-----------|------|-------------|
| 1 | Dropout | ⭐⭐ | 30 min | Random neuron deactivation |
| 2 | Layer Norm | ⭐⭐⭐ | 45 min | Activation normalization |
| 3 | Weight Decay | ⭐ | 20 min | L2 regularization |
| 4 | Early Stopping | ⭐⭐ | 30 min | Validation monitoring |
| 5 | Full Pipeline | ⭐⭐⭐⭐ | 60 min | Integration |

**Total Time:** 3-4 hours

---

## ✅ Completion Checklist

- [ ] Exercise 1: Dropout — Can implement forward and backward
- [ ] Exercise 2: Layer Norm — Understand normalization math
- [ ] Exercise 3: Weight Decay — Know how L2 regularization works
- [ ] Exercise 4: Early Stopping — Can monitor and stop training
- [ ] Exercise 5: Full Pipeline — Can combine all techniques

**Completed all 5?** You've mastered regularization! 🎉

Move on to [../exercises/solutions/README.md](solutions/README.md) for detailed solution explanations.

---

*Remember: Struggling is part of learning. Try each exercise for at least 20 minutes before checking solutions!*
