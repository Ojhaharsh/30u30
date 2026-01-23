# 🏋️ Day 4 Exercises: Mastering Bayesian Networks

Welcome to the dojo! Today you'll implement the core concepts of Minimum Description Length and understand why **uncertainty is a feature, not a bug**.

---

## 📋 How to Use These Exercises

1. **Read the exercise** (goal, learning objectives, time estimate)
2. **Open the exercise file** (e.g., `exercise_01_*.py`)
3. **Complete each TODO** with hints provided (don't peek at solutions!)
4. **Run your code** to verify it works
5. **Check `solutions.py`** if you get stuck
6. **Move to next exercise**

**Each exercise is independent** - you can do them in any order, but #2 and #3 build on #1.

---

## 🎯 The 5 Exercises

### **Exercise 1: The "Reparameterization" Trick** ⏱️ 20-30 min | Difficulty: Medium ⏱️⏱️
**File:** `exercise_01_reparameterization.py`

**What you'll implement:**
- The `softplus` activation function
- Gaussian sampling using the reparameterization trick
- Statistical verification (mean/std checks)

**Why it matters:**
The reparameterization trick is the **core innovation** that makes Bayesian neural networks differentiable. Without it, you can't backprop through randomness. This exercise implements the foundation.

**Learning objectives:**
1. How to make randomness differentiable
2. The formula: $w = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim N(0,1)$
3. Why Softplus instead of ReLU for $\sigma$
4. How to verify sampling with statistical tests

**What to look for:**
- Your sampled weights should have mean ≈ $\mu$ and std ≈ $\sigma$
- When $\rho = 0$, $\sigma \approx 0.693$ (verify this!)
- Different $\rho$ values should produce different uncertainty levels

---

### **Exercise 2: The "Gap Experiment"** ⏱️ 25-35 min | Difficulty: Hard ⏱️⏱️⏱️
**File:** `exercise_02_gap_experiment.py`

**What you'll implement:**
- Create a dataset with missing regions (gaps)
- Train a Bayesian network on gappy data
- Visualize uncertainty using Monte Carlo sampling
- Compare with what you'd expect from a regular network

**Why it matters:**
This is the **most important visualization** in Bayesian deep learning. It proves that the network knows what it doesn't know.

**Learning objectives:**
1. Epistemic uncertainty: "I haven't seen data here"
2. Why uncertainty spikes exactly at gaps
3. How Monte Carlo sampling reveals uncertainty
4. The difference between **aleatoric** (noise) and **epistemic** (ignorance) uncertainty

**What to look for:**
- Narrow prediction band where data exists
- **Wide confidence band in the gap** (this is the magic!)
- Uncertainty should be symmetric around the gap (good generalization)
- Regular networks either overfit (narrow everywhere) or underfit (wide everywhere)

---

### **Exercise 3: Breaking the Model (Beta Parameter)** ⏱️ 30-40 min | Difficulty: Medium ⏱️⏱️
**File:** `exercise_03_beta_parameter.py`

**What you'll implement:**
- Train models with different `kl_weight` values
- Measure test MSE, prediction uncertainty, and calibration
- Plot the tradeoff between fit quality and model simplicity
- Find the "sweet spot"

**Why it matters:**
`kl_weight` is the **most important hyperparameter** in Bayesian deep learning. It controls how much you trust the data vs. the prior.

**Learning objectives:**
1. What does `kl_weight` (β) do physically?
2. Low β → Overfit (confident but wrong)
3. High β → Underfit (uncertain but safe)
4. How to measure calibration (% of points in σ bands)

**What to look for:**
- MSE should **increase** as β increases (less data-fitting)
- Uncertainty should **increase** with β
- Calibration error should **decrease** with higher β
- Optimal β ≈ somewhere in the middle

**Tuning guide:**
| Symptom | Solution |
|---------|----------|
| Model ignores data (y ≈ 0) | ↓ Reduce β |
| Uncertainty too small | ↑ Increase β |
| Too wiggly (overfits) | ↑ Increase β |

---

### **Exercise 4: Monte Carlo Predictions** ⏱️ 20-30 min | Difficulty: Medium ⏱️⏱️
**File:** `exercise_04_monte_carlo.py`

**What you'll implement:**
- Run the network multiple times with different weight samples
- Aggregate predictions into mean and uncertainty
- Study how many samples are "enough"
- Verify the Law of Large Numbers

**Why it matters:**
In production, you can't predict once. You predict 50-100 times and aggregate. This exercise shows why.

**Learning objectives:**
1. Monte Carlo approximation of Bayesian inference
2. How uncertainty estimates stabilize with more samples
3. The Law of Large Numbers in action
4. Computational cost vs. accuracy tradeoff

**What to look for:**
- 1 sample: Noisy, jagged predictions
- 10 samples: Clearer but still wiggly
- 100 samples: Smooth, stable, boring (good!)
- Uncertainty estimates should **converge** to a stable value

**Production rule of thumb:** Use 50-100 MC samples. More is slower, less is unreliable.

---

### **Exercise 5: Advanced MDL (Pareto Frontier)** ⏱️ 30-40 min | Difficulty: Hard ⏱️⏱️⏱️
**File:** `exercise_05_advanced_mdl.py`

**What you'll implement:**
- Decompose loss into reconstruction vs. regularization
- Train models across the compression spectrum
- Plot the Pareto frontier
- Understand the fundamental tradeoff

**Why it matters:**
This is the **visual proof** of the MDL principle. You'll see the exact tradeoff curve.

**Learning objectives:**
1. MDL Loss = Reconstruction Loss + KL Divergence
2. Pareto efficiency (can't improve both objectives)
3. How compression ratio changes with β
4. Real-world compression (edge devices, mobile)

**What to look for:**
- The frontier should be a **smooth curve** from upper-left to lower-right
- Points lie **on or below** the frontier (that's optimal!)
- As β increases, you move left (simpler model) and up (worse fit)
- The "elbow" is often the best practical choice

---

## 📊 Summary Table

| Exercise | Topic | File | Time | Difficulty | Output |
|----------|-------|------|------|------------|--------|
| 1 | Reparameterization | `exercise_01_*.py` | 20-30 | ⏱️⏱️ | Stats printout |
| 2 | Gap visualization | `exercise_02_*.py` | 25-35 | ⏱️⏱️⏱️ | `gap_experiment.png` |
| 3 | Beta tuning | `exercise_03_*.py` | 30-40 | ⏱️⏱️ | `beta_study.png` |
| 4 | MC convergence | `exercise_04_*.py` | 20-30 | ⏱️⏱️ | `mc_convergence.png` |
| 5 | Pareto frontier | `exercise_05_*.py` | 30-40 | ⏱️⏱️⏱️ | `pareto_frontier.png` |

**Total time:** 2-3 hours (if you do all 5)

---

## 💡 Getting Help

1. **Stuck on a TODO?**
   - Re-read the hint (don't look at the solution yet)
   - Google the function you need
   - Ask: "What does this TODO need to compute?"

2. **Code runs but gives wrong results?**
   - Check intermediate values with `print()` statements
   - Visualize what you're computing (plot histograms, etc.)
   - Check dimensions: shapes of X, y, weights, etc.

3. **Still stuck?**
   - Look at the corresponding solution in `solutions.py`
   - Understand the solution, then implement your own version
   - Run both and check they give same results

---

## 🎓 Progressive Difficulty

**Beginner-friendly** → Go do Exercise 1 first

**Medium difficulty** → Then try Exercises 3 & 4

**Challenge mode** → Save Exercises 2 & 5 for last

---

## 🏆 What You'll Learn

After these 5 exercises, you'll understand:
- ✅ How Bayesian networks work (probabilistic thinking)
- ✅ Why uncertainty matters (gap experiment)
- ✅ How to tune them (beta parameter)
- ✅ How to use them in production (MC sampling)
- ✅ The compression principle behind all of this (Pareto frontier)

**You're not just implementing code. You're building intuition.**

---

## 📚 Related Files

- **`solutions.py`** - Reference solutions with implementation summaries
- **`README.md`** (parent) - Conceptual overview of MDL
- **`CHEATSHEET.md`** (parent) - Quick API reference
- **`implementation.py`** (parent) - Full Bayesian NN implementation
- **`notebook.ipynb`** (parent) - Interactive notebook walkthrough

---

## 🚀 Next Steps

After completing these exercises:
1. Read the paper/blog posts linked in the main README
2. Explore other Bayesian networks (Variational Inference, etc.)
3. Try on your own dataset (not sine waves!)
4. Compare with other uncertainty methods (ensembles, dropout)

**Happy learning!** 🎯
