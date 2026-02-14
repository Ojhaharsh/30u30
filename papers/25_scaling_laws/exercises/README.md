# Exercises: Scaling Laws for Neural Language Models

Welcome to the scaling law exercises. These tasks are designed to move you from theoretical understanding of scaling laws to practical engineering applications. You will learn to predict performance, estimate costs, and debug scaling bottlenecks.

These exercises are tiered from introductory math to complex architectural audits.

---

## Exercises Overview

| # | Task | Difficulty | What You'll Get Out of It |
|---|------|------------|---------------------------|
| 1 | **Power-Law Fitter** | Easy (2/5) | Master the math of transforming raw loss into a predictable curve. |
| 2 | **The 6N Estimator** | Easy (2/5) | Calculate exactly how many GPUs (and dollars) you need for a run. |
| 3 | **Overfitting Detector** | Medium (3/5) | Learn to identify the "Data Wall" where scaling laws break. |
| 4 | **GPT-3 Performance Predictor** | Medium (3/5) | Forecast performance across 3 orders of magnitude with precision. |
| 5 | **Minimal Scaling Transformer** | Hard (4/5) | Implement an architecture optimized for $12Ld^2$ parameter audits. |

---

## Logic Breakdown

### Exercise 1: Power-Law Fitter
The core of Kaplan's paper is Equation 1.1: $L(N) = (N_c / N)^\alpha$. In practice, we solve this by moving to log-space:
$\log(L) = \log(N_c^\alpha) - \alpha \log(N)$.
This is a simple linear regression. Your task is to implement the transformation and fit.

### Exercise 2: The 6N Estimator
Every forward-backward pass involves roughly $6N$ floating point operations. This exercise builds your "intuition for compute." You'll calculate PF-days for various OpenAI models.

### Exercise 3: Overfitting Detector
Models follow power laws ONLY if they aren't data-bottlenecked. In this exercise, you'll analyze a training sweep where the data $D$ stays constant while $N$ increases. You'll find the "knee" where the law breaks.

### Exercise 4: GPT-3 Predictor
If you know $L(N)$ for $N=10^6$ and $N=10^8$, can you predict $N=1.75 \cdot 10^{11}$? You'll use your fitter logic from Ex 1 to forecast the GPT-3 breakthrough.

### Exercise 5: Minimal Scaling Transformer
To match Kaplan's rigor, you must exclude embeddings. You'll build a standard block and verify that it contains exactly $12 \cdot d_{model}^2$ parameters. This is the difference between "I think it scales" and "I know why it scales."

---

## Instructions

1. **Start small**: Open `exercise_01_fitter.py`.
2. **Read the docstrings**: They follow the the project's NumPy-style standard and contain the specific math steps required.
3. **Run to verify**: Each exercise has a `if __name__ == "__main__":` block that acts as a unit test.
4. **Don't peek**: Solutions are in the `solutions/` folder. Use them only when you've been stuck for 15+ minutes.

---

## Success Criteria

- [ ] `exercise_01`: Your fitted $\alpha$ matches the target (0.076) within ±0.001.
- [ ] `exercise_02`: You can correctly differentiate between Petascale and Exascale compute for GPT-3.
- [ ] `exercise_03`: You identify the exact token count where the model begins to overfit.
- [ ] `exercise_05`: Your manual parameter count matches the theoretic count to 99.9% precision.

---

**Next Step:** Head into `exercise_01_fitter.py` to begin.
