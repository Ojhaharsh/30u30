# Day 25 Exercise Solutions: Scaling Laws

This directory contains the reference implementations for the Day 25 exercises. 

## Solutions Overview

| # | File | Technical Objective | Key Learning |
|---|------|---------------------|--------------|
| 1 | `solution_01_fitter.py` | Log-linear regression | Finding $\alpha$ in log-log space. |
| 2 | `solution_02_compute.py` | Compute estimation | Applying the $6NBS$ rule. |
| 3 | `solution_03_bottleneck.py` | Bottleneck analysis | Identifying $D$-bottlenecks. |
| 4 | `solution_04_predictor.py` | Extrapolation | Use power laws for forecasting. |
| 5 | `solution_05_transformer.py` | Architecture | Verifying $12 \times L \times d_{model}^2$. |

## How to Use These Solutions

1. **Self-Correction**: Use these to check your math and implementation logic after completing the exercises.
2. **Pedagogical Reference**: Each solution is heavily commented to explain the "why" behind the implementation, referencing specific sections of Kaplan et al. (2020).
3. **Debugging**: If your scaling plots are not linear, compare your `fitter` implementation with `solution_01`.

## Key Formulas for Verification
- **Compute**: $C \approx 6N \times \text{tokens}$
- **Optimal D**: $D_{opt} \approx 5000 \times N^{0.74}$
- **Transformer Parameters**: $N \approx 12 \times L \times d_{model}^2$
