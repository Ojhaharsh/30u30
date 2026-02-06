"""
Exercise 3: Scaling Analysis

Replicate the scaling results from Figures 6-8 of the paper.

The paper found:
  - Max entropy ~ n^2 (quadratic in grid size)
  - Max complexity ~ n (linear in grid size)
  - Time to max complexity ~ n^2 (quadratic)

Your task:
  1. Run the interacting model at multiple grid sizes
  2. Record max entropy, max complexity, and time to peak
  3. Fit curves and check the scaling exponents
  4. Plot your results

Paper reference: Section 5.2, Figures 6-8
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation import run_simulation


def scaling_experiment(sizes=None, steps_multiplier=40):
    """
    TODO: Run simulations at multiple grid sizes and collect scaling data.

    For each grid size n:
      - Run total_steps = steps_multiplier * n^2 (so larger grids get more time)
      - Record max entropy, max complexity, time of max complexity

    Args:
        sizes: list of grid sizes to test (e.g., [10, 20, 30, 50])
        steps_multiplier: steps = multiplier * n^2

    Returns:
        sizes: array of grid sizes
        max_entropy: array of max entropy values
        max_complexity: array of max complexity values
        peak_time: array of times to peak complexity
    """
    if sizes is None:
        sizes = [10, 20, 30, 50]

    # YOUR CODE HERE
    pass


def fit_and_plot(sizes, max_entropy, max_complexity, peak_time):
    """
    TODO: Fit curves and plot results.

    1. Fit quadratic to max_entropy vs n  (expect ~ n^2)
    2. Fit linear to max_complexity vs n  (expect ~ n)
    3. Fit quadratic to peak_time vs n    (expect ~ n^2)
    4. Plot all three with fits and r^2 values
    """
    import matplotlib.pyplot as plt

    # YOUR CODE HERE
    # Hint: np.polyfit(sizes, values, degree) gives coefficients
    #       R^2 = 1 - SS_res / SS_tot
    pass


if __name__ == '__main__':
    print("Scaling experiment (Figures 6-8)")
    print("This may take a few minutes for larger grid sizes.")
    print("=" * 50)

    result = scaling_experiment(sizes=[10, 20, 30, 50])

    if result is not None:
        sizes, max_e, max_c, peak_t = result
        print(f"\n{'Size':>6} {'Max Entropy':>12} {'Max Complexity':>14} {'Peak Time':>10}")
        for i in range(len(sizes)):
            print(f"{sizes[i]:>6} {max_e[i]:>12} {max_c[i]:>14} {peak_t[i]:>10}")

        fit_and_plot(sizes, max_e, max_c, peak_t)
