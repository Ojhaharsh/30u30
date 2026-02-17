# Solutions: Day 28 — CS231n CNNs for Visual Recognition

Reference implementations for all 5 exercises. Use these to check your work or if you're stuck.

| # | File | What It Implements |
|---|------|--------------------|
| 1 | `solution_01_conv_forward.py` | Naive conv forward pass (nested loops, output size formula) |
| 2 | `solution_02_pooling.py` | Max pooling forward + backward (gradient routing to max positions) |
| 3 | `solution_03_output_sizes.py` | Output dimension calculator for CONV and POOL layers |
| 4 | `solution_04_parameter_count.py` | Parameter counter with full VGG-16 breakdown |
| 5 | `solution_05_feature_viz.py` | Filter visualization + activation map computation and display |

## Running Solutions

From the `28_CS231n/` directory:

```bash
python exercises/solutions/solution_01_conv_forward.py
python exercises/solutions/solution_02_pooling.py
python exercises/solutions/solution_03_output_sizes.py
python exercises/solutions/solution_04_parameter_count.py
python exercises/solutions/solution_05_feature_viz.py
```

Each solution includes the same `check()` tests as the exercise file. All tests should pass.

## Try the Exercise First

These solutions exist for verification, not shortcuts. The learning happens when you get stuck and work through it. If you haven't attempted the exercise yet, go to `exercises/` and start there.
