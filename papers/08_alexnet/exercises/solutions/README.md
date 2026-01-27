# Solutions: AlexNet Exercises

Complete solutions for all 5 exercises.

---

## Solution Files

| Exercise | Solution File |
|----------|--------------|
| 1 | `solution_01_build_alexnet.py` |
| 2 | `solution_02_ablation_study.py` |
| 3 | `solution_03_feature_viz.py` |
| 4 | `solution_04_transfer_learning.py` |
| 5 | `solution_05_modern_improvements.py` |

---

## Key Insights

### Solution 1: Build AlexNet
- Total parameters: ~61 million
- Key: Large kernels (11×11, 5×5) in early layers
- Critical: Proper padding/stride calculations

### Solution 2: Ablation Study
- ReLU is critical (sigmoid → vanishing gradients)
- Dropout helps with small datasets
- Data augmentation always helps

### Solution 3: Feature Visualization
- Conv1: Gabor-like edge detectors
- Mid layers: Textures and patterns
- Later layers: Object parts

### Solution 4: Transfer Learning
- Pretrained >> Random init (especially with limited data)
- Freeze conv, train classifier = fastest
- Lower LR for fine-tuning

### Solution 5: Modern Improvements
- BatchNorm: Faster training, higher accuracy
- Kaiming init: Better gradient flow
- AdamW + cosine scheduler: Smoother training

---

## Running Solutions

```bash
python solution_01_build_alexnet.py
```

---

Good luck! 🚀
