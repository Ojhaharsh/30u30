# Solutions: ResNet Exercises

Complete solutions for ResNet exercises.

## Key Insights

### Skip Connections
- `output = F(x) + x` is the magic
- Gradients flow directly through identity path
- Enables training 100+ layer networks

### BasicBlock vs Bottleneck
- BasicBlock: Two 3x3 convs (ResNet-18/34)
- Bottleneck: 1x1 → 3x3 → 1x1 (ResNet-50/101/152)

### Why Residuals Work
1. Easy to learn identity mapping
2. Gradient highway through skip connection
3. Ensemble-like behavior

---

Run solutions with: `python solution_01_build_resnet18.py`
