# Solutions: ResNet V2 Exercises

Complete solutions for ResNet V2 exercises.

## Key Insight: Pre-Activation

```
Post-activation (V1): Conv → BN → ReLU → Conv → BN → Add → ReLU
Pre-activation (V2):  BN → ReLU → Conv → BN → ReLU → Conv → Add
```

The crucial difference: In V2, the skip connection is a **pure identity**!

### Why This Matters:
1. Gradient flows directly to any layer
2. No degradation even at 1000+ layers
3. Easier optimization landscape

---

Run solutions with: `python solution_01_pre_vs_post.py`
