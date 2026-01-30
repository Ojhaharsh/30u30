# Solutions: Dilated Convolutions Exercises

Complete solutions for dilated convolution exercises.

## Key Concepts

### Receptive Field Formula
For dilated convolution with kernel k and dilation d:
```
Effective RF = (k - 1) × d + 1
```

### Why Dilated Convolutions?
1. **Exponential RF growth**: Stack dilations 1,2,4,8... for huge RF
2. **No resolution loss**: Unlike pooling, keeps full spatial resolution
3. **Efficient**: Fewer parameters than larger kernels

### Common Dilation Patterns
- **Exponential**: 1, 2, 4, 8, 16 (WaveNet style)
- **Repeat**: 1, 2, 4, 1, 2, 4 (avoids gridding)
- **ASPP**: 6, 12, 18 in parallel (DeepLab)

### Gridding Problem
Using powers of 2 can cause "holes" in receptive field.
Solution: Use non-power-of-2 dilations or HDC.

---

Run solutions with: `python solution_01_receptive_field.py`
