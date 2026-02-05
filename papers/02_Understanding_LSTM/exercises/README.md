# Exercises: Understanding LSTM Networks

5 hands-on exercises to master LSTM fundamentals. Work through them in order - each builds on the previous!

---

## Exercise 1: Build LSTM from Scratch
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_01_build_lstm.py`

### Goal
Implement a complete LSTM cell from scratch in NumPy, including all 4 gates and backpropagation.

### What You'll Learn
- How each gate works mathematically
- How cell state and hidden state interact
- How gradients flow backward through the gates
- Why LSTMs solve vanishing gradients

### Tasks
1. Implement the forget gate: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. Implement the input gate: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
3. Implement the cell candidate: $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
4. Implement the output gate: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
5. Update cell state: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
6. Update hidden state: $h_t = o_t \odot \tanh(C_t)$
7. Implement backpropagation through time (BPTT)
8. Add gradient clipping

### Success Criteria
- Forward pass produces correct gate activations
- Cell state evolves properly
- Backward pass computes correct gradients
- Model can learn simple sequence (e.g., "hello" → "ello ")

### Hints
- Initialize forget gate bias to 1.0 (default to remembering)
- Use gradient clipping: `np.clip(grad, -5, 5)`
- Test each gate independently before combining
- Check shapes at every step!

---

## Exercise 2: Gate Activation Analysis
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_02_gate_analysis.py`

### Goal
Train an LSTM on a text dataset and analyze what each gate learns to do.

### What You'll Learn
- When forget gates activate (what to throw away)
- When input gates activate (what to add)
- When output gates activate (what to reveal)
- How gates coordinate to solve tasks

### Tasks
1. Train LSTM on simple text (e.g., repeated pattern "abcabc...")
2. Capture gate activations during forward pass
3. Plot heatmaps of each gate over time
4. Identify patterns:
   - Which gates are most active?
   - Do gates specialize (some always on/off)?
   - How do gates respond to specific characters?
5. Write a report on your findings

### Success Criteria
- Training loss decreases
- Gate visualizations are clear and interpretable
- You can explain why certain gates activate
- Report includes at least 3 insights

### Hints
- Use `visualization.py` functions: `plot_gate_activations()`, `analyze_gate_patterns()`
- Try simple patterns first (e.g., "ababab...")
- Look for gates that activate on specific characters
- Compare gate behavior early vs late in training

---

## Exercise 3: Ablation Study
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_03_ablation_study.py`

### Goal
Remove each gate one at a time and measure the impact on performance. Understand which components are critical.

### What You'll Learn
- Which gates are most important
- How gates complement each other
- What happens when you remove each gate
- Trade-offs between complexity and performance

### Tasks
1. Train baseline LSTM (all gates)
2. Train "No Forget Gate" LSTM (set $f_t = 1$)
3. Train "No Input Gate" LSTM (set $i_t = 1$)
4. Train "No Output Gate" LSTM (set $o_t = 1$)
5. Compare all 4 models on:
   - Training loss
   - Convergence speed
   - Sample quality
   - Long-range dependency performance

### Success Criteria
- All 4 models train successfully
- Clear comparison of performance
- Explanation of why each gate matters
- Insights on which gate is most critical

### Hints
- Keep all other hyperparameters constant
- Train for same number of iterations
- Use same random seed for fair comparison
- Expected: removing forget gate hurts most

---

## Exercise 4: Long-Range Dependencies
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_04_long_range_deps.py`

### Goal
Test LSTM's ability to remember information over different distances.

### What You'll Learn
- How far back LSTMs can "remember"
- When vanishing gradients still occur
- Practical limits of LSTM memory
- How to design tasks that test memory

### Tasks
1. Create synthetic task: "Remember the first character"
   - Input: `A _ _ _ _ ?` (N underscores)
   - Target: `A`
2. Test distances: 5, 10, 20, 50, 100 steps
3. Train LSTM on each distance
4. Measure accuracy vs distance
5. Compare with vanilla RNN (optional)

### Success Criteria
- Task is correctly implemented
- LSTM accuracy measured at each distance
- Clear plot of accuracy vs distance
- Analysis of where performance drops

### Hints
- Generate data programmatically
- Use random characters for "noise"
- Start with small distances to verify task
- LSTM should handle 50+ steps easily
- Vanilla RNN should fail beyond 10 steps

---

## Exercise 5: LSTM vs GRU Comparison
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_05_lstm_vs_gru.py`

### Goal
Implement a GRU (Gated Recurrent Unit) and compare it to LSTM.

### What You'll Learn
- GRU architecture (3 gates instead of 4)
- Trade-offs: speed vs capacity
- When to use GRU vs LSTM
- How simplifications affect performance

### Tasks
1. Implement GRU from scratch:
   - Reset gate: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
   - Update gate: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
   - Candidate: $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$
   - Hidden: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$
2. Train both LSTM and GRU on same dataset
3. Compare:
   - Training speed (time per iteration)
   - Final loss
   - Sample quality
   - Number of parameters
4. Write comparison report

### Success Criteria
- GRU correctly implemented
- Both models train successfully
- Fair comparison (same hyperparameters)
- Report explains trade-offs

### Hints
- GRU has 3 weight matrices (vs LSTM's 4)
- GRU is typically 20-30% faster
- Performance is often similar
- LSTM may be better for very long sequences
- GRU is simpler and often preferred today

---

## Bonus Challenges

### Challenge 1: Bidirectional LSTM
Implement a bidirectional LSTM that processes sequences forward AND backward. Combine the outputs.

### Challenge 2: Peephole Connections
Add "peephole" connections where gates can see the cell state:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t, C_{t-1}] + b_f)$$

### Challenge 3: Gradient Visualization
Implement gradient flow tracking and visualize how gradients change magnitude as they flow backward.

### Challenge 4: Layer Stacking
Stack multiple LSTM layers and see if deeper networks learn better.

### Challenge 5: Real-World Task
Train LSTM on a real task:
- Sentiment classification (movie reviews)
- Named entity recognition
- Time series prediction
- Music generation

---

## Tips for Success

### General Advice
- **Start simple**: Get basic version working first
- **Test incrementally**: Test each component before combining
- **Visualize everything**: Plot losses, gates, gradients
- **Compare baselines**: Always have a baseline to compare against
- **Debug systematically**: Print shapes, check ranges, verify equations

### Debugging Checklist
When things go wrong:
1. **Check shapes**: Print array shapes at each step
2. **Check ranges**: Gates should be [0,1], tanh [-1,1]
3. **Check gradients**: Should not be NaN or extremely large
4. **Check initialization**: Forget bias should be 1
5. **Check learning rate**: Try 0.001 first
6. **Check data**: Print samples to verify correctness

### Common Mistakes
- Forget to initialize forget gate bias to 1
- Mix up cell state (C) and hidden state (h)
- Forget to detach gradients between sequences
- Use wrong activation functions (tanh vs sigmoid)
- Don't clip gradients

---

## Solutions

Complete solutions are available in `solutions/` folder:
- `solution_01_lstm_implementation.py`
- `solution_02_gate_analysis.py`
- `solution_03_ablation_study.py`
- `solution_04_long_range_deps.py`
- `solution_05_lstm_vs_gru.py`

**Recommendation**: Try solving on your own first! Only check solutions if you're truly stuck.

---

## Estimated Time

| Exercise | Difficulty | Time       | Prerequisites |
|----------|-----------|------------|---------------|
| 1        | Hard      | 2-3 hours  | None          |
| 2        | Medium    | 1-2 hours  | Exercise 1    |
| 3        | Medium    | 1-2 hours  | Exercise 1    |
| 4        | Medium    | 1-2 hours  | Exercise 1    |
| 5        | Hard      | 2-3 hours  | Exercise 1    |

**Total**: 7-12 hours for all exercises

---

## Learning Path

```
Day 2 README → Day 2 Paper Notes → Day 2 Implementation → Day 2 Notebook
                                                               ↓
                                                          Exercise 1
                                                         (Build LSTM)
                                                               ↓
                                        ┌──────────────────────┼──────────────────────┐
                                        ↓                      ↓                      ↓
                                  Exercise 2              Exercise 3            Exercise 4
                                (Gate Analysis)        (Ablation Study)   (Long-Range Deps)
                                        └──────────────────────┼──────────────────────┘
                                                               ↓
                                                          Exercise 5
                                                       (LSTM vs GRU)
                                                               ↓
                                                          Day 3!
```

---

Good luck! Remember: **understanding comes from building**. Don't just read the code - type it out, break it, fix it, and make it your own.

**Questions?** Open an issue or discussion on GitHub.
