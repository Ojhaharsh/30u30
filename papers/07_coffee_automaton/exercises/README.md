# Exercises: Coffee Automaton - Complexity Theory

5 hands-on exercises to master complexity theory and emergent systems. Work through them in order - each builds on the previous!

---

## Exercise 1: Build the Coffee Automaton ⏱️⏱️
**Difficulty**: Medium  
**Time**: 1-2 hours  
**File**: `exercise_01_build_automaton.py`

### Goal
Implement a 1D/2D cellular automaton that models heat diffusion (the "coffee cooling" system).

### What You'll Learn
- How simple local rules create global behavior
- Heat diffusion dynamics
- Cellular automaton implementation
- Measuring complexity over time

### Tasks
1. Create a grid representing temperature values
2. Implement heat diffusion: each cell averages with neighbors
3. Add optional cooling (energy loss to environment)
4. Track and plot temperature evolution over time
5. Measure when the system reaches equilibrium

### Success Criteria
- Grid updates correctly at each timestep
- Heat spreads from hot spots to neighbors
- System eventually reaches uniform temperature
- Can visualize the evolution as animation

### Hints
- Start with 1D (simpler) before trying 2D
- Use numpy for efficient array operations
- Diffusion rate controls how fast heat spreads
- Check conservation of energy (total heat should decrease if cooling enabled)

---

## Exercise 2: Measure Complexity ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_02_complexity_measure.py`

### Goal
Implement multiple complexity measures and track how they change as the coffee automaton evolves.

### What You'll Learn
- Shannon entropy calculation
- Logical depth approximation
- The complexity "peak" phenomenon
- Why complexity rises then falls

### Tasks
1. Implement Shannon entropy: $H = -\sum p_i \log p_i$
2. Implement gradient-based complexity (structure measure)
3. Combine into unified complexity metric
4. Run automaton and plot complexity over time
5. Find the peak complexity and analyze what the system looks like at that moment

### Success Criteria
- Entropy calculation is correct
- Complexity curve shows rise-then-fall pattern
- Peak complexity occurs in "middle" of evolution
- Can explain why complexity peaks when it does

### Hints
- Histogram temperatures into bins for entropy
- Gradient = `np.abs(np.diff(state)).sum()`
- Complexity ≈ entropy × structure_measure
- Try different initial conditions to see how peak changes

---

## Exercise 3: Game of Life Complexity ⏱️⏱️⏱️
**Difficulty**: Hard  
**Time**: 2-3 hours  
**File**: `exercise_03_game_of_life.py`

### Goal
Apply complexity measures to Conway's Game of Life and observe the rise-and-fall pattern.

### What You'll Learn
- Game of Life rules
- Discrete vs continuous automata
- How structure emerges and decays
- Pattern classification (still lifes, oscillators, spaceships)

### Tasks
1. Implement Game of Life rules
2. Add complexity measurement from Exercise 2
3. Test with different initial conditions:
   - Random soup
   - Glider gun
   - R-pentomino
4. Plot complexity curves for each
5. Identify when stable structures form vs when chaos dominates

### Success Criteria
- Game of Life runs correctly
- Complexity measures work on discrete states
- Can identify different phases (chaos → structure → stability)
- Report on which initial conditions lead to highest complexity

### Hints
- Use `scipy.ndimage.convolve` for neighbor counting
- Life rule: exactly 3 neighbors = birth, 2-3 neighbors = survive
- Random soup: start with ~30% alive cells
- Watch for oscillators that keep complexity from fully decaying

---

## Exercise 4: Neural Network Training Complexity ⏱️⏱️⏱️⏱️
**Difficulty**: Very Hard  
**Time**: 3-4 hours  
**File**: `exercise_04_neural_training.py`

### Goal
Monitor the complexity of neural network activations during training. Does training follow a complexity curve?

### What You'll Learn
- How neural networks evolve during training
- Internal representations at different training stages
- Connection between complexity and generalization
- When does "emergence" happen in neural nets?

### Tasks
1. Create simple neural network (e.g., for MNIST)
2. Capture activations at each layer during training
3. Compute complexity of activations over training
4. Plot complexity curves for each layer
5. Correlate complexity peaks with:
   - Training loss
   - Validation accuracy
   - Generalization gap

### Success Criteria
- Network trains successfully
- Activation complexity is computed at checkpoints
- Visualization shows complexity evolution
- Analysis explains relationship between complexity and learning

### Hints
- Use hooks to capture activations: `layer.register_forward_hook()`
- Start with small network (2-3 layers)
- Sample activations on fixed batch for consistency
- Look for phase transitions in complexity

---

## Exercise 5: Cosmological Model ⏱️⏱️⏱️⏱️⏱️
**Difficulty**: Expert  
**Time**: 4-5 hours  
**File**: `exercise_05_cosmological_model.py`

### Goal
Build a toy model of cosmic evolution showing complexity peaking during "structure formation era."

### What You'll Learn
- Toy cosmology simulation
- Gravity-like attraction rules
- How structure forms from uniform beginnings
- Why life exists "now" in cosmic history

### Tasks
1. Create particle simulation with:
   - Expansion (like universe expanding)
   - Gravity (particles attract)
   - Random thermal motion
2. Measure complexity as particles cluster
3. Show three eras:
   - Early: Hot uniform soup (low complexity)
   - Middle: Structure formation (high complexity)
   - Late: Maximum entropy (low complexity again)
4. Plot complexity over "cosmic time"
5. Identify when "life" would be possible (peak complexity era)

### Success Criteria
- Particles cluster realistically
- Three eras are visible in simulation
- Complexity peaks during structure formation
- Can explain why complexity must eventually decrease

### Hints
- Use N-body simulation with softened gravity
- Add expansion factor that increases over time
- Complexity = balance of order (clusters) and disorder (randomness)
- This is a simplified model - focus on qualitative behavior

---

## Bonus Challenges 🌟

### Challenge 1: Edge of Chaos
Find the exact parameters where the coffee automaton shows maximum complexity. Plot a phase diagram.

### Challenge 2: Information Flow
Measure mutual information between distant regions. How does information propagate?

### Challenge 3: Predict the Peak
Given initial conditions, can you predict when complexity will peak without simulating?

### Challenge 4: Reversibility
What happens if you run the coffee automaton backwards in time? Does complexity behave symmetrically?

### Challenge 5: Multi-Scale Complexity
Measure complexity at different spatial scales. Is there a "fractal" structure to complexity?

---

## Tips for Success

### General Advice
- **Start simple**: 1D before 2D, small grids before large
- **Visualize everything**: Animations help understanding
- **Compare**: Always have baseline to compare against
- **Document patterns**: Note what you observe at each stage

### Debugging Checklist
When things go wrong:
1. **Check boundaries**: How do edges behave?
2. **Check conservation**: Is total energy preserved when expected?
3. **Check scales**: Are values in reasonable ranges?
4. **Check time**: Is simulation running long enough?

### Common Mistakes
- Forgetting to copy arrays (modifying in-place)
- Wrong boundary conditions
- Entropy calculation on wrong axis
- Not normalizing complexity measures

---

## Solutions

Complete solutions are available in `solutions/` folder:
- `solution_01_build_automaton.py`
- `solution_02_complexity_measure.py`
- `solution_03_game_of_life.py`
- `solution_04_neural_training.py`
- `solution_05_cosmological_model.py`

**Recommendation**: Try solving on your own first! Only check solutions if you're truly stuck.

---

## Estimated Time

| Exercise | Difficulty | Time       | Prerequisites |
|----------|-----------|------------|---------------|
| 1        | Medium    | 1-2 hours  | None          |
| 2        | Hard      | 2-3 hours  | Exercise 1    |
| 3        | Hard      | 2-3 hours  | Exercise 2    |
| 4        | V. Hard   | 3-4 hours  | Exercise 2    |
| 5        | Expert    | 4-5 hours  | Exercise 2    |

**Total**: 12-17 hours for all exercises

---

## Learning Path

```
Day 7 README → Paper Notes → Implementation → Notebook
                                                   ↓
                                             Exercise 1
                                          (Build Automaton)
                                                   ↓
                                             Exercise 2
                                         (Measure Complexity)
                                                   ↓
                               ┌───────────────────┼───────────────────┐
                               ↓                   ↓                   ↓
                          Exercise 3          Exercise 4          Exercise 5
                        (Game of Life)    (Neural Training)    (Cosmology)
                               └───────────────────┼───────────────────┘
                                                   ↓
                                              Day 8! 🚀
```

---

Good luck! Remember: **complexity emerges from simplicity**. Start with the simplest version and watch the magic happen! ✨

**Questions?** Check the main README or open an issue.
