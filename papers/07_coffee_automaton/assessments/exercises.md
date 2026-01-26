# Day 7: Coffee Automaton - Exercises

Practice problems to master complexity theory and emergent behavior.

---

## Exercise 1: Edge of Chaos Discovery ⭐⭐⭐

**Goal**: Find the critical point where complexity is maximized.

**Task**:
Implement a parameter sweep to find the "edge of chaos" - the diffusion rate that produces maximum complexity.

**Requirements**:
1. Create a Coffee Automaton with varying diffusion rates (0.05 to 0.30)
2. For each diffusion rate, run simulation for 100 steps
3. Measure peak complexity using `calculate_local_complexity()`
4. Plot complexity vs diffusion rate
5. Identify the critical point (peak complexity)

**Starting Code**:
```python
from implementation import CoffeeAutomaton, ComplexityMeasures
import numpy as np
import matplotlib.pyplot as plt

# Your code here
diffusion_rates = np.linspace(0.05, 0.30, 20)
peak_complexities = []

for diff_rate in diffusion_rates:
    # Create automaton
    coffee = CoffeeAutomaton(
        size=40,
        diffusion_rate=diff_rate,
        cooling_rate=0.02,
        noise_level=0.01
    )
    
    # Add initial hotspot
    coffee.add_hotspot(20, 20, intensity=30, radius=5)
    
    # TODO: Run simulation and track complexity
    # TODO: Find peak complexity
    # TODO: Store result

# TODO: Plot results and find critical point
```

**Expected Output**:
- Plot showing complexity vs diffusion rate
- Clear peak indicating edge of chaos
- Critical diffusion rate value (should be around 0.10-0.15)

**Hints**:
- Use `calc.calculate_local_complexity(coffee.grid)` each step
- Track the maximum complexity reached during evolution
- The peak should form a clear "mountain" shape

**Challenge**: Can you find how the critical point changes with different noise levels?

---

## Exercise 2: Pattern Classification ⭐⭐⭐⭐

**Goal**: Build a classifier to identify different emergent patterns.

**Task**:
Create a system that can automatically detect and classify different types of patterns in the Coffee Automaton:
- Still Life (stable patterns)
- Oscillators (periodic patterns)
- Chaos (random noise)
- Edge of Chaos (complex patterns)

**Requirements**:
1. Generate examples of each pattern type by varying parameters
2. Extract features (entropy, complexity, temporal variance)
3. Build a simple classifier (can use sklearn)
4. Test on new parameter combinations
5. Report classification accuracy

**Starting Code**:
```python
from implementation import CoffeeAutomaton, ComplexityMeasures
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def extract_features(coffee, steps=50):
    """Extract features from automaton evolution"""
    calc = ComplexityMeasures()
    
    features = {
        'complexity': [],
        'entropy': [],
        'variance': []
    }
    
    for step in range(steps):
        coffee.step()
        # TODO: Calculate features
        # complexity = calc.calculate_local_complexity(coffee.grid)
        # entropy = calc.calculate_entropy(coffee.grid)
        # variance = np.var(coffee.grid)
    
    # TODO: Return feature vector (mean, std, max, etc.)
    return feature_vector

# TODO: Generate training data for each pattern type
# TODO: Train classifier
# TODO: Test and report accuracy
```

**Expected Output**:
- Classification accuracy > 85%
- Confusion matrix showing pattern types
- Feature importance plot

**Pattern Generation Hints**:
- Still Life: Low diffusion (0.05), low noise (0.001)
- Oscillators: Medium diffusion (0.12), low noise (0.005)
- Chaos: High diffusion (0.25), high noise (0.03)
- Edge of Chaos: Medium diffusion (0.12), medium noise (0.01)

---

## Exercise 3: Information Flow Analysis ⭐⭐⭐⭐⭐

**Goal**: Measure how information propagates through the system.

**Task**:
Implement mutual information calculation to track information flow over time and space.

**Requirements**:
1. Record grid states over time (100+ steps)
2. Calculate mutual information between consecutive timesteps
3. Calculate spatial mutual information (between neighboring cells)
4. Plot information flow dynamics
5. Identify critical transitions

**Starting Code**:
```python
from implementation import CoffeeAutomaton
import numpy as np
from sklearn.metrics import mutual_info_score

def mutual_information_temporal(grid_t0, grid_t1, bins=10):
    """Calculate MI between two timesteps"""
    # Discretize grids
    flat_t0 = np.digitize(grid_t0.flatten(), bins=bins)
    flat_t1 = np.digitize(grid_t1.flatten(), bins=bins)
    
    # Calculate mutual information
    mi = mutual_info_score(flat_t0, flat_t1)
    return mi

def analyze_information_flow():
    coffee = CoffeeAutomaton(size=40, diffusion_rate=0.12)
    coffee.add_hotspot(20, 20, intensity=30, radius=5)
    
    grids = []
    mi_values = []
    
    # TODO: Collect grid states
    for step in range(100):
        coffee.step()
        grids.append(coffee.grid.copy())
    
    # TODO: Calculate MI between consecutive steps
    for i in range(1, len(grids)):
        mi = mutual_information_temporal(grids[i-1], grids[i])
        mi_values.append(mi)
    
    # TODO: Plot MI over time
    # TODO: Identify critical points (sharp drops/rises in MI)
    
    return mi_values

# TODO: Run analysis for different parameter regimes
```

**Expected Output**:
- MI vs time plot showing information dynamics
- Higher MI at edge of chaos
- Lower MI in ordered and chaotic regimes

**Challenge**: Calculate spatial mutual information between neighboring cells to find "information highways" in the grid.

---

## Exercise 4: Neural Network Initialization Experiment ⭐⭐⭐⭐

**Goal**: Apply edge-of-chaos principles to neural network initialization.

**Task**:
Use Coffee Automaton insights to initialize a neural network at the critical point.

**Requirements**:
1. Create a simple neural network (3 layers, 100 neurons each)
2. Implement Coffee Automaton-inspired initialization
3. Compare with standard initialization (Xavier, Kaiming)
4. Train on MNIST or CIFAR-10
5. Compare convergence speed and final accuracy

**Starting Code**:
```python
import torch
import torch.nn as nn
from implementation import CoffeeAutomaton, ComplexityMeasures

class EdgeOfChaosInit:
    """Initialize network at edge of chaos"""
    
    @staticmethod
    def initialize(layer):
        """Find optimal initialization by maximizing activation complexity"""
        # TODO: Try different initialization scales
        scales = np.logspace(-2, 0, 20)
        complexities = []
        
        for scale in scales:
            # Initialize with this scale
            nn.init.normal_(layer.weight, 0, scale)
            
            # Generate random input
            x = torch.randn(100, layer.in_features)
            
            # Forward pass
            with torch.no_grad():
                out = layer(torch.relu(x))
            
            # Measure complexity
            calc = ComplexityMeasures()
            # TODO: Adapt complexity measure for activations
            
        # TODO: Find scale with maximum complexity
        # TODO: Apply optimal initialization

class SimpleNN(nn.Module):
    def __init__(self, init_method='edge_of_chaos'):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        
        # TODO: Apply chosen initialization method
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# TODO: Train networks with different initializations
# TODO: Compare convergence curves
# TODO: Report final accuracies
```

**Expected Output**:
- Training curves for each initialization method
- Edge-of-chaos initialization should converge faster
- Table comparing final accuracies

---

## Exercise 5: Lyapunov Exponent Calculator ⭐⭐⭐⭐⭐

**Goal**: Measure chaos in the system using Lyapunov exponents.

**Task**:
Calculate the Lyapunov exponent to quantify sensitivity to initial conditions.

**Requirements**:
1. Create two Coffee Automatons with nearly identical initial conditions
2. Track divergence over time
3. Calculate Lyapunov exponent: λ = lim (1/t) log(d(t)/d(0))
4. Plot exponent vs diffusion rate
5. Verify negative λ (order), zero λ (edge), positive λ (chaos)

**Starting Code**:
```python
from implementation import CoffeeAutomaton
import numpy as np
import matplotlib.pyplot as plt

def calculate_lyapunov(diffusion_rate, steps=1000, perturbation=1e-6):
    """Calculate Lyapunov exponent for given parameters"""
    
    # Create two nearly identical automatons
    coffee1 = CoffeeAutomaton(size=40, diffusion_rate=diffusion_rate)
    coffee2 = CoffeeAutomaton(size=40, diffusion_rate=diffusion_rate)
    
    # Add same initial conditions
    coffee1.add_hotspot(20, 20, intensity=30, radius=5)
    coffee2.add_hotspot(20, 20, intensity=30, radius=5)
    
    # Add tiny perturbation to second automaton
    coffee2.grid[20, 20] += perturbation
    
    lyapunov_sum = 0
    
    for step in range(steps):
        coffee1.step()
        coffee2.step()
        
        # Calculate distance between states
        distance = np.linalg.norm(coffee1.grid - coffee2.grid)
        
        if distance > 0:
            # TODO: Accumulate log of divergence
            lyapunov_sum += np.log(distance / perturbation)
            
            # Renormalize to prevent overflow
            if step % 10 == 0:
                # TODO: Rescale perturbation
                pass
    
    # Calculate average Lyapunov exponent
    lyapunov = lyapunov_sum / steps
    return lyapunov

# TODO: Calculate for range of diffusion rates
diffusion_rates = np.linspace(0.05, 0.30, 20)
lyapunov_exponents = []

for diff_rate in diffusion_rates:
    lyap = calculate_lyapunov(diff_rate)
    lyapunov_exponents.append(lyap)

# TODO: Plot and identify regimes
# Negative λ → Ordered
# Zero λ → Critical (edge of chaos)
# Positive λ → Chaotic
```

**Expected Output**:
- Plot of Lyapunov exponent vs diffusion rate
- Clear transition from negative to positive
- Zero crossing near edge of chaos

**Theory Check**:
- What does λ < 0 mean? (Stable, attracting)
- What does λ = 0 mean? (Critical point)
- What does λ > 0 mean? (Chaotic, diverging)

---

## Bonus Challenge: Self-Organizing Criticality ⭐⭐⭐⭐⭐

**Goal**: Demonstrate that the system naturally evolves toward the critical point.

**Task**:
1. Start system far from criticality (very low or very high diffusion)
2. Implement adaptive rule: increase/decrease diffusion based on local complexity
3. Show system self-organizes to edge of chaos
4. Plot complexity and diffusion rate over time

**Adaptive Rule Example**:
```python
def adaptive_step(coffee, calc, target_complexity=1.0):
    """Adjust diffusion to maintain edge of chaos"""
    coffee.step()
    
    complexity = calc.calculate_local_complexity(coffee.grid).mean()
    
    # Adjust diffusion rate
    if complexity < target_complexity:
        coffee.diffusion_rate *= 1.01  # Increase slightly
    else:
        coffee.diffusion_rate *= 0.99  # Decrease slightly
    
    # Keep in valid range
    coffee.diffusion_rate = np.clip(coffee.diffusion_rate, 0.05, 0.30)
    
    return complexity
```

This demonstrates **self-organized criticality** - systems naturally evolve toward the edge of chaos!

---

## Submission Guidelines

For each exercise, submit:
1. **Code**: Well-commented Python implementation
2. **Results**: Plots, tables, and numerical outputs
3. **Analysis**: 1-2 paragraphs interpreting results
4. **Insights**: What did you learn about emergence/complexity?

**Grading Rubric**:
- Code Quality: 30%
- Correct Implementation: 40%
- Analysis & Insights: 30%

---

## Tips for Success

1. **Start simple**: Get basic version working first, then add complexity
2. **Visualize everything**: Plot intermediate results to debug
3. **Sanity checks**: Verify results make physical sense
4. **Compare regimes**: Test in ordered, edge, and chaotic regimes
5. **Ask questions**: If stuck, review the README and paper notes

Good luck! Remember: **The most interesting phenomena happen at the edge of chaos!** 🔥✨
