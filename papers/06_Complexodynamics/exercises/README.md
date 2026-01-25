# Complexodynamics Exercises

Practice problems for mastering the First Law of Complexodynamics!

## 📁 Exercise Structure

Each exercise has:
- `exercise_XX_name.py` - Problem statement with TODO sections
- `solutions/solution_XX_name.py` - Complete working solution
- Test cases to verify your implementation

## 📊 Difficulty Levels

⭐ **Easy** (30-45 min) - Single concept, straightforward implementation  
⭐⭐ **Medium** (45-90 min) - Multiple concepts, requires integration  
⭐⭐⭐ **Hard** (90-120 min) - Complex system, research + implementation

---

## Exercise List

### ⭐ Exercise 1: Shannon Complexity
**Time:** 30 minutes  
**File:** `exercise_01_shannon_complexity.py`

Implement Shannon entropy calculator for DNA sequences.

**Learning goals:**
- Understand information content measurement
- Frequency counting and probability calculation
- Edge case handling

**Key concepts:**
- $C = -\sum p_i \log_2 p_i$
- Maximum complexity for DNA: 2 bits/site

---

### ⭐⭐ Exercise 2: Information Flow
**Time:** 45 minutes  
**File:** `exercise_02_information_flow.py`

Calculate information gain (I_E) and loss (I_L) rates.

**Learning goals:**
- Model selection pressure as information flow
- Calculate mutation-induced information loss
- Find equilibrium conditions

**Key concepts:**
- $I_E$ = KL divergence between pre/post-selection
- $I_L = \mu \cdot L \cdot \log_2(alphabet\_size)$
- Equilibrium: $I_E = I_L$

---

### ⭐⭐ Exercise 3: Channel Capacity
**Time:** 45 minutes  
**File:** `exercise_03_channel_capacity.py`

Implement fidelity-complexity trade-off calculations.

**Learning goals:**
- Calculate maximum sustainable complexity
- Compare different noise models
- Understand Eigen's error threshold

**Key concepts:**
- Simple model: $C_{max} = -\log_2(\mu \cdot L)$
- Gaussian channel capacity
- Binomial noise model
- Error threshold: $L_{max} \sim \frac{1}{\mu} \log(1/\mu)$

---

### ⭐⭐⭐ Exercise 4: Evolutionary Dynamics
**Time:** 90 minutes  
**File:** `exercise_04_evolutionary_dynamics.py`

Build full population-based evolutionary simulator.

**Learning goals:**
- Implement mutation-selection-reproduction cycle
- Track complexity over time
- Detect equilibrium automatically
- Population size effects

**Key concepts:**
- Population dynamics
- Fitness-proportional selection
- Complexity trajectory: $C(t) = C_{max}(1 - e^{-t/\tau})$
- Stochastic effects in finite populations

---

### ⭐⭐⭐ Exercise 5: Real Genome Analysis
**Time:** 90 minutes  
**File:** `exercise_05_genome_analysis.py`

Analyze actual genomic data using complexodynamics.

**Learning goals:**
- Parse FASTA format files
- Calculate complexity of real genomes
- Infer mutation rates from codon usage
- Predict evolutionary status

**Key concepts:**
- FASTA parsing
- Genome-scale complexity calculation
- Codon usage bias as mutation signal
- Distance to C_max as evolutionary indicator

---

## 🎯 How to Use These Exercises

### 1. Start with Exercise 1
Build up from fundamentals!

### 2. Read the TODO sections
Each exercise has clear instructions in comments.

### 3. Run tests frequently
```bash
python exercise_01_shannon_complexity.py
```

### 4. Check solutions when stuck
But try for at least 15 minutes first!

### 5. Extend and experiment
Add your own test cases and variations.

---

## 💡 Tips for Success

### **For Easy Exercises (⭐)**
- Read the docstrings carefully
- Start with edge cases (empty input, single element)
- Use print statements to debug
- Test with simple examples first

### **For Medium Exercises (⭐⭐)**
- Break into smaller functions
- Draw diagrams of the algorithm
- Compare output to theory
- Use visualization to check correctness

### **For Hard Exercises (⭐⭐⭐)**
- Start with a simple version first
- Incremental development (one feature at a time)
- Profile performance on large inputs
- Compare to published results

---

## 🧪 Testing Your Solutions

Each exercise includes test cases:

```python
if __name__ == '__main__':
    # Test case 1
    result = your_function(test_input)
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test case 2
    ...
    
    print("✅ All tests passed!")
```

**Pro tip:** Add your own test cases!

---

## 📚 Background Reading

Before starting:
1. Read [README.md](../README.md) - Main tutorial
2. Skim [CHEATSHEET.md](../CHEATSHEET.md) - Quick reference
3. Review [paper_notes.md](../paper_notes.md) - ELI5 explanations

During exercises:
- Refer to `implementation.py` for examples
- Check [CHEATSHEET.md](../CHEATSHEET.md) for formulas
- Use `visualization.py` to plot your results

---

## 🏆 Challenge Mode

Finished all exercises? Try these:

1. **Speed challenge:** Optimize Exercise 4 to run 10x faster
2. **Accuracy challenge:** Match paper Figure 2 within 1% error
3. **Extension challenge:** Add sexual reproduction (recombination)
4. **Research challenge:** Test on real genome data (download from NCBI)
5. **Teaching challenge:** Explain your solution to someone else

---

## 🤝 Getting Help

**Stuck?**
1. Re-read the problem statement
2. Check the CHEATSHEET for formulas
3. Look at `implementation.py` for similar code
4. Peek at the solution (but try first!)
5. Open an issue on GitHub

**Found a bug?**
- Open an issue with details
- Include your code and error message
- Tag it with "exercise"

---

## 📈 Progress Tracking

- [ ] Exercise 1: Shannon Complexity
- [ ] Exercise 2: Information Flow
- [ ] Exercise 3: Channel Capacity
- [ ] Exercise 4: Evolutionary Dynamics
- [ ] Exercise 5: Real Genome Analysis

**Completion time:** _____ hours

---

## 🎓 What You'll Learn

By completing all exercises, you'll master:
- Information theory fundamentals
- Evolutionary dynamics simulation
- Complexity measurement techniques
- Real genomic data analysis
- Trade-offs in biological systems

**These skills transfer to:**
- Machine learning (capacity, regularization)
- Optimization (equilibration, convergence)
- Bioinformatics (sequence analysis)
- Systems biology (population dynamics)

---

**Ready? Start with Exercise 1!** →  `exercise_01_shannon_complexity.py`

🚀 Good luck!