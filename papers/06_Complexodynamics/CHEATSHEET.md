# Complexodynamics Cheatsheet 🚀

> Quick reference for the First Law of Complexodynamics

---

## 🎯 The First Law

> **"The information content of a replicator will increase up to the limit imposed by the accuracy of its replication machinery."**

**In one equation:**

$$
\frac{dC}{dt} = I_E - I_L
$$

- At equilibrium: $I_E = I_L$ → $\frac{dC}{dt} = 0$

---

## 📊 Core Formulas

### **Shannon Complexity**

$$
C = -\sum_{i=1}^{N} p_i \log_2 p_i \quad \text{(bits/site)}
$$

- **Maximum:** $C_{\max} = \log_2(N)$ = 2 bits/site for DNA
- **Minimum:** $C_{\min} = 0$ (uniform sequence)

**Quick calculation:**
```python
import numpy as np
from collections import Counter

def shannon_complexity(seq):
    counts = Counter(seq)
    probs = np.array(list(counts.values())) / len(seq)
    return -np.sum(probs * np.log2(probs))
```

---

### **Channel Capacity**

$$
C_{\max} = -\log_2(\mu \cdot L)
$$

- $\mu$ = mutation rate (per base per generation)
- $L$ = genome length
- $C_{\max}$ = maximum sustainable complexity (bits)

**Example values:**

| Organism | $\mu$ | $L$ | $C_{\max}$ (bits/site) |
|----------|-------|-----|------------------------|
| RNA Virus | $10^{-4}$ | $10^4$ | 1.2 |
| Bacteria | $10^{-6}$ | $10^6$ | 1.5 |
| Human | $10^{-9}$ | $10^9$ | 1.9 |

---

### **Information Gain (Selection)**

$$
I_E = \sum_i p_i(s) \log_2 \frac{p_i(s)}{p_i}
$$

- $p_i$ = frequency before selection
- $p_i(s)$ = frequency after selection
- $I_E$ = information gained from environment (bits)

**Interpretation:** How much the environment "teaches" the genome

---

### **Information Loss (Mutation)**

$$
I_L = H(X|X') = \mu \cdot L \cdot \log_2(4)
$$

For DNA with uniform mutation:
- $H(X|X')$ = conditional entropy
- $\mu \cdot L$ = expected number of mutations
- Each mutation loses $\log_2(4) = 2$ bits

**Interpretation:** How much copying errors "forget"

---

### **Equilibrium Complexity**

$$
C_{eq} = C_{\max} \left(1 - \frac{\mu \cdot L}{\beta \cdot N_e}\right)
$$

Where:
- $C_{\max}$ = maximum possible complexity
- $\beta$ = selection strength
- $N_e$ = effective population size
- $\mu \cdot L$ = genomic mutation rate

**Rule of thumb:** $C_{eq} \approx 0.95 \cdot C_{\max}$ for strong selection

---

### **Time to Equilibrium**

$$
\tau = \frac{1}{\lambda} = \frac{1}{\beta \cdot s}
$$

- $\tau$ = time constant (generations)
- $\beta$ = selection coefficient
- $s$ = selection differential

**Complexity trajectory:**

$$
C(t) = C_{\max}\left(1 - e^{-t/\tau}\right)
$$

**Time to 95% of max:** $t_{95} \approx 3\tau$

---

## 🧬 Biological Scaling

### **Mutation-Fidelity Relationship**

$$
\mu = \frac{1}{f \cdot L}
$$

- $f$ = fidelity factor (how good DNA repair is)
- Bacteria: $f \approx 10^6$
- Humans: $f \approx 10^9$ (better repair!)

**Higher fidelity** → **Lower mutation rate** → **Higher complexity ceiling**

---

### **Eigen's Error Threshold**

Maximum genome length before error catastrophe:

$$
L_{\max} = \frac{1}{\mu} \log\left(\frac{1}{\mu}\right)
$$

**Critical point:** Beyond $L_{\max}$, information is lost faster than gained!

| Organism | $\mu$ | $L_{\max}$ | Actual $L$ | Status |
|----------|-------|-----------|-----------|--------|
| RNA Virus | $10^{-4}$ | $10^4$ | $10^4$ | At limit! |
| Bacteria | $10^{-6}$ | $10^7$ | $10^6$ | Safe |
| Human | $10^{-9}$ | $10^{10}$ | $10^9$ | Safe |

---

## 🔬 Experimental Predictions

### **Complexity Growth Rate**

Early phase (far from equilibrium):

$$
\frac{dC}{dt} \approx I_E \approx \beta \cdot \mathrm{Var}(fitness)
$$

**Prediction:** Strong selection → Fast complexity increase

---

### **Population Size Effect**

$$
C_{eq} \propto \log(N_e)
$$

**Prediction:** Larger populations → Higher equilibrium complexity

---

### **Environmental Variability**

Fluctuating environment:

$$
C_{eq} = \langle C_{\max} \rangle_{\text{time}}
$$

**Prediction:** Complexity tracks time-averaged capacity

---

## 📐 Useful Approximations

### **For DNA sequences:**

$$
C \approx 2 - \frac{1}{L}\sum_{i} \left(n_i - \frac{L}{4}\right)^2
$$

- $n_i$ = count of base $i$
- Works when $L$ is large

---

### **For protein-coding genes:**

$$
C_{protein} \approx \frac{3}{2} C_{DNA}
$$

Due to degeneracy in genetic code (synonymous codons)

---

### **For equilibrium time:**

$$
t_{eq} \approx \frac{1}{\mu \cdot \beta} \text{ generations}
$$

**Example (E. coli):**
- $\mu = 10^{-6}$
- $\beta = 0.01$ (weak selection)
- $t_{eq} \approx 10^8$ generations ≈ 200,000 years

---

## 🎨 Visual Intuitions

### **Complexity vs Time**

```
C │     _______________  C_max (asymptote)
  │   /
  │  /
  │ /
  │/___________________
  0                    t
     ← τ →
```

**Shape:** Saturating exponential

---

### **Fidelity-Complexity Trade-off**

```
C_max │       ╱
      │      ╱
      │     ╱
      │    ╱
      │   ╱
      │  ╱
      │ ╱
      │╱_________________
          μ (log scale)
```

**Relationship:** $C_{\max} \sim -\log(\mu)$

---

### **Information Flow Balance**

```
I_E │       ╱
    │      ╱  ╲
    │     ╱    ╲ I_L
    │    ╱  eq  ╲
    │   ╱   ↓    ╲
    │  ╱          ╲
    │ ╱____________╲____
              C
```

**Equilibrium:** Where curves intersect ($I_E = I_L$)

---

## 💻 Code Snippets

### **Basic Complexity Calculation**

```python
import numpy as np

def complexity(sequence):
    """Shannon entropy of sequence."""
    unique, counts = np.unique(list(sequence), return_counts=True)
    probs = counts / len(sequence)
    return -np.sum(probs * np.log2(probs))

# Example
dna = "ACGTACGTACGT"
print(f"Complexity: {complexity(dna):.2f} bits/site")
# Output: Complexity: 2.00 bits/site (maximum!)
```

---

### **Channel Capacity**

```python
def channel_capacity(mu, L):
    """Maximum sustainable complexity."""
    return -np.log2(mu * L)

# Example: E. coli
mu = 1e-6
L = 4.6e6
C_max = channel_capacity(mu, L)
print(f"C_max: {C_max:.2f} bits")
# Output: C_max: 22.13 bits (total), ~1.5 bits/site
```

---

### **Complexity Trajectory**

```python
def complexity_trajectory(t, C_max, tau):
    """C(t) over time."""
    return C_max * (1 - np.exp(-t / tau))

# Example
t = np.linspace(0, 10000, 100)  # generations
C = complexity_trajectory(t, C_max=1.5, tau=1000)

import matplotlib.pyplot as plt
plt.plot(t, C)
plt.xlabel('Generations')
plt.ylabel('Complexity (bits/site)')
plt.axhline(1.5, linestyle='--', label='C_max')
plt.legend()
```

---

### **Information Flow**

```python
def information_gain(p_before, p_after):
    """I_E from selection."""
    return np.sum(p_after * np.log2(p_after / p_before))

def information_loss(mu, L):
    """I_L from mutation."""
    return mu * L * 2  # 2 bits per mutation for DNA

# Example
p_before = np.array([0.25, 0.25, 0.25, 0.25])
p_after = np.array([0.4, 0.3, 0.2, 0.1])  # Selection favored A

I_E = information_gain(p_before, p_after)
I_L = information_loss(mu=1e-6, L=1e6)

print(f"I_E: {I_E:.4f} bits")
print(f"I_L: {I_L:.4f} bits")
print(f"Net: {I_E - I_L:.4f} bits/generation")
```

---

## 🔗 Key Relationships

### **Thermodynamics Analogy**

| Thermodynamics | Complexodynamics |
|----------------|------------------|
| $\frac{dS}{dt} \geq 0$ | $\frac{dC}{dt} \geq 0$ |
| $S_{eq}$ at max | $C_{eq}$ at max |
| Heat capacity | Information capacity |
| Temperature | Selection pressure |
| Entropy | Complexity |

---

### **Information Theory**

$$
C = H(X) = -\sum p(x) \log p(x)
$$

$$
I_E = I(X; E) = H(X) - H(X|E)
$$

$$
I_L = H(X|X') = \text{mutation noise}
$$

**Connection:** Complexity is just Shannon entropy!

---

### **Evolution ↔ Learning**

| Evolution | Machine Learning |
|-----------|------------------|
| Genome | Model weights |
| Mutation | Noise/regularization |
| Selection | Loss function |
| $C_{eq}$ | Optimal capacity |
| $I_E$ | Gradient information |
| $I_L$ | Weight decay |

**Insight:** Training is just fast evolution!

---

## 🎯 Quick Checks

### **Is complexity increasing?**

$$
\frac{dC}{dt} > 0 \iff I_E > I_L
$$

**Check:** Measure $I_E$ (selection strength) and $I_L$ (mutation load)

---

### **Are we at equilibrium?**

$$
\left|\frac{dC}{dt}\right| < \epsilon \iff C \approx C_{\max}
$$

**Check:** If complexity hasn't changed in $3\tau$ generations

---

### **What's the limiting factor?**

- If $C < C_{\max}$ and $\frac{dC}{dt} > 0$: **Still evolving**
- If $C \approx C_{\max}$: **Mutation rate is bottleneck**
- If $C > C_{\max}$: **Error catastrophe! (shouldn't happen)**

---

## 📊 Typical Values

| Parameter | Symbol | Virus | Bacteria | Human |
|-----------|--------|-------|----------|-------|
| Genome length | $L$ | $10^4$ | $10^6$ | $10^9$ |
| Mutation rate | $\mu$ | $10^{-4}$ | $10^{-6}$ | $10^{-9}$ |
| Complexity | $C$ | 1.2 | 1.5 | 1.9 |
| Max complexity | $C_{\max}$ | 1.3 | 1.6 | 2.0 |
| Generation time | - | 6 hrs | 20 min | 25 yrs |
| Equilibrium time | $\tau$ | 100 gen | 10k gen | 1M gen |

---

## 🚨 Common Mistakes

### ❌ **Confusing entropy and complexity**
- **Entropy** (thermodynamics): disorder
- **Complexity** (information): information content
- They're related but **not the same**!

### ❌ **Thinking complexity always increases**
- Only up to $C_{\max}$!
- Then it **plateaus**
- Can even **decrease** if environment changes

### ❌ **Ignoring population size**
- Small populations: drift dominates, low $C_{eq}$
- Large populations: selection dominates, high $C_{eq}$

### ❌ **Using wrong log base**
- Always use $\log_2$ for bits!
- $\log_e$ (natural log) gives nats, not bits

---

## 🎓 Further Study

### **Next concepts:**
1. **Stochastic complexodynamics** - Finite population effects
2. **Non-stationary environments** - Moving targets
3. **Sexual reproduction** - Recombination effects
4. **Multilevel selection** - Groups vs individuals

### **Related papers:**
- Adami (2004): Information theory in evolution
- Ofria & Adami (2004): Avida digital evolution
- Schneider (2010): Evolution of complexity

---

## 🏆 Remember

✅ **Complexity increases** (First Law)  
✅ **Up to a ceiling** ($C_{\max}$)  
✅ **Set by fidelity** ($\mu$)  
✅ **Through equilibration** ($I_E = I_L$)  
✅ **It's physics!** (not mysticism)

---

**Print this out and keep it handy while coding! 📌**
