# Contributing to 30u30

Thank you for your interest in making this the **best learning resource** for Ilya's 30 papers! 🙏

This guide will help you contribute effectively.

---

## Ways to Contribute

### 🐛 Report Bugs

Found a bug in the code? Typo in documentation?

1. Check if it's already reported in [Issues](../../issues)
2. If not, create a new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version)

### 💡 Suggest Improvements

Have ideas for:
- Better explanations?
- Additional exercises?
- More visualizations?
- Code optimizations?

Open a [Discussion](../../discussions) or [Issue](../../issues)!

### 📝 Improve Documentation

Help make explanations clearer:
- Fix typos
- Add examples
- Improve analogies
- Translate to other languages

### 💻 Contribute Code

Add implementations, exercises, or tests.

---

## Code Contribution Guidelines

### Before You Start

1. **Check existing issues** - Someone might already be working on it
2. **Discuss major changes** - Open an issue/discussion first
3. **One issue per PR** - Keep changes focused

### Setting Up Development Environment

```bash
# Fork the repo on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/30u30.git
cd 30u30

# Create a branch
git checkout -b feature/your-feature-name

# Install dependencies
pip install -r requirements.txt

# Make your changes
# ...

# Test your changes
python papers/XX_Paper/implementation.py
```

### Code Style

We follow these principles:

#### 1. **Clarity over cleverness**
```python
# ✅ Good - Clear and readable
for i in range(len(data)):
    loss += compute_loss(data[i])

# ❌ Bad - Too clever
loss = sum(map(lambda x: compute_loss(x), data))
```

#### 2. **Heavy comments for educational code**
```python
# ✅ Good - Explain WHY
# Clip gradients to prevent explosion during backprop
# Without this, gradients can grow exponentially through time
np.clip(grad, -5, 5, out=grad)

# ❌ Bad - Just restating WHAT
# Clip gradients
np.clip(grad, -5, 5, out=grad)
```

#### 3. **Use NumPy, not PyTorch/TensorFlow**
We want educational, CPU-friendly code:
```python
# ✅ Good
h = np.tanh(Wxh @ x + Whh @ h_prev + bh)

# ❌ Bad - Don't use frameworks for core implementations
h = torch.tanh(torch.mm(Wxh, x) + torch.mm(Whh, h_prev) + bh)
```

#### 4. **Descriptive variable names**
```python
# ✅ Good
hidden_size = 100
learning_rate = 0.01
char_to_idx = {...}

# ❌ Bad
n = 100
lr = 0.01
c2i = {...}
```

#### 5. **Format code consistently**
We use `black` for formatting:
```bash
pip install black
black papers/XX_Paper/*.py
```

### File Organization

When adding a new paper, follow this structure:

```
papers/XX_Paper_Name/
├── README.md              # Complete guide
├── paper_notes.md         # ELI5 summary
├── implementation.py      # Core implementation
├── train_minimal.py       # Training script
├── visualization.py       # Plotting utilities
├── notebook.ipynb         # Jupyter notebook
├── exercises/
│   ├── README.md
│   ├── exercise_01_*.py
│   └── solutions/
│       └── exercise_01_solution.py
└── data/
    └── sample_data.txt
```

### Documentation Standards

#### README.md structure:
1. **Title + Paper link**
2. **Big Idea** (2-3 sentences)
3. **Why It Matters** (historical context)
4. **Prerequisites**
5. **The Intuition** (analogies)
6. **The Math** (with explanations)
7. **Implementation Walkthrough**
8. **Experiments** (what to try)
9. **Exercises** (5 practice problems)
10. **Going Further** (advanced topics)
11. **Resources** (videos, blogs, etc.)

#### paper_notes.md structure:
1. **ELI5 Summary** (for absolute beginners)
2. **Real-world Analogy**
3. **Key Concepts** (simplified)
4. **Why This Paper Changed Things**
5. **Connection to Modern AI**

### Pull Request Process

1. **Create a descriptive PR title:**
   ```
   Add Exercise 3 for Day 1: Custom Dataset Training
   Fix gradient clipping bug in LSTM implementation
   Improve explanation of attention mechanism in Day 13
   ```

2. **Write a clear description:**
   - What changes did you make?
   - Why did you make them?
   - How can reviewers test them?

3. **Link related issues:**
   ```
   Closes #42
   Relates to #38
   ```

4. **Ensure all tests pass:**
   - Code runs without errors
   - Output makes sense
   - No breaking changes

5. **Wait for review:**
   - Be patient - reviews take time
   - Be open to feedback
   - Make requested changes promptly

---

## Writing Guidelines

### Tone

- **Professional but accessible** - No jargon without explanation
- **Encouraging** - Learning should feel achievable
- **Practical** - Focus on understanding and building

### Examples of good vs bad writing:

#### ✅ Good - Clear and educational
```markdown
## Understanding Backpropagation Through Time

Imagine you're trying to teach someone to throw a ball. If they miss the 
target, you don't just say "try again." You tell them: "throw a bit lower" 
or "use less force."

BPTT does the same for neural networks. It figures out which weights need 
to change and by how much. The "through time" part means we trace the error 
backwards through the sequence.
```

#### ❌ Bad - Too technical, no intuition
```markdown
## Backpropagation Through Time

BPTT computes gradients by applying the chain rule recursively across 
temporal dependencies in the computational graph.
```

#### ✅ Good - Math with context
```markdown
The hidden state equation:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Breaking this down:
- $W_{xh} x_t$: Contribution from current input
- $W_{hh} h_{t-1}$: Contribution from previous memory
- $b_h$: Bias term (learned offset)
- $\tanh$: Squashes values to [-1, 1] range
```

#### ❌ Bad - Math without context
```markdown
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
```

### Use Analogies

Help beginners understand by connecting to familiar concepts:

- RNN hidden state = short-term memory
- Gradient clipping = putting speed limits on updates
- Temperature in sampling = creativity dial
- Attention = spotlight of focus
- Batch normalization = standardizing test scores

---

## Adding Exercises

Great exercises:
1. **Build understanding** - Not just coding practice
2. **Progressive difficulty** - Start simple, get harder
3. **Have clear goals** - What should students learn?
4. **Include solutions** - Fully worked out with explanations

### Exercise Template

```python
"""
Exercise X: [Clear, Descriptive Title]

Goal: [What will students learn?]

Difficulty: ⏱️ Quick / ⏱️⏱️ Medium / ⏱️⏱️⏱️ Project

Instructions:
1. [Step 1]
2. [Step 2]
...

Expected outcome:
[What should happen when it works?]

Hints:
- [Helpful hint 1]
- [Helpful hint 2]

Common mistakes:
- [Pitfall 1]
- [Pitfall 2]
"""

# Starter code with TODOs
def exercise_function():
    # TODO: Implement this
    pass

if __name__ == "__main__":
    # Test code
    pass
```

---

## Testing

Before submitting:

### 1. **Code runs without errors**
```bash
python implementation.py
python train_minimal.py --data data/sample.txt --epochs 10
```

### 2. **Outputs are reasonable**
- Loss decreases during training
- Generated samples improve over time
- Visualizations display correctly

### 3. **Notebook cells execute in order**
```bash
jupyter nbconvert --to notebook --execute notebook.ipynb
```

### 4. **Documentation is accurate**
- Code matches explanations
- Examples work as described
- Links aren't broken

---

## Recognition

Contributors will be:
- Listed in README.md
- Thanked in release notes
- Given credit in commit history

---

## Questions?

- 💬 [GitHub Discussions](../../discussions) - General questions
- 📧 Email: [your-email]
- 🐦 Twitter: #30u30

---

## Code of Conduct

### Be respectful
- Everyone is learning
- Questions are encouraged
- Criticism should be constructive

### Be patient
- Reviews take time
- Maintainers are volunteers
- Not every suggestion will be accepted

### Be helpful
- Answer questions when you can
- Share your learnings
- Help others debug

---

Thank you for making 30u30 better! 🎉

Every contribution - big or small - helps thousands of people learn.
