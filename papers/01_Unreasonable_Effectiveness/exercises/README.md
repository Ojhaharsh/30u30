# Exercises: The Unreasonable Effectiveness of RNNs

Learn by building! These exercises progress from simple to challenging.

**Time estimates:** 
- ⏱️ Quick (15-30 min)
- ⏱️⏱️ Medium (30-60 min)  
- ⏱️⏱️⏱️ Project (1-3 hours)

---

## Exercise 1: Build RNN from Scratch ⏱️⏱️⏱️

**Goal:** Implement a character-level RNN using only NumPy. Understand every line.

**What you'll learn:**
- How RNNs process sequences
- Forward propagation through time
- Backpropagation through time (BPTT)
- Gradient clipping

**Instructions:**

1. **Start with the skeleton** (`exercise_01_build_rnn.py`)

2. **Implement these functions:**
   ```python
   def forward_pass(inputs, h_prev, Wxh, Whh, Why, bh, by):
       """Compute hidden states and predictions"""
       pass
   
   def backward_pass(dh_next, ...):
       """Compute gradients via BPTT"""
       pass
   
   def update_weights(W, dW, learning_rate):
       """Update parameters"""
       pass
   ```

3. **Test on simple data:**
   ```python
   text = "abcabcabc"
   # Can your RNN learn this pattern?
   ```

4. **Verify your implementation:**
   - Loss should decrease
   - Generated text should improve
   - Compare with `implementation.py`

**Hints:**
- Hidden state equation: `h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)`
- Output equation: `y_t = Why @ h_t + by`
- Use gradient clipping: `np.clip(grad, -5, 5)`

**Success criteria:**
- ✅ Loss decreases over time
- ✅ Model generates recognizable patterns
- ✅ You can explain each line of code

**Solution:** `solutions/exercise_01_solution.py`

---

## Exercise 2: Temperature Sampling ⏱️

**Goal:** Understand how temperature affects text generation.

**What you'll learn:**
- Role of temperature in sampling
- Trade-off between creativity and coherence
- Controlling randomness

**Instructions:**

1. **Train a model on any text** (use `train_minimal.py`)

2. **Generate samples with different temperatures:**
   ```python
   temperatures = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
   for temp in temperatures:
       sample = model.sample(h, seed_idx, 200, temperature=temp)
       print(f"Temperature {temp}:")
       print(sample)
   ```

3. **Observe the differences:**
   - Low temperature (0.2): Conservative, repetitive
   - Medium temperature (0.8): Balanced
   - High temperature (2.0): Creative, chaotic

4. **Answer these questions:**
   - What temperature produces the best results?
   - Why does low temperature repeat phrases?
   - Why does high temperature produce nonsense?
   - What's the mathematical reason?

**Expected observations:**
- **T=0.2**: "the the the the" (too deterministic)
- **T=1.0**: "the cat sat on the mat" (balanced)
- **T=2.0**: "thq vzt wrt in thz xat" (too random)

**Deliverable:** 
- Write a short analysis (200 words)
- Include samples at 3 temperatures
- Explain the pattern

**Solution:** `solutions/exercise_02_solution.md`

---

## Exercise 3: Your Own Dataset ⏱️⏱️

**Goal:** Train on your own text and analyze what the model learns.

**What you'll learn:**
- Data preparation
- Domain-specific patterns
- Overfitting vs underfitting

**Instructions:**

1. **Choose your dataset:**
   - Your tweets / diary entries
   - Your favorite book
   - Code in your preferred language
   - Song lyrics from your favorite artist

2. **Prepare the data:**
   ```python
   # Must be plain text, UTF-8 encoded
   # At least 100KB for good results
   ```

3. **Train the model:**
   ```bash
   python train_minimal.py --data your_data.txt --epochs 200
   ```

4. **Analyze the results:**
   - What patterns did it learn?
   - What mistakes does it make?
   - How long until it produces coherent text?
   - Does it capture the "style"?

5. **Experiment with hyperparameters:**
   - Hidden size: 50, 100, 200
   - Learning rate: 0.01, 0.1, 0.5
   - Sequence length: 10, 25, 50

**Fun datasets to try:**
- 📚 **Books**: Shakespeare, Hemingway, Harry Potter
- 💻 **Code**: Your GitHub repos, Linux kernel, React source
- 🎵 **Lyrics**: Taylor Swift, Beatles, Hamilton
- 📱 **Social media**: Your Twitter archive
- 📰 **News**: News articles, Wikipedia

**Deliverable:**
- Trained model
- 3 generated samples
- Analysis of what it learned
- Hyperparameter comparison table

**Solution:** `solutions/exercise_03_solution.md`

---

## Exercise 4: Loss Visualization ⏱️

**Goal:** Visualize training dynamics and identify problems.

**What you'll learn:**
- How to diagnose training issues
- Overfitting detection
- When to stop training

**Instructions:**

1. **Modify training loop to log losses:**
   ```python
   losses = []
   for epoch in range(epochs):
       loss = train_one_epoch()
       losses.append(loss)
   ```

2. **Plot the loss curve:**
   ```python
   import matplotlib.pyplot as plt
   
   plt.plot(losses)
   plt.xlabel('Iteration')
   plt.ylabel('Loss')
   plt.title('Training Loss')
   plt.show()
   ```

3. **Identify these patterns:**
   - **Smooth decrease**: Good training
   - **Plateau**: Learning rate too low or model capacity exhausted
   - **Spikes**: Gradient explosion or learning rate too high
   - **Oscillation**: Unstable training

4. **Try these fixes:**
   - Gradient clipping (for explosions)
   - Learning rate decay (for oscillations)
   - Increase model size (for plateaus)

**Expected patterns:**

```
Good training:
Loss ↓ smoothly, then plateaus

Gradient explosion:
Loss ___/\___/\___/\___
    spike spike spike

Learning rate too high:
Loss \/\/\/\/\/\/
    oscillates

Overfitting:
Train loss ↓, Val loss ↑
```

**Deliverable:**
- Loss plot for your training run
- Annotated with observations
- Suggestions for improvement

**Solution:** `solutions/exercise_04_solution.py`

---

## Project: Shakespeare vs Hemingway Classifier ⏱️⏱️⏱️

**Goal:** Train two models and build a classifier to distinguish them.

**What you'll learn:**
- Style transfer
- Feature extraction from RNNs
- Classification using generative models

**The Challenge:**

1. **Train two models:**
   - Model A: Shakespeare's works
   - Model B: Hemingway's works

2. **Generate samples:**
   - 100 samples from each model
   - Mix them randomly

3. **Build a classifier:**
   - Can you tell which is which?
   - What features distinguish them?
   - Use hidden states as features

4. **Analyze the differences:**
   - Vocabulary
   - Sentence structure
   - Average word length
   - Punctuation patterns

**Approach:**

```python
# Train models
shakespeare_rnn = train_on(shakespeare_data)
hemingway_rnn = train_on(hemingway_data)

# Generate samples
shakespeare_samples = [shakespeare_rnn.sample(...) for _ in range(100)]
hemingway_samples = [hemingway_rnn.sample(...) for _ in range(100)]

# Extract features
def extract_features(text, model):
    # Run text through model, extract hidden states
    # Compute statistics: vocab richness, sentence length, etc.
    return features

# Train classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(features, labels)

# Evaluate
accuracy = clf.score(test_features, test_labels)
```

**Bonus challenges:**
- 🌟 Add more authors (3-way classification)
- 🌟 Use only vocabulary (no RNN features)
- 🌟 Build a "style transfer" system

**Deliverable:**
- Two trained models
- Classifier with >80% accuracy
- Report analyzing stylistic differences
- Code for the entire pipeline

**Solution:** `solutions/project_shakespeare_hemingway/`

---

## Bonus Exercise: Gradient Visualization ⏱️⏱️

**Goal:** Visualize how gradients flow through time.

**Instructions:**

1. **Compute gradients for a sequence:**
   ```python
   gradients = model.backward(...)
   ```

2. **Plot gradient magnitudes at each time step:**
   ```python
   plt.plot(gradient_norms)
   plt.xlabel('Time step')
   plt.ylabel('Gradient magnitude')
   ```

3. **Observe:**
   - Do gradients vanish (shrink) going back in time?
   - Do they explode (grow)?
   - How does sequence length affect this?

4. **Compare:**
   - Short sequences (10 steps)
   - Medium sequences (25 steps)
   - Long sequences (50 steps)

**This sets up Day 2:** LSTMs solve the vanishing gradient problem!

**Solution:** `solutions/bonus_gradient_viz.py`

---

## Tips for Success

1. **Start simple:** Get Exercise 1 working before moving on
2. **Test often:** Print shapes, check ranges, verify formulas
3. **Debug systematically:** One function at a time
4. **Ask questions:** Use GitHub discussions if stuck
5. **Compare with solutions:** But try first!

---

## Common Pitfalls

❌ **Forgetting to reset hidden state** between epochs  
✅ Set `h = np.zeros(...)` at start of each epoch

❌ **Not clipping gradients** → explosion  
✅ Always use `np.clip(grad, -5, 5)`

❌ **Wrong matrix dimensions**  
✅ Print shapes: `print(W.shape, x.shape)`

❌ **Dividing by zero in softmax**  
✅ Add small epsilon: `p = exp(x) / (sum(exp(x)) + 1e-8)`

❌ **Learning rate too high** → oscillation  
✅ Start with 0.01, increase gradually

---

## Going Further

Once you've completed these exercises, you're ready for:
- **Day 2:** LSTMs (solving the vanishing gradient problem)
- **Advanced:** Attention mechanisms
- **Projects:** Chatbots, code completion, music generation

---

**Need help?** Open an issue or join the discussion!

**Finished?** Share your results with #30u30 and move to Day 2!
