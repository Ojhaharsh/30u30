# Paper Notes: The Unreasonable Effectiveness of RNNs

## ELI5 (Explain Like I'm 5)

### The Story

Imagine you're learning to write by copying books, one letter at a time. At first, you just copy random letters. But after copying millions of letters, you start picking up patterns:

You begin to understand:
- Which letters usually come after others ('q' is followed by 'u')
- How to spell words correctly
- How sentences are structured
- Even the writing style of different authors

**That's the core idea behind character-level RNNs.** They learn to predict the next character by seeing tons of examples. And in learning to predict, they learn the structure of whatever text you feed them.

Note: This analogy is ours, not Karpathy's. But it captures his main point — that prediction forces understanding of structure.

---

## What the Post Actually Covers

Karpathy's blog post (May 2015) isn't a research paper — it's a practitioner's demo. He trains character-level language models on several datasets and shows that a surprisingly simple setup (predict the next character) learns rich structural patterns.

**Important detail he states upfront:** All experiments use **LSTMs**, not vanilla RNNs. He presents the vanilla RNN equations for simplicity, then says "I will use the terms RNN/LSTM interchangeably but all experiments in this post use an LSTM." Our implementation uses vanilla RNN for pedagogical reasons, but his results come from LSTMs.

---

## The Core Idea (From the Post)

**The Task:** Given a sequence of characters, predict the next one.

Karpathy uses the example of training on the string "hello" with vocabulary "h, e, l, o":

```
Feed 'h' → model should predict 'e'
Feed 'e' → model should predict 'l'
Feed 'l' → model should predict 'l'
Feed 'l' → model should predict 'o'
```

He explains that this is trained using one-hot encoding, softmax output, and cross-entropy loss. The RNN processes one character at a time, using its hidden state to remember what came before.

**His key observation:** To predict the next character well, the model must implicitly learn spelling, grammar, formatting, and domain structure. Nobody programs these rules in — they emerge from the prediction task.

He also links to his [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086) — about 100 lines of Python/NumPy that implements the core idea. Our implementation is based on this.

---

## The Experiments (All Six)

Karpathy trains on six different datasets. This is the heart of the post.

### 1. Paul Graham Essays (~1MB)

**Setup:** 2-layer LSTM, 512 hidden nodes, dropout 0.5.

The model generates text that sounds like Paul Graham writing about startups — sentence structure, vocabulary, even the habit of citing numbered references like [2]. It's not coherent, but it captures the *style*.

**Temperature demo (from this section):** He shows what happens at different temperatures:
- Low temperature: "is that they were all the same thing that was a startup is that they were all the same thing that was a startup" — gets stuck in a loop, always picking the highest-probability next character.
- Higher temperature: More variety but more spelling mistakes.

This is the only place in the post where he discusses temperature in detail.

### 2. Shakespeare (4.4MB)

**Setup:** 3-layer RNN, 512 hidden nodes per layer.

Generated output:
```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.
```

What it learned (without being told):
- Character names in ALL CAPS followed by colons
- Dialogue indentation
- Period-appropriate vocabulary
- Rough iambic rhythm

What it gets wrong: "srain" isn't a word, the meaning is incoherent. It learned the **form** of Shakespeare, not the **content**. Karpathy notes: "I can barely recognize these samples from actual Shakespeare."

He also explicitly verifies the samples are NOT memorized by computing nearest-neighbor on the training set.

### 3. Wikipedia (100MB, Hutter Prize dataset)

**Setup:** Following Graves et al., trained overnight.

The model learns:
- Markdown formatting (headings, bullet lists, `[[wiki links]]`)
- XML structure (proper tag opening/closing, nesting)
- Plausible-looking URLs (that don't actually exist — Karpathy notes the model "hallucinated" a yahoo URL)
- Cite journal templates

It switches between generating prose, structured markdown, and raw XML — each with correct formatting conventions.

### 4. Algebraic Geometry LaTeX (16MB)

**Setup:** Trained with his labmate Justin Johnson.

The generated LaTeX "almost compiles." After manual fixes, it produces plausible-looking math with proper equation environments, theorem/lemma structures, and mathematical notation. Common error: opening `\begin{proof}` but closing with `\end{lemma}` — the dependency is too long-term for the model to track.

### 5. Linux Source Code (474MB)

**Setup:** All source and header files from the Linux GitHub repo. 3-layer LSTM, ~10 million parameters, trained over several days.

Generated output:
```c
/*
 * Increment the size file of the new incorrect UI_FILTER group information
 * of the size generatively.
 */
static int indicate_policy(void)
{
  int error;
  if (fd == MARN_EPT) {
    /*
     * The kernel blank will coeld it to userspace.
     */
    if (ss->segment < mem_total)
      unblock_graph_and_set_blocked();
    else
      ret = 1;
    goto bail;
  }
  ...
}
```

What it gets right: brackets, indentation, semicolons, pointer notation, kernel macros, comments, `#include` statements, even the GPL license header character by character.

What it gets wrong: Uses undefined variables, declares variables it never uses, returns values from `void` functions, compares `tty == tty` (vacuously true). Karpathy says explicitly: "I don't think it compiles."

He also notes the model sometimes "decides it's time to sample a new file" and generates a full GPL header, includes, macros, then dives into code.

### 6. Baby Names (8000 names)

A fun small experiment. The model generates plausible new names: "Rudi", "Levette", "Berice", "Lussa". About 90% of generated names don't exist in the training data. Karpathy jokes it could be "useful inspiration when naming a new startup."

---

## Understanding What's Going On

This section of the post is often overlooked but is genuinely interesting.

### Training Evolution (War and Peace)

Karpathy shows how samples evolve during training:

- **Iteration 100:** Random jumbles, but starts learning that words are separated by spaces
- **Iteration 300:** Gets the idea about quotes and periods
- **Iteration 500:** Spells short common words ("we", "He", "His", "and")
- **Iteration 700:** More English-like, but still many misspellings
- **Iteration 1200:** Quotation marks, question marks, longer words
- **Iteration 2000:** Properly spelled words, names, quotations

**His observation:** "First the model discovers the general word-space structure, then starts to learn words (short ones first, then longer). Topics and themes that span multiple words start to emerge only much later."

This progression is worth understanding — it shows how the model builds up structure hierarchically.

### Hidden State Visualization

He visualizes individual neurons in the LSTM's hidden state while processing Wikipedia text. Most neurons aren't interpretable, but about 5% learn specific, interesting functions:

1. **URL detector neuron** — activates inside URLs, turns off outside them
2. **Wiki link detector** — activates inside `[[ ]]` markup, but notably waits for the *second* `[` before activating (the first `[` alone isn't enough)
3. **Position tracker** — varies linearly across `[[ ]]` scope, giving the model a sense of how far through the link text it is
4. **"www" counter** — turns off after the first "w" in "www", likely helping the model know when to stop emitting "w"s

He also found a **quote detection cell** that tracks whether the model is inside or outside quotation marks.

His key point: "We didn't have to hardcode at any point that if you're trying to predict the next character it might be useful to keep track of whether or not you are currently inside or outside of a quote. We just trained the LSTM on raw data and it decided that this is a useful quantity to keep track of."

---

## The Math

At each timestep, the vanilla RNN does:

### Hidden State Update
```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
```

- `h_t`: New hidden state at time t
- `x_t`: Current input (one-hot encoded character)
- `h_{t-1}`: Previous hidden state
- `W_xh, W_hh`: Weight matrices (learned)
- `tanh`: Squashes values to [-1, 1]

### Output
```
y_t = W_hy * h_t + b_y
p_t = softmax(y_t)
```

- `y_t`: Raw scores for each possible next character
- `p_t`: Probability distribution over next character

Karpathy presents these same equations in the post. Note: his actual experiments use the LSTM update equations, which are more complex (four gates instead of one tanh). We use vanilla RNN because it's easier to implement from scratch and the core principle is the same.

---

## The Title

The title references Eugene Wigner's 1960 essay "The Unreasonable Effectiveness of Mathematics in the Natural Sciences." Wigner's point: math works suspiciously well for describing physics.

Karpathy's parallel: character-level prediction works suspiciously well for learning language structure. The task is dead-simple (predict one character), but the model ends up learning spelling, grammar, formatting, and domain conventions.

---

## What the Post Gets Right

- The demo-driven approach is effective: showing Shakespeare, Linux, LaTeX output is more convincing than abstract claims
- Honest about limitations: he checks for memorization, notes the code doesn't compile, acknowledges the meaning is incoherent
- The hidden state visualization section is genuine research insight, not just a demo
- Temperature explanation is practical and grounded

## What the Post Doesn't Cover

- Why LSTMs work better than vanilla RNNs (he just says they do — that's covered in Colah's post, Day 2)
- Formal analysis of what the hidden states are computing
- Quantitative evaluation (no perplexity numbers, no benchmarks)
- Comparison to n-gram baselines (Yoav Goldberg later did this comparison and showed n-grams can match some of the results)
- How this connects to word-level models (he briefly mentions "word-level models work better but this is surely a temporary thing")
- Scaling laws or predictions about future models

---

## Looking Back (Our Retrospective, Not in the Post)

With 10 years of hindsight, the line from char-RNN to modern LLMs is clear:

| Karpathy's Setup (2015) | Modern LLMs (2024) |
|--------------------------|---------------------|
| Character-level | Subword tokens (BPE) |
| LSTM | Transformer |
| Hidden states | Attention + KV cache |
| ~10M parameters | ~100B+ parameters |
| Single GPU, days | Thousands of GPUs, weeks |
| Generates form, not meaning | Generates both (mostly) |

The core idea — **learn by predicting the next token** — hasn't changed. But this connection is retrospective. Karpathy wasn't predicting GPT; he was showing that RNNs could do surprising things with character prediction.

---

## Questions Worth Thinking About

These come from the post itself or directly from the experiments:

1. Why does the model learn to open and close brackets correctly but can't track which *type* of bracket it opened? (LaTeX `\begin{proof}` → `\end{lemma}` error)
2. The hidden state visualization shows ~5% of neurons learn interpretable functions. What are the other 95% doing?
3. Karpathy trained on War and Peace and showed the model learns short words first, then long words, then themes. What does this tell us about how sequential structure is encoded?
4. Yoav Goldberg showed that n-gram models can produce similar-looking outputs for some of these tasks. What does the RNN actually learn that n-grams don't?

---

**Next:** [Day 2 - Understanding LSTM Networks](../02_Understanding_LSTM/) - How to make RNNs remember longer sequences.
