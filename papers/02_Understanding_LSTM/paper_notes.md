# Paper Notes: Understanding LSTM Networks

> Christopher Olah's blog post (August 2015)
> http://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

## ELI5 (Explain Like I'm 5)

Imagine you're reading a really long story, and someone asks you "who was the main character?" If you only remember the last sentence, you're stuck. You need a way to keep important notes while you read — like writing key facts on a sticky note.

That's what an LSTM does. A regular RNN is like reading without notes — you forget stuff fast. An LSTM carries a "sticky note" (the cell state) that it can write on, erase from, and read at each step. Three "rules" control the sticky note:

1. **Erase rule** (forget gate): scratch out old notes that aren't relevant anymore
2. **Write rule** (input gate): add new important facts
3. **Read rule** (output gate): decide which notes are relevant right now

The sticky note travels alongside the reading, and information on it doesn't degrade — unlike a regular RNN where everything gets fuzzier over time.

Note: This analogy is ours, not Colah's. His metaphor is the "conveyor belt" — see below.

---

## What This Post Actually Is

This isn't a research paper — it's a visual explainer. Colah walks through how LSTMs work step by step, using diagrams that became the de facto reference for understanding LSTMs. If you've seen an LSTM diagram anywhere on the internet, it probably traces back to this post.

The post covers: RNN recap, the long-term dependency problem, the LSTM cell state and gates, and LSTM variants (including GRUs).

---

## The Long-Term Dependency Problem

Colah's key examples:

**Short-term (easy):** "the clouds are in the ___" — predicting "sky" only needs recent context. RNNs handle this fine.

**Long-term (hard):** "I grew up in France... I speak fluent ___" — predicting "French" requires remembering "France" from much earlier. The relevant information and where it's needed are far apart. RNNs struggle here.

He references Hochreiter (1991) and Bengio et al. (1994) as the work that identified fundamental reasons why this is hard — gradients either vanish or explode when backpropagating through many time steps.

---

## The Cell State: Colah's "Conveyor Belt"

This is the central metaphor of the post, in Colah's own words:

> "The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It's very easy for information to just flow along it unchanged."

The cell state is a separate path that runs parallel to the hidden state. Information can ride along it without degradation, unless the LSTM explicitly modifies it through gates.

Standard RNN: just a hidden state, passed through a tanh at each step.
LSTM: hidden state + cell state, where the cell state has a direct additive connection across time steps.

---

## The Three Gates (Step by Step from the Post)

Gates are sigmoid layers followed by pointwise multiplication. Sigmoid outputs values between 0 ("let nothing through") and 1 ("let everything through").

### 1. Forget Gate

> "The first step in our LSTM is to decide what information we're going to throw away from the cell state."

Looks at h_{t-1} and x_t, outputs a value between 0 and 1 for each element in the cell state.

**Colah's example:** In a language model tracking subject gender, when we encounter a new subject, we want to forget the old one's gender.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### 2. Input Gate + Cell Candidate

Two parts:
- **Input gate** (sigmoid): decides which values to update
- **Cell candidate** (tanh): creates a vector of new candidate values

**Colah's example:** We want to add the new subject's gender to the cell state, replacing the one we're forgetting.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

### 3. Cell State Update

This is where the actual memory update happens:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Forget gate scales old memory, input gate scales new candidates, then they're added. This additive update is why gradients flow well — the derivative of C_t with respect to C_{t-1} is just f_t, not a matrix multiplication.

### 4. Output Gate

> "Finally, we need to decide what we're going to output."

Sigmoid decides which parts of the cell state to expose. The cell state is pushed through tanh (to get values between -1 and 1), then multiplied by the output gate.

**Colah's example:** If we just saw a subject, we might want to output information relevant to a verb — like whether the subject is singular or plural, for verb conjugation.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

---

## Variants Covered in the Post

### Peephole Connections (Gers & Schmidhuber, 2000)
Gates can look at the cell state directly, not just h_{t-1} and x_t. Some papers use peepholes on all gates, some on just a few.

### Coupled Forget and Input Gates
Instead of deciding separately what to forget and what to add, make those decisions together: only forget when something new is coming in, only add when making room.

### GRU (Cho et al., 2014)
Merges forget and input gates into a single "update gate." Also merges cell state and hidden state. Simpler than LSTM, increasingly popular at the time of writing.

### Do Variants Matter?
Colah references two key studies:
- **Greff et al. (2015)** compared popular variants and found "they're all about the same"
- **Jozefowicz et al. (2015)** tested over 10,000 RNN architectures and found some that beat LSTMs on certain tasks

---

## The Post's Conclusion

Colah ends with: attention is the "next step." He references Xu et al. (2015) for attention-based image captioning and mentions Grid LSTMs (Kalchbrenner et al. 2015) and generative models (Gregor et al. 2015, Chung et al. 2015) as other exciting directions.

Note that this post was written in August 2015 — two years before "Attention Is All You Need" (2017). The post correctly predicted attention's importance but couldn't have known it would eventually replace LSTMs entirely in most NLP tasks.

---

## What the Post Doesn't Cover

- Vanishing gradient math in detail (it references Hochreiter and Bengio but doesn't derive it)
- Training procedures, hyperparameters, or practical implementation advice
- Benchmarks or performance numbers
- The forget gate bias initialization trick (b_f = 1, from Jozefowicz et al. 2015)
- Bidirectional LSTMs, stacked LSTMs, or attention mechanisms in detail
- Any comparison of LSTM vs vanilla RNN on actual tasks

This is an architecture explainer, not a benchmarking paper.

---

## Our Additions (Not from the Post)

### Why the Additive Update Matters for Gradients

The cell state update C_t = f_t * C_{t-1} + i_t * C_tilde means the gradient of C_t with respect to C_{t-1} is just f_t (a scalar gate value near 1), rather than a weight matrix multiplication. This is why gradients don't vanish — they take a direct path through the cell state.

In a vanilla RNN, gradients must pass through h_t = tanh(W_hh * h_{t-1} + ...), meaning they get multiplied by W_hh at every step. If eigenvalues of W_hh are below 1, gradients shrink exponentially.

### Practical Initialization Note

Initializing the forget gate bias to 1 (so forget gates start near 1, meaning "remember by default") is now standard practice. This comes from Jozefowicz et al. (2015), one of the papers Colah references.

### Where LSTMs Stand Now (Our Retrospective)

| Aspect | 2015 (when post was written) | Now |
|--------|------|-----|
| Dominant architecture | LSTMs ruled NLP, speech, translation | Transformers dominate most tasks |
| Attention | Emerging add-on to LSTMs | Replaced LSTMs in "Attention Is All You Need" |
| Where LSTMs still appear | Everywhere | Time series, streaming, edge devices, smaller models |
| Colah's prediction about attention | "There is a next step and it's attention!" | Exactly right |

---

## Questions Worth Thinking About

1. Colah says the cell state carries information "with only some minor linear interactions." What are those "minor linear interactions" specifically? (Answer: the forget gate multiplication and the input gate addition.)

2. The post mentions coupled forget/input gates as a variant. What's the tradeoff? (You can only add new information by forgetting old information, which might be too restrictive.)

3. Greff et al. found LSTM variants perform "about the same." Why might that be? (Maybe the core design — additive cell state with learned gates — is what matters, not the specific gate configuration.)

4. Colah correctly predicted attention's importance. But he framed it as an add-on to RNNs, not a replacement. What changed? (Vaswani et al. 2017 showed self-attention alone, without any recurrence, was sufficient and more parallelizable.)
