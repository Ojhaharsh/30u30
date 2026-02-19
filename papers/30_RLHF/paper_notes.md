# Paper Notes: Deep Reinforcement Learning from Human Feedback

> **Note:** This paper is the grandfather of ChatGPT. While it focuses on Atari and MuJoCo, the core algorithm (Reward Modeling + PPO) is exactly what was used years later to align GPT-3.

## ELI5 (Explain Like I'm 5)

### The Puppy Training School

Imagine you want to teach a puppy to do a backflip.

**Method 1: Writing a Rule (Traditional RL)**
You have to write a mathematical formula for "backflip."
> Reward = (feet_height > 1m) + (rotation_speed > 5rad/s) - (wobble) ...
This is incredibly hard. If you get the math wrong, the puppy might just jump up and down, or spin in circles, because that maximizes your formula.

**Method 2: Clicker Training (This Paper)**
You just watch the puppy play.
- Clip A: The puppy jumps.
- Clip B: The puppy does a somersault.
You pick **Clip B**.
You don't need to know the physics of a backflip. You just need to recognize one when you see it.
The "Reward Model" is like a clicker. It learns: "Oh, the human likes it when I rotate backwards."
Then, the puppy (the Policy) practices to hear the clicker (maximize the Reward Model) as much as possible.

> **Note:** This analogy is ours, not the authors'. The paper uses the example of a Hopper doing a backflip, which is mathematically hard to specify but easy to judge visually.

---

## What the Paper Actually Covers

### 1. The Problem with Hand-Crafted Rewards (Section 1)
In many real-world tasks (driving, conversation, cooking), we don't have a reward function. We can't write a Python function that returns `reward=1` if a summary is "helpful" and `reward=0` if it's "sarcastic."
We need to learn the reward function from data.

### 2. Learning from Comparisons (Section 2.1)
Instead of asking humans for an absolute score ("This backflip is a 7/10"), which is noisy and uncalibrated, we ask for **pairwise comparisons** ("Is Clip A better than Clip B?").
This is easier for humans and more robust.

### 3. The Algorithm (Section 2.2)
1.  **Collect** trajectories with the current policy $\pi$.
2.  **Select** pairs of segments $(\sigma^1, \sigma^2)$ and show them to a human.
3.  **Label** preferences: $\mu$ prefers $\sigma^1$ or $\sigma^2$.
4.  **Train Reward Model** $\hat{r}$ to predict these preferences.
5.  **Train Policy** $\pi$ to maximize $\hat{r}$ using PPO.

### 4. Results (Section 5)
-   **Atari:** Superhuman performance on many games with only ~1,000 queries (less than 1% of the interactions needed for evolution strategies).
-   **MuJoCo:** Trained a Hopper to do a backflip (Figure 4) without a reward function.
-   **Hallucination:** Sometimes the agent learns to exploit the reward model ("reward hacking").

---

## The Math

### The Bradley-Terry Model (Eq 1)
The probability that a human prefers segment $\sigma^1$ over $\sigma^2$ depends on the sum of rewards in each segment:

$$ \hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum_{t} \hat{r}(s_t^1, a_t^1)}{\exp \sum_{t} \hat{r}(s_t^1, a_t^1) + \exp \sum_{t} \hat{r}(s_t^2, a_t^2)} $$

This is basically a softmax over the two segments' total rewards.

### The Loss Function (Eq 2)
We minimize the cross-entropy between the human's choice $\mu(1$ or $2)$ and the model's prediction:

$$ \text{Loss}(\hat{r}) = - \sum_{(\sigma^1, \sigma^2, \mu) \in \mathcal{D}} \mu(1) \log \hat{P}[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat{P}[\sigma^2 \succ \sigma^1] $$

---

## Going Beyond the Paper (Our Retrospective)

### Connecting to ChatGPT (InstructGPT, 2022)
This paper (2017) laid the groundwork.
InstructGPT (2022) used the **exact same 3-step recipe**:
1.  **SFT:** Supervised Fine-Tuning (demonstration data).
2.  **RM:** Reward Modeling (pairwise ranking of outputs).
3.  **PPO:** Optimize the LLM against the RM.

The only difference is the domain (Text vs. Atari pixels) and the network architecture (Transformer vs. CNN). The RLHF logic is identical.

### Evaluation is "In the Eye of the Beholder"
RLHF aligns the model to the *labeler's* preferences, not necessarily "truth." If labelers prefer confident-sounding hallucinations, the model will learn to hallucinate confidently.

---

## Questions Worth Thinking About

1.  **Reward Hacking:** If the Reward Model is an imperfect proxy for what we want, will a super-powerful PPO agent "break" it? (Yes, this is a major safety concern.)
2.  **Sample Efficiency:** Why do we need PPO? Why not just train the policy directly on the preferred trajectories (like Behavioral Cloning)? (Answer: PPO allows the agent to discover *new* behaviors that get high reward, potentially better than the human demonstrations.)
