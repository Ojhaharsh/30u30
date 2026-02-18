# Day 29: Proximal Policy Optimization Algorithms

> Schulman, Wolski, Dhariwal, Radford, Klimov — OpenAI (2017) — [Paper](https://arxiv.org/abs/1707.06347)

**Time:** 4-6 hours
**Prerequisites:** Basic reinforcement learning (policy gradients, value functions), calculus, Day 25 (Scaling Laws) helpful but not required
**Code:** PyTorch (gym environments)

---

## What This Paper Is Actually About

The central problem in policy gradient RL is step size. Take too small a step and training is slow. Take too large a step and the policy collapses — you update so aggressively that the agent forgets what worked and never recovers. Trust Region Policy Optimization (TRPO, Schulman et al. 2015) solved this with a hard KL-divergence constraint, guaranteeing monotonic improvement, but required second-order optimization (computing the Hessian) which is expensive and hard to implement correctly.

PPO solves the same problem with a much simpler mechanism: clip the probability ratio between the new and old policy so that the objective function itself stops rewarding updates that move too far. No constraint. No Hessian. Just a min() in the loss function. The result is a first-order method that empirically matches or beats TRPO across a wide range of tasks.

The paper proposes two variants. PPO-Clip (the one everyone uses) clips the surrogate objective directly. PPO-Penalty adds a KL-divergence penalty to the objective and adapts the penalty coefficient automatically. The paper tests both, but PPO-Clip is what became the standard — it is the algorithm used in OpenAI's RLHF pipeline for InstructGPT and ChatGPT.

---

## What the Authors Actually Showed

The experiments (Section 6) compare PPO against A2C, A3C, TRPO, and CEM across two benchmark suites:

**Continuous control (MuJoCo locomotion tasks, Section 6.1):**
The paper reports average normalized scores across 7 MuJoCo tasks (HalfCheetah, Hopper, Walker2d, Ant, Humanoid, Swimmer, Reacher). PPO-Clip achieves the highest average score of 1.0 (normalized), compared to TRPO at 0.98, A2C at 0.62, and A3C at 0.64. The comparison is run for 1 million timesteps per task.

**Atari game playing (Section 6.3):**
PPO is tested on 49 Atari games for 40 million frames. PPO outperforms A2C on 30 of 49 games and matches or beats TRPO on 28 of 49 games. The paper reports a mean score of 2.5x the human baseline across all games.

**Ablation study (Section 6.1, Table 1):**
The paper systematically ablates the components of the PPO-Clip objective. The full PPO-Clip objective (clipping + entropy bonus + value function loss) outperforms each component removed. Clipping alone accounts for most of the gain over vanilla policy gradient.

---

## The Core Idea

Standard policy gradient methods maximize the expected return by computing:

$$L^{PG}(\theta) = \hat{E}_t \left[ \log \pi_\theta(a_t | s_t) \hat{A}_t \right]$$

where $\hat{A}_t$ is the estimated advantage at timestep $t$. The problem: nothing stops you from taking a huge gradient step that destroys the policy.

PPO introduces the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

This ratio is 1 when the new policy equals the old policy. It grows above 1 when the new policy assigns higher probability to an action than the old policy did, and falls below 1 when it assigns lower probability.

The clipped surrogate objective (Equation 7 in the paper) is:

$$L^{CLIP}(\theta) = \hat{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

The min() does the work. When the advantage is positive (the action was good), the objective increases as $r_t$ increases — but only up to $1+\epsilon$. After that, the clipped term takes over and the objective stops improving. The agent gains nothing from pushing the ratio further. When the advantage is negative (the action was bad), the same logic applies in reverse: the objective stops improving once $r_t$ falls below $1-\epsilon$.

The result: the agent is never incentivized to move the policy more than $\epsilon$ away from the old policy. The paper uses $\epsilon = 0.2$ as the default.

---

## The Architecture

### 1. Actor-Critic Network

PPO uses an actor-critic architecture. A single neural network (or two separate networks) outputs both the policy (actor) and the value function (critic):

- **Actor:** $\pi_\theta(a | s)$ — a probability distribution over actions given state $s$
- **Critic:** $V_\theta(s)$ — an estimate of the expected return from state $s$

For continuous action spaces, the actor outputs the mean (and optionally log-std) of a Gaussian distribution. For discrete action spaces, it outputs logits over actions.

### 2. Generalized Advantage Estimation (GAE)

PPO uses GAE (Schulman et al. 2015b, cited in the paper as the advantage estimator) to compute $\hat{A}_t$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual. The $\lambda$ parameter (typically 0.95) controls the bias-variance tradeoff: $\lambda=0$ gives one-step TD (low variance, high bias), $\lambda=1$ gives Monte Carlo returns (high variance, low bias).

### 3. The Full Objective

The paper's full objective (Equation 9) combines three terms:

$$L^{CLIP+VF+S}(\theta) = \hat{E}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

- $L^{CLIP}$: the clipped policy loss (described above)
- $L^{VF} = (V_\theta(s_t) - V_t^{targ})^2$: value function MSE loss
- $S[\pi_\theta](s_t)$: entropy bonus to encourage exploration
- $c_1, c_2$: coefficients (the paper uses $c_1 = 1, c_2 = 0.01$ for Atari)

### 4. The Training Loop

The paper's Algorithm 1 describes the core loop:

```
for each iteration:
    collect T timesteps of experience using current policy
    compute advantage estimates using GAE
    for K epochs:
        split data into minibatches of size M
        update policy by maximizing L^CLIP+VF+S
```

The key difference from vanilla policy gradient: the same batch of data is reused for K epochs (the paper uses K=3-10 epochs, minibatch size M=64). This is what makes PPO sample-efficient — you extract more gradient signal from each environment interaction.

---

## Implementation Notes

**Clipping epsilon:** The paper uses $\epsilon = 0.2$ for all experiments. Values of 0.1-0.3 are common in practice. Larger epsilon = more aggressive updates = faster but less stable training.

**Advantage normalization:** The paper normalizes advantages within each minibatch (subtract mean, divide by std). This is not in the original equations but is standard practice and significantly stabilizes training.

**Value function clipping:** Some implementations also clip the value function loss (analogous to the policy clipping). The paper mentions this variant but does not use it as the default.

**Entropy coefficient:** The paper uses $c_2 = 0.01$ for Atari (discrete actions) and $c_2 = 0$ for MuJoCo (continuous). Entropy bonus prevents premature convergence to a deterministic policy.

**Orthogonal initialization:** The paper's implementation uses orthogonal weight initialization with specific gain values. This matters more for deep networks and continuous control tasks.

**Things that will bite you:**
- Not normalizing observations (running mean/std normalization is critical for MuJoCo)
- Forgetting to detach the old policy's log probabilities before computing the ratio
- Using the wrong sign convention for advantages (maximize return = maximize advantage)
- Not clipping gradients (the paper clips to max norm 0.5)

---

## What to Build

### Quick Start

```bash
python setup.py
python implementation.py --demo
python train_minimal.py --env CartPole-v1 --epochs 100
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | Implement the clipped surrogate objective | Understand exactly what the clip() does and when it activates |
| 2 | Implement GAE advantage estimation | See how lambda controls the bias-variance tradeoff |
| 3 | Build the actor-critic network | Practice separating policy and value heads |
| 4 | Ablate the clipping epsilon | Reproduce the paper's finding that epsilon=0.2 is near-optimal |
| 5 | Compare PPO vs vanilla policy gradient | See the training stability difference firsthand |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **The clip is a pessimistic lower bound.** The min() in $L^{CLIP}$ takes the worse of the clipped and unclipped objectives. This means the objective is always a lower bound on the true policy improvement — it never overclaims. (Section 3 of the paper.)

2. **Multiple epochs on the same data is the efficiency gain.** Standard policy gradient discards data after one gradient step. PPO reuses each batch for K epochs. This is the primary reason PPO is more sample-efficient than A2C/A3C. (Algorithm 1, Section 3.)

3. **PPO-Clip outperforms PPO-Penalty in practice.** The paper's ablation (Table 1) shows that the adaptive KL penalty is harder to tune and less stable than clipping. PPO-Clip became the standard for this reason.

4. **GAE is doing most of the advantage estimation work.** PPO's performance is sensitive to the GAE lambda parameter. The paper uses lambda=0.95 throughout. (Section 4, citing Schulman et al. 2015b.)

5. **PPO is the algorithm inside RLHF.** InstructGPT (Ouyang et al. 2022) and ChatGPT use PPO to fine-tune language models on human preference signals. The reward model replaces the environment's reward function. This connection is the reason Day 30 (RLHF) follows directly from Day 29.

Note: The connection to InstructGPT/ChatGPT is our retrospective addition, not from the 2017 paper.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | PPO-Clip from scratch: actor-critic, GAE, clipped loss, training loop |
| `train_minimal.py` | Training script with CLI args for env, epochs, learning rate |
| `visualization.py` | Learning curves, clipping frequency plots, policy entropy over time |
| `setup.py` | Environment check (gym, torch, numpy) |
| `requirements.txt` | Dependencies |
| `notebook.ipynb` | Interactive walkthrough — build PPO step by step |
| `paper_notes.md` | Deep notes on the paper with ELI5 and math |
| `CHEATSHEET.md` | Quick reference for hyperparameters and debugging |
| `exercises/` | 5 exercises with solutions |
| `data/` | Saved training runs and logs |

---

## Further Reading

- [PPO Paper](https://arxiv.org/abs/1707.06347) — the original paper (7 pages, readable in one sitting)
- [TRPO Paper](https://arxiv.org/abs/1502.05477) — Schulman et al. 2015, the predecessor PPO is designed to simplify
- [GAE Paper](https://arxiv.org/abs/1506.02438) — Schulman et al. 2015b, the advantage estimator PPO uses
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — OpenAI's clean PPO implementation with documentation
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — Huang et al. 2022, a thorough audit of what actually matters in PPO implementations
- [InstructGPT](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022, PPO applied to RLHF for language models (preview of Day 30)

---

**Previous:** [Day 28 — CS231n: CNNs for Visual Recognition](../28_CS231n/)

**Next:** [Day 30 — Deep Reinforcement Learning from Human Feedback](../30_RLHF/)
