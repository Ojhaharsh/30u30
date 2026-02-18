# Paper Notes: Proximal Policy Optimization Algorithms

> Schulman, Wolski, Dhariwal, Radford, Klimov — OpenAI (2017)

---

## ELI5 (Explain Like I'm 5)

### The "Don't Overcorrect" Problem

Imagine you are learning to ride a bike. Every time you wobble left, you correct by leaning right. But if you overcorrect — lean too far right — you fall the other way. Now you're worse off than before.

Policy gradient RL has exactly this problem. The agent tries an action, sees it worked, and updates its policy to do that action more often. But if the update is too large, the policy changes so much that it forgets everything else it knew. The next episode, it collapses.

TRPO solved this by saying: "You are only allowed to change your policy by a certain amount, measured by KL divergence." It works, but computing that constraint requires second-order optimization — essentially computing the curvature of the loss landscape, which is expensive.

PPO says: "Instead of constraining the update, just make the objective function stop rewarding you for going too far." It clips the probability ratio at [1-epsilon, 1+epsilon]. Once you've moved the policy that far, you get no additional reward for moving it further. The incentive disappears.

> **Note:** This analogy is ours, not the authors'.

---

## What the Paper Actually Covers

The paper (Schulman et al. 2017) is 7 pages. It is primarily an empirical paper — the theoretical justification for PPO is brief (Section 3), and the bulk of the paper is experiments (Section 6).

### Section 1: Introduction
The authors frame PPO as the answer to a practical question: can we get the stability benefits of TRPO without the implementation complexity? They note that TRPO requires computing the Fisher information matrix (second-order method), which is expensive and hard to parallelize. PPO is designed to be implementable with standard first-order optimizers (Adam, SGD).

### Section 2: Background — Policy Gradient Methods
The paper reviews the standard policy gradient objective and its problems. The key issue: policy gradient methods perform one gradient update per data sample. This is wasteful — you collect expensive environment data and then discard it after a single gradient step.

### Section 3: Clipped Surrogate Objective
This is the core contribution. The authors introduce the probability ratio $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ and the clipped objective:

$$L^{CLIP}(\theta) = \hat{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

The authors describe this as a "pessimistic" lower bound: the min() ensures the objective never exceeds what the clipped version allows. The default epsilon is 0.2.

They also describe PPO-Penalty (Section 3, Equation 8), which adds an adaptive KL penalty instead of clipping. The penalty coefficient $\beta$ is adjusted up if the KL divergence exceeds a target $d_{targ}$, and down if it falls below. The paper tests both variants but finds PPO-Clip more robust.

### Section 4: Adaptive KL Penalty Coefficient
The adaptive penalty mechanism: if the measured KL divergence $d$ is greater than $1.5 \times d_{targ}$, multiply $\beta$ by 2. If $d < d_{targ} / 1.5$, divide $\beta$ by 2. This is a heuristic that works in practice but requires tuning $d_{targ}$.

### Section 5: Algorithm
Algorithm 1 in the paper describes the training loop for the parallel actor variant (multiple actors collecting data simultaneously). The key parameters: T timesteps per actor per iteration, K epochs of optimization per batch, minibatch size M.

The paper uses: T=2048, K=10, M=64 for MuJoCo; T=128, K=3, M=32*8 for Atari.

### Section 6: Experiments
Three sets of experiments:
1. **Comparison of surrogate objectives (Section 6.1):** Ablation on MuJoCo tasks comparing no clipping, clipping, KL penalty (fixed), KL penalty (adaptive), and other variants. PPO-Clip wins.
2. **Comparison with other algorithms (Section 6.2):** PPO vs. TRPO, A2C, A3C, CEM on 7 MuJoCo tasks. PPO achieves the highest average normalized score.
3. **Atari (Section 6.3):** PPO vs. A2C and TRPO on 49 Atari games, 40M frames. PPO outperforms A2C on 30/49 games.

---

## The Math

### Probability Ratio (Section 3)

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

- $r_t = 1$: new policy identical to old policy
- $r_t > 1$: new policy assigns higher probability to this action
- $r_t < 1$: new policy assigns lower probability to this action

### Clipped Surrogate Objective (Equation 7)

$$L^{CLIP}(\theta) = \hat{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

When advantage $\hat{A}_t > 0$ (good action): the min clips at $(1+\epsilon)\hat{A}_t$, preventing the ratio from growing beyond $1+\epsilon$.

When advantage $\hat{A}_t < 0$ (bad action): the min clips at $(1-\epsilon)\hat{A}_t$, preventing the ratio from falling below $1-\epsilon$.

### Generalized Advantage Estimation (cited from Schulman et al. 2015b)

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the one-step TD residual.

- $\lambda = 0$: $\hat{A}_t = \delta_t$ (one-step TD, low variance, high bias)
- $\lambda = 1$: $\hat{A}_t = \sum_l \gamma^l \delta_{t+l}$ (Monte Carlo, high variance, low bias)

The paper uses $\lambda = 0.95$ throughout.

### Full Objective (Equation 9)

$$L^{CLIP+VF+S}(\theta) = \hat{E}_t \left[ L^{CLIP}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

- $L^{VF}_t = (V_\theta(s_t) - V_t^{targ})^2$: value function MSE
- $S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$: entropy bonus
- $c_1 = 1, c_2 = 0.01$ for Atari; $c_2 = 0$ for MuJoCo

---

## The Experiments

### Setup (Section 6)

MuJoCo tasks: HalfCheetah, Hopper, Walker2d, Ant, Humanoid, Swimmer, Reacher. 1 million timesteps each. 3 random seeds.

Atari: 49 games. 40 million frames. Compared against A2C (synchronous A3C) and TRPO.

Hyperparameters for MuJoCo (Table 3 in paper): Adam optimizer, lr=3e-4, epsilon=0.2, lambda=0.95, gamma=0.99, T=2048, K=10 epochs, M=64 minibatch size.

### Results

| Algorithm | MuJoCo Average Score (normalized) |
|-----------|----------------------------------|
| PPO-Clip | 1.00 (best) |
| TRPO | 0.98 |
| A3C | 0.64 |
| A2C | 0.62 |
| CEM | 0.43 |

Source: Table 2 in the paper. Scores normalized so that PPO-Clip = 1.0.

On Atari: PPO achieves a mean score of approximately 2.5x human performance across 49 games (Figure 3 in paper). A2C achieves approximately 1.8x.

---

## What the Paper Gets Right

- The clipping mechanism is genuinely elegant. It achieves the same goal as TRPO's KL constraint with a single line of code.
- The ablation study (Table 1) is thorough. The authors test 8 variants of the objective and show that each component of the full PPO objective contributes.
- The paper is honest about limitations: PPO is on-policy, which means it cannot reuse data from old policies (unlike off-policy methods like SAC or TD3). The multiple-epoch trick is a partial workaround, not a full solution.

## What the Paper Does Not Cover

- **Continuous action spaces in depth:** The paper uses Gaussian policies for MuJoCo but does not discuss the implementation details (e.g., whether log-std is state-dependent or a learned parameter).
- **Hyperparameter sensitivity:** The paper reports results with specific hyperparameters but does not systematically study sensitivity. Later work (Huang et al. 2022) found that many implementation details matter significantly.
- **Off-policy data:** PPO discards data after K epochs. The paper does not explore whether the data could be reused further (importance sampling corrections would be needed).
- **Theoretical guarantees:** Unlike TRPO, PPO does not have a formal monotonic improvement guarantee. The clipping is a heuristic that works empirically.

---

## Going Beyond the Paper (Our Retrospective)

> **Note:** Everything in this section is our addition, not from the 2017 paper.

PPO became the dominant RL algorithm for a reason that the 2017 paper could not have anticipated: RLHF. When OpenAI developed InstructGPT (Ouyang et al. 2022), they needed an RL algorithm that could:
1. Fine-tune a large language model (the policy)
2. Optimize against a learned reward model (the environment)
3. Be stable enough to not destroy the pretrained weights
4. Be simple enough to implement at scale

PPO-Clip checked all four boxes. The KL penalty term in RLHF (which prevents the fine-tuned model from drifting too far from the pretrained model) is essentially the same idea as PPO's clipping — just applied at the token level.

The irony: PPO was designed to simplify TRPO for robotics tasks. It ended up being the algorithm that trained ChatGPT.

---

## Questions Worth Thinking About

1. The clipping prevents the policy from changing too much in one step. But what if the optimal policy is very far from the current policy? How does PPO eventually get there?

2. GAE with $\lambda = 0.95$ is a weighted average of all future TD residuals. Why does this reduce variance compared to pure Monte Carlo returns?

3. The paper uses the same network for actor and critic (shared parameters). What are the tradeoffs of sharing vs. separating these networks?

4. PPO is on-policy: it discards data after K epochs. SAC (Haarnoja et al. 2018) is off-policy and reuses all past data. When would you prefer PPO over SAC, and vice versa?

---

**Previous:** [Day 28 — CS231n: CNNs for Visual Recognition](../28_CS231n/)

**Next:** [Day 30 — Deep Reinforcement Learning from Human Feedback](../30_RLHF/)
