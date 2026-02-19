# Data for Day 30 (RLHF)

Since we are training an agent on a Gymnasium environment (`CartPole-v1`), we do not need to download a static dataset.

Instead, the **Preference Dataset** is generated on-the-fly:
1.  The agent interacts with the environment to create **trajectory segments**.
2.  The **Synthetic Oracle** labels pairs of segments as "better" or "worse".
3.  These labeled pairs are stored in a replay buffer (in memory) to train the Reward Model.

## Outputs
When you run `train_minimal.py`, the following files will be saved here (or in the root):
-   `reward_model.pth`: The weights of the learned reward function.
-   `rlhf_results.pt`: Training logs (accuracies, true rewards, learned rewards).
