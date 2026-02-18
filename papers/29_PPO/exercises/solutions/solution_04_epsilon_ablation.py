"""
Solution 4: Epsilon Ablation

This solution runs the ablation and prints the results.
The exercise_04 file already contains the full implementation.
Run exercise_04_epsilon_ablation.py directly to see the results.

Reference: Schulman et al. (2017), Table 1 and Section 6.1
https://arxiv.org/abs/1707.06347
"""

# The epsilon ablation is fully implemented in exercise_04_epsilon_ablation.py.
# This solution file explains the expected findings.

EXPECTED_FINDINGS = """
Expected findings from the epsilon ablation (spirit of Table 1 in the paper):

epsilon=0.05: Too conservative. The policy barely moves each iteration.
              Training is slow and may not converge within 50 iterations.

epsilon=0.1:  Slightly conservative. Usually works but slower than 0.2.

epsilon=0.2:  Paper default. Best balance of stability and speed.
              Should reach CartPole reward > 400 within 50 iterations.

epsilon=0.3:  Slightly aggressive. Usually still works on CartPole
              (which is a simple environment), but may show more variance.

epsilon=0.5:  Too aggressive. On CartPole this may still work (the environment
              is forgiving), but on harder tasks (LunarLander, MuJoCo) this
              would cause instability or policy collapse.

Key insight from the paper (Section 6.1, Table 1):
The clipping mechanism is the most important component of PPO.
Removing it entirely (epsilon=infinity, i.e., vanilla PG) causes
significantly worse performance on MuJoCo tasks.

Note: CartPole is too simple to clearly show the instability of large epsilon.
For a clearer demonstration, run exercise_05 (PPO vs. vanilla PG) on
LunarLander-v2.
"""

if __name__ == "__main__":
    print(EXPECTED_FINDINGS)
    print("To run the actual ablation, execute:")
    print("  python exercises/exercise_04_epsilon_ablation.py")
