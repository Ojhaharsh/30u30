# Solutions: Machine Super Intelligence

This directory contains the reference solutions for the Day 27 exercises. 

These solutions are here to help you **learn**, not to copy-paste. Here's the recommended approach:

1. **Attempt the exercise first** (spend at least 30-60 minutes).
2. **Get stuck?** Review the relevant section in the main `README.md`.
3. **Still stuck?** Look at just the function you need help with in the solution.
4. **Compare your solution** with ours after completing the exercise.


## Overview of Solutions

| Solution | Key Logic |
|----------|-----------|
| `solution_01_formula.py` | Uses `2**(-k)` as the weight for the summation. |
| `solution_02_environments.py` | Implements a simple branching observation logic. |
| `solution_03_epsilon.py` | Standard `if np.random.random() < self.epsilon` exploration. |
| `solution_04_agent.py` | Uses a dictionary to remember sequence transitions `(last_obs -> current_obs)`. |
| `solution_05_invariance.py` | Proves that shift in $K$ scales the final score by a constant factor. |

## Educational Notes

- **Exercise 1:** Notice how sensitive $\Upsilon$ is to $K$. An agent that masters many "Hard" tasks but fails "Easy" tasks will have a much lower intelligence score than a generalist.
- **Exercise 4:** This agent represents a very basic form of **inducting the environment's program**. In a more complex AIXI setting, this would involve searching over all programs, but here we just learn a simple state-transition map.
- **Exercise 5:** The Invariance Theorem is fundamental because it means our specific choice of "Coding Language" (Python vs. C++ vs. Turing Machine) only changes the final intelligence score by a constant, fixed multiplier. It doesn't change the *relative* ranking of agents.

---

**Next:** [Day 28 — CS231n: CNNs for Visual Recognition](../../28_cs231n/)
