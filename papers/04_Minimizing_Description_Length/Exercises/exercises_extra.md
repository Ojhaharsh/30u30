# Day 4 Extra Exercises: Bayesian Networks

These exercises build on the MDL implementation. All are our additions for practice.

## Exercise 1: The "Reparameterization" Trick (Math -> Code)
**Goal:** Implement the Gaussian sampling manually.
The core of this paper is $w = \mu + \sigma \cdot \epsilon$.
1. Create a function `sample_gaussian(mu, rho)`.
2. It must convert `rho` to `sigma` using Softplus: $\log(1 + e^\rho)$.
3. It must sample $\epsilon \sim N(0, 1)$.
4. It must return the sampled weight.
5. **Verify:** If you run it 1000 times with fixed mu/rho, does the mean/std match the expected values?

## Exercise 2: The "Gap" Experiment (The Hello World)
**Goal:** Prove the model knows what it doesn't know.
1. Generate data: $y = x^3$ for $x \in [-4, -2]$ AND $x \in [2, 4]$. (Leave a gap in $[-2, 2]$).
2. Train your `MDLNetwork` on this.
3. **Task:** Plot the predictions.
    * Does the model connect the two sides with a confident line? (Bad)
    * Or does the uncertainty bubble explode in the middle? (Good)

## Exercise 3: Breaking the Model (The Beta Parameter)
**Goal:** Understand `kl_weight` (Beta).
Train the model from Ex 2 three times with different betas:
1.  `kl_weight = 0.0` (Standard MLE). What happens to uncertainty?
2.  `kl_weight = 100.0` (Prior Dominated). What happens to the prediction?
3.  `kl_weight = 0.1` (Balanced).
**Deliverable:** A plot showing all three behaviors.

## Exercise 4: Monte Carlo Predictions
**Goal:** Understand why we sample multiple times.
1. Train a model.
2. Pick a test point $x_{test}$.
3. Predict ONCE. Record value.
4. Predict 100 times. Record mean value.
5. **Question:** Which one is smoother? Why does the single prediction look "jagged" when you plot a curve?

## Exercise 5: The "Free Lunch" (Pruning)
**Goal:** Compress the network.
1. Train a model until convergence.
2. Look at the `sigma` (uncertainty) of the weights in Layer 1.
3. **Task:**
    * Find all weights where $\text{SNR} = |\mu| / \sigma < 0.1$.
    * Hard-set those weights to 0.0.
    * Run the model again. Does the error increase?
    * Calculate how many parameters you just "deleted" for free.

---

**Need Help?** Check `solutions.py` for the answers.