# Day 12: Dropout - Exercises

Progressive exercises to master dropout and its variants.

---

## 🏋️ Exercise Overview

| Exercise | Difficulty | Time | Topic |
|----------|------------|------|-------|
| 1. Implement Dropout | ⏱️⏱️ | 30 min | Core dropout mechanics |
| 2. Dropout Rate Sweep | ⏱️⏱️⏱️ | 45 min | Finding optimal p |
| 3. Spatial Dropout | ⏱️⏱️⏱️ | 45 min | CNN-specific dropout |
| 4. MC Dropout | ⏱️⏱️⏱️⏱️ | 60 min | Uncertainty estimation |
| 5. Regularization Comparison | ⏱️⏱️⏱️⏱️⏱️ | 90 min | Dropout vs alternatives |

---

## Exercise 1: Implement Dropout from Scratch (⏱️⏱️)

**Goal:** Build a complete Dropout class with forward and backward passes.

### Tasks

1. Implement `forward()` with inverted dropout scaling
2. Implement `backward()` with proper gradient flow
3. Handle training vs evaluation mode correctly
4. Verify with numerical gradient checking

### Starter Code

```python
"""Exercise 1: Build Dropout from Scratch"""

import numpy as np

class Dropout:
    """
    Inverted Dropout Layer.
    
    TODO: Implement the forward and backward methods.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of KEEPING a neuron (not dropping!)
        """
        assert 0 < p <= 1, "Keep probability must be in (0, 1]"
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        During training:
        - Generate random mask (1 with prob p, 0 with prob 1-p)
        - Apply mask to input
        - Scale by 1/p (inverted dropout)
        
        During inference:
        - Return input unchanged
        
        Args:
            x: Input array of any shape
            
        Returns:
            Dropped and scaled output (same shape)
        """
        # TODO: Implement this
        pass
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Gradient only flows through kept neurons.
        Same mask and scaling applied.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        # TODO: Implement this
        pass


def test_dropout():
    """Test your implementation."""
    np.random.seed(42)
    
    # Create dropout layer
    dropout = Dropout(p=0.5)
    x = np.ones((4, 8))
    
    # Test training mode
    dropout.training = True
    y = dropout.forward(x)
    
    # Check: Some values should be 0, others scaled by 2
    assert np.any(y == 0), "Some values should be dropped"
    assert np.any(y == 2), "Kept values should be scaled by 1/p = 2"
    
    # Check: Expected value should be close to original
    np.random.seed(42)
    outputs = [dropout.forward(x) for _ in range(1000)]
    mean_output = np.mean(outputs, axis=0)
    assert np.allclose(mean_output, x, atol=0.1), f"Expected mean ~ 1, got {mean_output.mean()}"
    
    # Test eval mode
    dropout.training = False
    y_eval = dropout.forward(x)
    assert np.array_equal(y_eval, x), "Eval mode should return unchanged input"
    
    # Test backward
    dropout.training = True
    y = dropout.forward(x)
    grad = np.ones_like(y)
    grad_input = dropout.backward(grad)
    
    # Gradient should have same pattern as forward
    assert grad_input.shape == x.shape
    assert np.all((grad_input == 0) | (grad_input == 2)), "Gradient should be 0 or scaled"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    test_dropout()
```

### Expected Output

```
✓ All tests passed!
```

---

## Exercise 2: Dropout Rate Exploration (⏱️⏱️⏱️)

**Goal:** Train models with different dropout rates and find the sweet spot.

### Tasks

1. Train MNIST classifiers with p ∈ {0.0, 0.3, 0.5, 0.7, 0.9}
2. Plot training vs validation accuracy for each
3. Compute the train-valid gap
4. Identify the optimal dropout rate

### Starter Code

```python
"""Exercise 2: Find the Optimal Dropout Rate"""

import numpy as np
import matplotlib.pyplot as plt
from train_minimal import DropoutMLP, SGD, load_mnist, train_epoch, evaluate

def sweep_dropout_rates():
    """Train with different dropout rates and compare."""
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use subset for faster training
    X_train, y_train = X_train[:5000], y_train[:5000]
    
    dropout_rates = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for p in dropout_rates:
        print(f"\n--- Training with keep_prob = {p} ---")
        
        # TODO: Create model with this dropout rate
        # TODO: Train for 20 epochs
        # TODO: Record train and test accuracy
        
        pass
    
    # TODO: Plot results
    # - X-axis: dropout rate
    # - Y-axis: accuracy (train and test)
    # - Highlight the train-test gap
    
    pass


if __name__ == "__main__":
    sweep_dropout_rates()
```

### Questions to Answer

1. Which dropout rate gives the best test accuracy?
2. Which dropout rate has the smallest train-test gap?
3. What happens when dropout is too high (p=0.9)?
4. How does dropout affect training speed?

---

## Exercise 3: Spatial Dropout for CNNs (⏱️⏱️⏱️)

**Goal:** Implement Dropout2D and compare with standard dropout on images.

### Tasks

1. Implement Dropout2D (drop entire channels)
2. Build a simple CNN with both dropout types
3. Train on CIFAR-10 or MNIST
4. Compare performance and visualize dropped channels

### Starter Code

```python
"""Exercise 3: Spatial Dropout for CNNs"""

import numpy as np

class Dropout2D:
    """
    Spatial Dropout for 2D feature maps.
    
    Drops entire channels instead of individual elements.
    Better for convolutions because nearby pixels are correlated.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with channel-wise dropout.
        
        Args:
            x: Input of shape (batch, channels, height, width)
            
        Returns:
            Output with some channels zeroed
        """
        if not self.training:
            return x
        
        batch_size, channels, height, width = x.shape
        
        # TODO: Create mask for each channel (not each pixel!)
        # Shape should be (batch, channels, 1, 1)
        # Then broadcast to full spatial dimensions
        
        pass
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass with same channel mask."""
        # TODO: Implement
        pass


def visualize_dropout_vs_dropout2d():
    """Compare standard dropout vs spatial dropout on feature maps."""
    
    np.random.seed(42)
    
    # Create sample feature map
    x = np.random.randn(1, 4, 8, 8)  # 1 sample, 4 channels, 8x8
    
    # Standard dropout
    standard_dropout = Dropout(p=0.5)
    y_standard = standard_dropout.forward(x.copy())
    
    # Spatial dropout
    spatial_dropout = Dropout2D(p=0.5)
    y_spatial = spatial_dropout.forward(x.copy())
    
    # TODO: Visualize the difference
    # - Show original feature maps
    # - Show after standard dropout (scattered zeros)
    # - Show after spatial dropout (entire channels zero)
    
    pass


if __name__ == "__main__":
    visualize_dropout_vs_dropout2d()
```

---

## Exercise 4: MC Dropout for Uncertainty (⏱️⏱️⏱️⏱️)

**Goal:** Use dropout at inference time for uncertainty estimation.

### Tasks

1. Train a model with dropout
2. Keep dropout ON during inference
3. Run multiple forward passes (N=100)
4. Compute mean prediction and variance
5. Correlate uncertainty with prediction errors

### Starter Code

```python
"""Exercise 4: Monte Carlo Dropout for Uncertainty"""

import numpy as np
from train_minimal import DropoutMLP, load_mnist

def mc_dropout_inference(model, x, n_samples=100):
    """
    Run multiple forward passes with dropout enabled.
    
    Args:
        model: Model with dropout layers
        x: Input sample(s)
        n_samples: Number of forward passes
        
    Returns:
        mean_prediction: Average prediction
        uncertainty: Standard deviation (epistemic uncertainty)
    """
    model.train()  # Keep dropout ON!
    
    predictions = []
    for _ in range(n_samples):
        # TODO: Run forward pass
        # TODO: Collect predictions
        pass
    
    # TODO: Compute mean and std
    mean_prediction = None
    uncertainty = None
    
    return mean_prediction, uncertainty


def analyze_uncertainty():
    """Analyze relationship between uncertainty and errors."""
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Train model
    model = DropoutMLP(dropout_p=0.5)
    # TODO: Train the model
    
    # MC Dropout on test set
    n_test_samples = 100
    mean_preds, uncertainties = [], []
    
    for i in range(n_test_samples):
        mean, unc = mc_dropout_inference(model, X_test[i:i+1], n_samples=50)
        mean_preds.append(mean)
        uncertainties.append(unc)
    
    # TODO: Analyze
    # 1. Are wrong predictions more uncertain?
    # 2. Plot uncertainty vs accuracy
    # 3. Can we reject uncertain predictions to improve accuracy?
    
    pass


if __name__ == "__main__":
    analyze_uncertainty()
```

### Questions to Answer

1. Do incorrect predictions have higher uncertainty?
2. If we reject the top 10% most uncertain predictions, how much does accuracy improve?
3. How many forward passes are needed for stable uncertainty estimates?

---

## Exercise 5: Regularization Comparison (⏱️⏱️⏱️⏱️⏱️)

**Goal:** Compare dropout with other regularization techniques.

### Tasks

1. Implement/use multiple regularization techniques:
   - L2 regularization (weight decay)
   - L1 regularization
   - Dropout (various rates)
   - Early stopping
   - Data augmentation (for images)
   
2. Train on the same dataset with each technique
3. Compare: final accuracy, training curves, train-test gap
4. Find the best combination

### Starter Code

```python
"""Exercise 5: Regularization Showdown"""

import numpy as np
from typing import Dict

def train_with_regularization(
    X_train, y_train, X_test, y_test,
    regularization: str,
    **kwargs
) -> Dict:
    """
    Train a model with specified regularization.
    
    Args:
        regularization: One of 'none', 'l2', 'dropout', 'early_stopping', 'combined'
        
    Returns:
        Dictionary with train_acc, test_acc, history
    """
    
    if regularization == 'none':
        # TODO: Train without any regularization
        pass
    
    elif regularization == 'l2':
        # TODO: Train with L2 weight decay
        weight_decay = kwargs.get('weight_decay', 1e-4)
        pass
    
    elif regularization == 'dropout':
        # TODO: Train with dropout
        dropout_p = kwargs.get('dropout_p', 0.5)
        pass
    
    elif regularization == 'early_stopping':
        # TODO: Train with early stopping
        patience = kwargs.get('patience', 5)
        pass
    
    elif regularization == 'combined':
        # TODO: Train with dropout + L2 + early stopping
        pass
    
    return {
        'train_acc': None,
        'test_acc': None,
        'history': None
    }


def run_comparison():
    """Compare all regularization techniques."""
    
    from train_minimal import load_mnist
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Use subset for faster experiments
    X_train, y_train = X_train[:5000], y_train[:5000]
    
    techniques = [
        ('none', {}),
        ('l2', {'weight_decay': 1e-4}),
        ('l2', {'weight_decay': 1e-3}),
        ('dropout', {'dropout_p': 0.3}),
        ('dropout', {'dropout_p': 0.5}),
        ('dropout', {'dropout_p': 0.7}),
        ('early_stopping', {'patience': 5}),
        ('combined', {'dropout_p': 0.5, 'weight_decay': 1e-4}),
    ]
    
    results = {}
    for name, kwargs in techniques:
        key = f"{name}_{kwargs}" if kwargs else name
        print(f"\nTraining with {key}...")
        
        results[key] = train_with_regularization(
            X_train, y_train, X_test, y_test,
            regularization=name,
            **kwargs
        )
    
    # TODO: Create comparison table and plots
    # TODO: Determine best single technique and best combination
    
    pass


if __name__ == "__main__":
    run_comparison()
```

### Questions to Answer

1. Which single technique works best?
2. Does combining techniques help?
3. Is there such a thing as too much regularization?
4. When would you choose dropout over L2 or vice versa?

---

## 🎯 Success Criteria

- Exercise 1: All tests pass, numerical gradient check < 1e-5
- Exercise 2: Identify optimal dropout rate, explain tradeoffs
- Exercise 3: Spatial dropout implemented, visual comparison clear
- Exercise 4: Uncertainty correlates with errors (r > 0.3)
- Exercise 5: Find best regularization combo, explain reasoning

---

## 💡 Hints

**Exercise 1:**
- Remember: mask should be 0 or 1, not continuous
- Scaling factor is 1/p, not p
- During backward, gradient only flows through kept neurons

**Exercise 2:**
- Too little dropout → overfitting (large gap)
- Too much dropout → underfitting (low accuracy)
- Sweet spot is usually p ∈ [0.3, 0.7]

**Exercise 3:**
- For spatial dropout, mask shape is (batch, channels, 1, 1)
- Use np.broadcast_to to expand to full dimensions
- Adjacent pixels get same mask value

**Exercise 4:**
- Model must be in training mode (dropout ON)
- More samples = lower variance in uncertainty estimate
- 50-100 samples usually sufficient

**Exercise 5:**
- L2 works well for small datasets
- Dropout better for large networks
- Early stopping is almost always useful
- Data augmentation is most effective for images
