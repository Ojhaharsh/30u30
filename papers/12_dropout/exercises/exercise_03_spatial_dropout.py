"""
Exercise 3: Spatial Dropout for CNNs

Goal: Implement Dropout2D (spatial dropout) and compare with standard dropout.

Time: 45-60 minutes
Difficulty: Medium ⏱️⏱️⏱️
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Dropout2D:
    """
    Spatial Dropout for 2D feature maps.
    
    Instead of dropping individual pixels, drops entire channels.
    This is better for convolutions because nearby pixels are correlated.
    
    TODO: Implement forward and backward methods.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of KEEPING each channel (not dropping!)
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with channel-wise dropout.
        
        Args:
            x: Input of shape (batch, channels, height, width)
            
        Returns:
            Output with some channels zeroed (same shape)
            
        Steps:
        1. If not training, return x unchanged
        2. Get batch_size, channels, height, width from x.shape
        3. Create mask of shape (batch, channels, 1, 1) - one value per channel
        4. Broadcast mask to full spatial dimensions
        5. Apply mask and scale by 1/p
        """
        if not self.training:
            return x
        
        # TODO: Implement spatial dropout
        # Hint: Random mask should have shape (batch, channels, 1, 1)
        # Then broadcast to (batch, channels, height, width)
        
        raise NotImplementedError("Implement forward pass!")
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass with same channel mask."""
        if not self.training:
            return grad_output
        
        # TODO: Implement backward pass
        # Same mask and scaling as forward
        
        raise NotImplementedError("Implement backward pass!")
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


def visualize_dropout_vs_spatial():
    """Compare standard dropout vs spatial dropout on feature maps."""
    from implementation import Dropout
    
    print("=" * 60)
    print("EXERCISE 3: SPATIAL DROPOUT FOR CNNs")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample feature map: (batch=1, channels=4, height=8, width=8)
    x = np.random.randn(1, 4, 8, 8)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch: {x.shape[0]}")
    print(f"  Channels: {x.shape[1]}")
    print(f"  Height x Width: {x.shape[2]}x{x.shape[3]}")
    
    # Standard dropout (drops individual elements)
    print("\n--- Standard Dropout ---")
    standard_dropout = Dropout(p=0.5)
    
    # Reshape for standard dropout (flatten spatial)
    x_flat = x.reshape(1, -1)  # (1, 256)
    y_flat = standard_dropout.forward(x_flat)
    
    zeros_standard = np.sum(y_flat == 0)
    print(f"Elements dropped: {zeros_standard}/{y_flat.size} ({100*zeros_standard/y_flat.size:.1f}%)")
    
    # TODO: Implement and test spatial dropout
    print("\n--- Spatial Dropout (TODO) ---")
    # spatial_dropout = Dropout2D(p=0.5)
    # y_spatial = spatial_dropout.forward(x)
    
    # # Count dropped channels
    # channels_dropped = 0
    # for c in range(x.shape[1]):
    #     if np.all(y_spatial[0, c] == 0):
    #         channels_dropped += 1
    # print(f"Channels dropped: {channels_dropped}/{x.shape[1]}")
    
    print("\nExercise not yet implemented!")
    print("Implement Dropout2D and run again.")
    
    # TODO: Visualize the difference
    # - Show original feature maps
    # - Show after standard dropout (scattered zeros)
    # - Show after spatial dropout (entire channels zero)


if __name__ == "__main__":
    visualize_dropout_vs_spatial()
