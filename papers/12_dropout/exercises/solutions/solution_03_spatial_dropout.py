"""
Solution 3: Spatial Dropout for CNNs

Complete solution with Dropout2D implementation and comparison.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Dropout2D:
    """
    Spatial Dropout for 2D feature maps.
    
    Drops entire channels instead of individual elements.
    Better for CNNs because nearby pixels are correlated.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of KEEPING each channel (not dropping!)
        """
        if not 0 < p <= 1:
            raise ValueError(f"Keep probability must be in (0, 1], got {p}")
        
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply channel-wise dropout.
        
        Args:
            x: Input of shape (batch, channels, height, width)
            
        Returns:
            Output with some channels zeroed (same shape)
        """
        if not self.training:
            return x
        
        batch_size, channels, height, width = x.shape
        
        # Create mask: one value per channel, shape (batch, channels, 1, 1)
        channel_mask = (np.random.rand(batch_size, channels, 1, 1) < self.p)
        
        # Broadcast to full spatial dimensions
        self.mask = np.broadcast_to(channel_mask, x.shape).astype(np.float32)
        
        # Apply mask and scale by 1/p (inverted dropout)
        return x * self.mask / self.p
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass with same channel mask."""
        if not self.training:
            return grad_output
        
        return grad_output * self.mask / self.p
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


def compare_dropout_types():
    """Compare standard dropout vs spatial dropout."""
    from implementation import Dropout
    
    print("=" * 60)
    print("SOLUTION 3: SPATIAL DROPOUT FOR CNNs")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample feature map
    batch_size = 2
    channels = 8
    height = 8
    width = 8
    
    x = np.random.randn(batch_size, channels, height, width)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch: {batch_size}")
    print(f"  Channels: {channels}")
    print(f"  Height x Width: {height}x{width}")
    print(f"  Total elements: {x.size}")
    
    # Test 1: Standard Dropout
    print("\n" + "=" * 40)
    print("STANDARD DROPOUT")
    print("=" * 40)
    
    standard_dropout = Dropout(p=0.5)
    y_standard = standard_dropout.forward(x.reshape(batch_size, -1)).reshape(x.shape)
    
    zeros_standard = np.sum(y_standard == 0)
    print(f"Elements dropped: {zeros_standard}/{y_standard.size} ({100*zeros_standard/y_standard.size:.1f}%)")
    
    # Check channel-by-channel
    print("\nChannel-by-channel analysis (sample 0):")
    for c in range(min(4, channels)):
        channel_zeros = np.sum(y_standard[0, c] == 0)
        total = height * width
        print(f"  Channel {c}: {channel_zeros}/{total} zeros ({100*channel_zeros/total:.1f}%)")
    
    # Test 2: Spatial Dropout (Dropout2D)
    print("\n" + "=" * 40)
    print("SPATIAL DROPOUT (Dropout2D)")
    print("=" * 40)
    
    spatial_dropout = Dropout2D(p=0.5)
    y_spatial = spatial_dropout.forward(x)
    
    # Count dropped channels
    channels_dropped = 0
    for b in range(batch_size):
        for c in range(channels):
            if np.all(y_spatial[b, c] == 0):
                channels_dropped += 1
    
    print(f"Channels entirely dropped: {channels_dropped}/{batch_size*channels}")
    print(f"  (Expected ~50% = {batch_size*channels*0.5:.1f})")
    
    print("\nChannel-by-channel analysis (sample 0):")
    for c in range(min(4, channels)):
        if np.all(y_spatial[0, c] == 0):
            print(f"  Channel {c}: ENTIRELY DROPPED")
        else:
            print(f"  Channel {c}: KEPT (scaled by 2x)")
    
    # Test 3: Backward pass
    print("\n" + "=" * 40)
    print("BACKWARD PASS TEST")
    print("=" * 40)
    
    grad_output = np.ones_like(x)
    
    spatial_dropout.training = True
    _ = spatial_dropout.forward(x)  # Re-generate mask
    grad_input = spatial_dropout.backward(grad_output)
    
    print(f"Grad input shape: {grad_input.shape}")
    print(f"Gradient flows through kept channels only: ", end="")
    
    # Verify gradients
    kept_ok = True
    dropped_ok = True
    
    for b in range(batch_size):
        for c in range(channels):
            if np.all(spatial_dropout.mask[b, c] == 0):
                if not np.all(grad_input[b, c] == 0):
                    dropped_ok = False
            else:
                if np.any(grad_input[b, c] == 0):
                    kept_ok = False
    
    print("PASS" if (kept_ok and dropped_ok) else "FAIL")
    
    # Visual comparison
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        
        # Original channels
        for c in range(4):
            axes[0, c].imshow(x[0, c], cmap='viridis')
            axes[0, c].set_title(f'Original Ch {c}', fontsize=10)
            axes[0, c].axis('off')
        
        # After spatial dropout
        for c in range(4):
            axes[1, c].imshow(y_spatial[0, c], cmap='viridis')
            status = "DROPPED" if np.all(y_spatial[0, c] == 0) else "KEPT (2x)"
            axes[1, c].set_title(f'After Dropout: {status}', fontsize=10)
            axes[1, c].axis('off')
        
        axes[0, 0].set_ylabel('Original', fontsize=12, labelpad=40, rotation=0, ha='right')
        axes[1, 0].set_ylabel('Dropout2D', fontsize=12, labelpad=40, rotation=0, ha='right')
        
        plt.suptitle('Spatial Dropout: Entire Channels Dropped', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig('spatial_dropout_comparison.png', dpi=150)
        plt.show()
        
        print("\nPlot saved to 'spatial_dropout_comparison.png'")
        
    except ImportError:
        print("\nMatplotlib not available for visualization.")
    
    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
1. Standard Dropout drops individual elements (scattered pattern)
2. Spatial Dropout drops entire channels (structured pattern)
3. For CNNs, nearby pixels are CORRELATED:
   - Dropping one pixel: neighbors can "fill in"
   - Dropping entire channel: must use other features
4. Spatial Dropout forces more diverse feature usage
5. Use Dropout2D for conv layers, standard Dropout for FC layers
""")


if __name__ == "__main__":
    compare_dropout_types()
