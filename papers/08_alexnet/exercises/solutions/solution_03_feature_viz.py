"""
Solution 3: AlexNet Feature Visualization
=========================================

Visualize what AlexNet learns.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_alexnet():
    """Load pretrained AlexNet."""
    model = models.alexnet(pretrained=True)
    model.eval()
    return model


def visualize_conv1_filters(model):
    """Visualize first layer filters."""
    # Get conv1 weights: [96, 3, 11, 11]
    weights = model.features[0].weight.data.cpu()
    
    # Normalize to 0-1
    weights = weights - weights.min()
    weights = weights / weights.max()
    
    # Plot 96 filters in 8x12 grid
    fig, axes = plt.subplots(8, 12, figsize=(15, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < 96:
            # Convert to HWC for imshow
            img = weights[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle('AlexNet Conv1 Filters (96 Gabor-like detectors)', fontsize=14)
    plt.tight_layout()
    plt.show()


def extract_activations(model, image, layer_idx):
    """Extract feature maps at given layer."""
    activations = {}
    
    def hook(module, inp, out):
        activations['out'] = out.detach()
    
    handle = model.features[layer_idx].register_forward_hook(hook)
    
    with torch.no_grad():
        _ = model(image)
    
    handle.remove()
    return activations['out']


def visualize_activations(model, image_tensor, layer_indices=[0, 3, 6, 8, 10]):
    """Visualize activations at multiple layers."""
    layer_names = {0: 'Conv1', 3: 'Conv2', 6: 'Conv3', 8: 'Conv4', 10: 'Conv5'}
    
    fig, axes = plt.subplots(len(layer_indices), 8, figsize=(16, len(layer_indices) * 2))
    
    for row, layer_idx in enumerate(layer_indices):
        acts = extract_activations(model, image_tensor, layer_idx)
        
        for col in range(8):
            ax = axes[row, col] if len(layer_indices) > 1 else axes[col]
            if col < acts.shape[1]:
                ax.imshow(acts[0, col].cpu().numpy(), cmap='viridis')
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(layer_names.get(layer_idx, f'Layer {layer_idx}'))
    
    plt.suptitle('Feature Maps at Different Depths', fontsize=14)
    plt.tight_layout()
    plt.show()


def get_sample_image():
    """Create sample image tensor."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create random colorful image for demo
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    return preprocess(img).unsqueeze(0)


def demo():
    """Run feature visualization demo."""
    print("AlexNet Feature Visualization")
    print("=" * 60)
    
    model = get_alexnet()
    
    print("\n1. Visualizing Conv1 filters...")
    visualize_conv1_filters(model)
    
    print("\n2. Visualizing feature maps...")
    image = get_sample_image()
    visualize_activations(model, image)
    
    print("\nObservations:")
    print("  - Conv1: Gabor-like edge/color detectors")
    print("  - Conv2-3: Textures and patterns")
    print("  - Conv4-5: Object parts and higher semantics")


if __name__ == "__main__":
    demo()
