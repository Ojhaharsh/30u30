"""
Exercise 3: Feature Visualization
=================================

Goal: Visualize what AlexNet learns at each layer.

Time: 1-2 hours
Difficulty: Medium ⏱️⏱️
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_pretrained_alexnet():
    """Load pretrained AlexNet."""
    # TODO 1: Load pretrained model
    model = None  # TODO: models.alexnet(pretrained=True)
    model.eval()
    return model


def visualize_conv1_filters(model):
    """
    Visualize the first convolutional layer filters.
    
    Conv1 learns Gabor-like edge and color detectors.
    """
    # TODO 2: Get first conv layer weights
    # model.features[0] is the first conv layer
    weights = None  # TODO: model.features[0].weight.data.cpu()
    
    # Weights shape: [96, 3, 11, 11] - 96 filters, RGB, 11x11 kernel
    
    # Normalize for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # TODO 3: Plot filters in a grid
    fig, axes = plt.subplots(8, 12, figsize=(15, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < 96:
            # TODO: Get filter i and convert to displayable format
            # filter_img = weights[i].permute(1, 2, 0).numpy()
            pass
        ax.axis('off')
    
    plt.suptitle('AlexNet Conv1 Filters (Gabor-like detectors)', fontsize=14)
    plt.tight_layout()
    plt.show()


def get_activations(model, image, layer_idx):
    """
    Get feature map activations for a given layer.
    
    Args:
        model: AlexNet model
        image: Input image tensor
        layer_idx: Index of layer in model.features
    """
    activations = {}
    
    def hook(module, input, output):
        activations['output'] = output.detach()
    
    # TODO 4: Register hook on target layer
    handle = None  # TODO: model.features[layer_idx].register_forward_hook(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hook
    if handle:
        handle.remove()
    
    return activations.get('output')


def visualize_feature_maps(model, image, layer_idx=0, num_maps=16):
    """
    Visualize feature maps for a given image and layer.
    """
    # Get activations
    activations = get_activations(model, image, layer_idx)
    
    if activations is None:
        print("No activations captured!")
        return
    
    # TODO 5: Plot feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i < min(num_maps, activations.shape[1]):
            # TODO: Get feature map i
            # feature_map = activations[0, i].cpu().numpy()
            # ax.imshow(feature_map, cmap='viridis')
            pass
        ax.axis('off')
    
    plt.suptitle(f'Feature Maps at Layer {layer_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()


def load_and_preprocess_image(image_path):
    """Load and preprocess image for AlexNet."""
    
    # TODO 6: Define preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load image
    image = None  # TODO: Image.open(image_path).convert('RGB')
    
    # Preprocess
    image_tensor = None  # TODO: preprocess(image).unsqueeze(0)
    
    return image_tensor


def compare_activations(model, images, labels, layer_idx=0):
    """
    Compare activations for different image categories.
    """
    fig, axes = plt.subplots(len(images), 9, figsize=(18, 4 * len(images)))
    
    for row, (img, label) in enumerate(zip(images, labels)):
        # Original image
        if axes.ndim == 1:
            ax_img = axes[0]
        else:
            ax_img = axes[row, 0]
        # TODO 7: Show original image
        
        # Get activations
        activations = get_activations(model, img, layer_idx)
        
        if activations is not None:
            # Show top 8 feature maps
            for i in range(8):
                if axes.ndim == 1:
                    ax = axes[i + 1]
                else:
                    ax = axes[row, i + 1]
                # TODO 8: Show feature map i
                ax.axis('off')
        
        # Add label
        if axes.ndim == 1:
            axes[0].set_title(label)
        else:
            axes[row, 0].set_title(label)
    
    plt.suptitle('Activation Comparison Across Categories', fontsize=14)
    plt.tight_layout()
    plt.show()


def find_maximally_activating_regions(model, images, layer_idx, filter_idx):
    """
    Find which image regions maximally activate a specific filter.
    """
    max_activations = []
    
    for img in images:
        activations = get_activations(model, img, layer_idx)
        
        if activations is not None:
            # Get activation for specific filter
            filter_activation = activations[0, filter_idx]
            max_val = filter_activation.max().item()
            max_idx = (filter_activation == filter_activation.max()).nonzero()
            max_activations.append({
                'value': max_val,
                'location': max_idx[0] if len(max_idx) > 0 else None,
                'activation_map': filter_activation.cpu().numpy()
            })
    
    return max_activations


def demo():
    """Demonstrate feature visualization."""
    print("AlexNet Feature Visualization Demo")
    print("=" * 60)
    
    # Load model
    print("Loading pretrained AlexNet...")
    model = get_pretrained_alexnet()
    
    if model is None:
        print("Could not load model!")
        return
    
    print("\n1. Visualizing Conv1 filters...")
    visualize_conv1_filters(model)
    
    # Create a random test image
    print("\n2. Visualizing feature maps...")
    test_image = torch.randn(1, 3, 224, 224)
    
    for layer_idx in [0, 3, 6]:  # Conv1, Conv2, Conv3
        print(f"   Layer {layer_idx}...")
        visualize_feature_maps(model, test_image, layer_idx)
    
    print("\n" + "=" * 60)
    print("✅ Visualization complete!")
    print("Try with real images for more interesting results.")


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "=" * 60)
    print("Instructions:")
    print("1. Fill in all TODOs")
    print("2. Run demo() to test with random images")
    print("3. Try with real images for better visualization")
    print("=" * 60 + "\n")
    
    # Uncomment when ready:
    # demo()
