"""
Day 9: ResNet - Minimal Training Script

Experience the skip connection revolution! Train ResNet and see how
skip connections solve the degradation problem and enable deep learning.

This script demonstrates:
- The degradation problem (deeper networks performing worse)
- How skip connections solve this
- Training very deep networks successfully
- Gradient flow analysis

Author: 30u30 Project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from implementation import create_resnet, ResNetAnalyzer
from visualization import ResNetVisualizer


def demonstrate_skip_connections():
    """
    Demonstrate the power of skip connections through visual comparison.
    """
    print("🔗 SKIP CONNECTION DEMONSTRATION")
    print("=" * 50)
    
    # Create visualizer
    visualizer = ResNetVisualizer()
    
    print("📊 Showing the degradation problem and its solution...")
    
    # Show degradation problem
    fig1 = visualizer.plot_degradation_problem()
    plt.show()
    
    print("\n🔍 Understanding skip connections conceptually...")
    
    # Show skip connection concept
    fig2 = visualizer.plot_skip_connection_concept()
    plt.show()
    
    print("✨ Skip connections create gradient highways for deep learning!")


def compare_resnet_variants():
    """
    Compare different ResNet variants and their capabilities.
    """
    print("\n🏗️  RESNET VARIANT COMPARISON")
    print("=" * 50)
    
    variants = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    
    print("Creating and analyzing different ResNet variants...\n")
    
    for variant in variants:
        print(f"🔍 Analyzing {variant.upper()}:")
        
        # Create model
        model = create_resnet(variant, num_classes=10)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count layers
        conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        print(f"   🏗️  Convolutional layers: {conv_layers}")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        
        print(f"   ✅ Output shape: {y.shape}")
        print(f"   💡 Key insight: {variant.upper()} can train successfully with skip connections!")
        print()
    
    print("🚀 All variants work! Skip connections enable arbitrary depth!")


def analyze_skip_connections():
    """
    Analyze how skip connections work in practice.
    """
    print("\n🔬 SKIP CONNECTION ANALYSIS")
    print("=" * 50)
    
    # Create ResNet-18 for analysis
    model = create_resnet('resnet18', num_classes=10)
    analyzer = ResNetAnalyzer(model)
    
    # Create sample input
    x = torch.randn(2, 3, 224, 224)  # Small batch for analysis
    
    print("🔍 Analyzing skip connection contributions...")
    
    # Analyze skip connections
    skip_analysis = analyzer.analyze_skip_connections(x)
    
    print("\n📊 Skip Connection Analysis Results:")
    print("-" * 40)
    
    # Show first few layers
    for name, stats in list(skip_analysis.items())[:5]:
        layer_name = name.split('.')[-1] if '.' in name else name
        print(f"🔗 {layer_name}:")
        print(f"   Identity norm: {stats['identity_norm']:.3f}")
        print(f"   Residual norm: {stats['residual_norm']:.3f}")
        print(f"   Residual/Identity ratio: {stats['residual_ratio']:.3f}")
        
        if stats['residual_ratio'] < 0.5:
            print("   💡 Identity path dominates - preserving information")
        elif stats['residual_ratio'] > 2.0:
            print("   💡 Residual path dominates - learning new features")
        else:
            print("   💡 Balanced - both paths contribute")
        print()
    
    # Visualize the analysis
    visualizer = ResNetVisualizer()
    fig = visualizer.plot_skip_connection_analysis(skip_analysis)
    plt.show()
    
    print("✨ Skip connections automatically balance identity vs transformation!")


def mini_training_demo():
    """
    Run a mini training demo to see ResNet in action.
    """
    print("\n🎓 MINI TRAINING DEMO")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create ResNet-18 (smaller for demo)
    model = create_resnet('resnet18', num_classes=10)
    model = model.to(device)
    
    print(f"🧠 Created ResNet-18 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare CIFAR-10 data (small dataset for demo)
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("📊 Loading CIFAR-10 dataset...")
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    
    # Use smaller batches and subset for demo
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"📚 Training set: {len(train_dataset)} images")
    print(f"🔍 Test set: {len(test_dataset)} images")
    print(f"🏷️  Classes: {', '.join(classes)}")
    print()
    
    # Mini training loop (2 epochs for demo)
    print("🚀 Starting mini training session (2 epochs)...")
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(2):
        print(f"\n📅 Epoch {epoch + 1}/2")
        print("-" * 30)
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Print progress every 500 batches
            if batch_idx % 500 == 0 and batch_idx > 0:
                acc = 100. * train_correct / train_total
                print(f"   Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.1f}%")
                
                # Early break for demo
                if batch_idx >= 1000:
                    break
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / min(1000, len(train_loader))
        epoch_train_acc = 100. * train_correct / train_total
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
                
                # Early break for demo
                if batch_idx >= 200:
                    break
        
        epoch_test_loss = test_loss / min(200, len(test_loader))
        epoch_test_acc = 100. * test_correct / test_total
        
        # Store results
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        print(f"   🎯 Train: Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.1f}%")
        print(f"   ✅ Test:  Loss={epoch_test_loss:.4f}, Acc={epoch_test_acc:.1f}%")
    
    print(f"\n🎉 Training complete!")
    print(f"Final test accuracy: {history['test_acc'][-1]:.1f}%")
    print()
    
    if history['test_acc'][-1] > 30:
        print("🌟 Great! ResNet is learning successfully!")
        print("   Skip connections enabled this deep network to train effectively!")
    else:
        print("🚀 ResNet is making progress! More training would improve results.")
    
    # Show a simple training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['test_loss'], 'r-o', label='Test Loss')
    ax1.set_title('Training Progress: Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'g-o', label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], 'm-o', label='Test Accuracy')
    ax2.set_title('Training Progress: Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('🎓 ResNet Training Progress: Skip Connections Working!')
    plt.tight_layout()
    plt.show()
    
    return model, history


def main():
    """
    Complete ResNet demonstration showing the power of skip connections.
    """
    print("🌟 WELCOME TO THE SKIP CONNECTION REVOLUTION! 🌟")
    print()
    print("ResNet solved the degradation problem and enabled")
    print("training networks with 100+ layers successfully!")
    print("Let's see how skip connections work their magic! 🔗")
    print()
    
    # 1. Demonstrate skip connections conceptually
    demonstrate_skip_connections()
    
    # 2. Compare different variants
    compare_resnet_variants()
    
    # 3. Analyze skip connections in detail
    analyze_skip_connections()
    
    # 4. Mini training demo
    model, history = mini_training_demo()
    
    # Final insights
    print("\n" + "="*60)
    print("🎉 SKIP CONNECTION REVOLUTION COMPLETE!")
    print("="*60)
    print()
    print("🔍 WHAT WE DISCOVERED:")
    print("   🔗 Skip connections solve the degradation problem")
    print("   🚀 Enable training networks with 152+ layers")
    print("   🌊 Create gradient highways for deep learning")
    print("   ⚖️ Automatically balance identity vs transformation")
    print()
    print("💡 KEY INSIGHTS:")
    print("   • Residual learning: F(x) = H(x) - x")
    print("   • Identity mappings are easy to learn")
    print("   • Deeper networks actually work better!")
    print("   • Foundation for modern deep learning")
    print()
    print("🌍 IMPACT ON AI:")
    print("   🖼️ Computer vision revolution")
    print("   🧠 Enabled Transformers (attention + residuals)")
    print("   📱 Mobile AI applications")
    print("   🏥 Medical image analysis")
    print("   🚗 Autonomous vehicle vision")
    print()
    print("🚀 NEXT STEPS:")
    print("   📚 Day 10: ResNet V2 (pre-activation improvements)")
    print("   🌐 Day 11: Dilated convolutions (global context)")
    print("   🔮 Modern: Vision Transformers, EfficientNets")
    print()
    print("✨ ResNet didn't just enable deeper networks -")
    print("   it fundamentally changed how we think about")
    print("   neural network design! 🏗️")


if __name__ == "__main__":
    main()