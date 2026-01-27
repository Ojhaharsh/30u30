"""
Day 8: AlexNet - Minimal Training Script

A simple script to experience the Deep Learning Revolution firsthand.
Train AlexNet and see how it learns to recognize images just like it did in 2012!

This script demonstrates:
- AlexNet architecture and training
- The power of deep learning for image recognition
- Key innovations that started the revolution

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

from implementation import AlexNet, AlexNetTrainer, get_data_loaders
from visualization import AlexNetVisualizer


def run_alexnet_demo():
    """
    Run a simplified AlexNet training demo on CIFAR-10.
    Shows the key concepts without requiring the full ImageNet dataset.
    """
    print("🔥 WELCOME TO THE DEEP LEARNING REVOLUTION!")
    print("=" * 60)
    print("Experience AlexNet - the network that changed everything")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print("   ✨ GPU acceleration enabled - just like in 2012!")
    else:
        print("   💻 Using CPU - will be slower but still works")
    print()
    
    # Create model
    print("🧠 Creating AlexNet...")
    model = AlexNet(num_classes=10, dropout=0.5)  # CIFAR-10 has 10 classes
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    print(f"   🏗️  Architecture: 5 Conv + 3 FC layers")
    print()
    
    # Prepare data
    print("📊 Preparing CIFAR-10 dataset...")
    print("   (Using CIFAR-10 instead of ImageNet for faster demo)")
    
    # CIFAR-10 transforms (adapted for AlexNet)
    transform_train = transforms.Compose([
        transforms.Resize(224),                    # Resize to AlexNet input size
        transforms.RandomCrop(224, padding=4),    # Random crop with padding
        transforms.RandomHorizontalFlip(0.5),     # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(                      # ImageNet normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"   📚 Training samples: {len(train_dataset):,}")
    print(f"   🔍 Test samples: {len(test_dataset):,}")
    print(f"   📦 Batch size: 32")
    print()
    
    # CIFAR-10 class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"   🏷️  Classes: {', '.join(classes)}")
    print()
    
    # Quick visualization of data
    print("🖼️  Sample images from dataset:")
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Show a few sample images
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        img = images[i]
        # Denormalize for display
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'{classes[labels[i]]}')
        axes[i].axis('off')
    
    plt.suptitle('Sample CIFAR-10 Images (Resized to 224x224 for AlexNet)')
    plt.tight_layout()
    plt.show()
    
    # Initialize trainer
    print("🚀 Setting up training...")
    trainer = AlexNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=0.001,           # Lower learning rate for stability
        momentum=0.9,       # Original AlexNet momentum
        weight_decay=0.0005 # Original weight decay
    )
    
    print("   ⚙️  Optimizer: SGD with momentum")
    print("   📈 Learning rate: 0.001")
    print("   🎯 Loss function: Cross-entropy")
    print("   🔄 Weight decay: 0.0005")
    print()
    
    # Train for a few epochs (quick demo)
    num_epochs = 3
    print(f"🎓 Training AlexNet for {num_epochs} epochs...")
    print("   (In 2012, they trained for 90 epochs on ImageNet!)")
    print()
    
    # Training loop with progress
    model.to(device)
    model.train()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        print(f"📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            trainer.optimizer.zero_grad()
            output = model(data)
            loss = trainer.criterion(output, target)
            
            # Backward pass
            loss.backward()
            trainer.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0 and batch_idx > 0:
                acc = 100. * correct / total
                print(f"   Batch {batch_idx:3d}: Loss={loss.item():.4f}, Acc={acc:.1f}%")
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += trainer.criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # Store history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(trainer.optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"   🎯 Train: Loss={epoch_loss:.4f}, Acc={epoch_acc:.1f}%")
        print(f"   ✅ Test:  Loss={val_loss:.4f}, Acc={val_acc:.1f}%")
        print()
    
    # Final results
    final_acc = history['val_acc'][-1]
    print("🎉 TRAINING COMPLETE!")
    print("=" * 40)
    print(f"🏆 Final Test Accuracy: {final_acc:.1f}%")
    print()
    
    if final_acc > 60:
        print("🌟 Excellent! AlexNet is learning well!")
        print("   The deep learning revolution is working!")
    elif final_acc > 40:
        print("✨ Good progress! AlexNet is starting to see patterns.")
        print("   With more training, it would get much better!")
    else:
        print("🚀 Early stages! The network needs more time to learn.")
        print("   Remember: AlexNet was trained for 5-6 days in 2012!")
    
    # Show training curves
    print("\n📈 Visualizing training progress...")
    
    visualizer = AlexNetVisualizer()
    fig = visualizer.plot_training_curves(history)
    plt.show()
    
    # Test the model on some examples
    print("\n🔍 Testing AlexNet on individual images...")
    model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Show results for first 4 images
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        img = images[i].cpu()
        # Denormalize
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        
        true_class = classes[labels[i]]
        pred_class = classes[predicted[i]]
        
        if predicted[i] == labels[i]:
            color = 'green'
            result = '✅ CORRECT'
        else:
            color = 'red'
            result = '❌ WRONG'
        
        axes[i].set_title(f'{result}\nTrue: {true_class}\nPred: {pred_class}', 
                         color=color, fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('AlexNet Predictions on Test Images')
    plt.tight_layout()
    plt.show()
    
    return model, history


def analyze_alexnet_innovations():
    """
    Demonstrate the key innovations that made AlexNet revolutionary.
    """
    print("\n" + "="*60)
    print("🚀 ALEXNET INNOVATIONS: What Made It Revolutionary")
    print("="*60)
    
    print("\n1️⃣  ReLU ACTIVATION FUNCTION")
    print("-" * 40)
    print("Before: sigmoid, tanh (vanishing gradients)")
    print("AlexNet: ReLU(x) = max(0, x)")
    print()
    print("Benefits:")
    print("   ⚡ 6x faster training")
    print("   📈 No vanishing gradients")
    print("   🧠 Biological plausibility")
    print("   💾 Computational efficiency")
    
    print("\n2️⃣  DROPOUT REGULARIZATION")
    print("-" * 40)
    print("Randomly set 50% of neurons to zero during training")
    print()
    print("Benefits:")
    print("   🎯 Prevents overfitting")
    print("   🔀 Forces ensemble learning")
    print("   💪 Robust feature learning")
    print("   📊 Better generalization")
    
    print("\n3️⃣  DATA AUGMENTATION")
    print("-" * 40)
    print("Random crops, horizontal flips, color jittering")
    print()
    print("Benefits:")
    print("   📈 Increased dataset size")
    print("   🔄 Better invariance")
    print("   🎯 Reduced overfitting")
    print("   🌍 Better generalization")
    
    print("\n4️⃣  GPU PARALLELIZATION")
    print("-" * 40)
    print("Used 2x NVIDIA GTX 580 GPUs")
    print()
    print("Benefits:")
    print("   🚀 10-20x faster training")
    print("   🔢 Enabled larger networks")
    print("   💰 Made deep learning practical")
    print("   🌟 Sparked the GPU revolution")
    
    print("\n5️⃣  LOCAL RESPONSE NORMALIZATION")
    print("-" * 40)
    print("Lateral inhibition inspired by neuroscience")
    print()
    print("Benefits:")
    print("   🧠 Biological inspiration")
    print("   📊 Better contrast")
    print("   🎯 Improved selectivity")
    print("   ✨ Feature enhancement")
    
    print("\n💡 THE MAGIC COMBINATION:")
    print("Each innovation was known, but AlexNet combined them perfectly!")
    print("The result: A 10.8% improvement that changed AI forever! 🌟")


def main():
    """
    Complete AlexNet demonstration showing the Deep Learning Revolution.
    """
    print("🌟 ALEXNET: THE DEEP LEARNING REVOLUTION BEGINS 🌟")
    print()
    print("Welcome to the moment that changed AI forever!")
    print("In 2012, three researchers from Toronto showed that")
    print("deep neural networks could see better than any system before.")
    print()
    print("Let's experience this revolution firsthand! 🚀")
    print()
    
    # Main training demo
    model, history = run_alexnet_demo()
    
    # Explain the innovations
    analyze_alexnet_innovations()
    
    # Final thoughts
    print("\n" + "="*60)
    print("🎉 CONGRATULATIONS!")
    print("="*60)
    print()
    print("You've just experienced the Deep Learning Revolution!")
    print("This same architecture:")
    print("   🚗 Powers self-driving cars")
    print("   🏥 Diagnoses medical images")  
    print("   📱 Recognizes faces in photos")
    print("   🎨 Creates digital art")
    print("   🌍 Moderates social media content")
    print()
    print("💫 The journey from here:")
    print("   📚 Day 9: ResNet (going deeper)")
    print("   🔬 Day 10: ResNet V2 (better training)")
    print("   🌐 Day 11: Dilated convolutions (global context)")
    print()
    print("🚀 Ready to continue the deep learning adventure?")
    print()
    print("AlexNet didn't just win ImageNet - it won the future! ✨")


if __name__ == "__main__":
    main()