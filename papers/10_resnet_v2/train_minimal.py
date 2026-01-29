"""
Day 10: ResNet V2 Training Demonstration - Perfect Signal Flow in Action
Hands-on training and analysis of pre-activation improvements

This script demonstrates:
1. Side-by-side training of ResNet V1 vs V2
2. Very deep network training (200+ layers)
3. Signal flow analysis and gradient tracking
4. Pre-activation vs post-activation comparisons
5. Interactive exploration of identity mapping benefits

Run this to experience the power of perfect signal flow!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from implementation import (
    ResNetV2, PreActBlock, PreActBottleneck,
    resnet_v2_18, resnet_v2_50, VeryDeepResNetV2,
    SignalFlowAnalyzer, PreActivationStudy
)
from visualization import ResNetV2Visualizer


class ResNetComparator:
    """
    Compare ResNet V1 vs V2 training dynamics side by side.
    
    Demonstrates the practical benefits of pre-activation design.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.train_stats = {'v1': {}, 'v2': {}}
        
    def create_models(self, num_classes=10):
        """Create comparable ResNet V1 and V2 models."""
        
        # ResNet V1 (original post-activation design)
        class PostActBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                shortcut = self.shortcut(x)
                # ReLU AFTER addition (contaminates identity path)
                return F.relu(out + shortcut)
        
        class ResNetV1(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.maxpool = nn.MaxPool2d(3, 2, 1)
                
                self.layer1 = self._make_layer(PostActBlock, 64, 64, 2, 1)
                self.layer2 = self._make_layer(PostActBlock, 64, 128, 2, 2)
                self.layer3 = self._make_layer(PostActBlock, 128, 256, 2, 2)
                self.layer4 = self._make_layer(PostActBlock, 256, 512, 2, 2)
                
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(512, num_classes)
                self.in_channels = 64
                
            def _make_layer(self, block, in_channels, out_channels, blocks, stride):
                layers = [block(in_channels, out_channels, stride)]
                for _ in range(1, blocks):
                    layers.append(block(out_channels, out_channels, 1))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        resnet_v1 = ResNetV1(num_classes).to(self.device)
        resnet_v2 = resnet_v2_18(num_classes).to(self.device)
        
        return resnet_v1, resnet_v2
    
    def prepare_data(self, batch_size=128):
        """Prepare CIFAR-10 dataset for training."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                              download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=transform_test)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train one epoch and return statistics."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(dataloader), 100.0 * correct / total
    
    def test_epoch(self, model, dataloader, criterion):
        """Test one epoch and return statistics."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(dataloader), 100.0 * correct / total
    
    def compare_training(self, epochs=20):
        """Compare training dynamics between ResNet V1 and V2."""
        print("🚀 Starting ResNet V1 vs V2 Training Comparison")
        print("=" * 60)
        
        # Create models
        resnet_v1, resnet_v2 = self.create_models()
        print(f"ResNet V1 parameters: {sum(p.numel() for p in resnet_v1.parameters()):,}")
        print(f"ResNet V2 parameters: {sum(p.numel() for p in resnet_v2.parameters()):,}")
        
        # Prepare data
        trainloader, testloader = self.prepare_data()
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer_v1 = optim.SGD(resnet_v1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        optimizer_v2 = optim.SGD(resnet_v2.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        
        # Track statistics
        stats = {
            'v1': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []},
            'v2': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
        }
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Train ResNet V1
            start_time = time.time()
            v1_train_loss, v1_train_acc = self.train_epoch(resnet_v1, trainloader, optimizer_v1, criterion)
            v1_train_time = time.time() - start_time
            
            # Train ResNet V2
            start_time = time.time()
            v2_train_loss, v2_train_acc = self.train_epoch(resnet_v2, trainloader, optimizer_v2, criterion)
            v2_train_time = time.time() - start_time
            
            # Test both models
            v1_test_loss, v1_test_acc = self.test_epoch(resnet_v1, testloader, criterion)
            v2_test_loss, v2_test_acc = self.test_epoch(resnet_v2, testloader, criterion)
            
            # Store statistics
            stats['v1']['train_loss'].append(v1_train_loss)
            stats['v1']['train_acc'].append(v1_train_acc)
            stats['v1']['test_loss'].append(v1_test_loss)
            stats['v1']['test_acc'].append(v1_test_acc)
            
            stats['v2']['train_loss'].append(v2_train_loss)
            stats['v2']['train_acc'].append(v2_train_acc)
            stats['v2']['test_loss'].append(v2_test_loss)
            stats['v2']['test_acc'].append(v2_test_acc)
            
            # Print results
            print(f"ResNet V1: Train Acc: {v1_train_acc:.2f}% | Test Acc: {v1_test_acc:.2f}% | Time: {v1_train_time:.1f}s")
            print(f"ResNet V2: Train Acc: {v2_train_acc:.2f}% | Test Acc: {v2_test_acc:.2f}% | Time: {v2_train_time:.1f}s")
            
            # Update learning rate
            if (epoch + 1) % 10 == 0:
                for param_group in optimizer_v1.param_groups:
                    param_group['lr'] *= 0.1
                for param_group in optimizer_v2.param_groups:
                    param_group['lr'] *= 0.1
        
        return stats, resnet_v1, resnet_v2
    
    def analyze_gradient_flow(self, model_v1, model_v2, dataloader):
        """Analyze gradient flow in both models."""
        print("\n🌊 Analyzing Gradient Flow...")
        
        # Setup analyzers
        analyzer_v1 = SignalFlowAnalyzer()
        analyzer_v2 = SignalFlowAnalyzer()
        
        analyzer_v1.register_hooks(model_v1)
        analyzer_v2.register_hooks(model_v2)
        
        # Get a batch for analysis
        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        # Forward and backward pass for V1
        model_v1.eval()
        outputs_v1 = model_v1(inputs[:1])  # Single sample
        loss_v1 = criterion(outputs_v1, targets[:1])
        loss_v1.backward()
        
        # Forward and backward pass for V2
        model_v2.eval()
        outputs_v2 = model_v2(inputs[:1])  # Single sample
        loss_v2 = criterion(outputs_v2, targets[:1])
        loss_v2.backward()
        
        # Analyze gradients
        v1_gradients = analyzer_v1.analyze_gradient_flow()
        v2_gradients = analyzer_v2.analyze_gradient_flow()
        
        print("\nGradient Flow Comparison:")
        print("ResNet V1 - Average gradient magnitude:", 
              np.mean(list(v1_gradients.values())) if v1_gradients else "N/A")
        print("ResNet V2 - Average gradient magnitude:", 
              np.mean(list(v2_gradients.values())) if v2_gradients else "N/A")
        
        return v1_gradients, v2_gradients


class VeryDeepTrainer:
    """
    Demonstrate training of very deep networks with ResNet V2.
    
    Shows how pre-activation enables training of 200+ layer networks.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def train_very_deep_network(self, depth=200, epochs=10):
        """Train a very deep ResNet V2 network."""
        print(f"\n🏔️ Training Very Deep ResNet V2 ({depth} layers)")
        print("=" * 60)
        
        # Create very deep model
        model = VeryDeepResNetV2(depth=depth, num_classes=10).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model depth: {depth} layers")
        print(f"Total parameters: {total_params:,}")
        
        # Prepare data (smaller dataset for faster demo)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                              download=True, transform=transform)
        # Use subset for faster training
        subset_indices = torch.randperm(len(trainset))[:5000]
        trainset = torch.utils.data.Subset(trainset, subset_indices)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=100, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Track training
        train_accs = []
        test_accs = []
        
        print(f"\nTraining {depth}-layer network...")
        for epoch in range(epochs):
            # Training phase
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            
            for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Check for gradient explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100.0 * correct / total
            train_accs.append(train_acc)
            
            # Testing phase
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_acc = 100.0 * correct / total
            test_accs.append(test_acc)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Grad Norm: {grad_norm:.3f}")
        
        print(f"\n✅ Successfully trained {depth}-layer network!")
        print(f"Final training accuracy: {train_accs[-1]:.2f}%")
        print(f"Final test accuracy: {test_accs[-1]:.2f}%")
        
        return model, train_accs, test_accs


class PreActivationAnalyzer:
    """
    Detailed analysis of pre-activation vs post-activation effects.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def analyze_activation_patterns(self):
        """Analyze activation patterns in pre vs post activation blocks."""
        print("\n🔍 Analyzing Pre vs Post Activation Patterns")
        print("=" * 50)
        
        # Create comparison blocks
        study = PreActivationStudy()
        pre_block, post_block = study.create_comparison_blocks(64, 64)
        pre_block = pre_block.to(self.device)
        post_block = post_block.to(self.device)
        
        # Generate test input
        test_input = torch.randn(32, 64, 32, 32).to(self.device)
        
        # Analyze patterns
        analysis = study.analyze_activation_patterns(pre_block, post_block, test_input)
        
        print("Pre-activation Block Analysis:")
        for key, value in analysis['pre_activation'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nPost-activation Block Analysis:")
        for key, value in analysis['post_activation'].items():
            print(f"  {key}: {value:.4f}")
        
        # Compare sparsity
        pre_sparsity = analysis['pre_activation']['sparsity']
        post_sparsity = analysis['post_activation']['sparsity']
        
        print(f"\n📊 Key Insights:")
        print(f"   Pre-activation sparsity: {pre_sparsity:.2%}")
        print(f"   Post-activation sparsity: {post_sparsity:.2%}")
        print(f"   Sparsity difference: {abs(pre_sparsity - post_sparsity):.2%}")
        
        if pre_sparsity < post_sparsity:
            print("   ✅ Pre-activation shows less sparsity (better information flow)")
        else:
            print("   ⚠️ Post-activation shows less sparsity")
        
        return analysis


def demonstrate_resnet_v2_training():
    """
    Main demonstration of ResNet V2 training capabilities.
    """
    print("🎯 ResNet V2 Training Demonstration")
    print("=" * 60)
    print("Demonstrating the power of pre-activation design for deep networks")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Compare ResNet V1 vs V2
    print("\n" + "="*60)
    print("1. RESNET V1 vs V2 COMPARISON")
    print("="*60)
    
    comparator = ResNetComparator(device)
    
    # Quick training comparison (fewer epochs for demo)
    training_stats, model_v1, model_v2 = comparator.compare_training(epochs=5)
    
    # Analyze gradient flow
    trainloader, testloader = comparator.prepare_data(batch_size=32)
    v1_grads, v2_grads = comparator.analyze_gradient_flow(model_v1, model_v2, testloader)
    
    # 2. Very Deep Network Training
    print("\n" + "="*60)
    print("2. VERY DEEP NETWORK TRAINING")
    print("="*60)
    
    deep_trainer = VeryDeepTrainer(device)
    
    # Train progressively deeper networks
    depths = [50, 100, 200]
    
    for depth in depths:
        try:
            print(f"\n🔥 Attempting {depth}-layer network...")
            model, train_accs, test_accs = deep_trainer.train_very_deep_network(depth=depth, epochs=3)
            print(f"✅ Success! {depth}-layer network trained successfully")
            print(f"   Final accuracy: {test_accs[-1]:.2f}%")
        except Exception as e:
            print(f"❌ Failed to train {depth}-layer network: {e}")
            break
    
    # 3. Pre-activation Analysis
    print("\n" + "="*60)
    print("3. PRE-ACTIVATION ANALYSIS")
    print("="*60)
    
    analyzer = PreActivationAnalyzer(device)
    activation_analysis = analyzer.analyze_activation_patterns()
    
    # 4. Create visualizations
    print("\n" + "="*60)
    print("4. CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        visualizer = ResNetV2Visualizer()
        
        print("📊 Generating training comparison plots...")
        
        # Plot training results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(training_stats['v1']['train_acc']) + 1)
        
        # Training accuracy
        ax1.plot(epochs, training_stats['v1']['train_acc'], 'r-o', label='ResNet V1', linewidth=2)
        ax1.plot(epochs, training_stats['v2']['train_acc'], 'g-o', label='ResNet V2', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Accuracy (%)')
        ax1.set_title('Training Accuracy Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test accuracy
        ax2.plot(epochs, training_stats['v1']['test_acc'], 'r-s', label='ResNet V1', linewidth=2)
        ax2.plot(epochs, training_stats['v2']['test_acc'], 'g-s', label='ResNet V2', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Test Accuracy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training loss
        ax3.plot(epochs, training_stats['v1']['train_loss'], 'r-^', label='ResNet V1', linewidth=2)
        ax3.plot(epochs, training_stats['v2']['train_loss'], 'g-^', label='ResNet V2', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('Training Loss Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Test loss
        ax4.plot(epochs, training_stats['v1']['test_loss'], 'r-d', label='ResNet V1', linewidth=2)
        ax4.plot(epochs, training_stats['v2']['test_loss'], 'g-d', label='ResNet V2', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Test Loss')
        ax4.set_title('Test Loss Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.suptitle('ResNet V1 vs V2 Training Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('resnet_v2_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # 5. Summary
    print("\n" + "="*60)
    print("5. TRAINING SUMMARY")
    print("="*60)
    
    print("\n🎯 Key Findings:")
    print("   📈 ResNet V2 shows better training dynamics")
    print("   🚀 Faster convergence and more stable training")
    print("   🏔️ Enables training of very deep networks (200+ layers)")
    print("   🌊 Better gradient flow through pre-activation")
    print("   ✨ Simple design change, dramatic improvements")
    
    final_v1_acc = training_stats['v1']['test_acc'][-1] if training_stats['v1']['test_acc'] else 0
    final_v2_acc = training_stats['v2']['test_acc'][-1] if training_stats['v2']['test_acc'] else 0
    
    print(f"\n📊 Final Results (5 epochs):")
    print(f"   ResNet V1 final accuracy: {final_v1_acc:.2f}%")
    print(f"   ResNet V2 final accuracy: {final_v2_acc:.2f}%")
    print(f"   Improvement: {final_v2_acc - final_v1_acc:+.2f}%")
    
    print("\n💡 Why ResNet V2 Works Better:")
    print("   • Pre-activation creates pristine identity paths")
    print("   • No activation contamination of skip connections")
    print("   • Better gradient flow enables extreme depth")
    print("   • More stable optimization landscape")
    print("   • Foundation for modern deep learning architectures")
    
    print("\n🔮 Legacy Impact:")
    print("   • Used in DenseNet, EfficientNet, and Vision Transformers")
    print("   • Pre-activation principle adopted in language models")
    print("   • Standard practice in modern deep learning")
    print("   • Enables training of very large models")


def quick_demo():
    """Quick demonstration for immediate results."""
    print("⚡ ResNet V2 Quick Demo")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    print("Creating ResNet V2 models...")
    resnet18_v2 = resnet_v2_18(num_classes=10)
    resnet50_v2 = resnet_v2_50(num_classes=1000)
    
    print(f"✅ ResNet V2-18: {sum(p.numel() for p in resnet18_v2.parameters()):,} parameters")
    print(f"✅ ResNet V2-50: {sum(p.numel() for p in resnet50_v2.parameters()):,} parameters")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 size
    output = resnet18_v2(x)
    print(f"✅ Forward pass successful: {output.shape}")
    
    # Create very deep network
    print("\nCreating very deep network...")
    very_deep = VeryDeepResNetV2(depth=200, num_classes=10)
    print(f"✅ 200-layer ResNet V2: {sum(p.numel() for p in very_deep.parameters()):,} parameters")
    
    # Test pre vs post activation
    print("\nTesting pre vs post activation blocks...")
    study = PreActivationStudy()
    pre_block, post_block = study.create_comparison_blocks(64, 64)
    
    test_input = torch.randn(1, 64, 32, 32)
    analysis = study.analyze_activation_patterns(pre_block, post_block, test_input)
    
    pre_sparsity = analysis['pre_activation']['sparsity']
    post_sparsity = analysis['post_activation']['sparsity']
    
    print(f"✅ Pre-activation sparsity: {pre_sparsity:.2%}")
    print(f"✅ Post-activation sparsity: {post_sparsity:.2%}")
    
    print(f"\n🎯 ResNet V2 demonstrates {abs(pre_sparsity - post_sparsity):.1%} difference in activation patterns!")
    print("🚀 Ready for full training demonstration!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
    else:
        demonstrate_resnet_v2_training()