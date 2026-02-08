"""
Day 8: ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)

Complete implementation of AlexNet - the neural network that sparked the Deep Learning Revolution.
Includes the original architecture with all key innovations: ReLU, Dropout, LRN, and GPU optimization.

This implementation demonstrates:
1. The exact AlexNet architecture from 2012
2. Modern PyTorch training pipeline
3. ImageNet data handling
4. Feature visualization and analysis
5. Transfer learning capabilities

Author: 30u30 Project
Based on: "ImageNet Classification with Deep Convolutional Neural Networks"
         by Krizhevsky, Sutskever, Hinton (2012) - https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class LocalResponseNorm(nn.Module):
    """
    Local Response Normalization as described in the AlexNet paper.
    
    Implements lateral inhibition - bright neurons suppress their neighbors.
    This was later replaced by Batch Normalization, but was crucial for AlexNet's success.
    """
    
    def __init__(self, size: int = 5, alpha: float = 0.0001, beta: float = 0.75, k: float = 2.0):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local response normalization."""
        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.size()
        
        # Create padded version for neighborhood calculation
        pad = self.size // 2
        x_padded = F.pad(x, (0, 0, 0, 0, pad, pad))
        
        # Calculate local sum of squares
        x_squared = x_padded ** 2
        sum_squared = torch.zeros_like(x)
        
        for i in range(self.size):
            sum_squared += x_squared[:, i:i+channels, :, :]
        
        # Apply normalization
        denominator = (self.k + self.alpha * sum_squared) ** self.beta
        return x / denominator


class AlexNet(nn.Module):
    """
    AlexNet: The neural network that started the Deep Learning Revolution.
    
    Architecture:
    - 5 Convolutional layers with ReLU activations
    - 3 Fully connected layers  
    - Dropout for regularization
    - Local Response Normalization
    - Max pooling for dimension reduction
    
    Key innovations:
    1. ReLU activations (instead of sigmoid/tanh)
    2. Dropout regularization
    3. GPU parallelization
    4. Data augmentation
    5. Large-scale training on ImageNet
    """
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(AlexNet, self).__init__()
        
        # Feature extractor: Convolutional layers
        self.features = nn.Sequential(
            # Conv1: 11x11 conv, 96 filters, stride 4
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 5x5 conv, 256 filters 
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 3x3 conv, 384 filters
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 3x3 conv, 384 filters
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 3x3 conv, 256 filters
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier: Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights using the original scheme
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AlexNet."""
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classify
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize weights as in the original paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Gaussian initialization with std=0.01
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Gaussian initialization for fully connected layers
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def get_feature_maps(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Extract feature maps from a specific convolutional layer.
        Useful for visualization and analysis.
        """
        features = []
        current_x = x
        
        for i, layer in enumerate(self.features):
            current_x = layer(current_x)
            if isinstance(layer, nn.Conv2d):
                features.append(current_x)
                if len(features) - 1 == layer_idx:
                    return current_x
        
        return features[layer_idx] if layer_idx < len(features) else features[-1]


class ImageNetDataset(Dataset):
    """
    ImageNet dataset handler with AlexNet-specific preprocessing.
    
    Includes data augmentation techniques used in the original paper:
    - Random crops and horizontal flips
    - Color jittering
    - Normalization with ImageNet statistics
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.split = split
        
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(256),                    # Resize shorter side
                    transforms.RandomCrop(224),               # Random 224x224 crop
                    transforms.RandomHorizontalFlip(0.5),     # Random horizontal flip
                    transforms.ColorJitter(                   # Color augmentation
                        brightness=0.1,
                        contrast=0.1, 
                        saturation=0.1,
                        hue=0.05
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(                     # ImageNet normalization
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform
            
        # Load dataset
        try:
            self.dataset = datasets.ImageFolder(
                root=os.path.join(root_dir, split),
                transform=self.transform
            )
        except:
            # Fallback: use CIFAR-10 for demo if ImageNet not available
            if split == 'train':
                self.dataset = datasets.CIFAR10(
                    root=root_dir, 
                    train=True, 
                    download=True,
                    transform=self.transform
                )
            else:
                self.dataset = datasets.CIFAR10(
                    root=root_dir, 
                    train=False, 
                    download=True,
                    transform=self.transform
                )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class AlexNetTrainer:
    """
    Complete training pipeline for AlexNet following the original paper.
    
    Features:
    - SGD with momentum and weight decay
    - Learning rate scheduling
    - Training/validation monitoring
    - Model checkpointing
    - GPU acceleration
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer: SGD with momentum (as in original paper)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs: int):
        """Complete training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'alexnet_best.pth')
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')


class FeatureExtractor:
    """
    Extract and analyze features from different layers of AlexNet.
    Useful for understanding what the network learns.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
        # Register hooks to capture intermediate features
        self.features = {}
        self.hooks = []
        
        # Hook each convolutional layer
        conv_layers = [layer for layer in self.model.features if isinstance(layer, nn.Conv2d)]
        for i, layer in enumerate(conv_layers):
            hook = layer.register_forward_hook(self._make_hook(f'conv{i+1}'))
            self.hooks.append(hook)
    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from all convolutional layers."""
        self.features.clear()
        with torch.no_grad():
            _ = self.model(x)
        return self.features.copy()
    
    def get_filter_responses(self, x: torch.Tensor, layer: str = 'conv1') -> torch.Tensor:
        """Get responses of all filters in a specific layer."""
        features = self.extract_features(x)
        return features[layer]
    
    def cleanup(self):
        """Remove hooks to free memory."""
        for hook in self.hooks:
            hook.remove()


class AlexNetAnalyzer:
    """
    Analysis tools for understanding AlexNet's behavior and learned features.
    """
    
    @staticmethod
    def analyze_activation_patterns(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
        """
        Analyze activation patterns across the network.
        
        Returns statistics about:
        - Activation sparsity (how many neurons are zero)
        - Layer-wise activation statistics
        - ReLU effectiveness
        """
        model.eval()
        activation_stats = {
            'layer_sparsity': {},
            'layer_means': {},
            'layer_stds': {},
            'relu_effectiveness': {}
        }
        
        # Register hooks
        hooks = []
        layer_names = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Sparsity: percentage of zero activations
                    sparsity = (output == 0).float().mean().item()
                    activation_stats['layer_sparsity'][name] = sparsity
                    
                    # Statistics
                    activation_stats['layer_means'][name] = output.mean().item()
                    activation_stats['layer_stds'][name] = output.std().item()
                    
                    # ReLU effectiveness (for ReLU layers)
                    if isinstance(module, nn.ReLU):
                        # How much input was positive
                        input_tensor = input[0] if isinstance(input, tuple) else input
                        positive_ratio = (input_tensor > 0).float().mean().item()
                        activation_stats['relu_effectiveness'][name] = positive_ratio
            return hook
        
        # Hook all layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.Linear)):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                layer_names.append(name)
        
        # Run through a few batches
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 5:  # Analyze only a few batches
                    break
                data = data.to(device)
                _ = model(data)
        
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
        
        return activation_stats
    
    @staticmethod
    def compute_network_depth_effectiveness(model: nn.Module) -> Dict:
        """
        Analyze how effectively the network uses its depth.
        """
        # Count parameters and layers
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Layer analysis
        conv_layers = 0
        fc_layers = 0
        conv_params = 0
        fc_params = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers += 1
                conv_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Linear):
                fc_layers += 1
                fc_params += sum(p.numel() for p in module.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'conv_layers': conv_layers,
            'fc_layers': fc_layers,
            'conv_parameters': conv_params,
            'fc_parameters': fc_params,
            'conv_param_ratio': conv_params / total_params,
            'fc_param_ratio': fc_params / total_params
        }


def create_alexnet_model(num_classes: int = 10, pretrained: bool = False) -> AlexNet:
    """
    Factory function to create AlexNet model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (if available)
    
    Returns:
        AlexNet model
    """
    model = AlexNet(num_classes=num_classes)
    
    if pretrained:
        try:
            # Load pretrained weights if available
            checkpoint = torch.load('alexnet_pretrained.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pretrained AlexNet weights")
        except FileNotFoundError:
            print("No pretrained weights found, using random initialization")
    
    return model


def get_data_loaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir: Directory containing the data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ImageNetDataset(data_dir, split='train')
    val_dataset = ImageNetDataset(data_dir, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Demo: Create and analyze AlexNet
    print("🔥 AlexNet Implementation Demo")
    print("=" * 50)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_alexnet_model(num_classes=10)  # Using 10 classes for demo
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Analyze model architecture
    analyzer = AlexNetAnalyzer()
    depth_stats = analyzer.compute_network_depth_effectiveness(model)
    
    print(f"\n📊 Model Analysis:")
    print(f"   Total parameters: {depth_stats['total_parameters']:,}")
    print(f"   Convolutional layers: {depth_stats['conv_layers']}")
    print(f"   Fully connected layers: {depth_stats['fc_layers']}")
    print(f"   Conv parameters: {depth_stats['conv_param_ratio']:.2%}")
    print(f"   FC parameters: {depth_stats['fc_param_ratio']:.2%}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    
    print("\n✅ AlexNet ready! The Deep Learning Revolution begins here.")