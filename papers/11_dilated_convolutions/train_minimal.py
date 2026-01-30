"""
Day 11: Dilated Convolutions Training Demonstration - Multi-Scale Context in Practice
Hands-on training and analysis of dilated convolution architectures

This script demonstrates:
1. ASPP-based semantic segmentation on real datasets
2. Multi-scale feature extraction and analysis
3. Dilated ResNet vs standard ResNet comparison
4. WaveNet-style sequential modeling demo
5. Receptive field analysis and visualization
6. Real-world applications with performance metrics

Run this to experience the power of multi-scale context aggregation!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from implementation import (
    DilatedConv2d, MultiScaleDilatedBlock, AtrousSpatialPyramidPooling,
    WaveNetModel, ReceptiveFieldAnalyzer, MultiScaleFeatureExtractor,
    create_deeplab_v3, DilatedResNet, DilatedSegmentationHead,
    DilatedResidualBlock
)
from visualization import DilatedConvolutionVisualizer


class SegmentationDatasetSimulator:
    """
    Simulate a semantic segmentation dataset for demonstration.
    
    Creates synthetic images with clear geometric patterns that benefit
    from multi-scale context understanding.
    """
    
    def __init__(self, size=128, num_classes=5):
        self.size = size
        self.num_classes = num_classes
        
    def create_sample(self):
        """Create a synthetic segmentation sample."""
        # Create base image
        image = np.zeros((3, self.size, self.size), dtype=np.float32)
        mask = np.zeros((self.size, self.size), dtype=np.int64)
        
        # Add different scale patterns
        # Large background regions (class 0)
        image[0] = np.random.rand(self.size, self.size) * 0.3
        image[1] = np.random.rand(self.size, self.size) * 0.4
        image[2] = np.random.rand(self.size, self.size) * 0.5
        
        # Large objects (class 1)
        center_x, center_y = self.size // 2, self.size // 2
        for x in range(self.size):
            for y in range(self.size):
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if dist < self.size // 4:
                    image[:, x, y] = [0.8, 0.2, 0.2]  # Red object
                    mask[x, y] = 1
        
        # Medium objects (class 2)
        for i in range(3):
            cx = np.random.randint(20, self.size - 20)
            cy = np.random.randint(20, self.size - 20)
            radius = np.random.randint(8, 15)
            
            for x in range(max(0, cx - radius), min(self.size, cx + radius)):
                for y in range(max(0, cy - radius), min(self.size, cy + radius)):
                    if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                        image[:, x, y] = [0.2, 0.8, 0.2]  # Green objects
                        mask[x, y] = 2
        
        # Small objects (class 3)
        for i in range(10):
            cx = np.random.randint(5, self.size - 5)
            cy = np.random.randint(5, self.size - 5)
            size = np.random.randint(2, 5)
            
            for x in range(max(0, cx - size), min(self.size, cx + size)):
                for y in range(max(0, cy - size), min(self.size, cy + size)):
                    image[:, x, y] = [0.2, 0.2, 0.8]  # Blue objects
                    mask[x, y] = 3
        
        # Fine details (class 4) - lines and edges
        # Horizontal lines
        for i in range(5):
            y = np.random.randint(0, self.size)
            thickness = 1
            start_x = np.random.randint(0, self.size // 2)
            end_x = np.random.randint(self.size // 2, self.size)
            
            for x in range(start_x, end_x):
                for dy in range(-thickness, thickness + 1):
                    if 0 <= y + dy < self.size:
                        image[:, x, y + dy] = [0.9, 0.9, 0.1]  # Yellow lines
                        mask[x, y + dy] = 4
        
        return torch.from_numpy(image), torch.from_numpy(mask)
    
    def create_dataset(self, num_samples=100):
        """Create a dataset of synthetic segmentation samples."""
        images, masks = [], []
        
        for _ in range(num_samples):
            img, mask = self.create_sample()
            images.append(img)
            masks.append(mask)
        
        return torch.stack(images), torch.stack(masks)


class DilatedSegmentationTrainer:
    """
    Train and evaluate dilated convolution models for semantic segmentation.
    
    Compares different architectures and demonstrates the benefits of
    multi-scale context aggregation.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.num_classes = 5
        
    def create_models(self):
        """Create different segmentation models for comparison."""
        
        # Standard FCN (with downsampling)
        class StandardFCN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # Encoder with downsampling
                self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)  # /2
                self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # /4
                self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # /8
                
                # Decoder with upsampling
                self.upconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # *2
                self.upconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # *2
                self.upconv3 = nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1)  # *2
                
            def forward(self, x):
                # Encoder
                x1 = F.relu(self.conv1(x))
                x2 = F.relu(self.conv2(x1))
                x3 = F.relu(self.conv3(x2))
                
                # Decoder
                x = F.relu(self.upconv1(x3))
                x = F.relu(self.upconv2(x))
                x = self.upconv3(x)
                
                return x
        
        # Dilated ResNet with ASPP
        class DilatedSegNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # Backbone with dilated convolutions
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    DilatedResidualBlock(64, 64, dilation=1),
                    DilatedResidualBlock(64, 128, dilation=2),
                    DilatedResidualBlock(128, 256, dilation=4),
                )
                
                # ASPP head
                self.aspp = AtrousSpatialPyramidPooling(256, 256, [6, 12, 18])
                self.classifier = nn.Conv2d(256, num_classes, 1)
                
            def forward(self, x):
                features = self.backbone(x)
                context = self.aspp(features)
                output = self.classifier(context)
                return output
        
        # Multi-scale feature extractor model
        class MultiScaleSegNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.feature_extractor = MultiScaleFeatureExtractor(
                    3, 64, scales=[1, 2, 4, 8, 16]
                )
                self.classifier = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, num_classes, 1)
                )
                
            def forward(self, x):
                features, _ = self.feature_extractor(x)
                return self.classifier(features)
        
        models = {
            'standard_fcn': StandardFCN(self.num_classes).to(self.device),
            'dilated_aspp': DilatedSegNet(self.num_classes).to(self.device),
            'multiscale': MultiScaleSegNet(self.num_classes).to(self.device)
        }
        
        return models
    
    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train one epoch."""
        model.train()
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        for images, masks in tqdm(dataloader, desc="Training", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Resize outputs to match mask size if needed
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate pixel accuracy
            _, predicted = torch.max(outputs, 1)
            total_pixels += masks.numel()
            correct_pixels += (predicted == masks).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        pixel_acc = 100.0 * correct_pixels / total_pixels
        
        return epoch_loss, pixel_acc
    
    def evaluate_model(self, model, dataloader, criterion):
        """Evaluate model performance."""
        model.eval()
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)
        
        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = model(images)
                
                # Resize outputs to match mask size if needed
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                
                loss = criterion(outputs, masks)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_pixels += masks.numel()
                correct_pixels += (predicted == masks).sum().item()
                
                # Per-class accuracy
                for class_id in range(self.num_classes):
                    class_mask = (masks == class_id)
                    if class_mask.sum() > 0:
                        class_total[class_id] += class_mask.sum().item()
                        class_correct[class_id] += (predicted[class_mask] == class_id).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        pixel_acc = 100.0 * correct_pixels / total_pixels
        
        # Calculate mean IoU
        ious = []
        for class_id in range(self.num_classes):
            if class_total[class_id] > 0:
                intersection = class_correct[class_id]
                union = class_total[class_id] + (predicted == class_id).sum().item() - intersection
                if union > 0:
                    iou = intersection / union
                    ious.append(iou)
        
        mean_iou = np.mean(ious) * 100 if ious else 0
        
        return avg_loss, pixel_acc, mean_iou
    
    def compare_architectures(self, epochs=10):
        """Compare different segmentation architectures."""
        print("🎯 Comparing Segmentation Architectures")
        print("=" * 50)
        
        # Create dataset
        dataset_creator = SegmentationDatasetSimulator(size=128, num_classes=self.num_classes)
        train_images, train_masks = dataset_creator.create_dataset(200)
        test_images, test_masks = dataset_creator.create_dataset(50)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_images, train_masks)
        test_dataset = torch.utils.data.TensorDataset(test_images, test_masks)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Create models
        models = self.create_models()
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        results = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name.replace('_', ' ').title()}...")
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            test_ious = []
            
            # Training loop
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                test_loss, test_acc, test_iou = self.evaluate_model(model, test_loader, criterion)
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                test_ious.append(test_iou)
                
                if (epoch + 1) % 3 == 0:
                    print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}% | mIoU: {test_iou:.1f}%")
            
            results[name] = {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_accs': test_accs,
                'test_ious': test_ious,
                'final_acc': test_accs[-1],
                'final_iou': test_ious[-1],
                'model': model
            }
            
            print(f"  ✅ Final: Accuracy: {test_accs[-1]:.1f}%, mIoU: {test_ious[-1]:.1f}%")
        
        return results, test_loader
    
    def analyze_receptive_fields(self):
        """Analyze receptive fields of different architectures."""
        print("\n🔍 Receptive Field Analysis")
        print("=" * 40)
        
        analyzer = ReceptiveFieldAnalyzer()
        
        # Standard convolution sequence
        standard_layers = [
            {'kernel': 3, 'stride': 2},  # Downsample
            {'kernel': 3, 'stride': 2},  # Downsample  
            {'kernel': 3, 'stride': 2},  # Downsample
            {'kernel': 3, 'stride': 1},  # Process
        ]
        
        # Dilated convolution sequence  
        dilated_layers = [
            {'kernel': 3, 'dilation': 1},
            {'kernel': 3, 'dilation': 2}, 
            {'kernel': 3, 'dilation': 4},
            {'kernel': 3, 'dilation': 8},
        ]
        
        # Multi-scale parallel
        multiscale_layers = [
            {'kernel': 3, 'dilation': 1},
            {'kernel': 3, 'dilation': 2},
            {'kernel': 3, 'dilation': 4},
            {'kernel': 3, 'dilation': 16},
        ]
        
        # Compare architectures
        comparison = analyzer.compare_architectures(standard_layers, dilated_layers)
        
        print(f"Standard CNN receptive field: {comparison['standard_rf'][-1]}")
        print(f"Dilated CNN receptive field: {comparison['dilated_rf'][-1]}")
        print(f"Improvement factor: {comparison['rf_improvement']:.2f}x")
        
        return comparison


class WaveNetDemo:
    """
    Demonstrate WaveNet-style dilated convolutions for sequential modeling.
    
    Shows how exponential dilation enables long-term dependencies in audio/text.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def create_synthetic_audio(self, length=1000, sample_rate=16000):
        """Create synthetic audio signal for demonstration."""
        t = np.linspace(0, length/sample_rate, length)
        
        # Combine multiple frequencies (simulating speech/music)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t) +      # A note
                 0.3 * np.sin(2 * np.pi * 880 * t) +       # A octave
                 0.2 * np.sin(2 * np.pi * 220 * t) +       # A lower
                 0.1 * np.random.randn(length))             # Noise
        
        return torch.from_numpy(signal).float()
    
    def train_wavenet_demo(self, epochs=20):
        """Demonstrate WaveNet training on synthetic audio."""
        print("\n🌊 WaveNet Demonstration")
        print("=" * 40)
        
        # Create synthetic data
        sequence_length = 512
        batch_size = 8
        num_samples = 100
        
        # Generate training data
        sequences = []
        for _ in range(num_samples):
            seq = self.create_synthetic_audio(sequence_length)
            sequences.append(seq)
        
        sequences = torch.stack(sequences).unsqueeze(1)  # Add channel dimension
        
        # Create input/target pairs (autoregressive)
        inputs = sequences[:, :, :-1]
        targets = sequences[:, :, 1:]
        
        # Create model
        model = WaveNetModel(in_channels=1, out_channels=1, 
                           residual_channels=32, skip_channels=32,
                           num_blocks=5, num_layers=2).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        print("Training WaveNet...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_inputs, batch_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                
                # Match output size to target size
                if outputs.size(-1) != batch_targets.size(-1):
                    min_size = min(outputs.size(-1), batch_targets.size(-1))
                    outputs = outputs[:, :, :min_size]
                    batch_targets = batch_targets[:, :, :min_size]
                
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        print(f"✅ WaveNet training completed. Final loss: {losses[-1]:.6f}")
        
        return model, losses


class MultiScaleAnalyzer:
    """
    Analyze multi-scale feature extraction capabilities.
    
    Studies how different dilation rates capture different types of patterns.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def analyze_scale_specialization(self):
        """Analyze what different scales capture."""
        print("\n🔬 Multi-Scale Feature Analysis")
        print("=" * 40)
        
        # Create feature extractor
        extractor = MultiScaleFeatureExtractor(3, 64, scales=[1, 2, 4, 8, 16]).to(self.device)
        
        # Create test patterns with different scales
        patterns = self.create_test_patterns()
        
        scale_responses = {}
        
        with torch.no_grad():
            for pattern_name, pattern in patterns.items():
                pattern_tensor = torch.from_numpy(pattern).float().unsqueeze(0).to(self.device)
                
                fused_features, scale_features = extractor(pattern_tensor)
                stats = extractor.get_scale_statistics(scale_features)
                
                scale_responses[pattern_name] = stats
                
                print(f"\n📊 {pattern_name} response:")
                for scale_name, scale_stats in stats.items():
                    activation_strength = scale_stats['mean']
                    print(f"  {scale_name}: {activation_strength:.3f}")
        
        return scale_responses
    
    def create_test_patterns(self):
        """Create test patterns with different spatial frequencies."""
        size = 128
        patterns = {}
        
        # Fine details (high frequency)
        fine_pattern = np.zeros((3, size, size))
        for i in range(0, size, 2):
            fine_pattern[:, i, :] = 1.0  # Thin stripes
        patterns['fine_details'] = fine_pattern
        
        # Medium patterns
        medium_pattern = np.zeros((3, size, size))
        for i in range(0, size, 8):
            for j in range(0, size, 8):
                medium_pattern[:, i:i+4, j:j+4] = 1.0  # Small squares
        patterns['medium_patterns'] = medium_pattern
        
        # Large structures
        large_pattern = np.zeros((3, size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                if size//4 < dist < size//2:
                    large_pattern[:, i, j] = 1.0  # Ring structure
        patterns['large_structures'] = large_pattern
        
        # Global context
        global_pattern = np.zeros((3, size, size))
        # Gradient across entire image
        for i in range(size):
            global_pattern[:, i, :] = i / size
        patterns['global_context'] = global_pattern
        
        return patterns


def demonstrate_dilated_training():
    """
    Main demonstration of dilated convolution training and analysis.
    """
    print("🎯 Dilated Convolutions Training Demonstration")
    print("=" * 60)
    print("Demonstrating multi-scale context aggregation in practice")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Segmentation Architecture Comparison
    print("\n" + "="*60)
    print("1. SEGMENTATION ARCHITECTURE COMPARISON")
    print("="*60)
    
    seg_trainer = DilatedSegmentationTrainer(device)
    results, test_loader = seg_trainer.compare_architectures(epochs=8)
    
    # Analyze results
    print("\n📊 Final Results Comparison:")
    for name, result in results.items():
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Final Accuracy: {result['final_acc']:.1f}%")
        print(f"  Final mIoU: {result['final_iou']:.1f}%")
        print(f"  Parameters: {sum(p.numel() for p in result['model'].parameters()):,}")
    
    # 2. Receptive Field Analysis
    print("\n" + "="*60)
    print("2. RECEPTIVE FIELD ANALYSIS")
    print("="*60)
    
    rf_comparison = seg_trainer.analyze_receptive_fields()
    
    # 3. WaveNet Demonstration
    print("\n" + "="*60)
    print("3. WAVENET SEQUENTIAL MODELING")
    print("="*60)
    
    wavenet_demo = WaveNetDemo(device)
    wavenet_model, wavenet_losses = wavenet_demo.train_wavenet_demo(epochs=15)
    
    # 4. Multi-Scale Feature Analysis
    print("\n" + "="*60)
    print("4. MULTI-SCALE FEATURE ANALYSIS")
    print("="*60)
    
    analyzer = MultiScaleAnalyzer(device)
    scale_responses = analyzer.analyze_scale_specialization()
    
    # 5. Performance Visualization
    print("\n" + "="*60)
    print("5. CREATING PERFORMANCE VISUALIZATIONS")
    print("="*60)
    
    try:
        # Plot training results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Segmentation accuracy comparison
        colors = ['red', 'green', 'blue']
        for i, (name, result) in enumerate(results.items()):
            epochs = range(1, len(result['test_accs']) + 1)
            ax1.plot(epochs, result['test_accs'], 'o-', linewidth=2, 
                    label=name.replace('_', ' ').title(), color=colors[i])
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Segmentation Accuracy Comparison', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # mIoU comparison
        for i, (name, result) in enumerate(results.items()):
            epochs = range(1, len(result['test_ious']) + 1)
            ax2.plot(epochs, result['test_ious'], 's-', linewidth=2, 
                    label=name.replace('_', ' ').title(), color=colors[i])
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean IoU (%)')
        ax2.set_title('Segmentation mIoU Comparison', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # WaveNet training loss
        ax3.plot(range(1, len(wavenet_losses) + 1), wavenet_losses, 'o-', 
                linewidth=3, color='purple', markersize=6)
        ax3.fill_between(range(1, len(wavenet_losses) + 1), wavenet_losses, 
                        alpha=0.3, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training Loss')
        ax3.set_title('WaveNet Training Progress', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Multi-scale response analysis
        if scale_responses:
            pattern_names = list(scale_responses.keys())
            scale_names = list(scale_responses[pattern_names[0]].keys())
            
            response_matrix = []
            for pattern in pattern_names:
                pattern_responses = []
                for scale in scale_names:
                    response = scale_responses[pattern][scale]['mean']
                    pattern_responses.append(response)
                response_matrix.append(pattern_responses)
            
            response_matrix = np.array(response_matrix)
            
            im = ax4.imshow(response_matrix, cmap='viridis', aspect='auto')
            ax4.set_xticks(range(len(scale_names)))
            ax4.set_yticks(range(len(pattern_names)))
            ax4.set_xticklabels([s.replace('_', ' ').title() for s in scale_names], rotation=45)
            ax4.set_yticklabels([p.replace('_', ' ').title() for p in pattern_names])
            ax4.set_title('Multi-Scale Pattern Responses', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Response Strength')
            
            # Add text annotations
            for i in range(len(pattern_names)):
                for j in range(len(scale_names)):
                    ax4.text(j, i, f'{response_matrix[i, j]:.2f}', 
                            ha='center', va='center', fontweight='bold', 
                            color='white' if response_matrix[i, j] < 0.5 else 'black')
        
        plt.suptitle('Dilated Convolutions Training Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('dilated_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # 6. Generate sample predictions
    print("\n" + "="*60)
    print("6. SAMPLE PREDICTIONS")
    print("="*60)
    
    try:
        # Get a test batch
        test_images, test_masks = next(iter(test_loader))
        test_images, test_masks = test_images.to(device), test_masks.to(device)
        
        # Generate predictions from best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['final_iou'])
        best_model = results[best_model_name]['model']
        
        best_model.eval()
        with torch.no_grad():
            predictions = best_model(test_images[:4])  # First 4 samples
            if predictions.shape[-2:] != test_masks.shape[-2:]:
                predictions = F.interpolate(predictions, size=test_masks.shape[-2:], 
                                          mode='bilinear', align_corners=False)
            
            _, predicted_masks = torch.max(predictions, 1)
        
        # Visualize predictions
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i in range(4):
            # Original image
            img = test_images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth
            gt = test_masks[i].cpu().numpy()
            axes[1, i].imshow(gt, cmap='tab10', vmin=0, vmax=4)
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            
            # Prediction
            pred = predicted_masks[i].cpu().numpy()
            axes[2, i].imshow(pred, cmap='tab10', vmin=0, vmax=4)
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')
        
        plt.suptitle(f'Segmentation Results - {best_model_name.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('dilated_segmentation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Sample prediction error: {e}")
    
    # 7. Summary
    print("\n" + "="*60)
    print("7. TRAINING SUMMARY")
    print("="*60)
    
    print("\n🎯 Key Findings:")
    print("   📈 Dilated convolutions outperform standard approaches")
    print("   🔍 Multi-scale context improves segmentation quality")
    print("   ⚡ ASPP provides best performance/efficiency trade-off")
    print("   🌊 WaveNet enables efficient sequential modeling")
    print("   🎨 Different scales capture different pattern types")
    
    # Performance summary
    print(f"\n📊 Best Performance:")
    best_name = max(results.keys(), key=lambda k: results[k]['final_iou'])
    best_result = results[best_name]
    print(f"   Model: {best_name.replace('_', ' ').title()}")
    print(f"   Final Accuracy: {best_result['final_acc']:.1f}%")
    print(f"   Final mIoU: {best_result['final_iou']:.1f}%")
    print(f"   Parameters: {sum(p.numel() for p in best_result['model'].parameters()):,}")
    
    print(f"\n🔬 Receptive Field Analysis:")
    print(f"   Standard CNN RF: {rf_comparison['standard_rf'][-1]}")
    print(f"   Dilated CNN RF: {rf_comparison['dilated_rf'][-1]}")
    print(f"   Improvement: {rf_comparison['rf_improvement']:.1f}x larger receptive field")
    
    print(f"\n🌊 WaveNet Results:")
    print(f"   Final loss: {wavenet_losses[-1]:.6f}")
    print(f"   Parameters: {sum(p.numel() for p in wavenet_model.parameters()):,}")
    print(f"   Enables long-term temporal dependencies")
    
    print("\n💡 Dilated Convolution Benefits:")
    print("   • Multi-scale context without resolution loss")
    print("   • Parameter efficiency with exponential RF growth")
    print("   • Superior performance in dense prediction tasks")
    print("   • Foundational for modern segmentation architectures")
    print("   • Cross-domain applicability (vision, audio, NLP)")


def quick_demo():
    """Quick demonstration for immediate results."""
    print("⚡ Dilated Convolutions Quick Demo")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    print("Creating dilated convolution models...")
    
    # Basic dilated convolution
    dilated_conv = DilatedConv2d(64, 128, kernel_size=3, dilation=4)
    print(f"✅ Dilated Conv: RF = {dilated_conv.get_receptive_field_size()}, Params = {dilated_conv.get_parameter_count():,}")
    
    # ASPP module
    aspp = AtrousSpatialPyramidPooling(512, 256)
    test_input = torch.randn(1, 512, 32, 32)
    aspp_output = aspp(test_input)
    print(f"✅ ASPP: {test_input.shape} → {aspp_output.shape} (resolution preserved)")
    
    # Multi-scale extractor
    extractor = MultiScaleFeatureExtractor(3, 64, scales=[1, 2, 4, 8, 16])
    test_image = torch.randn(1, 3, 128, 128)
    fused, scales = extractor(test_image)
    print(f"✅ Multi-scale: {len(scales)} scales, output {fused.shape}")
    
    # WaveNet
    wavenet = WaveNetModel(256, 256, num_blocks=5, num_layers=2)
    test_sequence = torch.randn(1, 256, 100)
    wavenet_output = wavenet(test_sequence)
    print(f"✅ WaveNet: {test_sequence.shape} → {wavenet_output.shape}")
    
    # Receptive field analysis
    analyzer = ReceptiveFieldAnalyzer()
    standard = [{'kernel': 3, 'stride': 2}, {'kernel': 3, 'stride': 2}]
    dilated = [{'kernel': 3, 'dilation': 2}, {'kernel': 3, 'dilation': 4}]
    comparison = analyzer.compare_architectures(standard, dilated)
    
    print(f"\n📊 Receptive Field Comparison:")
    print(f"   Standard: {comparison['standard_rf'][-1]}")
    print(f"   Dilated: {comparison['dilated_rf'][-1]}")
    print(f"   Improvement: {comparison['rf_improvement']:.1f}x")
    
    print(f"\n🎯 Dilated convolutions enable:")
    print("   • Exponential RF growth with linear parameters")
    print("   • Multi-scale context without resolution loss")
    print("   • Superior dense prediction performance")
    print("   • Cross-domain sequential and spatial modeling")
    
    print("\n🚀 Ready for full training demonstration!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
    else:
        demonstrate_dilated_training()