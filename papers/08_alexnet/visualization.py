"""
Day 8: AlexNet - Visualization Suite

Beautiful visualizations for understanding AlexNet and the Deep Learning Revolution.
Shows feature maps, training dynamics, architectural insights, and learned representations.

Features:
- Architecture visualization
- Feature map analysis
- Training curve plotting  
- Filter visualization
- Activation pattern analysis
- Comparison with other approaches

Author: 30u30 Project
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


class AlexNetVisualizer:
    """
    Comprehensive visualization suite for AlexNet analysis.
    Creates publication-quality plots and interactive visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292', '#FFD93D', '#6BCF7F']
        
    def plot_architecture(self, model: nn.Module, save_path: str = None) -> plt.Figure:
        """
        Visualize the AlexNet architecture with all layers and connections.
        Shows the revolutionary structure that started the deep learning boom.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.suptitle('🔥 AlexNet Architecture: The Deep Learning Revolution Begins', 
                     fontsize=16, fontweight='bold')
        
        # Layer specifications
        layers = [
            {'name': 'Input\n224×224×3', 'type': 'input', 'pos': (1, 5), 'size': (1.5, 2), 'color': '#E8F4FD'},
            {'name': 'Conv1\n11×11×96\nstride=4', 'type': 'conv', 'pos': (3, 5), 'size': (2, 1.8), 'color': '#FFE5E5'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (3, 3), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            {'name': 'LRN', 'type': 'norm', 'pos': (3, 2), 'size': (1.5, 0.6), 'color': '#FFF5E5'},
            {'name': 'MaxPool\n3×3 s=2', 'type': 'pool', 'pos': (3, 1), 'size': (1.5, 0.8), 'color': '#F0E5FF'},
            
            {'name': 'Conv2\n5×5×256', 'type': 'conv', 'pos': (6, 5), 'size': (2, 1.6), 'color': '#FFE5E5'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (6, 3.2), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            {'name': 'LRN', 'type': 'norm', 'pos': (6, 2.2), 'size': (1.5, 0.6), 'color': '#FFF5E5'},
            {'name': 'MaxPool\n3×3 s=2', 'type': 'pool', 'pos': (6, 1.2), 'size': (1.5, 0.8), 'color': '#F0E5FF'},
            
            {'name': 'Conv3\n3×3×384', 'type': 'conv', 'pos': (9, 5), 'size': (2, 1.4), 'color': '#FFE5E5'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (9, 3.4), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            
            {'name': 'Conv4\n3×3×384', 'type': 'conv', 'pos': (12, 5), 'size': (2, 1.4), 'color': '#FFE5E5'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (12, 3.4), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            
            {'name': 'Conv5\n3×3×256', 'type': 'conv', 'pos': (15, 5), 'size': (2, 1.2), 'color': '#FFE5E5'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (15, 3.6), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            {'name': 'MaxPool\n3×3 s=2', 'type': 'pool', 'pos': (15, 2.6), 'size': (1.5, 0.8), 'color': '#F0E5FF'},
            
            {'name': 'FC1\n4096\n+Dropout', 'type': 'fc', 'pos': (19, 5.5), 'size': (2.5, 2), 'color': '#E5E5FF'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (19, 3.2), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            
            {'name': 'FC2\n4096\n+Dropout', 'type': 'fc', 'pos': (23, 5.5), 'size': (2.5, 2), 'color': '#E5E5FF'},
            {'name': 'ReLU', 'type': 'activation', 'pos': (23, 3.2), 'size': (1.5, 0.8), 'color': '#E5F5E5'},
            
            {'name': 'Output\n1000 classes', 'type': 'output', 'pos': (27, 5), 'size': (2.5, 1.5), 'color': '#FFE5F5'},
        ]
        
        # Draw layers
        for layer in layers:
            # Create fancy box
            box = FancyBboxPatch(
                layer['pos'], layer['size'][0], layer['size'][1],
                boxstyle="round,pad=0.1",
                facecolor=layer['color'],
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(box)
            
            # Add text
            ax.text(layer['pos'][0] + layer['size'][0]/2, 
                   layer['pos'][1] + layer['size'][1]/2,
                   layer['name'], 
                   ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Draw arrows (connections)
        arrow_props = dict(arrowstyle='->', lw=2, color='darkblue')
        connections = [
            ((2.5, 6), (3, 6)),      # Input to Conv1
            ((5, 6), (6, 6)),        # Conv1 to Conv2  
            ((8, 6), (9, 6)),        # Conv2 to Conv3
            ((11, 6), (12, 6)),      # Conv3 to Conv4
            ((14, 6), (15, 6)),      # Conv4 to Conv5
            ((17, 6), (19, 6.5)),    # Conv5 to FC1
            ((21.5, 6.5), (23, 6.5)), # FC1 to FC2
            ((25.5, 6), (27, 6)),    # FC2 to Output
        ]
        
        for start, end in connections:
            ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
        
        # Add innovations callouts
        innovations = [
            {'pos': (3, 0.2), 'text': '🚀 ReLU\nActivation', 'color': 'lightgreen'},
            {'pos': (6, 0.2), 'text': '🧠 Local Response\nNormalization', 'color': 'lightyellow'},
            {'pos': (19, 1.5), 'text': '🎯 Dropout\nRegularization', 'color': 'lightcoral'},
            {'pos': (15, 8), 'text': '💾 GPU\nParallelization', 'color': 'lightblue'},
        ]
        
        for innovation in innovations:
            ax.text(innovation['pos'][0], innovation['pos'][1], innovation['text'],
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=innovation['color'], alpha=0.8))
        
        # Add title sections
        ax.text(7, 8.5, '📊 Feature Extraction (Convolutional Layers)', 
               fontsize=14, fontweight='bold', ha='center')
        ax.text(22, 8.5, '🎯 Classification (Fully Connected)', 
               fontsize=14, fontweight='bold', ha='center')
        
        # Statistics box
        stats_text = """
        💎 AlexNet Statistics:
        
        📐 Parameters: ~62M
        🔢 Layers: 8 (5 Conv + 3 FC)
        📊 ImageNet Top-5: 84.7%
        🏆 2012 Winner by 10.8%
        ⚡ GPU: 2x GTX 580
        📚 Training: 1.2M images
        """
        
        ax.text(0.5, 4, stats_text, fontsize=11, 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9),
               verticalalignment='center')
        
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 9)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_feature_maps(self, model: nn.Module, input_image: torch.Tensor, 
                         layer_name: str = 'conv1', save_path: str = None) -> plt.Figure:
        """
        Visualize feature maps from a specific layer.
        Shows what the network has learned to detect.
        """
        model.eval()
        
        # Extract features
        features = {}
        def hook_fn(module, input, output):
            features['target'] = output
        
        # Register hook on target layer
        target_layer = None
        for name, module in model.named_modules():
            if layer_name in name and isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            print(f"Layer {layer_name} not found!")
            return
        
        hook = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_image.unsqueeze(0))
        
        # Remove hook
        hook.remove()
        
        # Get feature maps
        feature_maps = features['target'].squeeze(0)  # Remove batch dimension
        num_filters = min(feature_maps.shape[0], 64)  # Show max 64 filters
        
        # Create visualization
        rows = 8
        cols = 8
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        fig.suptitle(f'🔍 Feature Maps from {layer_name.upper()}: What AlexNet Sees', 
                     fontsize=16, fontweight='bold')
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                ax = axes[i, j]
                
                if idx < num_filters:
                    # Show feature map
                    fmap = feature_maps[idx].cpu().numpy()
                    im = ax.imshow(fmap, cmap='viridis', interpolation='bilinear')
                    ax.set_title(f'Filter {idx}', fontsize=8)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    # Empty subplot
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_training_curves(self, history: Dict, save_path: str = None) -> plt.Figure:
        """
        Plot training and validation curves showing AlexNet's learning progress.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 AlexNet Training Dynamics: The Deep Learning Revolution in Action', 
                     fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=3, label='Training Loss', alpha=0.8)
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=3, label='Validation Loss', alpha=0.8)
        ax1.fill_between(epochs, history['train_loss'], alpha=0.3, color='blue')
        ax1.fill_between(epochs, history['val_loss'], alpha=0.3, color='red')
        ax1.set_title('📉 Loss Curves: Learning to See')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'g-', linewidth=3, label='Training Accuracy', alpha=0.8)
        ax2.plot(epochs, history['val_acc'], 'orange', linewidth=3, label='Validation Accuracy', alpha=0.8)
        ax2.fill_between(epochs, history['train_acc'], alpha=0.3, color='green')
        ax2.fill_between(epochs, history['val_acc'], alpha=0.3, color='orange')
        ax2.set_title('📊 Accuracy Progress: Getting Smarter')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate schedule
        if 'lr' in history:
            ax3.semilogy(epochs, history['lr'], 'purple', linewidth=3, marker='o', markersize=4)
            ax3.set_title('⚡ Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate (log scale)')
            ax3.grid(True, alpha=0.3)
        
        # Overfitting analysis
        if len(history['train_acc']) > 1 and len(history['val_acc']) > 1:
            # Calculate overfitting gap
            overfitting_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
            ax4.plot(epochs, overfitting_gap, 'red', linewidth=3, alpha=0.8)
            ax4.fill_between(epochs, overfitting_gap, alpha=0.3, color='red')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('🎯 Generalization Gap')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Train Acc - Val Acc (%)')
            ax4.grid(True, alpha=0.3)
            
            # Add interpretation
            final_gap = overfitting_gap[-1]
            if final_gap > 5:
                ax4.text(0.02, 0.98, '⚠️ Overfitting Detected', transform=ax4.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            else:
                ax4.text(0.02, 0.98, '✅ Good Generalization', transform=ax4.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_filter_visualization(self, model: nn.Module, layer_name: str = 'features.0', 
                                 save_path: str = None) -> plt.Figure:
        """
        Visualize the actual learned filters (weights) in a convolutional layer.
        Shows what patterns the network has learned to detect.
        """
        # Get the target layer
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            print(f"Layer {layer_name} not found!")
            return
        
        # Get filters
        filters = target_layer.weight.data
        num_filters = min(filters.shape[0], 64)  # Show max 64 filters
        
        # Normalize filters for visualization
        filters_norm = filters.clone()
        for i in range(filters_norm.shape[0]):
            f = filters_norm[i]
            f = f - f.min()
            f = f / (f.max() + 1e-8)
            filters_norm[i] = f
        
        # Create visualization
        rows = 8
        cols = 8
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        fig.suptitle(f'🎨 Learned Filters in {layer_name}: What AlexNet Looks For', 
                     fontsize=16, fontweight='bold')
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                ax = axes[i, j]
                
                if idx < num_filters:
                    # Show filter
                    filter_img = filters_norm[idx]
                    
                    if filter_img.shape[0] == 3:  # RGB filter
                        # Convert to displayable RGB
                        filter_display = filter_img.permute(1, 2, 0).cpu().numpy()
                        ax.imshow(filter_display)
                    else:  # Single channel or multiple channels
                        # Show first channel
                        filter_display = filter_img[0].cpu().numpy()
                        ax.imshow(filter_display, cmap='viridis')
                    
                    ax.set_title(f'Filter {idx}', fontsize=8)
                else:
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add insights
        insight_text = f"""
        🔍 Filter Analysis:
        
        📊 Layer: {layer_name}
        🔢 Number of filters: {filters.shape[0]}
        📐 Filter size: {filters.shape[2]}×{filters.shape[3]}
        
        💡 What to look for:
        • Edge detectors (lines, curves)
        • Texture patterns
        • Color combinations
        • Oriented features
        
        Early layers learn simple features,
        deeper layers learn complex patterns!
        """
        
        fig.text(0.02, 0.5, insight_text, fontsize=11, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
                verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_activation_analysis(self, activation_stats: Dict, save_path: str = None) -> plt.Figure:
        """
        Analyze and visualize activation patterns throughout the network.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧠 AlexNet Activation Analysis: How Neurons Respond', 
                     fontsize=16, fontweight='bold')
        
        # Sparsity analysis
        if 'layer_sparsity' in activation_stats:
            layers = list(activation_stats['layer_sparsity'].keys())
            sparsity = list(activation_stats['layer_sparsity'].values())
            
            ax1.bar(range(len(layers)), sparsity, color=self.colors[:len(layers)], alpha=0.7)
            ax1.set_title('🎯 Activation Sparsity (ReLU Effect)')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Sparsity (% of zeros)')
            ax1.set_xticks(range(len(layers)))
            ax1.set_xticklabels(layers, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add insight
            avg_sparsity = np.mean(sparsity)
            ax1.text(0.02, 0.98, f'💡 Average Sparsity: {avg_sparsity:.1%}\nReLU creates selective neurons!', 
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Activation statistics
        if 'layer_means' in activation_stats and 'layer_stds' in activation_stats:
            layers = list(activation_stats['layer_means'].keys())
            means = list(activation_stats['layer_means'].values())
            stds = list(activation_stats['layer_stds'].values())
            
            x_pos = np.arange(len(layers))
            ax2.bar(x_pos - 0.2, means, 0.4, label='Mean', alpha=0.7, color='blue')
            ax2.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', alpha=0.7, color='red')
            ax2.set_title('📊 Activation Statistics')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Value')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(layers, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # ReLU effectiveness
        if 'relu_effectiveness' in activation_stats:
            layers = list(activation_stats['relu_effectiveness'].keys())
            effectiveness = list(activation_stats['relu_effectiveness'].values())
            
            ax3.plot(range(len(layers)), effectiveness, 'o-', linewidth=3, markersize=8, color='green')
            ax3.fill_between(range(len(layers)), effectiveness, alpha=0.3, color='green')
            ax3.set_title('⚡ ReLU Effectiveness')
            ax3.set_xlabel('ReLU Layer')
            ax3.set_ylabel('Fraction of Positive Inputs')
            ax3.set_xticks(range(len(layers)))
            ax3.set_xticklabels(layers, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Add interpretation
            avg_effectiveness = np.mean(effectiveness)
            if avg_effectiveness > 0.5:
                interpretation = f"✅ ReLUs are effective!\n{avg_effectiveness:.1%} inputs are positive"
                color = 'lightgreen'
            else:
                interpretation = f"⚠️ Many neurons are silent\n{avg_effectiveness:.1%} inputs are positive"
                color = 'lightyellow'
            
            ax3.text(0.02, 0.98, interpretation, transform=ax3.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
        
        # Layer comparison
        ax4.axis('off')
        summary_text = """
        🧠 ACTIVATION INSIGHTS:
        
        🎯 Sparsity Pattern:
        • ReLU creates sparse activations
        • Higher sparsity = more selectivity
        • Good sparsity: 50-80%
        
        ⚡ ReLU Benefits:
        • No vanishing gradients
        • Computational efficiency  
        • Biological plausibility
        • Sparse representations
        
        📊 What We Learn:
        • Early layers: broad activation
        • Later layers: sparse, selective
        • Healthy networks show ~60% sparsity
        
        💡 AlexNet Innovation:
        ReLU was revolutionary! Before AlexNet,
        everyone used sigmoid/tanh which 
        suffered from vanishing gradients.
        ReLU enabled training deep networks! 🚀
        """
        
        ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_revolution_timeline(self, save_path: str = None) -> plt.Figure:
        """
        Create a timeline showing AlexNet's impact on the Deep Learning Revolution.
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('🌟 The Deep Learning Revolution: AlexNet\'s Lasting Impact', 
                     fontsize=16, fontweight='bold')
        
        # Timeline events
        events = [
            {'year': 2012, 'event': 'AlexNet\nWins ImageNet', 'impact': 'Starting the\nRevolution', 'color': '#FF6B6B'},
            {'year': 2013, 'event': 'CNN Adoption\nExplodes', 'impact': 'Computer Vision\nRenaissance', 'color': '#4ECDC4'},
            {'year': 2014, 'event': 'VGG, GoogleNet\nEmerge', 'impact': 'Deeper Networks\nProve Superior', 'color': '#45B7D1'},
            {'year': 2015, 'event': 'ResNet\nRevolution', 'impact': 'Training Very\nDeep Networks', 'color': '#FFA07A'},
            {'year': 2016, 'event': 'Transfer Learning\nMainstream', 'impact': 'Pre-trained Models\nEverywhere', 'color': '#98D8C8'},
            {'year': 2017, 'event': 'Attention/\nTransformers', 'impact': 'Beyond CNNs\nfor Vision', 'color': '#F06292'},
            {'year': 2020, 'event': 'Vision\nTransformers', 'impact': 'Challenging CNN\nSupremacy', 'color': '#FFD93D'},
            {'year': 2023, 'event': 'Multimodal\nAI', 'impact': 'Vision + Language\nIntegration', 'color': '#6BCF7F'},
        ]
        
        # Plot timeline
        years = [event['year'] for event in events]
        y_pos = 5
        
        # Draw timeline line
        ax.plot(years, [y_pos] * len(years), 'k-', linewidth=3, alpha=0.3)
        
        # Add events
        for i, event in enumerate(events):
            # Event marker
            ax.scatter([event['year']], [y_pos], s=300, c=event['color'], 
                      zorder=10, edgecolors='black', linewidth=2)
            
            # Event text (alternating above/below)
            if i % 2 == 0:
                y_text = y_pos + 1.5
                va = 'bottom'
                arrow_y = y_pos + 0.3
            else:
                y_text = y_pos - 1.5
                va = 'top'
                arrow_y = y_pos - 0.3
            
            ax.text(event['year'], y_text, event['event'], ha='center', va=va,
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=event['color'], alpha=0.8))
            
            # Impact text
            ax.text(event['year'], y_text + (0.8 if i % 2 == 0 else -0.8), 
                   event['impact'], ha='center', va=va,
                   fontsize=9, style='italic',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Arrow
            ax.annotate('', xy=(event['year'], y_pos), xytext=(event['year'], arrow_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        # Add impact metrics
        metrics_text = """
        📊 ALEXNET'S MEASURABLE IMPACT:
        
        🎯 ImageNet Performance:
        • 2012: 84.7% Top-5 Accuracy
        • Previous best: 73.8%
        • Improvement: +10.9% (HUGE!)
        
        💰 Industry Impact:
        • GPU sales explosion
        • Deep learning startup boom
        • AI research funding 10x growth
        
        📚 Academic Impact:
        • 80,000+ citations (as of 2024)
        • Spawned thousands of papers
        • New research conferences
        
        🌍 Societal Impact:
        • Self-driving cars
        • Medical image analysis
        • Content moderation
        • Augmented reality
        """
        
        ax.text(2010.5, 2, metrics_text, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))
        
        # Add quote
        quote_text = """
        💬 "It was a big moment. Everyone could see 
        that this was the future of computer vision."
        
        - Fei-Fei Li, ImageNet creator
        """
        
        ax.text(2020, 8, quote_text, fontsize=12, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        ax.set_xlim(2010, 2025)
        ax.set_ylim(0, 10)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_title('From AlexNet to Modern AI: A Decade of Transformation')
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_alexnet_dashboard(model, history=None, save_dir=None):
    """
    Create a comprehensive dashboard for AlexNet analysis.
    
    Args:
        model: AlexNet model
        history: Training history (optional)
        save_dir: Directory to save plots (optional)
    """
    visualizer = AlexNetVisualizer()
    
    print("🎨 Creating AlexNet Visualization Dashboard...")
    
    # 1. Architecture diagram
    fig1 = visualizer.plot_architecture(model, 
                                       f"{save_dir}/alexnet_architecture.png" if save_dir else None)
    plt.show()
    
    # 2. Filter visualization
    fig2 = visualizer.plot_filter_visualization(model, 'features.0',
                                                f"{save_dir}/alexnet_filters.png" if save_dir else None)
    plt.show()
    
    # 3. Training curves (if history available)
    if history:
        fig3 = visualizer.plot_training_curves(history,
                                              f"{save_dir}/alexnet_training.png" if save_dir else None)
        plt.show()
    
    # 4. Deep Learning Revolution timeline
    fig4 = visualizer.create_revolution_timeline(
        f"{save_dir}/dl_revolution.png" if save_dir else None)
    plt.show()
    
    print("✅ AlexNet Dashboard complete! The revolution visualized.")
    
    return {
        'architecture': fig1,
        'filters': fig2,
        'training': fig3 if history else None,
        'timeline': fig4
    }


if __name__ == "__main__":
    # Demo visualization
    from implementation import create_alexnet_model
    
    print("🔥 AlexNet Visualization Demo")
    print("=" * 50)
    
    # Create model
    model = create_alexnet_model(num_classes=10)
    
    # Create visualizations
    visualizer = AlexNetVisualizer()
    
    # Show architecture
    fig = visualizer.plot_architecture(model)
    plt.show()
    
    print("✅ Demo complete! Ready to visualize the Deep Learning Revolution.")