"""
Day 9: ResNet - Visualization Suite

Beautiful visualizations showing skip connections, gradient flow, and the solution
to the degradation problem. Demonstrates why ResNet revolutionized deep learning.

Features:
- Architecture visualization with skip connections
- Gradient flow analysis
- Degradation problem demonstration
- Skip connection impact analysis
- Training depth comparison

Author: 30u30 Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


class ResNetVisualizer:
    """
    Comprehensive visualization suite for ResNet analysis.
    Shows the revolutionary impact of skip connections on deep learning.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292', '#FFD93D', '#6BCF7F']
        
    def plot_skip_connection_concept(self, save_path: str = None) -> plt.Figure:
        """
        Visualize the core concept of skip connections and residual learning.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('🔗 The Skip Connection Revolution: From H(x) to F(x) + x', 
                     fontsize=16, fontweight='bold')
        
        # Traditional network (left)
        ax1.set_title('❌ Traditional Deep Network\nLearns H(x) directly', fontsize=14, fontweight='bold')
        
        # Draw traditional network layers
        layer_positions = [(1, 4), (3, 4), (5, 4), (7, 4), (9, 4)]
        layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Output']
        
        for i, (pos, name) in enumerate(zip(layer_positions, layer_names)):
            # Layer box
            rect = plt.Rectangle((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6, 
                                facecolor=self.colors[i % len(self.colors)], 
                                edgecolor='black', linewidth=2, alpha=0.8)
            ax1.add_patch(rect)
            ax1.text(pos[0], pos[1], name, ha='center', va='center', fontweight='bold', fontsize=9)
            
            # Arrow to next layer
            if i < len(layer_positions) - 1:
                ax1.arrow(pos[0]+0.4, pos[1], 1.2, 0, head_width=0.1, head_length=0.1, 
                         fc='darkblue', ec='darkblue', linewidth=2)
        
        # Add input
        ax1.text(0.5, 4, 'Input\nx', ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        # Add problems
        problems_text = """
        ⚠️ PROBLEMS:
        
        🔴 Vanishing gradients
        🔴 Hard to optimize  
        🔴 Degradation problem
        🔴 Identity mapping difficult
        """
        ax1.text(5, 2, problems_text, fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        ax1.set_xlim(0, 10)
        ax1.set_ylim(1, 6)
        ax1.axis('off')
        
        # ResNet (right)
        ax2.set_title('✅ ResNet with Skip Connections\nLearns F(x) = H(x) - x', fontsize=14, fontweight='bold')
        
        # Draw ResNet blocks
        block_positions = [(2, 4), (5, 4), (8, 4)]
        block_names = ['Residual\nBlock 1', 'Residual\nBlock 2', 'Residual\nBlock 3']
        
        for i, (pos, name) in enumerate(zip(block_positions, block_names)):
            # Main path (residual branch)
            main_y = pos[1] + 1
            
            # Sub-layers in residual branch
            sub_layers = [(pos[0]-0.5, main_y), (pos[0], main_y), (pos[0]+0.5, main_y)]
            for j, sub_pos in enumerate(sub_layers):
                rect = plt.Rectangle((sub_pos[0]-0.2, sub_pos[1]-0.15), 0.4, 0.3,
                                    facecolor=self.colors[(i*3+j) % len(self.colors)],
                                    edgecolor='black', alpha=0.8)
                ax2.add_patch(rect)
                
                # Connect sub-layers
                if j < len(sub_layers) - 1:
                    ax2.arrow(sub_pos[0]+0.2, sub_pos[1], 0.3, 0, head_width=0.05, head_length=0.05,
                             fc='blue', ec='blue')
            
            # Skip connection (identity path)
            skip_y = pos[1] - 1
            
            # Draw skip connection arrow
            if i == 0:
                start_x = 1.5
            else:
                start_x = block_positions[i-1][0] + 1
            
            # Curved skip connection
            from matplotlib.patches import FancyArrowPatch
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            
            skip_arrow = FancyArrowPatch((start_x, pos[1]), (pos[0]+0.7, pos[1]),
                                        arrowstyle='->', mutation_scale=20,
                                        connectionstyle="arc3,rad=-0.3",
                                        color='red', linewidth=3, alpha=0.8)
            ax2.add_patch(skip_arrow)
            
            # Addition symbol
            ax2.scatter([pos[0]+0.8], [pos[1]], s=300, c='red', marker='+', linewidth=4, zorder=10)
            
            # Block label
            ax2.text(pos[0], pos[1]-1.5, name, ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Connect blocks
            if i < len(block_positions) - 1:
                ax2.arrow(pos[0]+1, pos[1], 1.5, 0, head_width=0.1, head_length=0.1,
                         fc='darkgreen', ec='darkgreen', linewidth=2)
        
        # Add input
        ax2.text(1, 4, 'Input\nx', ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        # Add benefits
        benefits_text = """
        ✅ BENEFITS:
        
        🟢 Gradient highways
        🟢 Easy optimization
        🟢 No degradation
        🟢 Identity mapping learned
        """
        ax2.text(5, 2, benefits_text, fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Add key insight
        insight_text = """
        💡 KEY INSIGHT:
        Learn the residual F(x) = H(x) - x
        Then H(x) = F(x) + x
        
        If optimal function is identity,
        just set F(x) = 0!
        """
        ax2.text(8.5, 0.5, insight_text, fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
        
        ax2.set_xlim(0.5, 10)
        ax2.set_ylim(0, 6)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_degradation_problem(self, save_path: str = None) -> plt.Figure:
        """
        Visualize the degradation problem that ResNet solves.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('📉 The Degradation Problem: Why Deeper Used to Mean Worse', 
                     fontsize=16, fontweight='bold')
        
        # Simulated data showing degradation problem
        depths = [8, 14, 20, 32, 44, 56, 110]
        
        # Traditional network performance (degrades with depth)
        traditional_train = [92, 89, 87, 84, 82, 78, 75]  # Gets worse
        traditional_test = [88, 85, 83, 80, 78, 74, 71]   # Gets worse
        
        # ResNet performance (improves with depth)
        resnet_train = [91, 92, 93, 94, 95, 95.5, 96]     # Gets better
        resnet_test = [88, 89, 90, 91, 91.5, 92, 92.5]    # Gets better
        
        # Plot traditional networks
        ax1.plot(depths, traditional_train, 'ro-', linewidth=3, markersize=8, 
                label='Training Accuracy', alpha=0.8)
        ax1.plot(depths, traditional_test, 'bo-', linewidth=3, markersize=8, 
                label='Test Accuracy', alpha=0.8)
        ax1.fill_between(depths, traditional_train, alpha=0.3, color='red')
        ax1.fill_between(depths, traditional_test, alpha=0.3, color='blue')
        
        ax1.set_title('❌ Traditional Deep Networks\n(Degradation Problem)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Network Depth (Number of Layers)', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(65, 100)
        
        # Add annotation
        ax1.annotate('🚫 Training accuracy gets WORSE\nwith more layers!\n(Not overfitting - degradation)', 
                    xy=(56, 78), xytext=(40, 85),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # Plot ResNet
        ax2.plot(depths, resnet_train, 'go-', linewidth=3, markersize=8, 
                label='Training Accuracy', alpha=0.8)
        ax2.plot(depths, resnet_test, 'mo-', linewidth=3, markersize=8, 
                label='Test Accuracy', alpha=0.8)
        ax2.fill_between(depths, resnet_train, alpha=0.3, color='green')
        ax2.fill_between(depths, resnet_test, alpha=0.3, color='magenta')
        
        ax2.set_title('✅ ResNet with Skip Connections\n(Problem Solved!)', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Network Depth (Number of Layers)', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(65, 100)
        
        # Add annotation
        ax2.annotate('🎉 Training accuracy IMPROVES\nwith more layers!\n(Deeper is better)', 
                    xy=(110, 96), xytext=(80, 89),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Add explanation
        explanation = """
        🧠 WHY THIS HAPPENS:
        
        Traditional networks:
        • Hard to learn identity mappings
        • Gradients vanish in deep networks  
        • Optimization becomes difficult
        • More parameters ≠ better performance
        
        ResNet:
        • Skip connections create identity paths
        • Gradients flow freely through shortcuts
        • Easy to learn identity when needed
        • More depth = more representational power! ✨
        """
        
        fig.text(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_gradient_flow(self, gradient_analysis: Dict, save_path: str = None) -> plt.Figure:
        """
        Visualize gradient flow through ResNet vs traditional networks.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('🌊 Gradient Flow: Why Skip Connections Create Gradient Highways', 
                     fontsize=16, fontweight='bold')
        
        layers = list(gradient_analysis.keys())[:10]  # Show first 10 layers
        gradients = [gradient_analysis[layer] for layer in layers]
        
        # Simulate traditional network gradients (exponential decay)
        traditional_gradients = [grad * (0.7 ** i) for i, grad in enumerate(gradients)]
        
        # Plot ResNet gradients
        x_pos = np.arange(len(layers))
        bars1 = ax1.bar(x_pos, gradients, alpha=0.8, color='green', label='ResNet (with skip connections)')
        ax1.set_title('🚀 ResNet: Healthy Gradient Flow', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Gradient Magnitude', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot traditional gradients  
        bars2 = ax2.bar(x_pos, traditional_gradients, alpha=0.8, color='red', label='Traditional (no skip connections)')
        ax2.set_title('💀 Traditional Network: Vanishing Gradients', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Layer (from output backward)', fontsize=12)
        ax2.set_ylabel('Gradient Magnitude', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'L{i+1}' for i in range(len(layers))], rotation=45)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            if height > 0.001:  # Only show if not too small
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add explanation
        explanation_text = """
        💡 GRADIENT FLOW ANALYSIS:
        
        🟢 ResNet maintains strong gradients throughout the network
           • Skip connections provide gradient highways
           • Information flows backward efficiently
           • All layers receive strong learning signals
        
        🔴 Traditional networks suffer exponential gradient decay
           • Each layer reduces gradient magnitude
           • Deep layers barely receive learning signals
           • Training becomes ineffective for deep networks
        """
        
        ax2.text(0.02, 0.98, explanation_text, transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_resnet_architecture(self, model_name: str = 'ResNet-50', save_path: str = None) -> plt.Figure:
        """
        Detailed architectural diagram of ResNet variants.
        """
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.suptitle(f'🏗️ {model_name} Architecture: Deep Learning with Skip Connections', 
                     fontsize=16, fontweight='bold')
        
        # Define architecture components
        components = [
            {'name': 'Input\n224×224×3', 'pos': (1, 6), 'size': (1.5, 1), 'color': '#E8F4FD'},
            {'name': 'Conv1\n7×7×64\nstride=2', 'pos': (3, 6), 'size': (2, 1), 'color': '#FFE5E5'},
            {'name': 'MaxPool\n3×3 s=2', 'pos': (6, 6), 'size': (1.5, 0.8), 'color': '#F0E5FF'},
            {'name': 'Layer1\n3×Bottleneck\n64→256', 'pos': (9, 6), 'size': (2.5, 1.2), 'color': '#E5F5E5'},
            {'name': 'Layer2\n4×Bottleneck\n128→512', 'pos': (13, 6), 'size': (2.5, 1.2), 'color': '#E5F5F5'},
            {'name': 'Layer3\n6×Bottleneck\n256→1024', 'pos': (17, 6), 'size': (2.5, 1.2), 'color': '#F5F5E5'},
            {'name': 'Layer4\n3×Bottleneck\n512→2048', 'pos': (21, 6), 'size': (2.5, 1.2), 'color': '#F5E5F5'},
            {'name': 'AvgPool\n7×7', 'pos': (25, 6), 'size': (1.5, 0.8), 'color': '#F0E5FF'},
            {'name': 'FC\n1000', 'pos': (28, 6), 'size': (1.5, 1), 'color': '#E5E5FF'},
        ]
        
        # Draw components
        for comp in components:
            rect = plt.Rectangle(comp['pos'], comp['size'][0], comp['size'][1],
                               facecolor=comp['color'], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2,
                   comp['name'], ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw connections
        connections = [
            ((2.5, 6.5), (3, 6.5)), ((5, 6.5), (6, 6.4)),
            ((7.5, 6.5), (9, 6.6)), ((11.5, 6.6), (13, 6.6)),
            ((15.5, 6.6), (17, 6.6)), ((19.5, 6.6), (21, 6.6)),
            ((23.5, 6.6), (25, 6.4)), ((26.5, 6.5), (28, 6.5))
        ]
        
        for start, end in connections:
            ax.arrow(start[0], start[1], end[0]-start[0]-0.1, end[1]-start[1], 
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
        
        # Draw detailed bottleneck block
        ax.text(10.25, 4, 'Bottleneck Block Detail:', ha='center', fontweight='bold', fontsize=12)
        
        # Bottleneck components
        bottleneck_parts = [
            {'name': '1×1 Conv\nBN + ReLU', 'pos': (8.5, 2.5), 'size': (1.5, 0.8), 'color': '#FFE5E5'},
            {'name': '3×3 Conv\nBN + ReLU', 'pos': (10.5, 2.5), 'size': (1.5, 0.8), 'color': '#FFE5E5'},
            {'name': '1×1 Conv\nBN', 'pos': (12.5, 2.5), 'size': (1.5, 0.8), 'color': '#FFE5E5'},
        ]
        
        for part in bottleneck_parts:
            rect = plt.Rectangle(part['pos'], part['size'][0], part['size'][1],
                               facecolor=part['color'], edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(part['pos'][0] + part['size'][0]/2, part['pos'][1] + part['size'][1]/2,
                   part['name'], ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw bottleneck connections
        bottleneck_connections = [((10, 2.9), (10.5, 2.9)), ((12, 2.9), (12.5, 2.9))]
        for start, end in bottleneck_connections:
            ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
                    head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        
        # Draw skip connection
        from matplotlib.patches import FancyArrowPatch
        skip_arrow = FancyArrowPatch((8.5, 2.9), (14.2, 2.9),
                                    arrowstyle='->', mutation_scale=20,
                                    connectionstyle="arc3,rad=-0.4",
                                    color='red', linewidth=3, alpha=0.8)
        ax.add_patch(skip_arrow)
        
        # Addition symbol
        ax.scatter([14], [2.9], s=400, c='red', marker='+', linewidth=5, zorder=10)
        ax.text(14, 2.3, 'Add', ha='center', fontweight='bold', color='red')
        
        # Skip connection label
        ax.text(11.25, 1.8, '🔗 Skip Connection (Identity Mapping)', ha='center', fontsize=11, 
                fontweight='bold', color='red')
        
        # Add statistics
        stats_text = f"""
        📊 {model_name} STATISTICS:
        
        🔢 Layers: 50 (conv) + FC
        📐 Parameters: ~25.6M
        🎯 Bottleneck blocks: 16
        📊 Skip connections: 16
        🏆 ImageNet Top-5: 92.2%
        ⚡ Breakthrough: 152 layers possible!
        """
        
        ax.text(1, 4.5, stats_text, fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))
        
        # Add key innovations
        innovations_text = """
        🚀 KEY INNOVATIONS:
        
        🔗 Skip connections enable deep training
        🏗️ Bottleneck design reduces parameters
        📈 Batch normalization for stability
        🎯 Residual learning F(x) = H(x) - x
        """
        
        ax.text(25, 4.5, innovations_text, fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        ax.set_xlim(0, 30)
        ax.set_ylim(1, 8)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_skip_connection_analysis(self, skip_analysis: Dict, save_path: str = None) -> plt.Figure:
        """
        Analyze the contribution of skip connections vs residual branches.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔬 Skip Connection Analysis: Identity vs Residual Contributions', 
                     fontsize=16, fontweight='bold')
        
        # Extract data
        layer_names = list(skip_analysis.keys())[:8]  # Show first 8 layers
        identity_norms = [skip_analysis[name]['identity_norm'] for name in layer_names]
        residual_norms = [skip_analysis[name]['residual_norm'] for name in layer_names]
        residual_ratios = [skip_analysis[name]['residual_ratio'] for name in layer_names]
        output_norms = [skip_analysis[name]['output_norm'] for name in layer_names]
        
        # Plot 1: Identity vs Residual magnitudes
        x = np.arange(len(layer_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, identity_norms, width, label='Identity Path', 
                       alpha=0.8, color='blue')
        bars2 = ax1.bar(x + width/2, residual_norms, width, label='Residual Path', 
                       alpha=0.8, color='red')
        
        ax1.set_title('📊 Path Contributions: Identity vs Residual')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('L2 Norm')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'L{i+1}' for i in range(len(layer_names))], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residual/Identity ratio
        ax2.plot(x, residual_ratios, 'go-', linewidth=3, markersize=8, alpha=0.8)
        ax2.fill_between(x, residual_ratios, alpha=0.3, color='green')
        ax2.set_title('⚖️ Residual/Identity Ratio')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Ratio')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'L{i+1}' for i in range(len(layer_names))], rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal contribution')
        ax2.legend()
        
        # Plot 3: Output magnitudes
        ax3.bar(x, output_norms, alpha=0.8, color='purple')
        ax3.set_title('🎯 Total Output Magnitudes')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('L2 Norm')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'L{i+1}' for i in range(len(layer_names))], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Analysis summary
        ax4.axis('off')
        
        avg_ratio = np.mean(residual_ratios)
        max_ratio = np.max(residual_ratios)
        min_ratio = np.min(residual_ratios)
        
        summary_text = f"""
        🔍 SKIP CONNECTION INSIGHTS:
        
        📊 Average residual/identity ratio: {avg_ratio:.3f}
        📈 Max ratio: {max_ratio:.3f}
        📉 Min ratio: {min_ratio:.3f}
        
        💡 INTERPRETATION:
        
        🔵 When ratio < 1:
           • Identity path dominates
           • Layer acts more like identity
           • Good for gradient flow
        
        🔴 When ratio > 1:
           • Residual path dominates  
           • Layer learns new features
           • Active transformation
        
        ⚖️ Balance is key:
           • Some layers focus on identity
           • Others learn new features
           • Network automatically decides! ✨
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_resnet_dashboard(model, skip_analysis=None, gradient_analysis=None, save_dir=None):
    """
    Create comprehensive ResNet visualization dashboard.
    
    Args:
        model: ResNet model
        skip_analysis: Skip connection analysis results
        gradient_analysis: Gradient flow analysis results  
        save_dir: Directory to save plots
    """
    visualizer = ResNetVisualizer()
    
    print("🎨 Creating ResNet Visualization Dashboard...")
    
    # 1. Skip connection concept
    fig1 = visualizer.plot_skip_connection_concept(
        f"{save_dir}/skip_connection_concept.png" if save_dir else None)
    plt.show()
    
    # 2. Degradation problem
    fig2 = visualizer.plot_degradation_problem(
        f"{save_dir}/degradation_problem.png" if save_dir else None)
    plt.show()
    
    # 3. Architecture diagram
    fig3 = visualizer.plot_resnet_architecture('ResNet-50',
        f"{save_dir}/resnet_architecture.png" if save_dir else None)
    plt.show()
    
    # 4. Gradient flow (if available)
    if gradient_analysis:
        fig4 = visualizer.plot_gradient_flow(gradient_analysis,
            f"{save_dir}/gradient_flow.png" if save_dir else None)
        plt.show()
    
    # 5. Skip connection analysis (if available)
    if skip_analysis:
        fig5 = visualizer.plot_skip_connection_analysis(skip_analysis,
            f"{save_dir}/skip_analysis.png" if save_dir else None)
        plt.show()
    
    print("✅ ResNet dashboard complete! Skip connections visualized beautifully.")
    
    return {
        'concept': fig1,
        'degradation': fig2,
        'architecture': fig3,
        'gradient_flow': fig4 if gradient_analysis else None,
        'skip_analysis': fig5 if skip_analysis else None
    }


if __name__ == "__main__":
    # Demo
    print("🔥 ResNet Visualization Demo")
    print("=" * 50)
    
    visualizer = ResNetVisualizer()
    
    # Show core concepts
    fig = visualizer.plot_skip_connection_concept()
    plt.show()
    
    print("✅ Demo complete! Ready to visualize the skip connection revolution.")