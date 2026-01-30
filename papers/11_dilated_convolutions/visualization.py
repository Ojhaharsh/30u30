"""
Day 11: Dilated Convolutions Visualization - Multi-Scale Context in Action
Interactive visualizations showing how dilated convolutions capture multi-scale context

This module creates comprehensive visualizations demonstrating:
1. Dilated convolution receptive field patterns and growth
2. Multi-scale context aggregation without resolution loss
3. ASPP (Atrous Spatial Pyramid Pooling) parallel processing
4. WaveNet exponential receptive field expansion
5. Comparison with traditional pooling-based approaches
6. Real-world applications in segmentation and audio processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.patches import ConnectionPatch, Ellipse
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional
from implementation import (
    DilatedConv2d, MultiScaleDilatedBlock, AtrousSpatialPyramidPooling,
    WaveNetModel, ReceptiveFieldAnalyzer, MultiScaleFeatureExtractor,
    create_deeplab_v3, DilatedResNet
)

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DilatedConvolutionVisualizer:
    """
    Comprehensive visualization suite for dilated convolution concepts.
    
    Demonstrates multi-scale context aggregation, receptive field analysis,
    and the advantages of dilated convolutions over traditional approaches.
    """
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'dilation1': '#FF6B6B',    # Light red
            'dilation2': '#4ECDC4',    # Turquoise  
            'dilation4': '#45B7D1',    # Sky blue
            'dilation8': '#96CEB4',    # Mint green
            'dilation16': '#FECA57',   # Golden yellow
            'receptive': '#9B59B6',    # Purple
            'context': '#E74C3C',      # Red
            'detail': '#3498DB',       # Blue
            'fusion': '#2ECC71',       # Green
            'traditional': '#95A5A6',  # Gray
            'dilated': '#E67E22'       # Orange
        }
    
    def plot_dilated_convolution_patterns(self):
        """
        Visualize how dilated convolutions sample input with different dilation rates.
        
        Shows the spatial pattern of kernel sampling for various dilation rates.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Different dilation rates to visualize
        dilations = [1, 2, 3, 4, 6, 8]
        kernel_size = 3
        
        for idx, dilation in enumerate(dilations):
            ax = axes[idx]
            
            # Create grid to show sampling pattern
            grid_size = 15
            grid = np.zeros((grid_size, grid_size))
            
            # Calculate effective kernel positions
            center = grid_size // 2
            kernel_positions = []
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    # Position with dilation
                    pos_i = center + (i - kernel_size//2) * dilation
                    pos_j = center + (j - kernel_size//2) * dilation
                    
                    if 0 <= pos_i < grid_size and 0 <= pos_j < grid_size:
                        grid[pos_i, pos_j] = 1
                        kernel_positions.append((pos_i, pos_j))
            
            # Create custom colormap
            colors = ['white', self.colors[f'dilation{min(dilation, 8)}']]
            n_bins = 2
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            
            # Plot the sampling pattern
            im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
            
            # Add grid lines
            for i in range(grid_size + 1):
                ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
                ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
            
            # Highlight center point
            ax.scatter([center], [center], color='red', s=100, marker='*', zorder=5)
            
            # Calculate effective receptive field size
            effective_rf = kernel_size + (kernel_size - 1) * (dilation - 1)
            
            ax.set_title(f'Dilation Rate: {dilation}\nEffective RF: {effective_rf}×{effective_rf}', 
                        fontweight='bold', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add text annotations
            ax.text(0.02, 0.98, f'Kernel: {kernel_size}×{kernel_size}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.text(0.02, 0.85, f'Samples: {len(kernel_positions)}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        plt.suptitle('Dilated Convolution Sampling Patterns\n(Red star = center, Blue = sampled positions)', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig('dilated_convolution_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🔍 Dilated Convolution Patterns:")
        print("   Larger dilation → Larger receptive field")
        print("   Same parameters → Different spatial coverage")
        print("   Sparse sampling → Efficient context capture")
    
    def plot_receptive_field_growth(self):
        """
        Visualize how receptive fields grow with different dilation strategies.
        
        Compares exponential vs linear dilation progressions.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Different dilation strategies
        depths = np.arange(1, 11)
        
        # Exponential dilation (powers of 2)
        exp_dilations = [2**i for i in range(len(depths))]
        exp_rf = self._calculate_cumulative_rf(depths, exp_dilations)
        
        # Linear dilation  
        lin_dilations = list(range(1, len(depths)+1))
        lin_rf = self._calculate_cumulative_rf(depths, lin_dilations)
        
        # Standard convolution (no dilation)
        std_dilations = [1] * len(depths)
        std_rf = self._calculate_cumulative_rf(depths, std_dilations)
        
        # Mixed dilation (DeepLab style)
        mixed_dilations = [1, 2, 4, 8, 16, 6, 12, 18, 24, 32]
        mixed_rf = self._calculate_cumulative_rf(depths, mixed_dilations)
        
        # Plot receptive field growth
        ax1.plot(depths, std_rf, 'o-', linewidth=3, label='Standard (d=1)', 
                color=self.colors['traditional'])
        ax1.plot(depths, lin_rf, 's-', linewidth=3, label='Linear dilation', 
                color=self.colors['detail'])
        ax1.plot(depths, exp_rf, '^-', linewidth=3, label='Exponential dilation', 
                color=self.colors['context'])
        ax1.plot(depths, mixed_rf, 'd-', linewidth=3, label='Mixed dilation', 
                color=self.colors['fusion'])
        
        ax1.set_xlabel('Network Depth (layers)')
        ax1.set_ylabel('Receptive Field Size')
        ax1.set_title('Receptive Field Growth Strategies', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot parameter efficiency (RF per parameter)
        params = depths * 9  # Assuming 3x3 kernels
        
        ax2.plot(params, std_rf, 'o-', linewidth=3, label='Standard', 
                color=self.colors['traditional'])
        ax2.plot(params, lin_rf, 's-', linewidth=3, label='Linear dilation', 
                color=self.colors['detail'])
        ax2.plot(params, exp_rf, '^-', linewidth=3, label='Exponential dilation', 
                color=self.colors['context'])
        ax2.plot(params, mixed_rf, 'd-', linewidth=3, label='Mixed dilation', 
                color=self.colors['fusion'])
        
        ax2.set_xlabel('Parameters (approximate)')
        ax2.set_ylabel('Receptive Field Size')
        ax2.set_title('Parameter Efficiency', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Dilation rate visualization
        ax3.bar(range(len(exp_dilations)), exp_dilations, alpha=0.7, 
               label='Exponential', color=self.colors['context'])
        ax3.bar(range(len(lin_dilations)), lin_dilations, alpha=0.7, 
               label='Linear', color=self.colors['detail'], width=0.6)
        
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Dilation Rate')
        ax3.set_title('Dilation Rate Progressions', fontweight='bold')
        ax3.legend()
        ax3.set_yscale('log')
        
        # Efficiency comparison table
        ax4.axis('off')
        efficiency_data = [
            ['Strategy', 'Final RF', 'Parameters', 'Efficiency'],
            ['Standard', f'{std_rf[-1]}', f'{params[-1]}', f'{std_rf[-1]/params[-1]:.2f}'],
            ['Linear', f'{lin_rf[-1]}', f'{params[-1]}', f'{lin_rf[-1]/params[-1]:.2f}'],
            ['Exponential', f'{exp_rf[-1]}', f'{params[-1]}', f'{exp_rf[-1]/params[-1]:.2f}'],
            ['Mixed', f'{mixed_rf[-1]}', f'{params[-1]}', f'{mixed_rf[-1]/params[-1]:.2f}']
        ]
        
        table = ax4.table(cellText=efficiency_data[1:], colLabels=efficiency_data[0],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code the table
        for i in range(1, len(efficiency_data)):
            for j in range(len(efficiency_data[0])):
                if j == 0:  # Strategy names
                    colors = [self.colors['traditional'], self.colors['detail'], 
                             self.colors['context'], self.colors['fusion']]
                    table[(i, j)].set_facecolor(colors[i-1])
                    table[(i, j)].set_alpha(0.7)
        
        ax4.set_title('Efficiency Comparison', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('receptive_field_growth.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Receptive Field Growth Analysis:")
        print("   Exponential dilation: Maximum RF growth")
        print("   Linear dilation: Balanced growth") 
        print("   Mixed dilation: Practical compromise")
        print("   Same parameters → Different context capture!")
    
    def _calculate_cumulative_rf(self, depths, dilations):
        """Calculate cumulative receptive field for a sequence of dilated convolutions."""
        rf_size = 1
        rf_sizes = [rf_size]
        
        for dilation in dilations:
            # Assuming 3x3 kernel
            kernel_size = 3
            effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
            rf_size = rf_size + effective_kernel - 1
            rf_sizes.append(rf_size)
        
        return rf_sizes[1:]  # Remove initial size
    
    def plot_multi_scale_context_aggregation(self):
        """
        Visualize how dilated convolutions aggregate multi-scale context.
        
        Shows parallel processing of different scales and their fusion.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Simulate multi-scale features
        scales = [1, 2, 4, 8, 16]
        feature_size = 64
        
        # Generate synthetic multi-scale feature responses
        np.random.seed(42)
        feature_responses = {}
        
        for scale in scales:
            # Different scales capture different frequency content
            base_frequency = 1.0 / scale
            x = np.linspace(0, 4*np.pi, feature_size)
            
            # Combine multiple frequency components
            response = (np.sin(base_frequency * x) + 
                       0.5 * np.sin(2 * base_frequency * x) +
                       0.25 * np.random.randn(feature_size) * 0.1)
            
            feature_responses[scale] = response
        
        # Plot individual scale responses
        for i, scale in enumerate(scales):
            color = plt.cm.viridis(i / len(scales))
            ax1.plot(feature_responses[scale], label=f'Scale {scale}', 
                    color=color, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Spatial Position')
        ax1.set_ylabel('Feature Response')
        ax1.set_title('Individual Scale Feature Responses', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Show scale-specific frequency content
        freqs = np.fft.fftfreq(feature_size)
        for i, scale in enumerate(scales):
            fft = np.abs(np.fft.fft(feature_responses[scale]))
            color = plt.cm.viridis(i / len(scales))
            ax2.plot(freqs[:feature_size//2], fft[:feature_size//2], 
                    label=f'Scale {scale}', color=color, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Content by Scale', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 0.2)
        
        # Fusion strategies comparison
        fusion_strategies = {
            'Concatenation': np.concatenate(list(feature_responses.values())),
            'Element-wise Sum': np.sum(list(feature_responses.values()), axis=0),
            'Weighted Average': np.average(list(feature_responses.values()), 
                                         axis=0, weights=[1, 2, 3, 2, 1]),
            'Max Pooling': np.maximum.reduce(list(feature_responses.values()))
        }
        
        for i, (strategy, fused) in enumerate(fusion_strategies.items()):
            if strategy == 'Concatenation':
                # Handle different length for concatenation
                x_vals = np.linspace(0, len(fused), len(fused))
            else:
                x_vals = np.arange(len(fused))
            
            ax3.plot(x_vals, fused, label=strategy, linewidth=3, alpha=0.8)
        
        ax3.set_xlabel('Feature Dimension')
        ax3.set_ylabel('Fused Response')
        ax3.set_title('Multi-Scale Fusion Strategies', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Context aggregation effectiveness
        scales_visual = ['Fine\n(d=1)', 'Local\n(d=2)', 'Medium\n(d=4)', 
                        'Large\n(d=8)', 'Global\n(d=16)']
        effectiveness = [0.9, 0.7, 0.85, 0.95, 0.8]  # Simulated effectiveness scores
        colors = [self.colors['dilation1'], self.colors['dilation2'], 
                 self.colors['dilation4'], self.colors['dilation8'], 
                 self.colors['dilation16']]
        
        bars = ax4.bar(scales_visual, effectiveness, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, eff in zip(bars, effectiveness):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('Context Scale')
        ax4.set_ylabel('Context Capture Effectiveness')
        ax4.set_title('Multi-Scale Context Effectiveness', fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('multi_scale_context_aggregation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🎯 Multi-Scale Context Aggregation:")
        print("   Different scales capture different frequency content")
        print("   Parallel processing enables efficient fusion")
        print("   Complementary information from each scale")
        print("   Fusion strategies affect final representation quality")
    
    def plot_aspp_architecture(self):
        """
        Visualize the Atrous Spatial Pyramid Pooling (ASPP) architecture.
        
        Shows parallel processing paths and feature fusion in ASPP.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # ASPP Architecture Diagram
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 12)
        
        # Input feature map
        input_rect = FancyBboxPatch((1, 10), 8, 1, boxstyle="round,pad=0.1",
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
        ax1.add_patch(input_rect)
        ax1.text(5, 10.5, 'Input Feature Map (H×W×C)', ha='center', va='center',
                fontweight='bold', fontsize=12)
        
        # Parallel branches
        branch_configs = [
            (1, 8, '1×1 Conv', self.colors['detail']),
            (1, 6.5, '3×3, d=6', self.colors['dilation1']),
            (1, 5, '3×3, d=12', self.colors['dilation2']),
            (1, 3.5, '3×3, d=18', self.colors['dilation4']),
            (1, 2, 'Global Pool', self.colors['context'])
        ]
        
        branch_outputs = []
        for x, y, label, color in branch_configs:
            # Branch box
            branch_rect = FancyBboxPatch((x, y), 8, 1, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='black', 
                                       linewidth=2, alpha=0.8)
            ax1.add_patch(branch_rect)
            ax1.text(5, y+0.5, label, ha='center', va='center',
                    fontweight='bold', fontsize=11, color='white')
            
            # Arrow from input
            ax1.annotate('', xy=(5, y+1), xytext=(5, 9.8),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
            
            branch_outputs.append((5, y))
        
        # Concatenation
        concat_rect = FancyBboxPatch((3, 0.5), 4, 0.8, boxstyle="round,pad=0.1",
                                    facecolor=self.colors['fusion'], edgecolor='black',
                                    linewidth=2)
        ax1.add_patch(concat_rect)
        ax1.text(5, 0.9, 'Concatenate', ha='center', va='center',
                fontweight='bold', fontsize=11, color='white')
        
        # Arrows to concatenation
        for x, y in branch_outputs:
            ax1.annotate('', xy=(5, 1.3), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        ax1.set_title('ASPP Architecture: Parallel Multi-Scale Processing', 
                     fontweight='bold', fontsize=14, pad=20)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.axis('off')
        
        # ASPP vs Traditional Comparison
        comparison_data = {
            'Metric': ['Receptive Field', 'Parameters', 'Resolution', 'Context Quality', 'Efficiency'],
            'Traditional Pyramid': [50, 100, 25, 60, 40],  # Relative scores
            'ASPP': [95, 60, 100, 95, 90]
        }
        
        x = np.arange(len(comparison_data['Metric']))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, comparison_data['Traditional Pyramid'], width,
                       label='Traditional Pyramid', color=self.colors['traditional'],
                       alpha=0.8, edgecolor='black')
        bars2 = ax2.bar(x + width/2, comparison_data['ASPP'], width,
                       label='ASPP', color=self.colors['fusion'],
                       alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Performance Metrics')
        ax2.set_ylabel('Relative Score (0-100)')
        ax2.set_title('ASPP vs Traditional Multi-Scale Processing', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_data['Metric'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig('aspp_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🏗️ ASPP Architecture Analysis:")
        print("   Parallel branches process different scales simultaneously")
        print("   Global pooling captures image-level context")
        print("   Concatenation preserves all scale information")
        print("   Superior to traditional pyramids in efficiency and quality")
    
    def plot_wavenet_temporal_modeling(self):
        """
        Visualize WaveNet's dilated convolution approach for temporal modeling.
        
        Shows exponential receptive field growth in sequential data.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # WaveNet dilation pattern
        layers = 10
        dilations = [2**i for i in range(layers)]
        
        # Visualization of temporal receptive field
        sequence_length = 32
        layer_positions = np.arange(layers)
        
        # Create receptive field visualization
        for layer, dilation in enumerate(dilations):
            # Calculate which time steps this layer can see
            center = sequence_length // 2
            reach = dilation * 2  # Simplified reach calculation
            
            start = max(0, center - reach)
            end = min(sequence_length, center + reach)
            
            # Color intensity based on layer depth
            alpha = 0.7 - 0.05 * layer
            color = plt.cm.viridis(layer / layers)
            
            ax1.barh(layer, end - start, left=start, alpha=alpha, 
                    color=color, edgecolor='white', linewidth=0.5)
            
            # Add dilation rate label
            ax1.text(sequence_length + 1, layer, f'd={dilation}',
                    va='center', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Layer Index')
        ax1.set_title('WaveNet Temporal Receptive Fields', fontweight='bold')
        ax1.set_xlim(0, sequence_length + 8)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Receptive field growth
        cumulative_rf = []
        rf = 1
        
        for dilation in dilations:
            rf = rf + dilation
            cumulative_rf.append(rf)
        
        ax2.plot(layer_positions, cumulative_rf, 'o-', linewidth=3, 
                markersize=8, color=self.colors['context'])
        ax2.fill_between(layer_positions, cumulative_rf, alpha=0.3, 
                        color=self.colors['context'])
        
        ax2.set_xlabel('Layer Depth')
        ax2.set_ylabel('Receptive Field Size')
        ax2.set_title('Exponential Receptive Field Growth', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Add annotations for key points
        for i, rf in enumerate(cumulative_rf[::2]):  # Every other point
            ax2.annotate(f'{rf}', (i*2, rf), textcoords="offset points",
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # Comparison with traditional approaches
        approaches = ['RNN\n(Sequential)', 'CNN\n(Fixed kernel)', 'Attention\n(All pairs)', 'WaveNet\n(Dilated)']
        complexity = [100, 25, 200, 30]  # Relative computational complexity
        quality = [70, 40, 95, 90]  # Relative modeling quality
        
        scatter = ax3.scatter(complexity, quality, s=[300, 200, 400, 350],
                            c=[self.colors['traditional'], self.colors['detail'], 
                               self.colors['receptive'], self.colors['context']],
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, approach in enumerate(approaches):
            ax3.annotate(approach, (complexity[i], quality[i]),
                        textcoords="offset points", xytext=(0,15), 
                        ha='center', fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Computational Complexity (relative)')
        ax3.set_ylabel('Modeling Quality (relative)')
        ax3.set_title('Sequential Modeling Approaches Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 220)
        ax3.set_ylim(30, 100)
        
        # WaveNet architecture benefits
        benefits = ['Parameter\nEfficiency', 'Parallel\nTraining', 'Long-term\nDependencies', 
                   'Audio\nQuality', 'Training\nSpeed']
        scores = [95, 90, 85, 95, 80]
        
        bars = ax4.bar(benefits, scores, color=[self.colors['fusion'], self.colors['context'],
                                              self.colors['receptive'], self.colors['dilation4'],
                                              self.colors['detail']], 
                      alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{score}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax4.set_ylabel('Score (0-100)')
        ax4.set_title('WaveNet Architecture Benefits', fontweight='bold')
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('wavenet_temporal_modeling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🌊 WaveNet Temporal Modeling:")
        print("   Exponential receptive field growth enables long-term dependencies")
        print("   Parallel training unlike RNNs")
        print("   High-quality audio generation with efficient architecture")
        print("   Foundation for modern autoregressive models")
    
    def plot_segmentation_revolution(self):
        """
        Visualize how dilated convolutions revolutionized semantic segmentation.
        
        Shows before/after comparison and performance improvements.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # Traditional FCN approach (top row)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Simulate traditional FCN pipeline
        resolutions = [256, 128, 64, 32, 16, 32, 64, 128, 256]
        stages = ['Input', 'Conv+Pool', 'Conv+Pool', 'Conv+Pool', 'Conv+Pool', 
                 'Upsample', 'Upsample', 'Upsample', 'Output']
        
        # Plot resolution changes
        x_positions = np.arange(len(resolutions))
        ax1.plot(x_positions, resolutions, 'ro-', linewidth=4, markersize=10, 
                label='Traditional FCN', color=self.colors['traditional'])
        ax1.fill_between(x_positions, resolutions, alpha=0.3, color=self.colors['traditional'])
        
        # Add stage labels
        for i, (pos, stage) in enumerate(zip(x_positions, stages)):
            ax1.annotate(stage, (pos, resolutions[i] + 20), ha='center', 
                        rotation=45 if len(stage) > 5 else 0, fontsize=10, fontweight='bold')
        
        # Highlight information loss
        loss_regions = [(1, 4), (4, 8)]
        for start, end in loss_regions:
            ax1.axvspan(start, end, alpha=0.2, color='red', 
                       label='Information Loss' if start == 1 else "")
        
        ax1.set_xlabel('Processing Stage')
        ax1.set_ylabel('Spatial Resolution')
        ax1.set_title('Traditional FCN: Resolution Loss Problem', fontweight='bold', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 300)
        
        # Dilated convolution approach (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Constant resolution with increasing dilation
        dilated_stages = ['Input', 'Conv d=1', 'Conv d=2', 'Conv d=4', 'Conv d=8', 'Output']
        dilated_resolution = [256] * len(dilated_stages)
        dilated_rf = [3, 7, 15, 31, 63, 63]  # Receptive field sizes
        
        x_pos = np.arange(len(dilated_stages))
        
        # Plot constant resolution
        ax2.plot(x_pos, dilated_resolution, 'go-', linewidth=4, markersize=10,
                color=self.colors['fusion'], label='Resolution')
        
        # Plot growing receptive field
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_pos, dilated_rf, 'bs-', linewidth=3, markersize=8,
                     color=self.colors['receptive'], label='Receptive Field')
        
        ax2.set_xlabel('Processing Stage')
        ax2.set_ylabel('Spatial Resolution', color=self.colors['fusion'])
        ax2_twin.set_ylabel('Receptive Field Size', color=self.colors['receptive'])
        ax2.set_title('Dilated CNN:\nResolution Preservation', fontweight='bold')
        
        # Add stage labels
        for i, stage in enumerate(dilated_stages):
            ax2.text(i, dilated_resolution[i] + 20, stage, ha='center', 
                    rotation=45, fontsize=9, fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 300)
        
        # Performance comparison (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        
        metrics = ['mIoU\n(%)', 'Inference\nSpeed', 'Memory\nUsage', 'Boundary\nQuality']
        fcn_scores = [65, 70, 60, 50]
        dilated_scores = [79, 85, 90, 85]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, fcn_scores, width, label='Traditional FCN',
                       color=self.colors['traditional'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, dilated_scores, width, label='Dilated CNN',
                       color=self.colors['fusion'], alpha=0.8)
        
        # Add improvement percentages
        for i, (fcn, dilated) in enumerate(zip(fcn_scores, dilated_scores)):
            improvement = ((dilated - fcn) / fcn) * 100
            ax3.text(i, max(fcn, dilated) + 5, f'+{improvement:.0f}%',
                    ha='center', va='bottom', fontweight='bold', 
                    color='green', fontsize=10)
        
        ax3.set_xlabel('Performance Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Improvements', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 100)
        
        # Timeline and impact (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        
        years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
        miou_progress = [62, 65, 72, 79, 82, 85, 87]  # Simulated mIoU progress
        
        ax4.plot(years, miou_progress, 'o-', linewidth=4, markersize=10,
                color=self.colors['context'])
        ax4.fill_between(years, miou_progress, alpha=0.3, color=self.colors['context'])
        
        # Annotate key milestones
        milestones = {
            2014: 'FCN',
            2016: 'DeepLab v1',
            2017: 'DeepLab v2',
            2018: 'DeepLab v3+'
        }
        
        for year, milestone in milestones.items():
            if year in years:
                idx = years.index(year)
                ax4.annotate(milestone, (year, miou_progress[idx]),
                           textcoords="offset points", xytext=(0,15),
                           ha='center', fontweight='bold', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('PASCAL VOC mIoU (%)')
        ax4.set_title('Segmentation Progress', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(60, 90)
        
        # Applications showcase (bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        
        applications = ['Medical\nImaging', 'Autonomous\nDriving', 'Satellite\nImagery', 
                       'Augmented\nReality', 'Industrial\nInspection']
        impact_scores = [95, 90, 85, 80, 88]
        colors_app = [self.colors['dilation1'], self.colors['dilation2'], 
                     self.colors['dilation4'], self.colors['dilation8'], 
                     self.colors['fusion']]
        
        bars = ax5.bar(applications, impact_scores, color=colors_app, 
                      alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add impact descriptions
        descriptions = ['Precise tumor\nsegmentation', 'Real-time scene\nunderstanding', 
                       'Land use\nclassification', 'Object\nocclusion', 
                       'Defect\ndetection']
        
        for bar, desc, score in zip(bars, descriptions, impact_scores):
            ax5.text(bar.get_x() + bar.get_width()/2, score/2, desc,
                    ha='center', va='center', fontweight='bold', 
                    fontsize=10, color='white')
        
        ax5.set_ylabel('Impact Score (0-100)')
        ax5.set_title('Real-World Applications of Dilated Convolution Segmentation', 
                     fontweight='bold', fontsize=14)
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('How Dilated Convolutions Revolutionized Semantic Segmentation', 
                     fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig('segmentation_revolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🎯 Segmentation Revolution Impact:")
        print("   Resolution preservation vs traditional downsampling")
        print("   Significant performance improvements across metrics")
        print("   Enabled real-world applications requiring precise segmentation")
        print("   Foundation for modern dense prediction architectures")
    
    def plot_comprehensive_dilated_summary(self):
        """
        Create a comprehensive summary of dilated convolution innovations and impact.
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Dilated Convolutions: Multi-Scale Context Revolution', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Core innovation (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        
        innovations = ['Parameter\nEfficiency', 'Resolution\nPreservation', 'Context\nAggregation', 'Parallel\nProcessing']
        before_scores = [30, 25, 40, 20]
        after_scores = [95, 100, 90, 85]
        
        x = np.arange(len(innovations))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_scores, width, label='Before Dilated Conv',
                       color=self.colors['traditional'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, after_scores, width, label='After Dilated Conv',
                       color=self.colors['fusion'], alpha=0.8)
        
        ax1.set_xlabel('Innovation Areas')
        ax1.set_ylabel('Capability Score (0-100)')
        ax1.set_title('Core Innovations Impact', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(innovations)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Architecture evolution timeline (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        years = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020])
        architectures = ['FCN', 'SegNet', 'DeepLab v1', 'DeepLab v2', 'DeepLab v3', 'PSPNet', 'DeepLab v3+']
        performance = np.array([62, 60, 70, 75, 79, 78, 82])
        
        ax2.plot(years, performance, 'o-', linewidth=4, markersize=10, color=self.colors['context'])
        ax2.fill_between(years, performance, alpha=0.3, color=self.colors['context'])
        
        # Annotate architectures
        for year, arch, perf in zip(years[::2], architectures[::2], performance[::2]):
            ax2.annotate(arch, (year, perf), textcoords="offset points",
                        xytext=(0,15), ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('PASCAL VOC mIoU (%)')
        ax2.set_title('Segmentation Architecture Evolution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Multi-scale visualization (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        
        scales = ['Fine\nDetails', 'Local\nPatterns', 'Object\nParts', 'Object\nBoundaries', 'Scene\nContext']
        dilation_rates = [1, 2, 4, 8, 16]
        effectiveness = [90, 85, 95, 88, 92]
        
        colors_scale = [self.colors['dilation1'], self.colors['dilation2'], 
                       self.colors['dilation4'], self.colors['dilation8'], 
                       self.colors['dilation16']]
        
        bars = ax3.barh(scales, effectiveness, color=colors_scale, alpha=0.8, edgecolor='black')
        
        # Add dilation rate labels
        for bar, dilation in zip(bars, dilation_rates):
            ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'd={dilation}', va='center', fontweight='bold')
        
        ax3.set_xlabel('Context Capture Effectiveness (%)')
        ax3.set_title('Multi-Scale Context Capture', fontweight='bold')
        ax3.set_xlim(0, 100)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Application domains (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        domains = ['Computer\nVision', 'Audio\nProcessing', 'NLP', 'Medical\nImaging', 'Robotics']
        adoption = [95, 90, 60, 85, 75]
        impact = [90, 95, 70, 95, 80]
        
        scatter = ax4.scatter(adoption, impact, s=[400, 350, 200, 380, 300],
                            c=[self.colors['detail'], self.colors['context'], 
                               self.colors['receptive'], self.colors['dilation4'],
                               self.colors['fusion']], alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, domain in enumerate(domains):
            ax4.annotate(domain, (adoption[i], impact[i]), textcoords="offset points",
                        xytext=(0,20), ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Adoption Rate (%)')
        ax4.set_ylabel('Impact Score (%)')
        ax4.set_title('Cross-Domain Applications', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(50, 100)
        ax4.set_ylim(60, 100)
        
        # 5. Computational efficiency (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        
        methods = ['Large\nKernels', 'Multi-Scale\nPyramid', 'Attention\nMechanism', 'Dilated\nConvolutions']
        compute_cost = [100, 150, 200, 60]
        quality = [70, 80, 90, 85]
        
        bubbles = ax5.scatter(compute_cost, quality, s=[300, 400, 500, 350],
                            c=[self.colors['traditional'], self.colors['detail'], 
                               self.colors['receptive'], self.colors['fusion']],
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, method in enumerate(methods):
            ax5.annotate(method, (compute_cost[i], quality[i]), textcoords="offset points",
                        xytext=(0,15), ha='center', fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax5.set_xlabel('Computational Cost (relative)')
        ax5.set_ylabel('Output Quality (%)')
        ax5.set_title('Efficiency vs Quality Trade-off', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add efficiency frontier
        ax5.plot([60, 100], [85, 70], 'r--', linewidth=2, alpha=0.7, label='Efficiency Frontier')
        ax5.legend()
        
        # 6. Future directions (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        
        future_trends = ['Adaptive\nDilation', 'Hardware\nOptimization', 'Transformer\nIntegration', 
                        '3D/4D\nExtensions', 'Efficient\nArchitecture']
        potential = [85, 90, 95, 80, 88]
        timeline = [2, 3, 1, 4, 2]  # Years to adoption
        
        colors_future = plt.cm.plasma(np.array(timeline) / max(timeline))
        
        bars = ax6.bar(future_trends, potential, color=colors_future, alpha=0.8, edgecolor='black')
        
        for bar, years in zip(bars, timeline):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{years}y', ha='center', va='bottom', fontweight='bold')
        
        ax6.set_ylabel('Potential Impact (%)')
        ax6.set_title('Future Research Directions', fontweight='bold')
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Mathematical foundation (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.text(0.5, 0.8, r'$\mathbf{Dilated\ Convolution\ Mathematics}$',
                ha='center', va='center', transform=ax7.transAxes, 
                fontsize=16, fontweight='bold')
        
        ax7.text(0.25, 0.6, r'Standard: $y[i] = \sum_k x[i+k] \cdot w[k]$',
                ha='center', va='center', transform=ax7.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
        
        ax7.text(0.75, 0.6, r'Dilated: $y[i] = \sum_k x[i+d \cdot k] \cdot w[k]$',
                ha='center', va='center', transform=ax7.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        ax7.text(0.5, 0.4, r'Receptive Field: $RF = k + (k-1) \sum (d_i - 1)$',
                ha='center', va='center', transform=ax7.transAxes, fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        ax7.text(0.5, 0.2, 'Key Insight: Exponential RF growth with linear parameter increase',
                ha='center', va='center', transform=ax7.transAxes, fontsize=12,
                fontweight='bold', style='italic')
        
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        
        plt.savefig('dilated_convolutions_comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Comprehensive Dilated Convolutions Analysis:")
        print("   🔍 Core Innovation: Multi-scale context without resolution loss")
        print("   🚀 Performance: Dramatic improvements in dense prediction tasks")
        print("   ⚡ Efficiency: Better quality/compute trade-offs than alternatives")
        print("   🌐 Impact: Cross-domain applications from vision to audio to NLP")
        print("   🔮 Future: Foundation for next-generation efficient architectures")


def demonstrate_dilated_convolutions():
    """
    Interactive demonstration of dilated convolution concepts and applications.
    """
    print("🌊 Dilated Convolutions Visualization Demonstration")
    print("=" * 60)
    
    # Create visualizer
    visualizer = DilatedConvolutionVisualizer()
    
    # Show all visualizations
    print("\n1. Dilated Convolution Patterns:")
    visualizer.plot_dilated_convolution_patterns()
    
    print("\n2. Receptive Field Growth Analysis:")
    visualizer.plot_receptive_field_growth()
    
    print("\n3. Multi-Scale Context Aggregation:")
    visualizer.plot_multi_scale_context_aggregation()
    
    print("\n4. ASPP Architecture:")
    visualizer.plot_aspp_architecture()
    
    print("\n5. WaveNet Temporal Modeling:")
    visualizer.plot_wavenet_temporal_modeling()
    
    print("\n6. Segmentation Revolution:")
    visualizer.plot_segmentation_revolution()
    
    print("\n7. Comprehensive Summary:")
    visualizer.plot_comprehensive_dilated_summary()
    
    print("\n🎯 Key Takeaways:")
    print("   • Dilated convolutions expand receptive fields without parameters")
    print("   • Multi-scale context capture without resolution loss")
    print("   • Revolutionary impact on dense prediction tasks")
    print("   • Foundation for modern efficient architectures")
    print("   • Cross-domain applications from vision to audio to NLP")


if __name__ == "__main__":
    demonstrate_dilated_convolutions()