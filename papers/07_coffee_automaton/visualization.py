"""
Day 7: Coffee Automaton - Visualization Suite

Beautiful visualizations showing the rise and fall of complexity in closed systems.
This module creates publication-quality plots and animations demonstrating
the Coffee Automaton dynamics and complexity metrics.

Features:
- Real-time complexity evolution curves
- Heat map animations of coffee cooling
- Multi-metric complexity comparisons  
- Phase space trajectories
- Interactive complexity landscapes

Author: 30u30 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")


class CoffeeVisualizer:
    """
    Master visualization class for Coffee Automaton dynamics.
    Creates beautiful, publication-quality plots and animations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F06292']
        
    def plot_complexity_evolution(self, tracker, save_path: str = None) -> plt.Figure:
        """
        Plot the evolution of all complexity metrics over time.
        Shows the characteristic 'hump' shape of complexity rising and falling.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔥 Coffee Automaton: The Rise and Fall of Complexity', 
                     fontsize=16, fontweight='bold')
        
        metrics = [
            ('shannon_entropy', 'Shannon Entropy\n(Information Content)', '📊'),
            ('spatial_entropy', 'Spatial Entropy\n(Pattern Complexity)', '🗺️'),
            ('logical_depth', 'Logical Depth\n(Computational Complexity)', '💭'),
            ('effective_complexity', 'Effective Complexity\n(Gell-Mann Measure)', '⭐'),
            ('thermodynamic_depth', 'Thermodynamic Depth\n(Energy Required)', '🔥'),
            ('lempel_ziv', 'Lempel-Ziv Complexity\n(Algorithmic Information)', '🧮')
        ]
        
        for idx, (metric, title, emoji) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            times, values = tracker.get_complexity_curve(metric)
            
            # Plot the curve
            ax.plot(times, values, linewidth=3, color=self.colors[idx], alpha=0.8)
            ax.fill_between(times, values, alpha=0.3, color=self.colors[idx])
            
            # Mark the peak
            peak_time = tracker.get_peak_complexity_time(metric)
            peak_value = max(values)
            peak_idx = values.index(peak_value)
            ax.scatter([peak_time], [peak_value], color='red', s=100, zorder=5)
            ax.annotate(f'Peak\nt={peak_time}', 
                       xy=(peak_time, peak_value), 
                       xytext=(peak_time + len(times)*0.1, peak_value),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10)
            
            ax.set_title(f'{emoji} {title}', fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Complexity Value')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_cooling_animation(self, automaton, save_path: str = None) -> animation.FuncAnimation:
        """
        Create an animation showing coffee cooling over time.
        Beautiful heat map visualization with complexity overlay.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('☕ Coffee Cooling: Watching Complexity Emerge and Fade', 
                     fontsize=14, fontweight='bold')
        
        # Setup heat map
        im = ax1.imshow(automaton.history[0], cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
        ax1.set_title('Temperature Field')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Position')
        cbar = plt.colorbar(im, ax=ax1, label='Temperature')
        
        # Setup complexity plot
        ax2.set_xlim(0, len(automaton.history))
        ax2.set_ylim(0, max(automaton.complexity_history) * 1.1)
        ax2.set_title('Real-time Complexity')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Complexity')
        complexity_line, = ax2.plot([], [], linewidth=3, color='#FF6B6B')
        complexity_fill = ax2.fill_between([], [], alpha=0.3, color='#FF6B6B')
        
        time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            # Update heat map
            im.set_array(automaton.history[frame])
            
            # Update complexity plot
            if frame > 0:
                times = list(range(frame + 1))
                complexities = automaton.complexity_history[:frame + 1]
                complexity_line.set_data(times, complexities)
                
                # Update fill
                ax2.clear()
                ax2.plot(times, complexities, linewidth=3, color='#FF6B6B')
                ax2.fill_between(times, complexities, alpha=0.3, color='#FF6B6B')
                ax2.set_xlim(0, len(automaton.history))
                ax2.set_ylim(0, max(automaton.complexity_history) * 1.1)
                ax2.set_title('Real-time Complexity')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Complexity')
                ax2.grid(True, alpha=0.3)
                
                # Mark current point
                ax2.scatter([frame], [complexities[-1]], color='red', s=100, zorder=5)
            
            # Update time text
            time_text.set_text(f'Time: {frame}\nComplexity: {automaton.complexity_history[frame]:.3f}')
            
            return [im, complexity_line, time_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(automaton.history), 
                                     interval=100, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            
        return anim
        
    def plot_complexity_landscape(self, life_results: Dict, save_path: str = None) -> plt.Figure:
        """
        Plot the complexity landscape showing the 'life sweet spot'.
        Demonstrates that maximum complexity occurs at intermediate energy levels.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('🌱 The Life Sweet Spot: Where Complexity Peaks', 
                     fontsize=14, fontweight='bold')
        
        energies = list(life_results.keys())
        complexities = list(life_results.values())
        
        # Main complexity curve
        ax1.plot(energies, complexities, linewidth=4, color='#4ECDC4', marker='o', markersize=8)
        ax1.fill_between(energies, complexities, alpha=0.3, color='#4ECDC4')
        
        # Mark the peak (life zone)
        max_idx = np.argmax(complexities)
        optimal_energy = energies[max_idx]
        max_complexity = complexities[max_idx]
        
        ax1.scatter([optimal_energy], [max_complexity], color='red', s=200, zorder=5)
        ax1.annotate('🌱 LIFE ZONE\n(Peak Complexity)', 
                    xy=(optimal_energy, max_complexity),
                    xytext=(optimal_energy + 0.1, max_complexity + 0.02),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Mark zones
        ax1.axvspan(0, 0.3, alpha=0.2, color='blue', label='❄️ Too Cold (Low Complexity)')
        ax1.axvspan(0.7, 1.0, alpha=0.2, color='red', label='🔥 Too Hot (Low Complexity)')
        ax1.axvspan(0.3, 0.7, alpha=0.2, color='green', label='🌱 Goldilocks Zone')
        
        ax1.set_xlabel('Initial Energy Level', fontsize=12)
        ax1.set_ylabel('Peak Complexity Achieved', fontsize=12)
        ax1.set_title('Complexity vs Energy: The Universal Pattern')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Phase diagram
        ax2.scatter(energies, complexities, c=energies, cmap='coolwarm', s=100, alpha=0.8)
        
        # Add arrows showing evolution direction
        for i, (e, c) in enumerate(zip(energies, complexities)):
            if i % 2 == 0:  # Every other point for clarity
                ax2.annotate('', xy=(0.5, 0), xytext=(e, c),
                           arrowprops=dict(arrowstyle='->', alpha=0.5, lw=1))
        
        ax2.set_xlabel('Energy Level')
        ax2.set_ylabel('Complexity')
        ax2.set_title('Phase Diagram: Energy → Equilibrium')
        ax2.grid(True, alpha=0.3)
        
        # Add equilibrium line
        ax2.axhline(y=np.mean(complexities), color='gray', linestyle='--', 
                   label='Equilibrium Level', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_phase_analysis(self, tracker, save_path: str = None) -> plt.Figure:
        """
        Analyze the three phases of complexity evolution:
        Growth → Peak → Decay
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 Three Phases of Complexity Evolution', fontsize=16, fontweight='bold')
        
        # Get analysis for effective complexity
        analysis = tracker.analyze_phases('effective_complexity')
        times, values = tracker.get_complexity_curve('effective_complexity')
        
        # Phase 1: Growth
        growth_times = times[:analysis['growth_duration']]
        growth_values = values[:analysis['growth_duration']]
        
        ax1.plot(growth_times, growth_values, linewidth=3, color='green', marker='o')
        ax1.fill_between(growth_times, growth_values, alpha=0.3, color='green')
        ax1.set_title('🌱 Phase 1: Growth (Simple → Complex)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Complexity')
        ax1.grid(True, alpha=0.3)
        
        # Add growth rate annotation
        ax1.text(0.05, 0.95, f'Growth Rate: {analysis["growth_rate"]:.4f}/step', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Phase 2: Peak Region
        peak_start = max(0, analysis['peak_time'] - 10)
        peak_end = min(len(times), analysis['peak_time'] + 10)
        peak_times = times[peak_start:peak_end]
        peak_values = values[peak_start:peak_end]
        
        ax2.plot(peak_times, peak_values, linewidth=3, color='red', marker='s')
        ax2.fill_between(peak_times, peak_values, alpha=0.3, color='red')
        ax2.scatter([analysis['peak_time']], [analysis['peak_value']], 
                   color='red', s=200, zorder=5)
        ax2.set_title('🎯 Phase 2: Peak (Maximum Complexity)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Complexity')
        ax2.grid(True, alpha=0.3)
        
        ax2.text(0.05, 0.95, f'Peak Value: {analysis["peak_value"]:.4f}\nPeak Time: {analysis["peak_time"]}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Phase 3: Decay
        decay_times = times[analysis['growth_duration']:]
        decay_values = values[analysis['growth_duration']:]
        
        ax3.plot(decay_times, decay_values, linewidth=3, color='blue', marker='^')
        ax3.fill_between(decay_times, decay_values, alpha=0.3, color='blue')
        ax3.set_title('🌊 Phase 3: Decay (Complex → Simple)')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Complexity')
        ax3.grid(True, alpha=0.3)
        
        ax3.text(0.05, 0.95, f'Decay Rate: {analysis["decay_rate"]:.4f}/step', 
                transform=ax3.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Summary Statistics
        ax4.axis('off')
        summary_text = f"""
        📊 COMPLEXITY EVOLUTION SUMMARY
        
        🕐 Total Evolution Time: {analysis['total_evolution_time']} steps
        
        🌱 Growth Phase:
           • Duration: {analysis['growth_duration']} steps
           • Rate: {analysis['growth_rate']:.4f} units/step
           
        🎯 Peak Phase:
           • Time: {analysis['peak_time']} steps
           • Value: {analysis['peak_value']:.4f}
           
        🌊 Decay Phase:
           • Duration: {analysis['decay_duration']} steps  
           • Rate: {analysis['decay_rate']:.4f} units/step
           
        💡 Key Insight: 
        Complexity is temporary! It emerges from 
        order, peaks in the middle, then fades 
        back to simplicity.
        
        This is why life and intelligence exist 
        in the universe's "middle age" - where 
        complexity blooms. ✨
        """
        
        ax4.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_comparative_metrics(self, tracker, save_path: str = None) -> plt.Figure:
        """
        Compare all complexity metrics on the same plot.
        Shows how different measures capture different aspects of complexity.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('🔬 Complexity Metrics Comparison: Different Perspectives', 
                     fontsize=14, fontweight='bold')
        
        metrics = ['shannon_entropy', 'spatial_entropy', 'logical_depth', 
                  'effective_complexity', 'thermodynamic_depth', 'lempel_ziv']
        
        labels = ['Shannon Entropy', 'Spatial Entropy', 'Logical Depth',
                 'Effective Complexity', 'Thermodynamic Depth', 'Lempel-Ziv']
        
        # Normalize each metric for comparison
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            times, values = tracker.get_complexity_curve(metric)
            
            # Normalize to [0, 1] for comparison
            normalized_values = np.array(values)
            if np.max(normalized_values) > 0:
                normalized_values = normalized_values / np.max(normalized_values)
            
            ax.plot(times, normalized_values, linewidth=2.5, label=label, 
                   color=self.colors[i % len(self.colors)], alpha=0.8)
            
            # Mark peaks
            peak_idx = np.argmax(normalized_values)
            peak_time = times[peak_idx]
            peak_value = normalized_values[peak_idx]
            ax.scatter([peak_time], [peak_value], color=self.colors[i % len(self.colors)], 
                      s=60, alpha=0.9, zorder=5)
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Normalized Complexity', fontsize=12)
        ax.set_title('All Metrics Show the Same Pattern: Rise, Peak, Fall')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add insight box
        insight_text = """
        💡 Universal Pattern:
        Despite measuring different 
        aspects, ALL complexity 
        metrics show the same 
        fundamental pattern:
        
        Simple → Complex → Simple
        
        This universality suggests 
        deep physical principles 
        at work! 🌌
        """
        
        ax.text(0.02, 0.98, insight_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_complexity_dashboard(automaton, tracker, life_results, save_dir: str = None):
    """
    Create a comprehensive dashboard showing all Coffee Automaton results.
    
    Args:
        automaton: CoffeeAutomaton instance
        tracker: ComplexityTracker instance  
        life_results: Results from life sweet spot experiment
        save_dir: Directory to save plots (optional)
    """
    visualizer = CoffeeVisualizer()
    
    print("🎨 Creating Coffee Automaton Visualization Dashboard...")
    
    # 1. Main complexity evolution plot
    fig1 = visualizer.plot_complexity_evolution(tracker, 
                                               f"{save_dir}/complexity_evolution.png" if save_dir else None)
    plt.show()
    
    # 2. Phase analysis
    fig2 = visualizer.plot_phase_analysis(tracker,
                                         f"{save_dir}/phase_analysis.png" if save_dir else None)
    plt.show()
    
    # 3. Life sweet spot
    fig3 = visualizer.plot_complexity_landscape(life_results,
                                               f"{save_dir}/life_sweet_spot.png" if save_dir else None)
    plt.show()
    
    # 4. Comparative metrics
    fig4 = visualizer.plot_comparative_metrics(tracker,
                                              f"{save_dir}/metric_comparison.png" if save_dir else None)
    plt.show()
    
    # 5. Create animation (optional - can be time consuming)
    print("🎬 Creating cooling animation...")
    anim = visualizer.create_cooling_animation(automaton,
                                              f"{save_dir}/coffee_cooling.gif" if save_dir else None)
    plt.show()
    
    print("✅ Dashboard complete! Coffee complexity visualized beautifully.")
    
    return {
        'evolution': fig1,
        'phases': fig2, 
        'landscape': fig3,
        'comparison': fig4,
        'animation': anim
    }


if __name__ == "__main__":
    # Demo visualization
    from implementation import CoffeeExperiments
    
    print("🔥 Coffee Automaton Visualization Demo")
    print("=" * 50)
    
    # Quick experiment
    automaton, tracker = CoffeeExperiments.basic_cooling_experiment(steps=100)
    life_results = CoffeeExperiments.life_sweet_spot_experiment()
    
    # Create visualizations
    visualizer = CoffeeVisualizer()
    
    # Show just the main complexity plot for demo
    fig = visualizer.plot_complexity_evolution(tracker)
    plt.show()
    
    print("✅ Demo complete! Ready to visualize complexity dynamics.")