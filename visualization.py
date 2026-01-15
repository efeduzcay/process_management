"""
Visualization Module
--------------------
Academic-grade plots for MG-RR research paper.

Produces publication-quality figures using Matplotlib/Seaborn.

Author: MG-RR Research Study
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# Academic style configuration
ACADEMIC_STYLE = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
}

# Color scheme (colorblind-friendly)
COLORS = {
    'rr': '#E69F00',      # Orange
    'mgrr': '#0072B2',    # Blue
    'interactive': '#009E73',  # Green
    'batch': '#D55E00',   # Red-orange
    'neutral': '#999999'  # Gray
}


def setup_style():
    """Configure matplotlib for academic plots."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.rcParams.update(ACADEMIC_STYLE)
    
    if SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid", palette="colorblind")


class ResultsVisualizer:
    """
    Creates publication-quality visualizations for MG-RR research.
    """
    
    def __init__(self, output_dir: str = "figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib is required for visualization")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        setup_style()
    
    def plot_waiting_time_boxplot(
        self,
        rr_wt: List[float],
        mgrr_wt: List[float],
        title: str = "Average Waiting Time Comparison",
        save_as: str = "waiting_time_boxplot.png"
    ) -> plt.Figure:
        """
        Create box plot comparing waiting times.
        
        Args:
            rr_wt: RR waiting times across simulations
            mgrr_wt: MG-RR waiting times across simulations
            title: Plot title
            save_as: Filename for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = [rr_wt, mgrr_wt]
        positions = [1, 2]
        
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
        
        # Color boxes
        colors = [COLORS['rr'], COLORS['mgrr']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style whiskers and caps
        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color('black')
                line.set_linewidth(1.5)
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        ax.set_xticklabels(['Standard RR', 'MG-RR'])
        ax.set_ylabel('Average Waiting Time (ticks)')
        ax.set_title(title)
        
        # Add mean markers
        means = [sum(d)/len(d) for d in data]
        ax.scatter(positions, means, marker='D', color='red', s=50, zorder=5, label='Mean')
        ax.legend(loc='upper right')
        
        # Add statistical annotation
        mean_diff = means[0] - means[1]
        pct_change = (mean_diff / means[0]) * 100 if means[0] != 0 else 0
        ax.text(0.02, 0.98, f'Δ = {mean_diff:.2f} ({pct_change:+.1f}%)',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            fig.savefig(os.path.join(self.output_dir, save_as))
        
        return fig
    
    def plot_stutter_comparison(
        self,
        rr_stutters: List[int],
        mgrr_stutters: List[int],
        title: str = "Stutter Count Comparison (Interactive Processes)",
        save_as: str = "stutter_comparison.png"
    ) -> plt.Figure:
        """
        Create bar chart comparing stutter counts.
        
        Args:
            rr_stutters: Total stutters per simulation for RR
            mgrr_stutters: Total stutters per simulation for MG-RR
            title: Plot title
            save_as: Filename for saved figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Total stutters bar chart
        total_rr = sum(rr_stutters)
        total_mgrr = sum(mgrr_stutters)
        
        bars = ax1.bar(['Standard RR', 'MG-RR'], [total_rr, total_mgrr],
                       color=[COLORS['rr'], COLORS['mgrr']], alpha=0.8, edgecolor='black')
        
        ax1.set_ylabel('Total Stutter Count')
        ax1.set_title('Total Stutters Across All Simulations')
        
        # Add value labels
        for bar, val in zip(bars, [total_rr, total_mgrr]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_rr*0.02,
                     f'{int(val):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Reduction percentage
        reduction = (total_rr - total_mgrr) / total_rr * 100 if total_rr > 0 else 0
        ax1.text(0.5, 0.95, f'Reduction: {reduction:.1f}%',
                 transform=ax1.transAxes, ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Right: Distribution comparison
        mean_rr = sum(rr_stutters) / len(rr_stutters)
        mean_mgrr = sum(mgrr_stutters) / len(mgrr_stutters)
        
        ax2.hist(rr_stutters, bins=20, alpha=0.6, label=f'RR (μ={mean_rr:.2f})',
                 color=COLORS['rr'], edgecolor='black')
        ax2.hist(mgrr_stutters, bins=20, alpha=0.6, label=f'MG-RR (μ={mean_mgrr:.2f})',
                 color=COLORS['mgrr'], edgecolor='black')
        
        ax2.axvline(mean_rr, color=COLORS['rr'], linestyle='--', linewidth=2)
        ax2.axvline(mean_mgrr, color=COLORS['mgrr'], linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Stutter Count per Simulation')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Stutter Distribution')
        ax2.legend()
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_as:
            fig.savefig(os.path.join(self.output_dir, save_as))
        
        return fig
    
    def plot_tat_distribution(
        self,
        rr_tat: List[float],
        mgrr_tat: List[float],
        title: str = "Turnaround Time Distribution",
        save_as: str = "tat_distribution.png"
    ) -> plt.Figure:
        """
        Create distribution plot for turnaround times.
        
        Args:
            rr_tat: RR turnaround times
            mgrr_tat: MG-RR turnaround times
            title: Plot title
            save_as: Filename
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Overlaid histograms
        ax1 = axes[0]
        
        if SEABORN_AVAILABLE:
            sns.kdeplot(rr_tat, ax=ax1, label='Standard RR', color=COLORS['rr'], 
                       linewidth=2, fill=True, alpha=0.3)
            sns.kdeplot(mgrr_tat, ax=ax1, label='MG-RR', color=COLORS['mgrr'],
                       linewidth=2, fill=True, alpha=0.3)
        else:
            ax1.hist(rr_tat, bins=30, alpha=0.5, label='Standard RR', 
                    color=COLORS['rr'], density=True)
            ax1.hist(mgrr_tat, bins=30, alpha=0.5, label='MG-RR',
                    color=COLORS['mgrr'], density=True)
        
        ax1.set_xlabel('Average Turnaround Time (ticks)')
        ax1.set_ylabel('Density')
        ax1.set_title('Turnaround Time Distribution')
        ax1.legend()
        
        # Right: Q-Q style comparison
        ax2 = axes[1]
        
        sorted_rr = sorted(rr_tat)
        sorted_mgrr = sorted(mgrr_tat)
        
        ax2.scatter(sorted_rr, sorted_mgrr, alpha=0.5, c=COLORS['neutral'], s=20)
        
        # Reference line (y = x)
        min_val = min(min(sorted_rr), min(sorted_mgrr))
        max_val = max(max(sorted_rr), max(sorted_mgrr))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y = x')
        
        ax2.set_xlabel('RR Turnaround Time')
        ax2.set_ylabel('MG-RR Turnaround Time')
        ax2.set_title('RR vs MG-RR Q-Q Plot')
        ax2.legend()
        
        # Points above line: MG-RR worse, below: MG-RR better
        above = sum(1 for r, m in zip(sorted_rr, sorted_mgrr) if m > r)
        below = len(sorted_rr) - above
        ax2.text(0.05, 0.95, f'MG-RR better: {below}\nMG-RR worse: {above}',
                transform=ax2.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_as:
            fig.savefig(os.path.join(self.output_dir, save_as))
        
        return fig
    
    def plot_interactive_vs_batch_tradeoff(
        self,
        rr_wt_interactive: List[float],
        mgrr_wt_interactive: List[float],
        rr_wt_batch: List[float],
        mgrr_wt_batch: List[float],
        title: str = "Interactive vs Batch Trade-off Analysis",
        save_as: str = "tradeoff_analysis.png"
    ) -> plt.Figure:
        """
        Visualize the trade-off between interactive and batch performance.
        
        Args:
            rr_wt_interactive: RR waiting times for interactive processes
            mgrr_wt_interactive: MG-RR waiting times for interactive processes
            rr_wt_batch: RR waiting times for batch processes
            mgrr_wt_batch: MG-RR waiting times for batch processes
            title: Plot title
            save_as: Filename
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Calculate means
        means = {
            'RR Interactive': sum(rr_wt_interactive) / len(rr_wt_interactive) if rr_wt_interactive else 0,
            'MG-RR Interactive': sum(mgrr_wt_interactive) / len(mgrr_wt_interactive) if mgrr_wt_interactive else 0,
            'RR Batch': sum(rr_wt_batch) / len(rr_wt_batch) if rr_wt_batch else 0,
            'MG-RR Batch': sum(mgrr_wt_batch) / len(mgrr_wt_batch) if mgrr_wt_batch else 0,
        }
        
        # Grouped bar chart
        x = [0, 1]
        width = 0.35
        
        interactive_vals = [means['RR Interactive'], means['MG-RR Interactive']]
        batch_vals = [means['RR Batch'], means['MG-RR Batch']]
        
        bars1 = ax.bar([i - width/2 for i in x], interactive_vals, width,
                       label='Interactive', color=COLORS['interactive'], alpha=0.8, edgecolor='black')
        bars2 = ax.bar([i + width/2 for i in x], batch_vals, width,
                       label='Batch', color=COLORS['batch'], alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Standard RR', 'MG-RR'])
        ax.set_ylabel('Average Waiting Time (ticks)')
        ax.set_title(title)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Add change annotations
        int_change = means['MG-RR Interactive'] - means['RR Interactive']
        batch_change = means['MG-RR Batch'] - means['RR Batch']
        
        int_pct = (int_change / means['RR Interactive']) * 100 if means['RR Interactive'] else 0
        batch_pct = (batch_change / means['RR Batch']) * 100 if means['RR Batch'] else 0
        
        # Arrow annotations
        y_max = max(max(interactive_vals), max(batch_vals)) * 1.15
        
        ax.annotate(f'Interactive: {int_pct:+.1f}%',
                   xy=(0.5, y_max * 0.95), fontsize=11,
                   color='green' if int_change < 0 else 'red',
                   ha='center', fontweight='bold')
        
        ax.annotate(f'Batch: {batch_pct:+.1f}%',
                   xy=(0.5, y_max * 0.88), fontsize=11,
                   color='green' if batch_change < 0 else 'red',
                   ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_as:
            fig.savefig(os.path.join(self.output_dir, save_as))
        
        return fig
    
    def plot_load_heatmap(
        self,
        results_by_load: Dict[str, Dict[str, float]],
        metric: str = 'stutter_reduction',
        title: str = "Performance Across Load Levels",
        save_as: str = "load_heatmap.png"
    ) -> plt.Figure:
        """
        Create heatmap showing performance across different load levels.
        
        Args:
            results_by_load: Dict mapping load_level to metric values
            metric: Metric to visualize
            title: Plot title
            save_as: Filename
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        load_levels = list(results_by_load.keys())
        metrics = ['Stutter Reduction %', 'WT Improvement %', 'TAT Change %']
        
        # Create data matrix
        data = []
        for load in load_levels:
            row = [
                results_by_load[load].get('stutter_reduction_pct', 0),
                results_by_load[load].get('wt_improvement_pct', 0),
                results_by_load[load].get('tat_change_pct', 0),
            ]
            data.append(row)
        
        if SEABORN_AVAILABLE:
            sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                       xticklabels=metrics, yticklabels=load_levels,
                       center=0, ax=ax, cbar_kws={'label': 'Improvement %'})
        else:
            im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
            
            # Add text annotations
            for i in range(len(load_levels)):
                for j in range(len(metrics)):
                    ax.text(j, i, f'{data[i][j]:.1f}', ha='center', va='center')
            
            ax.set_xticks(range(len(metrics)))
            ax.set_yticks(range(len(load_levels)))
            ax.set_xticklabels(metrics)
            ax.set_yticklabels(load_levels)
            plt.colorbar(im, ax=ax, label='Improvement %')
        
        ax.set_title(title)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Load Level')
        
        plt.tight_layout()
        
        if save_as:
            fig.savefig(os.path.join(self.output_dir, save_as))
        
        return fig
    
    def plot_context_switches(
        self,
        rr_cs: List[int],
        mgrr_cs: List[int],
        title: str = "Context Switch Overhead Comparison",
        save_as: str = "context_switches.png"
    ) -> plt.Figure:
        """
        Compare context switch counts between algorithms.
        
        Args:
            rr_cs: RR context switch counts
            mgrr_cs: MG-RR context switch counts
            title: Plot title
            save_as: Filename
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Violin plot for better distribution visualization
        if SEABORN_AVAILABLE:
            import pandas as pd
            data = pd.DataFrame({
                'Algorithm': ['Standard RR'] * len(rr_cs) + ['MG-RR'] * len(mgrr_cs),
                'Context Switches': list(rr_cs) + list(mgrr_cs)
            })
            sns.violinplot(data=data, x='Algorithm', y='Context Switches', ax=ax,
                          palette=[COLORS['rr'], COLORS['mgrr']])
        else:
            bp = ax.boxplot([rr_cs, mgrr_cs], patch_artist=True)
            for patch, color in zip(bp['boxes'], [COLORS['rr'], COLORS['mgrr']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticklabels(['Standard RR', 'MG-RR'])
        
        ax.set_ylabel('Context Switches per Simulation')
        ax.set_title(title)
        
        # Add means
        mean_rr = sum(rr_cs) / len(rr_cs)
        mean_mgrr = sum(mgrr_cs) / len(mgrr_cs)
        pct_change = (mean_mgrr - mean_rr) / mean_rr * 100 if mean_rr else 0
        
        ax.text(0.02, 0.98, f'RR mean: {mean_rr:.1f}\nMG-RR mean: {mean_mgrr:.1f}\nChange: {pct_change:+.1f}%',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_as:
            fig.savefig(os.path.join(self.output_dir, save_as))
        
        return fig
    
    def create_all_figures(
        self,
        paired_data: Dict[str, Tuple[List[float], List[float]]],
        show: bool = False
    ) -> List[plt.Figure]:
        """
        Generate all figures for the research paper.
        
        Args:
            paired_data: Dict from MonteCarloRunner.get_paired_arrays()
            show: Whether to display figures
            
        Returns:
            List of Figure objects
        """
        figures = []
        
        # 1. Waiting Time Box Plot
        if 'avg_wt' in paired_data:
            rr, mgrr = paired_data['avg_wt']
            fig = self.plot_waiting_time_boxplot(rr, mgrr)
            figures.append(fig)
        
        # 2. Stutter Comparison
        if 'total_stutter' in paired_data:
            rr, mgrr = paired_data['total_stutter']
            fig = self.plot_stutter_comparison(
                [int(x) for x in rr], [int(x) for x in mgrr]
            )
            figures.append(fig)
        
        # 3. TAT Distribution
        if 'avg_tat' in paired_data:
            rr, mgrr = paired_data['avg_tat']
            fig = self.plot_tat_distribution(rr, mgrr)
            figures.append(fig)
        
        # 4. Trade-off Analysis
        if all(k in paired_data for k in ['avg_wt_interactive', 'avg_wt_batch']):
            fig = self.plot_interactive_vs_batch_tradeoff(
                paired_data['avg_wt_interactive'][0],
                paired_data['avg_wt_interactive'][1],
                paired_data['avg_wt_batch'][0],
                paired_data['avg_wt_batch'][1]
            )
            figures.append(fig)
        
        # 5. Context Switches
        if 'context_switches' in paired_data:
            rr, mgrr = paired_data['context_switches']
            fig = self.plot_context_switches([int(x) for x in rr], [int(x) for x in mgrr])
            figures.append(fig)
        
        if show:
            plt.show()
        
        print(f"Generated {len(figures)} figures in '{self.output_dir}/'")
        return figures


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Visualization Demo ===\n")
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        exit(1)
    
    import random
    random.seed(42)
    
    n = 100
    
    # Synthetic data
    rr_wt = [random.uniform(25, 45) for _ in range(n)]
    mgrr_wt = [w + random.uniform(-3, 2) for w in rr_wt]
    
    rr_stutters = [random.randint(3, 10) for _ in range(n)]
    mgrr_stutters = [max(0, s - random.randint(1, 4)) for s in rr_stutters]
    
    rr_tat = [random.uniform(50, 90) for _ in range(n)]
    mgrr_tat = [t + random.uniform(-5, 5) for t in rr_tat]
    
    # Create visualizer
    viz = ResultsVisualizer(output_dir="demo_figures")
    
    # Generate figures
    viz.plot_waiting_time_boxplot(rr_wt, mgrr_wt)
    viz.plot_stutter_comparison(rr_stutters, mgrr_stutters)
    viz.plot_tat_distribution(rr_tat, mgrr_tat)
    
    print("Demo figures saved to 'demo_figures/' directory")
