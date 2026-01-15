#!/usr/bin/env python3
"""
MG-RR Research Study - Main Entry Point
========================================

This script runs the complete Monte Carlo simulation comparing:
- Standard Round Robin (RR)
- Minimum Guaranteed Round Robin (MG-RR)

Phases:
1. Monte Carlo Simulation (N=1000 runs)
2. Statistical Analysis (Paired T-Test, Cohen's d)
3. Visualization (Academic-grade plots)
4. Report Generation

Usage:
    python main.py                    # Full simulation (N=1000)
    python main.py --demo             # Quick demo (N=50)
    python main.py --n 500            # Custom number of runs
    python main.py --static           # Original static scenario

Author: MG-RR Research Study
"""

import argparse
import sys
import os
from datetime import datetime

from process import Process
from simulator import ProcessManagerSimulator
from workload_generator import WorkloadGenerator, clone_processes
from monte_carlo import MonteCarloRunner, SimulationConfig
from analysis import StatisticalAnalysis
from visualization import ResultsVisualizer, MATPLOTLIB_AVAILABLE


def run_static_scenario():
    """
    Run the original static scenario for verification.
    This is the same as the original main.py behavior.
    """
    print("=" * 70)
    print("STATIC SCENARIO TEST")
    print("=" * 70)
    
    # Scenario:
    # - Game: interactive, needs 5 ticks per 16-tick window
    # - Spotlight: interactive, needs 3 ticks per 16-tick window
    # - Render: CPU-bound batch job
    base = [
        Process(pid=1, name="Game",      arrival_time=0, burst_time=40, is_interactive=True,  min_cpu_per_window=5),
        Process(pid=2, name="Spotlight", arrival_time=2, burst_time=20, is_interactive=True,  min_cpu_per_window=3),
        Process(pid=3, name="Render",    arrival_time=1, burst_time=60, is_interactive=False, min_cpu_per_window=0),
    ]

    # 1) Standard Round Robin
    print("\n--- Standard Round Robin ---")
    rr_procs = clone_processes(base)
    sim_rr = ProcessManagerSimulator(
        processes=rr_procs,
        policy="rr",
        quantum=3,
        window_size=16,
        tick_ms=1,
        max_time=500,
        verbose=True
    )
    sim_rr.run()
    rr_results = sim_rr.get_results()

    print("\n" + "=" * 70 + "\n")

    # 2) MG-RR (Minimum Guaranteed Round Robin)
    print("--- MG-RR (Minimum Guaranteed Round Robin) ---")
    mgrr_procs = clone_processes(base)
    sim_mgrr = ProcessManagerSimulator(
        processes=mgrr_procs,
        policy="mgrr",
        quantum=3,
        window_size=16,
        tick_ms=1,
        max_time=500,
        verbose=True
    )
    sim_mgrr.run()
    mgrr_results = sim_mgrr.get_results()

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'RR':>15} {'MG-RR':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Total Stutter':<25} {rr_results['total_stutter']:>15} {mgrr_results['total_stutter']:>15} "
          f"{rr_results['total_stutter'] - mgrr_results['total_stutter']:>15}")
    print(f"{'Avg Waiting Time':<25} {rr_results['avg_waiting_time']:>15.2f} {mgrr_results['avg_waiting_time']:>15.2f} "
          f"{rr_results['avg_waiting_time'] - mgrr_results['avg_waiting_time']:>15.2f}")
    print(f"{'Avg Turnaround Time':<25} {rr_results['avg_turnaround_time']:>15.2f} {mgrr_results['avg_turnaround_time']:>15.2f} "
          f"{rr_results['avg_turnaround_time'] - mgrr_results['avg_turnaround_time']:>15.2f}")
    print(f"{'Context Switches':<25} {rr_results['total_context_switches']:>15} {mgrr_results['total_context_switches']:>15} "
          f"{rr_results['total_context_switches'] - mgrr_results['total_context_switches']:>15}")


def run_monte_carlo_simulation(n_simulations: int = 1000, output_dir: str = "results"):
    """
    Run full Monte Carlo simulation with statistical analysis and visualization.
    
    Args:
        n_simulations: Number of simulation runs
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("MG-RR MONTE CARLO RESEARCH STUDY")
    print("=" * 70)
    print(f"Simulations: {n_simulations}")
    print(f"Output Directory: {output_dir}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ========================================
    # PHASE 1: Monte Carlo Simulation
    # ========================================
    print("\n[PHASE 1] Running Monte Carlo Simulation...")
    
    config = SimulationConfig(
        n_simulations=n_simulations,
        base_seed=42,
        quantum=3,
        window_size=16,
        max_time=500,
        num_processes=10,
        interactive_ratio=0.3,
        load_level='medium',
        distribution='exponential',
        max_arrival_time=200,
        verbose=False,
        parallel=n_simulations > 50,
        n_workers=None  # Auto-detect
    )
    
    runner = MonteCarloRunner(config)
    results = runner.run(show_progress=True)
    
    # Save raw results
    csv_path = os.path.join(output_dir, "monte_carlo_results.csv")
    runner.save_results(csv_path)
    
    # Get paired data for analysis
    paired_data = runner.get_paired_arrays()
    
    # Quick summary
    summary = runner.summary()
    print(f"\n[Quick Summary]")
    print(f"  RR Total Stutters: {summary['rr']['total_stutters']:,}")
    print(f"  MG-RR Total Stutters: {summary['mgrr']['total_stutters']:,}")
    print(f"  Stutter Reduction: {summary['improvement']['stutter_reduction_pct']:.1f}%")
    
    # ========================================
    # PHASE 2: Statistical Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("[PHASE 2] Statistical Analysis")
    print("=" * 70)
    
    analyzer = StatisticalAnalysis()
    analysis_results = analyzer.analyze_all(paired_data)
    
    # Generate and save report
    report = analyzer.generate_report(analysis_results, "MG-RR vs Standard RR: Statistical Analysis")
    print(report)
    
    report_path = os.path.join(output_dir, "statistical_analysis.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Print key findings
    print("\n" + "-" * 50)
    print("KEY FINDINGS:")
    print("-" * 50)
    
    if 'total_stutter' in analysis_results:
        stutter_analysis = analysis_results['total_stutter']
        ttest = stutter_analysis['ttest']
        effect = stutter_analysis['effect_size']
        
        print(f"\n1. STUTTER COUNT (Primary Hypothesis)")
        print(f"   H0: No difference in stutter count between RR and MG-RR")
        print(f"   H1: MG-RR significantly reduces stutter count")
        print(f"   Result: t({ttest.degrees_of_freedom}) = {ttest.t_statistic:.4f}, p = {ttest.p_value:.6f}")
        
        if ttest.is_significant():
            print(f"   *** SIGNIFICANT at α = 0.05 ***")
            if effect.cohens_d > 0:
                print(f"   Effect Size: {effect}")
                print(f"   Conclusion: RR produces significantly MORE stutters than MG-RR")
        else:
            print(f"   Not significant at α = 0.05")
    
    if 'avg_wt_batch' in analysis_results:
        batch_analysis = analysis_results['avg_wt_batch']
        ttest = batch_analysis['ttest']
        
        print(f"\n2. BATCH PROCESS WAITING TIME (Trade-off Analysis)")
        print(f"   Question: Does MG-RR hurt batch job performance?")
        print(f"   RR Mean WT: {batch_analysis['rr_stats'].mean:.2f}")
        print(f"   MG-RR Mean WT: {batch_analysis['mgrr_stats'].mean:.2f}")
        print(f"   Change: {batch_analysis['improvement']['percentage']:+.2f}%")
        
        if ttest and ttest.is_significant():
            if batch_analysis['improvement']['absolute'] < 0:
                print(f"   Note: MG-RR causes HIGHER waiting time for batch (trade-off)")
            else:
                print(f"   Note: MG-RR does NOT significantly hurt batch performance")
    
    # ========================================
    # PHASE 3: Visualization
    # ========================================
    if MATPLOTLIB_AVAILABLE:
        print("\n" + "=" * 70)
        print("[PHASE 3] Generating Visualizations")
        print("=" * 70)
        
        figures_dir = os.path.join(output_dir, "figures")
        viz = ResultsVisualizer(output_dir=figures_dir)
        figures = viz.create_all_figures(paired_data, show=False)
        
        print(f"Figures saved to: {figures_dir}/")
    else:
        print("\n[PHASE 3] Skipping visualizations (matplotlib not installed)")
        print("  Install with: pip install matplotlib seaborn")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(f"Duration: {runner.execution_time:.2f} seconds")
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")
    if MATPLOTLIB_AVAILABLE:
        print(f"  - {figures_dir}/*.png")
    
    print("\nNext Steps:")
    print("  1. Review statistical_analysis.txt for detailed results")
    print("  2. Check figures/ for publication-ready plots")
    print("  3. Use results for Phase 5: Academic Paper Composition")
    
    return runner, analysis_results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MG-RR Research Study - Monte Carlo Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                # Full simulation (N=1000)
  python main.py --demo         # Quick demo (N=50)
  python main.py --n 500        # Custom simulation count
  python main.py --static       # Original static scenario
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Run quick demo with N=50 simulations')
    parser.add_argument('--static', action='store_true',
                        help='Run original static scenario (no Monte Carlo)')
    parser.add_argument('-n', '--n-simulations', type=int, default=1000,
                        help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    if args.static:
        run_static_scenario()
    elif args.demo:
        run_monte_carlo_simulation(n_simulations=50, output_dir=args.output)
    else:
        run_monte_carlo_simulation(n_simulations=args.n_simulations, output_dir=args.output)


if __name__ == "__main__":
    main()
