"""
Monte Carlo Simulation Runner
-----------------------------
Executes paired simulations (RR vs MG-RR) for statistical analysis.

For each run:
1. Generate a random workload using the same seed
2. Run Standard RR simulation
3. Run MG-RR simulation
4. Collect metrics for both

This ensures paired comparison where the only variable is the scheduling algorithm.

Author: MG-RR Research Study
"""

import time
import json
import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from process import Process
from simulator import ProcessManagerSimulator
from workload_generator import WorkloadGenerator, clone_processes


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 1000
    base_seed: int = 42
    
    # Simulation parameters
    quantum: int = 3
    window_size: int = 16
    max_time: int = 500
    
    # Workload parameters
    num_processes: int = 10
    interactive_ratio: float = 0.3
    load_level: str = 'medium'
    distribution: str = 'exponential'
    max_arrival_time: int = 200
    
    # Execution
    verbose: bool = False
    parallel: bool = True
    n_workers: int = None  # None = use CPU count


@dataclass 
class RunResult:
    """Results from a single simulation run."""
    run_id: int
    seed: int
    
    # RR results
    rr_avg_wt: float
    rr_avg_tat: float
    rr_total_stutter: int
    rr_context_switches: int
    rr_avg_wt_interactive: float
    rr_avg_tat_interactive: float
    rr_avg_wt_batch: float
    rr_avg_tat_batch: float
    
    # MG-RR results
    mgrr_avg_wt: float
    mgrr_avg_tat: float
    mgrr_total_stutter: int
    mgrr_context_switches: int
    mgrr_avg_wt_interactive: float
    mgrr_avg_tat_interactive: float
    mgrr_avg_wt_batch: float
    mgrr_avg_tat_batch: float
    
    # Workload info
    process_count: int
    interactive_count: int
    batch_count: int
    total_burst: int


def run_single_simulation(args: Tuple[int, int, SimulationConfig]) -> RunResult:
    """
    Execute a single paired simulation (RR + MG-RR with same workload).
    
    This function is designed to be called in parallel.
    
    Args:
        args: Tuple of (run_id, seed, config)
        
    Returns:
        RunResult with metrics from both algorithms
    """
    run_id, seed, config = args
    
    # Generate workload with this seed
    gen = WorkloadGenerator(seed=seed)
    workload = gen.generate(
        num_processes=config.num_processes,
        interactive_ratio=config.interactive_ratio,
        load_level=config.load_level,
        distribution=config.distribution,
        max_arrival_time=config.max_arrival_time,
        window_size=config.window_size
    )
    
    total_burst = sum(p.burst_time for p in workload)
    interactive_count = sum(1 for p in workload if p.is_interactive)
    
    # --- Run Standard RR ---
    rr_procs = clone_processes(workload)
    sim_rr = ProcessManagerSimulator(
        processes=rr_procs,
        policy="rr",
        quantum=config.quantum,
        window_size=config.window_size,
        max_time=config.max_time,
        disable_window_logic=False,  # Still track stutters for comparison
        verbose=config.verbose
    )
    sim_rr.run()
    rr_results = sim_rr.get_results()
    
    # --- Run MG-RR ---
    mgrr_procs = clone_processes(workload)
    sim_mgrr = ProcessManagerSimulator(
        processes=mgrr_procs,
        policy="mgrr",
        quantum=config.quantum,
        window_size=config.window_size,
        max_time=config.max_time,
        disable_window_logic=False,
        verbose=config.verbose
    )
    sim_mgrr.run()
    mgrr_results = sim_mgrr.get_results()
    
    return RunResult(
        run_id=run_id,
        seed=seed,
        
        # RR metrics
        rr_avg_wt=rr_results['avg_waiting_time'],
        rr_avg_tat=rr_results['avg_turnaround_time'],
        rr_total_stutter=rr_results['total_stutter'],
        rr_context_switches=rr_results['total_context_switches'],
        rr_avg_wt_interactive=rr_results['avg_wt_interactive'],
        rr_avg_tat_interactive=rr_results['avg_tat_interactive'],
        rr_avg_wt_batch=rr_results['avg_wt_batch'],
        rr_avg_tat_batch=rr_results['avg_tat_batch'],
        
        # MG-RR metrics
        mgrr_avg_wt=mgrr_results['avg_waiting_time'],
        mgrr_avg_tat=mgrr_results['avg_turnaround_time'],
        mgrr_total_stutter=mgrr_results['total_stutter'],
        mgrr_context_switches=mgrr_results['total_context_switches'],
        mgrr_avg_wt_interactive=mgrr_results['avg_wt_interactive'],
        mgrr_avg_tat_interactive=mgrr_results['avg_tat_interactive'],
        mgrr_avg_wt_batch=mgrr_results['avg_wt_batch'],
        mgrr_avg_tat_batch=mgrr_results['avg_tat_batch'],
        
        # Workload info
        process_count=len(workload),
        interactive_count=interactive_count,
        batch_count=len(workload) - interactive_count,
        total_burst=total_burst
    )


class MonteCarloRunner:
    """
    Runs Monte Carlo simulation comparing RR and MG-RR algorithms.
    
    Produces paired results for statistical testing.
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize Monte Carlo runner.
        
        Args:
            config: Simulation configuration (uses defaults if None)
        """
        self.config = config or SimulationConfig()
        self.results: List[RunResult] = []
        self.execution_time: float = 0
    
    def run(self, show_progress: bool = True) -> List[RunResult]:
        """
        Execute all Monte Carlo simulations.
        
        Args:
            show_progress: Print progress updates
            
        Returns:
            List of RunResult objects
        """
        start_time = time.time()
        
        # Prepare arguments for each run
        args_list = [
            (i, self.config.base_seed + i, self.config)
            for i in range(self.config.n_simulations)
        ]
        
        if self.config.parallel and self.config.n_simulations > 10:
            # Parallel execution
            n_workers = self.config.n_workers or multiprocessing.cpu_count()
            if show_progress:
                print(f"Running {self.config.n_simulations} simulations in parallel ({n_workers} workers)...")
            
            self.results = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(run_single_simulation, args): args[0] 
                          for args in args_list}
                
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)
                    completed += 1
                    
                    if show_progress and completed % 100 == 0:
                        print(f"  Completed {completed}/{self.config.n_simulations}")
            
            # Sort by run_id
            self.results.sort(key=lambda r: r.run_id)
        else:
            # Sequential execution
            if show_progress:
                print(f"Running {self.config.n_simulations} simulations sequentially...")
            
            self.results = []
            for i, args in enumerate(args_list):
                result = run_single_simulation(args)
                self.results.append(result)
                
                if show_progress and (i + 1) % 100 == 0:
                    print(f"  Completed {i + 1}/{self.config.n_simulations}")
        
        self.execution_time = time.time() - start_time
        
        if show_progress:
            print(f"Completed in {self.execution_time:.2f} seconds")
        
        return self.results
    
    def get_paired_arrays(self) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Get paired arrays for statistical testing.
        
        Returns:
            Dict mapping metric name to (rr_values, mgrr_values) tuples
        """
        metrics = {
            'avg_wt': ([], []),
            'avg_tat': ([], []),
            'total_stutter': ([], []),
            'context_switches': ([], []),
            'avg_wt_interactive': ([], []),
            'avg_tat_interactive': ([], []),
            'avg_wt_batch': ([], []),
            'avg_tat_batch': ([], []),
        }
        
        for r in self.results:
            metrics['avg_wt'][0].append(r.rr_avg_wt)
            metrics['avg_wt'][1].append(r.mgrr_avg_wt)
            
            metrics['avg_tat'][0].append(r.rr_avg_tat)
            metrics['avg_tat'][1].append(r.mgrr_avg_tat)
            
            metrics['total_stutter'][0].append(r.rr_total_stutter)
            metrics['total_stutter'][1].append(r.mgrr_total_stutter)
            
            metrics['context_switches'][0].append(r.rr_context_switches)
            metrics['context_switches'][1].append(r.mgrr_context_switches)
            
            metrics['avg_wt_interactive'][0].append(r.rr_avg_wt_interactive)
            metrics['avg_wt_interactive'][1].append(r.mgrr_avg_wt_interactive)
            
            metrics['avg_tat_interactive'][0].append(r.rr_avg_tat_interactive)
            metrics['avg_tat_interactive'][1].append(r.mgrr_avg_tat_interactive)
            
            metrics['avg_wt_batch'][0].append(r.rr_avg_wt_batch)
            metrics['avg_wt_batch'][1].append(r.mgrr_avg_wt_batch)
            
            metrics['avg_tat_batch'][0].append(r.rr_avg_tat_batch)
            metrics['avg_tat_batch'][1].append(r.mgrr_avg_tat_batch)
        
        return metrics
    
    def save_results(self, filepath: str = "monte_carlo_results.csv"):
        """
        Save results to CSV file for external analysis.
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.results:
            raise ValueError("No results to save. Run simulation first.")
        
        fieldnames = [
            'run_id', 'seed',
            'rr_avg_wt', 'rr_avg_tat', 'rr_total_stutter', 'rr_context_switches',
            'rr_avg_wt_interactive', 'rr_avg_tat_interactive',
            'rr_avg_wt_batch', 'rr_avg_tat_batch',
            'mgrr_avg_wt', 'mgrr_avg_tat', 'mgrr_total_stutter', 'mgrr_context_switches',
            'mgrr_avg_wt_interactive', 'mgrr_avg_tat_interactive',
            'mgrr_avg_wt_batch', 'mgrr_avg_tat_batch',
            'process_count', 'interactive_count', 'batch_count', 'total_burst'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in self.results:
                writer.writerow({
                    'run_id': r.run_id,
                    'seed': r.seed,
                    'rr_avg_wt': r.rr_avg_wt,
                    'rr_avg_tat': r.rr_avg_tat,
                    'rr_total_stutter': r.rr_total_stutter,
                    'rr_context_switches': r.rr_context_switches,
                    'rr_avg_wt_interactive': r.rr_avg_wt_interactive,
                    'rr_avg_tat_interactive': r.rr_avg_tat_interactive,
                    'rr_avg_wt_batch': r.rr_avg_wt_batch,
                    'rr_avg_tat_batch': r.rr_avg_tat_batch,
                    'mgrr_avg_wt': r.mgrr_avg_wt,
                    'mgrr_avg_tat': r.mgrr_avg_tat,
                    'mgrr_total_stutter': r.mgrr_total_stutter,
                    'mgrr_context_switches': r.mgrr_context_switches,
                    'mgrr_avg_wt_interactive': r.mgrr_avg_wt_interactive,
                    'mgrr_avg_tat_interactive': r.mgrr_avg_tat_interactive,
                    'mgrr_avg_wt_batch': r.mgrr_avg_wt_batch,
                    'mgrr_avg_tat_batch': r.mgrr_avg_tat_batch,
                    'process_count': r.process_count,
                    'interactive_count': r.interactive_count,
                    'batch_count': r.batch_count,
                    'total_burst': r.total_burst
                })
        
        print(f"Results saved to {filepath}")
    
    def summary(self) -> Dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}
        
        n = len(self.results)
        
        def avg(values):
            return sum(values) / len(values) if values else 0
        
        rr_stutters = [r.rr_total_stutter for r in self.results]
        mgrr_stutters = [r.mgrr_total_stutter for r in self.results]
        
        rr_wt = [r.rr_avg_wt for r in self.results]
        mgrr_wt = [r.mgrr_avg_wt for r in self.results]
        
        rr_tat = [r.rr_avg_tat for r in self.results]
        mgrr_tat = [r.mgrr_avg_tat for r in self.results]
        
        return {
            'n_simulations': n,
            'execution_time': self.execution_time,
            'config': {
                'quantum': self.config.quantum,
                'window_size': self.config.window_size,
                'num_processes': self.config.num_processes,
                'interactive_ratio': self.config.interactive_ratio,
                'load_level': self.config.load_level
            },
            'rr': {
                'mean_stutter': avg(rr_stutters),
                'total_stutters': sum(rr_stutters),
                'mean_wt': avg(rr_wt),
                'mean_tat': avg(rr_tat)
            },
            'mgrr': {
                'mean_stutter': avg(mgrr_stutters),
                'total_stutters': sum(mgrr_stutters),
                'mean_wt': avg(mgrr_wt),
                'mean_tat': avg(mgrr_tat)
            },
            'improvement': {
                'stutter_reduction_pct': (
                    (avg(rr_stutters) - avg(mgrr_stutters)) / avg(rr_stutters) * 100
                    if avg(rr_stutters) > 0 else 0
                )
            }
        }


if __name__ == "__main__":
    # Demo: Run a small Monte Carlo simulation
    print("=== Monte Carlo Simulation Demo ===\n")
    
    config = SimulationConfig(
        n_simulations=50,  # Small for demo
        base_seed=42,
        num_processes=8,
        interactive_ratio=0.3,
        load_level='medium',
        parallel=False,  # Sequential for cleaner demo output
        verbose=False
    )
    
    runner = MonteCarloRunner(config)
    results = runner.run()
    
    print("\n=== Summary ===")
    summary = runner.summary()
    print(f"Simulations: {summary['n_simulations']}")
    print(f"Execution time: {summary['execution_time']:.2f}s")
    print(f"\nRR Mean Stutter: {summary['rr']['mean_stutter']:.2f}")
    print(f"MG-RR Mean Stutter: {summary['mgrr']['mean_stutter']:.2f}")
    print(f"Stutter Reduction: {summary['improvement']['stutter_reduction_pct']:.1f}%")
    
    # Save results
    runner.save_results("demo_results.csv")
