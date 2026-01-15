"""
Workload Generator for Monte Carlo Simulation
----------------------------------------------
Generates random workloads with configurable parameters for MG-RR vs RR comparison.

Distributions:
- Arrival Times: Poisson process (exponential inter-arrival times)
- Burst Times: Exponential or Normal distribution
- Mix: Interactive vs CPU-Bound (batch) processes
- Load Levels: Low, Medium, High (affects CPU utilization)

Author: MG-RR Research Study
"""

import random
import math
from typing import List, Tuple
from process import Process


class WorkloadGenerator:
    """
    Monte Carlo simülasyonu için rastgele iş yükü üretici.
    Aynı seed ile aynı workload'u tekrar üretebilir (RR vs MG-RR karşılaştırması için).
    """
    
    # Load level presets (affects total burst vs simulation time)
    LOAD_PRESETS = {
        'low': {
            'arrival_rate': 0.3,      # λ: average arrivals per tick
            'burst_mean': 15,         # μ: average burst time
            'burst_std': 5,
            'description': 'CPU < 50% utilized'
        },
        'medium': {
            'arrival_rate': 0.2,
            'burst_mean': 25,
            'burst_std': 8,
            'description': 'CPU ~70-80% utilized'
        },
        'high': {
            'arrival_rate': 0.15,
            'burst_mean': 40,
            'burst_std': 12,
            'description': 'CPU >90% utilized (saturation)'
        }
    }
    
    def __init__(self, seed: int = None):
        """
        Initialize generator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible workloads
        """
        self.rng = random.Random(seed)
        self.seed = seed
    
    def reset(self, seed: int = None):
        """Reset the generator with a new seed."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
    
    def _poisson_arrivals(self, n: int, rate: float, max_time: int) -> List[int]:
        """
        Generate n arrival times using Poisson process.
        Inter-arrival times are exponentially distributed with mean 1/rate.
        
        Args:
            n: Number of arrivals to generate
            rate: λ (lambda) - average arrivals per time unit
            max_time: Maximum arrival time allowed
            
        Returns:
            List of arrival times (sorted)
        """
        arrivals = []
        current_time = 0
        
        for _ in range(n):
            # Exponential inter-arrival time with mean = 1/rate
            inter_arrival = self.rng.expovariate(rate)
            current_time += inter_arrival
            
            if current_time > max_time:
                break
                
            arrivals.append(int(current_time))
        
        return arrivals
    
    def _generate_burst_time(
        self, 
        mean: float, 
        std: float = None,
        distribution: str = 'exponential',
        min_burst: int = 3,
        max_burst: int = 100
    ) -> int:
        """
        Generate a single burst time.
        
        Args:
            mean: Mean burst time
            std: Standard deviation (for normal distribution)
            distribution: 'exponential' or 'normal'
            min_burst: Minimum allowed burst time
            max_burst: Maximum allowed burst time
            
        Returns:
            Integer burst time
        """
        if distribution == 'exponential':
            burst = self.rng.expovariate(1.0 / mean)
        elif distribution == 'normal':
            std = std or (mean * 0.3)  # Default std = 30% of mean
            burst = self.rng.gauss(mean, std)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Clamp to valid range
        burst = max(min_burst, min(max_burst, int(burst)))
        return burst
    
    def _assign_interactive_properties(
        self,
        process: Process,
        window_size: int = 16
    ) -> None:
        """
        Assign QoS properties to an interactive process.
        
        Interactive processes need a minimum CPU time per window to maintain
        responsiveness (e.g., 60 FPS = 16ms window).
        
        Args:
            process: Process to modify (mutated in place)
            window_size: QoS window size in ticks
        """
        process.is_interactive = True
        
        # min_cpu_per_window: typically 20-40% of window for interactive apps
        # This ensures they can maintain their frame rate or UI responsiveness
        min_ratio = self.rng.uniform(0.2, 0.4)
        process.min_cpu_per_window = max(2, int(window_size * min_ratio))
    
    def generate(
        self,
        num_processes: int,
        interactive_ratio: float = 0.3,
        load_level: str = 'medium',
        distribution: str = 'exponential',
        window_size: int = 16,
        max_arrival_time: int = 200,
        custom_params: dict = None
    ) -> List[Process]:
        """
        Generate a random workload.
        
        Args:
            num_processes: Total number of processes to generate
            interactive_ratio: Fraction of processes that are interactive (0.0 - 1.0)
            load_level: 'low', 'medium', or 'high' (preset configurations)
            distribution: 'exponential' or 'normal' for burst times
            window_size: QoS window size for interactive processes
            max_arrival_time: Maximum time for process arrivals
            custom_params: Override preset params (arrival_rate, burst_mean, burst_std)
            
        Returns:
            List of Process objects ready for simulation
        """
        # Get load preset or use custom params
        if load_level in self.LOAD_PRESETS:
            params = self.LOAD_PRESETS[load_level].copy()
        else:
            params = self.LOAD_PRESETS['medium'].copy()
        
        if custom_params:
            params.update(custom_params)
        
        arrival_rate = params['arrival_rate']
        burst_mean = params['burst_mean']
        burst_std = params.get('burst_std', burst_mean * 0.3)
        
        # Generate arrival times using Poisson process
        arrivals = self._poisson_arrivals(num_processes, arrival_rate, max_arrival_time)
        
        # If we didn't get enough arrivals, pad with arrivals at max_time
        while len(arrivals) < num_processes:
            arrivals.append(max_arrival_time)
        
        # Determine which processes are interactive
        num_interactive = int(num_processes * interactive_ratio)
        interactive_indices = set(self.rng.sample(range(num_processes), num_interactive))
        
        # Generate processes
        processes = []
        for i in range(num_processes):
            is_interactive = i in interactive_indices
            
            # Interactive processes tend to have shorter bursts (responsive apps)
            if is_interactive:
                burst = self._generate_burst_time(
                    mean=burst_mean * 0.6,  # Interactive: 60% of mean burst
                    std=burst_std * 0.5,
                    distribution=distribution,
                    min_burst=5,
                    max_burst=60
                )
                name = f"Interactive_{i}"
            else:
                burst = self._generate_burst_time(
                    mean=burst_mean,
                    std=burst_std,
                    distribution=distribution,
                    min_burst=10,
                    max_burst=150
                )
                name = f"Batch_{i}"
            
            process = Process(
                pid=i + 1,
                name=name,
                arrival_time=arrivals[i],
                burst_time=burst,
                is_interactive=False,
                min_cpu_per_window=0
            )
            
            if is_interactive:
                self._assign_interactive_properties(process, window_size)
            
            processes.append(process)
        
        return processes
    
    def generate_scenario(
        self,
        scenario_name: str,
        window_size: int = 16
    ) -> List[Process]:
        """
        Generate pre-defined test scenarios.
        
        Args:
            scenario_name: Name of the scenario to generate
            window_size: QoS window size
            
        Returns:
            List of Process objects
        """
        scenarios = {
            'balanced': {
                'num_processes': 10,
                'interactive_ratio': 0.3,
                'load_level': 'medium'
            },
            'interactive_heavy': {
                'num_processes': 12,
                'interactive_ratio': 0.6,
                'load_level': 'medium'
            },
            'batch_heavy': {
                'num_processes': 10,
                'interactive_ratio': 0.1,
                'load_level': 'high'
            },
            'stress_test': {
                'num_processes': 20,
                'interactive_ratio': 0.4,
                'load_level': 'high'
            },
            'light_load': {
                'num_processes': 6,
                'interactive_ratio': 0.3,
                'load_level': 'low'
            }
        }
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")
        
        config = scenarios[scenario_name]
        return self.generate(
            num_processes=config['num_processes'],
            interactive_ratio=config['interactive_ratio'],
            load_level=config['load_level'],
            window_size=window_size
        )


def clone_processes(processes: List[Process]) -> List[Process]:
    """
    Create deep copies of processes for running multiple simulations.
    
    Simulations modify process objects (remaining_time, state, etc.),
    so we need fresh copies for each simulation run.
    
    Args:
        processes: List of Process objects to clone
        
    Returns:
        List of new Process objects with same initial values
    """
    return [
        Process(
            pid=p.pid,
            name=p.name,
            arrival_time=p.arrival_time,
            burst_time=p.burst_time,
            priority=p.priority,
            is_interactive=p.is_interactive,
            min_cpu_per_window=p.min_cpu_per_window,
        )
        for p in processes
    ]


if __name__ == "__main__":
    # Demo: Generate and display workloads
    print("=== Workload Generator Demo ===\n")
    
    # Seeded generator for reproducibility
    gen = WorkloadGenerator(seed=42)
    
    # Generate different load levels
    for load_level in ['low', 'medium', 'high']:
        print(f"\n--- {load_level.upper()} Load ---")
        workload = gen.generate(
            num_processes=8,
            interactive_ratio=0.3,
            load_level=load_level
        )
        
        total_burst = sum(p.burst_time for p in workload)
        interactive_count = sum(1 for p in workload if p.is_interactive)
        
        print(f"Processes: {len(workload)} ({interactive_count} interactive)")
        print(f"Total burst: {total_burst} ticks")
        print(f"Arrivals: {[p.arrival_time for p in workload]}")
        print(f"Bursts: {[p.burst_time for p in workload]}")
        
        # Reset for next iteration with same seed
        gen.reset(seed=42 + hash(load_level))
