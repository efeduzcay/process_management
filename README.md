# Process Management Simulation: RR vs MG-RR

A Python-based process scheduling simulation and research project that compares **Standard Round Robin (RR)** with **Minimum Guaranteed Round Robin (MG-RR)** under dynamic workloads.

This project focuses on how scheduling policies affect **interactive responsiveness**, **waiting time**, **turnaround time**, **context switches**, and **stutter count**. It includes:

- A custom **process scheduler simulator**
- **Monte Carlo experiments** for large-scale comparison
- **Statistical analysis** with paired testing
- **Visualization outputs** for result interpretation

---

## Overview

Traditional Round Robin scheduling is fair, but it may not always provide the best responsiveness for interactive processes such as games, search tools, or UI-driven applications.

This project introduces **MG-RR (Minimum Guaranteed Round Robin)**, an approach that keeps the fairness of RR while giving interactive processes a minimum CPU guarantee within a scheduling window.

The goal is to evaluate whether MG-RR can:

- Reduce **stutter** for interactive workloads
- Improve perceived responsiveness
- Preserve reasonable performance for batch workloads

---

## Features

- **Standard Round Robin (RR)** simulation
- **MG-RR scheduler** with minimum CPU guarantee logic for interactive processes
- **Static scenario testing** for direct behavior comparison
- **Monte Carlo simulation runner** for repeated experiments
- **Paired statistical analysis**
- **Effect size calculation**
- **Automatic result export**
- **Visualization support** with generated figures

---

## Project Structure

```bash
process_management/
│
├── main.py                  # Main entry point
├── process.py               # Process model and state tracking
├── scheduler.py             # RR and MG-RR dispatcher logic
├── simulator.py             # Core simulation engine
├── workload_generator.py    # Random workload generation
├── monte_carlo.py           # Monte Carlo experiment runner
├── analysis.py              # Statistical analysis tools
├── visualization.py         # Plot generation
│
├── context_switches.png
├── stutter_comparison.png
├── tat_distribution.png
├── tradeoff_analysis.png
└── waiting_time_boxplot.png
