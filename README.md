# Portfolio Optimization using Metaheuristics

This repository contains the implementation and comparative analysis of three metaheuristic algorithms applied to the portfolio optimization problem using real-world financial data.

## 1. Problem and Constraints
* **Objective:** Maximize the **Sharpe Ratio** (finding the optimal balance between expected return and risk).
* **Constraint 1 (Full Budget):** The sum of all asset weights must equal **1.0** (100%).
* **Constraint 2 (No Short-Selling):** Asset weights must be strictly non-negative (between **0** and **1**).

## 2. Implemented Algorithms
1. **Memetic Algorithm (MA):** A genetic algorithm hybridized with *Hill Climbing* (local search) for fine-tuning continuous weights.
2. **Particle Swarm Optimization (PSO):** A swarm intelligence-based metaheuristic, highly effective for continuous optimization problems.
3. **SCA-BAS Hybrid:** An approach combining the *Sine-Cosine Algorithm* with *Beetle Antennae Search* to balance exploration and exploitation.

## 3. Methodology and Testing (Backtesting)
The experiments are designed to simulate real-world performance:
1. **Data Collection:** Fetching historical closing prices (e.g., via `yfinance`).
2. **Train/Test Split:**
   * **In-Sample (Training):** Algorithms search for optimal weights over a historical data period.
   * **Out-of-Sample (Testing):** The discovered weights are evaluated on unseen future data to validate real-world performance and prevent overfitting.
3. **Experiments:** Each algorithm is executed **30 times**. The final results compare the Best Sharpe Ratio, mean, standard deviation, and convergence time.

## 4. Project Structure
* `data/` - Historical price datasets.
* `utils.py` - Shared utility functions (data fetching, train/test split, Sharpe Ratio calculation).
* `algorithms/` - Individual algorithm implementations (`memetic_algo.py`, `pso_algo.py`, `sca_bas_algo.py`).
* `main_experiments.py` - The main orchestration script that runs all algorithms and generates the final reports.