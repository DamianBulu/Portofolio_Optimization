import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Project path management
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_processed_data, calculate_sharpe_ratio, calculate_markowitz_benchmark, validate_and_normalize_weights
from algorithms.pso_algo import PSOAlgorithm

def run_pso_experiments():
    # Setup results directory
    results_dir = os.path.join('main_experiments', 'results', 'results_pso')
    os.makedirs(results_dir, exist_ok=True)

    print("1. Loading processed datasets...")
    # This utility reads mu_train.csv, sigma_train.csv, and test_returns.csv
    mu_train, sigma_train, test_returns = get_processed_data()

    # Process test data for out-of-sample evaluation (Annualised)
    mu_test = test_returns.mean().values * 252
    sigma_test = test_returns.cov().values * 252

    print("2. Computing Benchmark (Exact Markowitz Solution)...")
    markowitz_weights, markowitz_train_sharpe = calculate_markowitz_benchmark(mu_train, sigma_train)
    markowitz_test_sharpe = calculate_sharpe_ratio(markowitz_weights, mu_test, sigma_test)

    print(f"Mathematical Maximums (Benchmark):")
    print(f"    Train Sharpe: {markowitz_train_sharpe:.4f}")
    print(f"    Test Sharpe:  {markowitz_test_sharpe:.4f}")

    # Define Experiment Configurations
    experiments = [
        {"name": "PSO Fast",       "pop": 20, "iter": 50,  "w": 0.7, "c1": 1.2, "c2": 1.2},
        {"name": "PSO Standard",   "pop": 40, "iter": 100, "w": 0.7298, "c1": 1.4962, "c2": 1.4962},
        {"name": "PSO Aggressive", "pop": 70, "iter": 200, "w": 0.8, "c1": 2.0, "c2": 2.0}
    ]

    results_summary = [{
        "Experiment Name": "Benchmark Markowitz",
        "Algorithm Settings": "Mathematical Optimization",
        "Sharpe Ratio performance in Train": round(markowitz_train_sharpe, 4),
        "Sharpe Ratio performance in Test": round(markowitz_test_sharpe, 4),
        "Evolution of performance(Test vs Train)": f"{round(markowitz_test_sharpe - markowitz_train_sharpe, 4)}"
    }]

    experiments_names = ["Markowitz"]
    train_scores = [markowitz_train_sharpe]
    test_scores = [markowitz_test_sharpe]
    convergence_histories = {}

    print("\n3. Running Particle Swarm Optimisation Experiments")
    for exp in experiments:
        print(f"Running {exp['name']}...")
        
        # Initialise PSO Class
        algo = PSOAlgorithm(
            mu=mu_train,
            sigma=sigma_train,
            population_size=exp['pop'],
            iterations=exp['iter'],
            w=exp['w'],
            c1=exp['c1'],
            c2=exp['c2'],
            early_stopping=False # Disabled for plotting full convergence
        )

        # Execute Algorithm
        best_weights, train_sharpe, history = algo.run_pso(verbose=False)
        
        # Out-of-sample performance
        test_sharpe = calculate_sharpe_ratio(best_weights, mu_test, sigma_test)

        # Store results for report and plots
        convergence_histories[exp['name']] = history
        experiments_names.append(exp['name'])
        train_scores.append(train_sharpe)
        test_scores.append(test_sharpe)

        results_summary.append({
            "Experiment Name": exp['name'],
            "Algorithm Settings": f"Pop:{exp['pop']} Iter:{exp['iter']} w:{exp['w']} c1:{exp['c1']}",
            "Sharpe Ratio performance in Train": round(train_sharpe, 4),
            "Sharpe Ratio performance in Test": round(test_sharpe, 4),
            "Evolution of performance(Test vs Train)": f"{round(test_sharpe - train_sharpe, 4)}"
        })

        print(f"   Done. Train Sharpe: {train_sharpe:.4f} | Test Sharpe: {test_sharpe:.4f}")

    # Export to CSV
    df = pd.DataFrame(results_summary)
    csv_path = os.path.join(results_dir, 'report_pso_experiments.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nReport saved to: {csv_path}')

    # Visualisation
    print('Generating visual captures...')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot 1: Convergence Curves
    plt.figure(figsize=(10, 6))
    plt.axhline(y=markowitz_train_sharpe, color='black', linestyle='--', label='Mathematical Max (Markowitz)')

    for i, (name, history) in enumerate(convergence_histories.items()):
        plt.plot(history, label=name, color=colors[i % len(colors)], linewidth=2)

    plt.title("PSO Convergence Evolution (Training Data)", fontsize=14)
    plt.xlabel("Iterations / Epochs")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pso_convergence_evolution.png"))
    plt.close()

    # Plot 2: Train vs Test Comparison (Generalisation)
    x = np.arange(len(experiments_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, train_scores, width, label='In-Sample (Train)', color='#7fbc41')
    plt.bar(x + width / 2, test_scores, width, label='Out-of-Sample (Test)', color='#de77ae')

    plt.ylabel('Sharpe Ratio')
    plt.title('PSO Generalisation Performance: Train vs Test', fontsize=14)
    plt.xticks(x, experiments_names)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pso_performance_comparison.png"))

    print("All captures saved in results/results_pso/")
    # plt.show()

if __name__ == "__main__":
    run_pso_experiments()