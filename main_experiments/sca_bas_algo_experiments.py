import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_processed_data, calculate_sharpe_ratio, calculate_markowitz_benchmark, validate_and_normalize_weights
from algorithms.sca_bas_algo import bassca_portfolio_optimization


def run_sca_bas_experiments():
    results_dir = os.path.join('results', 'results_sca_bas')
    os.makedirs(results_dir, exist_ok=True)

    print("1. Loading data...")
    mu_train, sigma_train, test_returns = get_processed_data()

    mu_test = test_returns.mean().values * 252
    sigma_test = test_returns.cov().values * 252

    print("2. Computing Benchmark(Markowitz)...")
    markowitz_weights, markowitz_train_sharpe = calculate_markowitz_benchmark(mu_train, sigma_train)
    markowitz_test_sharpe = calculate_sharpe_ratio(markowitz_weights, mu_test, sigma_test)

    print(f"2. Markowitz(mathematical) results:")
    print(f"    Train:{markowitz_train_sharpe:.4f}")
    print(f"    Test:{markowitz_test_sharpe:.4f}")

    experiments = [
        {"name": "SCA-BAS Fast",       "n_agents": 15, "max_iter": 60,  "sensing_distance": 0.15, "step_size": 0.15},
        {"name": "SCA-BAS Standard",   "n_agents": 25, "max_iter": 100, "sensing_distance": 0.1,  "step_size": 0.1},
        {"name": "SCA-BAS Aggressive", "n_agents": 40, "max_iter": 150, "sensing_distance": 0.2,  "step_size": 0.2}
    ]

    results_summary = [{
        "Experiment Name": "Benchmark Markowitz",
        "Algorithm Settings": "Mathematical Optimization",
        "Sharpe Ratio performance in Train": round(markowitz_train_sharpe, 4),
        "Sharpe Ratio performance in Test": round(markowitz_test_sharpe, 4),
        "Evolution of performance(Test vs Train)": f"{round(markowitz_test_sharpe - markowitz_train_sharpe, 4)}"
    }]

    experments_names = ["Markowitz"]
    train_scores = [markowitz_train_sharpe]
    test_scores = [markowitz_test_sharpe]
    convergence_histories = {}

    print("3.Running experiments for SCA-BAS")

    get_fitness_train = lambda weights: -calculate_sharpe_ratio(validate_and_normalize_weights(np.array(weights)), mu_train, sigma_train)

    for exp in experiments:
        print(f"Running {exp['name']}...")

        start = time.perf_counter()
        best_weights, history = bassca_portfolio_optimization(
            get_fitness=get_fitness_train,
            n_stocks=len(mu_train),
            n_agents=exp['n_agents'],
            max_iter=exp['max_iter'],
            sensing_distance=exp['sensing_distance'],
            step_size=exp['step_size'],
            early_stopping=False
        )
        end = time.perf_counter()
        print(f"{end - start}s")

        best_weights = validate_and_normalize_weights(best_weights)

        train_sharpe = calculate_sharpe_ratio(best_weights, mu_train, sigma_train)
        test_sharpe = calculate_sharpe_ratio(best_weights, mu_test, sigma_test)

        convergence_histories[exp['name']] = history
        experments_names.append(exp['name'])
        train_scores.append(train_sharpe)
        test_scores.append(test_sharpe)

        results_summary.append({
            "Experiment Name": exp['name'],
            "Algorithm Settings": f"Agents:{exp['n_agents']}   Iter:{exp['max_iter']}   Sensing:{exp['sensing_distance']}   Step:{exp['step_size']}",
            "Sharpe Ratio performance in Train": round(train_sharpe, 4),
            "Sharpe Ratio performance in Test": round(test_sharpe, 4),
            "Evolution of performance(Test vs Train)": f"{round(test_sharpe - train_sharpe, 4)}",
            "Runtime": f"{end - start}s"
        })

        print(f"Sharpe Ratio on train dataset: {train_sharpe:.4f}    Sharpe Ratio on test dataset: {test_sharpe:.4f}")

    df = pd.DataFrame(results_summary)
    csv_path = os.path.join(results_dir, 'rapport_sca_bas.csv')
    df.to_csv(csv_path, index=False)
    print(f'Rapport saved in {csv_path}')

    print('Generating plots...')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    plt.figure(figsize=(9, 5))
    plt.axhline(y=markowitz_train_sharpe, color='black', linestyle='--', label='Markowitz(Mathematical maximum)')

    for i, (name, history) in enumerate(convergence_histories.items()):
        plt.plot(history, label=name, color=colors[i % len(colors)], linewidth=2)

    plt.title("Evolution of SCA-BAS (train data)", fontsize=12)
    plt.xlabel("Iterations")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "convergence_plot_for_sca_bas.png"))
    plt.close()

    x = np.arange(len(experments_names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, train_scores, width, label='Train', color='#7fbc41')
    plt.bar(x + width / 2, test_scores, width, label='Test', color='#de77ae')

    plt.ylabel('Sharpe Ratio')
    plt.title('In-Sample performance vs Out-of-Sample performance', fontsize=12)
    plt.xticks(x, experments_names)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "performance_plot_sca_bas.png"))

    plt.show()


if __name__ == "__main__":
    run_sca_bas_experiments()

