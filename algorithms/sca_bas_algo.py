import os
import sys
import plotly.graph_objects as go
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_processed_data, calculate_sharpe_ratio, validate_and_normalize_weights

def should_stop(history):
    if len(history) < 5:
        return False

    val = history[-1]
    vals = history[-5:]
    return all(abs(val - v) < 1e-6 for v in vals)

def bassca_portfolio_optimization(
        get_fitness,
        n_stocks=25,
        n_agents=25,
        max_iter=100,
        sensing_distance = 0.1,
        step_size = 0.1,
        early_stopping = True,
):
    X = np.random.dirichlet(np.ones(n_stocks), size=n_agents)

    p_best = X.copy()

    fitness_scores = np.array([get_fitness(ind) for ind in X])
    g_best = X[np.argmin(fitness_scores)].copy()

    history = [-get_fitness(g_best)]

    for t in range(max_iter):
        if early_stopping and should_stop(history):
            print(f"Early stopping at iteration {t}")
            break

        r1 = 2 * (1 - t / max_iter)
        omega = 1 / (1 + 1.5 * np.exp(10 * t / max_iter - 5)) - 0.1 * np.random.rand()

        for i in range(n_agents):
            r2 = 2 * np.pi * np.random.rand()
            r3 = 2 * np.pi * np.random.rand()
            r4 = np.random.rand()

            if r4 < 0.5:
                X[i] = omega * X[i] + r1 * np.sin(r2) * np.abs(r3 * g_best - X[i])
            else:
                X[i] = omega * X[i] + r1 * np.cos(r2) * np.abs(r3 * g_best - X[i])

            X[i] = np.clip(X[i], 0, 1)
            X[i] /= (np.sum(X[i]) + 1e-12) # to ensure we don't divide by zero

            direction = np.random.randn(n_stocks)
            direction /= (np.linalg.norm(direction) + 1e-12)

            x_left = p_best[i] + sensing_distance * direction
            x_right = p_best[i] - sensing_distance * direction

            if get_fitness(x_left) < get_fitness(x_right):
                p_new = p_best[i] + step_size * direction
            else:
                p_new = p_best[i] - step_size * direction

            p_new = np.clip(p_new, 0, 1)
            p_new /= (np.sum(p_new) + 1e-12)

            if get_fitness(p_new) < get_fitness(p_best[i]):
                p_best[i] = p_new.copy()

            current_fit = get_fitness(X[i])
            if current_fit < get_fitness(p_best[i]):
                p_best[i] = X[i].copy()

            if get_fitness(p_best[i]) < get_fitness(g_best):
                g_best = p_best[i].copy()

        history.append(-get_fitness(g_best))

    return g_best, history