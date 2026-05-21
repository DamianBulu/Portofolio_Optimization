import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_processed_data, calculate_sharpe_ratio, validate_and_normalize_weights

class PSOAlgorithm:
    def __init__(self, mu, sigma, population_size=50, iterations=150, w=0.7298, c1=1.4962, c2=1.4962, early_stopping=True):
        """
        Class Constructor: Initializes the Particle Swarm Optimisation parameters.
        Parameters (w, c1, c2) are based on standard professional convergence coefficients
        (Clerc & Kennedy, 2002) to ensure swarm stability.
        """
        self.mu = mu
        self.sigma = sigma
        self.population_size = population_size
        self.iterations = iterations
        
        # PSO Specific Hyperparameters
        self.w = w    # Inertia weight: controls the impact of the previous velocity
        self.c1 = c1  # Cognitive coefficient: tendency to return to personal best
        self.c2 = c2  # Social coefficient: tendency to move toward global best
        
        self.early_stopping = early_stopping
        self.n_assets = len(mu)

    def init_particles(self):
        """
        Initialises particles with random weight distributions and zero initial velocities.
        """
        particles = []
        velocities = []
        for _ in range(self.population_size):
            # Initialise weights using Dirichlet to ensure they sum to 1 immediately
            weights = np.random.dirichlet(np.ones(self.n_assets))
            particles.append(weights)
            # Initialise velocity as small random values
            velocities.append(np.random.uniform(-0.01, 0.01, self.n_assets))
        
        return np.array(particles), np.array(velocities)

    def _should_stop(self, history):
        """
        Early stopping logic consistent with the SCA-BAS model.
        """
        if len(history) < 10:
            return False
        # Check if the improvement over the last 10 iterations is negligible
        return all(abs(history[-1] - v) < 1e-7 for v in history[-10:])

    def run_pso(self, verbose=False):
        """
        Main execution loop for the Particle Swarm Optimisation algorithm.
        Objective: Maximise the Sharpe Ratio through iterative swarm movement.
        """
        # Initialisation
        particles, velocities = self.init_particles()
        
        # Track personal bests for each particle
        p_best_positions = particles.copy()
        p_best_scores = np.array([calculate_sharpe_ratio(p, self.mu, self.sigma) for p in particles])
        
        # Track global best
        g_best_idx = np.argmax(p_best_scores)
        g_best_position = p_best_positions[g_best_idx].copy()
        g_best_score = p_best_scores[g_best_idx]
        
        history = [g_best_score]

        if verbose:
            print(f"Starting PSO Optimisation: {self.iterations} iterations, Swarm Size: {self.population_size}")

        for iteration in range(self.iterations):
            
            for i in range(self.population_size):
                # Random stochastic components
                r1 = np.random.rand(self.n_assets)
                r2 = np.random.rand(self.n_assets)
                
                # 1. Update Velocity
                # Formula: V(t+1) = w*V(t) + c1*r1*(pbest - X) + c2*r2*(gbest - X)
                cognitive = self.c1 * r1 * (p_best_positions[i] - particles[i])
                social = self.c2 * r2 * (g_best_position - particles[i])
                velocities[i] = (self.w * velocities[i]) + cognitive + social
                
                # 2. Update Position (Weights)
                particles[i] = particles[i] + velocities[i]
                
                # 3. Constraint Handling (Non-negativity and Sum-to-One)
                particles[i] = validate_and_normalize_weights(particles[i])
                
                # 4. Evaluate Performance
                current_score = calculate_sharpe_ratio(particles[i], self.mu, self.sigma)
                
                # 5. Update Personal Best
                if current_score > p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = particles[i].copy()
                    
                    # 6. Update Global Best
                    if current_score > g_best_score:
                        g_best_score = current_score
                        g_best_position = particles[i].copy()

            # Record history for convergence analysis
            history.append(g_best_score)

            # Early Stopping Check
            if self.early_stopping and self._should_stop(history):
                if verbose:
                    print(f"Convergence achieved: Early stopping at iteration {iteration + 1}")
                break

            if verbose and (iteration + 1) % 25 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations} | Global Best Sharpe: {g_best_score:.4f}")

        return g_best_position, g_best_score, history

# Example usage for testing the script standalone
if __name__ == "__main__":
    # Mock data for standalone test
    n_assets = 10
    mu = np.random.uniform(0.01, 0.05, n_assets)
    # Generate a positive semi-definite covariance matrix
    matrix = np.random.randn(n_assets, n_assets)
    sigma = np.dot(matrix, matrix.T)
    
    pso = PSOAlgorithm(mu, sigma, population_size=30, iterations=100)
    best_weights, best_score, hist = pso.run_pso(verbose=True)
    
    print(f"\nFinal Results:")
    print(f"Best Sharpe Ratio: {best_score:.4f}")
    print(f"Weights Sum: {np.sum(best_weights):.2f}")