import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from main_experiments.memetic_algo_experiments import run_hga_experiments
from main_experiments.pso_algo_experiments import run_pso_experiments
from main_experiments.sca_bas_algo_experiments import run_sca_bas_experiments


def main():
	run_hga_experiments()
	run_pso_experiments()
	run_sca_bas_experiments()


if __name__ == '__main__':
	main()
