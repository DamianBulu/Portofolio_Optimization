
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_processed_data,calculate_sharpe_ratio,calculate_markowitz_benchmark
from algorithms.memetic_algo import MemeticAlgorithm


def run_hga_experiments():
    results_dir=os.path.join('results','results_hga')

    os.makedirs(results_dir,exist_ok=True)

    print("1.Incarcare date...")
    mu_train,sigma_train,test_returns=get_processed_data()

    #Procesare date de test
    mu_test=test_returns.mean().values*252
    sigma_test=test_returns.cov().values*252

    print("2.Calculare Benchmark(Markowitz)...")
    markowitz_weights,markowitz_train_sharpe=calculate_markowitz_benchmark(mu_train,sigma_train)
    markowitz_test_sharpe=calculate_sharpe_ratio(markowitz_weights,mu_test,sigma_test)

    print(f"2.Markowitz(matematic) results:")
    print(f"    Train:{markowitz_train_sharpe:.4f}")
    print(f"    Test:{markowitz_test_sharpe:.4f}")

    experiments = [
        {"name": "HGA Standard", "pop": 50, "gen": 100, "hc_prob": 0.3, "hc_iter": 10},
        {"name": "HGA Rapid", "pop": 30, "gen": 50, "hc_prob": 0.2, "hc_iter": 5},
        {"name": "HGA Agresiv", "pop": 60, "gen": 150, "hc_prob": 0.5, "hc_iter": 15}
    ]

    results_summary=[{
        "Experiment Name":"Benchmark Markowitz",
        'Algorithm Settings':'Mathematical Optimization',
        "Sharpe Ratio performance in Train":round(markowitz_train_sharpe,4),
        "Sharpe Ratio performance in Test":round(markowitz_test_sharpe,4),
        'Evolutia performantei(Test vs Train)':f'{round(markowitz_test_sharpe-markowitz_train_sharpe,4)}'
    }]

    experments_names=["Markowitz"]
    train_scores=[markowitz_train_sharpe]
    test_scores=[markowitz_test_sharpe]
    convergence_histories={}

    print("3.Rulare Experimente HGA")
    for exp in experiments:
        print(f"Rulare {exp['name']}...")
        algo=MemeticAlgorithm(
            mu=mu_train,
            sigma=sigma_train,
            population_size=exp['pop'],
            generations=exp['gen'],
            hc_probability=exp['hc_prob'],
            hc_iterations=exp['hc_iter']
        )

        best_weights,train_sharpe,history=algo.run_memetic_algorithm(verbose=False)
        test_sharpe=calculate_sharpe_ratio(best_weights,mu_test,sigma_test)

        #Salvare date
        convergence_histories[exp['name']]=history
        experments_names.append(exp['name'])
        train_scores.append(train_sharpe)
        test_scores.append(test_sharpe)

        results_summary.append({
            'Experiment Name':exp['name'],
            'Algorithm Settings':f"Population:{exp['pop']}   Generations:{exp['gen']}   HC_Probability:{exp['hc_prob']}",
            'Sharpe Ratio performance in Train':round(train_sharpe,4),
            "Sharpe Ratio performance in Test":round(test_sharpe,4),
            'Evolutia performantei(Test vs Train)':f'{round(test_sharpe-train_sharpe,4)}'

        })
        print(f"Sharpe Ratio on train dataset: {train_sharpe:.4f}    Sharpe Ration on test dataset: {test_sharpe:.4f}")

    #Salvare csv
    df=pd.DataFrame(results_summary)
    csv_path=os.path.join(results_dir,'raport_hga.csv')
    df.to_csv(csv_path,index=False)
    print(f'Raport salvat in {csv_path}')




    #Generare plouturi simple
    print('Generare grafice...')
    colors=['#1f77b4', '#ff7f0e', '#2ca02c']

    #Plot 1: Convergenta
    plt.figure(figsize=(9,5))
    plt.axhline(y=markowitz_train_sharpe,color='black',linestyle='--',label='MArkowitz(maxim matematic)')

    for i,(name,history) in enumerate(convergence_histories.items()):
        plt.plot(history,label=name,color=colors[i],linewidth=2)

    plt.title('Evolutie Algoritm Memetic(train data)',fontsize=12)
    plt.xlabel('Generatii')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True,linestyle=':',alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'convergence_plot_for_hga.png'))
    plt.close()

    #Plot 2: Comparatie Train vs Test
    x=np.arange(len(experments_names))
    width=0.35

    plt.figure(figsize=(9,5))
    plt.bar(x-width/2,train_scores,width,label='Train(trecut)',color='#7fbc41')
    plt.bar(x+width/2,test_scores,width,label='Test(viitor)',color='#de77ae')

    plt.ylabel('Sharpe Ratio')
    plt.title('In-Sample performance vs Out-of-Sample performance',fontsize=12)
    plt.xticks(x,experments_names)
    plt.legend()
    plt.grid(axis='y',linestyle=':',alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'performance_plot_hga.png'))

    plt.show()


if __name__=="__main__":
    run_hga_experiments()



