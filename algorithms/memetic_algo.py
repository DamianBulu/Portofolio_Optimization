

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_processed_data,calculate_sharpe_ratio,validate_and_normalize_weights

#Parametri algoritm memetic
#Setarile sunt inspirate din lucrarea de cercetare pentru a obtine o convergenta buna
population_size=50
generations=150
crossover_rate=0.8
mutation_rate=0.1
mutation_std=0.05

#Parametri specifici pentru Hill Climbing(Local Search)
hc_probability=0.3 #Sansa ca un individ sa fie supus cautarii locale
hc_iterations=10  #De cate ori incercam sa imbunatatim individual (kHC)

def init_population(pop_size,n_assets):
    """
        Generare random de ponderi pt fiecare portofoliu + normalizare
    """

    population=[]
    for _ in range(pop_size):
        weights=np.random.rand(n_assets)
        weights=validate_and_normalize_weights(weights)
        population.append(weights)
    return population


def tournament_selection(population,fitnesees,k=3):
    """
        Se aleg cativa indivizi la intamplare si se returneaza cel mai bun
    """
    selected_indices=random.sample(range(len(population)),k)
    best_idx=max(selected_indices,key=lambda idx: fitnesees[idx])

    return population[best_idx]

def arithmetic_crossover(parent1,parent2):
    """
        Aici se combina 2 parinti folosind un factor alpha random intre 0 si 1
    """

    if random.random()<crossover_rate:
        alpha=random.random()
        offspring1=alpha*parent1+(1-alpha)*parent2
        offspring2=(1-alpha)*parent1+alpha*parent2

        return validate_and_normalize_weights(offspring1),validate_and_normalize_weights(offspring2)

    return parent1.copy(),parent2.copy()

def gaussian_mutation(individual):
    """
     Se adauga zgomot guassian peste genele individului
    """

    if random.random()<mutation_rate:
        noise=np.random.normal(0,mutation_std,len(individual))
        mutated=individual+noise
        return validate_and_normalize_weights(mutated)
    return individual

def hill_climbing(individual,mu,sigma):
    """
        Se ia un portofoliu si se ajusteaza ponderile pt a i creste Sharpe Ration
    """
    current_best=individual.copy()
    current_fittness=calculate_sharpe_ratio(current_best,mu,sigma)

    for _ in range(hc_iterations):
        neighbor=current_best.copy()

        #se aleg random 2 actove diferite
        idx1,idx2=random.sample(range(len(neighbor)),2)

        #se muta o mica pondere(de ex. intre 1% si 5%) de la activul 1 la acticul 2
        shift_amount=random.uniform(0.01,0.05)

        #verificre sa nu se scada sub zero
        if neighbor[idx1]>=shift_amount:
            neighbor[idx1]-=shift_amount
            neighbor[idx2]+=shift_amount

            neighbor=validate_and_normalize_weights(neighbor)
            neighbor_fitness=calculate_sharpe_ratio(neighbor,mu,sigma)

            #daca se gaseste un portofoliu vecin mai bun se pastreaza
            if neighbor_fitness>current_fittness:
                current_best=neighbor.copy()
                current_fittness=neighbor_fitness
    return current_best


def run_memetic_algorithm(mu,sigma):
    """
        Bucla principala pt algoritmul MEmetic (HGA -Hybrid Genetic Algoritm)


    """

    n_assets=len(mu)
    population=init_population(population_size,n_assets)

    best_fitness_history=[]
    global_best_individual=None
    global_best_fitness=-np.inf

    print(f"Incepere optimizare memetica pt un nr de {generations} generatii")

    for generation in range(generations):

        #evaluarea populatiilor
        fitnesses=[calculate_sharpe_ratio(ind,mu,sigma) for ind in population]

        #salvarea celui mai bun individ din generatia curenta
        gen_best_idx=np.argmax(fitnesses)
        if fitnesses[gen_best_idx]>global_best_fitness:
            global_best_fitness=fitnesses[gen_best_idx]
            global_best_individual=population[gen_best_idx].copy()

        best_fitness_history.append(global_best_fitness)

        new_population=[]

        #cel mai bun individ se pastreaza in noua populatie
        new_population.append(global_best_individual.copy())

        #restul noii populatii
        while len(new_population)<population_size:

            #selectie
            p1=tournament_selection(population,fitnesses)
            p2=tournament_selection(population,fitnesses)

            #crossover
            offspring1,offspring2=arithmetic_crossover(p1,p2)

            #mutatie
            offspring1=gaussian_mutation(offspring1)
            offspring2=gaussian_mutation(offspring2)

            #Local Search cu HC
            if random.random()<hc_probability:
                offspring1=hill_climbing(offspring1,mu,sigma)
            if random.random()<hc_probability:
                offspring2=hill_climbing(offspring2,mu,sigma)

            new_population.append(offspring1)
            if len(new_population)<population_size:
                new_population.append(offspring2)

        population=new_population

        if (generation+1)%25==0:
            print(f"Generația {generation + 1}/{generations} | Best Sharpe Ratio: {global_best_fitness:.4f}")

    return global_best_individual,global_best_fitness,best_fitness_history





