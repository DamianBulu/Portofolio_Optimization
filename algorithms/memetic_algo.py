

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_processed_data,calculate_sharpe_ratio,validate_and_normalize_weights


class MemeticAlgorithm:
    def __init__(self,mu,sigma,population_size=50,generations=150,crossover_rate=0.8,mutation_rate=0.1,mutation_std=0.05,hc_probability=0.3,hc_iterations=10):
        #Constructorul clasei:aici se initializeaza toti hiperparametrii algoritmului
        # Setarile sunt inspirate din lucrarea de cercetare pentru a obtine o convergenta buna

        self.mu=mu
        self.sigma=sigma
        self.population_size=population_size
        self.generations=generations
        self.crossover_rate=crossover_rate
        self.mutation_rate=mutation_rate
        self.mutation_std=mutation_std
        self.hc_probability=hc_probability
        self.hc_iterations=hc_iterations
        self.n_assets=len(mu)

    def init_population(self):
        """
            Generare random de ponderi pt fiecare portofoliu + normalizare
        """
        population=[]
        for _ in range(self.population_size):
            weights=np.random.rand(self.n_assets)
            weights=validate_and_normalize_weights(weights)
            population.append(weights)
        return population


    def tournament_selection(self,population,fitnesees,k=3):
        """
            Se aleg cativa indivizi la intamplare si se returneaza cel mai bun
        """
        selected_indices=random.sample(range(len(population)),k)
        best_idx=max(selected_indices,key=lambda idx: fitnesees[idx])

        return population[best_idx]

    def arithmetic_crossover(self,parent1,parent2):
        """
            Aici se combina 2 parinti folosind un factor alpha random intre 0 si 1
        """
        if random.random()<self.crossover_rate:
            alpha=random.random()
            offspring1=alpha*parent1+(1-alpha)*parent2
            offspring2=(1-alpha)*parent1+alpha*parent2

            return validate_and_normalize_weights(offspring1),validate_and_normalize_weights(offspring2)

        return parent1.copy(),parent2.copy()

    def gaussian_mutation(self,individual):
        """
         Se adauga zgomot guassian peste genele individului
        """

        if random.random()<self.mutation_rate:
            noise=np.random.normal(0,self.mutation_std,self.n_assets)
            mutated=individual+noise
            return validate_and_normalize_weights(mutated)
        return individual

    def hill_climbing(self,individual):
        """
            Se ia un portofoliu si se ajusteaza ponderile pt a i creste Sharpe Ration
        """
        current_best=individual.copy()
        current_fittness=calculate_sharpe_ratio(current_best,self.mu,self.sigma)

        for _ in range(self.hc_iterations):
            neighbor=current_best.copy()

            #se aleg random 2 actove diferite
            idx1,idx2=random.sample(range(self.n_assets),2)

            #se muta o mica pondere(de ex. intre 1% si 5%) de la activul 1 la acticul 2
            shift_amount=random.uniform(0.01,0.05)

            #verificre sa nu se scada sub zero
            if neighbor[idx1]>=shift_amount:
                neighbor[idx1]-=shift_amount
                neighbor[idx2]+=shift_amount

                neighbor=validate_and_normalize_weights(neighbor)
                neighbor_fitness=calculate_sharpe_ratio(neighbor,self.mu,self.sigma)

                #daca se gaseste un portofoliu vecin mai bun se pastreaza
                if neighbor_fitness>current_fittness:
                    current_best=neighbor.copy()
                    current_fittness=neighbor_fitness
        return current_best


    def run_memetic_algorithm(self,verbose=False):
        """
            Bucla principala pt algoritmul MEmetic (HGA -Hybrid Genetic Algoritm)


        """


        population=self.init_population()

        best_fitness_history=[]
        global_best_individual=None
        global_best_fitness=-np.inf

        if verbose:
            print(f"Incepere optimizare memetica pt un nr de {self.generations} generatii, Population: {self.population_size}:")

        for generation in range(self.generations):

            #evaluarea populatiilor
            fitnesses=[calculate_sharpe_ratio(ind,self.mu,self.sigma) for ind in population]

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
            while len(new_population)<self.population_size:

                #selectie
                p1=self.tournament_selection(population,fitnesses)
                p2=self.tournament_selection(population,fitnesses)

                #crossover
                offspring1,offspring2=self.arithmetic_crossover(p1,p2)

                #mutatie
                offspring1=self.gaussian_mutation(offspring1)
                offspring2=self.gaussian_mutation(offspring2)

                #Local Search cu HC
                if random.random()<self.hc_probability:
                    offspring1=self.hill_climbing(offspring1)
                if random.random()<self.hc_probability:
                    offspring2=self.hill_climbing(offspring2)

                new_population.append(offspring1)
                if len(new_population)<self.population_size:
                    new_population.append(offspring2)

            population=new_population

            if verbose and (generation+1)%25==0:
                print(f"Generația {generation + 1}/{self.generations} | Best Sharpe Ratio: {global_best_fitness:.4f}")

        return global_best_individual,global_best_fitness,best_fitness_history





