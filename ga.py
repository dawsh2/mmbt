"""
Genetic Algorithm module for optimizing trading strategy weights.
Based on the code provided in paste.txt.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

def cal_pop_fitness(equation_inputs, pop, opt=0):
    """
    Calculate fitness of each solution in the population.
    
    Args:
        equation_inputs: First column is returns, other columns are rule signals
        pop: Population of weights for each rule
        opt: Optimization criterion (0 is default SSR model)
        
    Returns:
        Fitness score for each solution
    """
    # Extract log returns and rule signals
    logr = equation_inputs[:, 0]  # returns
    positions = pop @ equation_inputs[:, 1:].T  # weighted signals
    
    # Calculate portfolio returns
    port_r = (positions * logr).astype(np.float64)
    
    # Calculate SSR (Signal to Standard Deviation Ratio)
    mean_return = np.mean(port_r, axis=1)
    std_return = np.std(port_r, axis=1)
    
    # Calculate total negative returns for each solution
    negative_returns = np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        negative_returns[i] = np.sum(port_r[i][port_r[i] < 0])
    
    # Avoid division by zero
    std_return = np.where(std_return == 0, 1e-10, std_return)
    negative_returns = np.where(negative_returns == 0, -1e-10, negative_returns)
    
    # SSR ratio: mean return / std dev / negative returns
    SSR = mean_return / std_return / (-negative_returns)
    
    return SSR

def select_mating_pool(pop, fitness, num_parents):
    """
    Select the best individuals for mating based on fitness.
    
    Args:
        pop: Population of potential solutions
        fitness: Fitness scores for each solution
        num_parents: Number of parents to select
        
    Returns:
        Selected parents
    """
    # Initialize array to hold selected parents
    parents = np.empty((num_parents, pop.shape[1]))
    
    # Select best-performing individuals
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999  # Ensure this individual is not selected again
        
    return parents

def crossover(parents, offspring_size):
    """
    Perform crossover between pairs of parents.
    
    Args:
        parents: Selected parents
        offspring_size: Size of the offspring (num_offspring, num_weights)
        
    Returns:
        Generated offspring
    """
    # Initialize array to hold offspring
    offspring = np.empty(offspring_size)
    
    # Set crossover point (typically in the middle)
    crossover_point = np.uint8(offspring_size[1]/2)

    # Create offspring
    for k in range(offspring_size[0]):
        # Select parents
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        
        # First half from parent 1
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # Second half from parent 2
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    """
    Apply random mutations to offspring.
    
    Args:
        offspring_crossover: Offspring from crossover
        num_mutations: Number of mutations to apply per offspring
        
    Returns:
        Mutated offspring
    """
    # Calculate how often to apply mutations
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    
    # Apply mutations
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for _ in range(num_mutations):
            # Add random value to selected gene
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
            
    return offspring_crossover

def GA_train(training_df, optimizing_selection=0, sol_per_pop=8, num_parents_mating=4, num_generations=200):
    """
    Train weights using genetic algorithm.
    
    Args:
        training_df: Training data with returns in first column and rule signals in other columns
        optimizing_selection: Optimization criterion (default: 0 for SSR)
        sol_per_pop: Population size
        num_parents_mating: Number of parents for mating
        num_generations: Number of generations to evolve
        
    Returns:
        Best solution (weights) found
    """
    # Check if input is DataFrame and convert to numpy array if needed
    if hasattr(training_df, 'values'):
        equation_inputs = training_df.values
    else:
        equation_inputs = training_df
    
    # Number of weights to optimize (number of rules)
    num_weights = equation_inputs.shape[1] - 1
    
    # Define population size
    pop_size = (sol_per_pop, num_weights)
    
    # Create initial population with random weights
    new_population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
    
    # Track best outputs for each generation
    best_outputs = []
    
    # Evolution loop
    for generation in range(num_generations):
        # Calculate fitness for current population
        fitness = cal_pop_fitness(equation_inputs, new_population, optimizing_selection)
        
        # Track best fitness in this generation
        best_outputs.append(np.max(fitness))
        
        # Display progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}/{num_generations} - Best fitness: {np.max(fitness):.6f}")
        
        # Select parents
        parents = select_mating_pool(new_population, fitness.copy(), num_parents_mating)
        
        # Create offspring through crossover
        offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        
        # Apply mutations
        offspring_mutation = mutation(offspring_crossover, num_mutations=2)
        
        # Create new population
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    
    # Get best solution after all generations
    fitness = cal_pop_fitness(equation_inputs, new_population, optimizing_selection)
    best_match_idx = np.where(fitness == np.max(fitness))
    
    # Plot evolution progress
    plt.figure(figsize=(10, 6))
    plt.plot(best_outputs)
    plt.xlabel("Generation")
    plt.ylabel('Fitness (SSR ratio)')
    plt.title('GA Optimization Progress')
    plt.grid(True)
    plt.show()
    
    # Return best solution
    best_solution = new_population[best_match_idx[0][0], :]
    return best_solution
