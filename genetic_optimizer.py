"""
Genetic Algorithm module for optimizing rule weights in an event-driven trading system.
"""

import numpy as np
import matplotlib.pyplot as plt
from backtester import Backtester
from strategy import TopNStrategy
import time

class GeneticOptimizer:
    """
    Optimizes trading rule weights using a genetic algorithm approach.
    
    This class implements a genetic algorithm to find optimal weights for combining
    signals from multiple trading rules. It's designed to work with the event-driven
    architecture and existing rules from the trading system.
    """
    
    def __init__(self, 
                 data_handler,
                 rule_objects,
                 population_size=20, 
                 num_parents=8, 
                 num_generations=50,
                 mutation_rate=0.1,
                 optimization_metric='sharpe'):
        """
        Initialize the genetic optimizer.
        
        Args:
            data_handler: The data handler object containing train/test data
            rule_objects: List of rule instances to optimize weights for
            population_size: Number of chromosomes in the population
            num_parents: Number of parents to select for mating
            num_generations: Number of generations to run the optimization
            mutation_rate: Rate of mutation in the genetic algorithm
            optimization_metric: Metric to optimize ('sharpe', 'return', 'drawdown', etc.)
        """
        self.data_handler = data_handler
        self.rule_objects = rule_objects
        self.population_size = population_size
        self.num_parents = num_parents
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.optimization_metric = optimization_metric
        self.num_weights = len(rule_objects)
        self.best_weights = None
        self.best_fitness = None
        self.fitness_history = []
    
    def _initialize_population(self):
        """
        Initialize a random population of weight vectors.
        
        Returns:
            numpy.ndarray: Initial population of chromosomes
        """
        # Create population with random weights between 0 and 1
        population = np.random.uniform(low=0.0, high=1.0, 
                                       size=(self.population_size, self.num_weights))
        
        # Normalize each chromosome so weights sum to 1
        row_sums = population.sum(axis=1)
        normalized_population = population / row_sums[:, np.newaxis]
        
        return normalized_population


    def _calculate_fitness(self, chromosome):
        """
        Calculate fitness for a single chromosome by backtesting the weighted strategy.

        Args:
            chromosome: Array of weights for the rules

        Returns:
            float: Fitness score based on the optimization metric
        """
        # Create a copy of TopNStrategy with the weighted combination method
        weighted_strategy = WeightedRuleStrategy(
            rule_objects=self.rule_objects,
            weights=chromosome
        )

        # Reset the strategy and run backtest
        weighted_strategy.reset()
        backtester = Backtester(self.data_handler, weighted_strategy)
        results = backtester.run(use_test_data=False)  # Train on training data

        # Calculate fitness based on selected metric
        if self.optimization_metric == 'sharpe':
            if results['num_trades'] >= 5:  # Require minimum trades for meaningful results
                fitness = backtester.calculate_sharpe()
            else:
                # Penalize strategies with too few trades
                fitness = -1.0
        elif self.optimization_metric == 'return':
            fitness = results['total_log_return'] if results['num_trades'] > 0 else -1.0
        elif self.optimization_metric == 'risk_adjusted':
            # Custom risk-adjusted return metric
            if results['num_trades'] >= 5:
                trades = np.array([t[5] for t in results['trades']])  # Log returns
                mean_return = np.mean(trades)
                std_return = np.std(trades) if len(trades) > 1 else float('inf')
                downside = np.sum(trades[trades < 0]) if any(trades < 0) else -0.0001
                fitness = mean_return / (std_return * (-downside))
            else:
                fitness = -1.0
        elif self.optimization_metric == 'win_rate':
            # Calculate win rate (percentage of profitable trades)
            if results['num_trades'] >= 5:  # Require minimum trades for meaningful results
                winning_trades = sum(1 for t in results['trades'] if t[5] > 0)
                win_rate = winning_trades / results['num_trades']

                # We might also want to ensure a minimum number of trades
                # to avoid strategies that make very few but lucky trades
                trade_count_factor = min(1.0, results['num_trades'] / 20)  # Reaches 1.0 at 20+ trades

                fitness = win_rate * trade_count_factor
            else:
                # Penalize strategies with too few trades
                fitness = -1.0
        else:
            # Default to total return
            fitness = results['total_log_return']

        return fitness
    
    def _calculate_population_fitness(self, population):
        """
        Calculate fitness for the entire population.
        
        Args:
            population: Array of chromosomes
            
        Returns:
            numpy.ndarray: Fitness scores for each chromosome
        """
        fitness_scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness_scores[i] = self._calculate_fitness(population[i])
        return fitness_scores

    def _select_parents(self, population, fitness):
        """
        Select the best parents for producing the offspring of the next generation.
        
        Args:
            population: Current population
            fitness: Fitness scores for the population
            
        Returns:
            numpy.ndarray: Selected parent chromosomes
        """
        parents = np.empty((self.num_parents, self.num_weights))
        for i in range(self.num_parents):
            max_fitness_idx = np.argmax(fitness)
            parents[i] = population[max_fitness_idx]
            fitness[max_fitness_idx] = -float('inf')  # Ensure this parent isn't selected again
        return parents

    def _crossover(self, parents):
        """
        Create offspring through crossover of the parents.
        
        Args:
            parents: Selected parent chromosomes
            
        Returns:
            numpy.ndarray: Offspring chromosomes
        """
        offspring_size = self.population_size - self.num_parents
        offspring = np.empty((offspring_size, self.num_weights))
        
        for k in range(offspring_size):
            # Select two parents
            parent1_idx = k % self.num_parents
            parent2_idx = (k + 1) % self.num_parents
            
            # Perform crossover - weighted average of parents
            crossover_ratio = np.random.random()
            offspring[k] = (crossover_ratio * parents[parent1_idx] + 
                           (1 - crossover_ratio) * parents[parent2_idx])
            
            # Ensure normalization
            offspring[k] = offspring[k] / np.sum(offspring[k])
            
        return offspring

    def _mutate(self, offspring):
        """
        Mutate the offspring by introducing random changes.
        
        Args:
            offspring: Offspring chromosomes after crossover
            
        Returns:
            numpy.ndarray: Mutated offspring
        """
        for i in range(len(offspring)):
            # Determine if this chromosome should be mutated
            if np.random.random() < self.mutation_rate:
                # Select two random positions to modify
                idx1, idx2 = np.random.choice(self.num_weights, 2, replace=False)
                
                # Get a random change amount (ensuring sum remains the same)
                change = np.random.uniform(-offspring[i, idx1]/2, offspring[i, idx1]/2)
                
                # Apply mutation
                offspring[i, idx1] -= change
                offspring[i, idx2] += change
                
                # Ensure no negative weights
                if offspring[i, idx1] < 0:
                    offspring[i, idx2] += offspring[i, idx1]
                    offspring[i, idx1] = 0
                if offspring[i, idx2] < 0:
                    offspring[i, idx1] += offspring[i, idx2]
                    offspring[i, idx2] = 0
                    
                # Re-normalize to ensure weights sum to 1
                offspring[i] = offspring[i] / np.sum(offspring[i])
                
        return offspring


    def optimize(self, verbose=True, early_stopping_generations=10, min_improvement=0.001):
        """
        Run the genetic algorithm to find optimal weights with early stopping.

        Args:
            verbose: Whether to print progress information
            early_stopping_generations: Stop if no improvement after this many generations
            min_improvement: Minimum improvement in fitness to be considered significant

        Returns:
            numpy.ndarray: Optimal weights for the rules
        """
        start_time = time.time()
        if verbose:
            print(f"Starting genetic optimization with {self.num_weights} rules...")
            print(f"Population size: {self.population_size}, Generations: {self.num_generations}")

        # Initialize population
        population = self._initialize_population()

        # Variables for early stopping
        generations_without_improvement = 0
        last_best_fitness = float('-inf')

        # Run genetic algorithm for specified number of generations
        for generation in range(self.num_generations):
            # Calculate fitness for current population
            fitness = self._calculate_population_fitness(population)

            # Track best fitness in this generation
            best_fitness = np.max(fitness)
            self.fitness_history.append(best_fitness)

            if verbose and (generation % 5 == 0 or generation == self.num_generations - 1):
                elapsed = time.time() - start_time
                print(f"Generation {generation+1}/{self.num_generations} | "
                      f"Best Fitness: {best_fitness:.4f} | "
                      f"Time: {elapsed:.1f}s")

            # Check for early stopping
            improvement = best_fitness - last_best_fitness
            if improvement > min_improvement:
                generations_without_improvement = 0
                last_best_fitness = best_fitness
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= early_stopping_generations:
                if verbose:
                    print(f"\nEarly stopping at generation {generation+1}: No improvement for {early_stopping_generations} generations")
                break

            # Store the best weights found so far
            if self.best_fitness is None or best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                best_idx = np.argmax(fitness)
                self.best_weights = population[best_idx].copy()

            # Select parents
            parents = self._select_parents(population, fitness.copy())

            # Create offspring through crossover
            offspring = self._crossover(parents)

            # Apply mutation
            offspring = self._mutate(offspring)

            # Create new population for next generation
            population[0:self.num_parents] = parents
            population[self.num_parents:] = offspring

        if verbose:
            total_time = time.time() - start_time
            print(f"\nOptimization completed in {total_time:.1f} seconds")
            print(f"Best fitness: {self.best_fitness:.4f}")
            print(f"Optimal weights: {self.best_weights}")

        return self.best_weights


    
    def plot_fitness_history(self):
        """
        Plot the evolution of fitness over generations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history)
        plt.title('Fitness Evolution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class WeightedRuleStrategy:
    """
    Strategy that combines multiple rule signals using weighted voting.
    
    This strategy extends the TopNStrategy concept by applying weights to each rule's
    signal when combining them, rather than using a simple average.
    """
    
    def __init__(self, rule_objects, weights=None):
        """
        Initialize the weighted rule strategy.
        
        Args:
            rule_objects: List of rule instances to use
            weights: List of weights for each rule (will be normalized)
        """
        self.rules = rule_objects
        
        # If weights not provided, use equal weighting
        if weights is None:
            self.weights = np.ones(len(rule_objects)) / len(rule_objects)
        else:
            # Ensure weights are normalized
            self.weights = np.array(weights) / np.sum(weights)
            
        self.last_signal = None
    
    def on_bar(self, event):
        """
        Process a new bar and generate signals.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            dict: Signal information including timestamp, signal, and price
        """
        bar = event.bar
        # Get signals from each rule
        rule_signals = [rule.on_bar(bar) for rule in self.rules]
        
        # Combine signals using weights
        combined = self.combine_signals(rule_signals)
        
        self.last_signal = {
            "timestamp": bar["timestamp"],
            "signal": combined,
            "price": bar["Close"]
        }
        return self.last_signal
    
    def combine_signals(self, signals):
        """
        Combine rule signals using weighted voting.
        
        Args:
            signals: List of signals from each rule
            
        Returns:
            int: Combined signal (-1, 0, or 1)
        """
        # Convert signals to numpy array
        signals_array = np.array(signals)
        
        # Calculate weighted average
        weighted_sum = np.sum(signals_array * self.weights)
        
        # Convert to discrete signal
        if weighted_sum > 0.2:  # Threshold for buy
            return 1
        elif weighted_sum < -0.2:  # Threshold for sell
            return -1
        else:
            return 0
    
    def reset(self):
        """
        Reset the strategy.
        """
        for rule in self.rules:
            rule.reset()
        self.last_signal = None
