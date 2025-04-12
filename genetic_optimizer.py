"""
Genetic Algorithm module for optimizing rule weights in an event-driven trading system.
"""

import numpy as np
import matplotlib.pyplot as plt
from backtester import Backtester
from strategy import TopNStrategy
import time
from signals import SignalType


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
                 optimization_metric='sharpe',
                 random_seed=None,
                 deterministic=False,
                 buy_threshold=0.5,  # New parameter
                 sell_threshold=-0.5):  # New parameter
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
            random_seed: Optional seed for random number generator to ensure reproducible results
            deterministic: If True, ensures deterministic behavior across multiple runs
            buy_threshold: Threshold for the weighted sum to trigger a BUY signal
            sell_threshold: Threshold for the weighted sum to trigger a SELL signal
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
        self.random_seed = random_seed
        self.deterministic = deterministic
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Set random seed if provided or if deterministic mode is enabled
        if self.deterministic and self.random_seed is None:
            self.random_seed = 42  # Default seed for deterministic mode

        # Initialize the random state
        self._set_random_seed()

    def _set_random_seed(self):
        """Set the random seed if specified for reproducible results."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

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
        # Create a copy of WeightedRuleStrategy with the weighted combination method
        weighted_strategy = WeightedRuleStrategy(
            rule_objects=self.rule_objects,
            weights=chromosome,
            buy_threshold=self.buy_threshold,  # Pass the threshold
            sell_threshold=self.sell_threshold  # Pass the threshold
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
        # Reset random seed for reproducibility if needed
        self._set_random_seed()

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
        # Reset random seed for reproducibility if needed
        self._set_random_seed()

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
            if self.deterministic:
                print(f"Running in deterministic mode with seed: {self.random_seed}")
            print(f"Buy Threshold: {self.buy_threshold}, Sell Threshold: {self.sell_threshold}")

        # Reset the random seed at the beginning of optimization
        self._set_random_seed()

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

    def set_random_seed(self, seed):
        """
        Set a new random seed for the optimizer.

        Args:
            seed: Integer seed for the random number generator
        """
        self.random_seed = seed
        self._set_random_seed()

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
    def __init__(self, rule_objects, weights, buy_threshold=0.5, sell_threshold=-0.5):
        self.rule_objects = rule_objects
        self.weights = np.array(weights)
        self.rule_signals = [None] * len(rule_objects)
        self.buy_threshold = buy_threshold  # Store the buy threshold
        self.sell_threshold = sell_threshold  # Store the sell threshold

    def on_bar(self, event):
        bar = event.bar  # Extract the bar dictionary from the BarEvent
        combined_signals = []
        for i, rule in enumerate(self.rule_objects):
            signal_object = rule.on_bar(bar) # Now we expect a Signal object
            #print(f"Signal Object from {rule.rule_id}: {signal_object}")
            #print(f"Type of signal_object: {type(signal_object)}")
            if signal_object and hasattr(signal_object, 'type'):
                combined_signals.append(signal_object.type.value * self.weights[i])
            else:
                combined_signals.append(0) # Or handle missing signal appropriately

        weighted_sum = np.sum(combined_signals)

        if weighted_sum > self.buy_threshold:
            final_signal = SignalType.BUY
        elif weighted_sum < self.sell_threshold:
            final_signal = SignalType.SELL
        else:
            final_signal = SignalType.NEUTRAL

        return {
            "timestamp": bar["timestamp"], # Use the extracted bar
            "signal": final_signal,
            "price": bar["Close"]        # Use the extracted bar
        }

    def reset(self):
        for rule in self.rule_objects:
            if hasattr(rule, 'reset'):
                rule.reset()
        self.rule_signals = [None] * len(self.rule_objects)
