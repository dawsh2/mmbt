
"""
Enhanced Genetic Algorithm module with regularization and cross-validation for rule weights.
"""

import numpy as np
import matplotlib.pyplot as plt
from backtester import Backtester
import time
from signals import Signal, SignalType
import gc  # For garbage collection

class GeneticOptimizer:
    """
    Optimizes trading rule weights using a genetic algorithm approach with regularization
    and cross-validation to reduce overfitting.
    
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
                 optimization_metric='return',
                 random_seed=None,
                 deterministic=False,
                 batch_size=None,
                 cv_folds=3,  # Cross-validation folds
                 regularization_factor=0.2,  # Weight regularization strength
                 balance_factor=0.3,  # Balance factor for equal vs optimized weights
                 max_weight_ratio=3.0,  # Maximum ratio between weights
                 optimize_thresholds=True):  # Whether to optimize threshold parameters
        """
        Initialize the enhanced genetic optimizer with regularization and cross-validation.
        
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
            batch_size: Optional batch size for fitness calculations (for memory efficiency)
            cv_folds: Number of cross-validation folds
            regularization_factor: Weight given to regularization term (0-1)
            balance_factor: Weight given to balancing toward equal weights (0-1) 
            max_weight_ratio: Maximum allowed ratio between highest and lowest weights
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
        self.best_thresholds = None
        self.best_fitness = None
        self.fitness_history = []
        self.random_seed = random_seed
        self.deterministic = deterministic
        self.batch_size = batch_size  # For batch processing
        self.cv_folds = cv_folds
        self.regularization_factor = regularization_factor
        self.balance_factor = balance_factor
        self.max_weight_ratio = max_weight_ratio
        self.optimize_thresholds = optimize_thresholds
        
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
        Initialize a random population of weight vectors with constraints.
        If optimize_thresholds is True, also include threshold parameters in chromosomes.
        
        Returns:
            numpy.ndarray: Initial population of chromosomes
        """
        # Determine chromosome size based on whether thresholds are optimized
        if self.optimize_thresholds:
            # Add 2 extra genes for buy_threshold and sell_threshold
            chromosome_size = self.num_weights + 2
        else:
            chromosome_size = self.num_weights
        
        # Create population with random weights between 0 and 1
        population = np.random.uniform(low=0.0, high=1.0, 
                                       size=(self.population_size, chromosome_size))
        
        # Add equal-weight chromosomes to ensure diversity
        equal_weights = np.ones(self.num_weights) / self.num_weights
        
        # Replace first few chromosomes with slightly perturbed equal weights
        for i in range(min(3, self.population_size)):
            # Start with equal weights and add small random perturbation
            perturbation = np.random.uniform(-0.1, 0.1, size=self.num_weights)
            perturbed_weights = equal_weights + perturbation
            # Ensure no negative weights
            perturbed_weights = np.maximum(perturbed_weights, 0.01)
            # Normalize
            population[i, :self.num_weights] = perturbed_weights / perturbed_weights.sum()
            
            # If optimizing thresholds, set reasonable initial values
            if self.optimize_thresholds:
                # Standard buy_threshold around 0.3-0.7
                population[i, -2] = np.random.uniform(0.3, 0.7)
                # Standard sell_threshold around -0.3 to -0.7
                population[i, -1] = np.random.uniform(0.3, 0.7) * -1
        
        # Normalize weight portion of each chromosome so weights sum to 1
        for i in range(self.population_size):
            weight_sum = np.sum(population[i, :self.num_weights])
            if weight_sum > 0:
                population[i, :self.num_weights] = population[i, :self.num_weights] / weight_sum
        
        # If optimizing thresholds, ensure they're in sensible ranges for the rest of chromosomes
        if self.optimize_thresholds:
            # Buy thresholds (0.1 to 0.9)
            population[:, -2] = np.random.uniform(0.1, 0.9, size=self.population_size)
            # Sell thresholds (-0.9 to -0.1)
            population[:, -1] = np.random.uniform(0.1, 0.9, size=self.population_size) * -1
            
        return population

    def _calculate_fitness(self, chromosome, data_handler=None):
        """
        Calculate fitness for a single chromosome by backtesting the weighted strategy.

        Args:
            chromosome: Array containing weights and possibly threshold parameters
            data_handler: Optional specific data handler for cross-validation

        Returns:
            float: Fitness score based on the optimization metric with regularization
        """
        # Use provided data handler or default
        dh = data_handler or self.data_handler
        
        # Extract weights and thresholds from chromosome
        if self.optimize_thresholds:
            weights = chromosome[:self.num_weights]
            buy_threshold = chromosome[-2]
            sell_threshold = chromosome[-1]
        else:
            weights = chromosome
            buy_threshold = 0.5  # Default value
            sell_threshold = -0.5  # Default value
        
        # Create a copy of WeightedRuleStrategy with the weighted combination method
        from genetic_optimizer import WeightedRuleStrategy
        weighted_strategy = WeightedRuleStrategy(
            rule_objects=self.rule_objects,
            weights=weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )

        # Reset the strategy and run backtest
        weighted_strategy.reset()
        backtester = Backtester(dh, weighted_strategy)
        results = backtester.run(use_test_data=False)  # Train on training data

        # Base fitness calculation based on selected metric
        if self.optimization_metric == 'sharpe':
            if results['num_trades'] >= 5:  # Require minimum trades for meaningful results
                base_fitness = backtester.calculate_sharpe()
            else:
                # Penalize strategies with too few trades
                base_fitness = -1.0
        elif self.optimization_metric == 'return':
            base_fitness = results['total_log_return'] if results['num_trades'] > 0 else -1.0
        elif self.optimization_metric == 'risk_adjusted':
            # Custom risk-adjusted return metric
            if results['num_trades'] >= 5:
                returns = [trade[5] for trade in results['trades']]
                mean_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else float('inf')
                downside = sum(r for r in returns if r < 0) if any(r < 0 for r in returns) else -0.0001
                base_fitness = mean_return / (std_return * abs(downside)) if abs(downside) > 0 else 0
            else:
                base_fitness = -1.0
        elif self.optimization_metric == 'win_rate':
            # Calculate win rate (percentage of profitable trades)
            if results['num_trades'] >= 5:  # Require minimum trades for meaningful results
                win_count = sum(1 for trade in results['trades'] if trade[5] > 0)
                win_rate = win_count / results['num_trades']
                # We might also want to ensure a minimum number of trades
                trade_count_factor = min(1.0, results['num_trades'] / 20)  # Reaches 1.0 at 20+ trades
                base_fitness = win_rate * trade_count_factor
            else:
                # Penalize strategies with too few trades
                base_fitness = -1.0
        else:
            # Default to total return
            base_fitness = results['total_log_return'] if 'total_log_return' in results else -1.0

        # Apply regularization - penalize extreme weights and encourage more balanced weights
        if self.optimize_thresholds:
            weights = chromosome[:self.num_weights]
        else:
            weights = chromosome
            
        weight_variance = np.var(weights)  # Variance measures how spread out the weights are
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        
        # Check if max/min ratio exceeds limit (handles division by zero)
        weight_ratio_penalty = 0
        if min_weight > 0:
            ratio = max_weight / min_weight
            if ratio > self.max_weight_ratio:
                weight_ratio_penalty = (ratio - self.max_weight_ratio) * 0.1  # Scale penalty
        
        # Calculate regularization terms
        regularization_term = weight_variance * self.regularization_factor
        
        # Equal weight direction - penalizes deviation from equal weights
        equal_weights = np.ones(self.num_weights) / self.num_weights
        balance_term = np.sum(np.abs(weights - equal_weights)) * self.balance_factor

        # Calculate number of "effective" rules (how many rules have meaningful weight)
        # This rewards using more rules rather than concentrating on just a few
        effective_rules = 1.0 / np.sum(weights**2)
        diversity_bonus = 0.05 * (effective_rules / self.num_weights)
        
        # Add threshold regularization if optimizing thresholds
        threshold_penalty = 0
        if self.optimize_thresholds:
            buy_threshold = chromosome[-2]
            sell_threshold = chromosome[-1]
            
            # Penalize thresholds that are too close to zero or too extreme
            if buy_threshold < 0.1:
                threshold_penalty += (0.1 - buy_threshold) * 2
            elif buy_threshold > 0.9:
                threshold_penalty += (buy_threshold - 0.9) * 2
                
            if sell_threshold > -0.1:
                threshold_penalty += (sell_threshold + 0.1) * 2
            elif sell_threshold < -0.9:
                threshold_penalty += (-0.9 - sell_threshold) * 2
                
            # Penalize if buy and sell thresholds are too close
            threshold_gap = buy_threshold - sell_threshold
            if threshold_gap < 0.3:
                threshold_penalty += (0.3 - threshold_gap) * 3
        
        # Final fitness with regularization
        final_fitness = base_fitness - regularization_term - balance_term + diversity_bonus - weight_ratio_penalty - threshold_penalty
        
        # Add info about trades and parameters
        trade_info = {
            'num_trades': results['num_trades'] if 'num_trades' in results else 0,
            'total_return': results['total_percent_return'] if 'total_percent_return' in results else 0
        }
        
        # Add threshold values to trade info if optimizing
        if self.optimize_thresholds:
            trade_info['buy_threshold'] = chromosome[-2]
            trade_info['sell_threshold'] = chromosome[-1]

        # Clean up
        del weighted_strategy
        del backtester
        del results
        
        return final_fitness, trade_info
        
    def _calculate_population_fitness(self, population):
        """
        Calculate fitness for the entire population using cross-validation.
        
        Args:
            population: Array of chromosomes
            
        Returns:
            numpy.ndarray: Fitness scores for each chromosome
        """
        fitness_scores = np.zeros(len(population))
        trade_info_list = []
        
        # Check if we should use cross-validation
        if self.cv_folds > 1:
            # Create cross-validation folds from training data
            cv_folds = self._create_cv_folds(self.cv_folds)
            
            for i, chromosome in enumerate(population):
                fold_scores = []
                fold_trade_info = []
                
                # Test on each fold
                for train_dh, test_dh in cv_folds:
                    # Calculate fitness on training data
                    train_fitness, train_info = self._calculate_fitness(chromosome, train_dh)
                    # Calculate fitness on validation data
                    test_fitness, test_info = self._calculate_fitness(chromosome, test_dh)
                    
                    # Use validation score as the true fitness
                    fold_scores.append(test_fitness)
                    fold_trade_info.append(test_info)
                
                # Average the scores across folds
                fitness_scores[i] = np.mean(fold_scores)
                avg_trade_info = {}
                for key in fold_trade_info[0].keys():
                    avg_trade_info[key] = np.mean([info[key] for info in fold_trade_info])
                trade_info_list.append(avg_trade_info)
        else:
            # Process without cross-validation
            if self.batch_size and self.batch_size < len(population):
                # Process in batches for memory efficiency
                num_batches = (len(population) + self.batch_size - 1) // self.batch_size
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(population))
                    
                    for i in range(start_idx, end_idx):
                        fitness_scores[i], trade_info = self._calculate_fitness(population[i])
                        trade_info_list.append(trade_info)
                    
                    # Force garbage collection after each batch
                    gc.collect()
            else:
                # Process all chromosomes
                for i, chromosome in enumerate(population):
                    fitness_scores[i], trade_info = self._calculate_fitness(chromosome)
                    trade_info_list.append(trade_info)
        
        return fitness_scores, trade_info_list

    def _create_cv_folds(self, n_folds):
        """
        Create cross-validation folds from the training data.
        
        Args:
            n_folds: Number of folds
            
        Returns:
            list: List of (train_dh, test_dh) tuples
        """
        # Save current state of data handler
        original_index = self.data_handler.current_train_index
        self.data_handler.reset_train()
        
        # Collect all training bars
        all_bars = []
        while True:
            bar = self.data_handler.get_next_train_bar()
            if bar is None:
                break
            all_bars.append(bar)
            
        # Restore original state
        self.data_handler.reset_train()
        self.data_handler.current_train_index = original_index
        
        # Create folds
        fold_size = len(all_bars) // n_folds
        folds = []
        
        for i in range(n_folds):
            # Define validation range
            val_start = i * fold_size
            val_end = val_start + fold_size if i < n_folds - 1 else len(all_bars)
            
            # Validation bars for this fold
            val_bars = all_bars[val_start:val_end]
            
            # Training bars for this fold (all others)
            train_bars = all_bars[:val_start] + all_bars[val_end:]
            
            # Create custom data handlers for this fold
            train_dh = self._create_custom_data_handler(train_bars)
            val_dh = self._create_custom_data_handler(val_bars)
            
            folds.append((train_dh, val_dh))
            
        return folds
    
    def _create_custom_data_handler(self, bars):
        """
        Create a custom data handler with specific bars.
        
        Args:
            bars: List of bar dictionaries
            
        Returns:
            object: A data handler-like object
        """
        # Simple data handler that works with the bars provided
        class CustomDataHandler:
            def __init__(self, train_bars, test_bars=None):
                self.train_bars = train_bars
                self.test_bars = test_bars or train_bars  # Default to same bars for test
                self.current_train_index = 0
                self.current_test_index = 0
            
            def get_next_train_bar(self):
                if self.current_train_index < len(self.train_bars):
                    bar = self.train_bars[self.current_train_index]
                    self.current_train_index += 1
                    return bar
                return None
            
            def get_next_test_bar(self):
                if self.current_test_index < len(self.test_bars):
                    bar = self.test_bars[self.current_test_index]
                    self.current_test_index += 1
                    return bar
                return None
            
            def reset_train(self):
                self.current_train_index = 0
            
            def reset_test(self):
                self.current_test_index = 0
        
        return CustomDataHandler(bars)

    def _select_parents(self, population, fitness, trade_info_list):
        """
        Select the best parents for producing the offspring of the next generation.
        
        Args:
            population: Current population
            fitness: Fitness scores for the population
            trade_info_list: List of dictionaries with trade info
            
        Returns:
            numpy.ndarray: Selected parent chromosomes
        """
        chromosome_size = population.shape[1]
        parents = np.empty((self.num_parents, chromosome_size))
        parent_info = []
        
        # Copy fitness array to avoid modifying original
        fitness_copy = fitness.copy()
        
        for i in range(self.num_parents):
            max_fitness_idx = np.argmax(fitness_copy)
            parents[i] = population[max_fitness_idx].copy()  # Make a copy to avoid reference issues
            parent_info.append(trade_info_list[max_fitness_idx])
            fitness_copy[max_fitness_idx] = -float('inf')  # Ensure this parent isn't selected again
        
        return parents, parent_info

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
        
        # Determine chromosome size 
        parent_size = parents.shape[1]
        
        offspring_size = self.population_size - self.num_parents
        offspring = np.empty((offspring_size, parent_size))
        
        for k in range(offspring_size):
            # Select two parents with tournament selection (select 3, pick best 2)
            indices = np.random.choice(self.num_parents, 3, replace=False)
            sorted_indices = sorted(indices, key=lambda i: np.sum(np.abs(parents[i, :self.num_weights] - np.ones(self.num_weights)/self.num_weights)))
            parent1_idx, parent2_idx = sorted_indices[0], sorted_indices[1]
            
            # Choose crossover method
            if self.optimize_thresholds:
                # 20% chance to use separate crossover for weights and thresholds
                if np.random.random() < 0.2:
                    # Separate crossover
                    # For weights
                    weight_ratio = np.random.random() * 0.6 + 0.2  # Between 0.2 and 0.8
                    offspring[k, :self.num_weights] = (weight_ratio * parents[parent1_idx, :self.num_weights] + 
                                              (1 - weight_ratio) * parents[parent2_idx, :self.num_weights])
                    
                    # For thresholds, randomly choose one parent's thresholds or average them
                    threshold_method = np.random.choice(['parent1', 'parent2', 'average'])
                    if threshold_method == 'parent1':
                        offspring[k, -2:] = parents[parent1_idx, -2:]
                    elif threshold_method == 'parent2':
                        offspring[k, -2:] = parents[parent2_idx, -2:]
                    else:  # average
                        offspring[k, -2:] = (parents[parent1_idx, -2:] + parents[parent2_idx, -2:]) / 2
                else:
                    # Uniform crossover with same ratio for everything
                    crossover_ratio = np.random.random() * 0.6 + 0.2  # Between 0.2 and 0.8
                    offspring[k] = (crossover_ratio * parents[parent1_idx] + 
                                  (1 - crossover_ratio) * parents[parent2_idx])
            else:
                # Introduce diversity occasionally by using extreme crossover
                if np.random.random() < 0.1:  # 10% chance
                    # Create extreme combinations (favor either first or second parent strongly)
                    crossover_ratio = np.random.choice([0.9, 0.1])
                else:
                    # Normally use a random crossover ratio
                    crossover_ratio = np.random.random() * 0.6 + 0.2  # Between 0.2 and 0.8
                
                offspring[k] = (crossover_ratio * parents[parent1_idx] + 
                              (1 - crossover_ratio) * parents[parent2_idx])
            
            # Ensure weight normalization
            weight_sum = np.sum(offspring[k, :self.num_weights])
            if weight_sum > 0:
                offspring[k, :self.num_weights] = offspring[k, :self.num_weights] / weight_sum
            
        return offspring

    def _mutate(self, offspring):
        """
        Mutate the offspring by introducing random changes with constraints.
        
        Args:
            offspring: Offspring chromosomes after crossover
            
        Returns:
            numpy.ndarray: Mutated offspring
        """
        # Reset random seed for reproducibility if needed
        self._set_random_seed()
        
        # Determine chromosome size
        chromosome_size = offspring.shape[1]
        
        for i in range(len(offspring)):
            # Determine if this chromosome should be mutated
            if np.random.random() < self.mutation_rate:
                mutation_type = np.random.choice(['standard', 'rebalance', 'reset', 'threshold'])
                
                if mutation_type == 'standard':
                    # Standard mutation: tweak two random weights
                    idx1, idx2 = np.random.choice(self.num_weights, 2, replace=False)
                    
                    # Get a random change amount (ensuring sum remains the same)
                    change = np.random.uniform(-offspring[i, idx1]/2, offspring[i, idx1]/2)
                    
                    # Apply mutation
                    offspring[i, idx1] -= change
                    offspring[i, idx2] += change
                    
                elif mutation_type == 'rebalance':
                    # Rebalance mutation: move weights slightly toward equal weights
                    equal_weights = np.ones(self.num_weights) / self.num_weights
                    rebalance_strength = np.random.uniform(0.05, 0.2)  # 5-20% rebalance
                    offspring[i, :self.num_weights] = (1 - rebalance_strength) * offspring[i, :self.num_weights] + rebalance_strength * equal_weights
                    
                elif mutation_type == 'reset':
                    # Reset mutation: randomly reset 1-3 weights
                    num_to_reset = np.random.randint(1, 4)  # 1 to 3 weights
                    reset_indices = np.random.choice(self.num_weights, num_to_reset, replace=False)
                    
                    # Remove weight from selected indices
                    total_removed = sum(offspring[i, idx] for idx in reset_indices)
                    for idx in reset_indices:
                        offspring[i, idx] = 0
                    
                    # Redistribute to other weights proportionally
                    remaining_indices = [j for j in range(self.num_weights) if j not in reset_indices]
                    if remaining_indices:
                        remaining_total = sum(offspring[i, idx] for idx in remaining_indices)
                        if remaining_total > 0:
                            for idx in remaining_indices:
                                offspring[i, idx] += total_removed * (offspring[i, idx] / remaining_total)
                        else:
                            # If all remaining weights are 0, distribute equally
                            for idx in remaining_indices:
                                offspring[i, idx] = total_removed / len(remaining_indices)
                
                elif mutation_type == 'threshold' and self.optimize_thresholds:
                    # Threshold mutation: adjust buy or sell threshold
                    if np.random.random() < 0.5:
                        # Mutate buy threshold (index -2)
                        offspring[i, -2] += np.random.uniform(-0.2, 0.2)
                        # Ensure it stays in sensible range
                        offspring[i, -2] = np.clip(offspring[i, -2], 0.1, 0.9)
                    else:
                        # Mutate sell threshold (index -1)
                        offspring[i, -1] += np.random.uniform(-0.2, 0.2)
                        # Ensure it stays in sensible range
                        offspring[i, -1] = np.clip(offspring[i, -1], -0.9, -0.1)
                
                # Ensure no negative weights
                offspring[i, :self.num_weights] = np.maximum(offspring[i, :self.num_weights], 0)
                    
                # Re-normalize to ensure weights sum to 1
                weight_sum = np.sum(offspring[i, :self.num_weights])
                if weight_sum > 0:
                    offspring[i, :self.num_weights] = offspring[i, :self.num_weights] / weight_sum
                else:
                    # Fallback to equal weights if all weights become 0
                    offspring[i, :self.num_weights] = np.ones(self.num_weights) / self.num_weights
                
                # If optimizing thresholds, ensure buy threshold is higher than sell threshold
                if self.optimize_thresholds:
                    buy_threshold = offspring[i, -2]
                    sell_threshold = offspring[i, -1]
                    
                    # If thresholds are inverted or too close, adjust them
                    if buy_threshold <= abs(sell_threshold) or (buy_threshold - abs(sell_threshold)) < 0.1:
                        gap = np.random.uniform(0.3, 0.6)  # Desired gap between thresholds
                        midpoint = (buy_threshold + abs(sell_threshold)) / 2
                        
                        # Set new thresholds centered around midpoint
                        offspring[i, -2] = min(0.9, midpoint + gap/2)
                        offspring[i, -1] = max(-0.9, -(midpoint + gap/2))
                
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
            print(f"Starting enhanced genetic optimization with {self.num_weights} rules...")
            print(f"Population size: {self.population_size}, Generations: {self.num_generations}")
            print(f"Cross-validation folds: {self.cv_folds}, Regularization: {self.regularization_factor}")
            if self.deterministic:
                print(f"Running in deterministic mode with seed: {self.random_seed}")

        # Reset the random seed at the beginning of optimization
        self._set_random_seed()

        # Initialize population
        population = self._initialize_population()

        # Variables for early stopping
        generations_without_improvement = 0
        last_best_fitness = float('-inf')
        
        # Clear any previous history
        self.fitness_history = []
        self.trade_info_history = []

        # Run genetic algorithm for specified number of generations
        for generation in range(self.num_generations):
            # Calculate fitness for current population
            fitness, trade_info_list = self._calculate_population_fitness(population)

            # Track best fitness in this generation
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            self.fitness_history.append(best_fitness)
            self.trade_info_history.append(trade_info_list[best_idx])

            # Update best weights and thresholds if improved
            if self.best_fitness is None or best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                
                # Store weights and thresholds separately for clarity
                if self.optimize_thresholds:
                    self.best_weights = population[best_idx, :self.num_weights].copy()
                    self.best_thresholds = population[best_idx, -2:].copy()
                else:
                    self.best_weights = population[best_idx].copy()
                    self.best_thresholds = np.array([0.5, -0.5])  # Default values
                
                self.best_trade_info = trade_info_list[best_idx]

            # Print progress
            if verbose and (generation % 5 == 0 or generation == self.num_generations - 1):
                elapsed = time.time() - start_time
                best_trade_info = trade_info_list[best_idx]
                print(f"Generation {generation+1}/{self.num_generations} | "
                      f"Best Fitness: {best_fitness:.4f} | "
                      f"Return: {best_trade_info['total_return']:.2f}% | "
                      f"Trades: {best_trade_info['num_trades']} | "
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

            # Select parents
            parents, parent_info = self._select_parents(population, fitness, trade_info_list)

            # Create offspring through crossover
            offspring = self._crossover(parents)

            # Apply mutation
            offspring = self._mutate(offspring)

            # Create new population for next generation
            population[:self.num_parents] = parents
            population[self.num_parents:] = offspring
            
            # Force garbage collection every few generations
            if generation % 5 == 0:
                gc.collect()

        if verbose:
            total_time = time.time() - start_time
            print(f"\nOptimization completed in {total_time:.1f} seconds")
            print(f"Best fitness: {self.best_fitness:.4f}")
            print(f"Optimal weights: {self.best_weights}")
            print(f"Effective number of rules: {1.0 / np.sum(self.best_weights**2):.1f} of {self.num_weights}")
            
            if self.optimize_thresholds:
                buy_threshold, sell_threshold = self.best_thresholds
                print(f"Optimal thresholds: Buy: {buy_threshold:.4f}, Sell: {sell_threshold:.4f}")
            
        # Clean up memory
        del population
        del parents
        del offspring
        gc.collect()

        # Return the best parameters
        if self.optimize_thresholds:
            return {
                'weights': self.best_weights,
                'buy_threshold': self.best_thresholds[0],
                'sell_threshold': self.best_thresholds[1]
            }
        else:
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
        if not self.fitness_history:
            print("No fitness history available to plot.")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot fitness
        ax1.plot(range(1, len(self.fitness_history) + 1), self.fitness_history)
        ax1.set_title('Fitness Evolution Over Generations')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.grid(True)
        
        # Plot trade metrics from history
        if self.trade_info_history:
            generations = range(1, len(self.trade_info_history) + 1)
            returns = [info['total_return'] for info in self.trade_info_history]
            trades = [info['num_trades'] for info in self.trade_info_history]
            
            ax2.plot(generations, returns, 'g-', label='Return (%)')
            
            # Create second y-axis for number of trades
            ax3 = ax2.twinx()
            ax3.plot(generations, trades, 'r--', label='Trades')
            
            ax2.set_title('Performance Metrics Over Generations')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Return (%)')
            ax3.set_ylabel('Number of Trades')
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def cleanup(self):
        """
        Clean up resources and free memory.
        """
        # Keep only the best weights and fitness
        self.fitness_history = []
        self.trade_info_history = []
        gc.collect()


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
            signal_object = rule.on_bar(bar)  # Now we expect a Signal object
            if signal_object and hasattr(signal_object, 'signal_type'):  # Check for signal_type
                combined_signals.append(signal_object.signal_type.value * self.weights[i])
            else:
                combined_signals.append(0)  # Or handle missing signal appropriately

        weighted_sum = np.sum(combined_signals)

        if weighted_sum > self.buy_threshold:
            final_signal_type = SignalType.BUY
        elif weighted_sum < self.sell_threshold:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL

        return Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id="weighted_strategy"
        )

    def reset(self):
        for rule in self.rule_objects:
            if hasattr(rule, 'reset'):
                rule.reset()
        self.rule_signals = [None] * len(self.rule_objects)        
