"""
Script to test the deterministic behavior of the genetic algorithm with fixed seed.
"""

import os
import numpy as np
import pandas as pd
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7,
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy

def run_optimization(random_seed=None, deterministic=False):
    """
    Run genetic optimization with specified seed settings.
    
    Args:
        random_seed: Integer seed for random number generator (or None for random behavior)
        deterministic: Whether to use deterministic mode
        
    Returns:
        tuple: (best_weights, best_fitness) from optimization
    """
    # Load data
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    # Simple error handling for file path
    if not os.path.exists(filepath):
        filepath = "data/data.csv"  # Try alternative path
        if not os.path.exists(filepath):
            print("Data file not found, please update the filepath in the script")
            return None, None
    
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    # Define minimal rule configuration for faster testing
    rules_config = [
        (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30]}),
        (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]}),
        (Rule2, {'ema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule3, {'ema1_period': [10, 20], 'ema2_period': [30, 50]})
    ]
    
    # Train rules
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=4)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    # Create genetic optimizer with specified seed settings
    optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=10,  # Small population for faster testing
        num_generations=10,  # Fewer generations for faster testing
        optimization_metric='sharpe',
        random_seed=random_seed,
        deterministic=deterministic
    )
    
    # Run optimization
    best_weights = optimizer.optimize(verbose=True)
    best_fitness = optimizer.best_fitness
    
    return best_weights, best_fitness

def test_reproducibility():
    """Test reproducibility of results with fixed seed."""
    print("=== Testing Reproducibility with Fixed Seed ===")
    
    # Run first optimization with seed 42
    print("\nRun 1 with seed 42:")
    weights1, fitness1 = run_optimization(random_seed=42)
    
    # Run second optimization with the same seed
    print("\nRun 2 with seed 42:")
    weights2, fitness2 = run_optimization(random_seed=42)
    
    # Run third optimization with a different seed
    print("\nRun 3 with seed 123:")
    weights3, fitness3 = run_optimization(random_seed=123)
    
    # Run fourth optimization with deterministic mode (implicit seed 42)
    print("\nRun 4 with deterministic mode (implicit seed 42):")
    weights4, fitness4 = run_optimization(deterministic=True)
    
    # Check if results are equal
    if weights1 is not None and weights2 is not None:
        weights_match_12 = np.allclose(weights1, weights2)
        fitness_match_12 = np.isclose(fitness1, fitness2)
        print(f"\nWeights Match (Run 1 vs Run 2): {weights_match_12}")
        print(f"Fitness Match (Run 1 vs Run 2): {fitness_match_12}")
        
        if weights3 is not None:
            weights_match_13 = np.allclose(weights1, weights3)
            fitness_match_13 = np.isclose(fitness1, fitness3)
            print(f"Weights Match (Run 1 vs Run 3): {weights_match_13}")
            print(f"Fitness Match (Run 1 vs Run 3): {fitness_match_13}")
        
        if weights4 is not None:
            weights_match_14 = np.allclose(weights1, weights4)
            fitness_match_14 = np.isclose(fitness1, fitness4)
            print(f"Weights Match (Run 1 vs Run 4): {weights_match_14}")
            print(f"Fitness Match (Run 1 vs Run 4): {fitness_match_14}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_reproducibility()
