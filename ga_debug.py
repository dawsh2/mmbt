"""
Debug script for genetic algorithm optimization to compare in-sample and out-of-sample performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy


def plot_equity_curve(trades, title, initial_capital=10000):
    """Plot equity curve from trade data."""
    equity = [initial_capital]
    for trade in trades:
        log_return = trade[5]
        equity.append(equity[-1] * np.exp(log_return))
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title(title)
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def run_ga_debug():
    """Run GA optimization and compare in-sample and out-of-sample performance."""
    # Define filepath
    filepath = "data/data.csv"  # Update this path to your data file location
    
    print(f"Looking for data file at: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir())
        exit(1)
    
    # 1. Load and prepare data
    print("\n=== Loading Data ===")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    # 2. Train basic rules and select top performers
    print("\n=== Training Basic Rules ===")
    rules_config = [
        (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30, 50]}),
        (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]}),
        (Rule2, {'ema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule3, {'ema1_period': [10, 20], 'ema2_period': [30, 50]}),
        (Rule4, {'dema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule5, {'dema1_period': [10, 20], 'dema2_period': [30, 50]}),
    ]
    
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    print("\nSelected Top Rules:")
    for i, rule in enumerate(top_rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # 3. Run baseline strategy on both training and test data
    print("\n=== Running Baseline Strategy ===")
    top_n_strategy = rule_system.get_top_n_strategy()
    
    # 3.1 In-sample (training) results for baseline
    baseline_backtester_train = Backtester(data_handler, top_n_strategy)
    baseline_results_train = baseline_backtester_train.run(use_test_data=False)
    
    # 3.2 Out-of-sample (test) results for baseline
    baseline_backtester_test = Backtester(data_handler, top_n_strategy)
    baseline_results_test = baseline_backtester_test.run(use_test_data=True)
    
    print("\nBaseline Strategy Performance:")
    print(f"  IN-SAMPLE Total Return: {baseline_results_train['total_percent_return']:.2f}%")
    print(f"  IN-SAMPLE Sharpe Ratio: {baseline_backtester_train.calculate_sharpe():.4f}")
    print(f"  IN-SAMPLE Trades: {baseline_results_train['num_trades']}")
    print(f"  OUT-OF-SAMPLE Total Return: {baseline_results_test['total_percent_return']:.2f}%")
    print(f"  OUT-OF-SAMPLE Sharpe Ratio: {baseline_backtester_test.calculate_sharpe():.4f}")
    print(f"  OUT-OF-SAMPLE Trades: {baseline_results_test['num_trades']}")
    
    # 4. Run genetic optimization
    print("\n=== Running Genetic Optimization ===")
    
    # Create genetic optimizer with sharpe ratio as the objective
    genetic_optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=20,
        num_generations=30,
        optimization_metric='sharpe'  # Try different metrics: 'sharpe', 'return', 'win_rate'
    )
    
    # Run optimization
    optimal_weights = genetic_optimizer.optimize(verbose=True)
    print(f"Optimal weights: {optimal_weights}")
    
    # Create weighted strategy with optimal weights
    weighted_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights
    )
    
    # 5.1 In-sample (training) results for optimized strategy
    ga_backtester_train = Backtester(data_handler, weighted_strategy)
    ga_results_train = ga_backtester_train.run(use_test_data=False)
    
    # 5.2 Out-of-sample (test) results for optimized strategy
    ga_backtester_test = Backtester(data_handler, weighted_strategy)
    ga_results_test = ga_backtester_test.run(use_test_data=True)
    
    print("\nGenetically Optimized Strategy Performance:")
    print(f"  IN-SAMPLE Total Return: {ga_results_train['total_percent_return']:.2f}%")
    print(f"  IN-SAMPLE Sharpe Ratio: {ga_backtester_train.calculate_sharpe():.4f}")
    print(f"  IN-SAMPLE Trades: {ga_results_train['num_trades']}")
    print(f"  OUT-OF-SAMPLE Total Return: {ga_results_test['total_percent_return']:.2f}%")
    print(f"  OUT-OF-SAMPLE Sharpe Ratio: {ga_backtester_test.calculate_sharpe():.4f}")
    print(f"  OUT-OF-SAMPLE Trades: {ga_results_test['num_trades']}")
    
    # 6. Compare results
    print("\n=== Performance Comparison ===")
    print(f"{'Strategy':<25} {'Train Return':<15} {'Train Sharpe':<15} {'Test Return':<15} {'Test Sharpe':<15}")
    print("-" * 85)
    print(f"{'Baseline (Equal Weights)':<25} {baseline_results_train['total_percent_return']:>13.2f}% {baseline_backtester_train.calculate_sharpe():>14.4f} {baseline_results_test['total_percent_return']:>13.2f}% {baseline_backtester_test.calculate_sharpe():>14.4f}")
    print(f"{'Genetically Optimized':<25} {ga_results_train['total_percent_return']:>13.2f}% {ga_backtester_train.calculate_sharpe():>14.4f} {ga_results_test['total_percent_return']:>13.2f}% {ga_backtester_test.calculate_sharpe():>14.4f}")
    
    # Check if optimization improved IN-SAMPLE performance
    if ga_results_train['total_percent_return'] <= baseline_results_train['total_percent_return']:
        print("\nWARNING: Genetic optimization did not improve in-sample returns!")
        print("This suggests a problem with the optimization implementation.")
        
        # Additional diagnostics
        print("\n=== Diagnostic Information ===")
        print(f"Fitness History: {genetic_optimizer.fitness_history}")
        print(f"Best Fitness: {genetic_optimizer.best_fitness}")
        
        # Check if weights sum to 1
        weight_sum = sum(optimal_weights)
        print(f"Sum of weights: {weight_sum}")
        if abs(weight_sum - 1.0) > 0.01:
            print("WARNING: Weights do not sum to 1.0, which is required!")
    
    # 7. Plot fitness history
    plt.figure(figsize=(12, 6))
    plt.plot(genetic_optimizer.fitness_history)
    plt.title('Genetic Optimization Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig("GA_Fitness_History.png")
    plt.show()
    
    # 8. Plot equity curves
    plot_equity_curve(baseline_results_train["trades"], "Baseline Strategy Equity Curve (IN-SAMPLE)")
    plot_equity_curve(baseline_results_test["trades"], "Baseline Strategy Equity Curve (OUT-OF-SAMPLE)")
    plot_equity_curve(ga_results_train["trades"], "Optimized Strategy Equity Curve (IN-SAMPLE)")
    plot_equity_curve(ga_results_test["trades"], "Optimized Strategy Equity Curve (OUT-OF-SAMPLE)")
    
    # Return all results for further analysis
    return {
        'baseline_train': baseline_results_train,
        'baseline_test': baseline_results_test,
        'ga_train': ga_results_train,
        'ga_test': ga_results_test,
        'optimal_weights': optimal_weights,
        'fitness_history': genetic_optimizer.fitness_history
    }


if __name__ == "__main__":
    results = run_ga_debug()
    
    # If needed, you can analyze the results further here
    # For example, compute in-sample vs out-of-sample percentage differences
    baseline_train_return = results['baseline_train']['total_percent_return']
    baseline_test_return = results['baseline_test']['total_percent_return']
    ga_train_return = results['ga_train']['total_percent_return']
    ga_test_return = results['ga_test']['total_percent_return']
    
    baseline_diff = baseline_test_return - baseline_train_return
    ga_diff = ga_test_return - ga_train_return
    
    print("\n=== Additional Analysis ===")
    print(f"Baseline in-sample to out-of-sample difference: {baseline_diff:.2f}%")
    print(f"GA optimized in-sample to out-of-sample difference: {ga_diff:.2f}%")
    
    print("\nDone! Results and charts saved.")
