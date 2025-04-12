"""
Run genetic algorithm optimization with existing trading rules.
"""

import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import argparse
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7,
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from memory_profiler import profile


#@profile
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
    plt.close()


#@profile
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run genetic algorithm optimization')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic mode (default seed 42)')
    parser.add_argument('--population', type=int, default=8, help='Population size for GA')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations for GA')
    parser.add_argument('--metric', type=str, default='return', choices=['return', 'sharpe', 'win_rate', 'risk_adjusted'],
                        help='Optimization metric to use')
    return parser.parse_args()


#@profile
def run_genetic_optimization():
    """Main function to run genetic algorithm optimization."""
    # Parse command line arguments
    args = parse_arguments()

    # Define filepath - adjust as needed
    filepath = "data/data.csv"  # Update this path to your data file location

    print(f"Looking for data file at: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir())
        exit(1)
    else:
        print(f"File found, starting optimization...")

    # 1. Load and prepare data
    start_time = time.time()
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    # 2. Train basic rules and select top performers
    print("\n=== Training Basic Rules ===")
    # Simplified rule config for faster execution
    rules_config = [
        (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30, 50]}),
        (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]}),
        (Rule2, {'ema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule3, {'ema1_period': [10, 20], 'ema2_period': [30, 50]}),
        (Rule4, {'dema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule5, {'dema1_period': [10, 20], 'dema2_period': [30, 50]}),
        (Rule6, {'tema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule7, {'stoch1_period': [10, 14], 'stochma2_period': [3, 5]}),
        (Rule8, {'vortex1_period': [10, 14], 'vortex2_period': [10, 14]}),
        (Rule9, {'p1': [9, 12], 'p2': [26, 30]}),
        (Rule10, {'rsi1_period': [10, 14]}),
        (Rule11, {'cci1_period': [14, 20]}),
        (Rule12, {'rsi_period': [10, 14]}),
        (Rule13, {'stoch_period': [10, 14], 'stoch_d_period': [3, 5]}),
        (Rule14, {'atr_period': [14, 20]}),
        (Rule15, {'bb_period': [20, 25], 'bb_std_dev_multiplier': [1.5, 2.0, 2.5]})
    ]

    rule_system_start = time.time()
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=16)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    print(f"Rule training completed in {time.time() - rule_system_start:.2f} seconds")

    print("\nSelected Top Rules:")
    for i, rule in enumerate(top_rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")

    # 3. Run baseline strategy
    print("\n=== Running Baseline (Equal-Weighted) Strategy ===")
    top_n_strategy = rule_system.get_top_n_strategy()
    baseline_backtester = Backtester(data_handler, top_n_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)

    print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {baseline_results['num_trades']}")
    sharpe_baseline = baseline_backtester.calculate_sharpe()
    print(f"Sharpe Ratio: {sharpe_baseline:.4f}")

    plot_equity_curve(baseline_results["trades"], "Baseline Strategy Equity Curve")

    # 4. Run genetic optimization
    print("\n=== Running Genetic Optimization ===")
    ga_start = time.time()

    # Additional info about seed settings
    seed_info = ""
    if args.deterministic:
        seed_info = " with deterministic mode (seed 42)"
    elif args.seed is not None:
        seed_info = f" with fixed seed {args.seed}"

    print(f"Optimizing using {args.metric.upper()} metric{seed_info}")
    print(f"Population size: {args.population}, Generations: {args.generations}")

    genetic_optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=args.population,
        num_generations=args.generations,
        optimization_metric=args.metric,
        random_seed=args.seed,
        deterministic=args.deterministic
    )

    # Run optimization
    optimal_weights = genetic_optimizer.optimize(
        verbose=True,
        early_stopping_generations=4,  # Stop after 4 generations without improvement
        min_improvement=0.0005  # Consider improvements below 0.0005 as insignificant
    )
    print(f"Genetic optimization completed in {time.time() - ga_start:.2f} seconds")
    print(f"Optimal weights: {optimal_weights}")

    # Create weighted strategy with optimal weights
    weighted_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights
    )

    # Backtest the optimized strategy
    print("\n=== Backtesting Genetically Optimized Strategy ===")
    weighted_backtester = Backtester(data_handler, weighted_strategy)
    ga_results = weighted_backtester.run(use_test_data=True)

    print(f"Total Return: {ga_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {ga_results['num_trades']}")
    sharpe_ga = weighted_backtester.calculate_sharpe()
    print(f"Sharpe Ratio: {sharpe_ga:.4f}")

    plot_equity_curve(ga_results["trades"], "Genetically Optimized Strategy Equity Curve")

    # Compare results
    improvement = (ga_results['total_percent_return'] / baseline_results['total_percent_return'] - 1) * 100 if baseline_results['total_percent_return'] > 0 else 0
    print("\n=== Performance Comparison ===")
    print(f"{'Strategy':<25} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 55)
    print(f"{'Baseline (Equal Weights)':<25} {baseline_results['total_percent_return']:>8.2f}% {baseline_results['num_trades']:>10} {sharpe_baseline:>9.4f}")
    print(f"{'Genetically Optimized':<25} {ga_results['total_percent_return']:>8.2f}% {ga_results['num_trades']:>10} {sharpe_ga:>9.4f}")
    if baseline_results['total_percent_return'] > 0:
        print(f"\nReturn improvement from genetic optimization: {improvement:.2f}%")

    # Save genetic optimization fitness history
    plt.figure(figsize=(12, 6))
    plt.plot(genetic_optimizer.fitness_history)
    plt.title('Genetic Optimization Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GA_Fitness_History.png")
    plt.close()

    # Save optimization details for reproducibility
    with open("optimization_details.txt", "w") as f:
        f.write(f"Optimization Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Optimization Metric: {args.metric}\n")
        f.write(f"Population Size: {args.population}\n")
        f.write(f"Number of Generations: {args.generations}\n")
        f.write(f"Deterministic Mode: {args.deterministic}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Best Fitness: {genetic_optimizer.best_fitness:.6f}\n")
        f.write(f"Optimal Weights: {optimal_weights}\n")
        f.write(f"Final Performance:\n")
        f.write(f"  Total Return: {ga_results['total_percent_return']:.2f}%\n")
        f.write(f"  Number of Trades: {ga_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {sharpe_ga:.4f}\n")

    print("\nOptimization complete! Results and charts saved.")
    print(f"Reproducibility information saved to optimization_details.txt")


# Main program
if __name__ == "__main__":
    run_genetic_optimization()
