"""
Run combined optimization (genetic + regime) with flexible optimization sequences.
"""

import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from regime_detection import (
    RegimeType, TrendStrengthRegimeDetector, 
    VolatilityRegimeDetector, RegimeManager
)
from optimizer_manager import (
    OptimizerManager, OptimizationMethod, OptimizationSequence
)


def plot_equity_curve(rules_first_backtest["trades"], "Rules-First Optimization Equity Curve")

# Approach 2: Regimes-First Optimization
print("\n=== Approach 2: Regimes-First Optimization ===")
optimizer_regimes_first = OptimizerManager(
    data_handler=data_handler,
    rule_objects=top_rule_objects
)

regimes_first_start = time.time()
regimes_first_results = optimizer_regimes_first.optimize(
    method=OptimizationMethod.GENETIC,
    sequence=OptimizationSequence.REGIMES_FIRST,
    metrics='sharpe',
    regime_detector=trend_detector,
    optimization_params=genetic_params,
    verbose=True
)
print(f"Regimes-First optimization completed in {time.time() - regimes_first_start:.2f} seconds")

# Get the optimized strategy
regimes_first_strategy = optimizer_regimes_first.get_optimized_strategy()

# Backtest the strategy
regimes_first_backtester = Backtester(data_handler, regimes_first_strategy)
regimes_first_backtest = regimes_first_backtester.run(use_test_data=True)
regimes_first_sharpe = regimes_first_backtester.calculate_sharpe()

plot_equity_curve(regimes_first_backtest["trades"], "Regimes-First Optimization Equity Curve")

# Approach 3: Iterative Optimization (if time permits)
print("\n=== Approach 3: Iterative Optimization ===")
optimizer_iterative = OptimizerManager(
    data_handler=data_handler,
    rule_objects=top_rule_objects
)

# Configure iterative optimization (use smaller values for faster execution)
iterative_params = {
    'genetic': {
        'population_size': 15,
        'num_generations': 20
    },
    'iterations': 2  # Reduce to 2 iterations for faster execution
}

iterative_start = time.time()
iterative_results = optimizer_iterative.optimize(
    method=OptimizationMethod.GENETIC,
    sequence=OptimizationSequence.ITERATIVE,
    metrics='sharpe',
    regime_detector=trend_detector,
    optimization_params=iterative_params,
    verbose=True
)
print(f"Iterative optimization completed in {time.time() - iterative_start:.2f} seconds")

# Get the optimized strategy
iterative_strategy = optimizer_iterative.get_optimized_strategy()

# Backtest the strategy
iterative_backtester = Backtester(data_handler, iterative_strategy)
iterative_backtest = iterative_backtester.run(use_test_data=True)
iterative_sharpe = iterative_backtester.calculate_sharpe()

plot_equity_curve(iterative_backtest["trades"], "Iterative Optimization Equity Curve")

# 6. Compare all approaches
print("\n=== Final Performance Comparison ===")

# Collect results for comparison
comparison = {
    "Baseline": {
        'total_return': baseline_results['total_percent_return'],
        'num_trades': baseline_results['num_trades'],
        'sharpe': sharpe_baseline,
        'trades': baseline_results['trades']
    },
    "Rules-First": {
        'total_return': rules_first_backtest['total_percent_return'],
        'num_trades': rules_first_backtest['num_trades'],
        'sharpe': rules_first_sharpe,
        'trades': rules_first_backtest['trades']
    },
    "Regimes-First": {
        'total_return': regimes_first_backtest['total_percent_return'],
        'num_trades': regimes_first_backtest['num_trades'],
        'sharpe': regimes_first_sharpe,
        'trades': regimes_first_backtest['trades']
    },
    "Iterative": {
        'total_return': iterative_backtest['total_percent_return'],
        'num_trades': iterative_backtest['num_trades'],
        'sharpe': iterative_sharpe,
        'trades': iterative_backtest['trades']
    }
}

# Print summary table
print("\nOut-of-Sample Performance Comparison:")
print(f"{'Approach':<15} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
print("-" * 45)

for name, results in comparison.items():
    print(f"{name:<15} {results['total_return']:>8.2f}% {results['num_trades']:>10} {results['sharpe']:>9.4f}")

# Plot equity curves for comparison
plot_performance_comparison(comparison, "Optimization Approach Comparison")

# 7. Determine the best approach
best_approach = max(comparison.items(), key=lambda x: x[1]['sharpe'])[0]
best_return = max(comparison.items(), key=lambda x: x[1]['total_return'])[0]

print(f"\nBest approach by Sharpe ratio: {best_approach} (Sharpe: {comparison[best_approach]['sharpe']:.4f})")
print(f"Best approach by total return: {best_return} (Return: {comparison[best_return]['total_return']:.2f}%)")

# 8. Analyze regime distribution (optional)
print("\n=== Regime Distribution Analysis ===")

# Reset detector and analyze regime distribution in test data
trend_detector.reset()
data_handler.reset_test()

regime_counts = {
    RegimeType.TRENDING_UP: 0,
    RegimeType.TRENDING_DOWN: 0,
    RegimeType.RANGE_BOUND: 0,
    RegimeType.UNKNOWN: 0
}

# Map trade timestamps to regimes
trade_regimes = {}

while True:
    bar = data_handler.get_next_test_bar()
    if bar is None:
        break
        
    regime = trend_detector.detect_regime(bar)
    regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    # Store timestamp to regime mapping
    trade_regimes[bar['timestamp']] = regime

# Print regime distribution
print("\nMarket Regime Distribution in Test Data:")
total_bars = sum(regime_counts.values())
for regime, count in regime_counts.items():
    percentage = (count / total_bars) * 100 if total_bars > 0 else 0
    print(f"  {regime.name}: {count} bars ({percentage:.1f}%)")

# Determine best strategy and save results
print("\n=== Saving Best Strategy Results ===")

if best_approach == "Baseline":
    best_strategy = top_n_strategy
    best_trades = baseline_results['trades']
elif best_approach == "Rules-First":
    best_strategy = rules_first_strategy
    best_trades = rules_first_backtest['trades']
elif best_approach == "Regimes-First":
    best_strategy = regimes_first_strategy
    best_trades = regimes_first_backtest['trades']
elif best_approach == "Iterative":
    best_strategy = iterative_strategy
    best_trades = iterative_backtest['trades']

# Plot detailed performance of best strategy
plot_equity_curve(best_trades, f"Best Strategy: {best_approach}")

print(f"\nBest strategy: {best_approach}")
print(f"Sharpe Ratio: {comparison[best_approach]['sharpe']:.4f}")
print(f"Total Return: {comparison[best_approach]['total_return']:.2f}%")
print(f"Number of Trades: {comparison[best_approach]['num_trades']}")

# Here you could save the best strategy for later use
# For example, using pickle:
# import pickle
# with open('best_strategy.pkl', 'wb') as f:
#     pickle.dump(best_strategy, f)

print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
print("\nOptimization complete! Results and charts saved.")
trades, title, initial_capital=10000):
    """Plot equity curve from trade data."""
    if not trades:
        print(f"Warning: No trades to plot for {title}")
        return
        
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
    plt.savefig(f"{title.replace(' ', '_').replace(':', '')}.png")
    plt.close()


def plot_performance_comparison(results_dict, title="Strategy Performance Comparison"):
    """Plot equity curves for multiple strategies."""
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    for approach_name, results in results_dict.items():
        if 'trades' not in results or not results['trades']:
            print(f"Warning: No trades for {approach_name}")
            continue
            
        # Create equity curve from trade results
        equity = [10000]  # Start with $10,000
        for trade in results['trades']:
            log_return = trade[5]
            equity.append(equity[-1] * np.exp(log_return))
        
        plt.plot(equity, label=f"{approach_name} ({results['total_return']:.2f}%)")
    
    plt.title(title)
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


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
    (Rule15, {'bb_period': [20, 25]}),
]

rule_system_start = time.time()
rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
rule_system.train_rules(data_handler)
top_rule_objects = list(rule_system.trained_rule_objects.values())
print(f"Rule training completed in {time.time() - rule_system_start:.2f} seconds")

print("\nSelected Top Rules:")
for i, rule in enumerate(top_rule_objects):
    rule_name = rule.__class__.__name__
    print(f"  {i+1}. {rule_name}")

# 3. Run baseline strategy
print("\n=== Running Baseline Strategy ===")
top_n_strategy = rule_system.get_top_n_strategy()
baseline_backtester = Backtester(data_handler, top_n_strategy)
baseline_results = baseline_backtester.run(use_test_data=True)

print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
print(f"Number of Trades: {baseline_results['num_trades']}")
sharpe_baseline = baseline_backtester.calculate_sharpe()
print(f"Sharpe Ratio: {sharpe_baseline:.4f}")

plot_equity_curve(baseline_results["trades"], "Baseline Strategy Equity Curve")

# 4. Create regime detector
trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)

# 5. Run different optimization approaches using OptimizerManager
print("\n=== Running Combined Optimization Approaches ===")

# Configure genetic optimization parameters (use smaller values for faster execution)
genetic_params = {
    'genetic': {
        'population_size': 15,
        'num_generations': 30,
        'mutation_rate': 0.1
    }
}

# Approach 1: Rules-First Optimization
print("\n=== Approach 1: Rules-First Optimization ===")
optimizer_rules_first = OptimizerManager(
    data_handler=data_handler,
    rule_objects=top_rule_objects
)

rules_first_start = time.time()
rules_first_results = optimizer_rules_first.optimize(
    method=OptimizationMethod.GENETIC,
    sequence=OptimizationSequence.RULES_FIRST,
    metrics='sharpe',
    regime_detector=trend_detector,
    optimization_params=genetic_params,
    verbose=True
)
print(f"Rules-First optimization completed in {time.time() - rules_first_start:.2f} seconds")

# Get the optimized strategy
rules_first_strategy = optimizer_rules_first.get_optimized_strategy()

# Backtest the strategy
rules_first_backtester = Backtester(data_handler, rules_first_strategy)
rules_first_backtest = rules_first_backtester.run(use_test_data=True)
rules_first_sharpe = rules_first_backtester.calculate_sharpe()


