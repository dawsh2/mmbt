"""
Enhanced trading system integrating the OptimizerManager for flexible optimization approaches.

This example demonstrates how to use the OptimizerManager to coordinate different
optimization methods and sequences for an advanced trading system.
"""

import os
import numpy as np
import pandas as pd
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


def plot_performance_comparison(results_dict, title):
    """
    Plot performance comparison across different optimization approaches.
    
    Args:
        results_dict: Dictionary mapping approach names to backtest results
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    for approach_name, results in results_dict.items():
        # Create equity curve from trade results
        equity = [1.0]  # Start with $1
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
    plt.show()


if __name__ == "__main__":
    # Load data
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    # --- Train basic rules to get our rule objects ---
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
    
    # Train rules and select top performers
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    print("\nSelected Top Rules:")
    for i, rule in enumerate(top_rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # --- Baseline Strategy ---
    print("\n=== Baseline Strategy ===")
    top_n_strategy = rule_system.get_top_n_strategy()
    baseline_backtester = Backtester(data_handler, top_n_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)
    
    print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {baseline_results['num_trades']}")
    print(f"Sharpe Ratio: {baseline_backtester.calculate_sharpe():.4f}")
    
    # --- Create Regime Detectors ---
    print("\n=== Creating Regime Detectors ===")
    trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
    volatility_detector = VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015)
    
    # --- Approach 1: Rules-First Optimization ---
    print("\n=== Approach 1: Rules-First Optimization ===")
    optimizer_rules_first = OptimizerManager(
        data_handler=data_handler,
        rule_objects=top_rule_objects
    )
    
    # Configure genetic optimization parameters
    genetic_params = {
        'genetic': {
            'population_size': 20,
            'num_generations': 30,
            'mutation_rate': 0.1
        }
    }
    
    # Run optimization
    rules_first_results = optimizer_rules_first.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.RULES_FIRST,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=genetic_params,
        verbose=True
    )
    
    # Get the optimized strategy
    rules_first_strategy = optimizer_rules_first.get_optimized_strategy()
    
    # --- Approach 2: Regimes-First Optimization ---
    print("\n=== Approach 2: Regimes-First Optimization ===")
    optimizer_regimes_first = OptimizerManager(
        data_handler=data_handler,
        rule_objects=top_rule_objects
    )
    
    # Run optimization
    regimes_first_results = optimizer_regimes_first.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.REGIMES_FIRST,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=genetic_params,
        verbose=True
    )
    
    # Get the optimized strategy
    regimes_first_strategy = optimizer_regimes_first.get_optimized_strategy()
    
    # --- Approach 3: Iterative Optimization ---
    print("\n=== Approach 3: Iterative Optimization ===")
    optimizer_iterative = OptimizerManager(
        data_handler=data_handler,
        rule_objects=top_rule_objects
    )
    
    # Configure iterative optimization
    iterative_params = {
        'genetic': {
            'population_size': 15,
            'num_generations': 20
        },
        'iterations': 3  # Number of iterations between rule and regime optimization
    }
    
    # Run optimization
    iterative_results = optimizer_iterative.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.ITERATIVE,
        metrics='sharpe',
        regime_detector=trend_detector,
        optimization_params=iterative_params,
        verbose=True
    )
    
    # Get the optimized strategy
    iterative_strategy = optimizer_iterative.get_optimized_strategy()
    
    # --- Backtest all strategies on test data ---
    print("\n=== Final Performance Comparison ===")
    
    # Run the backtests
    rules_first_backtester = Backtester(data_handler, rules_first_strategy)
    rules_first_backtest = rules_first_backtester.run(use_test_data=True)
    rules_first_sharpe = rules_first_backtester.calculate_sharpe()
    
    regimes_first_backtester = Backtester(data_handler, regimes_first_strategy)
    regimes_first_backtest = regimes_first_backtester.run(use_test_data=True)
    regimes_first_sharpe = regimes_first_backtester.calculate_sharpe()
    
    iterative_backtester = Backtester(data_handler, iterative_strategy)
    iterative_backtest = iterative_backtester.run(use_test_data=True)
    iterative_sharpe = iterative_backtester.calculate_sharpe()
    
    # Collect results for comparison
    comparison = {
        "Baseline": {
            'total_return': baseline_results['total_percent_return'],
            'num_trades': baseline_results['num_trades'],
            'sharpe': baseline_backtester.calculate_sharpe(),
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
    
    # Summary table
    print("\nOut-of-Sample Performance Comparison:")
    print(f"{'Approach':<15} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 45)
    
    for name, results in comparison.items():
        print(f"{name:<15} {results['total_return']:>8.2f}% {results['num_trades']:>10} {results['sharpe']:>9.4f}")
    
    # Plot equity curves for comparison
    plot_performance_comparison(comparison, "Optimization Approach Comparison")
    
    # --- Additional Analysis: Regime Distribution ---
    print("\n=== Regime Distribution Analysis ===")
    
    # This would analyze how the strategies performed in different regimes
    # To do this, we need to track which trades occurred in which regimes
    
    # Reset detector and analyze regime distribution
    trend_detector.reset()
    data_handler.reset_test()
    
    regime_counts = {
        RegimeType.TRENDING_UP: 0,
        RegimeType.TRENDING_DOWN: 0,
        RegimeType.RANGE_BOUND: 0,
        RegimeType.UNKNOWN: 0
    }
    
    regime_performance = {
        RegimeType.TRENDING_UP: {'returns': [], 'count': 0},
        RegimeType.TRENDING_DOWN: {'returns': [], 'count': 0},
        RegimeType.RANGE_BOUND: {'returns': [], 'count': 0},
        RegimeType.UNKNOWN: {'returns': [], 'count': 0}
    }
    
    # Map trade timestamps to regimes
    trade_regimes = {}
    
    bar_index = 0
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
            
        regime = trend_detector.detect_regime(bar)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Store timestamp to regime mapping
        trade_regimes[bar['timestamp']] = regime
        bar_index += 1
    
    # Analyze performance by regime for the iterative strategy
    for trade in iterative_backtest['trades']:
        entry_time = trade[0]
        exit_time = trade[3]
        log_return = trade[5]
        
        # Find the regime at entry (simplification - could be more complex)
        if entry_time in trade_regimes:
            regime = trade_regimes[entry_time]
            regime_performance[regime]['returns'].append(log_return)
            regime_performance[regime]['count'] += 1
    
    # Print regime distribution
    print("\nMarket Regime Distribution in Test Data:")
    total_bars = sum(regime_counts.values())
    for regime, count in regime_counts.items():
        percentage = (count / total_bars) * 100 if total_bars > 0 else 0
        print(f"  {regime.name}: {count} bars ({percentage:.1f}%)")
    
    # Print performance by regime
    print("\nIterative Strategy Performance by Regime:")
    print(f"{'Regime':<15} {'Avg Return':<12} {'# Trades':<10} {'Win Rate':<10}")
    print("-" * 47)
    
    for regime, data in regime_performance.items():
        if data['count'] > 0:
            avg_return = np.mean(data['returns'])
            win_rate = sum(1 for r in data['returns'] if r > 0) / len(data['returns']) * 100
            print(f"{regime.name:<15} {avg_return:>10.4f} {data['count']:>10} {win_rate:>9.1f}%")
    
    # --- Customize and Save Final Strategy ---
    print("\n=== Saving Best Strategy ===")
    
    # Determine which approach performed best
    best_approach = max(comparison.items(), key=lambda x: x[1]['sharpe'])[0]
    best_strategy = None
    
    if best_approach == "Baseline":
        best_strategy = top_n_strategy
    elif best_approach == "Rules-First":
        best_strategy = rules_first_strategy
    elif best_approach == "Regimes-First":
        best_strategy = regimes_first_strategy
    elif best_approach == "Iterative":
        best_strategy = iterative_strategy
    
    print(f"Best approach: {best_approach}")
    print(f"Sharpe Ratio: {comparison[best_approach]['sharpe']:.4f}")
    print(f"Total Return: {comparison[best_approach]['total_return']:.2f}%")
    
    # Here you could save the strategy to disk for later use
    # For example, using pickle:
    # import pickle
    # with open('best_strategy.pkl', 'wb') as f:
    #     pickle.dump(best_strategy, f)
    
    print("\n=== Optimization Complete ===")
