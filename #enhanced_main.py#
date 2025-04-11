"""
Enhanced trading system integrating genetic optimization and regime filtering.

This example demonstrates how to use the new modules to create an advanced
trading system that adapts to different market conditions.
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


def plot_regime_performance(results_dict, title):
    """
    Plot performance comparison across different strategies/regimes.
    
    Args:
        results_dict: Dictionary mapping strategy names to backtest results
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    for strategy_name, results in results_dict.items():
        # Create equity curve from trade results
        equity = [1.0]  # Start with $1
        for trade in results['trades']:
            log_return = trade[5]
            equity.append(equity[-1] * np.exp(log_return))
        
        plt.plot(equity, label=f"{strategy_name} ({results['total_percent_return']:.2f}%)")
    
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
    
    # --- Rule Training and Strategy Building ---
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
    
    # --- Approach 1: Simple TopN Strategy (original approach) ---
    print("\n=== Approach 1: Simple TopN Strategy (Baseline) ===")
    top_n_strategy = rule_system.get_top_n_strategy()
    baseline_backtester = Backtester(data_handler, top_n_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)
    
    print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {baseline_results['num_trades']}")
    print(f"Sharpe Ratio: {baseline_backtester.calculate_sharpe():.4f}")
    
    # --- Approach 2: Genetically Optimized Weights ---
    print("\n=== Approach 2: Genetically Optimized Weights ===")
    genetic_optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=20,
        num_generations=50,
        optimization_metric='sharpe'
    )
    
    # Run optimization
    optimal_weights = genetic_optimizer.optimize()
    print(f"Optimal weights: {optimal_weights}")
    
    # Create and backtest the weighted strategy
    weighted_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights
    )
    
    weighted_backtester = Backtester(data_handler, weighted_strategy)
    weighted_results = weighted_backtester.run(use_test_data=True)
    
    print(f"Total Return: {weighted_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {weighted_results['num_trades']}")
    print(f"Sharpe Ratio: {weighted_backtester.calculate_sharpe():.4f}")
    
    # Plot fitness evolution
    genetic_optimizer.plot_fitness_history()
    
    # --- Approach 3: Regime-Based Strategy ---
    print("\n=== Approach 3: Regime-Based Adaptive Strategy ===")
    
    # Create regime detector
    regime_detector = TrendStrengthRegimeDetector(
        adx_period=14,
        adx_threshold=25
    )
    
    # Create regime manager
    regime_manager = RegimeManager(
        regime_detector=regime_detector,
        rule_objects=top_rule_objects,
        data_handler=data_handler
    )
    
    # Optimize strategies for different regimes
    print("\nOptimizing strategies for different market regimes...")
    regime_manager.optimize_regime_strategies()
    
    # Backtest the regime-based strategy
    regime_backtester = Backtester(data_handler, regime_manager)
    regime_results = regime_backtester.run(use_test_data=True)
    
    print(f"Total Return: {regime_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {regime_results['num_trades']}")
    print(f"Sharpe Ratio: {regime_backtester.calculate_sharpe():.4f}")
    
    # --- Approach 4: Combined Volatility & Trend Regimes ---
    print("\n=== Approach 4: Combined Volatility & Trend Regimes ===")
    
    # This approach would combine two regime detectors for more granular regime classification
    # For example, we could categorize the market into:
    # - Trending Up + Low Volatility
    # - Trending Up + High Volatility
    # - Trending Down + Low Volatility
    # - Trending Down + High Volatility
    # - Range-Bound + Low Volatility
    # - Range-Bound + High Volatility
    
    # This implementation is beyond the scope of this example, but follows
    # the same principles as Approach 3
    
    # --- Performance Comparison ---
    print("\n=== Performance Comparison ===")
    comparison = {
        "Baseline (Simple TopN)": baseline_results,
        "Genetically Optimized": weighted_results,
        "Regime-Based": regime_results
    }
    
    # Summary table
    print(f"{'Strategy':<25} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 55)
    for name, results in comparison.items():
        sharpe = baseline_backtester.calculate_sharpe() if name == "Baseline (Simple TopN)" else \
                 weighted_backtester.calculate_sharpe() if name == "Genetically Optimized" else \
                 regime_backtester.calculate_sharpe()
        
        print(f"{name:<25} {results['total_percent_return']:>8.2f}% {results['num_trades']:>10} {sharpe:>9.4f}")
    
    # Plot equity curves for comparison
    plot_regime_performance(comparison, "Strategy Performance Comparison")
