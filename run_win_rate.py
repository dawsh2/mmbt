# Example of how to use win rate optimization with the OptimizerManager

import os
import numpy as np
import pandas as pd
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from strategy import (Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
                     Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15)
from regime_detection import TrendStrengthRegimeDetector
from optimizer_manager import (OptimizerManager, OptimizationMethod, OptimizationSequence)

if __name__ == "__main__":
    # Load data
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    # Train rules to get our rule objects
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
    
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    # Create regime detector
    trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
    
    # --- Win Rate Optimization Approach ---
    print("\n=== Win Rate Optimization Approach ===")
    optimizer_win_rate = OptimizerManager(
        data_handler=data_handler,
        rule_objects=top_rule_objects
    )
    
    # Configure optimization parameters
    optimization_params = {
        'genetic': {
            'population_size': 20,
            'num_generations': 30,
            'mutation_rate': 0.1
        }
    }
    
    # Run optimization using win_rate as the metric
    win_rate_results = optimizer_win_rate.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.REGIMES_FIRST,  # This will optimize each regime separately
        metrics='win_rate',  # Use win_rate as the optimization metric
        regime_detector=trend_detector,
        optimization_params=optimization_params,
        verbose=True
    )
    
    # Get the optimized strategy
    win_rate_strategy = optimizer_win_rate.get_optimized_strategy()
    
    # Print detailed results
    print("\nWin Rate Optimization Results:")
    print(f"Total Return: {win_rate_results['total_return']:.2f}%")
    print(f"Number of Trades: {win_rate_results['num_trades']}")
    print(f"Sharpe Ratio: {win_rate_results['sharpe']:.4f}")
    print(f"Win Rate: {win_rate_results['win_rate']:.2%}")
    
    # --- Compare with Sharpe Ratio Optimization ---
    print("\n=== Sharpe Ratio Optimization (for comparison) ===")
    optimizer_sharpe = OptimizerManager(
        data_handler=data_handler,
        rule_objects=top_rule_objects
    )
    
    # Run optimization using Sharpe ratio as the metric
    sharpe_results = optimizer_sharpe.optimize(
        method=OptimizationMethod.GENETIC,
        sequence=OptimizationSequence.REGIMES_FIRST,
        metrics='sharpe',  # Use sharpe as the optimization metric
        regime_detector=trend_detector,
        optimization_params=optimization_params,
        verbose=True
    )
    
    # Print detailed results
    print("\nSharpe Ratio Optimization Results:")
    print(f"Total Return: {sharpe_results['total_return']:.2f}%")
    print(f"Number of Trades: {sharpe_results['num_trades']}")
    print(f"Sharpe Ratio: {sharpe_results['sharpe']:.4f}")
    print(f"Win Rate: {sharpe_results['win_rate']:.2%}")
    
    # --- Print comparison ---
    print("\n=== Optimization Approach Comparison ===")
    print(f"{'Approach':<20} {'Return':<10} {'# Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 60)
    print(f"{'Win Rate Optimization':<20} {win_rate_results['total_return']:>8.2f}% {win_rate_results['num_trades']:>10} {win_rate_results['sharpe']:>9.4f} {win_rate_results['win_rate']:>9.2%}")
    print(f"{'Sharpe Optimization':<20} {sharpe_results['total_return']:>8.2f}% {sharpe_results['num_trades']:>10} {sharpe_results['sharpe']:>9.4f} {sharpe_results['win_rate']:>9.2%}")
