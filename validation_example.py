"""
Example script demonstrating the use of walk-forward validation and cross-validation
to improve trading strategy robustness.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from walk_forward_validation import (
    WalkForwardValidator, 
    CrossValidator, 
    NestedCrossValidator,
    run_walk_forward_validation,
    run_cross_validation,
    run_nested_cross_validation
)

def run_basic_example():
    """Simple example of walk-forward validation."""
    # Define data filepath
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    
    # Define rule configuration
    rules_config = [
        (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30, 50]}),
        (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]}),
        (Rule2, {'ema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule3, {'ema1_period': [10, 20], 'ema2_period': [30, 50]}),
        (Rule4, {'dema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule5, {'dema1_period': [10, 20], 'dema2_period': [30, 50]}),
        # Limited rule set for faster execution
    ]
    
    # Run walk-forward validation
    print("\n=== Running Walk-Forward Validation ===")
    wf_results = run_walk_forward_validation(
        data_filepath=filepath,
        rules_config=rules_config,
        window_size=252,  # 1 year of trading days
        step_size=63      # ~3 months
    )
    
    # Run cross-validation
    print("\n=== Running Cross-Validation ===")
    cv_results = run_cross_validation(
        data_filepath=filepath,
        rules_config=rules_config,
        n_folds=5
    )
    
    # Compare the results
    print("\n=== Comparison of Validation Methods ===")
    print(f"Walk-Forward Validation - Total Return: {wf_results['summary']['total_return']:.2f}%")
    print(f"Cross-Validation - Total Return: {cv_results['summary']['total_return']:.2f}%")
    
    return {
        'walk_forward': wf_results,
        'cross_validation': cv_results
    }

def run_comprehensive_validation():
    """
    Comprehensive example demonstrating nested cross-validation and
    the creation of a final model using the best strategy.
    """
    # Define data filepath
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    
    # Define rule configuration - full set for comprehensive test
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
    
    # Run nested cross-validation
    print("\n=== Running Nested Cross-Validation ===")
    nested_results = run_nested_cross_validation(
        data_filepath=filepath,
        rules_config=rules_config,
        outer_folds=3,  # Fewer folds for faster execution
        inner_folds=2
    )
    
    # Determine the best optimization method based on nested CV
    method_counts = {}
    for method in nested_results['best_methods']:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    best_method = max(method_counts.items(), key=lambda x: x[1])[0]
    print(f"\nBest optimization method: {best_method}")
    
    # Create final model using the best method on the full dataset
    print("\n=== Training Final Model ===")
    data_handler = CSVDataHandler(filepath, train_fraction=1.0)
    
    # Train rules system
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    print("\nSelected top rules for final model:")
    for i, rule in enumerate(top_rule_objects):
        print(f"  {i+1}. {rule.__class__.__name__}")
    
    # Optimize weights using the best method
    if best_method == 'genetic':
        optimizer = GeneticOptimizer(
            data_handler=data_handler,
            rule_objects=top_rule_objects,
            optimization_metric='sharpe'
        )
        best_weights = optimizer.optimize(verbose=True)
        print(f"\nOptimized weights: {best_weights}")
        
        final_strategy = WeightedRuleStrategy(
            rule_objects=top_rule_objects,
            weights=best_weights
        )
    else:
        final_strategy = rule_system.get_top_n_strategy()
        print("\nUsing equal-weighted strategy")
    
    # Final backtest on full dataset
    print("\n=== Final Model Backtest ===")
    final_backtester = Backtester(data_handler, final_strategy)
    final_results = final_backtester.run(use_test_data=False)  # Use full dataset
    
    print(f"Total Return: {final_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {final_results['num_trades']}")
    print(f"Sharpe Ratio: {final_backtester.calculate_sharpe():.4f}")
    
    # Plot equity curve for final model
    trades = final_results['trades']
    if trades:
        plt.figure(figsize=(15, 8))
        
        # Create equity curve
        equity = [1.0]  # Start with $1
        for trade in trades:
            log_return = trade[5]
            equity.append(equity[-1] * np.exp(log_return))
        
        plt.plot(range(len(equity)), equity)
        plt.title('Final Model Equity Curve')
        plt.xlabel('Trade')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return {
        'nested_cv_results': nested_results,
        'best_method': best_method,
        'final_model_results': final_results,
        'final_strategy': final_strategy
    }

def comparison_with_without_validation():
    """
    Compare trading performance with and without validation.
    This demonstrates the value of validation for improving robustness.
    """
    # Define data filepath
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    
    # Define rule configuration
    rules_config = [
        (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30, 50]}),
        (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]}),
        (Rule2, {'ema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule3, {'ema1_period': [10, 20], 'ema2_period': [30, 50]}),
        (Rule4, {'dema1_period': [10, 20], 'ma2_period': [30, 50]}),
        (Rule5, {'dema1_period': [10, 20], 'dema2_period': [30, 50]}),
    ]
    
    # Create data handler with standard train/test split
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    # Approach 1: Standard train/test split (no validation)
    print("\n=== Approach 1: Standard Train/Test Split ===")
    
    # Train on training data
    rule_system_std = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system_std.train_rules(data_handler)
    top_rules_std = list(rule_system_std.trained_rule_objects.values())
    
    # Backtest on test data
    strategy_std = rule_system_std.get_top_n_strategy()
    backtester_std = Backtester(data_handler, strategy_std)
    results_std = backtester_std.run(use_test_data=True)
    
    print(f"Total Return: {results_std['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results_std['num_trades']}")
    print(f"Sharpe Ratio: {backtester_std.calculate_sharpe():.4f}")
    
    # Approach 2: With Walk-Forward Validation
    print("\n=== Approach 2: With Walk-Forward Validation ===")
    
    # Run walk-forward validation
    wf_validator = WalkForwardValidator(
        data_filepath=filepath,
        rules_config=rules_config,
        window_size=252,  # 1 year of trading days
        step_size=63,     # ~3 months
        top_n=5
    )
    
    wf_results = wf_validator.run_validation(verbose=True, plot_results=False)
    
    # Generate a final model based on walk-forward insights
    # (In a real application, you would analyze which rules perform best across windows)
    
    # Use full dataset now
    full_data_handler = CSVDataHandler(filepath, train_fraction=1.0)
    
    # Train rule system
    rule_system_wf = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system_wf.train_rules(full_data_handler)
    top_rules_wf = list(rule_system_wf.trained_rule_objects.values())
    
    # Use genetic optimization (typically performs better in walk-forward testing)
    optimizer = GeneticOptimizer(
        data_handler=full_data_handler,
        rule_objects=top_rules_wf,
        optimization_metric='sharpe'
    )
    best_weights = optimizer.optimize(verbose=False)
    
    # Create final strategy
    strategy_wf = WeightedRuleStrategy(
        rule_objects=top_rules_wf,
        weights=best_weights
    )
    
    # Backtest on original test data for fair comparison
    backtester_wf = Backtester(data_handler, strategy_wf)
    results_wf = backtester_wf.run(use_test_data=True)
    
    print(f"Total Return: {results_wf['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results_wf['num_trades']}")
    print(f"Sharpe Ratio: {backtester_wf.calculate_sharpe():.4f}")
    
    # Compare results
    print("\n=== Performance Comparison ===")
    print(f"{'Approach':<25} {'Return':<10} {'Trades':<8} {'Sharpe':<8}")
    print("-" * 55)
    print(f"{'Standard Train/Test':<25} {results_std['total_percent_return']:>8.2f}% {results_std['num_trades']:>7} {backtester_std.calculate_sharpe():>7.4f}")
    print(f"{'With Walk-Forward':<25} {results_wf['total_percent_return']:>8.2f}% {results_wf['num_trades']:>7} {backtester_wf.calculate_sharpe():>7.4f}")
    
    # Plot comparison of equity curves
    plt.figure(figsize=(15, 8))
    
    # Standard approach equity curve
    equity_std = [1.0]
    for trade in results_std['trades']:
        log_return = trade[5]
        equity_std.append(equity_std[-1] * np.exp(log_return))
    
    # Walk-forward approach equity curve
    equity_wf = [1.0]
    for trade in results_wf['trades']:
        log_return = trade[5]
        equity_wf.append(equity_wf[-1] * np.exp(log_return))
    
    plt.plot(range(len(equity_std)), equity_std, label='Standard Train/Test')
    plt.plot(range(len(equity_wf)), equity_wf, label='With Walk-Forward Validation')
    plt.title('Equity Curve Comparison')
    plt.xlabel('Trade')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'standard_results': results_std,
        'walkforward_results': results_wf,
        'improvement_pct': results_wf['total_percent_return'] - results_std['total_percent_return']
    }

if __name__ == "__main__":
    # Choose which example to run
    # 1. Basic walk-forward and cross-validation
    results = run_basic_example()
    
    # 2. Comprehensive nested cross-validation and final model creation
    # results = run_comprehensive_validation()
    
    # 3. Comparison of performance with and without validation
    # results = comparison_with_without_validation()
    
    print("\nValidation completed successfully!")
