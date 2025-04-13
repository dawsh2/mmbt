"""
Run genetic optimization with threshold parameter tuning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7,
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import WeightedRuleStrategy
from genetic_optimizer import EnhancedGeneticOptimizer


def calculate_annualized_return(total_return, days, trading_days_per_year=252):
    """
    Calculate annualized return from total return and number of days.
    
    Args:
        total_return: Total return in percentage
        days: Number of days in the period
        trading_days_per_year: Number of trading days in a year
        
    Returns:
        float: Annualized return percentage
    """
    # Convert total_return from percentage to decimal
    decimal_return = total_return / 100
    
    # Calculate years
    years = days / trading_days_per_year
    
    # Calculate annualized return (avoiding negative numbers in power calculation)
    if decimal_return >= 0:
        annualized_return = ((1 + decimal_return) ** (1 / years) - 1) * 100
    else:
        annualized_return = -((1 / (1 + decimal_return)) ** (1 / years) - 1) * 100
    
    return annualized_return


def plot_equity_curve(trades, title, initial_capital=10000, save_path=None):
    """Plot equity curve from trade data with more details."""
    if isinstance(trades, pd.DataFrame) and len(trades) > 0:
        # If trades is a DataFrame, extract log returns
        if 'log_return' in trades.columns:
            log_returns = trades['log_return'].values
        elif 'profit' in trades.columns:
            # Convert profit to log return if needed
            log_returns = np.log(1 + trades['profit'] / 100)
        else:
            print(f"No return data found in trades for {title}")
            return
    elif isinstance(trades, list) and len(trades) > 0 and isinstance(trades[0], (list, tuple)):
        # If trades is a list of tuples/lists
        log_returns = [trade[5] for trade in trades]
    else:
        print(f"No valid trades found for {title}")
        return
        
    # Create equity curve
    equity = [initial_capital]
    for log_return in log_returns:
        equity.append(equity[-1] * np.exp(log_return))
    
    # Calculate additional metrics
    total_return = (equity[-1] / equity[0] - 1) * 100
    max_equity = max(equity)
    min_equity_after_peak = min(equity[equity.index(max_equity):]) if max_equity in equity else equity[-1]
    max_drawdown = ((max_equity - min_equity_after_peak) / max_equity) * 100
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(equity)
    ax1.set_title(f"{title}\nTotal Return: {total_return:.2f}%, Max Drawdown: {max_drawdown:.2f}%")
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True)
    
    # Calculate drawdown series
    drawdown = []
    peak = equity[0]
    for e in equity:
        peak = max(peak, e)
        dd = ((peak - e) / peak) * 100
        drawdown.append(dd)
    
    # Plot drawdown
    ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax2.plot(drawdown, color='red')
    ax2.set_title('Drawdown (%)')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return total_return, max_drawdown


def run_optimizer_grid():
    """
    Run grid search on threshold parameters to analyze their impact.
    This function runs multiple optimization runs with different threshold values.
    """
    # Define file path
    filepath = "data/data.csv"  # Update this path to your data file location
    
    print(f"Looking for data file at: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir())
        exit(1)
        
    # Load data
    data_handler = CSVDataHandler(filepath, train_fraction=0.8)
    
    # Train rules and select top performers
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
        (Rule15, {'bb_period': [20, 25]})
    ]
    
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=12)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    print("\nSelected Top Rules:")
    for i, rule in enumerate(top_rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # 1. Run baseline strategy (equal weights, default thresholds)
    print("\n=== Running Baseline Strategy ===")
    baseline_strategy = rule_system.get_top_n_strategy()
    baseline_backtester = Backtester(data_handler, baseline_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)
    baseline_sharpe = baseline_backtester.calculate_sharpe()
    
    print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {baseline_results['num_trades']}")
    print(f"Sharpe Ratio: {baseline_sharpe:.4f}")
    
    # 2. Run optimizer with weight optimization only
    print("\n=== Running Weight Optimization Only ===")
    weight_optimizer = EnhancedGeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=20,
        num_generations=30,
        optimization_metric='sharpe',
        random_seed=42,
        regularization_factor=0.2,
        balance_factor=0.3,
        optimize_thresholds=False
    )
    
    optimal_weights = weight_optimizer.optimize(verbose=True)
    
    # Create strategy with optimized weights but default thresholds
    weight_only_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights,
        buy_threshold=0.5,  # Default
        sell_threshold=-0.5  # Default
    )
    
    # Test weight-only optimized strategy
    weight_backtester = Backtester(data_handler, weight_only_strategy)
    weight_results = weight_backtester.run(use_test_data=True)
    weight_sharpe = weight_backtester.calculate_sharpe()
    
    print(f"Weight-Only Optimization:")
    print(f"Total Return: {weight_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {weight_results['num_trades']}")
    print(f"Sharpe Ratio: {weight_sharpe:.4f}")
    
    # 3. Run optimizer with both weight and threshold optimization
    print("\n=== Running Weight + Threshold Optimization ===")
    full_optimizer = EnhancedGeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=20,
        num_generations=30,
        optimization_metric='sharpe',
        random_seed=42,
        regularization_factor=0.2,
        balance_factor=0.3,
        optimize_thresholds=True
    )
    
    optimal_params = full_optimizer.optimize(verbose=True)
    
    # Extract parameters
    optimal_weights = optimal_params['weights']
    buy_threshold = optimal_params['buy_threshold']
    sell_threshold = optimal_params['sell_threshold']
    
    # Create strategy with optimized weights and thresholds
    full_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )
    
    # Test fully optimized strategy
    full_backtester = Backtester(data_handler, full_strategy)
    full_results = full_backtester.run(use_test_data=True)
    full_sharpe = full_backtester.calculate_sharpe()
    
    print(f"Full Optimization:")
    print(f"Total Return: {full_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {full_results['num_trades']}")
    print(f"Sharpe Ratio: {full_sharpe:.4f}")
    
    # 4. Run a grid of threshold values to see their impact
    print("\n=== Running Threshold Grid Analysis ===")
    
    # Define threshold grid
    buy_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    sell_thresholds = [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    
    # Create grid results storage
    grid_results = []
    
    # Use the optimized weights from weight-only optimization
    # This isolates the effect of threshold changes
    for buy_thresh in buy_thresholds:
        for sell_thresh in sell_thresholds:
            print(f"Testing thresholds: Buy={buy_thresh}, Sell={sell_thresh}")
            
            # Skip if buy threshold <= abs(sell threshold)
            if buy_thresh <= abs(sell_thresh):
                continue
                
            # Create strategy with these thresholds
            test_strategy = WeightedRuleStrategy(
                rule_objects=top_rule_objects,
                weights=optimal_weights,  # From weight-only optimization
                buy_threshold=buy_thresh,
                sell_threshold=sell_thresh
            )
            
            # Test this strategy
            test_backtester = Backtester(data_handler, test_strategy)
            test_results = test_backtester.run(use_test_data=True)
            test_sharpe = test_backtester.calculate_sharpe()
            
            # Record results
            grid_results.append({
                'buy_threshold': buy_thresh,
                'sell_threshold': sell_thresh,
                'total_return': test_results['total_percent_return'],
                'num_trades': test_results['num_trades'],
                'sharpe': test_sharpe
            })
    
    # Plot heatmap of returns across threshold grid
    print("\n=== Threshold Grid Analysis Results ===")
    
    # Convert grid results to dataframe
    grid_df = pd.DataFrame(grid_results)
    
    # Create heatmap of total returns
    plt.figure(figsize=(10, 8))
    
    # Convert to pivot table for heatmap
    pivot_returns = grid_df.pivot_table(
        index='buy_threshold', 
        columns='sell_threshold', 
        values='total_return',
        aggfunc='mean'
    )
    
    # Plot heatmap with values displayed
    plt.imshow(pivot_returns, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Total Return (%)')
    
    # Add text annotations with returns
    for i in range(len(pivot_returns.index)):
        for j in range(len(pivot_returns.columns)):
            if not pd.isna(pivot_returns.iloc[i, j]):
                plt.text(j, i, f"{pivot_returns.iloc[i, j]:.1f}%", 
                       ha="center", va="center", color="black")
    
    # Set tick labels
    plt.xticks(range(len(pivot_returns.columns)), pivot_returns.columns)
    plt.yticks(range(len(pivot_returns.index)), pivot_returns.index)
    
    plt.xlabel('Sell Threshold')
    plt.ylabel('Buy Threshold')
    plt.title('Total Return (%) by Threshold Parameters')
    plt.tight_layout()
    plt.savefig("threshold_returns_heatmap.png")
    
    # Also create heatmap of Sharpe ratios
    plt.figure(figsize=(10, 8))
    
    # Convert to pivot table for heatmap
    pivot_sharpe = grid_df.pivot_table(
        index='buy_threshold', 
        columns='sell_threshold', 
        values='sharpe',
        aggfunc='mean'
    )
    
    # Plot heatmap with values displayed
    plt.imshow(pivot_sharpe, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Sharpe Ratio')
    
    # Add text annotations with Sharpe values
    for i in range(len(pivot_sharpe.index)):
        for j in range(len(pivot_sharpe.columns)):
            if not pd.isna(pivot_sharpe.iloc[i, j]):
                plt.text(j, i, f"{pivot_sharpe.iloc[i, j]:.2f}", 
                       ha="center", va="center", color="black")
    
    # Set tick labels
    plt.xticks(range(len(pivot_sharpe.columns)), pivot_sharpe.columns)
    plt.yticks(range(len(pivot_sharpe.index)), pivot_sharpe.index)
    
    plt.xlabel('Sell Threshold')
    plt.ylabel('Buy Threshold')
    plt.title('Sharpe Ratio by Threshold Parameters')
    plt.tight_layout()
    plt.savefig("threshold_sharpe_heatmap.png")
    
    # Plot trade counts across threshold grid
    plt.figure(figsize=(10, 8))
    
    # Convert to pivot table for heatmap
    pivot_trades = grid_df.pivot_table(
        index='buy_threshold', 
        columns='sell_threshold', 
        values='num_trades',
        aggfunc='mean'
    )
    
    # Plot heatmap with values displayed
    plt.imshow(pivot_trades, cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Number of Trades')
    
    # Add text annotations with trade counts
    for i in range(len(pivot_trades.index)):
        for j in range(len(pivot_trades.columns)):
            if not pd.isna(pivot_trades.iloc[i, j]):
                plt.text(j, i, f"{int(pivot_trades.iloc[i, j])}", 
                       ha="center", va="center", color="black")
    
    # Set tick labels
    plt.xticks(range(len(pivot_trades.columns)), pivot_trades.columns)
    plt.yticks(range(len(pivot_trades.index)), pivot_trades.index)
    
    plt.xlabel('Sell Threshold')
    plt.ylabel('Buy Threshold')
    plt.title('Number of Trades by Threshold Parameters')
    plt.tight_layout()
    plt.savefig("threshold_trades_heatmap.png")
    
    # 5. Compare all approaches - baseline, weight-only, full optimization, and best from grid search
    
    # Find the best result from grid search
    best_grid_idx = grid_df['sharpe'].idxmax()
    best_grid = grid_df.iloc[best_grid_idx]
    
    # Create strategy with best grid parameters
    best_grid_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights,  # From weight-only optimization
        buy_threshold=best_grid['buy_threshold'],
        sell_threshold=best_grid['sell_threshold']
    )
    
    # Test best grid strategy
    best_grid_backtester = Backtester(data_handler, best_grid_strategy)
    best_grid_results = best_grid_backtester.run(use_test_data=True)
    best_grid_sharpe = best_grid_backtester.calculate_sharpe()
    
    # Print summary comparison
    print("\n=== Strategy Comparison ===")
    print(f"{'Strategy':<25} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 55)
    print(f"{'Baseline (Equal Weights)':<25} {baseline_results['total_percent_return']:>8.2f}% {baseline_results['num_trades']:>10} {baseline_sharpe:>9.4f}")
    print(f"{'Weight-Only Optimized':<25} {weight_results['total_percent_return']:>8.2f}% {weight_results['num_trades']:>10} {weight_sharpe:>9.4f}")
    print(f"{'Full Optimization':<25} {full_results['total_percent_return']:>8.2f}% {full_results['num_trades']:>10} {full_sharpe:>9.4f}")
    print(f"{'Best Grid Search':<25} {best_grid_results['total_percent_return']:>8.2f}% {best_grid_results['num_trades']:>10} {best_grid_sharpe:>9.4f}")
    
    # Save optimization details for reproducibility
    with open("threshold_optimization_details.txt", "w") as f:
        f.write(f"Optimization Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameter Summary:\n")
        f.write(f"\nBaseline (Equal Weights):\n")
        f.write(f"  Buy Threshold: 0.5, Sell Threshold: -0.5\n")
        f.write(f"  Total Return: {baseline_results['total_percent_return']:.2f}%\n")
        f.write(f"  Number of Trades: {baseline_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {baseline_sharpe:.4f}\n")
        
        f.write(f"\nWeight-Only Optimization:\n")
        f.write(f"  Buy Threshold: 0.5, Sell Threshold: -0.5\n")
        f.write(f"  Total Return: {weight_results['total_percent_return']:.2f}%\n")
        f.write(f"  Number of Trades: {weight_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {weight_sharpe:.4f}\n")
        
        f.write(f"\nFull Optimization (Weights + Thresholds):\n")
        f.write(f"  Buy Threshold: {buy_threshold:.4f}, Sell Threshold: {sell_threshold:.4f}\n")
        f.write(f"  Total Return: {full_results['total_percent_return']:.2f}%\n")
        f.write(f"  Number of Trades: {full_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {full_sharpe:.4f}\n")
        
        f.write(f"\nBest Grid Search:\n")
        f.write(f"  Buy Threshold: {best_grid['buy_threshold']}, Sell Threshold: {best_grid['sell_threshold']}\n")
        f.write(f"  Total Return: {best_grid_results['total_percent_return']:.2f}%\n")
        f.write(f"  Number of Trades: {best_grid_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {best_grid_sharpe:.4f}\n")
        
        f.write("\nDetailed Grid Results:\n")
        for i, result in grid_df.iterrows():
            f.write(f"  Buy: {result['buy_threshold']}, Sell: {result['sell_threshold']}, "
                   f"Return: {result['total_return']:.2f}%, Trades: {result['num_trades']}, "
                   f"Sharpe: {result['sharpe']:.4f}\n")
    
    print("\nOptimization complete! Results and charts saved.")
    print("Reproducibility information saved to threshold_optimization_details.txt")
    
    # Plot equity curves for comparison
    plt.figure(figsize=(14, 7))
    
    # Create equity curves for all strategies
    baseline_equity = [10000]
    for trade in baseline_results['trades']:
        baseline_equity.append(baseline_equity[-1] * np.exp(trade[5]))
    
    weight_equity = [10000]
    for trade in weight_results['trades']:
        weight_equity.append(weight_equity[-1] * np.exp(trade[5]))
    
    full_equity = [10000]
    for trade in full_results['trades']:
        full_equity.append(full_equity[-1] * np.exp(trade[5]))
    
    best_grid_equity = [10000]
    for trade in best_grid_results['trades']:
        best_grid_equity.append(best_grid_equity[-1] * np.exp(trade[5]))
    
    # Plot all equity curves
    plt.plot(baseline_equity, label=f"Baseline ({baseline_results['total_percent_return']:.2f}%)")
    plt.plot(weight_equity, label=f"Weight-Only ({weight_results['total_percent_return']:.2f}%)")
    plt.plot(full_equity, label=f"Full Optimization ({full_results['total_percent_return']:.2f}%)")
    plt.plot(best_grid_equity, label=f"Best Grid ({best_grid_results['total_percent_return']:.2f}%)")
    
    plt.title('Strategy Comparison')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("threshold_strategy_comparison.png")
    plt.close()


if __name__ == "__main__":
    run_optimizer_grid()
