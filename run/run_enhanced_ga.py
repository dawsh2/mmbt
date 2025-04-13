"""
Run improved genetic algorithm optimization with regularization and cross-validation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime, timedelta
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
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
    
    # Calculate annualized return
    annualized_return = ((1 + decimal_return) ** (1 / years) - 1) * 100
    
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
    min_equity = min(equity[1:]) if len(equity) > 1 else equity[0]
    max_drawdown = ((max_equity - min_equity) / max_equity) * 100
    
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
        plt.close()
    else:
        plt.show()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run improved genetic algorithm optimization')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic mode (default seed 42)')
    parser.add_argument('--population', type=int, default=20, help='Population size for GA')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations for GA')
    parser.add_argument('--metric', type=str, default='sharpe', choices=['return', 'sharpe', 'win_rate', 'risk_adjusted'],
                        help='Optimization metric to use')
    parser.add_argument('--cv_folds', type=int, default=3, help='Number of cross-validation folds (1 for no CV)')
    parser.add_argument('--reg_factor', type=float, default=0.2, help='Regularization factor (0-1)')
    parser.add_argument('--balance_factor', type=float, default=0.3, help='Balance factor toward equal weights (0-1)')
    return parser.parse_args()


def run_genetic_optimization():
    """Main function to run improved genetic algorithm optimization."""
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

    # Calculate trading days in the test period
    data_handler.reset_test()
    first_bar = data_handler.get_next_test_bar()
    data_handler.reset_test()
    last_bar = None
    count = 0
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is not None:
            last_bar = bar
            count += 1
        else:
            break
    
    # Calculate date range
    first_date = datetime.strptime(first_bar['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(first_bar['timestamp'], str) else first_bar['timestamp']
    last_date = datetime.strptime(last_bar['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(last_bar['timestamp'], str) else last_bar['timestamp']
    days_diff = (last_date - first_date).days
    
    # Calculate annualized returns
    baseline_annual_return = calculate_annualized_return(
        baseline_results['total_percent_return'], 
        days_diff
    )

    print(f"Test Period: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} ({days_diff} days)")
    print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
    print(f"Annualized Return: {baseline_annual_return:.2f}%")
    print(f"Number of Trades: {baseline_results['num_trades']}")
    sharpe_baseline = baseline_backtester.calculate_sharpe()
    print(f"Sharpe Ratio: {sharpe_baseline:.4f}")

    plot_equity_curve(baseline_results["trades"], "Baseline Strategy Equity Curve (OOS)", save_path="Baseline_Strategy_Equity_Curve.png")

    # 4. Run enhanced genetic optimization
    print("\n=== Running Enhanced Genetic Optimization ===")
    ga_start = time.time()

    # Additional info about settings
    print(f"Optimizing using {args.metric.upper()} metric with {args.cv_folds} CV folds")
    print(f"Regularization Factor: {args.reg_factor}, Balance Factor: {args.balance_factor}")
    print(f"Population size: {args.population}, Generations: {args.generations}")

    genetic_optimizer = EnhancedGeneticOptimizer(
        data_handler=data_handler,
        rule_objects=top_rule_objects,
        population_size=args.population,
        num_generations=args.generations,
        optimization_metric=args.metric,
        random_seed=args.seed,
        deterministic=args.deterministic,
        cv_folds=args.cv_folds,
        regularization_factor=args.reg_factor,
        balance_factor=args.balance_factor
    )

    # Run optimization
    optimal_weights = genetic_optimizer.optimize(
        verbose=True,
        early_stopping_generations=5,  # Early stopping after 5 generations without improvement
        min_improvement=0.0005  # Consider improvements below 0.0005 as insignificant
    )
    print(f"Genetic optimization completed in {time.time() - ga_start:.2f} seconds")
    print(f"Optimal weights: {optimal_weights}")

    # Calculate effective number of rules (measure of weight distribution)
    effective_rules = 1.0 / np.sum(optimal_weights**2)
    print(f"Effective number of rules: {effective_rules:.1f} out of {len(top_rule_objects)}")

    # Create weighted strategy with optimal weights
    weighted_strategy = WeightedRuleStrategy(
        rule_objects=top_rule_objects,
        weights=optimal_weights
    )

    # Backtest the optimized strategy
    print("\n=== Backtesting Genetically Optimized Strategy ===")
    weighted_backtester = Backtester(data_handler, weighted_strategy)
    ga_results = weighted_backtester.run(use_test_data=True)

    # Calculate annualized return for GA strategy
    ga_annual_return = calculate_annualized_return(
        ga_results['total_percent_return'], 
        days_diff
    )

    print(f"Total Return: {ga_results['total_percent_return']:.2f}%")
    print(f"Annualized Return: {ga_annual_return:.2f}%")
    print(f"Number of Trades: {ga_results['num_trades']}")
    sharpe_ga = weighted_backtester.calculate_sharpe()
    print(f"Sharpe Ratio: {sharpe_ga:.4f}")

    plot_equity_curve(ga_results["trades"], "Genetically Optimized Strategy Equity Curve", save_path="GA_Strategy_Equity_Curve.png")

    # Create a blended strategy (50% baseline, 50% GA)
    print("\n=== Running Blended Strategy (50% baseline, 50% GA) ===")
    
    # Get list of all rules
    all_rules = []
    all_weights = []
    
    # Add baseline weights (equal weights)
    equal_weights = np.ones(len(top_rule_objects)) / len(top_rule_objects)
    for i, rule in enumerate(top_rule_objects):
        all_rules.append(rule)
        all_weights.append(equal_weights[i] * 0.5)  # 50% weight to baseline
        
    # Add GA weights
    for i, rule in enumerate(top_rule_objects):
        all_rules.append(rule)
        all_weights.append(optimal_weights[i] * 0.5)  # 50% weight to GA
    
    # Normalize weights
    all_weights = np.array(all_weights) / sum(all_weights)
    
    # Create blended strategy
    blended_strategy = WeightedRuleStrategy(
        rule_objects=all_rules,
        weights=all_weights
    )
    
    # Backtest the blended strategy
    blended_backtester = Backtester(data_handler, blended_strategy)
    blended_results = blended_backtester.run(use_test_data=True)
    
    # Calculate annualized return for blended strategy
    blended_annual_return = calculate_annualized_return(
        blended_results['total_percent_return'], 
        days_diff
    )
    
    print(f"Total Return: {blended_results['total_percent_return']:.2f}%")
    print(f"Annualized Return: {blended_annual_return:.2f}%")
    print(f"Number of Trades: {blended_results['num_trades']}")
    sharpe_blended = blended_backtester.calculate_sharpe()
    print(f"Sharpe Ratio: {sharpe_blended:.4f}")
    
    plot_equity_curve(blended_results["trades"], "Blended Strategy Equity Curve", save_path="Blended_Strategy_Equity_Curve.png")

    # Compare results
    improvement_percent = (ga_results['total_percent_return'] / baseline_results['total_percent_return'] - 1) * 100 if baseline_results['total_percent_return'] > 0 else 0
    blended_improvement_percent = (blended_results['total_percent_return'] / baseline_results['total_percent_return'] - 1) * 100 if baseline_results['total_percent_return'] > 0 else 0
    
    print("\n=== Performance Comparison ===")
    print(f"{'Strategy':<25} {'Return':<10} {'Annual':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 65)
    print(f"{'Baseline (Equal Weights)':<25} {baseline_results['total_percent_return']:>8.2f}% {baseline_annual_return:>8.2f}% {baseline_results['num_trades']:>10} {sharpe_baseline:>9.4f}")
    print(f"{'Genetically Optimized':<25} {ga_results['total_percent_return']:>8.2f}% {ga_annual_return:>8.2f}% {ga_results['num_trades']:>10} {sharpe_ga:>9.4f}")
    print(f"{'Blended (50/50)':<25} {blended_results['total_percent_return']:>8.2f}% {blended_annual_return:>8.2f}% {blended_results['num_trades']:>10} {sharpe_blended:>9.4f}")
    
    if baseline_results['total_percent_return'] > 0:
        print(f"\nGA Return vs Baseline: {improvement_percent:+.2f}%")
        print(f"Blended Return vs Baseline: {blended_improvement_percent:+.2f}%")

    # Save genetic optimization fitness history
    plt.figure(figsize=(12, 6))
    plt.plot(genetic_optimizer.fitness_history)
    plt.title('Genetic Optimization Fitness Progress')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("GA_Fitness_History.png")
    plt.close()

    # Show the full fitness history with additional metrics
    genetic_optimizer.plot_fitness_history()

    # Plot comparison of all three strategies together
    plt.figure(figsize=(14, 7))
    
    # Create equity curves for all strategies
    baseline_equity = [10000]
    for trade in baseline_results['trades']:
        baseline_equity.append(baseline_equity[-1] * np.exp(trade[5]))
    
    ga_equity = [10000]
    for trade in ga_results['trades']:
        ga_equity.append(ga_equity[-1] * np.exp(trade[5]))
    
    blended_equity = [10000]
    for trade in blended_results['trades']:
        blended_equity.append(blended_equity[-1] * np.exp(trade[5]))
    
    # Plot all equity curves
    plt.plot(baseline_equity, label=f"Baseline ({baseline_results['total_percent_return']:.2f}%)")
    plt.plot(ga_equity, label=f"GA ({ga_results['total_percent_return']:.2f}%)")
    plt.plot(blended_equity, label=f"Blended ({blended_results['total_percent_return']:.2f}%)")
    
    plt.title('Strategy Comparison')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Strategy_Comparison.png")
    plt.close()

    # Save optimization details for reproducibility
    with open("optimization_details.txt", "w") as f:
        f.write(f"Optimization Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Optimization Metric: {args.metric}\n")
        f.write(f"Population Size: {args.population}\n")
        f.write(f"Number of Generations: {args.generations}\n")
        f.write(f"Cross-Validation Folds: {args.cv_folds}\n")
        f.write(f"Regularization Factor: {args.reg_factor}\n")
        f.write(f"Balance Factor: {args.balance_factor}\n")
        f.write(f"Deterministic Mode: {args.deterministic}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Best Fitness: {genetic_optimizer.best_fitness:.6f}\n")
        f.write(f"Optimal Weights: {optimal_weights}\n")
        f.write(f"Effective Number of Rules: {effective_rules:.1f}\n")
        f.write(f"\nBaseline Performance:\n")
        f.write(f"  Total Return: {baseline_results['total_percent_return']:.2f}%\n")
        f.write(f"  Annualized Return: {baseline_annual_return:.2f}%\n")
        f.write(f"  Number of Trades: {baseline_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {sharpe_baseline:.4f}\n")
        f.write(f"\nGA Performance:\n")
        f.write(f"  Total Return: {ga_results['total_percent_return']:.2f}%\n")
        f.write(f"  Annualized Return: {ga_annual_return:.2f}%\n")
        f.write(f"  Number of Trades: {ga_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {sharpe_ga:.4f}\n")
        f.write(f"\nBlended Performance:\n")
        f.write(f"  Total Return: {blended_results['total_percent_return']:.2f}%\n")
        f.write(f"  Annualized Return: {blended_annual_return:.2f}%\n")
        f.write(f"  Number of Trades: {blended_results['num_trades']}\n")
        f.write(f"  Sharpe Ratio: {sharpe_blended:.4f}\n")

    print("\nOptimization complete! Results and charts saved.")
    print(f"Detailed information saved to optimization_details.txt")


# Main program
if __name__ == "__main__":
    run_genetic_optimization()
