"""
Run walk-forward validation with enhanced genetic optimization.
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
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7,
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import WeightedRuleStrategy
from genetic_optimizer import EnhancedGeneticOptimizer
from validator import WalkForwardValidator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run walk-forward validation with enhanced genetic optimization')
    parser.add_argument('--window_size', type=int, default=252, help='Window size in bars')
    parser.add_argument('--step_size', type=int, default=63, help='Step size in bars')
    parser.add_argument('--train_pct', type=float, default=0.7, help='Percentage of window for training')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top rules to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible results')
    parser.add_argument('--metric', type=str, default='sharpe', 
                        choices=['return', 'sharpe', 'win_rate', 'risk_adjusted'],
                        help='Optimization metric to use')
    parser.add_argument('--reg_factor', type=float, default=0.2, help='Regularization factor (0-1)')
    parser.add_argument('--balance_factor', type=float, default=0.3, help='Balance factor toward equal weights (0-1)')
    parser.add_argument('--population', type=int, default=15, help='Population size for GA')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations for GA')
    return parser.parse_args()


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
    
    # Calculate annualized return (avoid negative numbers in power calculation)
    if decimal_return >= 0:
        annualized_return = ((1 + decimal_return) ** (1 / years) - 1) * 100
    else:
        annualized_return = -((1 / (1 + decimal_return)) ** (1 / years) - 1) * 100
    
    return annualized_return


class EnhancedWalkForwardValidator(WalkForwardValidator):
    """
    Enhanced walk-forward validator using the improved genetic optimizer.
    """
    
    def __init__(self, 
                 data_filepath, 
                 rules_config, 
                 window_size=252,
                 step_size=63, 
                 train_pct=0.7,
                 top_n=16,
                 optimization_method='genetic', 
                 optimization_metric='sharpe',
                 random_seed=42,
                 reg_factor=0.2,
                 balance_factor=0.3,
                 population_size=15,
                 num_generations=20):
        """
        Initialize the enhanced walk-forward validator.
        
        Args:
            data_filepath: Path to CSV data file
            rules_config: Configuration of rules and parameters to test
            window_size: Size of each window in trading days
            step_size: Number of days to roll forward between windows
            train_pct: Percentage of window to use for training
            top_n: Number of top rules to select
            optimization_method: Method to use for optimization ('genetic', 'grid', etc.)
            optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
            random_seed: Random seed for reproducibility
            reg_factor: Regularization factor for genetic optimizer
            balance_factor: Balance factor for genetic optimizer
            population_size: Population size for genetic optimizer
            num_generations: Number of generations for genetic optimizer
        """
        super().__init__(
            data_filepath=data_filepath,
            rules_config=rules_config,
            window_size=window_size,
            step_size=step_size,
            train_pct=train_pct,
            top_n=top_n,
            optimization_method=optimization_method,
            optimization_metric=optimization_metric
        )
        
        self.random_seed = random_seed
        self.reg_factor = reg_factor
        self.balance_factor = balance_factor
        self.population_size = population_size
        self.num_generations = num_generations
        
    def run_validation(self, verbose=True, plot_results=True):
        """
        Run walk-forward validation with enhanced genetic optimization.
        
        Args:
            verbose: Whether to print progress information
            plot_results: Whether to plot validation results
            
        Returns:
            dict: Summary of validation results
        """
        # This method will be similar to the parent class but using the enhanced optimizer
        if self.full_df is None:
            self.load_data()
            
        if verbose:
            print(f"Running walk-forward validation with {self.window_size} day windows "
                  f"and {self.step_size} day steps")
            print(f"Total data size: {len(self.full_df)} bars")
        
        # Create windows
        self._create_windows()
        
        if verbose:
            print(f"Created {len(self.windows)} validation windows")
        
        # Run validation for each window
        all_trades = []
        window_results = []
        
        for i, (train_window, test_window) in enumerate(self.windows):
            if verbose:
                train_start = train_window[0]['timestamp'] if len(train_window) > 0 else "N/A"
                train_end = train_window[-1]['timestamp'] if len(train_window) > 0 else "N/A"
                test_start = test_window[0]['timestamp'] if len(test_window) > 0 else "N/A"
                test_end = test_window[-1]['timestamp'] if len(test_window) > 0 else "N/A"
                
                print(f"\nWindow {i+1}/{len(self.windows)}:")
                print(f"  Train: {train_start} to {train_end} ({len(train_window)} bars)")
                print(f"  Test:  {test_start} to {test_end} ({len(test_window)} bars)")
            
            # Skip windows with insufficient data
            if len(train_window) < 30 or len(test_window) < 5:
                if verbose:
                    print("  Insufficient data in window, skipping...")
                continue
            
            # Create data handler for this window
            window_data_handler = self._create_window_data_handler(train_window, test_window)
            
            # Train strategy on window
            if verbose:
                print("  Training strategy on window...")
                
            try:
                # Train rules and get top performers
                rule_system = EventDrivenRuleSystem(rules_config=self.rules_config, top_n=self.top_n)
                rule_system.train_rules(window_data_handler)
                top_rule_objects = list(rule_system.trained_rule_objects.values())
                
                if verbose:
                    print(f"  Selected {len(top_rule_objects)} top rules")
                
                # Test baseline equal-weighted strategy
                baseline_strategy = rule_system.get_top_n_strategy()
                baseline_backtester = Backtester(window_data_handler, baseline_strategy)
                baseline_results = baseline_backtester.run(use_test_data=True)
                baseline_sharpe = baseline_backtester.calculate_sharpe()
                
                if verbose:
                    print(f"  Baseline results: Return: {baseline_results['total_percent_return']:.2f}%, "
                         f"Trades: {baseline_results['num_trades']}, Sharpe: {baseline_sharpe:.4f}")
                
                # Use enhanced genetic optimizer
                if self.optimization_method == 'genetic':
                    # Use enhanced genetic optimizer
                    optimizer = EnhancedGeneticOptimizer(
                        data_handler=window_data_handler,
                        rule_objects=top_rule_objects,
                        population_size=self.population_size,
                        num_generations=self.num_generations,
                        optimization_metric=self.optimization_metric,
                        random_seed=self.random_seed,
                        regularization_factor=self.reg_factor,
                        balance_factor=self.balance_factor,
                        cv_folds=2  # Use 2-fold CV within each window
                    )
                    
                    if verbose:
                        print(f"  Running genetic optimization with {len(top_rule_objects)} rules...")
                    
                    best_weights = optimizer.optimize(verbose=False)
                    
                    # Create strategy with optimized weights
                    strategy = WeightedRuleStrategy(
                        rule_objects=top_rule_objects,
                        weights=best_weights
                    )
                else:
                    # Use equal-weighted strategy if no optimization
                    strategy = baseline_strategy
                
                # Backtest on test window
                if verbose:
                    print("  Testing strategy on out-of-sample window...")
                    
                backtester = Backtester(window_data_handler, strategy)
                results = backtester.run(use_test_data=True)
                
                # Calculate performance metrics
                total_return = results['total_percent_return']
                num_trades = results['num_trades']
                sharpe = backtester.calculate_sharpe() if num_trades > 1 else 0
                
                # Calculate days in test window
                test_start_date = pd.to_datetime(test_start) if isinstance(test_start, str) else test_start
                test_end_date = pd.to_datetime(test_end) if isinstance(test_end, str) else test_end
                days_diff = (test_end_date - test_start_date).days
                annual_return = calculate_annualized_return(total_return, days_diff)
                
                if verbose:
                    print(f"  Results: Return: {total_return:.2f}% (Ann: {annual_return:.2f}%), "
                          f"Trades: {num_trades}, Sharpe: {sharpe:.4f}")
                
                # Compare with baseline
                if baseline_results['total_percent_return'] != 0:
                    improvement = ((total_return / baseline_results['total_percent_return']) - 1) * 100
                    if verbose:
                        print(f"  Improvement over baseline: {improvement:+.2f}%")
                
                # Store results
                window_results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'num_trades': num_trades,
                    'sharpe': sharpe,
                    'avg_trade': results['average_log_return'] if num_trades > 0 else 0,
                    'train_size': len(train_window),
                    'test_size': len(test_window),
                    'baseline_return': baseline_results['total_percent_return'],
                    'baseline_trades': baseline_results['num_trades'],
                    'baseline_sharpe': baseline_sharpe,
                    'improvement': improvement if baseline_results['total_percent_return'] != 0 else 0
                })
                
                # Collect trades for equity curve
                for trade in results['trades']:
                    all_trades.append({
                        'window': i + 1,
                        'entry_time': trade[0],
                        'exit_time': trade[3],
                        'entry_price': trade[2],
                        'exit_price': trade[4],
                        'log_return': trade[5],
                        'type': trade[1]
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"  Error in window {i+1}: {str(e)}")
                window_results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'total_return': 0,
                    'annual_return': 0,
                    'num_trades': 0,
                    'sharpe': 0,
                    'avg_trade': 0,
                    'train_size': len(train_window),
                    'test_size': len(test_window),
                    'error': str(e),
                    'baseline_return': 0,
                    'baseline_trades': 0,
                    'baseline_sharpe': 0,
                    'improvement': 0
                })
        
        # Compile results
        self.results = {
            'window_results': window_results,
            'all_trades': all_trades,
            'summary': self._calculate_summary(window_results)
        }
        
        if verbose:
            self._print_summary(self.results['summary'])
            
        if plot_results:
            self._plot_results(window_results, all_trades)
            
        return self.results
    
    def _calculate_summary(self, window_results):
        """
        Calculate extended summary statistics from window results.
        
        Args:
            window_results: List of result dictionaries for each window
            
        Returns:
            dict: Summary statistics
        """
        if not window_results:
            return {
                'num_windows': 0,
                'avg_return': 0,
                'total_return': 0,
                'avg_annual_return': 0,
                'avg_sharpe': 0,
                'pct_profitable_windows': 0,
                'avg_trades_per_window': 0,
                'total_trades': 0,
                'avg_baseline_return': 0,
                'avg_improvement': 0,
                'pct_outperforming_baseline': 0
            }
            
        num_windows = len(window_results)
        returns = [r['total_return'] for r in window_results]
        annual_returns = [r['annual_return'] for r in window_results]
        sharpes = [r['sharpe'] for r in window_results]
        trades = [r['num_trades'] for r in window_results]
        baseline_returns = [r['baseline_return'] for r in window_results]
        improvements = [r['improvement'] for r in window_results]
        
        # Calculate compound return across all windows
        compound_return = np.prod([1 + r/100 for r in returns]) - 1
        compound_baseline = np.prod([1 + r/100 for r in baseline_returns]) - 1
        
        # Count windows that outperform baseline
        outperforming = sum(1 for r in window_results if r['total_return'] > r['baseline_return'])
        
        return {
            'num_windows': num_windows,
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'total_return': compound_return * 100,  # Convert to percentage
            'avg_annual_return': np.mean(annual_returns),
            'avg_sharpe': np.mean(sharpes),
            'median_sharpe': np.median(sharpes),
            'pct_profitable_windows': sum(1 for r in returns if r > 0) / num_windows * 100,
            'avg_trades_per_window': np.mean(trades),
            'total_trades': sum(trades),
            'avg_baseline_return': np.mean(baseline_returns),
            'baseline_total_return': compound_baseline * 100,
            'avg_improvement': np.mean(improvements),
            'pct_outperforming_baseline': outperforming / num_windows * 100
        }
    
    def _print_summary(self, summary):
        """
        Print summary statistics from walk-forward validation.
        
        Args:
            summary: Dictionary of summary statistics
        """
        print("\n=== Walk-Forward Validation Summary ===")
        print(f"Number of windows: {summary['num_windows']}")
        print(f"Total compound return: {summary['total_return']:.2f}%")
        print(f"Baseline compound return: {summary['baseline_total_return']:.2f}%")
        print(f"Average window return: {summary['avg_return']:.2f}%")
        print(f"Average annualized return: {summary['avg_annual_return']:.2f}%")
        print(f"Average Sharpe ratio: {summary['avg_sharpe']:.4f}")
        print(f"Percentage of profitable windows: {summary['pct_profitable_windows']:.1f}%")
        print(f"Percentage outperforming baseline: {summary['pct_outperforming_baseline']:.1f}%")
        print(f"Average improvement over baseline: {summary['avg_improvement']:+.2f}%")
        print(f"Average trades per window: {summary['avg_trades_per_window']:.1f}")
        print(f"Total trades: {summary['total_trades']}")
    
    def _plot_results(self, window_results, all_trades):
        """
        Plot the walk-forward validation results with comparison to baseline.
        
        Args:
            window_results: List of result dictionaries for each window
            all_trades: List of all trades across windows
        """
        if not window_results:
            print("No results to plot")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 15))
        
        # 1. Plot returns by window compared to baseline
        ax1 = fig.add_subplot(3, 1, 1)
        windows = [r['window'] for r in window_results]
        returns = [r['total_return'] for r in window_results]
        baseline_returns = [r['baseline_return'] for r in window_results]
        
        # Create grouped bar chart
        bar_width = 0.35
        ax1.bar([w - bar_width/2 for w in windows], returns, bar_width, label='Optimized', color='blue', alpha=0.7)
        ax1.bar([w + bar_width/2 for w in windows], baseline_returns, bar_width, label='Baseline', color='orange', alpha=0.7)
        
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax1.set_title('Returns by Window')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return (%)')
        ax1.set_xticks(windows)
        ax1.legend()
        
        # 2. Plot cumulative equity curve for optimized strategy
        if all_trades:
            ax2 = fig.add_subplot(3, 1, 2)
            trades_df = pd.DataFrame(all_trades)
            trades_df = trades_df.sort_values('entry_time')
            
            # Create equity curve
            equity = [1.0]  # Start with $1
            for log_return in trades_df['log_return']:
                equity.append(equity[-1] * np.exp(log_return))
            
            ax2.plot(range(len(equity)), equity)
            ax2.set_title('Cumulative Equity Curve')
            ax2.set_xlabel('Trade')
            ax2.set_ylabel('Equity ($)')
            ax2.grid(True)
        
        # 3. Plot improvement over baseline by window
        ax3 = fig.add_subplot(3, 1, 3)
        improvements = [r['improvement'] for r in window_results]
        
        # Create bar chart with color based on improvement
        bars = ax3.bar(windows, improvements)
        
        # Color bars based on positive/negative improvement
        for i, bar in enumerate(bars):
            bar.set_color('green' if improvements[i] > 0 else 'red')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Improvement Over Baseline (%)')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_xticks(windows)
        
        plt.tight_layout()
        plt.savefig('walk_forward_results.png')
        plt.show()
        
        # Additional plot: Trade distribution
        if all_trades:
            plt.figure(figsize=(15, 6))
            
            # Plot trade returns distribution
            plt.subplot(1, 2, 1)
            plt.hist(trades_df['log_return'], bins=20)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Distribution of Trade Returns')
            plt.xlabel('Log Return')
            plt.ylabel('Frequency')
            
            # Plot trades per window
            plt.subplot(1, 2, 2)
            trade_counts = trades_df['window'].value_counts().sort_index()
            plt.bar(trade_counts.index, trade_counts.values)
            plt.title('Number of Trades per Window')
            plt.xlabel('Window')
            plt.ylabel('Number of Trades')
            plt.xticks(trade_counts.index)
            
            plt.tight_layout()
            plt.savefig('walk_forward_trade_analysis.png')
            plt.show()
if __name__ == "__main__":
    args = parse_arguments()
    # Create the validator with arguments
    validator = EnhancedWalkForwardValidator(
        data_filepath="data/data.csv",
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
            (Rule15, {'bb_period': [20, 25], 'bb_std_dev_multiplier': [1.5, 2.0, 2.5]}),
            ],
        window_size=args.window_size,
        step_size=args.step_size,
        train_pct=args.train_pct,
        top_n=args.top_n,
        optimization_method='genetic',
        optimization_metric=args.metric,
        random_seed=args.seed,
        reg_factor=args.reg_factor,
        balance_factor=args.balance_factor,
        population_size=args.population,
        num_generations=args.generations
    )
    # Run the validation
    results = validator.run_validation(verbose=True, plot_results=True)
