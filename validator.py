"""
Walk-Forward Validation Module for Algorithmic Trading

This module provides tools for walk-forward validation and cross-validation
to improve the robustness of trading strategies and prevent overfitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from backtester import Backtester, BarEvent
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from optimizer_manager import OptimizerManager, OptimizationMethod, OptimizationSequence


class WalkForwardValidator:
    """
    Walk-Forward Validation for trading strategies.
    
    This class performs rolling walk-forward validation to test strategy robustness
    by repeatedly training on in-sample data and testing on out-of-sample data.
    """
    
    def __init__(self, 
                 data_filepath, 
                 rules_config, 
                 window_size=252,       # Default: 1 year of trading days
                 step_size=63,          # Default: 3 months
                 train_pct=0.7,         # Default: 70% training, 30% testing
                 top_n=5,               # Number of top rules to use
                 optimization_method='genetic', 
                 optimization_metric='sharpe'):
        """
        Initialize the walk-forward validator.
        
        Args:
            data_filepath: Path to CSV data file
            rules_config: Configuration of rules and parameters to test
            window_size: Size of each window in trading days
            step_size: Number of days to roll forward between windows
            train_pct: Percentage of window to use for training
            top_n: Number of top rules to select
            optimization_method: Method to use for optimization ('genetic', 'grid', etc.)
            optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
        """
        self.data_filepath = data_filepath
        self.rules_config = rules_config
        self.window_size = window_size
        self.step_size = step_size
        self.train_pct = train_pct
        self.top_n = top_n
        self.optimization_method = optimization_method
        self.optimization_metric = optimization_metric
        
        # Results storage
        self.results = []
        self.full_df = None
        self.windows = []
    
    def load_data(self):
        """
        Load the full dataset for walk-forward validation.
        
        Returns:
            pd.DataFrame: The loaded data
        """
        # Create a temporary data handler to load the data
        temp_handler = CSVDataHandler(self.data_filepath, train_fraction=1.0)
        self.full_df = temp_handler.full_df
        return self.full_df
    
    @staticmethod
    def run_nested_cross_validation(data_filepath, rules_config, outer_folds=5, inner_folds=3):
        """
        Run nested cross-validation on a trading strategy.

        Args:
            data_filepath: Path to CSV data file
            rules_config: Configuration of rules and parameters
            outer_folds: Number of outer folds for evaluation
            inner_folds: Number of inner folds for optimization
            
        Returns:
            dict: Validation results
        """
        validator = NestedCrossValidator(
            data_filepath=data_filepath,
            rules_config=rules_config,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            top_n=5,
            optimization_methods=['genetic', 'equal'],
            optimization_metric='sharpe'
        )
        
        results = validator.run_validation(verbose=True, plot_results=True)
        return results
    
    def run_validation(self, verbose=True, plot_results=True):
        """
        Run walk-forward validation.

        Args:
            verbose: Whether to print progress information
            plot_results: Whether to plot validation results

        Returns:
            dict: Summary of validation results
        """
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

                if self.optimization_method == 'genetic':
                    # Optimize weights using genetic algorithm
                    optimizer = GeneticOptimizer(
                        data_handler=window_data_handler,
                        rule_objects=top_rule_objects,
                        optimization_metric=self.optimization_metric
                    )
                    best_weights = optimizer.optimize(verbose=False)

                    # Create strategy with optimized weights
                    strategy = WeightedRuleStrategy(
                        rule_objects=top_rule_objects,
                        weights=best_weights
                    )
                else:
                    # Use equal-weighted strategy if no optimization
                    strategy = rule_system.get_top_n_strategy()

                # Backtest on test window
                if verbose:
                    print("  Testing strategy on out-of-sample window...")

                backtester = Backtester(window_data_handler, strategy)
                results = backtester.run(use_test_data=True)

                # Calculate performance metrics
                total_return = results['total_percent_return']
                num_trades = results['num_trades']
                sharpe = backtester.calculate_sharpe() if num_trades > 1 else 0

                if verbose:
                    print(f"  Results: Return: {total_return:.2f}%, "
                          f"Trades: {num_trades}, Sharpe: {sharpe:.4f}")

                # Store results
                window_results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'total_return': total_return,
                    'num_trades': num_trades,
                    'sharpe': sharpe,
                    'avg_trade': results['average_log_return'] if num_trades > 0 else 0,
                    'train_size': len(train_window),
                    'test_size': len(test_window)
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
                    'num_trades': 0,
                    'sharpe': 0,
                    'avg_trade': 0,
                    'train_size': len(train_window),
                    'test_size': len(test_window),
                    'error': str(e)
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
    
    def _create_windows(self):
        """
        Create the training and testing windows for walk-forward validation.
        
        Returns:
            list: List of (train_window, test_window) tuples
        """
        self.windows = []
        
        if self.full_df is None:
            self.load_data()
            
        # Convert dataframe to list of dictionaries for consistency with data handler
        data_list = self.full_df.to_dict('records')
        
        # Calculate the number of windows
        total_length = len(data_list)
        num_windows = (total_length - self.window_size) // self.step_size + 1
        
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # Skip if window exceeds data length
            if end_idx > total_length:
                break
                
            window_data = data_list[start_idx:end_idx]
            
            # Split into training and testing
            train_size = int(self.window_size * self.train_pct)
            train_window = window_data[:train_size]
            test_window = window_data[train_size:end_idx]
            
            self.windows.append((train_window, test_window))
            
        return self.windows
    
    def _create_window_data_handler(self, train_window, test_window):
        """
        Create a data handler for a specific window.
        
        Args:
            train_window: List of training data dictionaries
            test_window: List of testing data dictionaries
            
        Returns:
            object: A data handler-like object for the specific window
        """
        # Simple mock data handler that will return only the selected window data
        class WindowDataHandler:
            def __init__(self, train_data, test_data):
                self.train_df = pd.DataFrame(train_data)
                self.test_df = pd.DataFrame(test_data)
                self.current_train_index = 0
                self.current_test_index = 0
                
            def get_next_train_bar(self):
                if self.current_train_index < len(self.train_df):
                    bar = self.train_df.iloc[self.current_train_index].to_dict()
                    self.current_train_index += 1
                    return bar
                return None
                
            def get_next_test_bar(self):
                if self.current_test_index < len(self.test_df):
                    bar = self.test_df.iloc[self.current_test_index].to_dict()
                    self.current_test_index += 1
                    return bar
                return None
                
            def reset_train(self):
                self.current_train_index = 0
                
            def reset_test(self):
                self.current_test_index = 0
        
        return WindowDataHandler(train_window, test_window)
    
    def _calculate_summary(self, window_results):
        """
        Calculate summary statistics from window results.
        
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
                'avg_sharpe': 0,
                'pct_profitable_windows': 0,
                'avg_trades_per_window': 0,
                'total_trades': 0
            }
            
        num_windows = len(window_results)
        returns = [r['total_return'] for r in window_results]
        sharpes = [r['sharpe'] for r in window_results]
        trades = [r['num_trades'] for r in window_results]
        
        # Calculate compound return across all windows
        compound_return = np.prod([1 + r/100 for r in returns]) - 1
        
        return {
            'num_windows': num_windows,
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'total_return': compound_return * 100,  # Convert to percentage
            'avg_sharpe': np.mean(sharpes),
            'median_sharpe': np.median(sharpes),
            'pct_profitable_windows': sum(1 for r in returns if r > 0) / num_windows * 100,
            'avg_trades_per_window': np.mean(trades),
            'total_trades': sum(trades)
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
        print(f"Average window return: {summary['avg_return']:.2f}%")
        print(f"Median window return: {summary['median_return']:.2f}%")
        print(f"Average Sharpe ratio: {summary['avg_sharpe']:.4f}")
        print(f"Median Sharpe ratio: {summary['median_sharpe']:.4f}")
        print(f"Percentage of profitable windows: {summary['pct_profitable_windows']:.1f}%")
        print(f"Average trades per window: {summary['avg_trades_per_window']:.1f}")
        print(f"Total trades: {summary['total_trades']}")
    
    def _plot_results(self, window_results, all_trades):
        """
        Plot the walk-forward validation results.
        
        Args:
            window_results: List of result dictionaries for each window
            all_trades: List of all trades across windows
        """
        if not window_results:
            print("No results to plot")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Plot returns by window
        ax1 = fig.add_subplot(3, 1, 1)
        windows = [r['window'] for r in window_results]
        returns = [r['total_return'] for r in window_results]
        ax1.bar(windows, returns)
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax1.set_title('Returns by Window')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return (%)')
        ax1.set_xticks(windows)
        
        # 2. Plot cumulative equity curve
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
        
        # 3. Plot Sharpe ratio by window
        ax3 = fig.add_subplot(3, 1, 3)
        sharpes = [r['sharpe'] for r in window_results]
        ax3.bar(windows, sharpes)
        ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax3.set_title('Sharpe Ratio by Window')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xticks(windows)
        
        plt.tight_layout()
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
            plt.show()
            

class CrossValidator:
    """
    Cross-Validation for trading strategies.
    
    This class performs k-fold cross-validation to assess strategy robustness
    by dividing the dataset into k folds and using each fold as a test set.
    """
    
    def __init__(self, 
                 data_filepath, 
                 rules_config, 
                 n_folds=5,               # Number of folds for cross-validation
                 top_n=5,                 # Number of top rules to use
                 optimization_method='genetic', 
                 optimization_metric='sharpe'):
        """
        Initialize the cross-validator.
        
        Args:
            data_filepath: Path to CSV data file
            rules_config: Configuration of rules and parameters to test
            n_folds: Number of folds for cross-validation
            top_n: Number of top rules to select
            optimization_method: Method to use for optimization ('genetic', 'grid', etc.)
            optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
        """
        self.data_filepath = data_filepath
        self.rules_config = rules_config
        self.n_folds = n_folds
        self.top_n = top_n
        self.optimization_method = optimization_method
        self.optimization_metric = optimization_metric
        
        # Results storage
        self.results = []
        self.full_df = None
        self.folds = []
    
    def load_data(self):
        """
        Load the full dataset for cross-validation.
        
        Returns:
            pd.DataFrame: The loaded data
        """
        # Create a temporary data handler to load the data
        temp_handler = CSVDataHandler(self.data_filepath, train_fraction=1.0)
        self.full_df = temp_handler.full_df
        return self.full_df
    
    def run_validation(self, verbose=True, plot_results=True):
        """
        Run cross-validation.
        
        Args:
            verbose: Whether to print progress information
            plot_results: Whether to plot validation results
            
        Returns:
            dict: Summary of validation results
        """
        if self.full_df is None:
            self.load_data()
            
        if verbose:
            print(f"Running {self.n_folds}-fold cross-validation")
            print(f"Total data size: {len(self.full_df)} bars")
        
        # Create folds
        self._create_folds()
        
        if verbose:
            print(f"Created {len(self.folds)} folds")
        
        # Run validation for each fold
        all_trades = []
        fold_results = []
        
        for i, (train_data, test_data) in enumerate(self.folds):
            if verbose:
                train_start = train_data[0]['timestamp'] if len(train_data) > 0 else "N/A"
                train_end = train_data[-1]['timestamp'] if len(train_data) > 0 else "N/A"
                test_start = test_data[0]['timestamp'] if len(test_data) > 0 else "N/A"
                test_end = test_data[-1]['timestamp'] if len(test_data) > 0 else "N/A"
                
                print(f"\nFold {i+1}/{len(self.folds)}:")
                print(f"  Train: {train_start} to {train_end} ({len(train_data)} bars)")
                print(f"  Test:  {test_start} to {test_end} ({len(test_data)} bars)")
            
            # Skip folds with insufficient data
            if len(train_data) < 30 or len(test_data) < 5:
                if verbose:
                    print("  Insufficient data in fold, skipping...")
                continue
            
            # Create data handler for this fold
            fold_data_handler = self._create_fold_data_handler(train_data, test_data)
            
            # Train strategy on fold
            if verbose:
                print("  Training strategy on fold...")
                
            try:
                # Train rules and get top performers
                rule_system = EventDrivenRuleSystem(rules_config=self.rules_config, top_n=self.top_n)
                rule_system.train_rules(fold_data_handler)
                top_rule_objects = list(rule_system.trained_rule_objects.values())
                
                if self.optimization_method == 'genetic':
                    # Optimize weights using genetic algorithm
                    optimizer = GeneticOptimizer(
                        data_handler=fold_data_handler,
                        rule_objects=top_rule_objects,
                        optimization_metric=self.optimization_metric
                    )
                    best_weights = optimizer.optimize(verbose=False)
                    
                    # Create strategy with optimized weights
                    strategy = WeightedRuleStrategy(
                        rule_objects=top_rule_objects,
                        weights=best_weights
                    )
                else:
                    # Use equal-weighted strategy if no optimization
                    strategy = rule_system.get_top_n_strategy()
                
                # Backtest on test fold
                if verbose:
                    print("  Testing strategy on out-of-sample fold...")
                    
                backtester = Backtester(fold_data_handler, strategy)
                results = backtester.run(use_test_data=True)
                
                # Calculate performance metrics
                total_return = results['total_percent_return']
                num_trades = results['num_trades']
                sharpe = backtester.calculate_sharpe() if num_trades > 1 else 0
                
                if verbose:
                    print(f"  Results: Return: {total_return:.2f}%, "
                          f"Trades: {num_trades}, Sharpe: {sharpe:.4f}")
                
                # Store results
                fold_results.append({
                    'fold': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'total_return': total_return,
                    'num_trades': num_trades,
                    'sharpe': sharpe,
                    'avg_trade': results['average_log_return'] if num_trades > 0 else 0,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                })
                
                # Collect trades for equity curve
                for trade in results['trades']:
                    all_trades.append({
                        'fold': i + 1,
                        'entry_time': trade[0],
                        'exit_time': trade[3],
                        'entry_price': trade[2],
                        'exit_price': trade[4],
                        'log_return': trade[5],
                        'type': trade[1]
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"  Error in fold {i+1}: {str(e)}")
                fold_results.append({
                    'fold': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'total_return': 0,
                    'num_trades': 0,
                    'sharpe': 0,
                    'avg_trade': 0,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'error': str(e)
                })
        
        # Compile results
        self.results = {
            'fold_results': fold_results,
            'all_trades': all_trades,
            'summary': self._calculate_summary(fold_results)
        }
        
        if verbose:
            self._print_summary(self.results['summary'])
            
        if plot_results:
            self._plot_results(fold_results, all_trades)
            
        return self.results
    
    def _create_folds(self):
        """
        Create folds for cross-validation.
        
        Returns:
            list: List of (train_data, test_data) tuples
        """
        self.folds = []
        
        if self.full_df is None:
            self.load_data()
            
        # Convert dataframe to list of dictionaries for consistency with data handler
        data_list = self.full_df.to_dict('records')
        
        # Create evenly-sized folds, one for each test set
        fold_size = len(data_list) // self.n_folds
        
        for i in range(self.n_folds):
            # Define test fold indices
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_folds - 1 else len(data_list)
            
            # Get test data
            test_data = data_list[test_start:test_end]
            
            # Get training data (all other folds)
            train_data = data_list[:test_start] + data_list[test_end:]
            
            self.folds.append((train_data, test_data))
            
        return self.folds
    
    def _create_fold_data_handler(self, train_data, test_data):
        """
        Create a data handler for a specific fold.
        
        Args:
            train_data: List of training data dictionaries
            test_data: List of testing data dictionaries
            
        Returns:
            object: A data handler-like object for the specific fold
        """
        # Same implementation as in WalkForwardValidator
        class FoldDataHandler:
            def __init__(self, train_data, test_data):
                self.train_df = pd.DataFrame(train_data)
                self.test_df = pd.DataFrame(test_data)
                self.current_train_index = 0
                self.current_test_index = 0
                
            def get_next_train_bar(self):
                if self.current_train_index < len(self.train_df):
                    bar = self.train_df.iloc[self.current_train_index].to_dict()
                    self.current_train_index += 1
                    return bar
                return None
                
            def get_next_test_bar(self):
                if self.current_test_index < len(self.test_df):
                    bar = self.test_df.iloc[self.current_test_index].to_dict()
                    self.current_test_index += 1
                    return bar
                return None
                
            def reset_train(self):
                self.current_train_index = 0
                
            def reset_test(self):
                self.current_test_index = 0
        
        return FoldDataHandler(train_data, test_data)
    
    def _calculate_summary(self, fold_results):
        """
        Calculate summary statistics from fold results.
        
        Args:
            fold_results: List of result dictionaries for each fold
            
        Returns:
            dict: Summary statistics
        """
        if not fold_results:
            return {
                'num_folds': 0,
                'avg_return': 0,
                'total_return': 0,
                'avg_sharpe': 0,
                'pct_profitable_folds': 0,
                'avg_trades_per_fold': 0,
                'total_trades': 0
            }
            
        num_folds = len(fold_results)
        returns = [r['total_return'] for r in fold_results]
        sharpes = [r['sharpe'] for r in fold_results]
        trades = [r['num_trades'] for r in fold_results]
        
        # Calculate compound return across all folds
        compound_return = np.prod([1 + r/100 for r in returns]) - 1
        
        return {
            'num_folds': num_folds,
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'total_return': compound_return * 100,  # Convert to percentage
            'avg_sharpe': np.mean(sharpes),
            'median_sharpe': np.median(sharpes),
            'pct_profitable_folds': sum(1 for r in returns if r > 0) / num_folds * 100,
            'avg_trades_per_fold': np.mean(trades),
            'total_trades': sum(trades)
        }
    
    def _print_summary(self, summary):
        """
        Print summary statistics from cross-validation.
        
        Args:
            summary: Dictionary of summary statistics
        """
        print("\n=== Cross-Validation Summary ===")
        print(f"Number of folds: {summary['num_folds']}")
        print(f"Total compound return: {summary['total_return']:.2f}%")
        print(f"Average fold return: {summary['avg_return']:.2f}%")
        print(f"Median fold return: {summary['median_return']:.2f}%")
        print(f"Average Sharpe ratio: {summary['avg_sharpe']:.4f}")
        print(f"Median Sharpe ratio: {summary['median_sharpe']:.4f}")
        print(f"Percentage of profitable folds: {summary['pct_profitable_folds']:.1f}%")
        print(f"Average trades per fold: {summary['avg_trades_per_fold']:.1f}")
        print(f"Total trades: {summary['total_trades']}")
    
    def _plot_results(self, fold_results, all_trades):
        """
        Plot the cross-validation results.
        
        Args:
            fold_results: List of result dictionaries for each fold
            all_trades: List of all trades across folds
        """
        if not fold_results:
            print("No results to plot")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Plot returns by fold
        ax1 = fig.add_subplot(3, 1, 1)
        folds = [r['fold'] for r in fold_results]
        returns = [r['total_return'] for r in fold_results]
        ax1.bar(folds, returns)
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax1.set_title('Returns by Fold')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Return (%)')
        ax1.set_xticks(folds)
        
        # 2. Plot cumulative equity curve
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
        
        # 3. Plot Sharpe ratio by fold
        ax3 = fig.add_subplot(3, 1, 3)
        sharpes = [r['sharpe'] for r in fold_results]
        ax3.bar(folds, sharpes)
        ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax3.set_title('Sharpe Ratio by Fold')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xticks(folds)
        
        plt.tight_layout()
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
            
            # Plot trades per fold
            plt.subplot(1, 2, 2)
            trade_counts = trades_df['fold'].value_counts().sort_index()
            plt.bar(trade_counts.index, trade_counts.values)
            plt.title('Number of Trades per Fold')
            plt.xlabel('Fold')
            plt.ylabel('Number of Trades')
            plt.xticks(trade_counts.index)
            
            plt.tight_layout()
            plt.show()


class NestedCrossValidator:
    """
    Nested Cross-Validation for more robust evaluation of trading strategies.
    
    This class performs nested cross-validation with an inner loop for
    hyperparameter optimization and an outer loop for performance evaluation.
    """
    
    def __init__(self, 
                 data_filepath, 
                 rules_config, 
                 outer_folds=5,           # Number of outer folds
                 inner_folds=3,           # Number of inner folds for optimization
                 top_n=5,                 # Number of top rules to use
                 optimization_methods=None, # Methods to compare in inner loop
                 optimization_metric='sharpe'):
        """
        Initialize the nested cross-validator.
        
        Args:
            data_filepath: Path to CSV data file
            rules_config: Configuration of rules and parameters to test
            outer_folds: Number of outer folds for final evaluation
            inner_folds: Number of inner folds for hyperparameter optimization
            top_n: Number of top rules to select
            optimization_methods: List of optimization methods to compare
            optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
        """
        self.data_filepath = data_filepath
        self.rules_config = rules_config
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.top_n = top_n
        self.optimization_methods = optimization_methods or ['genetic', 'equal']
        self.optimization_metric = optimization_metric
        
        # Results storage
        self.results = {}
        self.full_df = None
        self.outer_folds_data = []
        self.best_methods = []
    
    def load_data(self):
        """
        Load the full dataset for nested cross-validation.
        
        Returns:
            pd.DataFrame: The loaded data
        """
        # Create a temporary data handler to load the data
        temp_handler = CSVDataHandler(self.data_filepath, train_fraction=1.0)
        self.full_df = temp_handler.full_df
        return self.full_df
    
    def run_validation(self, verbose=True, plot_results=True):
        """
        Run nested cross-validation.
        
        Args:
            verbose: Whether to print progress information
            plot_results: Whether to plot validation results
            
        Returns:
            dict: Summary of validation results
        """
        if self.full_df is None:
            self.load_data()
            
        if verbose:
            print(f"Running {self.outer_folds}x{self.inner_folds} nested cross-validation")
            print(f"Total data size: {len(self.full_df)} bars")
            print(f"Comparing optimization methods: {', '.join(self.optimization_methods)}")
        
        # Create outer folds
        self._create_outer_folds()
        
        if verbose:
            print(f"Created {len(self.outer_folds_data)} outer folds")
        
        # Results storage for each optimization method
        method_results = {method: [] for method in self.optimization_methods}
        best_method_results = []
        all_trades = []
        
        # Run outer folds
        for i, (outer_train, outer_test) in enumerate(self.outer_folds_data):
            if verbose:
                print(f"\n=== Outer Fold {i+1}/{len(self.outer_folds_data)} ===")
                print(f"  Outer train size: {len(outer_train)}")
                print(f"  Outer test size: {len(outer_test)}")
            
            # Inner loop: Find best optimization method
            if verbose:
                print("  Running inner cross-validation to select best method...")
                
            inner_results = self._run_inner_cv(outer_train, verbose=verbose)
            
            # Select best method based on inner CV
            best_method = max(inner_results.items(), key=lambda x: x[1]['summary']['avg_sharpe'])[0]
            self.best_methods.append(best_method)
            
            if verbose:
                print(f"  Best method from inner CV: {best_method}")
                
            # Train with the best method on full outer training set
            if verbose:
                print(f"  Training with {best_method} on full outer training set...")
                
            outer_train_handler = self._create_data_handler(outer_train, [])
            
            try:
                # Train rules and get top performers
                rule_system = EventDrivenRuleSystem(rules_config=self.rules_config, top_n=self.top_n)
                rule_system.train_rules(outer_train_handler)
                top_rule_objects = list(rule_system.trained_rule_objects.values())
                
                # Create strategy based on best method
                if best_method == 'genetic':
                    optimizer = GeneticOptimizer(
                        data_handler=outer_train_handler,
                        rule_objects=top_rule_objects,
                        optimization_metric=self.optimization_metric
                    )
                    best_weights = optimizer.optimize(verbose=False)
                    best_strategy = WeightedRuleStrategy(
                        rule_objects=top_rule_objects,
                        weights=best_weights
                    )
                else:  # 'equal' or default
                    best_strategy = rule_system.get_top_n_strategy()
                
                # For comparison, also create strategies for all methods
                method_strategies = {}
                for method in self.optimization_methods:
                    if method == 'genetic':
                        optimizer = GeneticOptimizer(
                            data_handler=outer_train_handler,
                            rule_objects=top_rule_objects,
                            optimization_metric=self.optimization_metric
                        )
                        weights = optimizer.optimize(verbose=False)
                        method_strategies[method] = WeightedRuleStrategy(
                            rule_objects=top_rule_objects,
                            weights=weights
                        )
                    else:  # 'equal' or default
                        method_strategies[method] = rule_system.get_top_n_strategy()
                
                # Evaluate all methods on the outer test set
                for method, strategy in method_strategies.items():
                    # Create test data handler
                    outer_test_handler = self._create_data_handler([], outer_test)
                    outer_test_handler.train_df = pd.DataFrame(outer_train)  # Just for completeness
                    
                    # Backtest
                    backtester = Backtester(outer_test_handler, strategy)
                    results = backtester.run(use_test_data=True)
                    
                    # Calculate metrics
                    total_return = results['total_percent_return']
                    num_trades = results['num_trades']
                    sharpe = backtester.calculate_sharpe() if num_trades > 1 else 0
                    
                    # Store result for this method
                    method_results[method].append({
                        'fold': i + 1,
                        'total_return': total_return,
                        'num_trades': num_trades,
                        'sharpe': sharpe,
                        'avg_trade': results['average_log_return'] if num_trades > 0 else 0
                    })
                    
                    if verbose:
                        print(f"  {method} method on outer test: "
                              f"Return: {total_return:.2f}%, "
                              f"Trades: {num_trades}, "
                              f"Sharpe: {sharpe:.4f}")
                    
                    # If this is the best method, store its trades
                    if method == best_method:
                        for trade in results['trades']:
                            all_trades.append({
                                'fold': i + 1,
                                'method': method,
                                'entry_time': trade[0],
                                'exit_time': trade[3],
                                'entry_price': trade[2],
                                'exit_price': trade[4],
                                'log_return': trade[5],
                                'type': trade[1]
                            })
                        
                        best_method_results.append({
                            'fold': i + 1,
                            'method': best_method,
                            'total_return': total_return,
                            'num_trades': num_trades,
                            'sharpe': sharpe,
                            'avg_trade': results['average_log_return'] if num_trades > 0 else 0
                        })
                
            except Exception as e:
                if verbose:
                    print(f"  Error in outer fold {i+1}: {str(e)}")
                    
                # Record empty results for this fold
                for method in self.optimization_methods:
                    method_results[method].append({
                        'fold': i + 1,
                        'total_return': 0,
                        'num_trades': 0,
                        'sharpe': 0,
                        'avg_trade': 0,
                        'error': str(e)
                    })
        
        # Compile results
        summaries = {}
        for method in self.optimization_methods:
            summaries[method] = self._calculate_summary(method_results[method])
            
        best_methods_summary = self._calculate_summary(best_method_results)
        
        self.results = {
            'method_results': method_results,
            'best_method_results': best_method_results,
            'best_methods': self.best_methods,
            'all_trades': all_trades,
            'method_summaries': summaries,
            'best_methods_summary': best_methods_summary
        }
        
        if verbose:
            self._print_summaries(summaries, best_methods_summary)
            
        if plot_results:
            self._plot_results(method_results, best_method_results, all_trades)
            
        return self.results
    
    def _create_outer_folds(self):
        """
        Create outer folds for nested cross-validation.
        
        Returns:
            list: List of (train_data, test_data) tuples
        """
        self.outer_folds_data = []
        
        if self.full_df is None:
            self.load_data()
            
        # Convert dataframe to list of dictionaries
        data_list = self.full_df.to_dict('records')
        
        # Create evenly-sized folds
        fold_size = len(data_list) // self.outer_folds
        
        for i in range(self.outer_folds):
            # Define test fold indices
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.outer_folds - 1 else len(data_list)
            
            # Get test data
            test_data = data_list[test_start:test_end]
            
            # Get training data (all other folds)
            train_data = data_list[:test_start] + data_list[test_end:]
            
            self.outer_folds_data.append((train_data, test_data))
            
        return self.outer_folds_data
    
    def _run_inner_cv(self, train_data, verbose=False):
        """
        Run inner cross-validation to select the best method.
        
        Args:
            train_data: Training data for the current outer fold
            verbose: Whether to print progress information
            
        Returns:
            dict: Results for each optimization method
        """
        # Create inner folds
        inner_folds = []
        fold_size = len(train_data) // self.inner_folds
        
        for i in range(self.inner_folds):
            # Define test fold indices
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.inner_folds - 1 else len(train_data)
            
            # Get test data
            test_data = train_data[test_start:test_end]
            
            # Get training data (all other folds)
            inner_train = train_data[:test_start] + train_data[test_end:]
            
            inner_folds.append((inner_train, test_data))
        
        # Results for each method
        method_results = {method: [] for method in self.optimization_methods}
        
        # Run inner cross-validation for each fold
        for i, (inner_train, inner_test) in enumerate(inner_folds):
            if verbose:
                print(f"    Inner fold {i+1}/{len(inner_folds)}")
            
            # Create data handler for this inner fold
            inner_data_handler = self._create_data_handler(inner_train, inner_test)
            
            try:
                # Train rules and get top performers
                rule_system = EventDrivenRuleSystem(rules_config=self.rules_config, top_n=self.top_n)
                rule_system.train_rules(inner_data_handler)
                top_rule_objects = list(rule_system.trained_rule_objects.values())
                
                # Test each optimization method
                for method in self.optimization_methods:
                    if method == 'genetic':
                        # Use genetic algorithm
                        optimizer = GeneticOptimizer(
                            data_handler=inner_data_handler,
                            rule_objects=top_rule_objects,
                            optimization_metric=self.optimization_metric,
                            population_size=10,  # Smaller for faster inner loop
                            num_generations=20
                        )
                        weights = optimizer.optimize(verbose=False)
                        strategy = WeightedRuleStrategy(
                            rule_objects=top_rule_objects,
                            weights=weights
                        )
                    else:  # 'equal' or default
                        # Use equal-weighted strategy
                        strategy = rule_system.get_top_n_strategy()
                    
                    # Backtest on inner test set
                    backtester = Backtester(inner_data_handler, strategy)
                    results = backtester.run(use_test_data=True)
                    
                    # Calculate metrics
                    total_return = results['total_percent_return']
                    num_trades = results['num_trades']
                    sharpe = backtester.calculate_sharpe() if num_trades > 1 else 0
                    
                    # Store result for this method and fold
                    method_results[method].append({
                        'fold': i + 1,
                        'total_return': total_return,
                        'num_trades': num_trades,
                        'sharpe': sharpe,
                        'avg_trade': results['average_log_return'] if num_trades > 0 else 0
                    })
                    
                    if verbose:
                        print(f"      {method}: Return: {total_return:.2f}%, "
                              f"Sharpe: {sharpe:.4f}")
                
            except Exception as e:
                if verbose:
                    print(f"    Error in inner fold {i+1}: {str(e)}")
                
                # Record empty results for this fold
                for method in self.optimization_methods:
                    method_results[method].append({
                        'fold': i + 1,
                        'total_return': 0,
                        'num_trades': 0,
                        'sharpe': 0,
                        'avg_trade': 0,
                        'error': str(e)
                    })
        
        # Calculate summary for each method
        summaries = {}
        for method in self.optimization_methods:
            summaries[method] = self._calculate_summary(method_results[method])
            
        # Final results
        return {
            'method_results': method_results,
            'summary': summaries
        }
    
    def _create_data_handler(self, train_data, test_data):
        """
        Create a data handler for a specific dataset.
        
        Args:
            train_data: List of training data dictionaries
            test_data: List of testing data dictionaries
            
        Returns:
            object: A data handler-like object
        """
        class CustomDataHandler:
            def __init__(self, train_data, test_data):
                self.train_df = pd.DataFrame(train_data) if train_data else pd.DataFrame()
                self.test_df = pd.DataFrame(test_data) if test_data else pd.DataFrame()
                self.current_train_index = 0
                self.current_test_index = 0
                
            def get_next_train_bar(self):
                if self.current_train_index < len(self.train_df):
                    bar = self.train_df.iloc[self.current_train_index].to_dict()
                    self.current_train_index += 1
                    return bar
                return None
                
            def get_next_test_bar(self):
                if self.current_test_index < len(self.test_df):
                    bar = self.test_df.iloc[self.current_test_index].to_dict()
                    self.current_test_index += 1
                    return bar
                return None
                
            def reset_train(self):
                self.current_train_index = 0
                
            def reset_test(self):
                self.current_test_index = 0
        
        return CustomDataHandler(train_data, test_data)
    
    def _calculate_summary(self, fold_results):
        """
        Calculate summary statistics from fold results.
        
        Args:
            fold_results: List of result dictionaries
            
        Returns:
            dict: Summary statistics
        """
        if not fold_results:
            return {
                'num_folds': 0,
                'avg_return': 0,
                'total_return': 0,
                'avg_sharpe': 0,
                'pct_profitable_folds': 0,
                'avg_trades_per_fold': 0,
                'total_trades': 0
            }
            
        num_folds = len(fold_results)
        returns = [r['total_return'] for r in fold_results]
        sharpes = [r['sharpe'] for r in fold_results]
        trades = [r['num_trades'] for r in fold_results]
        
        # Calculate compound return across all folds
        compound_return = np.prod([1 + r/100 for r in returns]) - 1
        
        return {
            'num_folds': num_folds,
            'avg_return': np.mean(returns),
            'median_return': np.median(returns),
            'total_return': compound_return * 100,  # Convert to percentage
            'avg_sharpe': np.mean(sharpes),
            'median_sharpe': np.median(sharpes),
            'pct_profitable_folds': sum(1 for r in returns if r > 0) / num_folds * 100,
            'avg_trades_per_fold': np.mean(trades),
            'total_trades': sum(trades)
        }
    
    def _print_summaries(self, method_summaries, best_methods_summary):
        """
        Print summary statistics from nested cross-validation.
        
        Args:
            method_summaries: Dictionary of summary statistics for each method
            best_methods_summary: Summary statistics for best method selection
        """
        print("\n=== Nested Cross-Validation Summary ===")
        
        # Print summary for each method
        print("\nPerformance by Method (Inner CV):")
        methods = list(method_summaries.keys())
        print(f"{'Method':<10} {'Avg Return':<12} {'Avg Sharpe':<12} {'Avg Trades':<12}")
        print("-" * 50)
        
        for method in methods:
            summary = method_summaries[method]
            print(f"{method:<10} {summary['avg_return']:>10.2f}% {summary['avg_sharpe']:>10.4f} {summary['avg_trades_per_fold']:>10.1f}")
        
        # Print best method results
        print("\nBest Method Selection (Outer CV):")
        method_counts = {}
        for method in self.best_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
            
        total_folds = len(self.best_methods)
        for method, count in method_counts.items():
            print(f"  {method}: selected in {count}/{total_folds} folds ({count/total_folds*100:.1f}%)")
        
        # Print final performance with method selection
        print("\nFinal Performance (Outer CV with best method selection):")
        print(f"Total compound return: {best_methods_summary['total_return']:.2f}%")
        print(f"Average fold return: {best_methods_summary['avg_return']:.2f}%")
        print(f"Average Sharpe ratio: {best_methods_summary['avg_sharpe']:.4f}")
        print(f"Percentage of profitable folds: {best_methods_summary['pct_profitable_folds']:.1f}%")
        print(f"Average trades per fold: {best_methods_summary['avg_trades_per_fold']:.1f}")
        print(f"Total trades: {best_methods_summary['total_trades']}")
    
    def _plot_results(self, method_results, best_method_results, all_trades):
        """
        Plot the nested cross-validation results.
        
        Args:
            method_results: Dictionary of results for each method
            best_method_results: Results using the best method selection
            all_trades: List of all trades across folds
        """
        if not method_results or not best_method_results:
            print("No results to plot")
            return
            
        # Plot returns by fold for each method
        plt.figure(figsize=(15, 8))
        
        # Set up bar positions
        methods = list(method_results.keys())
        num_methods = len(methods)
        num_folds = max(len(method_results[m]) for m in methods)
        bar_width = 0.8 / num_methods
        
        # Plot returns by fold for each method
        for i, method in enumerate(methods):
            results = method_results[method]
            folds = [r['fold'] for r in results]
            returns = [r['total_return'] for r in results]
            positions = [fold + (i - num_methods/2 + 0.5) * bar_width for fold in folds]
            plt.bar(positions, returns, width=bar_width, label=method)
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Returns by Fold and Method')
        plt.xlabel('Fold')
        plt.ylabel('Return (%)')
        plt.xticks(range(1, num_folds + 1))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot Sharpe ratios by fold for each method
        plt.figure(figsize=(15, 8))
        
        for i, method in enumerate(methods):
            results = method_results[method]
            folds = [r['fold'] for r in results]
            sharpes = [r['sharpe'] for r in results]
            positions = [fold + (i - num_methods/2 + 0.5) * bar_width for fold in folds]
            plt.bar(positions, sharpes, width=bar_width, label=method)
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Sharpe Ratio by Fold and Method')
        plt.xlabel('Fold')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(range(1, num_folds + 1))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Plot the equity curve for the best method selection
        if all_trades:
            plt.figure(figsize=(15, 8))
            trades_df = pd.DataFrame(all_trades)
            trades_df = trades_df.sort_values('entry_time')
            
            # Create equity curve
            equity = [1.0]  # Start with $1
            for log_return in trades_df['log_return']:
                equity.append(equity[-1] * np.exp(log_return))
            
            plt.plot(range(len(equity)), equity)
            plt.title('Cumulative Equity Curve (Best Method Selection)')
            plt.xlabel('Trade')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Plot trade returns distribution
            plt.figure(figsize=(15, 6))
            plt.hist(trades_df['log_return'], bins=20)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Distribution of Trade Returns (Best Method Selection)')
            plt.xlabel('Log Return')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


# Usage examples
def run_walk_forward_validation(data_filepath, rules_config, window_size=252, step_size=63):
    """
    Run walk-forward validation on a trading strategy.
    
    Args:
        data_filepath: Path to CSV data file
        rules_config: Configuration of rules and parameters
        window_size: Size of each window in trading days
        step_size: Number of days to roll forward between windows
        
    Returns:
        dict: Validation results
    """
    validator = WalkForwardValidator(
        data_filepath=data_filepath,
        rules_config=rules_config,
        window_size=window_size,
        step_size=step_size,
        top_n=5,
        optimization_method='genetic',
        optimization_metric='sharpe'
    )
    
    results = validator.run_validation(verbose=True, plot_results=True)
    return results

def run_cross_validation(data_filepath, rules_config, n_folds=5):
    """
    Run cross-validation on a trading strategy.
    
    Args:
        data_filepath: Path to CSV data file
        rules_config: Configuration of rules and parameters
        n_folds: Number of folds for cross-validation
        
    Returns:
        dict: Validation results
    """
    validator = CrossValidator(
        data_filepath=data_filepath,
        rules_config=rules_config,
        n_folds=n_folds,
        top_n=5,
        optimization_method='genetic',
        optimization_metric='sharpe'
    )
    
    results = validator.run_validation(verbose=True, plot_results=True)
    return results
