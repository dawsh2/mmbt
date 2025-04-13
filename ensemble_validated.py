"""
Validated Ensemble Trading System

This script integrates walk-forward validation with ensemble trading strategies
to create a more robust trading system that adapts to different market conditions
and provides more reliable out-of-sample performance measurement.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from datetime import datetime

from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
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
from event import MarketEvent

class EnsembleStrategy:
    """
    Strategy that combines signals from multiple optimized strategies.
    
    This class takes multiple strategies that have been optimized with
    different metrics and combines their signals to make trading decisions.
    """
    
    def __init__(self, strategies, combination_method='voting', weights=None):
        """
        Initialize the ensemble strategy.
        
        Args:
            strategies: Dictionary mapping strategy names to strategy objects
            combination_method: Method to combine signals ('voting', 'weighted', or 'consensus')
            weights: Optional dictionary of weights for each strategy (for 'weighted' method)
        """
        self.strategies = strategies
        self.combination_method = combination_method
        self.weights = weights or {}
        self.last_signal = None
        
        # Ensure we have weights for each strategy if using weighted method
        if combination_method == 'weighted' and not all(name in self.weights for name in strategies):
            print("Warning: Not all strategies have weights. Using equal weights.")
            total_strategies = len(strategies)
            self.weights = {name: 1.0 / total_strategies for name in strategies}
    
    def on_bar(self, event):
        """
        Process a bar and generate a signal by combining multiple strategies.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            dict: Signal information
        """
        bar = event.bar
        strategy_signals = {}
        
        # Collect signals from all strategies
        for name, strategy in self.strategies.items():
            signal_info = strategy.on_bar(event)
            strategy_signals[name] = signal_info['signal'] if signal_info else 0
        
        # Combine signals using the specified method
        if self.combination_method == 'voting':
            # Simple majority vote
            votes = defaultdict(int)
            for name, signal in strategy_signals.items():
                votes[signal] += 1
            
            # Find signal with most votes (in case of tie, use 0)
            max_votes = -1
            final_signal = 0
            for signal, vote_count in votes.items():
                if vote_count > max_votes:
                    max_votes = vote_count
                    final_signal = signal
        
        elif self.combination_method == 'weighted':
            # Weighted average of signals
            weighted_sum = 0
            total_weight = 0
            for name, signal in strategy_signals.items():
                weight = self.weights.get(name, 1.0)
                weighted_sum += signal * weight
                total_weight += weight
            
            avg_signal = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Convert to discrete signal
            if avg_signal > 0.3:
                final_signal = 1
            elif avg_signal < -0.3:
                final_signal = -1
            else:
                final_signal = 0
        
        elif self.combination_method == 'consensus':
            # Require consensus (all agree) for entry, any disagreement for exit
            if all(signal == 1 for signal in strategy_signals.values()):
                final_signal = 1
            elif all(signal == -1 for signal in strategy_signals.values()):
                final_signal = -1
            else:
                final_signal = 0
        
        else:
            # Default to simple average
            avg_signal = sum(strategy_signals.values()) / len(strategy_signals)
            if avg_signal > 0.3:
                final_signal = 1
            elif avg_signal < -0.3:
                final_signal = -1
            else:
                final_signal = 0
        
        self.last_signal = {
            "timestamp": bar["timestamp"],
            "signal": final_signal,
            "price": bar["Close"],
            "strategy_signals": strategy_signals  # Include individual strategy signals for analysis
        }
        
        return self.last_signal
    
    def reset(self):
        """Reset all strategies in the ensemble."""
        for strategy in self.strategies.values():
            strategy.reset()
        self.last_signal = None


class ValidatedEnsemble:
    """
    Class for creating and validating ensemble trading strategies using walk-forward validation.
    """
    
    def __init__(self, 
                 data_filepath, 
                 rules_config, 
                 window_size=252,        # Default: 1 year of trading days
                 step_size=63,           # Default: 3 months
                 train_pct=0.7,          # Default: 70% training, 30% testing
                 top_n=5,                # Number of top rules to select
                 optimization_metrics=None,  # Metrics to optimize for
                 ensemble_method='weighted',  # How to combine strategies
                 verbose=True):
        """
        Initialize the validated ensemble trading system.
        
        Args:
            data_filepath: Path to CSV data file
            rules_config: Configuration of rules and parameters to test
            window_size: Size of each window in trading days
            step_size: Number of days to roll forward between windows
            train_pct: Percentage of window to use for training
            top_n: Number of top rules to select
            optimization_metrics: List of metrics to optimize for
            ensemble_method: Method to combine signals ('voting', 'weighted', or 'consensus')
            verbose: Whether to print progress information
        """
        self.data_filepath = data_filepath
        self.rules_config = rules_config
        self.window_size = window_size
        self.step_size = step_size
        self.train_pct = train_pct
        self.top_n = top_n
        self.optimization_metrics = optimization_metrics or ['sharpe', 'return', 'win_rate']
        self.ensemble_method = ensemble_method
        self.verbose = verbose
        
        # Results storage
        self.results = {}
        self.full_df = None
        self.windows = []
        self.optimized_strategies = {}
        self.ensemble_strategy = None
        
        # Optimization parameters
        self.optimization_params = {
            'genetic': {
                'population_size': 20,
                'num_generations': 30,
                'mutation_rate': 0.1
            }
        }
    
    def load_data(self):
        """
        Load the full dataset for validation.
        
        Returns:
            pd.DataFrame: The loaded data
        """
        # Create a temporary data handler to load the data
        temp_handler = CSVDataHandler(self.data_filepath, train_fraction=1.0)
        self.full_df = temp_handler.full_df
        return self.full_df
    
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
    
    def _train_top_rules(self, data_handler):
        """
        Train and select top performing rules on a specific dataset.
        
        Args:
            data_handler: Data handler for this window
            
        Returns:
            list: Top rule objects
        """
        if self.verbose:
            print("  Training and selecting top rules...")
            
        rule_system = EventDrivenRuleSystem(rules_config=self.rules_config, top_n=self.top_n)
        rule_system.train_rules(data_handler)
        top_rule_objects = list(rule_system.trained_rule_objects.values())
        
        if self.verbose:
            print(f"  Selected {len(top_rule_objects)} top rules")
            
        return top_rule_objects
    
    def _optimize_strategies(self, data_handler, rule_objects):
        """
        Optimize strategies using different metrics.
        
        Args:
            data_handler: Data handler for this window
            rule_objects: List of top rule objects
            
        Returns:
            dict: Optimized strategies for each metric
        """
        optimized_strategies = {}
        
        for metric in self.optimization_metrics:
            if self.verbose:
                print(f"  Optimizing with {metric.upper()} metric...")
                
            # Create optimizer
            optimizer = OptimizerManager(
                data_handler=data_handler,
                rule_objects=rule_objects
            )
            
            # Run optimization with appropriate sequence based on metric
            if metric == 'win_rate':
                # For win rate, use regime-specific optimization
                regime_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
                results = optimizer.optimize(
                    method=OptimizationMethod.GENETIC,
                    sequence=OptimizationSequence.REGIMES_FIRST,
                    metrics=metric,
                    regime_detector=regime_detector,
                    optimization_params=self.optimization_params,
                    verbose=False  # Keep quiet during optimization
                )
            else:
                # For other metrics, optimize rule weights directly
                results = optimizer.optimize(
                    method=OptimizationMethod.GENETIC,
                    sequence=OptimizationSequence.RULES_FIRST,
                    metrics=metric,
                    optimization_params=self.optimization_params,
                    verbose=False  # Keep quiet during optimization
                )
            
            # Store optimized strategy
            optimized_strategies[metric] = optimizer.get_optimized_strategy()
        
        return optimized_strategies
    
    def _create_ensemble_strategy(self, strategies, backtest_results=None):
        """
        Create an ensemble strategy from individual optimized strategies.
        
        Args:
            strategies: Dictionary of optimized strategies
            backtest_results: Optional results from backtesting individual strategies
            
        Returns:
            EnsembleStrategy: Combined strategy
        """
        # Calculate weights based on backtest results if available
        weights = {}
        if backtest_results and self.ensemble_method == 'weighted':
            # Use Sharpe ratios as weights (with a minimum of 0.1)
            for metric, results in backtest_results.items():
                # Extract the metric name from the results key
                metric_name = metric.split()[0].lower()
                if metric_name in strategies:
                    # Use backtest Sharpe or default to 0.1 if negative
                    sharpe = max(0.1, results['sharpe'])
                    weights[metric_name] = sharpe
            
            # Normalize weights to sum to 1
            if weights:
                total_weight = sum(weights.values())
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # If no weights could be calculated, use equal weights
                weights = {metric: 1.0 / len(strategies) for metric in strategies}
        else:
            # Use equal weights if no backtest results
            weights = {metric: 1.0 / len(strategies) for metric in strategies}
        
        # Create ensemble strategy
        ensemble = EnsembleStrategy(
            strategies=strategies,
            combination_method=self.ensemble_method,
            weights=weights
        )
        
        return ensemble, weights
    
    def _backtest_strategies(self, data_handler, strategies, use_test_data=True):
        """
        Backtest multiple strategies on the same dataset.
        
        Args:
            data_handler: Data handler for this window
            strategies: Dictionary of strategies to test
            use_test_data: Whether to use test data (True) or train data (False)
            
        Returns:
            dict: Backtest results for each strategy
        """
        results = {}
        
        for name, strategy in strategies.items():
            # Reset strategy before backtesting
            strategy.reset()
            
            # Run backtest
            backtester = Backtester(data_handler, strategy)
            backtest_results = backtester.run(use_test_data=use_test_data)
            
            # Calculate Sharpe ratio
            sharpe = backtester.calculate_sharpe()
            backtest_results['sharpe'] = sharpe
            
            # Calculate win rate
            if backtest_results['num_trades'] > 0:
                win_rate = sum(1 for t in backtest_results['trades'] if t[5] > 0) / backtest_results['num_trades']
                backtest_results['win_rate'] = win_rate
            else:
                backtest_results['win_rate'] = 0.0
            
            # Store results
            results[f"{name.capitalize()} Strategy"] = backtest_results
        
        return results
    
    def run_validation(self, plot_results=True):
        """
        Run walk-forward validation with ensemble strategies.
        
        Args:
            plot_results: Whether to plot validation results
            
        Returns:
            dict: Summary of validation results
        """
        start_time = time.time()
        
        if self.full_df is None:
            self.load_data()
            
        if self.verbose:
            print(f"Running walk-forward validation with {self.window_size} day windows "
                  f"and {self.step_size} day steps")
            print(f"Total data size: {len(self.full_df)} bars")
            print(f"Optimization metrics: {', '.join(self.optimization_metrics)}")
            print(f"Ensemble method: {self.ensemble_method}")
        
        # Create windows
        self._create_windows()
        
        if self.verbose:
            print(f"Created {len(self.windows)} validation windows")
        
        # Results storage
        all_window_results = []
        all_strategy_trades = {metric: [] for metric in self.optimization_metrics}
        all_ensemble_trades = []
        
        # Process each window
        for i, (train_window, test_window) in enumerate(self.windows):
            if self.verbose:
                train_start = train_window[0]['timestamp'] if train_window else "N/A"
                train_end = train_window[-1]['timestamp'] if train_window else "N/A"
                test_start = test_window[0]['timestamp'] if test_window else "N/A" 
                test_end = test_window[-1]['timestamp'] if test_window else "N/A"
                
                print(f"\n=== Window {i+1}/{len(self.windows)} ===")
                print(f"  Train: {train_start} to {train_end} ({len(train_window)} bars)")
                print(f"  Test:  {test_start} to {test_end} ({len(test_window)} bars)")
            
            # Skip windows with insufficient data
            if len(train_window) < 30 or len(test_window) < 5:
                if self.verbose:
                    print("  Insufficient data in window, skipping...")
                continue
            
            try:
                # Create data handler for this window
                window_data_handler = self._create_window_data_handler(train_window, test_window)
                
                # 1. Train and select top rules
                top_rule_objects = self._train_top_rules(window_data_handler)
                
                # 2. Optimize strategies with different metrics on training data
                optimized_strategies = self._optimize_strategies(window_data_handler, top_rule_objects)
                
                # 3. Backtest individual strategies on training data (for ensemble weights)
                if self.verbose:
                    print("  Backtesting individual strategies on training data...")
                    
                train_backtest_results = self._backtest_strategies(
                    window_data_handler, 
                    optimized_strategies,
                    use_test_data=False  # Use training data for weight calculation
                )
                
                # 4. Create ensemble strategy
                if self.verbose:
                    print("  Creating ensemble strategy...")
                    
                ensemble_strategy, weights = self._create_ensemble_strategy(
                    optimized_strategies, 
                    train_backtest_results
                )
                
                if self.verbose:
                    print("  Ensemble weights:")
                    for metric, weight in weights.items():
                        print(f"    {metric.capitalize()}: {weight:.4f}")
                
                # 5. Backtest all strategies on the test data
                if self.verbose:
                    print("  Backtesting all strategies on test data...")
                
                # Include ensemble in the strategies to test
                test_strategies = {**optimized_strategies, 'ensemble': ensemble_strategy}
                
                test_backtest_results = self._backtest_strategies(
                    window_data_handler, 
                    test_strategies,
                    use_test_data=True  # Use test data for final evaluation
                )
                
                # 6. Store results for this window
                window_result = {
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'weights': weights,
                    'train_results': train_backtest_results,
                    'test_results': test_backtest_results
                }
                
                all_window_results.append(window_result)
                
                # Track trades for equity curves
                for metric in self.optimization_metrics:
                    strategy_results = test_backtest_results.get(f"{metric.capitalize()} Strategy", {})
                    if 'trades' in strategy_results:
                        for trade in strategy_results['trades']:
                            all_strategy_trades[metric].append({
                                'window': i + 1,
                                'entry_time': trade[0],
                                'exit_time': trade[3],
                                'entry_price': trade[2],
                                'exit_price': trade[4],
                                'log_return': trade[5],
                                'type': trade[1]
                            })
                
                # Track ensemble trades
                ensemble_results = test_backtest_results.get("Ensemble Strategy", {})
                if 'trades' in ensemble_results:
                    for trade in ensemble_results['trades']:
                        all_ensemble_trades.append({
                            'window': i + 1,
                            'entry_time': trade[0],
                            'exit_time': trade[3],
                            'entry_price': trade[2],
                            'exit_price': trade[4],
                            'log_return': trade[5],
                            'type': trade[1]
                        })
                
                # 7. Show window results
                if self.verbose:
                    print("\n  Test Results:")
                    print(f"  {'Strategy':<20} {'Return':<10} {'# Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
                    print("  " + "-" * 62)
                    
                    for strategy_name, results in test_backtest_results.items():
                        win_rate = results.get('win_rate', 0)
                        print(f"  {strategy_name:<20} {results['total_percent_return']:>8.2f}% "
                              f"{results['num_trades']:>10} {results['sharpe']:>9.2f} "
                              f"{win_rate:>9.2%}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error in window {i+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Calculate overall performance
        overall_results = self._calculate_overall_performance(all_window_results, all_strategy_trades, all_ensemble_trades)
        
        # Save results
        self.results = {
            'window_results': all_window_results,
            'strategy_trades': all_strategy_trades,
            'ensemble_trades': all_ensemble_trades,
            'overall_results': overall_results
        }
        
        # Print summary
        if self.verbose:
            total_time = time.time() - start_time
            print(f"\n=== Validation Complete in {total_time:.1f} seconds ===")
            self._print_overall_summary(overall_results)
        
        # Plot results
        if plot_results:
            self._plot_validation_results()
        
        return self.results
    
    def _calculate_overall_performance(self, window_results, strategy_trades, ensemble_trades):
        """
        Calculate overall performance across all validation windows.
        
        Args:
            window_results: List of results from each window
            strategy_trades: Dictionary of trades for each strategy
            ensemble_trades: List of trades for the ensemble strategy
        
        Returns:
            dict: Overall performance metrics
        """
        # Prepare result containers
        strategy_metrics = {
            metric: {
                'returns': [],
                'sharpes': [],
                'win_rates': [],
                'num_trades': [],
                'profitable_windows': 0
            } for metric in self.optimization_metrics
        }
        ensemble_metrics = {
            'returns': [],
            'sharpes': [],
            'win_rates': [],
            'num_trades': [],
            'profitable_windows': 0
        }
        
        # Collect metrics from each window
        for window in window_results:
            test_results = window.get('test_results', {})
            
            # Process individual strategy results
            for metric in self.optimization_metrics:
                strategy_name = f"{metric.capitalize()} Strategy"
                if strategy_name in test_results:
                    result = test_results[strategy_name]
                    strategy_metrics[metric]['returns'].append(result['total_percent_return'])
                    strategy_metrics[metric]['sharpes'].append(result['sharpe'])
                    strategy_metrics[metric]['win_rates'].append(result.get('win_rate', 0))
                    strategy_metrics[metric]['num_trades'].append(result['num_trades'])
                    
                    if result['total_percent_return'] > 0:
                        strategy_metrics[metric]['profitable_windows'] += 1
            
            # Process ensemble results
            if "Ensemble Strategy" in test_results:
                result = test_results["Ensemble Strategy"]
                ensemble_metrics['returns'].append(result['total_percent_return'])
                ensemble_metrics['sharpes'].append(result['sharpe'])
                ensemble_metrics['win_rates'].append(result.get('win_rate', 0))
                ensemble_metrics['num_trades'].append(result['num_trades'])
                
                if result['total_percent_return'] > 0:
                    ensemble_metrics['profitable_windows'] += 1
        
        # Calculate aggregate metrics for each strategy
        strategy_summaries = {}
        for metric, data in strategy_metrics.items():
            num_windows = len(data['returns'])
            if num_windows > 0:
                # Calculate compound return across all windows
                compound_return = np.prod([1 + r/100 for r in data['returns']]) - 1
                
                strategy_summaries[metric] = {
                    'num_windows': num_windows,
                    'avg_return': np.mean(data['returns']),
                    'median_return': np.median(data['returns']),
                    'total_return': compound_return * 100,  # Convert to percentage
                    'avg_sharpe': np.mean(data['sharpes']),
                    'median_sharpe': np.median(data['sharpes']),
                    'avg_win_rate': np.mean(data['win_rates']),
                    'pct_profitable_windows': data['profitable_windows'] / num_windows * 100,
                    'avg_trades_per_window': np.mean(data['num_trades']),
                    'total_trades': sum(data['num_trades'])
                }
        
        # Calculate aggregate metrics for ensemble
        ensemble_summary = {}
        num_windows = len(ensemble_metrics['returns'])
        if num_windows > 0:
            # Calculate compound return across all windows
            compound_return = np.prod([1 + r/100 for r in ensemble_metrics['returns']]) - 1
            
            ensemble_summary = {
                'num_windows': num_windows,
                'avg_return': np.mean(ensemble_metrics['returns']),
                'median_return': np.median(ensemble_metrics['returns']),
                'total_return': compound_return * 100,  # Convert to percentage
                'avg_sharpe': np.mean(ensemble_metrics['sharpes']),
                'median_sharpe': np.median(ensemble_metrics['sharpes']),
                'avg_win_rate': np.mean(ensemble_metrics['win_rates']),
                'pct_profitable_windows': ensemble_metrics['profitable_windows'] / num_windows * 100,
                'avg_trades_per_window': np.mean(ensemble_metrics['num_trades']),
                'total_trades': sum(ensemble_metrics['num_trades'])
            }
        
        # Calculate equity curves
        strategy_equity = {}
        for metric, trades in strategy_trades.items():
            if trades:
                # Sort trades by entry time
                sorted_trades = sorted(trades, key=lambda x: x['entry_time'])
                
                # Calculate equity curve
                equity = [1.0]  # Start with $1
                for trade in sorted_trades:
                    equity.append(equity[-1] * np.exp(trade['log_return']))
                
                strategy_equity[metric] = equity
        
        # Calculate ensemble equity curve
        ensemble_equity = []
        if ensemble_trades:
            # Sort trades by entry time
            sorted_trades = sorted(ensemble_trades, key=lambda x: x['entry_time'])
            
            # Calculate equity curve
            equity = [1.0]  # Start with $1
            for trade in sorted_trades:
                equity.append(equity[-1] * np.exp(trade['log_return']))
            
            ensemble_equity = equity
        
        return {
            'strategy_summaries': strategy_summaries,
            'ensemble_summary': ensemble_summary,
            'strategy_equity': strategy_equity,
            'ensemble_equity': ensemble_equity
        }
    
    def _print_overall_summary(self, overall_results):
        """
        Print overall summary of validation results.
        
        Args:
            overall_results: Dictionary of overall performance metrics
        """
        print("\n=== Overall Walk-Forward Validation Results ===")
        
        # Print summaries for individual strategies
        strategy_summaries = overall_results.get('strategy_summaries', {})
        ensemble_summary = overall_results.get('ensemble_summary', {})
        
        if strategy_summaries:
            print("\nIndividual Strategy Performance:")
            print(f"{'Strategy':<15} {'Avg Return':<12} {'Total Return':<15} {'Avg Sharpe':<12} {'Win Rate':<10} {'Profitable':<10} {'Trades':<8}")
            print("-" * 85)
            
            for metric, summary in strategy_summaries.items():
                print(f"{metric.capitalize():<15} "
                      f"{summary['avg_return']:>10.2f}% "
                      f"{summary['total_return']:>13.2f}% "
                      f"{summary['avg_sharpe']:>11.2f} "
                      f"{summary['avg_win_rate']:>9.2%} "
                      f"{summary['pct_profitable_windows']:>9.1f}% "
                      f"{summary['total_trades']:>7}")
        
        # Print ensemble summary
        if ensemble_summary:
            print("\nEnsemble Strategy Performance:")
            print(f"Total Return: {ensemble_summary['total_return']:.2f}%")
            print(f"Average Window Return: {ensemble_summary['avg_return']:.2f}%")
            print(f"Average Sharpe Ratio: {ensemble_summary['avg_sharpe']:.2f}")
            print(f"Average Win Rate: {ensemble_summary['avg_win_rate']:.2f}")
            print(f"Percentage of Profitable Windows: {ensemble_summary['pct_profitable_windows']:.1f}%")
            print(f"Total Trades: {ensemble_summary['total_trades']}")
            print(f"Average Trades per Window: {ensemble_summary['avg_trades_per_window']:.1f}")
    
    def _plot_validation_results(self):
        """
        Plot validation results including equity curves and performance metrics.
        """
        if not self.results:
            print("No results to plot")
            return
            
        overall_results = self.results.get('overall_results', {})
        strategy_equity = overall_results.get('strategy_equity', {})
        ensemble_equity = overall_results.get('ensemble_equity', [])
        
        # 1. Plot equity curves
        if strategy_equity or ensemble_equity:
            plt.figure(figsize=(14, 8))
            
            # Plot individual strategy equity curves
            for metric, equity in strategy_equity.items():
                plt.plot(equity, label=f"{metric.capitalize()} Strategy")
            
            # Plot ensemble equity curve
            if ensemble_equity:
                plt.plot(ensemble_equity, linewidth=2, label="Ensemble Strategy")
            
            plt.title('Equity Curves Across All Validation Windows')
            plt.xlabel('Trade Number')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            plt.savefig("equity_curves.png")
            plt.show()
        
        # 2. Plot window returns by strategy
        window_results = self.results.get('window_results', [])
        if window_results:
            # Extract returns by window and strategy
            windows = [r['window'] for r in window_results]
            strategy_returns = {metric: [] for metric in self.optimization_metrics}
            strategy_returns['ensemble'] = []
            
            for window in window_results:
                test_results = window.get('test_results', {})
                
                for metric in self.optimization_metrics:
                    strategy_name = f"{metric.capitalize()} Strategy"
                    if strategy_name in test_results:
                        strategy_returns[metric].append(test_results[strategy_name]['total_percent_return'])
                    else:
                        strategy_returns[metric].append(0)
                
                if "Ensemble Strategy" in test_results:
                    strategy_returns['ensemble'].append(test_results["Ensemble Strategy"]['total_percent_return'])
                else:
                    strategy_returns['ensemble'].append(0)
            
            # Plot returns by window
            plt.figure(figsize=(14, 8))
            
            # Set width and positions for grouped bars
            num_strategies = len(strategy_returns)
            width = 0.8 / num_strategies
            
            # Plot each strategy
            for i, (strategy_name, returns) in enumerate(strategy_returns.items()):
                positions = [w + (i - num_strategies/2 + 0.5) * width for w in windows]
                plt.bar(positions, returns, width=width, label=f"{strategy_name.capitalize()}")
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Returns by Window and Strategy')
            plt.xlabel('Window')
            plt.ylabel('Return (%)')
            plt.xticks(windows)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("window_returns.png")
            plt.show()
            
            # 3. Plot Sharpe ratios by window
            strategy_sharpes = {metric: [] for metric in self.optimization_metrics}
            strategy_sharpes['ensemble'] = []
            
            for window in window_results:
                test_results = window.get('test_results', {})
                
                for metric in self.optimization_metrics:
                    strategy_name = f"{metric.capitalize()} Strategy"
                    if strategy_name in test_results:
                        strategy_sharpes[metric].append(test_results[strategy_name]['sharpe'])
                    else:
                        strategy_sharpes[metric].append(0)
                
                if "Ensemble Strategy" in test_results:
                    strategy_sharpes['ensemble'].append(test_results["Ensemble Strategy"]['sharpe'])
                else:
                    strategy_sharpes['ensemble'].append(0)
            
            # Plot Sharpe ratios by window
            plt.figure(figsize=(14, 8))
            
            # Plot each strategy
            for i, (strategy_name, sharpes) in enumerate(strategy_sharpes.items()):
                positions = [w + (i - num_strategies/2 + 0.5) * width for w in windows]
                plt.bar(positions, sharpes, width=width, label=f"{strategy_name.capitalize()}")
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Sharpe Ratios by Window and Strategy')
            plt.xlabel('Window')
            plt.ylabel('Sharpe Ratio')
            plt.xticks(windows)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("window_sharpes.png")
            plt.show()
        
        # 4. Plot average weights used in ensemble across windows
        if window_results:
            # Extract weights by window
            metrics = self.optimization_metrics
            all_weights = {metric: [] for metric in metrics}
            
            for window in window_results:
                weights = window.get('weights', {})
                
                for metric in metrics:
                    all_weights[metric].append(weights.get(metric, 0))
            
            # Calculate average weights
            avg_weights = {metric: np.mean(weights) for metric, weights in all_weights.items()}
            
            # Plot average weights
            plt.figure(figsize=(10, 6))
            plt.bar(avg_weights.keys(), avg_weights.values())
            plt.title('Average Ensemble Weights Across All Windows')
            plt.xlabel('Strategy')
            plt.ylabel('Average Weight')
            plt.ylim(0, max(avg_weights.values()) * 1.1)  # Add some headroom
            
            # Add value labels on bars
            for metric, weight in avg_weights.items():
                plt.text(metric, weight + 0.01, f"{weight:.2f}", ha='center')
                
            plt.grid(axis='y', alpha=0.3)
            plt.savefig("ensemble_weights.png")
            plt.show()
        
        # 5. Plot trade return distribution
        ensemble_trades = self.results.get('ensemble_trades', [])
        if ensemble_trades:
            log_returns = [trade['log_return'] for trade in ensemble_trades]
            
            plt.figure(figsize=(12, 6))
            
            plt.hist(log_returns, bins=20)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.title('Distribution of Ensemble Strategy Trade Returns')
            plt.xlabel('Log Return')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig("trade_distribution.png")
            plt.show()
            
            # Plot trade returns vs trade duration
            if ensemble_trades and 'entry_time' in ensemble_trades[0] and 'exit_time' in ensemble_trades[0]:
                # Calculate trade durations in days
                durations = []
                for trade in ensemble_trades:
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    duration = (exit_time - entry_time).days
                    durations.append(max(1, duration))  # Ensure at least 1 day
                
                # Plot
                plt.figure(figsize=(12, 6))
                plt.scatter(durations, log_returns)
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.title('Trade Returns vs. Duration')
                plt.xlabel('Trade Duration (days)')
                plt.ylabel('Log Return')
                plt.grid(True, alpha=0.3)
                plt.savefig("trade_duration_returns.png")
                plt.show()
    
    def save_to_file(self, filename='validated_ensemble_results.pkl'):
        """
        Save validation results to a file.
        
        Args:
            filename: Name of the file to save results to
        """
        import pickle
        
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
            
        if self.verbose:
            print(f"Results saved to {filename}")
    
    def load_from_file(self, filename='validated_ensemble_results.pkl'):
        """
        Load validation results from a file.
        
        Args:
            filename: Name of the file to load results from
        """
        import pickle
        
        try:
            with open(filename, 'rb') as f:
                self.results = pickle.load(f)
                
            if self.verbose:
                print(f"Results loaded from {filename}")
                
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error loading results: {str(e)}")
            return False
    
    def get_best_ensemble(self):
        """
        Create the best ensemble strategy based on validation results.
        
        Returns:
            EnsembleStrategy: Optimized ensemble strategy
        """
        if not self.results:
            raise ValueError("No validation results available. Run validation first.")
        
        # Train on the full dataset
        data_handler = CSVDataHandler(self.data_filepath, train_fraction=0.8)
        
        # Train and select top rules
        rule_system = EventDrivenRuleSystem(rules_config=self.rules_config, top_n=self.top_n)
        rule_system.train_rules(data_handler)
        top_rule_objects = list(rule_system.trained_rule_objects.values())
        
        # Calculate optimal weights based on validation
        strategy_summaries = self.results['overall_results']['strategy_summaries']
        
        # Use Sharpe ratios as weights
        weights = {}
        for metric, summary in strategy_summaries.items():
            # Use average Sharpe or a small positive value if negative
            weights[metric] = max(0.1, summary['avg_sharpe'])
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        if self.verbose:
            print("\nCreating best ensemble with weights:")
            for metric, weight in weights.items():
                print(f"  {metric.capitalize()}: {weight:.4f}")
        
        # Create strategies for each metric
        optimized_strategies = {}
        
        for metric in self.optimization_metrics:
            # Create optimizer
            optimizer = OptimizerManager(
                data_handler=data_handler,
                rule_objects=top_rule_objects
            )
            
            # Run optimization with appropriate sequence based on metric
            if metric == 'win_rate':
                # For win rate, use regime-specific optimization
                regime_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
                results = optimizer.optimize(
                    method=OptimizationMethod.GENETIC,
                    sequence=OptimizationSequence.REGIMES_FIRST,
                    metrics=metric,
                    regime_detector=regime_detector,
                    optimization_params=self.optimization_params,
                    verbose=self.verbose
                )
            else:
                # For other metrics, optimize rule weights directly
                results = optimizer.optimize(
                    method=OptimizationMethod.GENETIC,
                    sequence=OptimizationSequence.RULES_FIRST,
                    metrics=metric,
                    optimization_params=self.optimization_params,
                    verbose=self.verbose
                )
            
            # Store optimized strategy
            optimized_strategies[metric] = optimizer.get_optimized_strategy()
        
        # Create ensemble strategy
        ensemble_strategy = EnsembleStrategy(
            strategies=optimized_strategies,
            combination_method=self.ensemble_method,
            weights=weights
        )
        
        return ensemble_strategy


# Main execution 
if __name__ == "__main__":
    # =========================
    # Configuration Settings
    # =========================
    DATA_FILE = os.path.expanduser("~/mmbt/data/data.csv")
    WINDOW_SIZE = 252  # ~ 1 year of trading days
    STEP_SIZE = 63    # ~ 3 months
    TRAIN_PCT = 0.7   # 70% training, 30% testing
    TOP_N_RULES = 8   # Number of top rules to use
    
    # Rules configuration
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
    
    # Alternative rule configuration with expanded parameters
    expanded_rules_config = [
        # Rule0: Simple Moving Average Crossover
        (Rule0, {'fast_window': [5, 10, 15], 'slow_window': [20, 30, 50, 100]}),
        
        # Rule1: Simple Moving Average Crossover with MA1 and MA2
        (Rule1, {'ma1': [5, 10, 15, 20], 'ma2': [30, 50, 100]}),
        
        # Rule2: EMA and MA Crossover
        (Rule2, {'ema1_period': [5, 10, 15, 20], 'ma2_period': [30, 50, 100]}),
        
        # Rule3: EMA and EMA Crossover
        (Rule3, {'ema1_period': [5, 10, 15], 'ema2_period': [20, 30, 50]}),
        
        # Rule4: DEMA and MA Crossover
        (Rule4, {'dema1_period': [5, 10, 15], 'ma2_period': [20, 30, 50]}),
        
        # Rule5: DEMA and DEMA Crossover
        (Rule5, {'dema1_period': [5, 10, 15], 'dema2_period': [20, 30, 50]}),
        
        # Rule6: TEMA and MA Crossover
        (Rule6, {'tema1_period': [5, 10, 15], 'ma2_period': [20, 30, 50]}),
        
        # Rule7: Stochastic Oscillator
        (Rule7, {'stoch1_period': [10, 14, 20], 'stochma2_period': [3, 5, 7]}),
        
        # Rule8: Vortex Indicator
        (Rule8, {'vortex1_period': [10, 14, 20], 'vortex2_period': [10, 14, 20]}),
        
        # Rule9: Ichimoku Cloud
        (Rule9, {'p1': [7, 9, 11], 'p2': [26, 30, 52]}),
        
        # Rule10: RSI Overbought/Oversold
        (Rule10, {'rsi1_period': [7, 10, 14], 'c2_threshold': [30, 40, 50]}),
        
        # Rule11: CCI Overbought/Oversold
        (Rule11, {'cci1_period': [10, 14, 20], 'c2_threshold': [80, 100, 120]}),
        
        # Rule12: RSI-based strategy
        (Rule12, {'rsi_period': [7, 10, 14], 'overbought': [70, 80], 'oversold': [20, 30]}),
        
        # Rule13: Stochastic Oscillator strategy
        (Rule13, {'stoch_period': [10, 14, 20], 'stoch_d_period': [3, 5, 7]}),
        
        # Rule14: ATR Trailing Stop
        (Rule14, {'atr_period': [10, 14, 20], 'atr_multiplier': [2, 3]}),
        
        # Rule15: Bollinger Bands strategy
        (Rule15, {'bb_period': [15, 20, 25], 'bb_std_dev': [2, 2.5]}),
    ]
    
    # Create and run validated ensemble
    validator = ValidatedEnsemble(
        data_filepath=DATA_FILE,
        rules_config=rules_config,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        train_pct=TRAIN_PCT,
        top_n=TOP_N_RULES,
        optimization_metrics=['sharpe', 'return', 'win_rate'],
        ensemble_method='weighted',
        verbose=True
    )
    
    # Run validation
    results = validator.run_validation(plot_results=True)
    
    # Save results
    validator.save_to_file('validated_ensemble_results.pkl')
    
    # Get best ensemble strategy
    best_ensemble = validator.get_best_ensemble()
    
    # Optional: Backtest on full out-of-sample data
    print("\n=== Final Backtest on Full Out-of-Sample Data ===")
    data_handler = CSVDataHandler(DATA_FILE, train_fraction=0.8)
    backtester = Backtester(data_handler, best_ensemble)
    final_results = backtester.run(use_test_data=True)
    
    # Print final results
    print(f"Total Return: {final_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {final_results['num_trades']}")
    print(f"Sharpe Ratio: {backtester.calculate_sharpe():.4f}")
    
    if final_results['num_trades'] > 0:
        win_rate = sum(1 for t in final_results['trades'] if t[5] > 0) / final_results['num_trades']
        print(f"Win Rate: {win_rate:.2%}")
        
    # Plot final equity curve
    if final_results['trades']:
        equity = [1.0]  # Start with $1
        for trade in final_results['trades']:
            log_return = trade[5]
            equity.append(equity[-1] * np.exp(log_return))
        
        plt.figure(figsize=(14, 8))
        plt.plot(equity)
        plt.title('Final Ensemble Strategy Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.savefig("final_equity_curve.png")
        plt.show()
