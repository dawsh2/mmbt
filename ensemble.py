"""
Ensemble Optimization Trading System

This script implements an ensemble approach to trading strategy optimization,
where multiple strategies are optimized with different metrics and then combined
to create a more robust overall strategy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
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


def plot_performance_comparison(results_dict, title, metrics):
    """
    Plot performance comparison across different strategies.
    
    Args:
        results_dict: Dictionary mapping strategy names to backtest results
        title: Plot title
        metrics: List of metrics to show in the legend
    """
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    for strategy_name, results in results_dict.items():
        # Create equity curve from trade results
        equity = [1.0]  # Start with $1
        for trade in results['trades']:
            log_return = trade[5]
            equity.append(equity[-1] * np.exp(log_return))
        
        # Create legend label with selected metrics
        metrics_str = []
        for metric in metrics:
            if metric == 'return':
                metrics_str.append(f"Return: {results['total_percent_return']:.2f}%")
            elif metric == 'sharpe':
                metrics_str.append(f"Sharpe: {results['sharpe']:.2f}")
            elif metric == 'win_rate':
                win_rate = sum(1 for t in results['trades'] if t[5] > 0) / len(results['trades'])
                metrics_str.append(f"Win Rate: {win_rate:.2f}")
            elif metric == 'trades':
                metrics_str.append(f"Trades: {results['num_trades']}")
        
        legend_label = f"{strategy_name} ({', '.join(metrics_str)})"
        plt.plot(equity, label=legend_label)
    
    plt.title(title)
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def print_detailed_results(results, strategy_name):
    """
    Print detailed backtest results for a strategy.
    
    Args:
        results: Dictionary of backtest results
        strategy_name: Name of the strategy
    """
    print(f"\n=== {strategy_name} Performance ===")
    print(f"Total Return: {results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Calculate additional metrics
    if results['num_trades'] > 0:
        trades = results['trades']
        win_trades = [t for t in trades if t[5] > 0]
        loss_trades = [t for t in trades if t[5] < 0]
        
        win_rate = len(win_trades) / len(trades)
        avg_win = np.mean([t[5] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t[5] for t in loss_trades]) if loss_trades else 0
        
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Win: {avg_win:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Profit Factor: {abs(sum(t[5] for t in win_trades) / sum(t[5] for t in loss_trades)) if sum(t[5] for t in loss_trades) != 0 else 'N/A'}")


if __name__ == "__main__":
    # =========================
    # Configuration Settings
    # =========================
    DATA_FILE = os.path.expanduser("~/mmbt/data/data.csv")
    TRAIN_FRACTION = 0.8
    TOP_N_RULES = 8
    
    # Optimization parameters
    POPULATION_SIZE = 20
    NUM_GENERATIONS = 30
    MUTATION_RATE = 0.1
    
    # Metrics to optimize for
    OPTIMIZATION_METRICS = ['return', 'sharpe', 'win_rate']
    
    # Ensemble combination method ('voting', 'weighted', or 'consensus')
    ENSEMBLE_METHOD = 'weighted'
    
    # =========================
    # Load and Prepare Data
    # =========================
    print("\n=== Loading Data ===")
    data_handler = CSVDataHandler(DATA_FILE, train_fraction=TRAIN_FRACTION)
    
    # =========================
    # Train Basic Rules
    # =========================
    print("\n=== Training Individual Rules ===")
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
    
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=TOP_N_RULES)
    rule_system.train_rules(data_handler)
    top_rule_objects = list(rule_system.trained_rule_objects.values())
    
    print("\nSelected Top Rules:")
    for i, rule in enumerate(top_rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # =========================
    # Create Regime Detector
    # =========================
    print("\n=== Creating Regime Detector ===")
    trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
    volatility_detector = VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015)
    
    # =========================
    # Optimize Strategies with Different Metrics
    # =========================
    print("\n=== Optimizing Strategies with Different Metrics ===")
    optimized_strategies = {}
    optimization_results = {}
    
    # Configure genetic optimization parameters
    genetic_params = {
        'genetic': {
            'population_size': POPULATION_SIZE,
            'num_generations': NUM_GENERATIONS,
            'mutation_rate': MUTATION_RATE
        }
    }
    
    # Optimize using different metrics
    for metric in OPTIMIZATION_METRICS:
        print(f"\n--- Optimizing with {metric.upper()} metric ---")
        
        # Create optimizer
        optimizer = OptimizerManager(
            data_handler=data_handler,
            rule_objects=top_rule_objects
        )
        
        # Run optimization
        if metric == 'win_rate':
            # For win rate, we use regime-specific optimization
            results = optimizer.optimize(
                method=OptimizationMethod.GENETIC,
                sequence=OptimizationSequence.REGIMES_FIRST,
                metrics=metric,
                regime_detector=trend_detector,
                optimization_params=genetic_params,
                verbose=True
            )
        else:
            # For other metrics, we optimize rule weights directly
            results = optimizer.optimize(
                method=OptimizationMethod.GENETIC,
                sequence=OptimizationSequence.RULES_FIRST,
                metrics=metric,
                optimization_params=genetic_params,
                verbose=True
            )
        
        # Store results and strategy
        optimization_results[metric] = results
        optimized_strategies[metric] = optimizer.get_optimized_strategy()
    
    # =========================
    # Backtest Individual Strategies
    # =========================
    print("\n=== Backtesting Individual Strategies ===")
    backtest_results = {}
    
    # Run backtests for each optimized strategy
    for metric, strategy in optimized_strategies.items():
        print(f"\nBacktesting {metric.upper()} optimized strategy...")
        
        backtester = Backtester(data_handler, strategy)
        results = backtester.run(use_test_data=True)
        
        # Calculate Sharpe ratio
        sharpe = backtester.calculate_sharpe()
        results['sharpe'] = sharpe
        
        # Print results
        print(f"Total Return: {results['total_percent_return']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
        if results['num_trades'] > 0:
            win_rate = sum(1 for t in results['trades'] if t[5] > 0) / results['num_trades']
            print(f"Win Rate: {win_rate:.2%}")
        
        backtest_results[f"{metric.capitalize()} Strategy"] = results
    
    # =========================
    # Create and Backtest Ensemble Strategy
    # =========================
    print("\n=== Creating Ensemble Strategy ===")
    
    # For weighted combination, calculate weights based on Sharpe ratios
    weights = {}
    if ENSEMBLE_METHOD == 'weighted':
        # Use Sharpe ratios as weights (with a minimum of 0.1)
        for metric, results in optimization_results.items():
            # Use backtest Sharpe or default to 0.1 if negative
            sharpe = max(0.1, backtest_results[f"{metric.capitalize()} Strategy"]['sharpe'])
            weights[metric] = sharpe
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        print("Ensemble Weights:")
        for metric, weight in weights.items():
            print(f"  {metric.capitalize()}: {weight:.4f}")
    
    # Create ensemble strategy
    ensemble_strategy = EnsembleStrategy(
        strategies=optimized_strategies,
        combination_method=ENSEMBLE_METHOD,
        weights=weights
    )
    
    # Backtest ensemble strategy
    print("\n=== Backtesting Ensemble Strategy ===")
    ensemble_backtester = Backtester(data_handler, ensemble_strategy)
    ensemble_results = ensemble_backtester.run(use_test_data=True)
    ensemble_results['sharpe'] = ensemble_backtester.calculate_sharpe()
    
    # Add ensemble to backtest results for comparison
    backtest_results["Ensemble Strategy"] = ensemble_results
    
    # =========================
    # Print Detailed Results
    # =========================
    for strategy_name, results in backtest_results.items():
        print_detailed_results(results, strategy_name)
    
    # =========================
    # Plot Performance Comparison
    # =========================
    print("\n=== Generating Performance Comparison Plot ===")
    plot_performance_comparison(
        backtest_results,
        "Strategy Performance Comparison",
        metrics=['return', 'sharpe', 'win_rate']
    )
    
    # =========================
    # Analyze Ensemble Behavior
    # =========================
    print("\n=== Analyzing Ensemble Behavior ===")
    
    # Reset everything for analysis
    data_handler.reset_test()
    for strategy in optimized_strategies.values():
        strategy.reset()
    ensemble_strategy.reset()
    
    # Collect signals from each strategy and the ensemble
    signal_analysis = []
    
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
            
        event = MarketEvent(bar)
        
        # Get signals from individual strategies
        individual_signals = {}
        for metric, strategy in optimized_strategies.items():
            signal_info = strategy.on_bar(event)
            if signal_info:
                individual_signals[metric] = signal_info['signal']
        
        # Get ensemble signal
        ensemble_signal = ensemble_strategy.on_bar(event)
        
        # Record for analysis
        if ensemble_signal:
            signal_analysis.append({
                'timestamp': bar['timestamp'],
                'price': bar['Close'],
                'ensemble_signal': ensemble_signal['signal'],
                **{f"{metric}_signal": individual_signals.get(metric, 0) 
                   for metric in OPTIMIZATION_METRICS}
            })
    
    # Convert to DataFrame for easier analysis
    signals_df = pd.DataFrame(signal_analysis)
    
    # Calculate agreement statistics
    if len(signals_df) > 0:
        # Count how often strategies agree/disagree
        agreement_count = 0
        full_agreement_count = 0
        
        for _, row in signals_df.iterrows():
            signals = [row[f"{metric}_signal"] for metric in OPTIMIZATION_METRICS]
            if all(s == signals[0] for s in signals):
                full_agreement_count += 1
                agreement_count += 1
            elif sum(s != 0 for s in signals) < len(signals):  # Some agree on direction
                agreement_count += 1
        
        print(f"Full Agreement Rate: {full_agreement_count / len(signals_df):.2%}")
        print(f"Partial Agreement Rate: {agreement_count / len(signals_df):.2%}")
        
        # Count signal distribution
        ensemble_signals = signals_df['ensemble_signal'].value_counts()
        print("\nEnsemble Signal Distribution:")
        for signal, count in ensemble_signals.items():
            signal_name = "BUY" if signal == 1 else "SELL" if signal == -1 else "NEUTRAL"
            print(f"  {signal_name}: {count} ({count / len(signals_df):.2%})")
        
        # Save signal analysis to CSV
        signals_df.to_csv("ensemble_signal_analysis.csv", index=False)
        print("\nSignal analysis saved to 'ensemble_signal_analysis.csv'")
    
    # =========================
    # Final Summary
    # =========================
    print("\n=== Final Performance Summary ===")
    print(f"{'Strategy':<20} {'Return':<10} {'# Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 62)
    
    for strategy_name, results in backtest_results.items():
        if results['num_trades'] > 0:
            win_rate = sum(1 for t in results['trades'] if t[5] > 0) / results['num_trades']
            print(f"{strategy_name:<20} {results['total_percent_return']:>8.2f}% {results['num_trades']:>10} {results['sharpe']:>9.2f} {win_rate:>9.2%}")
        else:
            print(f"{strategy_name:<20} {results['total_percent_return']:>8.2f}% {results['num_trades']:>10} {results['sharpe']:>9.2f} {'N/A':>10}")
