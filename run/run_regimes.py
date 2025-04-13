"""
Run script for regime-based strategy training and evaluation.

This script trains trading rules under different market regimes defined
in regime_detection.py and compares their performance on out-of-sample data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester, BarEvent
from strategy import TopNStrategy
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from regime_detection import (
    RegimeType, TrendStrengthRegimeDetector, 
    VolatilityRegimeDetector
)
from trade_analyzer import TradeAnalyzer
from trade_visualizer import TradeVisualizer

def plot_performance_comparison(results_dict, title="Regime-Based Strategy Performance"):
    """Plot equity curves for multiple regime-based strategies."""
    plt.figure(figsize=(14, 8))
    
    # Plot equity curves
    for approach_name, results in results_dict.items():
        if 'trades' not in results or not results['trades']:
            print(f"Warning: No trades for {approach_name}")
            continue
            
        # Create equity curve from trade results
        equity = [10000]  # Start with $10,000
        for trade in results['trades']:
            log_return = trade[5]
            equity.append(equity[-1] * np.exp(log_return))
        
        plt.plot(equity, label=f"{approach_name} ({results['total_percent_return']:.2f}%)")
    
    plt.title(title)
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def run_regime_based_evaluation():
    # Configuration
    data_filepath = "data/data.csv"
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
    ]
    
    print("Loading data...")
    data_handler = CSVDataHandler(data_filepath, train_fraction=0.8)
    
    # Initialize regime detectors
    trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)
    volatility_detector = VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015)
    
    # Step 1: Identify different regimes in the training data
    print("\nIdentifying market regimes in training data...")
    data_handler.reset_train()
    trend_regime_bars = defaultdict(list)
    volatility_regime_bars = defaultdict(list)
    all_bars = []
    bar_index = 0
    
    trend_detector.reset()
    volatility_detector.reset()
    
    while True:
        bar = data_handler.get_next_train_bar()
        if bar is None:
            break
            
        # Identify regimes
        trend_regime = trend_detector.detect_regime(bar)
        volatility_regime = volatility_detector.detect_regime(bar)
        
        # Store bars by regime
        trend_regime_bars[trend_regime].append((bar_index, bar))
        volatility_regime_bars[volatility_regime].append((bar_index, bar))
        all_bars.append((bar_index, bar))
        bar_index += 1
    
    # Print regime distribution
    print("\nTrend Regime Distribution in Training Data:")
    for regime, bars in trend_regime_bars.items():
        print(f"  {regime.name}: {len(bars)} bars ({len(bars)/len(all_bars)*100:.1f}%)")
    
    print("\nVolatility Regime Distribution in Training Data:")
    for regime, bars in volatility_regime_bars.items():
        print(f"  {regime.name}: {len(bars)} bars ({len(bars)/len(all_bars)*100:.1f}%)")
    
    # Step 2: Create separate rule systems and data handlers for each regime
    rule_systems = {}
    
    # For trending up regime
    if RegimeType.TRENDING_UP in trend_regime_bars and len(trend_regime_bars[RegimeType.TRENDING_UP]) >= 50:
        print(f"\nTraining rules for {RegimeType.TRENDING_UP.name} regime...")
        trending_up_data = create_regime_specific_data(trend_regime_bars[RegimeType.TRENDING_UP])
        rule_systems[RegimeType.TRENDING_UP] = train_rules_for_regime(trending_up_data, rules_config)
    
    # For trending down regime
    if RegimeType.TRENDING_DOWN in trend_regime_bars and len(trend_regime_bars[RegimeType.TRENDING_DOWN]) >= 50:
        print(f"\nTraining rules for {RegimeType.TRENDING_DOWN.name} regime...")
        trending_down_data = create_regime_specific_data(trend_regime_bars[RegimeType.TRENDING_DOWN])
        rule_systems[RegimeType.TRENDING_DOWN] = train_rules_for_regime(trending_down_data, rules_config)
    
    # For range-bound regime
    if RegimeType.RANGE_BOUND in trend_regime_bars and len(trend_regime_bars[RegimeType.RANGE_BOUND]) >= 50:
        print(f"\nTraining rules for {RegimeType.RANGE_BOUND.name} regime...")
        range_bound_data = create_regime_specific_data(trend_regime_bars[RegimeType.RANGE_BOUND])
        rule_systems[RegimeType.RANGE_BOUND] = train_rules_for_regime(range_bound_data, rules_config)
    
    # For volatile regime
    if RegimeType.VOLATILE in volatility_regime_bars and len(volatility_regime_bars[RegimeType.VOLATILE]) >= 50:
        print(f"\nTraining rules for {RegimeType.VOLATILE.name} regime...")
        volatile_data = create_regime_specific_data(volatility_regime_bars[RegimeType.VOLATILE])
        rule_systems[RegimeType.VOLATILE] = train_rules_for_regime(volatile_data, rules_config)
    
    # For low volatility regime
    if RegimeType.LOW_VOLATILITY in volatility_regime_bars and len(volatility_regime_bars[RegimeType.LOW_VOLATILITY]) >= 50:
        print(f"\nTraining rules for {RegimeType.LOW_VOLATILITY.name} regime...")
        low_vol_data = create_regime_specific_data(volatility_regime_bars[RegimeType.LOW_VOLATILITY])
        rule_systems[RegimeType.LOW_VOLATILITY] = train_rules_for_regime(low_vol_data, rules_config)
    
    # Train a general strategy on all data for comparison
    print("\nTraining baseline strategy on all data...")
    baseline_rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    baseline_rule_system.train_rules(data_handler)
    baseline_strategy = baseline_rule_system.get_top_n_strategy()
    
    # Step 3: Evaluate each regime-specific rule system on out-of-sample data
    print("\n=== Evaluating strategies on out-of-sample data ===")
    
    # For evaluation, we need to identify the regimes in the test data
    data_handler.reset_test()
    trend_detector.reset()
    volatility_detector.reset()
    
    test_trend_regimes = {}
    test_volatility_regimes = {}
    
    test_bars = []
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
            
        trend_regime = trend_detector.detect_regime(bar)
        volatility_regime = volatility_detector.detect_regime(bar)
        
        # Store the regime for this timestamp
        test_trend_regimes[bar['timestamp']] = trend_regime
        test_volatility_regimes[bar['timestamp']] = volatility_regime
        test_bars.append(bar)
    
    # Create regime-aware strategy for evaluation
    class RegimeAwareStrategy:
        def __init__(self, regime_strategies, default_strategy):
            self.regime_strategies = regime_strategies
            self.default_strategy = default_strategy
            self.current_trend_regime = None
            self.current_volatility_regime = None
            
        def on_bar(self, event):
            bar = event.bar
            timestamp = bar['timestamp']
            
            # Determine current regimes
            self.current_trend_regime = test_trend_regimes.get(timestamp, RegimeType.UNKNOWN)
            self.current_volatility_regime = test_volatility_regimes.get(timestamp, RegimeType.UNKNOWN)
            
            # Select appropriate strategy
            if self.current_trend_regime in self.regime_strategies:
                return self.regime_strategies[self.current_trend_regime].on_bar(event)
            elif self.current_volatility_regime in self.regime_strategies:
                return self.regime_strategies[self.current_volatility_regime].on_bar(event)
            else:
                return self.default_strategy.on_bar(event)
                
        def reset(self):
            for strategy in self.regime_strategies.values():
                strategy.reset()
            self.default_strategy.reset()
            self.current_trend_regime = None
            self.current_volatility_regime = None
    
    # Create regime-specific strategies
    regime_strategies = {}
    for regime, rule_system in rule_systems.items():
        regime_strategies[regime] = rule_system.get_top_n_strategy()
    
    # Create the adaptive strategy that uses different strategies based on regime
    adaptive_strategy = RegimeAwareStrategy(regime_strategies, baseline_strategy)
    
    # Run backtests on test data
    results = {}
    
    # First, run the baseline
    print("\nBacktesting baseline strategy...")
    baseline_backtester = Backtester(data_handler, baseline_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)
    results["Baseline"] = baseline_results
    print(f"  Return: {baseline_results['total_percent_return']:.2f}%, "
          f"Trades: {baseline_results['num_trades']}, "
          f"Sharpe: {baseline_backtester.calculate_sharpe():.4f}")
    
    # Then run the adaptive strategy
    print("\nBacktesting adaptive strategy...")
    adaptive_backtester = Backtester(data_handler, adaptive_strategy)
    adaptive_results = adaptive_backtester.run(use_test_data=True)
    results["Adaptive"] = adaptive_results
    print(f"  Return: {adaptive_results['total_percent_return']:.2f}%, "
          f"Trades: {adaptive_results['num_trades']}, "
          f"Sharpe: {adaptive_backtester.calculate_sharpe():.4f}")
    
    # Also test each regime-specific strategy individually across all test data
    for regime, strategy in regime_strategies.items():
        print(f"\nBacktesting {regime.name} strategy on all test data...")
        regime_backtester = Backtester(data_handler, strategy)
        regime_results = regime_backtester.run(use_test_data=True)
        results[f"{regime.name}"] = regime_results
        print(f"  Return: {regime_results['total_percent_return']:.2f}%, "
              f"Trades: {regime_results['num_trades']}, "
              f"Sharpe: {regime_backtester.calculate_sharpe():.4f}")
    
    # Step 4: Compare performance across strategies
    print("\n=== Performance Comparison ===")
    print(f"{'Strategy':<20} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 50)
    
    for name, result in results.items():
        if name == "Baseline":
            sharpe = baseline_backtester.calculate_sharpe()
        elif name == "Adaptive":
            sharpe = adaptive_backtester.calculate_sharpe()
        else:
            # For regime-specific strategies, calculate their sharpe
            strategy_backtester = Backtester(data_handler, regime_strategies[next(r for r in RegimeType if r.name == name)])
            sharpe = strategy_backtester.calculate_sharpe()
            
        print(f"{name:<20} {result['total_percent_return']:>8.2f}% {result['num_trades']:>10} {sharpe:>9.4f}")
    
    # Step 5: Visualize the results
    plot_performance_comparison(results, "Strategy Performance by Market Regime")
    
    # Create more detailed analytics and visualizations
    try:
        print("\nGenerating trade analysis and visualizations...")
        
        # Create price data for regime chart
        price_data = pd.DataFrame([{
            'timestamp': bar['timestamp'],
            'price': bar['Close']
        } for bar in test_bars])
        
        # Convert regime data to the format expected by the visualizer
        regime_data = {timestamp: regime.name for timestamp, regime in test_trend_regimes.items()}
        
        # Create trade analyzer for the adaptive strategy
        analyzer = TradeAnalyzer(adaptive_results)
        performance_metrics = analyzer.calculate_performance_metrics()
        
        print("\nAdaptive Strategy Performance Metrics:")
        for metric, value in performance_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Use the trade visualizer to create charts
        visualizer = TradeVisualizer()
        
        # Create regime chart
        fig1 = visualizer.create_regime_chart(price_data, regime_data, adaptive_results['trades'], 
                                              "Price Action with Market Regimes")
        fig1.savefig("regime_chart.png")
        
        # Create performance chart
        fig2 = visualizer.create_comparison_chart(results, "Strategy Comparison")
        fig2.savefig("strategy_comparison.png")
        
        print("Analysis and visualizations saved to current directory.")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    
    return results


def create_regime_specific_data(regime_bars):
    """Create a mock data handler with only bars from a specific regime."""
    class RegimeSpecificDataHandler:
        def __init__(self, bars):
            self.bars = [bar for _, bar in bars]
            self.index = 0

        def get_next_train_bar(self):
            if self.index < len(self.bars):
                bar = self.bars[self.index]
                self.index += 1
                return bar
            return None

        def get_next_test_bar(self):
            return self.get_next_train_bar()

        def reset_train(self):
            self.index = 0

        def reset_test(self):
            self.index = 0
    
    return RegimeSpecificDataHandler(regime_bars)


def train_rules_for_regime(data_handler, rules_config):
    """Train rules using data from a specific regime."""
    rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=5)
    rule_system.train_rules(data_handler)
    
    # Print selected rules for this regime
    print("\nSelected Top Rules for this regime:")
    for i, rule in enumerate(rule_system.trained_rule_objects.values()):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    return rule_system


if __name__ == "__main__":
    results = run_regime_based_evaluation()
    print("\nEvaluation complete!")
