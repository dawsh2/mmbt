"""
Run script for comparing different regime detection methods with optimized rules.

This script compares multiple regime detection algorithms (Trend, Volatility, Kaufman),
optimizes rule weights for each regime type, and evaluates the performance of each
regime detection approach in an adaptive trading system.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import defaultdict

from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester, BarEvent
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from regime_detection import (
    RegimeType, TrendStrengthRegimeDetector, 
    VolatilityRegimeDetector, KaufmanRegimeDetector
)
from trade_analyzer import TradeAnalyzer
from trade_visualizer import TradeVisualizer

# At the top of the file, add this function for KAMA calculation
def kaufman_adaptive_ma(series, n=10, fast_ema=2, slow_ema=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA) for a series.
    
    Args:
        series: Price series
        n: Period for Efficiency Ratio
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        
    Returns:
        pandas.Series: KAMA values
    """
    # Calculate direction and volatility
    direction = abs(series.diff(n))
    volatility = series.diff().abs().rolling(n).sum()
    
    # Calculate efficiency ratio
    er = direction / volatility
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2/(fast_ema + 1)
    slow_sc = 2/(slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with first price
    kama = pd.Series(index=series.index)
    kama.iloc[0] = series.iloc[0]
    
    # Calculate KAMA iteratively
    for i in range(1, len(series)):
        if i < n:
            kama.iloc[i] = series.iloc[i]
        else:
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
            
    return kama

# Then modify the regime_detectors initialization part of the script:
regime_detectors = {
    "Trend": TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25),
    "Volatility": VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015),
    # Add Kaufman detector with custom parameters
    "Kaufman": KaufmanRegimeDetector({
        'kama_fast_period': 20,
        'kama_slow_period': 50,
        'efficiency_ratio_period': 10,
        'efficiency_strong_threshold': 0.6,
        'efficiency_weak_threshold': 0.3,
        'ma_proximity_threshold': 0.001
    })
}

def plot_performance_comparison(results_dict, title="Regime Detector Comparison"):
    """Plot equity curves for multiple regime detection strategies."""
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

def run_regime_detector_comparison():
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
    
    # GA optimization parameters
    ga_params = {
        'population_size': 20,
        'num_generations': 30,
        'mutation_rate': 0.1,
        'optimization_metric': 'sharpe'  # Options: 'sharpe', 'return', 'win_rate'
    }
    
    start_time = time.time()
    print("Loading data...")
    data_handler = CSVDataHandler(data_filepath, train_fraction=0.8)
    
    # Initialize different regime detectors
    regime_detectors = {
        "Trend": TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25),
        "Volatility": VolatilityRegimeDetector(lookback_period=20, volatility_threshold=0.015),
        # Add Kaufman detector with custom parameters
        "Kaufman": KaufmanRegimeDetector({
            'kama_fast_period': 20,
            'kama_slow_period': 50,
            'efficiency_ratio_period': 10,
            'efficiency_strong_threshold': 0.6,
            'efficiency_weak_threshold': 0.3,
            'ma_proximity_threshold': 0.001
        })
    }

    
    # First, train the rule system to get all rule objects with best parameters
    print("\nTraining rule system to get optimized rule objects...")
    full_rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=16)
    full_rule_system.train_rules(data_handler)
    all_rule_objects = list(full_rule_system.trained_rule_objects.values())
    
    print(f"\nTrained {len(all_rule_objects)} rule objects:")
    for i, rule in enumerate(all_rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # Create baseline strategy with GA-optimized weights on all data
    print("\nOptimizing baseline strategy on all data...")
    data_handler.reset_train()
    baseline_optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=all_rule_objects,
        population_size=ga_params['population_size'],
        num_generations=ga_params['num_generations'],
        mutation_rate=ga_params['mutation_rate'],
        optimization_metric=ga_params['optimization_metric']
    )
    baseline_weights = baseline_optimizer.optimize(verbose=True)
    baseline_strategy = WeightedRuleStrategy(
        rule_objects=all_rule_objects,
        weights=baseline_weights
    )
    
    # Main loop: Process each regime detector
    detector_strategies = {}
    regime_distributions = {}
    
    for detector_name, detector in regime_detectors.items():
        print(f"\n==== Processing {detector_name} Regime Detector ====")
        
        # Step 1: Identify regimes in training data
        data_handler.reset_train()
        detector.reset()
        regime_bars = defaultdict(list)
        bar_index = 0
        
        while True:
            bar = data_handler.get_next_train_bar()
            if bar is None:
                break
                
            regime = detector.detect_regime(bar)
            regime_bars[regime].append((bar_index, bar))
            bar_index += 1
        
        # Store regime distribution for analysis
        regime_distributions[detector_name] = {
            regime.name: len(bars) for regime, bars in regime_bars.items()
        }
        
        # Print regime distribution
        print(f"\n{detector_name} Regime Distribution in Training Data:")
        total_bars = sum(len(bars) for bars in regime_bars.values())
        for regime, bars in regime_bars.items():
            print(f"  {regime.name}: {len(bars)} bars ({len(bars)/total_bars*100:.1f}%)")
        
        # Step 2: Optimize rule weights for each regime
        regime_strategies = {}
        for regime, bars in regime_bars.items():
            if len(bars) >= 50:  # Need enough data for meaningful optimization
                print(f"\nOptimizing weights for {regime.name} regime...")
                regime_data = create_regime_specific_data(bars)
                weights = optimize_weights_for_regime(regime_data, all_rule_objects, ga_params)
                regime_strategies[regime] = WeightedRuleStrategy(
                    rule_objects=all_rule_objects,
                    weights=weights
                )
        
        # Step 3: Identify regimes in test data
        data_handler.reset_test()
        detector.reset()
        test_regimes = {}
        test_bars = []
        
        while True:
            bar = data_handler.get_next_test_bar()
            if bar is None:
                break
                
            regime = detector.detect_regime(bar)
            test_regimes[bar['timestamp']] = regime
            test_bars.append(bar)
        
        # Step 4: Create a regime-aware strategy for this detector
        detector_strategies[detector_name] = create_regime_aware_strategy(
            test_regimes, regime_strategies, baseline_strategy
        )
    
    # Step 5: Evaluate all strategies on test data
    results = {}
    
    # First, run the baseline
    print("\nBacktesting baseline strategy...")
    baseline_backtester = Backtester(data_handler, baseline_strategy)
    baseline_results = baseline_backtester.run(use_test_data=True)
    results["Baseline (GA)"] = baseline_results
    print(f"  Return: {baseline_results['total_percent_return']:.2f}%, "
          f"Trades: {baseline_results['num_trades']}, "
          f"Sharpe: {baseline_backtester.calculate_sharpe():.4f}")
    
    # Also run an equal-weighted strategy for comparison
    equal_weights = np.ones(len(all_rule_objects)) / len(all_rule_objects)
    equal_strategy = WeightedRuleStrategy(
        rule_objects=all_rule_objects,
        weights=equal_weights
    )
    equal_backtester = Backtester(data_handler, equal_strategy)
    equal_results = equal_backtester.run(use_test_data=True)
    results["Equal Weights"] = equal_results
    print(f"  Return: {equal_results['total_percent_return']:.2f}%, "
          f"Trades: {equal_results['num_trades']}, "
          f"Sharpe: {equal_backtester.calculate_sharpe():.4f}")
    
    # Then run each regime detector's strategy
    for detector_name, strategy in detector_strategies.items():
        print(f"\nBacktesting {detector_name} regime-based strategy...")
        detector_backtester = Backtester(data_handler, strategy)
        detector_results = detector_backtester.run(use_test_data=True)
        results[f"{detector_name} Regime"] = detector_results
        print(f"  Return: {detector_results['total_percent_return']:.2f}%, "
              f"Trades: {detector_results['num_trades']}, "
              f"Sharpe: {detector_backtester.calculate_sharpe():.4f}")
    
    # Step 6: Compare performance across strategies
    print("\n=== Performance Comparison ===")
    print(f"{'Strategy':<20} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
    print("-" * 50)
    
    for name, result in results.items():
        if name == "Baseline (GA)":
            sharpe = baseline_backtester.calculate_sharpe()
        elif name == "Equal Weights":
            sharpe = equal_backtester.calculate_sharpe()
        else:
            # For regime-based strategies, we need to recalculate
            detector_name = name.split()[0]  # Extract detector name
            detector_backtester = Backtester(data_handler, detector_strategies[detector_name])
            sharpe = detector_backtester.calculate_sharpe()
            
        print(f"{name:<20} {result['total_percent_return']:>8.2f}% {result['num_trades']:>10} {sharpe:>9.4f}")
    
    # Step 7: Visualize the results
    plot_performance_comparison(results, "Comparison of Regime Detection Methods")
    
    # Step 8: Visualize regime distributions
    visualize_regime_distributions(regime_distributions, "Regime Distributions by Detector")
    
    # Create more detailed analytics and visualizations
    try:
        print("\nGenerating trade analysis and visualizations...")
        
        # Create price data for regime chart
        price_data = pd.DataFrame([{
            'timestamp': bar['timestamp'],
            'price': bar['Close']
        } for bar in test_bars])
        
        # Create trade visualizer
        visualizer = TradeVisualizer()
        
        # For each detector, create a regime chart
        for detector_name, strategy in detector_strategies.items():
            # Get detector's test regimes
            data_handler.reset_test()
            regime_detector = regime_detectors[detector_name]
            regime_detector.reset()
            
            test_regimes = {}
            while True:
                bar = data_handler.get_next_test_bar()
                if bar is None:
                    break
                    
                regime = regime_detector.detect_regime(bar)
                test_regimes[bar['timestamp']] = regime.name
            
            # Get detector's results
            detector_results = results[f"{detector_name} Regime"]
            
            # Create regime chart
            fig = visualizer.create_regime_chart(
                price_data, test_regimes, detector_results['trades'],
                f"{detector_name} Regime Detection with Trades"
            )
            fig.savefig(f"{detector_name}_regime_chart.png")
            
            # Create performance analysis
            analyzer = TradeAnalyzer(detector_results)
            performance_metrics = analyzer.calculate_performance_metrics()
            
            # Print key metrics
            print(f"\n{detector_name} Regime Performance Metrics:")
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        # Create comparison chart for all strategies
        fig = visualizer.create_comparison_chart(results, "Strategy Performance Comparison")
        fig.savefig("regime_detector_comparison.png")
        
        print("Analysis and visualizations saved to current directory.")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    
    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    return results, regime_distributions


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


def optimize_weights_for_regime(data_handler, rule_objects, ga_params):
    """Optimize rule weights using genetic algorithm for a specific regime."""
    # Create genetic optimizer for this regime's data
    optimizer = GeneticOptimizer(
        data_handler=data_handler,
        rule_objects=rule_objects,
        population_size=ga_params['population_size'],
        num_generations=ga_params['num_generations'],
        mutation_rate=ga_params['mutation_rate'],
        optimization_metric=ga_params['optimization_metric']
    )
    
    # Run optimization
    optimal_weights = optimizer.optimize(verbose=True)
    
    print(f"Optimal weights: {optimal_weights}")
    return optimal_weights


def create_regime_aware_strategy(test_regimes, regime_strategies, default_strategy):
    """Create a regime-aware strategy that switches based on detected regime."""
    class RegimeAwareStrategy:
        def __init__(self, regimes, strategies, default):
            self.regimes = regimes
            self.strategies = strategies
            self.default_strategy = default
            self.current_regime = None
            
        def on_bar(self, event):
            bar = event.bar
            timestamp = bar['timestamp']
            
            # Determine current regime
            self.current_regime = self.regimes.get(timestamp, None)
            
            # Select appropriate strategy
            if self.current_regime in self.strategies:
                return self.strategies[self.current_regime].on_bar(event)
            else:
                return self.default_strategy.on_bar(event)
                
        def reset(self):
            for strategy in self.strategies.values():
                strategy.reset()
            self.default_strategy.reset()
            self.current_regime = None
    
    return RegimeAwareStrategy(test_regimes, regime_strategies, default_strategy)


def visualize_regime_distributions(regime_distributions, title):
    """Create a bar chart of regime distributions for each detector."""
    # Determine all unique regimes across all detectors
    all_regimes = set()
    for distributions in regime_distributions.values():
        all_regimes.update(distributions.keys())
    
    # Sort regimes for consistent ordering
    sorted_regimes = sorted(list(all_regimes))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width and positions for grouped bars
    num_detectors = len(regime_distributions)
    width = 0.8 / num_detectors
    
    # Plot each detector's distribution
    for i, (detector_name, distribution) in enumerate(regime_distributions.items()):
        # Get counts for each regime (using 0 for missing regimes)
        counts = [distribution.get(regime, 0) for regime in sorted_regimes]
        
        # Calculate bar positions
        positions = [j + (i - num_detectors/2 + 0.5) * width for j in range(len(sorted_regimes))]
        
        # Plot bars
        bars = ax.bar(positions, counts, width=width, label=detector_name)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontsize=8)
    
    # Set labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_ylabel('Number of Bars', fontsize=12)
    ax.set_xticks(range(len(sorted_regimes)))
    ax.set_xticklabels(sorted_regimes, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


if __name__ == "__main__":
    try:
        results, distributions = run_regime_detector_comparison()
        print("\nComparison complete!")
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
