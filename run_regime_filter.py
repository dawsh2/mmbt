"""
Run regime filtering optimization with existing trading rules.
"""

import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from data_handler import CSVDataHandler
from rule_system import EventDrivenRuleSystem
from backtester import Backtester
from strategy import TopNStrategy
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from regime_detection import (
    RegimeType, TrendStrengthRegimeDetector, 
    VolatilityRegimeDetector, RegimeManager
)


def plot_equity_curve(trades, title, initial_capital=10000):
    """Plot equity curve from trade data."""
    equity = [initial_capital]
    for trade in trades:
        log_return = trade[5]
        equity.append(equity[-1] * np.exp(log_return))
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title(title)
    plt.xlabel('Trade Number')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


def plot_regime_distribution(regime_counts, title):
    """Plot distribution of market regimes."""
    labels = [r.name for r in regime_counts.keys()]
    values = list(regime_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.title(title)
    plt.ylabel('Number of Bars')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


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
    (Rule15, {'bb_period': [20, 25]}),
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
print("\n=== Running Baseline Strategy ===")
# top_n_strategy = rule_system.get_top_n_strategy()
# data_handler.reset_test()
# while True:
#     bar = data_handler.get_next_test_bar()
#     if bar is None:
#         break
#     event = {"bar": bar}
#     top_n_strategy.on_bar(event)

# baseline_backtester = Backtester(data_handler, top_n_strategy)
# baseline_results = baseline_backtester.calculate_returns()
top_n_strategy = rule_system.get_top_n_strategy()
baseline_backtester = Backtester(data_handler, top_n_strategy)
baseline_results = baseline_backtester.run(use_test_data=True)

print(f"Total Return: {baseline_results['total_percent_return']:.2f}%")
print(f"Number of Trades: {baseline_results['num_trades']}")
sharpe_baseline = baseline_backtester.calculate_sharpe()
print(f"Sharpe Ratio: {sharpe_baseline:.4f}")

plot_equity_curve(baseline_results["trades"], "Baseline Strategy Equity Curve")

# 4. Set up regime detection and regime-specific strategies
print("\n=== Setting Up Regime Detection ===")
regime_start = time.time()

# Create regime detector
trend_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)

# Create regime manager
regime_manager = RegimeManager(
    regime_detector=trend_detector,
    rule_objects=top_rule_objects,
    data_handler=data_handler
)

# Analyze regime distribution in the training data
data_handler.reset_train()
trend_detector.reset()
regime_counts = {
    RegimeType.TRENDING_UP: 0,
    RegimeType.TRENDING_DOWN: 0,
    RegimeType.RANGE_BOUND: 0,
    RegimeType.UNKNOWN: 0
}

print("Analyzing regime distribution in training data...")
while True:
    bar = data_handler.get_next_train_bar()
    if bar is None:
        break
    regime = trend_detector.detect_regime(bar)
    regime_counts[regime] = regime_counts.get(regime, 0) + 1

# Print regime distribution
total_bars = sum(regime_counts.values())
print("\nRegime Distribution in Training Data:")
for regime, count in regime_counts.items():
    percentage = (count / total_bars) * 100 if total_bars > 0 else 0
    print(f"  {regime.name}: {count} bars ({percentage:.1f}%)")

# Plot regime distribution
plot_regime_distribution(regime_counts, "Market Regime Distribution in Training Data")

# Optimize regime-specific strategies
print("\n=== Optimizing Regime-Specific Strategies ===")
regime_manager.optimize_regime_strategies(verbose=True)
print(f"Regime optimization completed in {time.time() - regime_start:.2f} seconds")

# 5. Backtest the regime-based strategy
print("\n=== Backtesting Regime-Based Strategy ===")
regime_backtester = Backtester(data_handler, regime_manager)
regime_results = regime_backtester.run(use_test_data=True)

print(f"Total Return: {regime_results['total_percent_return']:.2f}%")
print(f"Number of Trades: {regime_results['num_trades']}")
sharpe_regime = regime_backtester.calculate_sharpe()
print(f"Sharpe Ratio: {sharpe_regime:.4f}")

plot_equity_curve(regime_results["trades"], "Regime-Based Strategy Equity Curve")

# 6. Analyze regime distribution and performance in the test data
data_handler.reset_test()
trend_detector.reset()
regime_counts_test = {
    RegimeType.TRENDING_UP: 0,
    RegimeType.TRENDING_DOWN: 0,
    RegimeType.RANGE_BOUND: 0,
    RegimeType.UNKNOWN: 0
}

# Map trade timestamps to regimes
trade_regimes = {}

print("\nAnalyzing regime distribution in test data...")
while True:
    bar = data_handler.get_next_test_bar()
    if bar is None:
        break
    regime = trend_detector.detect_regime(bar)
    regime_counts_test[regime] = regime_counts_test.get(regime, 0) + 1
    trade_regimes[bar['timestamp']] = regime

# Print regime distribution
total_bars_test = sum(regime_counts_test.values())
print("\nRegime Distribution in Test Data:")
for regime, count in regime_counts_test.items():
    percentage = (count / total_bars_test) * 100 if total_bars_test > 0 else 0
    print(f"  {regime.name}: {count} bars ({percentage:.1f}%)")

# Plot regime distribution
plot_regime_distribution(regime_counts_test, "Market Regime Distribution in Test Data")

# Analyze performance by regime
regime_performance = {
    RegimeType.TRENDING_UP: {'returns': [], 'count': 0},
    RegimeType.TRENDING_DOWN: {'returns': [], 'count': 0},
    RegimeType.RANGE_BOUND: {'returns': [], 'count': 0},
    RegimeType.UNKNOWN: {'returns': [], 'count': 0}
}

for trade in regime_results['trades']:
    entry_time = trade[0]
    exit_time = trade[3]
    log_return = trade[5]
    
    # Find the regime at entry (simplification - could track regime at both entry and exit)
    if entry_time in trade_regimes:
        regime = trade_regimes[entry_time]
        regime_performance[regime]['returns'].append(log_return)
        regime_performance[regime]['count'] += 1

# Print performance by regime
print("\nRegime-Based Strategy Performance by Regime:")
print(f"{'Regime':<15} {'Avg Return':<12} {'# Trades':<10} {'Win Rate':<10}")
print("-" * 47)

for regime, data in regime_performance.items():
    if data['count'] > 0:
        avg_return = np.mean(data['returns'])
        win_rate = sum(1 for r in data['returns'] if r > 0) / len(data['returns']) * 100
        print(f"{regime.name:<15} {avg_return:>10.4f} {data['count']:>10} {win_rate:>9.1f}%")

# Compare results
improvement = (regime_results['total_percent_return'] / baseline_results['total_percent_return'] - 1) * 100 if baseline_results['total_percent_return'] > 0 else 0
print("\n=== Performance Comparison ===")
print(f"{'Strategy':<25} {'Return':<10} {'# Trades':<10} {'Sharpe':<10}")
print("-" * 55)
print(f"{'Baseline':<25} {baseline_results['total_percent_return']:>8.2f}% {baseline_results['num_trades']:>10} {sharpe_baseline:>9.4f}")
print(f"{'Regime-Based':<25} {regime_results['total_percent_return']:>8.2f}% {regime_results['num_trades']:>10} {sharpe_regime:>9.4f}")
if baseline_results['total_percent_return'] > 0:
    print(f"\nReturn improvement from regime detection: {improvement:.2f}%")

print("\nOptimization complete! Results and charts saved.")
