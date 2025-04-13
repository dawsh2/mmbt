"""
Enhanced Regime-Based Trading Strategy with Calmar Ratio Optimization

This script runs an optimized version of the regime-based strategy that performed well,
incorporating Rules0-15 with expanded parameter sets and using Calmar ratio for optimization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

from data_handler import CSVDataHandler
from backtester import Backtester
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)

# Import from the confidence-based system
from run_signals import (
    EnhancedWaveletRegimeDetector, 
    ConfidenceWeightedStrategy,
    MAConfidenceRule,
    RSIConfidenceRule,
    VolatilityRule
)

from regime_detection import RegimeType, RegimeManager

# First, add Calmar ratio calculation to Backtester class if not already there
def add_calmar_calculation_to_backtester():
    """Add Calmar ratio calculation to Backtester class if not already present."""
    import backtester
    
    if not hasattr(backtester.Backtester, 'calculate_calmar'):
        def calculate_calmar(self):
            """
            Calculate Calmar ratio (return divided by maximum drawdown).
            
            Returns:
                float: Calmar ratio
            """
            if not self.trades or len(self.trades) < 2:
                return 0.0
            
            # Calculate total return from trades
            total_log_return = sum(trade[5] for trade in self.trades)  # Assuming log_return is at index 5
            total_return = (np.exp(total_log_return) - 1)  # Convert to decimal (not percentage)
            
            # Calculate maximum drawdown
            equity = [1.0]  # Start with $1
            for trade in self.trades:
                log_return = trade[5]
                equity.append(equity[-1] * np.exp(log_return))
            
            # Calculate running maximum and drawdowns
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_drawdown = np.max(drawdowns)
            
            if max_drawdown == 0:
                return float('inf') if total_return > 0 else 0
            
            # Calculate Calmar ratio
            calmar = total_return / max_drawdown
            
            return calmar
        
        # Add method to Backtester class
        backtester.Backtester.calculate_calmar = calculate_calmar
        print("Added Calmar ratio calculation to Backtester class")


# Custom RegimeAwareStrategyFactory with Calmar optimization
class EnhancedRegimeStrategyFactory:
    """Creates strategies with awareness of current regime and calmar optimization."""
    
    def create_strategy(self, regime, rule_objects, optimization_params):
        """Create a strategy for a specific regime with optimized parameters."""
        # If optimization_params is a list/array of weights, use them directly
        if isinstance(optimization_params, (list, np.ndarray)):
            weights = optimization_params
        else:
            # Otherwise assume it's a dictionary with weights
            weights = optimization_params.get('weights', np.ones(len(rule_objects)) / len(rule_objects))
        
        # Create confidence-weighted strategy with regime-specific thresholds
        if regime == RegimeType.TRENDING_UP:
            buy_threshold = 0.25  # More aggressive for buy in uptrend
            sell_threshold = -0.35  # More conservative for sell in uptrend
        elif regime == RegimeType.TRENDING_DOWN:
            buy_threshold = 0.35  # More conservative for buy in downtrend
            sell_threshold = -0.25  # More aggressive for sell in downtrend
        elif regime == RegimeType.VOLATILE:
            buy_threshold = 0.4  # More conservative in volatile markets
            sell_threshold = -0.4
        else:  # Range-bound or unknown
            buy_threshold = 0.3
            sell_threshold = -0.3
        
        return ConfidenceWeightedStrategy(
            rule_objects=rule_objects, 
            weights=weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
    
    def create_default_strategy(self, rule_objects):
        """Create a default strategy with equal weights."""
        weights = np.ones(len(rule_objects)) / len(rule_objects)
        return ConfidenceWeightedStrategy(
            rule_objects=rule_objects,
            weights=weights
        )


# Custom optimization for the RegimeManager
def optimize_regime_specific_strategies(regime_manager, optimization_metric='calmar'):
    """
    Optimize strategies for different market regimes using the specified metric.
    
    Args:
        regime_manager: The RegimeManager instance
        optimization_metric: Metric to optimize ('calmar', 'sharpe', 'sortino', etc.)
        
    Returns:
        dict: Mapping from regime to optimized parameters
    """
    from genetic_optimizer import GeneticOptimizer
    
    if regime_manager.data_handler is None:
        raise ValueError("Data handler must be provided for optimization")
    
    # Identify bars in each regime using the full dataset
    regime_bars = {}
    regime_manager.regime_detector.reset()
    regime_manager.data_handler.reset_train()
    
    bar_index = 0
    while True:
        bar = regime_manager.data_handler.get_next_train_bar()
        if bar is None:
            break
        regime = regime_manager.regime_detector.detect_regime(bar)
        if regime not in regime_bars:
            regime_bars[regime] = []
        regime_bars[regime].append((bar_index, bar))
        bar_index += 1
    
    # Initialize the dictionary to store optimal parameters
    optimal_params = {}
    
    print("\nRegime Distribution in Training Data:")
    for regime, bars in regime_bars.items():
        print(f"  {regime.name}: {len(bars)} bars")
    
    for regime, bars in regime_bars.items():
        if len(bars) >= 100:  # Need enough data for meaningful optimization
            print(f"\nOptimizing strategy for {regime.name} regime "
                  f"({len(bars)} bars) using {optimization_metric} metric")
            
            # Create a regime-specific data handler
            regime_specific_data = create_regime_specific_data(bars)
            
            # Optimize parameters for this regime
            optimizer = GeneticOptimizer(
                data_handler=regime_specific_data,
                rule_objects=regime_manager.rule_objects,
                population_size=20,
                num_generations=30,
                optimization_metric=optimization_metric
            )
            
            # Run optimization
            optimal_params[regime] = optimizer.optimize(verbose=True)
            
            # Create and store the optimized strategy
            regime_manager.regime_strategies[regime] = regime_manager.strategy_factory.create_strategy(
                regime, regime_manager.rule_objects, optimal_params[regime]
            )
            
            print(f"Optimized weights for {regime.name}: {optimal_params[regime]}")
        else:
            print(f"Insufficient data for {regime.name} regime "
                  f"({len(bars)} bars). Using default strategy.")
            regime_manager.regime_strategies[regime] = regime_manager.strategy_factory.create_default_strategy(
                regime_manager.rule_objects
            )
    
    return optimal_params


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


def plot_equity_curve(trades, title, initial_capital=10000):
    """Plot equity curve from trade data."""
    if not trades:
        print(f"Warning: No trades to plot for {title}")
        return
        
    equity = [initial_capital]
    
    for trade in trades:
        log_return = trade[5]  # Assuming log return is at index 5
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


def run_enhanced_regime_strategy():
    """
    Run the enhanced regime-based strategy with expanded rules and Calmar optimization.
    """
    # Ensure Calmar ratio calculation is available
    add_calmar_calculation_to_backtester()
    
    # Load data
    data_file = os.path.expanduser("~/mmbt/data/data.csv")
    if not os.path.exists(data_file):
        data_file = "data/data.csv"  # Try alternative path
    
    data_handler = CSVDataHandler(data_file, train_fraction=0.8)
    print(f"Loaded data from {data_file}")
    
    # Create expanded parameter set for Rules0-15
    expanded_rules_config = [
        # Rule0: Simple Moving Average Crossover
        (Rule0, {'fast_window': [5, 8, 10, 12, 15], 'slow_window': [20, 30, 40, 50, 60, 100]}),
        
        # Rule1: Simple Moving Average Crossover with MA1 and MA2
        (Rule1, {'ma1': [5, 8, 10, 15, 20], 'ma2': [30, 40, 50, 60, 100]}),
        
        # Rule2: EMA and MA Crossover
        (Rule2, {'ema1_period': [5, 8, 10, 15, 20], 'ma2_period': [30, 40, 50, 60, 100]}),
        
        # Rule3: EMA and EMA Crossover
        (Rule3, {'ema1_period': [5, 8, 10, 15, 20], 'ema2_period': [30, 40, 50, 60, 100]}),
        
        # Rule4: DEMA and MA Crossover
        (Rule4, {'dema1_period': [5, 8, 10, 15, 20], 'ma2_period': [30, 40, 50, 60, 100]}),
        
        # Rule5: DEMA and DEMA Crossover
        (Rule5, {'dema1_period': [5, 8, 10, 15, 20], 'dema2_period': [30, 40, 50, 60, 100]}),
        
        # Rule6: TEMA and MA Crossover
        (Rule6, {'tema1_period': [5, 8, 10, 15, 20], 'ma2_period': [30, 40, 50, 60, 100]}),
        
        # Rule7: Stochastic Oscillator
        (Rule7, {'stoch1_period': [5, 8, 10, 14, 20], 'stochma2_period': [3, 5, 7, 9]}),
        
        # Rule8: Vortex Indicator
        (Rule8, {'vortex1_period': [7, 10, 14, 20, 30], 'vortex2_period': [7, 10, 14, 20, 30]}),
        
        # Rule9: Ichimoku Cloud
        (Rule9, {'p1': [7, 9, 11, 13], 'p2': [22, 26, 30, 44, 52]}),
        
        # Rule10: RSI Overbought/Oversold
        (Rule10, {'rsi1_period': [7, 9, 11, 14, 21], 'c2_threshold': [20, 30, 40, 50, 60, 70]}),
        
        # Rule11: CCI Overbought/Oversold
        (Rule11, {'cci1_period': [7, 10, 14, 20, 30], 'c2_threshold': [70, 90, 110, 130, 150]}),
        
        # Rule12: RSI-based strategy
        (Rule12, {'rsi_period': [7, 10, 14, 21], 'overbought': [65, 70, 75, 80], 'oversold': [20, 25, 30, 35]}),
        
        # Rule13: Stochastic Oscillator strategy
        (Rule13, {'stoch_period': [7, 10, 14, 21], 'stoch_d_period': [3, 5, 7], 'overbought': [70, 75, 80, 85], 'oversold': [15, 20, 25, 30]}),
        
        # Rule14: ATR Trailing Stop
        (Rule14, {'atr_period': [7, 10, 14, 21], 'atr_multiplier': [1.5, 2.0, 2.5, 3.0, 3.5]}),
        
        # Rule15: Bollinger Bands strategy
        (Rule15, {'bb_period': [15, 20, 25, 30], 'bb_std_dev_multiplier': [1.5, 2.0, 2.5, 3.0]})
    ]
    
    print("\n=== Training Individual Rules ===")
    from rule_system import EventDrivenRuleSystem
    
    rule_system = EventDrivenRuleSystem(rules_config=expanded_rules_config, top_n=15)
    rule_system.train_rules(data_handler)
    
    # Get the trained rule objects
    rule_objects = list(rule_system.trained_rule_objects.values())
    
    print("\nSelected Top Rules:")
    for i, rule in enumerate(rule_objects):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # Add the confidence-based rules
    confidence_rules = [
        MAConfidenceRule({'short_period': 5, 'long_period': 20}),
        MAConfidenceRule({'short_period': 10, 'long_period': 30}),
        MAConfidenceRule({'short_period': 20, 'long_period': 50}),
        RSIConfidenceRule({'period': 14, 'overbought': 70, 'oversold': 30}),
        RSIConfidenceRule({'period': 7, 'overbought': 80, 'oversold': 20}),
        VolatilityRule({'period': 20})
    ]
    
    # Combine standard rules with confidence rules
    all_rules = rule_objects + confidence_rules
    
    print(f"\nTotal rules in combined set: {len(all_rules)}")
    
    # Create enhanced wavelet regime detector
    regime_detector = EnhancedWaveletRegimeDetector(lookback_period=100, threshold=0.5)
    
    # Create regime-aware strategy factory
    strategy_factory = EnhancedRegimeStrategyFactory()
    
    # Create regime manager
    regime_manager = RegimeManager(
        regime_detector=regime_detector,
        strategy_factory=strategy_factory,
        rule_objects=all_rules,
        data_handler=data_handler
    )
    
    # Override the optimize_regime_strategies method to use our custom implementation
    # This is a bit of a hack, but it allows us to use Calmar ratio for optimization
    original_optimize = regime_manager.optimize_regime_strategies
    regime_manager.optimize_regime_strategies = lambda verbose=True: optimize_regime_specific_strategies(
        regime_manager, optimization_metric='calmar')
    
    # Optimize regime-specific strategies
    print("\n=== Optimizing Regime-Specific Strategies using Calmar Ratio ===")
    start_time = time.time()
    regime_manager.optimize_regime_strategies()
    print(f"\nOptimization completed in {time.time() - start_time:.2f} seconds")
    
    # Run backtest on training data first
    print("\n=== Backtesting on Training Data ===")
    training_backtester = Backtester(data_handler, regime_manager)
    train_results = training_backtester.run(use_test_data=False)
    
    # Calculate metrics
    train_sharpe = training_backtester.calculate_sharpe()
    train_calmar = training_backtester.calculate_calmar()
    train_win_rate = sum(1 for t in train_results['trades'] if t[5] > 0) / train_results['num_trades'] if train_results['num_trades'] > 0 else 0
    
    print(f"Training Results:")
    print(f"  Total Return: {train_results['total_percent_return']:.2f}%")
    print(f"  Number of Trades: {train_results['num_trades']}")
    print(f"  Sharpe Ratio: {train_sharpe:.4f}")
    print(f"  Calmar Ratio: {train_calmar:.4f}")
    print(f"  Win Rate: {train_win_rate:.2f}")
    
    # Plot training equity curve
    plot_equity_curve(train_results["trades"], "Enhanced Regime Strategy - Training")
    
    # Backtest on test data
    print("\n=== Backtesting on Out-of-Sample Data ===")
    test_backtester = Backtester(data_handler, regime_manager)
    test_results = test_backtester.run(use_test_data=True)
    
    # Calculate metrics
    test_sharpe = test_backtester.calculate_sharpe()
    test_calmar = test_backtester.calculate_calmar()
    test_win_rate = sum(1 for t in test_results['trades'] if t[5] > 0) / test_results['num_trades'] if test_results['num_trades'] > 0 else 0
    
    print(f"Out-of-Sample Results:")
    print(f"  Total Return: {test_results['total_percent_return']:.2f}%")
    print(f"  Number of Trades: {test_results['num_trades']}")
    print(f"  Sharpe Ratio: {test_sharpe:.4f}")
    print(f"  Calmar Ratio: {test_calmar:.4f}")
    print(f"  Win Rate: {test_win_rate:.2f}")
    
    # Plot test equity curve
    plot_equity_curve(test_results["trades"], "Enhanced Regime Strategy - Out-of-Sample")
    
    # Analyze performance by regime
    print("\n=== Performance by Market Regime ===")
    regime_performance = analyze_performance_by_regime(test_results['trades'], data_handler, regime_detector)
    
    # Display regime performance
    print(f"{'Regime':<15} {'Return':<10} {'Trades':<10} {'Win Rate':<10} {'Avg Return':<10}")
    print("-" * 55)
    
    for regime, metrics in regime_performance.items():
        print(f"{regime.name:<15} {metrics['total_return']:>8.2f}% {metrics['trade_count']:>10} {metrics['win_rate']:>9.2f} {metrics['avg_return']:>9.4f}")
    
    return {
        'train': train_results,
        'test': test_results,
        'regime_performance': regime_performance,
        'train_sharpe': train_sharpe,
        'train_calmar': train_calmar,
        'test_sharpe': test_sharpe,
        'test_calmar': test_calmar,
        'win_rate': test_win_rate
    }


def analyze_performance_by_regime(trades, data_handler, regime_detector):
    """
    Analyze trading performance broken down by market regime.
    
    Args:
        trades: List of trade tuples
        data_handler: Data handler object
        regime_detector: Regime detector object
        
    Returns:
        dict: Performance metrics by regime
    """
    # Map timestamps to regimes
    regime_map = {}
    data_handler.reset_test()
    regime_detector.reset()
    
    # Process test data to identify regimes
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
            
        regime = regime_detector.detect_regime(bar)
        regime_map[bar['timestamp']] = regime
    
    # Group trades by regime
    regime_trades = {}
    
    for trade in trades:
        entry_time = trade[0]
        
        # Convert timestamp to datetime if it's a string
        if isinstance(entry_time, str):
            entry_time = pd.to_datetime(entry_time)
            
        # Find the regime at entry time
        if entry_time in regime_map:
            regime = regime_map[entry_time]
        else:
            # Try to find the closest timestamp
            closest_time = min(regime_map.keys(), key=lambda x: abs(pd.to_datetime(x) - entry_time))
            regime = regime_map[closest_time]
        
        if regime not in regime_trades:
            regime_trades[regime] = []
            
        regime_trades[regime].append(trade)
    
    # Calculate performance metrics for each regime
    performance = {}
    
    for regime, trades in regime_trades.items():
        # Skip regimes with too few trades
        if len(trades) < 5:
            continue
            
        # Calculate metrics
        returns = [trade[5] for trade in trades]  # log returns
        win_count = sum(1 for r in returns if r > 0)
        
        total_return = (np.exp(sum(returns)) - 1) * 100  # percentage
        avg_return = np.mean(returns)
        win_rate = win_count / len(trades)
        
        performance[regime] = {
            'trade_count': len(trades),
            'win_count': win_count,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return
        }
    
    return performance


# Run the enhanced regime strategy
if __name__ == "__main__":
    results = run_enhanced_regime_strategy()
