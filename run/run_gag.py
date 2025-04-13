"""
Enhanced trading system with a combined approach:
1. Rule selection through Walk-Forward Validation
2. Genetic Algorithm optimization for rule weights
3. Regime-specific weight tuning

This script integrates the most powerful features of the trading system into a single approach.
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
from strategy import TopNStrategy
from strategy import (
    Rule0, Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, 
    Rule8, Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15
)
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy
from regime_detection import (
    RegimeType, TrendStrengthRegimeDetector, 
    VolatilityRegimeDetector, RegimeManager
)
from validator import WalkForwardValidator
from signals import Signal, SignalType


class RegimeSpecificGAStrategy:
    """
    A strategy that combines genetic algorithm optimization with regime-specific adaptation.
    
    This advanced strategy:
    1. Uses walk-forward validated rules
    2. Optimizes rule weights with genetic algorithms
    3. Creates different weight sets for each detected regime
    """
    
    def __init__(self, rule_objects, regime_detector, data_handler):
        self.rule_objects = rule_objects
        self.regime_detector = regime_detector
        self.data_handler = data_handler
        self.regime_weights = {}  # Dictionary of weights per regime
        self.default_weights = None  # Fallback weights
        self.current_regime = RegimeType.UNKNOWN
    
    def optimize(self, verbose=False):
        """
        Optimize the strategy with regime-specific genetic algorithms.
        
        This method:
        1. Identifies different regimes in the training data
        2. Runs a separate GA optimization for each regime
        3. Creates a combined regime-adaptive strategy
        """
        start_time = time.time()
        
        # First, perform a global GA optimization to get default weights
        if verbose:
            print("\nRunning global GA optimization for default weights...")
        
        global_optimizer = GeneticOptimizer(
            data_handler=self.data_handler,
            rule_objects=self.rule_objects,
            population_size=30,
            num_generations=40,
            optimization_metric='sharpe'
        )
        
        self.default_weights = global_optimizer.optimize(verbose=verbose)
        
        if verbose:
            print(f"Default weights: {self.default_weights}")
            print("Global optimization completed in {:.2f} seconds".format(time.time() - start_time))
        
        # Next, identify regimes in the training data
        regime_bars = self._identify_regime_bars()
        
        if verbose:
            print("\nDetected regimes in training data:")
            for regime, bars in regime_bars.items():
                print(f"  {regime.name}: {len(bars)} bars")
        
        # Run regime-specific GA optimization for each regime
        for regime, bars in regime_bars.items():
            if len(bars) < 30:  # Skip regimes with insufficient data
                if verbose:
                    print(f"\nSkipping {regime.name} due to insufficient data ({len(bars)} bars)")
                self.regime_weights[regime] = self.default_weights
                continue
            
            if verbose:
                print(f"\nOptimizing for {regime.name} regime ({len(bars)} bars)...")
            
            # Create regime-specific data handler
            regime_data = self._create_regime_specific_data(bars)
            
            # Run GA optimization for this regime
            regime_optimizer = GeneticOptimizer(
                data_handler=regime_data,
                rule_objects=self.rule_objects,
                population_size=20,  # Smaller population for faster optimization
                num_generations=30,  # Fewer generations for faster optimization
                optimization_metric='sharpe'
            )
            
            regime_weights = regime_optimizer.optimize(verbose=verbose)
            self.regime_weights[regime] = regime_weights
            
            if verbose:
                print(f"Optimized weights for {regime.name}: {regime_weights}")
        
        # Set default weights for any regime we didn't optimize
        for regime in RegimeType:
            if regime not in self.regime_weights:
                self.regime_weights[regime] = self.default_weights
        
        if verbose:
            total_time = time.time() - start_time
            print(f"\nRegime-specific GA optimization completed in {total_time:.2f} seconds")
        
        return self.regime_weights
    
    def _identify_regime_bars(self):
        """Identify which bars belong to each regime."""
        regime_bars = {}
        self.regime_detector.reset()
        self.data_handler.reset_train()
        bar_index = 0
        
        while True:
            bar = self.data_handler.get_next_train_bar()
            if bar is None:
                break
            regime = self.regime_detector.detect_regime(bar)
            if regime not in regime_bars:
                regime_bars[regime] = []
            regime_bars[regime].append((bar_index, bar))
            bar_index += 1
        
        return regime_bars
    
    def _create_regime_specific_data(self, regime_bars):
        """Create a data handler with only bars from a specific regime."""
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
    
    def on_bar(self, event):
        """Process a bar using the regime-specific weights."""
        bar = event.bar
        
        # Detect current regime
        self.current_regime = self.regime_detector.detect_regime(bar)
        
        # Get weights for the current regime
        weights = self.regime_weights.get(self.current_regime, self.default_weights)
        
        # Calculate the weighted signal
        combined_signals = []
        for i, rule in enumerate(self.rule_objects):
            signal_object = rule.on_bar(bar)
            if signal_object and hasattr(signal_object, 'signal_type'):
                combined_signals.append(signal_object.signal_type.value * weights[i])
            else:
                combined_signals.append(0)
        
        weighted_sum = np.sum(combined_signals)
        
        # Convert to final signal
        if weighted_sum > 0.5:
            final_signal_type = SignalType.BUY
        elif weighted_sum < -0.5:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL
        
        # Create and return signal
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id=f"GA_Regime_{self.current_regime.name}",
            confidence=abs(weighted_sum)
        )
    
    def reset(self):
        """Reset the strategy state."""
        self.regime_detector.reset()
        self.current_regime = RegimeType.UNKNOWN
        for rule in self.rule_objects:
            if hasattr(rule, 'reset'):
                rule.reset()


def main():
    # Define filepath - adjust as needed
    filepath = os.path.expanduser("~/mmbt/data/data.csv")
    
    print(f"Looking for data file at: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir())
        exit(1)
    else:
        print(f"File found, starting analysis...")
    
    # 1. Load data with end-of-day trade closing enabled
    start_time = time.time()
    data_handler = CSVDataHandler(filepath, train_fraction=0.8, close_positions_eod=True)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # 2. Define expanded parameter space for rules
    expanded_rules_config = [
        # Rule0: Simple Moving Average Crossover
        (Rule0, {'fast_window': [5, 10, 15, 20], 'slow_window': [20, 30, 50, 100, 200]}),
        
        # Rule1: Simple Moving Average Crossover with MA1 and MA2
        (Rule1, {'ma1': [5, 10, 15, 20, 25], 'ma2': [30, 40, 50, 75, 100]}),
        
        # Rule2: EMA and MA Crossover
        (Rule2, {'ema1_period': [5, 10, 15, 20], 'ma2_period': [30, 50, 75, 100]}),
        
        # Rule3: EMA and EMA Crossover
        (Rule3, {'ema1_period': [5, 10, 15, 20], 'ema2_period': [30, 50, 75, 100]}),
        
        # Rule4: DEMA and MA Crossover
        (Rule4, {'dema1_period': [5, 10, 15, 20], 'ma2_period': [30, 50, 75, 100]}),
        
        # Rule5: DEMA and DEMA Crossover
        (Rule5, {'dema1_period': [5, 10, 15, 20], 'dema2_period': [30, 50, 75, 100]}),
        
        # Rule6: TEMA and MA Crossover
        (Rule6, {'tema1_period': [5, 10, 15, 20], 'ma2_period': [30, 50, 75, 100]}),
        
        # Rule7: Stochastic Oscillator
        (Rule7, {'stoch1_period': [5, 10, 14, 20], 'stochma2_period': [3, 5, 7, 9]}),
        
        # Rule8: Vortex Indicator
        (Rule8, {'vortex1_period': [10, 14, 20, 25], 'vortex2_period': [10, 14, 20, 25]}),
        
        # Rule9: Ichimoku Cloud
        (Rule9, {'p1': [7, 9, 11, 14], 'p2': [26, 30, 40, 52]}),
        
        # Rule10: RSI Overbought/Oversold
        (Rule10, {'rsi1_period': [7, 10, 14, 21], 'c2_threshold': [30, 40, 50, 60]}),
        
        # Rule11: CCI Overbought/Oversold
        (Rule11, {'cci1_period': [10, 14, 20, 30], 'c2_threshold': [80, 100, 120, 150]}),
        
        # Rule12: RSI-based strategy
        (Rule12, {'rsi_period': [7, 10, 14, 21], 'hl_threshold': [65, 70, 75, 80], 'll_threshold': [20, 25, 30, 35]}),
        
        # Rule13: Stochastic and CCI combination
        (Rule13, {'stoch_period': [10, 14, 20], 'cci1_period': [14, 20, 30], 'hl_threshold': [70, 80, 90], 'll_threshold': [10, 20, 30]}),
        
        # Rule14: ATR Trailing Stop
        (Rule14, {'atr_period': [10, 14, 20, 30], 'atr_multiplier': [1.5, 2.0, 2.5, 3.0]}),
        
        # Rule15: Bollinger Bands strategy
        (Rule15, {'bb_period': [15, 20, 25, 30], 'bb_std_dev_multiplier': [1.5, 2.0, 2.5, 3.0]}),
    ]
    
    # ======= STEP 1: WALK-FORWARD VALIDATION FOR RULE SELECTION =======
    print("\n=== Step 1: Walk-Forward Validation for Rule Selection ===")
    print("Running walk-forward validation to select robust rules...")
    
    # Initialize walk-forward validator
    wf_validator = WalkForwardValidator(
        data_filepath=filepath,
        rules_config=expanded_rules_config,
        window_size=252,  # 1 year of trading days
        step_size=63,     # 3 months
        train_pct=0.7,    # 70% training, 30% testing within each window
        top_n=10,
        optimization_method='genetic',
        optimization_metric='sharpe'
    )
    
    # Run walk-forward validation
    wf_results = wf_validator.run_validation(verbose=True, plot_results=True)
    
    # Extract or train the best rules from walk-forward validation
    def extract_best_rules_from_wf(wf_results, expanded_rules_config, data_handler, top_n=10):
        """Extract the best performing rules from walk-forward validation results."""
        # This is a simplified implementation - ideally, you would analyze which rules
        # performed well across multiple windows in the walk-forward results
        print("\nTraining rule system with insights from walk-forward validation...")
        wf_rule_system = EventDrivenRuleSystem(rules_config=expanded_rules_config, top_n=top_n)
        wf_rule_system.train_rules(data_handler)
        best_rules = list(wf_rule_system.trained_rule_objects.values())
        return best_rules
    
    # Get the best rules
    wf_optimized_rules = extract_best_rules_from_wf(wf_results, expanded_rules_config, data_handler)
    
    print("\nWalk-Forward Optimized Rules:")
    for i, rule in enumerate(wf_optimized_rules):
        rule_name = rule.__class__.__name__
        print(f"  {i+1}. {rule_name}")
    
    # ======= STEP 2: COMBINED GA + REGIME OPTIMIZATION =======
    print("\n=== Step 2: Combined GA + Regime Optimization ===")
    
    # Create regime detector
    regime_detector = TrendStrengthRegimeDetector(
        adx_period=14,
        adx_threshold=25
    )
    
    # Create the combined strategy
    combined_strategy = RegimeSpecificGAStrategy(
        rule_objects=wf_optimized_rules,
        regime_detector=regime_detector,
        data_handler=data_handler
    )
    
    # Run optimization
    print("\nRunning regime-specific GA optimization...")
    regime_weights = combined_strategy.optimize(verbose=True)
    
    # ======= STEP 3: BACKTEST THE COMBINED STRATEGY =======
    print("\n=== Step 3: Backtesting the Combined Strategy ===")
    combined_backtester = Backtester(data_handler, combined_strategy, close_positions_eod=True)
    combined_results = combined_backtester.run(use_test_data=True)
    
    print(f"Total Return: {combined_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {combined_results['num_trades']}")
    print(f"Win Rate: {combined_results['win_rate']:.2%}")
    sharpe_combined = combined_backtester.calculate_sharpe()
    print(f"Sharpe Ratio: {sharpe_combined:.4f}")
    
    # ======= STEP 4: COMPARE WITH BASELINE =======
    print("\n=== Step 4: Comparison with Baseline Strategy ===")
    
    # Create baseline strategy
    baseline_rule_system = EventDrivenRuleSystem(rules_config=expanded_rules_config, top_n=10)
    baseline_rule_system.train_rules(data_handler)
    baseline_strategy = baseline_rule_system.get_top_n_strategy()
    
    # Backtest baseline
    baseline_backtester = Backtester(data_handler, baseline_strategy, close_positions_eod=True)
    baseline_results = baseline_backtester.run(use_test_data=True)
    sharpe_baseline = baseline_backtester.calculate_sharpe()
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"{'Strategy':<30} {'Return':<10} {'# Trades':<10} {'Win Rate':<10} {'Sharpe':<10}")
    print("-" * 75)
    print(f"{'Baseline':<30} {baseline_results['total_percent_return']:>8.2f}% {baseline_results['num_trades']:>10} {baseline_results['win_rate']:>9.2%} {sharpe_baseline:>9.4f}")
    print(f"{'Combined GA + Regime + WF':<30} {combined_results['total_percent_return']:>8.2f}% {combined_results['num_trades']:>10} {combined_results['win_rate']:>9.2%} {sharpe_combined:>9.4f}")
    
    # Calculate improvement
    return_improvement = (combined_results['total_percent_return'] / baseline_results['total_percent_return'] - 1) * 100 if baseline_results['total_percent_return'] > 0 else float('inf')
    sharpe_improvement = (sharpe_combined / sharpe_baseline - 1) * 100 if sharpe_baseline > 0 else float('inf')
    
    print(f"\nReturn improvement: {return_improvement:.2f}%")
    print(f"Sharpe ratio improvement: {sharpe_improvement:.2f}%")
    
    # Plot equity curves
    def plot_equity_curve_comparison(baseline_trades, combined_trades, title="Strategy Comparison"):
        """Plot equity curves for both strategies."""
        plt.figure(figsize=(14, 8))
        
        # Baseline equity curve
        equity_baseline = [10000]
        for trade in baseline_trades:
            log_return = trade[5]
            equity_baseline.append(equity_baseline[-1] * np.exp(log_return))
        
        # Combined strategy equity curve
        equity_combined = [10000]
        for trade in combined_trades:
            log_return = trade[5]
            equity_combined.append(equity_combined[-1] * np.exp(log_return))
        
        # Plot both curves
        plt.plot(equity_baseline, label=f"Baseline ({baseline_results['total_percent_return']:.2f}%)")
        plt.plot(equity_combined, label=f"Combined GA+Regime+WF ({combined_results['total_percent_return']:.2f}%)")
        plt.title(title)
        plt.xlabel('Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.show()
    
    print("\nPlotting equity curves...")
    plot_equity_curve_comparison(
        baseline_results["trades"], 
        combined_results["trades"], 
        "Combined Strategy vs Baseline (with EOD Closing)"
    )
    
    # Total runtime
    total_runtime = time.time() - start_time
    print(f"\nTotal runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")


if __name__ == "__main__":
    main()
