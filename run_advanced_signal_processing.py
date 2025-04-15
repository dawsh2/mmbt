"""
Enhanced Trading System Test Script

Tests new signal processing approaches including Wavelet-based regime detection
and Kalman filter-based rules in comparison with traditional methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time
import pywt  # You'll need to install PyWavelets: pip install PyWavelets
from sklearn.feature_selection import mutual_info_regression


from data_handler import CSVDataHandler
from backtester import Backtester, BarEvent
from signals import Signal, SignalType
from rule_system import EventDrivenRuleSystem
from strategy import WeightedRuleStrategy
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5
from regime_detection import RegimeDetector, RegimeType, RegimeManager
from strategy import WeightedRuleStrategyFactory



# New Rule Implementation
class KalmanFilterRule:
    def __init__(self, params):
        self.measurement_noise = params.get('measurement_noise', 0.1)
        self.process_noise = params.get('process_noise', 0.01)
        self.threshold = params.get('threshold', 0.005)  # Much lower threshold
        self.state = None
        self.covariance = None
        self.close_history = deque(maxlen=200)
        self.rule_id = "KalmanFilterRule"
        
    def on_bar(self, bar):
        close = bar['Close']
        self.close_history.append(close)
        
        # Simple Kalman filter for trend estimation
        if self.state is None:
            self.state = close
            self.covariance = 1.0
        else:
            # Prediction step
            prediction = self.state
            prediction_covariance = self.covariance + self.process_noise
            
            # Update step
            kalman_gain = prediction_covariance / (prediction_covariance + self.measurement_noise)
            self.state = prediction + kalman_gain * (close - prediction)
            self.covariance = (1 - kalman_gain) * prediction_covariance
        
        # Generate signal based on relationship between price and filtered state
        # Use a much lower threshold - 0.5% instead of 2%
        if close > self.state * (1 + self.threshold):
            signal_type = SignalType.SELL  # Potential mean reversion
        elif close < self.state * (1 - self.threshold):
            signal_type = SignalType.BUY   # Potential mean reversion
        else:
            signal_type = SignalType.NEUTRAL
        
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=close,
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={"filtered_state": self.state}
        )

    def reset(self):
        self.state = None
        self.covariance = None
        self.close_history = deque(maxlen=200)


# New Regime Detector

class WaveletRegimeDetector(RegimeDetector):
    def __init__(self, lookback_period=100, threshold=0.5):
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.price_history = deque(maxlen=lookback_period)
        self.current_regime = RegimeType.UNKNOWN
        
    def detect_regime(self, bar):
        self.price_history.append(bar['Close'])
        
        if len(self.price_history) >= self.lookback_period:
            try:
                # Convert deque to list for wavelet transform
                prices = list(self.price_history)
                
                # Calculate returns
                returns = np.diff(prices) / prices[:-1]
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(returns, 'db1', level=2)
                
                # Calculate energy in different frequency bands
                detail_coeffs = coeffs[1]  # Detail coefficients (high frequency)
                approx_coeffs = coeffs[0]  # Approximation coefficients (low frequency)
                
                # Calculate energy (variance) in each band
                detail_energy = np.var(detail_coeffs) if len(detail_coeffs) > 1 else 0
                approx_energy = np.var(approx_coeffs) if len(approx_coeffs) > 1 else 0
                
                # Calculate ratio of high-frequency to low-frequency energy
                energy_ratio = detail_energy / approx_energy if approx_energy > 0 else 0
                
                # Calculate trend direction - use average of recent returns
                trend = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                
                # Determine regime
                if energy_ratio > self.threshold * 2:  # Much more volatile threshold
                    self.current_regime = RegimeType.VOLATILE
                elif trend > 0.001:  # Positive trend
                    self.current_regime = RegimeType.TRENDING_UP
                elif trend < -0.001:  # Negative trend
                    self.current_regime = RegimeType.TRENDING_DOWN
                else:  # No clear trend
                    self.current_regime = RegimeType.RANGE_BOUND
                    
            except Exception as e:
                print(f"Error in wavelet calculation: {str(e)}")
                # Fall back to a simple regime determination
                returns = np.diff(list(self.price_history)) / list(self.price_history)[:-1]
                volatility = np.std(returns)
                avg_return = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                
                if volatility > np.mean(np.abs(returns)) * 1.5:
                    self.current_regime = RegimeType.VOLATILE
                elif avg_return > 0.001:
                    self.current_regime = RegimeType.TRENDING_UP
                elif avg_return < -0.001:
                    self.current_regime = RegimeType.TRENDING_DOWN
                else:
                    self.current_regime = RegimeType.RANGE_BOUND
        
        return self.current_regime

# Enhanced weight optimization function
def optimize_rule_weights(rule_objects, data_handler, use_information_theory=True):
    """
    Optimize rule weights using information theory metrics.
    
    Args:
        rule_objects: List of rule objects
        data_handler: The data handler for accessing market data
        use_information_theory: Whether to use information theory for weighting
        
    Returns:
        numpy.ndarray: Optimized weights
    """
    if not use_information_theory:
        # Fall back to equal weights
        return np.ones(len(rule_objects)) / len(rule_objects)
    
    # Collect signals from all rules
    rule_signals = []
    data_handler.reset_train()
    
    # Pre-collect all bars for efficiency
    all_bars = []
    forward_returns = []
    while True:
        bar = data_handler.get_next_train_bar()
        if bar is None:
            break
        all_bars.append(bar)
        
        # Calculate forward returns (for next bar) for prediction target
        if len(all_bars) > 1:
            prev_close = all_bars[-2]['Close']
            curr_close = all_bars[-1]['Close']
            forward_returns.append((curr_close / prev_close) - 1)
    
    # We have one less return than bars
    forward_returns.append(0)  # Add placeholder for last bar
    
    # Generate signals for each rule
    for rule in rule_objects:
        rule.reset()
        signals = []
        
        for bar in all_bars:
            signal = rule.on_bar(bar)
            if signal and hasattr(signal, 'signal_type'):
                signals.append(signal.signal_type.value)  # Convert to numeric (-1, 0, 1)
            else:
                signals.append(0)
        
        rule_signals.append(signals)
    
    # Calculate mutual information between each rule's signals and forward returns
    information_scores = []
    
    for signals in rule_signals:
        # Skip if we don't have enough data
        if len(signals) < 10 or len(forward_returns) < 10:
            information_scores.append(0.001)  # Default small value
            continue
            
        # Ensure signals and returns are same length
        min_len = min(len(signals), len(forward_returns))
        signals = signals[:min_len]
        returns = forward_returns[:min_len]
        
        try:
            # Calculate mutual information
            mi_score = mutual_info_regression(
                np.array(signals).reshape(-1, 1),
                returns
            )[0]
            
            # Ensure positive value
            information_scores.append(max(0.001, mi_score))
        except Exception as e:
            print(f"Error calculating mutual information: {str(e)}")
            information_scores.append(0.001)  # Default small value
    
    # Normalize to create weights
    information_scores = np.array(information_scores)
    if sum(information_scores) > 0:
        weights = information_scores / sum(information_scores)
    else:
        # Fall back to equal weights if information content is too low
        weights = np.ones(len(rule_objects)) / len(rule_objects)
    
    return weights

# Main test function
def run_advanced_signal_processing_test():
    """Test advanced signal processing techniques."""
    print("=== Advanced Signal Processing Test ===")
    
    # Load data
    data_file = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = CSVDataHandler(data_file, train_fraction=0.8)
    
    # Create rules including our new Kalman Filter rule
    standard_rules = [
        Rule0({'fast_window': 10, 'slow_window': 30}),
        Rule1({'ma1': 10, 'ma2': 50}),
        Rule2({'ema1_period': 12, 'ma2_period': 26}),
        Rule3({'ema1_period': 10, 'ema2_period': 50})
    ]
    
    # Create Kalman Filter rule with different parameter sets
    kalman_rules = [
        KalmanFilterRule({'measurement_noise': 0.1, 'process_noise': 0.01}),
        KalmanFilterRule({'measurement_noise': 0.5, 'process_noise': 0.05}),
        KalmanFilterRule({'measurement_noise': 0.05, 'process_noise': 0.1})
    ]
    
    # Combine all rules
    all_rules = standard_rules + kalman_rules
    
    # Test different strategy approaches
    strategies = {
        "Equal Weights": WeightedRuleStrategy(
            rule_objects=all_rules,
            weights=np.ones(len(all_rules)) / len(all_rules)
        ),
        "Information Theory Weights": WeightedRuleStrategy(
            rule_objects=all_rules,
            weights=optimize_rule_weights(all_rules, data_handler, use_information_theory=True)
        ),
        "Standard Rules Only": WeightedRuleStrategy(
            rule_objects=standard_rules,
            weights=np.ones(len(standard_rules)) / len(standard_rules)
        ),
        "Kalman Rules Only": WeightedRuleStrategy(
            rule_objects=kalman_rules,
            weights=np.ones(len(kalman_rules)) / len(kalman_rules)
        )
    }
    
    # Test each strategy
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name} strategy...")
        
        # Run backtest
        backtester = Backtester(data_handler, strategy)
        result = backtester.run(use_test_data=True)
        
        # Calculate metrics
        sharpe = backtester.calculate_sharpe()
        
        print(f"  Total Return: {result['total_percent_return']:.2f}%")
        print(f"  Number of Trades: {result['num_trades']}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        
        # Store results
        results[name] = {
            'return': result['total_percent_return'],
            'trades': result['num_trades'],
            'sharpe': sharpe,
            'log_return': result['total_log_return'],
            'win_rate': sum(1 for t in result['trades'] if t[5] > 0) / result['num_trades'] if result['num_trades'] > 0 else 0
        }
    
    # Compare strategies
    print("\nStrategy Comparison Summary:")
    print(f"{'Strategy':<25} {'Return':<10} {'Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 65)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['return']:>8.2f}% {metrics['trades']:>10} {metrics['sharpe']:>9.4f} {metrics['win_rate']:>9.2f}")
    
    # Test regime detection
    print("\n=== Testing Wavelet Regime Detection ===")
    
    # Create regime detectors
    detectors = {
        "Wavelet": WaveletRegimeDetector(lookback_period=100, threshold=0.5)
    }
    
    # Analyze regime distribution
    for name, detector in detectors.items():
        print(f"\nAnalyzing regimes with {name} detector...")
        detector.reset()
        data_handler.reset_test()
        
        regime_counts = {
            RegimeType.TRENDING_UP: 0,
            RegimeType.TRENDING_DOWN: 0,
            RegimeType.RANGE_BOUND: 0,
            RegimeType.VOLATILE: 0,
            RegimeType.UNKNOWN: 0
        }
        
        while True:
            bar = data_handler.get_next_test_bar()
            if bar is None:
                break
            
            regime = detector.detect_regime(bar)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Print regime distribution
        total_bars = sum(regime_counts.values())
        print(f"Regime distribution across {total_bars} bars:")
        
        for regime, count in regime_counts.items():
            percentage = (count / total_bars) * 100 if total_bars > 0 else 0
            print(f"  {regime.name}: {count} bars ({percentage:.1f}%)")
    
    # Test regime-based strategy using Wavelet detector
    print("\n=== Testing Regime-Based Strategy with Wavelet Detector ===")
    
    # Create regime manager with wavelet detector
    strategy_factory = WeightedRuleStrategyFactory()
    regime_detector = WaveletRegimeDetector(lookback_period=100, threshold=0.5)
    regime_manager = RegimeManager(
        regime_detector=regime_detector,
        strategy_factory=strategy_factory,  # Add this parameter
        rule_objects=all_rules,
        data_handler=data_handler
    )
    
    # Optimize regime-specific strategies
    print("Optimizing regime-specific strategies...")
    regime_manager.optimize_regime_strategies(verbose=True)
    
    # Backtest the regime-based strategy
    regime_backtester = Backtester(data_handler, regime_manager)
    regime_results = regime_backtester.run(use_test_data=True)
    
    regime_sharpe = regime_backtester.calculate_sharpe()
    regime_win_rate = sum(1 for t in regime_results['trades'] if t[5] > 0) / regime_results['num_trades'] if regime_results['num_trades'] > 0 else 0
    
    print("\nRegime-Based Strategy Results:")
    print(f"  Total Return: {regime_results['total_percent_return']:.2f}%")
    print(f"  Number of Trades: {regime_results['num_trades']}")
    print(f"  Sharpe Ratio: {regime_sharpe:.4f}")
    print(f"  Win Rate: {regime_win_rate:.2f}")
    
    # Add regime results to comparison
    results["Wavelet Regime-Based"] = {
        'return': regime_results['total_percent_return'],
        'trades': regime_results['num_trades'],
        'sharpe': regime_sharpe,
        'win_rate': regime_win_rate
    }
    
    # Plot equity curves
    plot_equity_curves(data_handler, strategies, regime_manager)
    
    return results

def plot_equity_curves(data_handler, strategies, regime_manager=None):
    """Plot equity curves for different strategies."""
    plt.figure(figsize=(12, 8))
    
    # Create strategies dictionary including regime manager
    all_strategies = dict(strategies)
    if regime_manager:
        all_strategies["Wavelet Regime-Based"] = regime_manager
    
    # Test each strategy and plot equity curve
    for name, strategy in all_strategies.items():
        # Reset data handler
        data_handler.reset_test()
        
        # Run backtest
        backtester = Backtester(data_handler, strategy)
        result = backtester.run(use_test_data=True)
        
        # Calculate equity curve
        equity = [10000]  # Start with $10,000
        for trade in result['trades']:
            equity.append(equity[-1] * np.exp(trade[5]))
        
        # Plot
        plt.plot(equity, label=f"{name} ({result['total_percent_return']:.1f}%)")
    
    plt.title("Strategy Comparison - Equity Curves")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("strategy_comparison.png")
    plt.show()

# Run the test
if __name__ == "__main__":
    run_advanced_signal_processing_test()
