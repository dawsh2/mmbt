"""
Enhanced Trading System with Confidence Metrics and Information Theory

This script implements an advanced trading system that combines:
1. Signal strength confidence metrics
2. Regime-based confidence adjustments
3. Information theory for optimal rule weighting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import time
import pywt
from sklearn.feature_selection import mutual_info_regression

from data_handler import CSVDataHandler
from backtester import Backtester, BarEvent
from signals import Signal, SignalType
from strategy import Rule0, Rule1, Rule2, Rule3, Rule4, Rule5
from regime_detection import RegimeDetector, RegimeType, RegimeManager

# ====== Enhanced Rules with Confidence Metrics ======

class ConfidenceRule:
    """Base class for rules that provide confidence metrics with their signals."""
    
    def __init__(self, params):
        self.params = params
        self.rule_id = "ConfidenceRule"
        
    def calculate_confidence(self, *args, **kwargs):
        """Calculate confidence of the signal based on market conditions."""
        # Default implementation returns constant confidence
        return 1.0

    def on_bar(self, event):
        # Check if we received a BarEvent object or a regular bar dictionary
        if hasattr(event, 'bar'):
            bar = event.bar  # Extract the actual bar dictionary from the event
        else:
            bar = event  # Already a dictionary

        # Now we can use bar as a dictionary
        signal_type, metadata = self._calculate_signal(bar)

        return Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )


    
    def _calculate_signal(self, bar):
        """Calculate the signal type for the given bar."""
        raise NotImplementedError("Subclasses must implement _calculate_signal")
    
    def reset(self):
        """Reset the rule state."""
        pass


class MAConfidenceRule(ConfidenceRule):
    """Moving average rule with strength-based confidence."""
    
    def __init__(self, params):
        super().__init__(params)
        self.short_period = params.get('short_period', 10)
        self.long_period = params.get('long_period', 30)
        self.price_history = deque(maxlen=max(self.short_period, self.long_period) + 10)
        self.rule_id = "MAConfidenceRule"
    
    def _calculate_signal(self, bar):
        close = bar['Close']
        self.price_history.append(close)
        
        if len(self.price_history) < self.long_period:
            return SignalType.NEUTRAL, {"ma_short": None, "ma_long": None}
        
        # Calculate moving averages
        ma_short = sum(list(self.price_history)[-self.short_period:]) / self.short_period
        ma_long = sum(list(self.price_history)[-self.long_period:]) / self.long_period
        
        # Generate signal
        if ma_short > ma_long:
            signal_type = SignalType.BUY
        elif ma_short < ma_long:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
            
        return signal_type, {"ma_short": ma_short, "ma_long": ma_long}
    
    def calculate_confidence(self, bar, signal_type, metadata):
        """Calculate confidence based on distance between moving averages."""
        if signal_type == SignalType.NEUTRAL or metadata["ma_short"] is None:
            return 0.0
            
        # Calculate percentage distance between MAs
        ma_short = metadata["ma_short"]
        ma_long = metadata["ma_long"]
        distance = abs(ma_short - ma_long) / ma_long
        
        # Convert to confidence score (0.5 to 1.0)
        # Small differences get 0.5, larger differences get up to 1.0
        confidence = 0.5 + min(0.5, distance / 0.01)  # Scale based on 1% difference
        
        return confidence
    
    def reset(self):
        self.price_history.clear()


class RSIConfidenceRule(ConfidenceRule):
    """RSI rule with extremity-based confidence."""
    
    def __init__(self, params):
        super().__init__(params)
        self.period = params.get('period', 14)
        self.overbought = params.get('overbought', 70)
        self.oversold = params.get('oversold', 30)
        self.price_history = deque(maxlen=self.period + 10)
        self.rule_id = "RSIConfidenceRule"
    
    def _calculate_signal(self, bar):
        close = bar['Close']
        self.price_history.append(close)
        
        if len(self.price_history) < self.period + 1:
            return SignalType.NEUTRAL, {"rsi": None}
        
        # Calculate RSI
        prices = list(self.price_history)
        deltas = np.diff(prices)
        gains = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Generate signal
        if rsi > self.overbought:
            signal_type = SignalType.SELL
        elif rsi < self.oversold:
            signal_type = SignalType.BUY
        else:
            signal_type = SignalType.NEUTRAL
            
        return signal_type, {"rsi": rsi}
    
    def calculate_confidence(self, bar, signal_type, metadata):
        """Calculate confidence based on RSI extremity."""
        if signal_type == SignalType.NEUTRAL or metadata["rsi"] is None:
            return 0.0
            
        rsi = metadata["rsi"]
        
        # More extreme RSI values get higher confidence
        if signal_type == SignalType.BUY:
            # RSI near 0 gets highest confidence, near oversold threshold gets lowest
            confidence = 0.5 + 0.5 * (1 - rsi / self.oversold)
        else:  # SELL
            # RSI near 100 gets highest confidence, near overbought threshold gets lowest
            confidence = 0.5 + 0.5 * ((rsi - self.overbought) / (100 - self.overbought))
            
        return max(0.5, min(1.0, confidence))  # Clamp between 0.5 and 1.0
    
    def reset(self):
        self.price_history.clear()


class VolatilityRule(ConfidenceRule):
    """Volatility-based rule with confidence."""
    
    def __init__(self, params):
        super().__init__(params)
        self.period = params.get('period', 20)
        self.price_history = deque(maxlen=self.period + 10)
        self.rule_id = "VolatilityRule"
    
    def _calculate_signal(self, bar):
        close = bar['Close']
        self.price_history.append(close)
        
        if len(self.price_history) < self.period:
            return SignalType.NEUTRAL, {"volatility": None}
        
        # Calculate price changes
        prices = list(self.price_history)
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(returns)
        
        # Generate signal based on recent price vs moving average
        ma = sum(prices[-self.period:]) / self.period
        
        if close > ma and volatility < 0.01:  # Low volatility and above MA
            signal_type = SignalType.BUY
        elif close < ma and volatility < 0.01:  # Low volatility and below MA
            signal_type = SignalType.SELL
        elif volatility > 0.02:  # High volatility, stay neutral
            signal_type = SignalType.NEUTRAL
        else:
            signal_type = SignalType.NEUTRAL
            
        return signal_type, {"volatility": volatility, "ma": ma}
    
    def calculate_confidence(self, bar, signal_type, metadata):
        """Calculate confidence based on volatility level."""
        if signal_type == SignalType.NEUTRAL or metadata["volatility"] is None:
            return 0.0
            
        volatility = metadata["volatility"]
        
        # Lower volatility gives higher confidence for directional signals
        if signal_type in (SignalType.BUY, SignalType.SELL):
            # Scale from 0.5 (at 1% volatility) to 1.0 (at 0% volatility)
            confidence = 1.0 - min(0.5, volatility / 0.01)
        else:
            confidence = 0.5  # Default for neutral signals
            
        return confidence
    
    def reset(self):
        self.price_history.clear()


# ====== Enhanced Weighted Strategy with Confidence ======

class ConfidenceWeightedStrategy:
    """Trading strategy that weights rules by both fixed weights and signal confidence."""
    
    def __init__(self, rule_objects, weights, buy_threshold=0.3, sell_threshold=-0.3):
        self.rule_objects = rule_objects
        self.weights = np.array(weights)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_signal = None
    
    def on_bar(self, event):
        """Process a bar and generate a trading signal."""
        bar = event.bar
        
        # Collect signals from all rules
        signals = []
        for rule in self.rule_objects:
            signal = rule.on_bar(bar)
            if signal and hasattr(signal, 'signal_type'):
                signals.append(signal)
            
        if not signals:
            return Signal(
                timestamp=bar["timestamp"],
                signal_type=SignalType.NEUTRAL,
                price=bar["Close"],
                rule_id="ConfidenceWeightedStrategy",
                confidence=0.0
            )
        
        # Calculate weighted signal
        weighted_signals = []
        confidence_values = []
        
        for i, signal in enumerate(signals):
            # Get rule weight and signal confidence
            rule_weight = self.weights[i] if i < len(self.weights) else 0.0
            signal_value = signal.signal_type.value
            signal_confidence = getattr(signal, 'confidence', 1.0)
            
            # Combine weight and confidence
            weighted_signal = signal_value * rule_weight * signal_confidence
            weighted_signals.append(weighted_signal)
            confidence_values.append(signal_confidence)
        
        # Calculate final signal
        weighted_sum = sum(weighted_signals)
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        
        # Adjust thresholds based on average confidence
        adjusted_buy = self.buy_threshold * avg_confidence
        adjusted_sell = self.sell_threshold * avg_confidence
        
        # Determine final signal type
        if weighted_sum >= adjusted_buy:
            final_signal_type = SignalType.BUY
        elif weighted_sum <= adjusted_sell:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL
        
        # Create final signal
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id="ConfidenceWeightedStrategy",
            confidence=avg_confidence,
            metadata={"weighted_sum": weighted_sum, "rule_signals": [s.signal_type.value for s in signals]}
        )
        
        return self.last_signal
    
    def reset(self):
        """Reset the strategy and all rules."""
        for rule in self.rule_objects:
            if hasattr(rule, 'reset'):
                rule.reset()
        self.last_signal = None


# ====== Improved Wavelet Regime Detector ======

class EnhancedWaveletRegimeDetector(RegimeDetector):
    """Enhanced wavelet-based regime detector with confidence metrics."""
    
    def __init__(self, lookback_period=100, threshold=0.5):
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.price_history = deque(maxlen=lookback_period)
        self.regime_history = deque(maxlen=5)  # Store recent regimes for stability
        self.current_regime = RegimeType.UNKNOWN
        self.confidence = 0.0
    
    def detect_regime(self, bar):
        """Detect the current market regime with confidence."""
        self.price_history.append(bar['Close'])
        
        if len(self.price_history) >= self.lookback_period:
            try:
                # Convert price history to returns
                prices = list(self.price_history)
                returns = np.diff(prices) / prices[:-1]
                
                # Use wavelet decomposition to analyze returns at different scales
                coeffs = pywt.wavedec(returns, 'db1', level=2)
                
                # Analyze coefficients to determine regime
                detail_coeffs = coeffs[1]  # High frequency detail
                approx_coeffs = coeffs[0]  # Low frequency approximation
                
                # Calculate energy in each frequency band
                detail_energy = np.sum(detail_coeffs**2)
                approx_energy = np.sum(approx_coeffs**2)
                
                # Calculate volatility from returns
                volatility = np.std(returns)
                
                # Calculate trend strength
                trend = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                trend_strength = abs(trend) / volatility if volatility > 0 else 0
                
                # Determine regime with confidence
                if volatility > 0.015:  # High volatility threshold
                    regime = RegimeType.VOLATILE
                    conf = min(1.0, volatility / 0.02)  # Higher volatility = higher confidence
                elif trend_strength > 0.5:  # Strong trend
                    if trend > 0:
                        regime = RegimeType.TRENDING_UP
                    else:
                        regime = RegimeType.TRENDING_DOWN
                    conf = min(1.0, trend_strength / 1.0)  # Stronger trend = higher confidence
                else:  # No strong trend or volatility
                    regime = RegimeType.RANGE_BOUND
                    conf = 1.0 - min(0.5, trend_strength + volatility / 0.01)  # Lower vol & trend = higher conf
                
                # Update confidence
                self.confidence = conf
                
                # Use history for stability (avoid frequent regime changes)
                self.regime_history.append(regime)
                
                # Only change regime if we have consistent signals
                if len(self.regime_history) >= 3:
                    recent_regimes = list(self.regime_history)[-3:]
                    if all(r == recent_regimes[0] for r in recent_regimes):
                        self.current_regime = recent_regimes[0]
                    # Otherwise keep current regime for stability
            
            except Exception as e:
                print(f"Error in wavelet calculation: {str(e)}")
                # Fall back to a simple analysis
                returns = np.diff(list(self.price_history)) / list(self.price_history)[:-1]
                volatility = np.std(returns)
                trend = np.mean(returns[-20:]) if len(returns) >= 20 else 0
                
                if volatility > 0.015:
                    self.current_regime = RegimeType.VOLATILE
                elif trend > 0.001:
                    self.current_regime = RegimeType.TRENDING_UP
                elif trend < -0.001:
                    self.current_regime = RegimeType.TRENDING_DOWN
                else:
                    self.current_regime = RegimeType.RANGE_BOUND
                
                self.confidence = 0.6  # Lower confidence for fallback method
        
        return self.current_regime
    
    def get_confidence(self):
        """Get the confidence level of the current regime determination."""
        return self.confidence
    
    def reset(self):
        """Reset the detector state."""
        self.price_history.clear()
        self.regime_history.clear()
        self.current_regime = RegimeType.UNKNOWN
        self.confidence = 0.0


# ====== Regime-Aware Strategy Factory ======

class RegimeAwareStrategyFactory:
    """Creates strategies with awareness of current regime and confidence."""
    
    def create_strategy(self, regime, rule_objects, optimization_params):
        """Create a strategy for a specific regime with optimized parameters."""
        # If optimization_params is a list/array of weights, use them directly
        if isinstance(optimization_params, (list, np.ndarray)):
            weights = optimization_params
        else:
            # Otherwise assume it's a dictionary with weights
            weights = optimization_params.get('weights', np.ones(len(rule_objects)) / len(rule_objects))
        
        # Create confidence-weighted strategy
        return ConfidenceWeightedStrategy(
            rule_objects=rule_objects, 
            weights=weights,
            buy_threshold=0.3,  # Can be customized per regime
            sell_threshold=-0.3
        )
    
    def create_default_strategy(self, rule_objects):
        """Create a default strategy with equal weights."""
        weights = np.ones(len(rule_objects)) / len(rule_objects)
        return ConfidenceWeightedStrategy(
            rule_objects=rule_objects,
            weights=weights
        )


# ====== Information Theory Weight Optimizer ======

def optimize_with_information_theory(rule_objects, data_handler):
    """
    Optimize rule weights using information theory.
    
    Args:
        rule_objects: List of rule objects
        data_handler: Data handler object
        
    Returns:
        numpy.ndarray: Optimized weights
    """
    # Collect data for all rules
    data_handler.reset_train()
    
    # Pre-collect all bars for efficiency
    all_bars = []
    forward_returns = []
    
    while True:
        bar = data_handler.get_next_train_bar()
        if bar is None:
            break
        all_bars.append(bar)
        
        if len(all_bars) > 1:
            # Calculate forward return (next bar's return)
            prev_close = all_bars[-2]['Close']
            curr_close = all_bars[-1]['Close']
            forward_returns.append((curr_close / prev_close) - 1)
    
    # Add placeholder for last bar
    forward_returns.append(0)
    
    # Collect signals from each rule
    rule_signals = []
    
    for rule in rule_objects:
        rule.reset()
        signals = []
        
        for bar in all_bars:
            # Create event object for the rule
            event = BarEvent(bar)
            
            # Get signal from rule
            signal = rule.on_bar(event)
            
            # Extract signal value (-1, 0, 1)
            if signal and hasattr(signal, 'signal_type'):
                signals.append(signal.signal_type.value)
            else:
                signals.append(0)
        
        rule_signals.append(signals)
    
    # Calculate mutual information between each rule's signals and forward returns
    information_scores = []
    
    for signals in rule_signals:
        # Skip if not enough data
        if len(signals) < 10:
            information_scores.append(0.001)
            continue
        
        # Ensure signals and returns have same length
        min_len = min(len(signals), len(forward_returns))
        signals = signals[:min_len]
        returns = forward_returns[:min_len]
        
        try:
            # Calculate mutual information (how much each rule predicts returns)
            mi_score = mutual_info_regression(
                np.array(signals).reshape(-1, 1),
                returns
            )[0]
            
            # Ensure positive value for weighting
            information_scores.append(max(0.001, mi_score))
        
        except Exception as e:
            print(f"Error calculating mutual information: {str(e)}")
            information_scores.append(0.001)
    
    # Normalize scores to create weights
    total_info = sum(information_scores)
    if total_info > 0:
        weights = np.array(information_scores) / total_info
    else:
        # Fall back to equal weights
        weights = np.ones(len(rule_objects)) / len(rule_objects)
    
    return weights


# ====== Main Test Function ======

def run_confidence_based_system_test():
    """Run tests of the confidence-based trading system."""
    print("=== Confidence-Based Trading System Test ===")
    
    # Load data
    data_file = os.path.expanduser("~/mmbt/data/data.csv")
    data_handler = CSVDataHandler(data_file, train_fraction=0.8)
    
    # Create rules with confidence metrics
    ma_rules = [
        MAConfidenceRule({'short_period': 5, 'long_period': 20}),
        MAConfidenceRule({'short_period': 10, 'long_period': 30}),
        MAConfidenceRule({'short_period': 20, 'long_period': 50})
    ]
    
    rsi_rules = [
        RSIConfidenceRule({'period': 14, 'overbought': 70, 'oversold': 30}),
        RSIConfidenceRule({'period': 7, 'overbought': 80, 'oversold': 20})
    ]
    
    volatility_rules = [
        VolatilityRule({'period': 20})
    ]
    
    # Combine all rules
    all_rules = ma_rules + rsi_rules + volatility_rules
    
    # 1. Test Equal Weights strategy
    print("\n--- Testing Equal Weights Strategy ---")
    equal_weights = np.ones(len(all_rules)) / len(all_rules)
    
    equal_strategy = ConfidenceWeightedStrategy(
        rule_objects=all_rules,
        weights=equal_weights
    )
    
    equal_backtester = Backtester(data_handler, equal_strategy)
    equal_results = equal_backtester.run(use_test_data=True)
    
    equal_sharpe = equal_backtester.calculate_sharpe()
    equal_win_rate = sum(1 for t in equal_results['trades'] if t[5] > 0) / equal_results['num_trades'] if equal_results['num_trades'] > 0 else 0
    
    print(f"Total Return: {equal_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {equal_results['num_trades']}")
    print(f"Sharpe Ratio: {equal_sharpe:.4f}")
    print(f"Win Rate: {equal_win_rate:.2f}")
    
    # 2. Test Information Theory Weighted strategy
    print("\n--- Testing Information Theory Weighted Strategy ---")
    info_weights = optimize_with_information_theory(all_rules, data_handler)
    
    print("Information Theory Weights:")
    for i, weight in enumerate(info_weights):
        rule_name = all_rules[i].__class__.__name__
        print(f"  {rule_name}: {weight:.4f}")
    
    info_strategy = ConfidenceWeightedStrategy(
        rule_objects=all_rules,
        weights=info_weights
    )
    
    info_backtester = Backtester(data_handler, info_strategy)
    info_results = info_backtester.run(use_test_data=True)
    
    info_sharpe = info_backtester.calculate_sharpe()
    info_win_rate = sum(1 for t in info_results['trades'] if t[5] > 0) / info_results['num_trades'] if info_results['num_trades'] > 0 else 0
    
    print(f"Total Return: {info_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {info_results['num_trades']}")
    print(f"Sharpe Ratio: {info_sharpe:.4f}")
    print(f"Win Rate: {info_win_rate:.2f}")
    
    # 3. Test Regime-Based strategy with confidence
    print("\n--- Testing Regime-Based Strategy with Confidence ---")
    
    # Create enhanced regime detector
    regime_detector = EnhancedWaveletRegimeDetector(lookback_period=100, threshold=0.5)
    
    # Create regime-aware strategy factory
    strategy_factory = RegimeAwareStrategyFactory()
    
    # Create regime manager
    regime_manager = RegimeManager(
        regime_detector=regime_detector,
        strategy_factory=strategy_factory,
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
    
    print(f"Total Return: {regime_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {regime_results['num_trades']}")
    print(f"Sharpe Ratio: {regime_sharpe:.4f}")
    print(f"Win Rate: {regime_win_rate:.2f}")
    
    # 4. Combine Information Theory and Regime-Based approach
    print("\n--- Testing Combined Info Theory + Regime-Based Strategy ---")
    
    # Use information theory weights as base weights for each regime
    info_weights_by_regime = {}
    
    # Analyze regime distribution in training data
    data_handler.reset_train()
    regime_detector.reset()
    
    regime_bars = {}
    
    while True:
        bar = data_handler.get_next_train_bar()
        if bar is None:
            break
            
        regime = regime_detector.detect_regime(bar)
        if regime not in regime_bars:
            regime_bars[regime] = []
            
        regime_bars[regime].append(bar)
    
    # Create regime-specific data handlers and optimize with info theory
    for regime, bars in regime_bars.items():
        if len(bars) < 100:  # Skip regimes with too little data
            info_weights_by_regime[regime] = info_weights  # Use global weights
            continue
            
        # Create a custom data handler for this regime
        class RegimeSpecificDataHandler:
            def __init__(self, bars):
                self.bars = bars
                self.index = 0
                
            def get_next_train_bar(self):
                if self.index < len(self.bars):
                    bar = self.bars[self.index]
                    self.index += 1
                    return bar
                return None
                
            def reset_train(self):
                self.index = 0
        
        # Create data handler and optimize weights
        regime_data_handler = RegimeSpecificDataHandler(bars)
        regime_weights = optimize_with_information_theory(all_rules, regime_data_handler)
        
        info_weights_by_regime[regime] = regime_weights
        
        print(f"Optimized weights for {regime.name}:")
        for i, weight in enumerate(regime_weights):
            rule_name = all_rules[i].__class__.__name__
            print(f"  {rule_name}: {weight:.4f}")
    
    # Create a custom regime manager that uses info theory weights
    class InfoTheoryRegimeManager:
        def __init__(self, regime_detector, rule_objects, weights_by_regime):
            self.regime_detector = regime_detector
            self.rule_objects = rule_objects
            self.weights_by_regime = weights_by_regime
            self.default_weights = np.ones(len(rule_objects)) / len(rule_objects)
            
        def on_bar(self, event):
            bar = event.bar
            
            # Detect regime
            regime = self.regime_detector.detect_regime(bar)
            regime_confidence = self.regime_detector.get_confidence()
            
            # Get weights for this regime
            weights = self.weights_by_regime.get(regime, self.default_weights)
            
            # Create strategy with these weights
            strategy = ConfidenceWeightedStrategy(
                rule_objects=self.rule_objects,
                weights=weights
            )
            
            # Generate signal
            signal = strategy.on_bar(event)
            
            # Add regime info to metadata
            if signal.metadata is None:
                signal.metadata = {}
                
            signal.metadata['regime'] = regime.name
            signal.metadata['regime_confidence'] = regime_confidence
            
            return signal
            
        def reset(self):
            self.regime_detector.reset()
            for rule in self.rule_objects:
                if hasattr(rule, 'reset'):
                    rule.reset()
    
    # Create and test the combined strategy
    combined_manager = InfoTheoryRegimeManager(
        regime_detector=regime_detector,
        rule_objects=all_rules,
        weights_by_regime=info_weights_by_regime
    )
    
    combined_backtester = Backtester(data_handler, combined_manager)
    combined_results = combined_backtester.run(use_test_data=True)
    
    combined_sharpe = combined_backtester.calculate_sharpe()
    combined_win_rate = sum(1 for t in combined_results['trades'] if t[5] > 0) / combined_results['num_trades'] if combined_results['num_trades'] > 0 else 0
    
    print(f"Total Return: {combined_results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {combined_results['num_trades']}")
    print(f"Sharpe Ratio: {combined_sharpe:.4f}")
    print(f"Win Rate: {combined_win_rate:.2f}")
    
    # 5. Compare all strategies
    print("\n=== Strategy Comparison ===")
    print(f"{'Strategy':<30} {'Return':<10} {'Trades':<10} {'Sharpe':<10} {'Win Rate':<10}")
    print("-" * 70)
    print(f"{'Equal Weights':<30} {equal_results['total_percent_return']:>8.2f}% {equal_results['num_trades']:>10} {equal_sharpe:>9.4f} {equal_win_rate:>9.2f}")
    print(f"{'Information Theory':<30} {info_results['total_percent_return']:>8.2f}% {info_results['num_trades']:>10} {info_sharpe:>9.4f} {info_win_rate:>9.2f}")
    print(f"{'Regime-Based':<30} {regime_results['total_percent_return']:>8.2f}% {regime_results['num_trades']:>10} {regime_sharpe:>9.4f} {regime_win_rate:>9.2f}")
    print(f"{'Info Theory + Regime':<30} {combined_results['total_percent_return']:>8.2f}% {combined_results['num_trades']:>10} {combined_sharpe:>9.4f} {combined_win_rate:>9.2f}")
    
    # 6. Plot equity curves
    plot_strategy_comparison(
        {
            "Equal Weights": equal_results,
            "Information Theory": info_results,
            "Regime-Based": regime_results,
            "Info Theory + Regime": combined_results
        },
        "Strategy Comparison - Confidence-Based Systems"
    )
    
    return {
        "equal": equal_results,
        "info_theory": info_results,
        "regime": regime_results,
        "combined": combined_results
    }


def plot_strategy_comparison(results_dict, title):
    """Plot equity curves for multiple strategies."""
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    for name, results in results_dict.items():
        # Calculate equity curve
        equity = [10000]  # Start with $10,000
        for trade in results['trades']:
            equity.append(equity[-1] * np.exp(trade[5]))
        
        plt.plot(equity, label=f"{name} ({results['total_percent_return']:.1f}%)")
    
    plt.title(title)
    plt.xlabel("Trade Number")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("confidence_strategy_comparison.png")
    plt.show()


# Run the test
if __name__ == "__main__":
    run_confidence_based_system_test()
