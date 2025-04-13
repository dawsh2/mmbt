"""
Market regime detection and management for adaptive trading strategies.

This module provides tools for identifying market regimes and adapting trading
strategies accordingly, using a decoupled strategy factory pattern.
"""

import numpy as np
import pandas as pd
from collections import deque
from enum import Enum, auto
from abc import ABC, abstractmethod
from backtester import Backtester
from genetic_optimizer import GeneticOptimizer, WeightedRuleStrategy  # Keep WeightedRuleStrategy for now
from signals import Signal, SignalType
from strategy import Strategy, StrategyFactory

# from src.strategy.indicators.indicators import kaufman_adaptive_ma
# from src.utils.config.config_utils import get_regime_detection_params  # Assuming this exists
# from strategy import Strategy, StrategyFactory



class RegimeType(Enum):
    """Enumeration of different market regime types."""
    UNKNOWN = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGE_BOUND = auto()
    VOLATILE = auto()
    LOW_VOLATILITY = auto()
    BULL = auto()
    BEAR = auto()
    CHOPPY = auto()


class RegimeDetector:
    """
    Base class for regime detection methods.

    This abstract class defines the interface for regime detection algorithms.
    Subclasses should implement the detect_regime method.
    """
    @abstractmethod
    def detect_regime(self, bar_data):
        """Detect the current market regime based on bar data."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the regime detector state."""
        pass

class TrendStrengthRegimeDetector(RegimeDetector):
    # ... (rest of TrendStrengthRegimeDetector implementation - no changes needed for decoupling) ...
    def __init__(self, adx_period=14, adx_threshold=25):
        super().__init__()
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.high_history = deque(maxlen=adx_period + 1)
        self.low_history = deque(maxlen=adx_period + 1)
        self.close_history = deque(maxlen=adx_period + 1)
        self.tr_history = deque(maxlen=adx_period)
        self.plus_dm_history = deque(maxlen=adx_period)
        self.minus_dm_history = deque(maxlen=adx_period)
        self.adx_history = deque(maxlen=adx_period)
        self.current_regime = RegimeType.UNKNOWN

    def detect_regime(self, bar):
        # ... (same logic as before) ...
        # Add current price data to history
        self.high_history.append(bar['High'])
        self.low_history.append(bar['Low'])
        self.close_history.append(bar['Close'])

        # Need at least 2 bars to start calculations
        if len(self.high_history) < 2:
            return RegimeType.UNKNOWN

        # Calculate True Range (TR)
        high = self.high_history[-1]
        low = self.low_history[-1]
        prev_close = self.close_history[-2]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        self.tr_history.append(tr)

        # Calculate +DM and -DM
        prev_high = self.high_history[-2]
        prev_low = self.low_history[-2]

        plus_dm = max(0, high - prev_high)
        minus_dm = max(0, prev_low - low)

        if plus_dm > minus_dm:
            minus_dm = 0
        elif minus_dm > plus_dm:
            plus_dm = 0

        self.plus_dm_history.append(plus_dm)
        self.minus_dm_history.append(minus_dm)

        # Need enough history to calculate ADX
        if len(self.tr_history) < self.adx_period:
            return RegimeType.UNKNOWN

        # Calculate smoothed TR, +DM, and -DM
        tr_sum = sum(self.tr_history)
        plus_dm_sum = sum(self.plus_dm_history)
        minus_dm_sum = sum(self.minus_dm_history)

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
        minus_di = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0

        # Calculate DX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * di_diff / di_sum if di_sum > 0 else 0

        # Calculate ADX (smoothed DX)
        self.adx_history.append(dx)
        adx = sum(self.adx_history) / len(self.adx_history)

        # Determine regime based on ADX and directional indicators
        if adx > self.adx_threshold:
            if plus_di > minus_di:
                self.current_regime = RegimeType.TRENDING_UP
            else:
                self.current_regime = RegimeType.TRENDING_DOWN
        else:
            self.current_regime = RegimeType.RANGE_BOUND

        return self.current_regime

    def reset(self):
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.tr_history.clear()
        self.plus_dm_history.clear()
        self.minus_dm_history.clear()
        self.adx_history.clear()
        self.current_regime = RegimeType.UNKNOWN

class VolatilityRegimeDetector(RegimeDetector):
    # ... (rest of VolatilityRegimeDetector implementation - no changes needed) ...
    def __init__(self, lookback_period=20, volatility_threshold=0.015):
        super().__init__()
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.close_history = deque(maxlen=lookback_period + 1)
        self.returns_history = deque(maxlen=lookback_period)
        self.current_regime = RegimeType.UNKNOWN

    def detect_regime(self, bar):
        # Add current price to history
        self.close_history.append(bar['Close'])

        # Need at least 2 bars to calculate returns
        if len(self.close_history) < 2:
            return RegimeType.UNKNOWN

        # Calculate return and add to history
        current_close = self.close_history[-1]
        prev_close = self.close_history[-2]
        daily_return = (current_close / prev_close) - 1
        self.returns_history.append(daily_return)

        # Need enough history to calculate volatility
        if len(self.returns_history) < self.lookback_period:
            return RegimeType.UNKNOWN

        # Calculate volatility (standard deviation of returns)
        volatility = np.std(list(self.returns_history))

        # Determine regime based on volatility
        if volatility > self.volatility_threshold:
            self.current_regime = RegimeType.VOLATILE
        else:
            self.current_regime = RegimeType.LOW_VOLATILITY

        return self.current_regime

    def reset(self):
        self.close_history.clear()
        self.returns_history.clear()
        self.current_regime = RegimeType.UNKNOWN

class WaveletRegimeDetector(RegimeDetector):
    def __init__(self, lookback_period=100, threshold=0.5):
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.price_history = deque(maxlen=lookback_period)
        
    def detect_regime(self, bar):
        self.price_history.append(bar['Close'])
        
        if len(self.price_history) >= self.lookback_period:
            # Perform wavelet decomposition
            import pywt
            coeffs = pywt.wavedec(list(self.price_history), 'db1', level=3)
            
            # Calculate energy in different frequency bands
            energy_high = np.sum(coeffs[1] ** 2)  # High frequency (noise)
            energy_low = np.sum(coeffs[3] ** 2)   # Low frequency (trend)
            
            # Determine regime based on energy distribution
            if energy_high > self.threshold * energy_low:
                return RegimeType.VOLATILE
            elif np.abs(coeffs[3][-1] - coeffs[3][0]) > self.threshold:
                return RegimeType.TRENDING_UP if coeffs[3][-1] > coeffs[3][0] else RegimeType.TRENDING_DOWN
            else:
                return RegimeType.RANGE_BOUND
        

class KaufmanRegimeDetector(RegimeDetector):
    # ... (rest of KaufmanRegimeDetector implementation - no changes needed) ...
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.params = get_regime_detection_params()
        else:
            self.params = config

        self.kama_fast_period = self.params.get('kama_fast_period', 20)
        self.kama_slow_period = self.params.get('kama_slow_period', 50)
        self.efficiency_ratio_period = self.params.get('efficiency_ratio_period', 10)
        self.efficiency_strong_threshold = self.params.get('efficiency_strong_threshold', 0.6)
        self.efficiency_weak_threshold = self.params.get('efficiency_weak_threshold', 0.3)
        self.ma_proximity_threshold = self.params.get('ma_proximity_threshold', 0.001) # Percentage

        self.close_history = deque(maxlen=max(self.kama_slow_period, self.efficiency_ratio_period) + 1)
        self.kama_fast_history = deque(maxlen=self.kama_fast_period)
        self.kama_slow_history = deque(maxlen=self.kama_slow_period)
        self.efficiency_ratio_history = deque(maxlen=self.efficiency_ratio_period)
        self.current_regime = RegimeType.UNKNOWN

    def detect_regime(self, bar):
        close_price = bar['Close']
        self.close_history.append(close_price)

        if len(self.close_history) > max(self.kama_slow_period, self.efficiency_ratio_period):
            kama20 = kaufman_adaptive_ma(pd.Series(list(self.close_history)),
                                             n=self.kama_fast_period, fast_ema=2, slow_ema=20).iloc[-1]
            kama50 = kaufman_adaptive_ma(pd.Series(list(self.close_history)),
                                             n=self.kama_slow_period, fast_ema=4, slow_ema=50).iloc[-1]

            direction = abs(self.close_history[-1] - self.close_history[0])
            volatility = sum(abs(self.close_history[i] - self.close_history[i-1])
                             for i in range(1, len(self.close_history)))
            efficiency_ratio = direction / (volatility + 1e-10) # Avoid division by zero

            if close_price > kama50 and kama20 > kama50 and efficiency_ratio > self.efficiency_strong_threshold:
                self.current_regime = RegimeType.BULL
            elif close_price < kama50 and kama20 < kama50 and efficiency_ratio > self.efficiency_strong_threshold:
                self.current_regime = RegimeType.BEAR
            elif efficiency_ratio < self.efficiency_weak_threshold or \
                 abs(kama20 - kama50) / kama50 < self.ma_proximity_threshold:
                self.current_regime = RegimeType.CHOPPY
            elif close_price > kama50:
                self.current_regime = RegimeType.TRENDING_UP # Weak bull indication
            elif close_price < kama50:
                self.current_regime = RegimeType.TRENDING_DOWN # Weak bear indication
            else:
                self.current_regime = RegimeType.UNKNOWN # Catch-all

            return self.current_regime
        else:
            return RegimeType.UNKNOWN

    def reset(self):
        self.close_history.clear()
        self.kama_fast_history.clear()
        self.kama_slow_history.clear()
        self.efficiency_ratio_history.clear()
        self.current_regime = RegimeType.UNKNOWN


class RegimeManager:
    """
    Manages trading strategies based on detected market regimes using a StrategyFactory.
    """
    def __init__(self, regime_detector: RegimeDetector, strategy_factory: StrategyFactory,
                 rule_objects: list, data_handler=None):
        """
        Initialize the regime manager.

        Args:
            regime_detector: RegimeDetector object for identifying regimes
            strategy_factory: StrategyFactory object for creating strategies
            rule_objects: List of trading rule objects (passed to the factory)
            data_handler: Optional data handler for optimization
        """
        self.regime_detector = regime_detector
        self.strategy_factory = strategy_factory
        self.rule_objects = rule_objects
        self.data_handler = data_handler
        self.current_regime = RegimeType.UNKNOWN
        self.regime_strategies = {}  # Mapping from regime to strategy
        self.default_strategy = self.strategy_factory.create_default_strategy(rule_objects)

    def optimize_regime_strategies(self, regimes_to_optimize=None, optimization_metric='win_rate', verbose=True):
        """
        Optimize strategies for different market regimes.

        Args:
            regimes_to_optimize: List of regimes to optimize for (or None for all)
            optimization_metric: Metric to optimize ('win_rate', 'sharpe', 'return', etc.)
            verbose: Whether to print optimization progress

        Returns:
            dict: Mapping from regime to optimized parameters (e.g., weights)
        """
        if self.data_handler is None:
            raise ValueError("Data handler must be provided for optimization")

        if regimes_to_optimize is None:
            regimes_to_optimize = list(RegimeType)
            regimes_to_optimize.remove(RegimeType.UNKNOWN)

        # First, identify bars in each regime using the full dataset
        regime_bars = self._identify_regime_bars()

        # Initialize the dictionary to store optimal parameters
        optimal_params = {}

        for regime in regimes_to_optimize:
            if regime in regime_bars and len(regime_bars[regime]) >= 30:
                if verbose:
                    print(f"\nOptimizing strategy for {regime.name} regime "
                          f"({len(regime_bars[regime])} bars) using {optimization_metric} metric")

                # Create a regime-specific data handler
                regime_specific_data = self._create_regime_specific_data(regime_bars[regime])

                # Optimize parameters for this regime
                optimizer = GeneticOptimizer(
                    data_handler=regime_specific_data,
                    rule_objects=self.rule_objects,
                    population_size=15,
                    num_generations=30,
                    optimization_metric=optimization_metric
                )
                optimal_params[regime] = optimizer.optimize(verbose=verbose)

                # Create and store the optimized strategy
                self.regime_strategies[regime] = self.strategy_factory.create_strategy(
                    regime, self.rule_objects, optimal_params[regime]
                )

                if verbose:
                    print(f"Optimized parameters for {regime.name}: {optimal_params[regime]}")
            else:
                if verbose and regime in regime_bars:
                    print(f"Insufficient data for {regime.name} regime "
                          f"({len(regime_bars.get(regime, []))} bars). Using default strategy.")
                elif verbose and regime not in regime_bars:
                    print(f"No bars found for {regime.name} regime. Using default strategy.")
                self.regime_strategies.setdefault(regime, self.default_strategy) # Ensure a strategy exists

        return optimal_params

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
            regime = self.regime_detector.detect_regime(bar)  # Corrected line
            if regime not in regime_bars:
                regime_bars[regime] = []
            regime_bars[regime].append((bar_index, bar))
            bar_index += 1
        return regime_bars

    def _create_regime_specific_data(self, regime_bars):
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

    def get_strategy_for_regime(self, regime: RegimeType) -> Strategy:
        """Get the optimized strategy for a specific regime."""
        return self.regime_strategies.get(regime, self.default_strategy)

    def on_bar(self, event):
        """Process a bar and generate trading signals using the appropriate strategy."""
        bar = event.bar
        
        # Handle end-of-day closing if present
        if bar.get('is_eod', False):
            # We don't need to do anything special here since the backtester
            # already handles EOD position closing
            pass
            
        self.current_regime = self.regime_detector.detect_regime(bar)
        strategy = self.get_strategy_for_regime(self.current_regime)
        if hasattr(strategy, 'on_bar'):
            return strategy.on_bar(event)
        return None
    
    def on_bar(self, event):
        """Process a bar and generate trading signals using the appropriate strategy."""
        bar = event.bar
        self.current_regime = self.regime_detector.detect_regime(bar)
        strategy = self.get_strategy_for_regime(self.current_regime)
        if hasattr(strategy, 'on_bar'):
            return strategy.on_bar(event)
        return None

    def reset(self):
        """Reset the regime manager and its components."""
        self.regime_detector.reset()
        self.default_strategy.reset()
        for strategy in self.regime_strategies.values():
            strategy.reset()
        self.current_regime = RegimeType.UNKNOWN

# Example of how to use the decoupled RegimeManager:
if __name__ == '__main__':
    # Assume you have your rule objects defined
    from src.strategy.rules import Rule0, Rule1, Rule2
    rule_objects = [Rule0(), Rule1(), Rule2()]

    # Assume you have a data handler
    class MockDataHandler:
        def __init__(self, data):
            self.train_data = data
            self.train_index = 0
            self.test_data = data
            self.test_index = 0

        def get_next_train_bar(self):
            if self.train_index < len(self.train_data):
                bar = self.train_data[self.train_index]
                self.train_index += 1
                return bar
            return None

        def get_next_test_bar(self):
            if self.test_index < len(self.test_data):
                bar = self.test_data[self.test_index]
                self.test_index += 1
                return bar
            return None

        def reset_train(self):
            self.train_index = 0

        def reset_test(self):
            self.test_index = 0

    # Create some mock data
    mock_data = [{'High': 105, 'Low': 100, 'Close': 102},
                 {'High': 108, 'Low': 103, 'Close': 107},
                 {'High': 110, 'Low': 105, 'Close': 109},
                 {'High': 107, 'Low': 102, 'Close': 104},
                 {'High': 103, 'Low': 98, 'Close': 100}]
    data_handler = MockDataHandler(mock_data)

    # Create a regime detector
    regime_detector = TrendStrengthRegimeDetector() # Or any other detector

    # Create a strategy factory
    strategy_factory = WeightedRuleStrategyFactory()

    # Instantiate the RegimeManager
    regime_manager = RegimeManager(regime_detector, strategy_factory, rule_objects, data_handler)

    # Example of optimizing strategies (requires more data for meaningful optimization)
    # optimized_params = regime_manager.optimize_regime_strategies(verbose=True)
    # print("Optimized Parameters:", optimized_params)

    # Example of processing a bar
    class BarEvent:
        def __init__(self, bar):
            self.bar = bar

    data_handler.reset_train()
    while True:
        bar = data_handler.get_next_train_bar()
        if bar is None:
            break
        event = BarEvent(bar)
        signal = regime_manager.on_bar(event)
        print(f"Regime: {regime_manager.current_regime}, Signal: {signal}")

    # Reset the regime manager
    regime_manager.reset()
