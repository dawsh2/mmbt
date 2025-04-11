"""
Market regime detection and management for adaptive trading strategies.

This module provides tools for identifying market regimes and adapting trading
strategies accordingly.
"""

import numpy as np
import pandas as pd
from collections import deque
from enum import Enum, auto
from backtester import Backtester
from genetic_optimizer import WeightedRuleStrategy

class RegimeType(Enum):
    """Enumeration of different market regime types."""
    UNKNOWN = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGE_BOUND = auto()
    VOLATILE = auto()
    LOW_VOLATILITY = auto()

class RegimeDetector:
    """
    Base class for regime detection methods.
    
    This abstract class defines the interface for regime detection algorithms.
    Subclasses should implement the detect_regime method.
    """
    
    def __init__(self):
        """Initialize the regime detector."""
        pass
    
    def detect_regime(self, bar_data):
        """
        Detect the current market regime based on bar data.
        
        Args:
            bar_data: Market data to analyze
            
        Returns:
            RegimeType: The detected market regime
        """
        raise NotImplementedError("Subclasses must implement detect_regime method")
    
    def reset(self):
        """Reset the regime detector state."""
        pass


class TrendStrengthRegimeDetector(RegimeDetector):
    """
    Detects market regimes based on trend strength indicators.
    
    Uses ADX (Average Directional Index) and price movement 
    to categorize the market into trending or range-bound regimes.
    """
    
    def __init__(self, adx_period=14, adx_threshold=25):
        """
        Initialize the trend strength regime detector.
        
        Args:
            adx_period: Period for calculating ADX
            adx_threshold: Threshold for determining strong trends
        """
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
        """
        Detect market regime based on ADX and price movement.
        
        Args:
            bar: Current market data bar
            
        Returns:
            RegimeType: Detected market regime
        """
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
        """Reset the detector's state."""
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.tr_history.clear()
        self.plus_dm_history.clear()
        self.minus_dm_history.clear()
        self.adx_history.clear()
        self.current_regime = RegimeType.UNKNOWN


class VolatilityRegimeDetector(RegimeDetector):
    """
    Detects market regimes based on volatility levels.
    
    Uses historical volatility to categorize the market into 
    high-volatility or low-volatility regimes.
    """
    
    def __init__(self, lookback_period=20, volatility_threshold=0.015):
        """
        Initialize the volatility regime detector.
        
        Args:
            lookback_period: Period for volatility calculation
            volatility_threshold: Threshold for high vs low volatility
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.close_history = deque(maxlen=lookback_period + 1)
        self.returns_history = deque(maxlen=lookback_period)
        self.current_regime = RegimeType.UNKNOWN
    
    def detect_regime(self, bar):
        """
        Detect market regime based on historical volatility.
        
        Args:
            bar: Current market data bar
            
        Returns:
            RegimeType: Detected market regime
        """
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
        """Reset the detector's state."""
        self.close_history.clear()
        self.returns_history.clear()
        self.current_regime = RegimeType.UNKNOWN


class RegimeManager:
    """
    Manages trading strategies based on detected market regimes.
    
    This class maintains multiple strategy configurations optimized for
    different market regimes and switches between them as regimes change.
    """
    
    def __init__(self, regime_detector, rule_objects, data_handler=None):
        """
        Initialize the regime manager.
        
        Args:
            regime_detector: RegimeDetector object for identifying regimes
            rule_objects: List of trading rule objects
            data_handler: Optional data handler for optimization
        """
        self.regime_detector = regime_detector
        self.rule_objects = rule_objects
        self.data_handler = data_handler
        self.current_regime = RegimeType.UNKNOWN
        self.regime_strategies = {}  # Mapping from regime to strategy
        self.default_strategy = WeightedRuleStrategy(rule_objects)

    # Modify the optimize_regime_strategies method in RegimeManager class
    # in regime_detection.py

    def optimize_regime_strategies(self, regimes_to_optimize=None, optimization_metric='win_rate', verbose=True):
        """
        Optimize strategies for different market regimes.

        Args:
            regimes_to_optimize: List of regimes to optimize for (or None for all)
            optimization_metric: Metric to optimize ('win_rate', 'sharpe', 'return', etc.)
            verbose: Whether to print optimization progress

        Returns:
            dict: Mapping from regime to optimized weights
        """
        if self.data_handler is None:
            raise ValueError("Data handler must be provided for optimization")

        if regimes_to_optimize is None:
            regimes_to_optimize = [
                RegimeType.TRENDING_UP,
                RegimeType.TRENDING_DOWN,
                RegimeType.RANGE_BOUND,
                RegimeType.VOLATILE,
                RegimeType.LOW_VOLATILITY
            ]

        # First, identify bars in each regime using the full dataset
        regime_bars = self._identify_regime_bars()

        # For each regime, optimize a separate strategy
        for regime in regimes_to_optimize:
            if regime in regime_bars and len(regime_bars[regime]) >= 30:
                if verbose:
                    print(f"\nOptimizing strategy for {regime.name} regime "
                          f"({len(regime_bars[regime])} bars) using {optimization_metric} metric")

                # Create a regime-specific data handler with only bars from this regime
                regime_specific_data = self._create_regime_specific_data(regime_bars[regime])

                # Optimize weights for this regime using the specified metric
                from genetic_optimizer import GeneticOptimizer
                optimizer = GeneticOptimizer(
                    data_handler=regime_specific_data,
                    rule_objects=self.rule_objects,
                    population_size=15,
                    num_generations=30,
                    optimization_metric=optimization_metric  # Use the specified metric
                )
                optimal_weights = optimizer.optimize(verbose=verbose)

                # Create a strategy with optimized weights
                self.regime_strategies[regime] = WeightedRuleStrategy(
                    rule_objects=self.rule_objects,
                    weights=optimal_weights
                )

                if verbose:
                    print(f"Optimized weights for {regime.name}: {optimal_weights}")
            else:
                if verbose and regime in regime_bars:
                    print(f"Insufficient data for {regime.name} regime "
                          f"({len(regime_bars.get(regime, []))} bars). Using default strategy.")

        return {regime: strategy.weights for regime, strategy in self.regime_strategies.items()}
        

    def _identify_regime_bars(self):
        """
        Identify which bars belong to each regime.
        
        Returns:
            dict: Mapping from regime to list of bar indices
        """
        regime_bars = {}
        self.regime_detector.reset()
        
        # Process all bars in the training dataset
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
        """
        Create a mock data handler with only bars from a specific regime.
        
        Args:
            regime_bars: List of (index, bar) tuples for the regime
            
        Returns:
            object: A data handler-like object for the specific regime
        """
        # Simple mock data handler that will return only the selected bars
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
    
    def get_strategy_for_regime(self, regime):
        """
        Get the optimized strategy for a specific regime.
        
        Args:
            regime: The market regime
            
        Returns:
            WeightedRuleStrategy: Strategy optimized for the regime
        """
        return self.regime_strategies.get(regime, self.default_strategy)
    
    def on_bar(self, event):
        """
        Process a bar and generate trading signals using the appropriate strategy.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            dict: Signal information
        """
        bar = event.bar
        
        # Detect current regime
        self.current_regime = self.regime_detector.detect_regime(bar)
        
        # Get the appropriate strategy for this regime
        strategy = self.get_strategy_for_regime(self.current_regime)
        
        # Generate signal using the selected strategy
        return strategy.on_bar(event)
    
    def reset(self):
        """Reset the regime manager and its components."""
        self.regime_detector.reset()
        self.default_strategy.reset()
        for strategy in self.regime_strategies.values():
            strategy.reset()
        self.current_regime = RegimeType.UNKNOWN
