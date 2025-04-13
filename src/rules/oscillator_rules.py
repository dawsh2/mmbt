"""
Oscillator Rules Module

This module implements various oscillator-based trading rules such as
RSI, Stochastic, CCI, and other momentum oscillators.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque

from ..signals import Signal, SignalType
from .rule_base import Rule
from .rule_registry import register_rule


@register_rule(category="oscillator")
class RSIRule(Rule):
    """
    Relative Strength Index (RSI) Rule.
    
    This rule generates signals based on RSI, an oscillator that measures the 
    speed and change of price movements on a scale of 0 to 100.
    """
    
    def __init__(self, 
                 name: str = "rsi_rule", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "RSI overbought/oversold rule"):
        """
        Initialize the RSI rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - rsi_period: Period for RSI calculation (default: 14)
                - overbought: Overbought level (default: 70)
                - oversold: Oversold level (default: 30)
                - signal_type: Signal generation method ('levels', 'divergence', 'midline') (default: 'levels')
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=self.params['rsi_period'] * 3)  # Store more prices for calculation
        self.price_changes = deque(maxlen=self.params['rsi_period'] * 3)
        self.rsi_history = deque(maxlen=20)  # Store RSI history for divergence
        self.current_signal_type = SignalType.NEUTRAL
        
        # For divergence detection
        self.price_highs = deque(maxlen=5)
        self.price_lows = deque(maxlen=5)
        self.rsi_highs = deque(maxlen=5)
        self.rsi_lows = deque(maxlen=5)
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'rsi_period': 14,
            'overbought': 70,
            'oversold': 30,
            'signal_type': 'levels'  # 'levels', 'divergence', or 'midline'
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['rsi_period'] <= 0:
            raise ValueError("RSI period must be positive")
            
        if not 0 <= self.params['oversold'] < self.params['overbought'] <= 100:
            raise ValueError("Oversold must be less than overbought, both between 0 and 100")
            
        valid_signal_types = ['levels', 'divergence', 'midline']
        if self.params['signal_type'] not in valid_signal_types:
            raise ValueError(f"signal_type must be one of {valid_signal_types}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on RSI.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if 'Close' not in data:
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing Close price data'}
            )
            
        # Get parameters
        rsi_period = self.params['rsi_period']
        overbought = self.params['overbought']
        oversold = self.params['oversold']
        signal_type = self.params['signal_type']
        
        # Extract price data
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Store price for highest high/lowest low tracking (for divergence)
        high = data.get('High', close)
        low = data.get('Low', close)
        
        # Update price history
        self.prices.append(close)
        
        # Need at least 2 prices to calculate changes
        if len(self.prices) < 2:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'initializing'}
            )
        
        # Calculate price change
        price_change = self.prices[-1] - self.prices[-2]
        self.price_changes.append(price_change)
        
        # Need enough price changes to calculate RSI
        if len(self.price_changes) < rsi_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate RSI
        gains = [max(0, change) for change in list(self.price_changes)[-rsi_period:]]
        losses = [abs(min(0, change)) for change in list(self.price_changes)[-rsi_period:]]
        
        avg_gain = sum(gains) / rsi_period
        avg_loss = sum(losses) / rsi_period
        
        if avg_loss == 0:
            rs = float('inf')
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.rsi_history.append(rsi)
        
        # Update price and RSI extremes for divergence detection
        if len(self.prices) >= 3 and self.prices[-2] >= self.prices[-1] and self.prices[-2] >= self.prices[-3]:
            # Local price high
            self.price_highs.append((len(self.prices) - 2, self.prices[-2]))
        
        if len(self.prices) >= 3 and self.prices[-2] <= self.prices[-1] and self.prices[-2] <= self.prices[-3]:
            # Local price low
            self.price_lows.append((len(self.prices) - 2, self.prices[-2]))
        
        if len(self.rsi_history) >= 3 and self.rsi_history[-2] >= self.rsi_history[-1] and self.rsi_history[-2] >= self.rsi_history[-3]:
            # Local RSI high
            self.rsi_highs.append((len(self.rsi_history) - 2, self.rsi_history[-2]))
        
        if len(self.rsi_history) >= 3 and self.rsi_history[-2] <= self.rsi_history[-1] and self.rsi_history[-2] <= self.rsi_history[-3]:
            # Local RSI low
            self.rsi_lows.append((len(self.rsi_history) - 2, self.rsi_history[-2]))
        
        # Generate signal based on selected method
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        signal_info = {}
        
        if signal_type == 'levels':
            # Classic overbought/oversold levels
            if rsi >= overbought:
                signal_type_enum = SignalType.SELL
                # Calculate confidence based on how far into overbought
                overbought_penetration = min(1.0, (rsi - overbought) / (100 - overbought))
                confidence = min(0.9, 0.6 + overbought_penetration * 0.3)
                signal_info['condition'] = 'overbought'
                
            elif rsi <= oversold:
                signal_type_enum = SignalType.BUY
                # Calculate confidence based on how far into oversold
                oversold_penetration = min(1.0, (oversold - rsi) / oversold)
                confidence = min(0.9, 0.6 + oversold_penetration * 0.3)
                signal_info['condition'] = 'oversold'
                
            # Exits from extreme conditions
            elif len(self.rsi_history) >= 2 and self.rsi_history[-2] >= overbought and rsi < overbought:
                signal_type_enum = SignalType.NEUTRAL  # Exit overbought condition
                confidence = 0.7
                signal_info['condition'] = 'exit_overbought'
                
            elif len(self.rsi_history) >= 2 and self.rsi_history[-2] <= oversold and rsi > oversold:
                signal_type_enum = SignalType.NEUTRAL  # Exit oversold condition
                confidence = 0.7
                signal_info['condition'] = 'exit_oversold'
                
        elif signal_type == 'divergence' and len(self.price_highs) >= 2 and len(self.price_lows) >= 2 and len(self.rsi_highs) >= 2 and len(self.rsi_lows) >= 2:
            # Bearish divergence: Price making higher highs but RSI making lower highs
            if self.price_highs[-1][1] > self.price_highs[-2][1] and self.rsi_highs[-1][1] < self.rsi_highs[-2][1]:
                signal_type_enum = SignalType.SELL
                confidence = 0.8  # High confidence for divergence signals
                signal_info['condition'] = 'bearish_divergence'
                
            # Bullish divergence: Price making lower lows but RSI making higher lows
            elif self.price_lows[-1][1] < self.price_lows[-2][1] and self.rsi_lows[-1][1] > self.rsi_lows[-2][1]:
                signal_type_enum = SignalType.BUY
                confidence = 0.8  # High confidence for divergence signals
                signal_info['condition'] = 'bullish_divergence'
                
        elif signal_type == 'midline' and len(self.rsi_history) >= 2:
            # Crossing above/below the midline (50)
            if self.rsi_history[-2] < 50 and rsi >= 50:
                signal_type_enum = SignalType.BUY  # Bullish momentum
                confidence = 0.6
                signal_info['condition'] = 'cross_above_midline'
                
            elif self.rsi_history[-2] > 50 and rsi <= 50:
                signal_type_enum = SignalType.SELL  # Bearish momentum
                confidence = 0.6
                signal_info['condition'] = 'cross_below_midline'
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'rsi': rsi,
                'overbought': overbought,
                'oversold': oversold,
                **signal_info
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=self.params['rsi_period'] * 3)
        self.price_changes = deque(maxlen=self.params['rsi_period'] * 3)
        self.rsi_history = deque(maxlen=20)
        self.price_highs = deque(maxlen=5)
        self.price_lows = deque(maxlen=5)
        self.rsi_highs = deque(maxlen=5)
        self.rsi_lows = deque(maxlen=5)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="oscillator")
class StochasticRule(Rule):
    """
    Stochastic Oscillator Rule.
    
    This rule generates signals based on the Stochastic Oscillator, which measures
    the current price relative to the price range over a period of time.
    """
    
    def __init__(self, 
                 name: str = "stochastic_rule", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Stochastic oscillator rule"):
        """
        Initialize the Stochastic Oscillator rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - k_period: %K period (default: 14)
                - k_slowing: %K slowing period (default: 3)
                - d_period: %D period (default: 3)
                - overbought: Overbought level (default: 80)
                - oversold: Oversold level (default: 20)
                - signal_type: Signal generation method ('levels', 'crossover', 'both') (default: 'both')
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=self.params['k_period'] * 2)
        self.low_history = deque(maxlen=self.params['k_period'] * 2)
        self.close_history = deque(maxlen=self.params['k_period'] * 2)
        self.k_history = deque(maxlen=max(self.params['k_slowing'], self.params['d_period']) * 2)
        self.k_slow_history = deque(maxlen=self.params['d_period'] * 2)
        self.d_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'k_period': 14,
            'k_slowing': 3,
            'j_period': 3,  # Often used for the J-Line
            'd_period': 3,
            'overbought': 80,
            'oversold': 20,
            'signal_type': 'both'  # 'levels', 'crossover', or 'both'
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        for param in ['k_period', 'k_slowing', 'd_period']:
            if self.params[param] <= 0:
                raise ValueError(f"{param} must be positive")
                
        if not 0 <= self.params['oversold'] < self.params['overbought'] <= 100:
            raise ValueError("Oversold must be less than overbought, both between 0 and 100")
            
        valid_signal_types = ['levels', 'crossover', 'both']
        if self.params['signal_type'] not in valid_signal_types:
            raise ValueError(f"signal_type must be one of {valid_signal_types}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on Stochastic Oscillator.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if not all(k in data for k in ['High', 'Low', 'Close']):
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=data.get('Close', None),
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing required price data'}
            )
            
        # Get parameters
        k_period = self.params['k_period']
        k_slowing = self.params['k_slowing']
        d_period = self.params['d_period']
        overbought = self.params['overbought']
        oversold = self.params['oversold']
        signal_type = self.params['signal_type']
        
        # Extract price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Need enough price history to calculate stochastic
        if len(self.high_history) < k_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate raw %K (Fast Stochastic)
        highest_high = max(list(self.high_history)[-k_period:])
        lowest_low = min(list(self.low_history)[-k_period:])
        
        if highest_high == lowest_low:
            # Avoid division by zero
            raw_k = 50
        else:
            # Calculate %K value
            raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            
        self.k_history.append(raw_k)
        
        # Need enough %K history for slowing
        if len(self.k_history) < k_slowing:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting %K data'}
            )
        
        # Calculate slowed %K
        k_slow = sum(list(self.k_history)[-k_slowing:]) / k_slowing
        self.k_slow_history.append(k_slow)
        
        # Need enough slowed %K history for %D
        if len(self.k_slow_history) < d_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting %K slow data'}
            )
        
        # Calculate %D
        d = sum(list(self.k_slow_history)[-d_period:]) / d_period
        self.d_history.append(d)
        
        # Generate signal
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        signal_info = {}
        
        # We need at least 2 values of %K and %D for crossover signals
        if len(self.k_slow_history) >= 2 and len(self.d_history) >= 2:
            current_k = self.k_slow_history[-1]
            current_d = self.d_history[-1]
            prev_k = self.k_slow_history[-2]
            prev_d = self.d_history[-2]
            
            # Check for overbought/oversold condition
            if signal_type in ['levels', 'both']:
                if current_k > overbought and current_d > overbought:
                    signal_type_enum = SignalType.SELL
                    # Calculate confidence based on how far into overbought
                    overbought_penetration = min(1.0, (current_k - overbought) / (100 - overbought))
                    confidence = min(0.9, 0.6 + overbought_penetration * 0.3)
                    signal_info['condition'] = 'overbought'
                    
                elif current_k < oversold and current_d < oversold:
                    signal_type_enum = SignalType.BUY
                    # Calculate confidence based on how far into oversold
                    oversold_penetration = min(1.0, (oversold - current_k) / oversold)
                    confidence = min(0.9, 0.6 + oversold_penetration * 0.3)
                    signal_info['condition'] = 'oversold'
            
            # Check for crossover
            if signal_type in ['crossover', 'both']:
                if prev_k <= prev_d and current_k > current_d:
                    # Bullish crossover
                    crossover_signal = SignalType.BUY
                    # Crossover strength
                    crossover_strength = abs(current_k - current_d)
                    crossover_confidence = min(1.0, 0.6 + (crossover_strength / 10) * 0.4)
                    
                    # Combine with levels signal if using 'both'
                    if signal_type == 'both':
                        if signal_type_enum == SignalType.BUY:
                            # Both signals agree, increase confidence
                            signal_type_enum = SignalType.BUY
                            confidence = min(1.0, confidence + 0.1)
                            signal_info['condition'] = 'oversold_with_bullish_cross'
                        elif signal_type_enum == SignalType.NEUTRAL:
                            # Only crossover signal present
                            signal_type_enum = SignalType.BUY
                            confidence = crossover_confidence
                            signal_info['condition'] = 'bullish_cross'
                        # Ignore if signals contradict (overbought but bullish cross)
                    else:
                        # Only using crossover signal
                        signal_type_enum = SignalType.BUY
                        confidence = crossover_confidence
                        signal_info['condition'] = 'bullish_cross'
                        
                elif prev_k >= prev_d and current_k < current_d:
                    # Bearish crossover
                    crossover_signal = SignalType.SELL
                    # Crossover strength
                    crossover_strength = abs(current_k - current_d)
                    crossover_confidence = min(1.0, 0.6 + (crossover_strength / 10) * 0.4)
                    
                    # Combine with levels signal if using 'both'
                    if signal_type == 'both':
                        if signal_type_enum == SignalType.SELL:
                            # Both signals agree, increase confidence
                            signal_type_enum = SignalType.SELL
                            confidence = min(1.0, confidence + 0.1)
                            signal_info['condition'] = 'overbought_with_bearish_cross'
                        elif signal_type_enum == SignalType.NEUTRAL:
                            # Only crossover signal present
                            signal_type_enum = SignalType.SELL
                            confidence = crossover_confidence
                            signal_info['condition'] = 'bearish_cross'
                        # Ignore if signals contradict (oversold but bearish cross)
                    else:
                        # Only using crossover signal
                        signal_type_enum = SignalType.SELL
                        confidence = crossover_confidence
                        signal_info['condition'] = 'bearish_cross'
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'k': k_slow,
                'd': d,
                'overbought': overbought,
                'oversold': oversold,
                **signal_info
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=self.params['k_period'] * 2)
        self.low_history = deque(maxlen=self.params['k_period'] * 2)
        self.close_history = deque(maxlen=self.params['k_period'] * 2)
        self.k_history = deque(maxlen=max(self.params['k_slowing'], self.params['d_period']) * 2)
        self.k_slow_history = deque(maxlen=self.params['d_period'] * 2)
        self.d_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="oscillator")
class CCIRule(Rule):
    """
    Commodity Channel Index (CCI) Rule.
    
    This rule uses the CCI oscillator to identify overbought and oversold 
    conditions as well as trend strength and potential reversals.
    """
    
    def __init__(self, 
                 name: str = "cci_rule", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "CCI oscillator rule"):
        """
        Initialize the CCI rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - period: CCI calculation period (default: 20)
                - overbought: Overbought level (default: 100)
                - oversold: Oversold level (default: -100)
                - extreme_overbought: Extreme overbought level (default: 200)
                - extreme_oversold: Extreme oversold level (default: -200)
                - zero_line_cross: Whether to use zero line crossovers (default: True)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=self.params['period'] * 2)
        self.low_history = deque(maxlen=self.params['period'] * 2)
        self.close_history = deque(maxlen=self.params['period'] * 2)
        self.cci_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'period': 20,
            'overbought': 100,
            'oversold': -100,
            'extreme_overbought': 200,
            'extreme_oversold': -200,
            'zero_line_cross': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['period'] <= 0:
            raise ValueError("Period must be positive")
            
        if not self.params['oversold'] < 0 < self.params['overbought']:
            raise ValueError("Oversold must be negative and overbought must be positive")
            
        if not self.params['extreme_oversold'] < self.params['oversold'] < self.params['overbought'] < self.params['extreme_overbought']:
            raise ValueError("Levels must be in order: extreme_oversold < oversold < overbought < extreme_overbought")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on CCI.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if not all(k in data for k in ['High', 'Low', 'Close']):
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=data.get('Close', None),
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing required price data'}
            )
            
        # Get parameters
        period = self.params['period']
        overbought = self.params['overbought']
        oversold = self.params['oversold']
        extreme_overbought = self.params['extreme_overbought']
        extreme_oversold = self.params['extreme_oversold']
        zero_line_cross = self.params['zero_line_cross']
        
        # Extract price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Need enough price history to calculate CCI
        if len(self.high_history) < period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate typical price for each bar
        typical_prices = [(self.high_history[i] + self.low_history[i] + self.close_history[i]) / 3 
                         for i in range(-period, 0)]
        
        # Calculate simple moving average of typical prices
        sma_tp = sum(typical_prices) / period
        
        # Calculate mean deviation
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices) / period
        
        # Calculate CCI
        current_tp = (high + low + close) / 3
        
        if mean_deviation == 0:
            # Avoid division by zero
            cci = 0
        else:
            # Standard CCI formula
            cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
            
        self.cci_history.append(cci)
        
        # Generate signal
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        signal_info = {}
        
        # Check for extreme conditions (higher priority)
        if cci >= extreme_overbought:
            signal_type_enum = SignalType.SELL
            # Calculate confidence based on how far past extreme level
            overbought_excess = (cci - extreme_overbought) / extreme_overbought
            confidence = min(1.0, 0.8 + overbought_excess * 0.2)
            signal_info['condition'] = 'extreme_overbought'
            
        elif cci <= extreme_oversold:
            signal_type_enum = SignalType.BUY
            # Calculate confidence based on how far past extreme level
            oversold_excess = (extreme_oversold - cci) / abs(extreme_oversold)
            confidence = min(1.0, 0.8 + oversold_excess * 0.2)
            signal_info['condition'] = 'extreme_oversold'
            
        # Check for regular overbought/oversold conditions
        elif cci >= overbought:
            signal_type_enum = SignalType.SELL
            # Calculate confidence based on how far into overbought
            overbought_penetration = (cci - overbought) / (extreme_overbought - overbought)
            confidence = min(0.9, 0.6 + overbought_penetration * 0.3)
            signal_info['condition'] = 'overbought'
            
        elif cci <= oversold:
            signal_type_enum = SignalType.BUY
            # Calculate confidence based on how far into oversold
            oversold_penetration = (oversold - cci) / (oversold - extreme_oversold)
            confidence = min(0.9, 0.6 + oversold_penetration * 0.3)
            signal_info['condition'] = 'oversold'
            
        # Check for zero line crossovers
        elif zero_line_cross and len(self.cci_history) >= 2:
            prev_cci = self.cci_history[-2]
            
            if prev_cci <= 0 and cci > 0:
                signal_type_enum = SignalType.BUY
                confidence = 0.6
                signal_info['condition'] = 'zero_line_cross_up'
                
            elif prev_cci >= 0 and cci < 0:
                signal_type_enum = SignalType.SELL
                confidence = 0.6
                signal_info['condition'] = 'zero_line_cross_down'
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'cci': cci,
                'overbought': overbought,
                'oversold': oversold,
                'extreme_overbought': extreme_overbought,
                'extreme_oversold': extreme_oversold,
                **signal_info
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=self.params['period'] * 2)
        self.low_history = deque(maxlen=self.params['period'] * 2)
        self.close_history = deque(maxlen=self.params['period'] * 2)
        self.cci_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="oscillator")
class MACDHistogramRule(Rule):
    """
    MACD Histogram Rule.
    
    This rule uses the MACD histogram to identify momentum shifts in a trend,
    focusing on changes in the histogram rather than just MACD line crossovers.
    """
    
    def __init__(self, 
                 name: str = "macd_histogram", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "MACD histogram rule"):
        """
        Initialize the MACD Histogram rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - fast_period: Fast EMA period (default: 12)
                - slow_period: Slow EMA period (default: 26)
                - signal_period: Signal line period (default: 9)
                - zero_line_cross: Whether to use zero line crossovers (default: True)
                - divergence: Whether to detect divergence (default: True)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=self.params['slow_period'] * 3)
        self.fast_ema = None
        self.slow_ema = None
        self.macd_history = deque(maxlen=self.params['signal_period'] * 3)
        self.signal_line = None
        self.histogram_history = deque(maxlen=20)  # Store more history for divergence detection
        
        # Store price highs/lows and histogram highs/lows for divergence detection
        self.price_highs = deque(maxlen=5)
        self.price_lows = deque(maxlen=5)
        self.hist_highs = deque(maxlen=5)
        self.hist_lows = deque(maxlen=5)
        
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'zero_line_cross': True,
            'divergence': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['fast_period'] >= self.params['slow_period']:
            raise ValueError("Fast period must be smaller than slow period")
            
        for param in ['fast_period', 'slow_period', 'signal_period']:
            if self.params[param] <= 0:
                raise ValueError(f"{param} must be positive")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on MACD histogram.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if 'Close' not in data:
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing Close price data'}
            )
            
        # Get parameters
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        signal_period = self.params['signal_period']
        zero_line_cross = self.params['zero_line_cross']
        use_divergence = self.params['divergence']
        
        # Extract price data
        close = data['Close']
        high = data.get('High', close)
        low = data.get('Low', close)
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.prices.append(close)
        
        # Need enough price history to start calculations
        if len(self.prices) < slow_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate EMAs for MACD
        if self.fast_ema is None:
            # Initialize EMAs with SMA
            self.fast_ema = sum(list(self.prices)[-fast_period:]) / fast_period
            
        if self.slow_ema is None:
            # Initialize EMAs with SMA
            self.slow_ema = sum(list(self.prices)[-slow_period:]) / slow_period
            
        # Update EMAs
        alpha_fast = 2 / (fast_period + 1)
        self.fast_ema = (close * alpha_fast) + (self.fast_ema * (1 - alpha_fast))
            
        alpha_slow = 2 / (slow_period + 1)
        self.slow_ema = (close * alpha_slow) + (self.slow_ema * (1 - alpha_slow))
        
        # Calculate MACD
        macd = self.fast_ema - self.slow_ema
        self.macd_history.append(macd)
        
        # Need enough MACD history to calculate signal line
        if len(self.macd_history) < signal_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting MACD data'}
            )
        
        # Calculate signal line
        if self.signal_line is None:
            # Initialize signal line with SMA
            self.signal_line = sum(list(self.macd_history)[-signal_period:]) / signal_period
            
        # Update signal line
        alpha_signal = 2 / (signal_period + 1)
        self.signal_line = (macd * alpha_signal) + (self.signal_line * (1 - alpha_signal))
        
        # Calculate histogram
        histogram = macd - self.signal_line
        self.histogram_history.append(histogram)
        
        # Update price and histogram extremes for divergence detection
        if use_divergence and len(self.prices) >= 3 and self.prices[-2] >= self.prices[-1] and self.prices[-2] >= self.prices[-3]:
            # Local price high
            self.price_highs.append((len(self.prices) - 2, self.prices[-2]))
        
        if use_divergence and len(self.prices) >= 3 and self.prices[-2] <= self.prices[-1] and self.prices[-2] <= self.prices[-3]:
            # Local price low
            self.price_lows.append((len(self.prices) - 2, self.prices[-2]))
        
        if use_divergence and len(self.histogram_history) >= 3 and self.histogram_history[-2] >= self.histogram_history[-1] and self.histogram_history[-2] >= self.histogram_history[-3]:
            # Local histogram high
            self.hist_highs.append((len(self.histogram_history) - 2, self.histogram_history[-2]))
        
        if use_divergence and len(self.histogram_history) >= 3 and self.histogram_history[-2] <= self.histogram_history[-1] and self.histogram_history[-2] <= self.histogram_history[-3]:
            # Local histogram low
            self.hist_lows.append((len(self.histogram_history) - 2, self.histogram_history[-2]))
        
        # Generate signal
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        signal_info = {}
        
        if len(self.histogram_history) >= 2:
            curr_hist = self.histogram_history[-1]
            prev_hist = self.histogram_history[-2]
            
            # Check for zero line crossover
            if zero_line_cross:
                if prev_hist <= 0 and curr_hist > 0:
                    signal_type_enum = SignalType.BUY
                    # Confidence based on strength of crossover
                    confidence = min(0.8, 0.5 + abs(curr_hist) * 10)
                    signal_info['condition'] = 'hist_cross_above_zero'
                    
                elif prev_hist >= 0 and curr_hist < 0:
                    signal_type_enum = SignalType.SELL
                    # Confidence based on strength of crossover
                    confidence = min(0.8, 0.5 + abs(curr_hist) * 10)
                    signal_info['condition'] = 'hist_cross_below_zero'
            
            # Check for histogram turning points
            if prev_hist < 0 and prev_hist < self.histogram_history[-3] and curr_hist > prev_hist:
                # Histogram is negative but turning upward - potential buy
                signal_type_enum = SignalType.BUY
                confidence = 0.6
                signal_info['condition'] = 'hist_turning_up_from_negative'
                
            elif prev_hist > 0 and prev_hist > self.histogram_history[-3] and curr_hist < prev_hist:
                # Histogram is positive but turning downward - potential sell
                signal_type_enum = SignalType.SELL
                confidence = 0.6
                signal_info['condition'] = 'hist_turning_down_from_positive'
            
            # Check for divergence
            if use_divergence and len(self.price_highs) >= 2 and len(self.hist_highs) >= 2:
                # Bearish divergence: Price making higher highs but histogram making lower highs
                if self.price_highs[-1][1] > self.price_highs[-2][1] and self.hist_highs[-1][1] < self.hist_highs[-2][1]:
                    signal_type_enum = SignalType.SELL
                    confidence = 0.8  # High confidence for divergence signals
                    signal_info['condition'] = 'bearish_divergence'
                    
            if use_divergence and len(self.price_lows) >= 2 and len(self.hist_lows) >= 2:
                # Bullish divergence: Price making lower lows but histogram making higher lows
                if self.price_lows[-1][1] < self.price_lows[-2][1] and self.hist_lows[-1][1] > self.hist_lows[-2][1]:
                    signal_type_enum = SignalType.BUY
                    confidence = 0.8  # High confidence for divergence signals
                    signal_info['condition'] = 'bullish_divergence'
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'macd': macd,
                'signal': self.signal_line,
                'histogram': histogram,
                **signal_info
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=self.params['slow_period'] * 3)
        self.fast_ema = None
        self.slow_ema = None
        self.macd_history = deque(maxlen=self.params['signal_period'] * 3)
        self.signal_line = None
        self.histogram_history = deque(maxlen=20)
        self.price_highs = deque(maxlen=5)
        self.price_lows = deque(maxlen=5)
        self.hist_highs = deque(maxlen=5)
        self.hist_lows = deque(maxlen=5)
        self.current_signal_type = SignalType.NEUTRAL
