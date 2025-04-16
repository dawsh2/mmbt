"""
Volatility Rules Module

This module implements various volatility-based trading rules such as
Bollinger Bands, ATR strategies, and volatility breakout systems.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque

from src.signals import Signal, SignalType
from src.rules.rule_base import Rule
from src.rules.rule_registry import register_rule



@register_rule(category="volatility")
class BollingerBandRule(Rule): # BollingerBreakout
    """
    Bollinger Bands Breakout Rule.
    
    This rule generates signals when price breaks out of Bollinger Bands,
    indicating potential trend reversals or continuation based on volatility.
    """
    
    def __init__(self, 
                 name: str = "bollinger_breakout", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Bollinger Bands breakout rule"):
        """
        Initialize the Bollinger Bands Breakout rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - period: Period for SMA calculation (default: 20)
                - std_dev: Number of standard deviations for bands (default: 2.0)
                - breakout_type: Type of breakout to monitor ('upper', 'lower', 'both') (default: 'both')
                - use_confirmations: Whether to require confirmation (default: True)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.prices = deque(maxlen=self.params['period'] * 3)
        self.upper_band_history = deque(maxlen=10)
        self.middle_band_history = deque(maxlen=10)
        self.lower_band_history = deque(maxlen=10)
        self.breakout_detected = False
        self.breakout_direction = 0  # 1 for upper, -1 for lower
        self.confirmation_bar = False
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'period': 20,
            'std_dev': 2.0,
            'breakout_type': 'both',  # 'upper', 'lower', or 'both'
            'use_confirmations': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['period'] <= 0:
            raise ValueError("Period must be positive")
        
        if self.params['std_dev'] <= 0:
            raise ValueError("Standard deviation must be positive")
            
        valid_breakout_types = ['upper', 'lower', 'both']
        if self.params['breakout_type'] not in valid_breakout_types:
            raise ValueError(f"Breakout type must be one of {valid_breakout_types}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on Bollinger Bands breakout.
        
        Args:
            data: Dictionary containing price data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Check for required data
        if not all(k in data for k in ['Close', 'High', 'Low']):
            return Signal(
                timestamp=data.get('timestamp', None),
                signal_type=SignalType.NEUTRAL,
                price=None,
                rule_id=self.name,
                confidence=0.0,
                metadata={'error': 'Missing required price data'}
            )
            
        # Get parameters
        period = self.params['period']
        std_dev = self.params['std_dev']
        breakout_type = self.params['breakout_type']
        use_confirmations = self.params['use_confirmations']
        
        # Extract price data
        close = data['Close']
        high = data['High']
        low = data['Low']
        timestamp = data.get('timestamp', None)
        symbol = data.get('symbol', 'default')
        
        # Update price history
        self.prices.append(close)
        
        # Need enough history to calculate Bollinger Bands
        if len(self.prices) < period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={
                    'status': 'collecting data',
                    'symbol': symbol
                }
            )
        
        # Calculate SMA (middle band)
        middle_band = sum(list(self.prices)[-period:]) / period
        
        # Calculate standard deviation
        variance = sum((price - middle_band) ** 2 for price in list(self.prices)[-period:]) / period
        std = np.sqrt(variance)
        
        # Calculate bands
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # Store band history
        self.upper_band_history.append(upper_band)
        self.middle_band_history.append(middle_band)
        self.lower_band_history.append(lower_band)
        
        # Initialize signal variables
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        signal_info = {}
        
        # Check for breakouts based on specified type
        if breakout_type in ['upper', 'both'] and high > upper_band:
            # Upper band breakout detected
            if not self.breakout_detected or self.breakout_direction != 1:
                self.breakout_detected = True
                self.breakout_direction = 1
                self.confirmation_bar = use_confirmations
                signal_info['breakout'] = 'upper'
                
                # Calculate confidence based on breakout strength
                breakout_strength = (high - upper_band) / upper_band
                confidence = min(0.9, 0.6 + breakout_strength * 2)
                
                if not use_confirmations:
                    signal_type_enum = SignalType.SELL
                
        elif breakout_type in ['lower', 'both'] and low < lower_band:
            # Lower band breakout detected
            if not self.breakout_detected or self.breakout_direction != -1:
                self.breakout_detected = True
                self.breakout_direction = -1
                self.confirmation_bar = use_confirmations
                signal_info['breakout'] = 'lower'
                
                # Calculate confidence based on breakout strength
                breakout_strength = (lower_band - low) / lower_band
                confidence = min(0.9, 0.6 + breakout_strength * 2)
                
                if not use_confirmations:
                    signal_type_enum = SignalType.BUY
                
        # Handle confirmation if needed
        if self.confirmation_bar:
            if self.breakout_direction == 1:
                # Confirm upper breakout - check if price closes above middle band
                if close > middle_band:
                    signal_type_enum = SignalType.SELL
                    signal_info['confirmation'] = 'confirmed_upper'
                else:
                    # Failed confirmation
                    signal_info['confirmation'] = 'failed_upper'
                    self.breakout_detected = False
                    
            elif self.breakout_direction == -1:
                # Confirm lower breakout - check if price closes below middle band
                if close < middle_band:
                    signal_type_enum = SignalType.BUY
                    signal_info['confirmation'] = 'confirmed_lower'
                else:
                    # Failed confirmation
                    signal_info['confirmation'] = 'failed_lower'
                    self.breakout_detected = False
                    
            # Reset confirmation flag
            self.confirmation_bar = False
        
        # Reset breakout if price returns to within bands
        if self.breakout_detected and lower_band < close < upper_band:
            signal_info['status'] = 'returned_to_bands'
            self.breakout_detected = False
            
        # Calculate relative position within bands (percent B)
        band_width = upper_band - lower_band
        if band_width > 0:
            percent_b = (close - lower_band) / band_width
            signal_info['percent_b'] = percent_b
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'std_dev': std,
                'symbol': symbol,  # Add symbol to metadata
                **signal_info
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.prices = deque(maxlen=self.params['period'] * 3)
        self.upper_band_history = deque(maxlen=10)
        self.middle_band_history = deque(maxlen=10)
        self.lower_band_history = deque(maxlen=10)
        self.breakout_detected = False
        self.breakout_direction = 0
        self.confirmation_bar = False
        self.current_signal_type = SignalType.NEUTRAL

@register_rule(category="volatility")
class ATRTrailingStopRule(Rule):
    """
    Average True Range (ATR) Trailing Stop Rule.
    
    This rule uses ATR to set dynamic trailing stops that adapt to market volatility,
    providing a strategy for managing exits.
    """
    
    def __init__(self, 
                 name: str = "atr_trailing_stop", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "ATR trailing stop rule"):
        """
        Initialize the ATR Trailing Stop rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - atr_period: Period for ATR calculation (default: 14)
                - atr_multiplier: Multiplier for ATR to set stop distance (default: 3.0)
                - use_trend_filter: Whether to use trend filter for entries (default: True)
                - trend_ma_period: Period for trend moving average (default: 50)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=max(self.params['atr_period'], self.params.get('trend_ma_period', 50)) * 2)
        self.low_history = deque(maxlen=max(self.params['atr_period'], self.params.get('trend_ma_period', 50)) * 2)
        self.close_history = deque(maxlen=max(self.params['atr_period'], self.params.get('trend_ma_period', 50)) * 2)
        self.tr_history = deque(maxlen=self.params['atr_period'])
        self.atr_history = deque(maxlen=10)
        self.trailing_stop = None
        self.position = 0  # 0=none, 1=long, -1=short
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'atr_period': 14,
            'atr_multiplier': 3.0,
            'use_trend_filter': True,
            'trend_ma_period': 50
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['atr_period'] <= 0:
            raise ValueError("ATR period must be positive")
            
        if self.params['atr_multiplier'] <= 0:
            raise ValueError("ATR multiplier must be positive")
            
        if self.params['use_trend_filter'] and self.params.get('trend_ma_period', 0) <= 0:
            raise ValueError("Trend MA period must be positive when trend filter is enabled")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on ATR trailing stop.
        
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
        atr_period = self.params['atr_period']
        atr_multiplier = self.params['atr_multiplier']
        use_trend_filter = self.params['use_trend_filter']
        trend_ma_period = self.params.get('trend_ma_period', 50)
        
        # Extract price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Need at least 2 bars to start calculations
        if len(self.high_history) < 2:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'initializing'}
            )
        
        # Calculate True Range (TR)
        high_val = self.high_history[-1]
        low_val = self.low_history[-1]
        prev_close = self.close_history[-2]
        
        tr = max(
            high_val - low_val,
            abs(high_val - prev_close),
            abs(low_val - prev_close)
        )
        self.tr_history.append(tr)
        
        # Calculate ATR
        if len(self.tr_history) < atr_period:
            # Not enough data for ATR calculation
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        atr = sum(self.tr_history) / len(self.tr_history)
        self.atr_history.append(atr)
        
        # Determine trend direction if using trend filter
        trend = 0  # 0=no trend, 1=uptrend, -1=downtrend
        
        if use_trend_filter and len(self.close_history) >= trend_ma_period:
            ma = sum(list(self.close_history)[-trend_ma_period:]) / trend_ma_period
            trend = 1 if close > ma else -1
        elif not use_trend_filter:
            # If not using trend filter, use recent price action for trend
            if len(self.close_history) >= 3:
                # Simple trend detection
                if self.close_history[-1] > self.close_history[-2] > self.close_history[-3]:
                    trend = 1
                elif self.close_history[-1] < self.close_history[-2] < self.close_history[-3]:
                    trend = -1
        
        # Calculate ATR-based stop
        atr_band = atr * atr_multiplier
        
        # Signal logic based on position and trailing stop
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        
        if self.position == 0:  # Not in a position
            # Enter position in the direction of the trend
            if trend > 0:
                self.position = 1
                self.trailing_stop = close - atr_band
                signal_type_enum = SignalType.BUY
                confidence = 0.7
            elif trend < 0:
                self.position = -1
                self.trailing_stop = close + atr_band
                signal_type_enum = SignalType.SELL
                confidence = 0.7
        
        elif self.position == 1:  # Long position
            # Update trailing stop if price moves higher
            new_stop = close - atr_band
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
            
            # Exit if price falls below trailing stop
            if low <= self.trailing_stop:
                signal_type_enum = SignalType.NEUTRAL  # Exit signal
                confidence = 0.8  # High confidence for stop exits
                self.position = 0
                self.trailing_stop = None
            else:
                signal_type_enum = SignalType.BUY  # Maintain long signal
                stop_distance = (close - self.trailing_stop) / close if close > 0 else 0
                confidence = min(0.9, 0.5 + stop_distance * 2)  # Confidence based on distance to stop
        
        elif self.position == -1:  # Short position
            # Update trailing stop if price moves lower
            new_stop = close + atr_band
            if new_stop < self.trailing_stop or self.trailing_stop is None:
                self.trailing_stop = new_stop
            
            # Exit if price rises above trailing stop
            if high >= self.trailing_stop:
                signal_type_enum = SignalType.NEUTRAL  # Exit signal
                confidence = 0.8  # High confidence for stop exits
                self.position = 0
                self.trailing_stop = None
            else:
                signal_type_enum = SignalType.SELL  # Maintain short signal
                stop_distance = (self.trailing_stop - close) / close if close > 0 else 0
                confidence = min(0.9, 0.5 + stop_distance * 2)  # Confidence based on distance to stop
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'atr': atr,
                'trailing_stop': self.trailing_stop,
                'position': 'long' if self.position == 1 else 'short' if self.position == -1 else 'none',
                'trend': 'up' if trend == 1 else 'down' if trend == -1 else 'none'
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=max(self.params['atr_period'], self.params.get('trend_ma_period', 50)) * 2)
        self.low_history = deque(maxlen=max(self.params['atr_period'], self.params.get('trend_ma_period', 50)) * 2)
        self.close_history = deque(maxlen=max(self.params['atr_period'], self.params.get('trend_ma_period', 50)) * 2)
        self.tr_history = deque(maxlen=self.params['atr_period'])
        self.atr_history = deque(maxlen=10)
        self.trailing_stop = None
        self.position = 0
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="volatility")
class VolatilityBreakoutRule(Rule):
    """
    Volatility Breakout Rule.
    
    This rule generates signals based on price breaking out of a volatility-adjusted range,
    which can identify the start of new trends after periods of consolidation.
    """
    
    def __init__(self, 
                 name: str = "volatility_breakout", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Volatility breakout rule"):
        """
        Initialize the Volatility Breakout rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - lookback_period: Period for calculating the range (default: 20)
                - volatility_measure: Method to measure volatility ('atr', 'stdev', 'range') (default: 'atr')
                - breakout_multiplier: Multiplier for breakout threshold (default: 1.5)
                - require_confirmation: Whether to require confirmation (default: True)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=self.params['lookback_period'] * 2)
        self.low_history = deque(maxlen=self.params['lookback_period'] * 2)
        self.close_history = deque(maxlen=self.params['lookback_period'] * 2)
        self.range_history = deque(maxlen=self.params['lookback_period'])
        self.volatility_history = deque(maxlen=10)
        self.breakout_level_high = None
        self.breakout_level_low = None
        self.pending_signal = None
        self.confirmation_bar = False
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'lookback_period': 20,
            'volatility_measure': 'atr',  # 'atr', 'stdev', or 'range'
            'breakout_multiplier': 1.5,
            'require_confirmation': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['lookback_period'] <= 0:
            raise ValueError("Lookback period must be positive")
            
        if self.params['breakout_multiplier'] <= 0:
            raise ValueError("Breakout multiplier must be positive")
            
        valid_volatility_measures = ['atr', 'stdev', 'range']
        if self.params['volatility_measure'] not in valid_volatility_measures:
            raise ValueError(f"Volatility measure must be one of {valid_volatility_measures}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on volatility breakout.
        
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
        lookback_period = self.params['lookback_period']
        volatility_measure = self.params['volatility_measure']
        breakout_multiplier = self.params['breakout_multiplier']
        require_confirmation = self.params['require_confirmation']
        
        # Extract price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        timestamp = data.get('timestamp', None)
        
        # Update price history
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Calculate daily range and add to history
        if len(self.high_history) > 0 and len(self.low_history) > 0:
            daily_range = self.high_history[-1] - self.low_history[-1]
            self.range_history.append(daily_range)
        
        # Need enough history for volatility calculation
        if len(self.high_history) < lookback_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate volatility based on selected measure
        volatility = 0
        
        if volatility_measure == 'atr':
            # Calculate ATR
            tr_values = []
            for i in range(1, len(self.high_history)):
                high_val = self.high_history[-i]
                low_val = self.low_history[-i]
                prev_close = self.close_history[-(i+1)]
                
                tr = max(
                    high_val - low_val,
                    abs(high_val - prev_close),
                    abs(low_val - prev_close)
                )
                tr_values.append(tr)
                
                if len(tr_values) >= lookback_period:
                    break
                    
            volatility = sum(tr_values) / len(tr_values) if tr_values else 0
            
        elif volatility_measure == 'stdev':
            # Calculate standard deviation of closes
            prices = list(self.close_history)[-lookback_period:]
            mean_price = sum(prices) / len(prices)
            squared_diffs = [(price - mean_price) ** 2 for price in prices]
            variance = sum(squared_diffs) / len(squared_diffs)
            volatility = np.sqrt(variance)
            
        elif volatility_measure == 'range':
            # Use average daily range
            ranges = list(self.range_history)[-lookback_period:]
            volatility = sum(ranges) / len(ranges) if ranges else 0
        
        self.volatility_history.append(volatility)
        
        # Calculate breakout levels
        if self.breakout_level_high is None or self.breakout_level_low is None:
            # Initialize breakout levels
            max_high = max(list(self.high_history)[-lookback_period:])
            min_low = min(list(self.low_history)[-lookback_period:])
            
            self.breakout_level_high = max_high + (volatility * breakout_multiplier)
            self.breakout_level_low = min_low - (volatility * breakout_multiplier)
        else:
            # Update breakout levels if current high/low exceed existing levels
            current_high = self.high_history[-1]
            current_low = self.low_history[-1]
            
            if current_high > self.breakout_level_high:
                self.breakout_level_high = current_high + (volatility * breakout_multiplier * 0.5)  # Reduced multiplier for updates
                
            if current_low < self.breakout_level_low:
                self.breakout_level_low = current_low - (volatility * breakout_multiplier * 0.5)  # Reduced multiplier for updates
        
        # Generate signal
        signal_type_enum = self.current_signal_type
        confidence = 0.5
        
        # Check for breakouts
        if high > self.breakout_level_high:
            # Bullish breakout detected
            if require_confirmation:
                # Set pending signal, need confirmation
                self.pending_signal = SignalType.BUY
                self.confirmation_bar = True
                signal_type_enum = SignalType.NEUTRAL  # Stay neutral until confirmed
                confidence = 0.3  # Low confidence without confirmation
            else:
                # Immediate signal
                signal_type_enum = SignalType.BUY
                # Calculate confidence based on breakout strength
                breakout_strength = (high - self.breakout_level_high) / volatility if volatility > 0 else 0
                confidence = min(0.9, 0.6 + breakout_strength * 0.2)
                
        elif low < self.breakout_level_low:
            # Bearish breakout detected
            if require_confirmation:
                # Set pending signal, need confirmation
                self.pending_signal = SignalType.SELL
                self.confirmation_bar = True
                signal_type_enum = SignalType.NEUTRAL  # Stay neutral until confirmed
                confidence = 0.3  # Low confidence without confirmation
            else:
                # Immediate signal
                signal_type_enum = SignalType.SELL
                # Calculate confidence based on breakout strength
                breakout_strength = (self.breakout_level_low - low) / volatility if volatility > 0 else 0
                confidence = min(0.9, 0.6 + breakout_strength * 0.2)
                
        elif self.confirmation_bar and self.pending_signal is not None:
            # This is the confirmation bar after a breakout
            self.confirmation_bar = False
            
            if self.pending_signal == SignalType.BUY and close > self.high_history[-2]:
                # Confirmed bullish breakout
                signal_type_enum = SignalType.BUY
                # Calculate confidence based on confirmation strength
                confirmation_strength = (close - self.high_history[-2]) / volatility if volatility > 0 else 0
                confidence = min(0.9, 0.7 + confirmation_strength * 0.2)
                
            elif self.pending_signal == SignalType.SELL and close < self.low_history[-2]:
                # Confirmed bearish breakout
                signal_type_enum = SignalType.SELL
                # Calculate confidence based on confirmation strength
                confirmation_strength = (self.low_history[-2] - close) / volatility if volatility > 0 else 0
                confidence = min(0.9, 0.7 + confirmation_strength * 0.2)
                
            else:
                # Failed confirmation
                signal_type_enum = SignalType.NEUTRAL
                confidence = 0.4
                
            self.pending_signal = None
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'volatility': volatility,
                'breakout_high': self.breakout_level_high,
                'breakout_low': self.breakout_level_low,
                'pending_signal': 'buy' if self.pending_signal == SignalType.BUY else 
                                 'sell' if self.pending_signal == SignalType.SELL else None,
                'confirmation_needed': self.confirmation_bar
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=self.params['lookback_period'] * 2)
        self.low_history = deque(maxlen=self.params['lookback_period'] * 2)
        self.close_history = deque(maxlen=self.params['lookback_period'] * 2)
        self.range_history = deque(maxlen=self.params['lookback_period'])
        self.volatility_history = deque(maxlen=10)
        self.breakout_level_high = None
        self.breakout_level_low = None
        self.pending_signal = None
        self.confirmation_bar = False
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="volatility")
class KeltnerChannelRule(Rule):
    """
    Keltner Channel Rule.
    
    This rule uses Keltner Channels which are similar to Bollinger Bands but use ATR
    for volatility measurement instead of standard deviation.
    """
    
    def __init__(self, 
                 name: str = "keltner_channel", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Keltner Channel volatility rule"):
        """
        Initialize the Keltner Channel rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - ema_period: Period for EMA calculation (default: 20)
                - atr_period: Period for ATR calculation (default: 10)
                - multiplier: Multiplier for channels (default: 2.0)
                - signal_type: Signal generation method ('channel_cross', 'channel_touch') (default: 'channel_cross')
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=max(self.params['ema_period'], self.params['atr_period']) * 2)
        self.low_history = deque(maxlen=max(self.params['ema_period'], self.params['atr_period']) * 2)
        self.close_history = deque(maxlen=max(self.params['ema_period'], self.params['atr_period']) * 2)
        self.tr_history = deque(maxlen=self.params['atr_period'])
        self.atr_history = deque(maxlen=10)
        self.ema_history = deque(maxlen=10)
        self.upper_channel_history = deque(maxlen=10)
        self.lower_channel_history = deque(maxlen=10)
        self.position = 0  # 0=none, 1=long, -1=short
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'ema_period': 20,
            'atr_period': 10,
            'multiplier': 2.0,
            'signal_type': 'channel_cross'  # 'channel_cross' or 'channel_touch'
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['ema_period'] <= 0:
            raise ValueError("EMA period must be positive")
            
        if self.params['atr_period'] <= 0:
            raise ValueError("ATR period must be positive")
            
        if self.params['multiplier'] <= 0:
            raise ValueError("Multiplier must be positive")
            
        valid_signal_types = ['channel_cross', 'channel_touch']
        if self.params['signal_type'] not in valid_signal_types:
            raise ValueError(f"signal_type must be one of {valid_signal_types}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on Keltner Channels.
        
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
        ema_period = self.params['ema_period']
        atr_period = self.params['atr_period']
        multiplier = self.params['multiplier']
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
        
        # Need at least 2 bars to start TR calculations
        if len(self.high_history) < 2:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'initializing'}
            )
        
        # Calculate True Range (TR)
        high_val = self.high_history[-1]
        low_val = self.low_history[-1]
        prev_close = self.close_history[-2]
        
        tr = max(
            high_val - low_val,
            abs(high_val - prev_close),
            abs(low_val - prev_close)
        )
        self.tr_history.append(tr)
        
        # Need enough data for EMA and ATR calculations
        min_periods = max(ema_period, atr_period)
        if len(self.close_history) < min_periods:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate EMA for middle line
        if len(self.ema_history) == 0:
            # Initialize EMA with SMA
            ema = sum(list(self.close_history)[-ema_period:]) / ema_period
        else:
            # Update EMA
            alpha = 2 / (ema_period + 1)
            ema = (close * alpha) + (self.ema_history[-1] * (1 - alpha))
            
        self.ema_history.append(ema)
        
        # Calculate ATR
        if len(self.tr_history) >= atr_period:
            atr = sum(list(self.tr_history)[-atr_period:]) / atr_period
            self.atr_history.append(atr)
        else:
            # Not enough data for ATR yet
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting TR data'}
            )
        
        # Calculate Keltner Channels
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        
        self.upper_channel_history.append(upper_channel)
        self.lower_channel_history.append(lower_channel)
        
        # Generate signal
        signal_type_enum = self.current_signal_type
        confidence = 0.5
        
        if signal_type == 'channel_cross' and len(self.close_history) >= 2 and len(self.upper_channel_history) >= 2 and len(self.lower_channel_history) >= 2:
            # Signal on price crossing the channels
            prev_close = self.close_history[-2]
            prev_upper = self.upper_channel_history[-2]
            prev_lower = self.lower_channel_history[-2]
            
            # Bullish signal: Price was below lower channel and crossed above
            if prev_close <= prev_lower and close > lower_channel:
                signal_type_enum = SignalType.BUY
                # Calculate confidence based on how far price moved above the lower channel
                penetration = (close - lower_channel) / (ema - lower_channel) if (ema - lower_channel) > 0 else 0
                confidence = min(1.0, 0.5 + penetration)
                self.position = 1
                
            # Bearish signal: Price was above upper channel and crossed below
            elif prev_close >= prev_upper and close < upper_channel:
                signal_type_enum = SignalType.SELL
                # Calculate confidence based on how far price moved below the upper channel
                penetration = (upper_channel - close) / (upper_channel - ema) if (upper_channel - ema) > 0 else 0
                confidence = min(1.0, 0.5 + penetration)
                self.position = -1
                
            # Exit signals: Price crosses back to the middle band (EMA)
            elif self.position == 1 and close > ema:
                signal_type_enum = SignalType.NEUTRAL  # Take profit on long
                confidence = 0.6
                self.position = 0
                
            elif self.position == -1 and close < ema:
                signal_type_enum = SignalType.NEUTRAL  # Take profit on short
                confidence = 0.6
                self.position = 0
                
        elif signal_type == 'channel_touch':
            # Signal on price touching the channels
            
            # Bearish signal: Price touches or exceeds upper channel
            if high >= upper_channel:
                signal_type_enum = SignalType.SELL
                # Calculate confidence based on how far price penetrated the upper channel
                penetration = (high - upper_channel) / (upper_channel - ema) if (upper_channel - ema) > 0 else 0
                confidence = min(1.0, 0.6 + penetration)
                
            # Bullish signal: Price touches or exceeds lower channel
            elif low <= lower_channel:
                signal_type_enum = SignalType.BUY
                # Calculate confidence based on how far price penetrated the lower channel
                penetration = (lower_channel - low) / (ema - lower_channel) if (ema - lower_channel) > 0 else 0
                confidence = min(1.0, 0.6 + penetration)
                
            # Neutral when price is between channels
            else:
                signal_type_enum = SignalType.NEUTRAL
                # Calculate confidence based on position within the channels
                relative_position = (close - lower_channel) / (upper_channel - lower_channel) if (upper_channel - lower_channel) > 0 else 0.5
                confidence = 0.3  # Lower confidence when not at extremes
        
        self.current_signal_type = signal_type_enum
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'ema': ema,
                'atr': atr,
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'channel_width': upper_channel - lower_channel
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=max(self.params['ema_period'], self.params['atr_period']) * 2)
        self.low_history = deque(maxlen=max(self.params['ema_period'], self.params['atr_period']) * 2)
        self.close_history = deque(maxlen=max(self.params['ema_period'], self.params['atr_period']) * 2)
        self.tr_history = deque(maxlen=self.params['atr_period'])
        self.atr_history = deque(maxlen=10)
        self.ema_history = deque(maxlen=10)
        self.upper_channel_history = deque(maxlen=10)
        self.lower_channel_history = deque(maxlen=10)
        self.position = 0
        self.current_signal_type = SignalType.NEUTRAL                
