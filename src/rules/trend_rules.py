"""
Trend Rules Module

This module implements various trend-based trading rules such as
ADX rules, trend strength indicators, and directional movement.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque

from src.signals import Signal, SignalType
from src.rules.rule_base import Rule
from src.rules.rule_registry import register_rule


@register_rule(category="trend")
class ADXRule(Rule):
    """
    Average Directional Index (ADX) Rule.
    
    This rule generates signals based on the ADX indicator and the directional movement indicators
    (+DI and -DI) to identify strong trends and their direction.
    """
    
    def __init__(self, 
                 name: str = "adx_rule", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "ADX trend strength rule"):
        """
        Initialize the ADX rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - adx_period: Period for ADX calculation (default: 14)
                - adx_threshold: Threshold to consider a trend strong (default: 25)
                - use_di_cross: Whether to use DI crossovers for signals (default: True)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=self.params['adx_period'] * 3)
        self.low_history = deque(maxlen=self.params['adx_period'] * 3)
        self.close_history = deque(maxlen=self.params['adx_period'] * 3)
        self.tr_history = deque(maxlen=self.params['adx_period'])
        self.plus_dm_history = deque(maxlen=self.params['adx_period'])
        self.minus_dm_history = deque(maxlen=self.params['adx_period'])
        self.adx_history = deque(maxlen=self.params['adx_period'])
        self.plus_di_history = deque(maxlen=5)
        self.minus_di_history = deque(maxlen=5)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'adx_period': 14,
            'adx_threshold': 25,
            'use_di_cross': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['adx_period'] <= 0:
            raise ValueError("ADX period must be positive")
        
        if self.params['adx_threshold'] <= 0:
            raise ValueError("ADX threshold must be positive")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on ADX and directional movement.
        
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
        adx_period = self.params['adx_period']
        adx_threshold = self.params['adx_threshold']
        use_di_cross = self.params['use_di_cross']
        
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
        
        # Calculate +DM and -DM
        prev_high = self.high_history[-2]
        prev_low = self.low_history[-2]
        
        plus_dm = max(0, high_val - prev_high)
        minus_dm = max(0, prev_low - low_val)
        
        if plus_dm > minus_dm:
            minus_dm = 0
        elif minus_dm > plus_dm:
            plus_dm = 0
            
        self.plus_dm_history.append(plus_dm)
        self.minus_dm_history.append(minus_dm)
        
        # Need enough history to calculate ADX
        if len(self.tr_history) < adx_period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate smoothed TR, +DM, and -DM
        tr_sum = sum(self.tr_history)
        plus_dm_sum = sum(self.plus_dm_history)
        minus_dm_sum = sum(self.minus_dm_history)
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_sum / tr_sum if tr_sum > 0 else 0
        minus_di = 100 * minus_dm_sum / tr_sum if tr_sum > 0 else 0
        
        self.plus_di_history.append(plus_di)
        self.minus_di_history.append(minus_di)
        
        # Calculate DX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * di_diff / di_sum if di_sum > 0 else 0
        
        # Calculate ADX (smoothed DX)
        self.adx_history.append(dx)
        adx = sum(self.adx_history) / len(self.adx_history)
        
        # Generate signal
        signal_type = self.current_signal_type
        
        if use_di_cross and len(self.plus_di_history) >= 2 and len(self.minus_di_history) >= 2:
            # Check for DI crossovers
            current_plus_di = self.plus_di_history[-1]
            current_minus_di = self.minus_di_history[-1]
            prev_plus_di = self.plus_di_history[-2]
            prev_minus_di = self.minus_di_history[-2]
            
            if prev_plus_di <= prev_minus_di and current_plus_di > current_minus_di:
                signal_type = SignalType.BUY
            elif prev_plus_di >= prev_minus_di and current_plus_di < current_minus_di:
                signal_type = SignalType.SELL
        else:
            # Use ADX and DI values for signals
            if adx > adx_threshold:
                if plus_di > minus_di:
                    signal_type = SignalType.BUY
                else:
                    signal_type = SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL
                
        # Calculate confidence based on ADX strength and DI difference
        adx_confidence = min(1.0, adx / 50)  # ADX of 50+ = full confidence
        di_diff_confidence = min(1.0, di_diff / 40)  # DI diff of 40+ = full confidence
        confidence = (adx_confidence + di_diff_confidence) / 2
        
        self.current_signal_type = signal_type
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'trend_strength': 'strong' if adx > adx_threshold else 'weak'
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=self.params['adx_period'] * 3)
        self.low_history = deque(maxlen=self.params['adx_period'] * 3)
        self.close_history = deque(maxlen=self.params['adx_period'] * 3)
        self.tr_history = deque(maxlen=self.params['adx_period'])
        self.plus_dm_history = deque(maxlen=self.params['adx_period'])
        self.minus_dm_history = deque(maxlen=self.params['adx_period'])
        self.adx_history = deque(maxlen=self.params['adx_period'])
        self.plus_di_history = deque(maxlen=5)
        self.minus_di_history = deque(maxlen=5)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="trend")
class IchimokuRule(Rule):
    """
    Ichimoku Cloud Rule.
    
    This rule generates signals based on the Ichimoku Cloud indicator,
    which provides multiple signals including trend direction, support/resistance,
    and momentum.
    """
    
    def __init__(self, 
                 name: str = "ichimoku_rule", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Ichimoku Cloud rule"):
        """
        Initialize the Ichimoku Cloud rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - tenkan_period: Tenkan-sen period (default: 9)
                - kijun_period: Kijun-sen period (default: 26)
                - senkou_span_b_period: Senkou Span B period (default: 52)
                - signal_type: Signal generation method ('cloud', 'tk_cross', 'price_cross') (default: 'cloud')
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=max(self.params['tenkan_period'], 
                                            self.params['kijun_period'], 
                                            self.params['senkou_span_b_period']) * 2)
        self.low_history = deque(maxlen=max(self.params['tenkan_period'], 
                                          self.params['kijun_period'], 
                                          self.params['senkou_span_b_period']) * 2)
        self.close_history = deque(maxlen=max(self.params['tenkan_period'], 
                                            self.params['kijun_period'], 
                                            self.params['senkou_span_b_period']) * 2)
        self.tenkan_history = deque(maxlen=10)
        self.kijun_history = deque(maxlen=10)
        self.senkou_a_history = deque(maxlen=10)
        self.senkou_b_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_span_b_period': 52,
            'signal_type': 'cloud'  # 'cloud', 'tk_cross', or 'price_cross'
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        for param in ['tenkan_period', 'kijun_period', 'senkou_span_b_period']:
            if self.params[param] <= 0:
                raise ValueError(f"{param} must be positive")
                
        valid_signal_types = ['cloud', 'tk_cross', 'price_cross']
        if self.params['signal_type'] not in valid_signal_types:
            raise ValueError(f"signal_type must be one of {valid_signal_types}")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on Ichimoku Cloud.
        
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
        tenkan_period = self.params['tenkan_period']
        kijun_period = self.params['kijun_period']
        senkou_span_b_period = self.params['senkou_span_b_period']
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
        
        # Need enough history for calculations
        min_periods = max(tenkan_period, kijun_period, senkou_span_b_period)
        if len(self.high_history) < min_periods:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate Tenkan-sen (Conversion Line)
        highest_high_tenkan = max(list(self.high_history)[-tenkan_period:])
        lowest_low_tenkan = min(list(self.low_history)[-tenkan_period:])
        tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2
        self.tenkan_history.append(tenkan_sen)
        
        # Calculate Kijun-sen (Base Line)
        highest_high_kijun = max(list(self.high_history)[-kijun_period:])
        lowest_low_kijun = min(list(self.low_history)[-kijun_period:])
        kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2
        self.kijun_history.append(kijun_sen)
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        self.senkou_a_history.append(senkou_span_a)
        
        # Calculate Senkou Span B (Leading Span B)
        highest_high_senkou_b = max(list(self.high_history)[-senkou_span_b_period:])
        lowest_low_senkou_b = min(list(self.low_history)[-senkou_span_b_period:])
        senkou_span_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
        self.senkou_b_history.append(senkou_span_b)
        
        # Generate signal based on selected method
        signal_type_enum = SignalType.NEUTRAL
        confidence = 0.5
        signal_info = {}
        
        # Check which signal type to use
        if signal_type == 'cloud':
            # Use price position relative to cloud
            if close > max(senkou_span_a, senkou_span_b):
                signal_type_enum = SignalType.BUY
                distance = (close - max(senkou_span_a, senkou_span_b)) / close
                confidence = min(1.0, distance * 10 + 0.5)  # Base 0.5 + distance factor
                signal_info['cloud_position'] = 'above'
            elif close < min(senkou_span_a, senkou_span_b):
                signal_type_enum = SignalType.SELL
                distance = (min(senkou_span_a, senkou_span_b) - close) / close
                confidence = min(1.0, distance * 10 + 0.5)  # Base 0.5 + distance factor
                signal_info['cloud_position'] = 'below'
            else:
                # Price is inside the cloud (indecision)
                signal_type_enum = SignalType.NEUTRAL
                signal_info['cloud_position'] = 'inside'
                confidence = 0.3  # Low confidence in the cloud
                
        elif signal_type == 'tk_cross' and len(self.tenkan_history) >= 2 and len(self.kijun_history) >= 2:
            # Use Tenkan/Kijun cross
            current_tenkan = self.tenkan_history[-1]
            current_kijun = self.kijun_history[-1]
            prev_tenkan = self.tenkan_history[-2]
            prev_kijun = self.kijun_history[-2]
            
            if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
                signal_type_enum = SignalType.BUY
                signal_info['cross_type'] = 'tenkan_above_kijun'
                confidence = 0.7
            elif prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
                signal_type_enum = SignalType.SELL
                signal_info['cross_type'] = 'tenkan_below_kijun'
                confidence = 0.7
            else:
                # No cross, use current position
                signal_type_enum = SignalType.BUY if current_tenkan > current_kijun else \
                                  SignalType.SELL if current_tenkan < current_kijun else \
                                  SignalType.NEUTRAL
                signal_info['current_position'] = 'tenkan_above_kijun' if current_tenkan > current_kijun else \
                                                 'tenkan_below_kijun' if current_tenkan < current_kijun else \
                                                 'equal'
                confidence = 0.4  # Lower confidence without a cross
                
        elif signal_type == 'price_cross':
            # Use price crossing the Kijun-sen
            if len(self.close_history) >= 2 and len(self.kijun_history) >= 2:
                prev_close = self.close_history[-2]
                prev_kijun = self.kijun_history[-2]
                
                if prev_close <= prev_kijun and close > kijun_sen:
                    signal_type_enum = SignalType.BUY
                    signal_info['cross_type'] = 'price_above_kijun'
                    confidence = 0.8
                elif prev_close >= prev_kijun and close < kijun_sen:
                    signal_type_enum = SignalType.SELL
                    signal_info['cross_type'] = 'price_below_kijun'
                    confidence = 0.8
                else:
                    # No cross, use current position
                    signal_type_enum = SignalType.BUY if close > kijun_sen else \
                                      SignalType.SELL if close < kijun_sen else \
                                      SignalType.NEUTRAL
                    signal_info['current_position'] = 'price_above_kijun' if close > kijun_sen else \
                                                     'price_below_kijun' if close < kijun_sen else \
                                                     'equal'
                    confidence = 0.5
        
        self.current_signal_type = signal_type_enum
        
        # Add calculated values to metadata
        metadata = {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'cloud_direction': 'bullish' if senkou_span_a > senkou_span_b else 'bearish',
            **signal_info
        }
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type_enum,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata=metadata
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=max(self.params['tenkan_period'], 
                                            self.params['kijun_period'], 
                                            self.params['senkou_span_b_period']) * 2)
        self.low_history = deque(maxlen=max(self.params['tenkan_period'], 
                                          self.params['kijun_period'], 
                                          self.params['senkou_span_b_period']) * 2)
        self.close_history = deque(maxlen=max(self.params['tenkan_period'], 
                                            self.params['kijun_period'], 
                                            self.params['senkou_span_b_period']) * 2)
        self.tenkan_history = deque(maxlen=10)
        self.kijun_history = deque(maxlen=10)
        self.senkou_a_history = deque(maxlen=10)
        self.senkou_b_history = deque(maxlen=10)
        self.current_signal_type = SignalType.NEUTRAL


@register_rule(category="trend")
class VortexRule(Rule):
    """
    Vortex Indicator Rule.
    
    This rule uses the Vortex Indicator (VI) to identify trend reversals based on
    the relationship between the positive and negative VI lines.
    """
    
    def __init__(self, 
                 name: str = "vortex_rule", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Vortex indicator rule"):
        """
        Initialize the Vortex Indicator rule.
        
        Args:
            name: Rule name
            params: Dictionary containing:
                - period: Calculation period for VI+ and VI- (default: 14)
                - smooth_signals: Whether to generate signals when VIs are aligned (default: True)
            description: Rule description
        """
        super().__init__(name, params or self.default_params(), description)
        self.high_history = deque(maxlen=self.params['period'] * 3)
        self.low_history = deque(maxlen=self.params['period'] * 3)
        self.close_history = deque(maxlen=self.params['period'] * 3)
        self.tr_history = deque(maxlen=self.params['period'])
        self.plus_vm_history = deque(maxlen=self.params['period'])
        self.minus_vm_history = deque(maxlen=self.params['period'])
        self.vi_plus_history = deque(maxlen=5)
        self.vi_minus_history = deque(maxlen=5)
        self.current_signal_type = SignalType.NEUTRAL
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """Default parameters for the rule."""
        return {
            'period': 14,
            'smooth_signals': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this rule."""
        if self.params['period'] <= 0:
            raise ValueError("Period must be positive")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on the Vortex Indicator.
        
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
        smooth_signals = self.params['smooth_signals']
        
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
        
        # Calculate +VM and -VM (Vortex Movements)
        prev_high = self.high_history[-2]
        prev_low = self.low_history[-2]
        
        plus_vm = abs(high_val - prev_low)
        minus_vm = abs(low_val - prev_high)
        
        self.plus_vm_history.append(plus_vm)
        self.minus_vm_history.append(minus_vm)
        
        # Need enough history to calculate Vortex Indicators
        if len(self.tr_history) < period:
            return Signal(
                timestamp=timestamp,
                signal_type=SignalType.NEUTRAL,
                price=close,
                rule_id=self.name,
                confidence=0.0,
                metadata={'status': 'collecting data'}
            )
        
        # Calculate VI+ and VI-
        tr_sum = sum(list(self.tr_history)[-period:])
        plus_vm_sum = sum(list(self.plus_vm_history)[-period:])
        minus_vm_sum = sum(list(self.minus_vm_history)[-period:])
        
        vi_plus = plus_vm_sum / tr_sum if tr_sum > 0 else 0
        vi_minus = minus_vm_sum / tr_sum if tr_sum > 0 else 0
        
        self.vi_plus_history.append(vi_plus)
        self.vi_minus_history.append(vi_minus)
        
        # Generate signal
        signal_type = self.current_signal_type
        
        if len(self.vi_plus_history) >= 2 and len(self.vi_minus_history) >= 2:
            current_vi_plus = self.vi_plus_history[-1]
            current_vi_minus = self.vi_minus_history[-1]
            prev_vi_plus = self.vi_plus_history[-2]
            prev_vi_minus = self.vi_minus_history[-2]
            
            # Check for crossovers
            if prev_vi_plus <= prev_vi_minus and current_vi_plus > current_vi_minus:
                signal_type = SignalType.BUY
            elif prev_vi_plus >= prev_vi_minus and current_vi_plus < current_vi_minus:
                signal_type = SignalType.SELL
            elif smooth_signals:
                # If smooth signals are enabled, use relationship between indicators
                if current_vi_plus > current_vi_minus:
                    signal_type = SignalType.BUY
                elif current_vi_plus < current_vi_minus:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.NEUTRAL
            else:
                # Otherwise, revert to neutral after crossover
                signal_type = SignalType.NEUTRAL
        
        # Calculate confidence based on distance between VI+ and VI-
        confidence = 0.5
        if len(self.vi_plus_history) > 0 and len(self.vi_minus_history) > 0:
            current_vi_plus = self.vi_plus_history[-1]
            current_vi_minus = self.vi_minus_history[-1]
            distance = abs(current_vi_plus - current_vi_minus)
            confidence = min(1.0, distance * 5 + 0.3)  # Scale distance for confidence
            
        self.current_signal_type = signal_type
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=close,
            rule_id=self.name,
            confidence=confidence,
            metadata={
                'vi_plus': vi_plus,
                'vi_minus': vi_minus,
                'ratio': vi_plus / vi_minus if vi_minus != 0 else float('inf')
            }
        )
    
    def reset(self) -> None:
        """Reset the rule's internal state."""
        super().reset()
        self.high_history = deque(maxlen=self.params['period'] * 3)
        self.low_history = deque(maxlen=self.params['period'] * 3)
        self.close_history = deque(maxlen=self.params['period'] * 3)
        self.tr_history = deque(maxlen=self.params['period'])
        self.plus_vm_history = deque(maxlen=self.params['period'])
        self.minus_vm_history = deque(maxlen=self.params['period'])
        self.vi_plus_history = deque(maxlen=5)
        self.vi_minus_history = deque(maxlen=5)
        self.current_signal_type = SignalType.NEUTRAL
