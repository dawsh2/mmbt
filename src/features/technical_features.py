"""
Technical Features Module

This module provides features derived from technical indicators, such as
moving average crossovers, oscillator states, and indicator divergences.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque

from src.features.feature_base import Feature, StatefulFeature
from src.features.feature_registry import register_feature



@register_feature(category="technical")
class VolatilityFeature(Feature):
    """
    Volatility measurement feature.
    
    This feature analyzes market volatility using indicators like ATR,
    Bollinger Band width, or standard deviation of returns.
    """
    
    def __init__(self, 
                 name: str = "volatility", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Market volatility measurement"):
        """
        Initialize the volatility feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - method: Method for volatility calculation ('atr', 'bb_width', 'std_dev')
                - period: Period for calculation (default: 14)
                - normalize: Whether to normalize the volatility value (default: True)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for volatility calculation."""
        return {
            'method': 'atr',
            'period': 14,
            'normalize': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_methods = ['atr', 'bb_width', 'std_dev']
        if self.params.get('method') not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate volatility metrics.
        
        Args:
            data: Dictionary containing price data and indicators
        
        Returns:
            Dictionary with volatility information
        """
        method = self.params.get('method', 'atr')
        period = self.params.get('period', 14)
        normalize = self.params.get('normalize', True)
        
        if method == 'atr':
            return self._calculate_atr(data, period, normalize)
        elif method == 'bb_width':
            return self._calculate_bb_width(data, normalize)
        elif method == 'std_dev':
            return self._calculate_std_dev(data, period, normalize)
        
        # Fallback
        return {
            'value': 0,             # Volatility value
            'is_increasing': False, # Whether volatility is increasing
            'percentile': 0,        # Current volatility percentile
            'regime': 'normal'      # Volatility regime: 'low', 'normal', 'high'
        }
    
    def _calculate_atr(self, data: Dict[str, Any], period: int, normalize: bool) -> Dict[str, Any]:
        """Calculate volatility using Average True Range."""
        # Check for required indicator
        atr_key = f'ATR_{period}' if f'ATR_{period}' in data else 'ATR'
        
        if atr_key not in data:
            return {'value': 0, 'is_increasing': False, 'percentile': 0, 'regime': 'normal'}
        
        # Extract values
        atr = data[atr_key]
        
        if not isinstance(atr, (list, np.ndarray, pd.Series)) or len(atr) < 2:
            current_atr = atr if not isinstance(atr, (list, np.ndarray, pd.Series)) else atr[-1]
            return {
                'value': current_atr,
                'is_increasing': False,
                'percentile': 0,
                'regime': 'normal'
            }
        
        current_atr = atr[-1]
        previous_atr = atr[-2]
        
        # Normalize if requested and possible
        if normalize and 'Close' in data:
            close_price = data['Close'][-1] if isinstance(data['Close'], (list, np.ndarray, pd.Series)) else data['Close']
            if close_price > 0:
                current_atr = (current_atr / close_price) * 100  # Convert to percentage of price
        
        # Determine if volatility is increasing
        is_increasing = current_atr > previous_atr
        
        # Calculate percentile if we have enough history
        percentile = 0
        if len(atr) >= 20:
            atr_history = atr[-20:]
            percentile = sum(1 for x in atr_history if x < current_atr) / len(atr_history) * 100
        
        # Determine volatility regime
        if percentile > 80:
            regime = 'high'
        elif percentile < 20:
            regime = 'low'
        else:
            regime = 'normal'
        
        return {
            'value': current_atr,
            'is_increasing': is_increasing,
            'percentile': percentile,
            'regime': regime
        }
    
    def _calculate_bb_width(self, data: Dict[str, Any], normalize: bool) -> Dict[str, Any]:
        """Calculate volatility using Bollinger Band width."""
        # Check for required indicators
        if 'BB_upper' not in data or 'BB_lower' not in data or 'BB_middle' not in data:
            return {'value': 0, 'is_increasing': False, 'percentile': 0, 'regime': 'normal'}
        
        # Extract values
        bb_upper = data['BB_upper']
        bb_lower = data['BB_lower']
        bb_middle = data['BB_middle']
        
        if not all(isinstance(x, (list, np.ndarray, pd.Series)) for x in [bb_upper, bb_lower, bb_middle]):
            # Handle scalar values
            if not isinstance(bb_upper, (list, np.ndarray, pd.Series)):
                current_upper = bb_upper
                previous_upper = bb_upper
            else:
                current_upper = bb_upper[-1]
                previous_upper = bb_upper[-2] if len(bb_upper) > 1 else bb_upper[-1]
            
            if not isinstance(bb_lower, (list, np.ndarray, pd.Series)):
                current_lower = bb_lower
                previous_lower = bb_lower
            else:
                current_lower = bb_lower[-1]
                previous_lower = bb_lower[-2] if len(bb_lower) > 1 else bb_lower[-1]
            
            if not isinstance(bb_middle, (list, np.ndarray, pd.Series)):
                current_middle = bb_middle
                previous_middle = bb_middle
            else:
                current_middle = bb_middle[-1]
                previous_middle = bb_middle[-2] if len(bb_middle) > 1 else bb_middle[-1]
        else:
            # Handle sequence values
            if len(bb_upper) < 2 or len(bb_lower) < 2 or len(bb_middle) < 2:
                return {'value': 0, 'is_increasing': False, 'percentile': 0, 'regime': 'normal'}
            
            current_upper = bb_upper[-1]
            current_lower = bb_lower[-1]
            current_middle = bb_middle[-1]
            
            previous_upper = bb_upper[-2]
            previous_lower = bb_lower[-2]
            previous_middle = bb_middle[-2]
        
        # Calculate band width
        current_width = current_upper - current_lower
        previous_width = previous_upper - previous_lower
        
        # Normalize if requested
        if normalize and current_middle > 0:
            current_width = (current_width / current_middle) * 100
            previous_width = (previous_width / previous_middle) * 100
        
        # Determine if volatility is increasing
        is_increasing = current_width > previous_width
        
        # Calculate percentile if we have enough history
        percentile = 0
        if isinstance(bb_upper, (list, np.ndarray, pd.Series)) and isinstance(bb_lower, (list, np.ndarray, pd.Series)) and len(bb_upper) >= 20 and len(bb_lower) >= 20:
            width_history = [(u - l) for u, l in zip(bb_upper[-20:], bb_lower[-20:])]
            
            if normalize and isinstance(bb_middle, (list, np.ndarray, pd.Series)) and len(bb_middle) >= 20:
                width_history = [(w / m) * 100 for w, m in zip(width_history, bb_middle[-20:])]
                
            percentile = sum(1 for x in width_history if x < current_width) / len(width_history) * 100
        
        # Determine volatility regime
        if percentile > 80:
            regime = 'high'
        elif percentile < 20:
            regime = 'low'
        else:
            regime = 'normal'
        
        return {
            'value': current_width,
            'is_increasing': is_increasing,
            'percentile': percentile,
            'regime': regime
        }
    
    def _calculate_std_dev(self, data: Dict[str, Any], period: int, normalize: bool) -> Dict[str, Any]:
        """Calculate volatility using standard deviation of returns."""
        # Check for required price data
        if 'Close' not in data:
            return {'value': 0, 'is_increasing': False, 'percentile': 0, 'regime': 'normal'}
        
        prices = data['Close']
        
        # Ensure we have enough data
        if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < period + 1:
            return {'value': 0, 'is_increasing': False, 'percentile': 0, 'regime': 'normal'}
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
            else:
                returns.append(0)
        
        # Calculate standard deviation over the specified period
        current_std = np.std(returns[-period:])
        
        # Calculate previous period for comparison
        if len(returns) > period * 2:
            previous_std = np.std(returns[-(period*2):-period])
        else:
            previous_std = current_std
        
        # Normalize if requested (std dev of returns is already a percentage)
        if not normalize:
            # Convert from percentage to absolute
            avg_price = np.mean(prices[-period:])
            current_std = current_std * avg_price
            previous_std = previous_std * avg_price
        
        # Determine if volatility is increasing
        is_increasing = current_std > previous_std
        
        # Calculate percentile if we have enough history
        percentile = 0
        if len(returns) >= 100:  # Need more history for reliable percentiles
            std_history = []
            for i in range(20):  # Calculate 20 historical periods
                start_idx = len(returns) - period - i
                if start_idx >= 0:
                    period_std = np.std(returns[start_idx:start_idx+period])
                    std_history.append(period_std)
            
            if std_history:
                percentile = sum(1 for x in std_history if x < current_std) / len(std_history) * 100
        
        # Determine volatility regime
        if percentile > 80:
            regime = 'high'
        elif percentile < 20:
            regime = 'low'
        else:
            regime = 'normal'
        
        return {
            'value': current_std,
            'is_increasing': is_increasing,
            'percentile': percentile,
            'regime': regime
        }


@register_feature(category="technical")
class SupportResistanceFeature(Feature):
    """
    Support and Resistance Levels feature.
    
    This feature identifies key support and resistance levels and measures
    the distance of the current price from these levels.
    """
    
    def __init__(self, 
                 name: str = "support_resistance", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Support and resistance analysis"):
        """
        Initialize the support/resistance feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - method: Method for level detection ('peaks', 'pivots', 'volume')
                - lookback: Lookback period for calculation (default: 100)
                - strength_threshold: Threshold for level strength (default: 2)
                - proximity_threshold: Threshold for level proximity in % (default: 3.0)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for support/resistance calculation."""
        return {
            'method': 'peaks',
            'lookback': 100,
            'strength_threshold': 2,
            'proximity_threshold': 3.0
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_methods = ['peaks', 'pivots', 'volume']
        if self.params.get('method') not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify support/resistance levels and calculate price distance.
        
        Args:
            data: Dictionary containing price data and indicators
        
        Returns:
            Dictionary with support/resistance information
        """
        method = self.params.get('method', 'peaks')
        lookback = self.params.get('lookback', 100)
        strength_threshold = self.params.get('strength_threshold', 2)
        proximity_threshold = self.params.get('proximity_threshold', 3.0)
        
        # Check for required price data
        if not all(k in data for k in ['High', 'Low', 'Close']):
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None,
                'price_position': 0
            }
        
        # Extract price data
        highs = data['High']
        lows = data['Low']
        closes = data['Close']
        
        # Ensure we have enough data
        if not all(isinstance(x, (list, np.ndarray, pd.Series)) for x in [highs, lows, closes]):
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None,
                'price_position': 0
            }
        
        if len(closes) < lookback:
            lookback = len(closes)
        
        # Get the current price
        current_price = closes[-1]
        
        # Calculate levels based on selected method
        if method == 'peaks':
            levels = self._find_peak_levels(highs[-lookback:], lows[-lookback:], strength_threshold)
        elif method == 'pivots':
            levels = self._find_pivot_levels(highs[-lookback:], lows[-lookback:], closes[-lookback:])
        elif method == 'volume':
            if 'Volume' in data and isinstance(data['Volume'], (list, np.ndarray, pd.Series)) and len(data['Volume']) >= lookback:
                levels = self._find_volume_levels(highs[-lookback:], lows[-lookback:], closes[-lookback:], data['Volume'][-lookback:])
            else:
                levels = self._find_peak_levels(highs[-lookback:], lows[-lookback:], strength_threshold)
        else:
            levels = []
        
        # Separate into support and resistance levels
        support_levels = [level for level in levels if level < current_price]
        resistance_levels = [level for level in levels if level > current_price]
        
        # Find nearest levels
        nearest_support = max(support_levels) if support_levels else None
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        
        # Calculate price position between nearest levels
        price_position = 0
        if nearest_support is not None and nearest_resistance is not None:
            range_size = nearest_resistance - nearest_support
            if range_size > 0:
                price_position = (current_price - nearest_support) / range_size * 100
        
        # Check for nearby levels within proximity threshold
        nearby_level = None
        for level in levels:
            distance_pct = abs(level - current_price) / current_price * 100
            if distance_pct <= proximity_threshold:
                nearby_level = level
                break
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'price_position': price_position,
            'nearby_level': nearby_level,
            'proximity_pct': abs(nearby_level - current_price) / current_price * 100 if nearby_level else None
        }
    
    def _find_peak_levels(self, highs, lows, strength_threshold):
        """Find support/resistance levels based on price peaks."""
        # Identify significant highs and lows
        significant_levels = []
        
        # Find peaks in high prices
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                significant_levels.append(highs[i])
        
        # Find troughs in low prices
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                significant_levels.append(lows[i])
        
        # Group similar levels
        grouped_levels = []
        if significant_levels:
            # Sort levels
            sorted_levels = sorted(significant_levels)
            
            # Group levels that are within 0.5% of each other
            current_group = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if level / current_group[0] <= 1.005 and level / current_group[0] >= 0.995:
                    current_group.append(level)
                else:
                    # Add the average of the current group if it has enough points
                    if len(current_group) >= strength_threshold:
                        grouped_levels.append(sum(current_group) / len(current_group))
                    current_group = [level]
            
            # Add the last group if it has enough points
            if len(current_group) >= strength_threshold:
                grouped_levels.append(sum(current_group) / len(current_group))
        
        return grouped_levels
    
    def _find_pivot_levels(self, highs, lows, closes):
        """Find support/resistance levels based on pivot points."""
        # Calculate pivot points for each period
        pivot_levels = []
        
        for i in range(5, len(closes) - 5, 5):  # Use 5-day chunks
            # Standard pivot point
            pivot = (highs[i-1] + lows[i-1] + closes[i-1]) / 3
            
            # Support and resistance levels
            s1 = (2 * pivot) - highs[i-1]
            s2 = pivot - (highs[i-1] - lows[i-1])
            r1 = (2 * pivot) - lows[i-1]
            r2 = pivot + (highs[i-1] - lows[i-1])
            
            pivot_levels.extend([pivot, s1, s2, r1, r2])
        
        # Add fibonacci retracement levels
        if len(closes) >= 2:
            overall_high = max(highs)
            overall_low = min(lows)
            range_size = overall_high - overall_low
            
            fib_levels = [
                overall_low + range_size * 0.236,
                overall_low + range_size * 0.382,
                overall_low + range_size * 0.5,
                overall_low + range_size * 0.618,
                overall_low + range_size * 0.786
            ]
            
            pivot_levels.extend(fib_levels)
        
        # Group similar levels
        grouped_levels = []
        if pivot_levels:
            # Sort levels
            sorted_levels = sorted(pivot_levels)
            
            # Group levels that are within 0.5% of each other
            current_group = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if level / current_group[0] <= 1.005 and level / current_group[0] >= 0.995:
                    current_group.append(level)
                else:
                    grouped_levels.append(sum(current_group) / len(current_group))
                    current_group = [level]
            
            # Add the last group
            grouped_levels.append(sum(current_group) / len(current_group))
        
        return grouped_levels
    
    def _find_volume_levels(self, highs, lows, closes, volumes):
        """Find support/resistance levels based on volume profile."""
        # Create price bins
        min_price = min(lows)
        max_price = max(highs)
        num_bins = 20
        
        if min_price == max_price:
            return []
            
        bin_edges = np.linspace(min_price, max_price, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Distribute volume into price bins
        volume_profile = np.zeros(num_bins)
        
        for i in range(len(closes)):
            # Find which bin this price falls into
            price = closes[i]
            volume = volumes[i]
            
            bin_idx = np.searchsorted(bin_edges, price) - 1
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            elif bin_idx < 0:
                bin_idx = 0
                
            volume_profile[bin_idx] += volume
        
        # Find local maxima in the volume profile
        volume_levels = []
        for i in range(1, num_bins - 1):
            if volume_profile[i] > volume_profile[i-1] and volume_profile[i] > volume_profile[i+1]:
                volume_levels.append(bin_centers[i])
        
        return volume_levels


@register_feature(category="technical")
class SignalAgreementFeature(Feature):
    """
    Signal Agreement feature.
    
    This feature analyzes the agreement or disagreement among multiple
    technical indicators to provide a consensus signal.
    """
    
    def __init__(self, 
                 name: str = "signal_agreement", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Technical indicator consensus"):
        """
        Initialize the signal agreement feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - indicators: List of indicators to include in consensus
                - weights: Optional dictionary of weights for each indicator
                - threshold: Threshold for signal consensus (default: 0.6)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for signal agreement."""
        return {
            'indicators': ['MA_crossover', 'RSI', 'MACD', 'BB'],
            'weights': None,
            'threshold': 0.6
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate signal agreement among indicators.
        
        Args:
            data: Dictionary containing technical indicators
        
        Returns:
            Dictionary with consensus information
        """
        indicators = self.params.get('indicators', ['MA_crossover', 'RSI', 'MACD', 'BB'])
        weights = self.params.get('weights', None)
        threshold = self.params.get('threshold', 0.6)
        
        # Initialize signal counts
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_weight = 0
        
        # Process each indicator
        signals_detail = {}
        
        for indicator in indicators:
            # Skip if indicator not available
            if indicator not in data:
                continue
            
            # Determine indicator signal
            signal = self._get_indicator_signal(indicator, data[indicator])
            signals_detail[indicator] = signal
            
            # Apply weighting if provided
            weight = weights.get(indicator, 1.0) if weights else 1.0
            total_weight += weight
            
            # Count signals
            if signal == 1:
                bullish_count += weight
            elif signal == -1:
                bearish_count += weight
            else:
                neutral_count += weight
        
        # Calculate consensus percentage
        if total_weight > 0:
            bullish_pct = bullish_count / total_weight
            bearish_pct = bearish_count / total_weight
            neutral_pct = neutral_count / total_weight
        else:
            bullish_pct = bearish_pct = neutral_pct = 0
        
        # Determine consensus signal
        if bullish_pct >= threshold:
            consensus = 1  # Bullish consensus
        elif bearish_pct >= threshold:
            consensus = -1  # Bearish consensus
        else:
            consensus = 0  # No clear consensus
        
        # Calculate agreement strength
        agreement_strength = max(bullish_pct, bearish_pct)
        
        return {
            'consensus': consensus,
            'agreement_strength': agreement_strength,
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'neutral_pct': neutral_pct,
            'signals': signals_detail
        }
    
    def _get_indicator_signal(self, indicator, value):
        """Extract signal from indicator value."""
        # Handle different indicator formats
        if isinstance(value, dict) and 'signal' in value:
            return value['signal']
        
        # Handle common indicators
        if indicator == 'RSI':
            # For RSI: below 30 = buy, above 70 = sell
            if isinstance(value, (list, np.ndarray, pd.Series)):
                rsi = value[-1]
            else:
                rsi = value
                
            if rsi < 30:
                return 1
            elif rsi > 70:
                return -1
            else:
                return 0
                
        elif indicator == 'MACD':
            # For MACD: signal line crossover
            if isinstance(value, dict):
                if 'histogram' in value:
                    # Positive histogram = buy, negative = sell
                    hist = value['histogram']
                    if isinstance(hist, (list, np.ndarray, pd.Series)):
                        hist = hist[-1]
                    return 1 if hist > 0 else (-1 if hist < 0 else 0)
                    
                elif 'value' in value and 'signal' in value:
                    # MACD line above signal line = buy, below = sell
                    macd = value['value']
                    signal = value['signal']
                    
                    if isinstance(macd, (list, np.ndarray, pd.Series)):
                        macd = macd[-1]
                    if isinstance(signal, (list, np.ndarray, pd.Series)):
                        signal = signal[-1]
                        
                    return 1 if macd > signal else (-1 if macd < signal else 0)
            
            return 0
            
        elif indicator == 'BB':
            # For Bollinger Bands: price near upper band = sell, near lower = buy
            if isinstance(value, dict):
                if all(k in value for k in ['upper', 'lower', 'middle']) and 'Close' in data:
                    upper = value['upper']
                    lower = value['lower']
                    
                    if isinstance(upper, (list, np.ndarray, pd.Series)):
                        upper = upper[-1]
                    if isinstance(lower, (list, np.ndarray, pd.Series)):
                        lower = lower[-1]
                        
                    close = data['Close'][-1] if isinstance(data['Close'], (list, np.ndarray, pd.Series)) else data['Close']
                    
                    # Calculate position within bands
                    band_width = upper - lower
                    if band_width > 0:
                        position = (close - lower) / band_width
                        
                        if position > 0.8:
                            return -1  # Near upper band = sell
                        elif position < 0.2:
                            return 1   # Near lower band = buy
            
            return 0
            
        elif indicator == 'MA_crossover':
            # For MA crossover: fast MA above slow MA = buy, below = sell
            if isinstance(value, dict) and 'state' in value:
                state = value['state']
                return state  # Already in the correct format
            
            return 0
        
        # Default: try to interpret numeric values as a signal
        if isinstance(value, (int, float)):
            if value > 0.5:
                return 1
            elif value < -0.5:
                return -1
            else:
                return 0
        
        # Fallback for complex objects
        return 0


@register_feature(category="technical")
class DivergenceFeature(Feature):
    """
    Divergence Detection feature.
    
    This feature detects divergences between price and indicators,
    which can signal potential trend reversals.
    """
    
    def __init__(self, 
                 name: str = "divergence", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Price-indicator divergence detection"):
        """
        Initialize the divergence feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - indicator: Indicator to check for divergence ('RSI', 'MACD', 'CCI', etc.)
                - lookback: Lookback period for divergence detection (default: 20)
                - peak_threshold: Threshold for identifying peaks/troughs (default: 3)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for divergence detection."""
        return {
            'indicator': 'RSI',
            'lookback': 20,
            'peak_threshold': 3
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect divergences between price and indicator.
        
        Args:
            data: Dictionary containing price data and indicators
        
        Returns:
            Dictionary with divergence information
        """
        indicator_key = self.params.get('indicator', 'RSI')
        lookback = self.params.get('lookback', 20)
        peak_threshold = self.params.get('peak_threshold', 3)
        
        # Check for required data
        if 'Close' not in data or indicator_key not in data:
            return {
                'bullish_divergence': False,
                'bearish_divergence': False,
                'hidden_bullish': False,
                'hidden_bearish': False,
                'strength': 0
            }
        
        # Extract price and indicator data
        prices = data['Close']
        indicator = data[indicator_key]
        
        # Ensure we have enough data
        if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < lookback:
            return {
                'bullish_divergence': False,
                'bearish_divergence': False,
                'hidden_bullish': False,
                'hidden_bearish': False,
                'strength': 0
            }
            
        if not isinstance(indicator, (list, np.ndarray, pd.Series)) or len(indicator) < lookback:
            return {
                'bullish_divergence': False,
                'bearish_divergence': False,
                'hidden_bullish': False,
                'hidden_bearish': False,
                'strength': 0
            }
        
        # Get the relevant data window
        price_window = prices[-lookback:]
        indicator_window = indicator[-lookback:]
        
        # Find peaks and troughs in price
        price_peaks = self._find_peaks(price_window, peak_threshold)
        price_troughs = self._find_troughs(price_window, peak_threshold)
        
        # Find peaks and troughs in indicator
        indicator_peaks = self._find_peaks(indicator_window, peak_threshold)
        indicator_troughs = self._find_troughs(indicator_window, peak_threshold)
        
        # Check for regular divergence
        bullish_divergence = self._check_bullish_divergence(price_window, indicator_window, price_troughs, indicator_troughs)
        bearish_divergence = self._check_bearish_divergence(price_window, indicator_window, price_peaks, indicator_peaks)
        
        # Check for hidden divergence
        hidden_bullish = self._check_hidden_bullish(price_window, indicator_window, price_troughs, indicator_troughs)
        hidden_bearish = self._check_hidden_bearish(price_window, indicator_window, price_peaks, indicator_peaks)
        
        # Calculate divergence strength (0-100)
        strength = 0
        if bullish_divergence or bearish_divergence:
            # Regular divergence is stronger
            strength = 80
        elif hidden_bullish or hidden_bearish:
            # Hidden divergence is more subtle
            strength = 50
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'hidden_bullish': hidden_bullish,
            'hidden_bearish': hidden_bearish,
            'strength': strength
        }
    
    def _find_peaks(self, data, threshold):
        """Find peaks in data."""
        peaks = []
        for i in range(threshold, len(data) - threshold):
            is_peak = True
            for j in range(1, threshold + 1):
                if data[i] <= data[i-j] or data[i] <= data[i+j]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data, threshold):
        """Find troughs in data."""
        troughs = []
        for i in range(threshold, len(data) - threshold):
            is_trough = True
            for j in range(1, threshold + 1):
                if data[i] >= data[i-j] or data[i] >= data[i+j]:
                    is_trough = False
                    break
            if is_trough:
                troughs.append(i)
        return troughs
    
    def _check_bullish_divergence(self, prices, indicator, price_troughs, indicator_troughs):
        """Check for bullish divergence: lower price lows but higher indicator lows."""
        if len(price_troughs) < 2 or len(indicator_troughs) < 2:
            return False
        
        # Get the two most recent price troughs
        last_price_trough_idx = price_troughs[-1]
        prev_price_trough_idx = price_troughs[-2]
        
        # Get the two most recent indicator troughs
        last_ind_trough_idx = indicator_troughs[-1]
        prev_ind_trough_idx = indicator_troughs[-2]
        
        # Check if price is making lower lows
        price_lower_low = prices[last_price_trough_idx] < prices[prev_price_trough_idx]
        
        # Check if indicator is making higher lows
        indicator_higher_low = indicator[last_ind_trough_idx] > indicator[prev_ind_trough_idx]
        
        # Both conditions must be true for bullish divergence
        return price_lower_low and indicator_higher_low
    
    def _check_bearish_divergence(self, prices, indicator, price_peaks, indicator_peaks):
        """Check for bearish divergence: higher price highs but lower indicator highs."""
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return False
        
        # Get the two most recent price peaks
        last_price_peak_idx = price_peaks[-1]
        prev_price_peak_idx = price_peaks[-2]
        
        # Get the two most recent indicator peaks
        last_ind_peak_idx = indicator_peaks[-1]
        prev_ind_peak_idx = indicator_peaks[-2]
        
        # Check if price is making higher highs
        price_higher_high = prices[last_price_peak_idx] > prices[prev_price_peak_idx]
        
        # Check if indicator is making lower highs
        indicator_lower_high = indicator[last_ind_peak_idx] < indicator[prev_ind_peak_idx]
        
        # Both conditions must be true for bearish divergence
        return price_higher_high and indicator_lower_high
    
    def _check_hidden_bullish(self, prices, indicator, price_troughs, indicator_troughs):
        """Check for hidden bullish divergence: higher price lows but lower indicator lows."""
        if len(price_troughs) < 2 or len(indicator_troughs) < 2:
            return False
        
        # Get the two most recent price troughs
        last_price_trough_idx = price_troughs[-1]
        prev_price_trough_idx = price_troughs[-2]
        
        # Get the two most recent indicator troughs
        last_ind_trough_idx = indicator_troughs[-1]
        prev_ind_trough_idx = indicator_troughs[-2]
        
        # Check if price is making higher lows
        price_higher_low = prices[last_price_trough_idx] > prices[prev_price_trough_idx]
        
        # Check if indicator is making lower lows
        indicator_lower_low = indicator[last_ind_trough_idx] < indicator[prev_ind_trough_idx]
        
        # Both conditions must be true for hidden bullish divergence
        return price_higher_low and indicator_lower_low
    
    def _check_hidden_bearish(self, prices, indicator, price_peaks, indicator_peaks):
        """Check for hidden bearish divergence: lower price highs but higher indicator highs."""
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return False
        
        # Get the two most recent price peaks
        last_price_peak_idx = price_peaks[-1]
        prev_price_peak_idx = price_peaks[-2]
        
        # Get the two most recent indicator peaks
        last_ind_peak_idx = indicator_peaks[-1]
        prev_ind_peak_idx = indicator_peaks[-2]
        
        # Check if price is making lower highs
        price_lower_high = prices[last_price_peak_idx] < prices[prev_price_peak_idx]
        
        # Check if indicator is making higher highs
        indicator_higher_high = indicator[last_ind_peak_idx] > indicator[prev_ind_peak_idx]
        
        # Both conditions must be true for hidden bearish divergence
        return price_lower_high and indicator_higher_high

class MACrossoverFeature(Feature):
    """
    Moving Average Crossover feature.
    
    This feature detects crossovers between two moving averages and provides
    information about the state and direction of the crossover.
    """
    
    def __init__(self, 
                 name: str = "ma_crossover", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Moving average crossover detection"):
        """
        Initialize the moving average crossover feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - fast_ma: Name of the fast MA indicator (default: 'SMA_10')
                - slow_ma: Name of the slow MA indicator (default: 'SMA_30')
                - smooth: Whether to smooth the crossover signal (default: False)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for moving average crossover."""
        return {
            'fast_ma': 'SMA_10',
            'slow_ma': 'SMA_30',
            'smooth': False
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate moving average crossover state and signals.
        
        Args:
            data: Dictionary containing technical indicators
        
        Returns:
            Dictionary with crossover information
        """
        fast_ma_key = self.params.get('fast_ma', 'SMA_10')
        slow_ma_key = self.params.get('slow_ma', 'SMA_30')
        smooth = self.params.get('smooth', False)
        
        # Check if MAs are available in data
        if fast_ma_key not in data or slow_ma_key not in data:
            return {
                'state': 0,       # 0 = neutral, 1 = fast above slow, -1 = fast below slow
                'crossover': 0,   # 0 = no crossover, 1 = bullish crossover, -1 = bearish crossover
                'distance': 0,    # Distance between the MAs as percentage of slow MA
                'signal': 0       # Trading signal: 1 = buy, -1 = sell, 0 = neutral
            }
        
        # Extract MA values
        fast_ma = data[fast_ma_key]
        slow_ma = data[slow_ma_key]
        
        # Handle different data types (scalar vs sequence)
        if isinstance(fast_ma, (list, np.ndarray, pd.Series)):
            if len(fast_ma) < 2:
                return {'state': 0, 'crossover': 0, 'distance': 0, 'signal': 0}
            current_fast = fast_ma[-1]
            prev_fast = fast_ma[-2]
        else:
            current_fast = fast_ma
            prev_fast = None
            
        if isinstance(slow_ma, (list, np.ndarray, pd.Series)):
            if len(slow_ma) < 2:
                return {'state': 0, 'crossover': 0, 'distance': 0, 'signal': 0}
            current_slow = slow_ma[-1]
            prev_slow = slow_ma[-2]
        else:
            current_slow = slow_ma
            prev_slow = None
        
        # Calculate current state
        if current_fast > current_slow:
            state = 1  # Fast above slow (bullish)
        elif current_fast < current_slow:
            state = -1  # Fast below slow (bearish)
        else:
            state = 0  # Equal (neutral)
        
        # Calculate distance between MAs
        if current_slow != 0:
            distance = (current_fast - current_slow) / current_slow * 100
        else:
            distance = 0
        
        # Detect crossover
        crossover = 0
        if prev_fast is not None and prev_slow is not None:
            if prev_fast <= prev_slow and current_fast > current_slow:
                crossover = 1  # Bullish crossover
            elif prev_fast >= prev_slow and current_fast < current_slow:
                crossover = -1  # Bearish crossover
        
        # Generate signal
        signal = 0
        if crossover != 0:
            # Crossover signal
            signal = crossover
        elif smooth and state != 0:
            # Smooth signal based on current state
            signal = state
        
        return {
            'state': state,
            'crossover': crossover,
            'distance': distance,
            'signal': signal
        }


@register_feature(category="technical")
class OscillatorStateFeature(Feature):
    """
    Oscillator State feature.
    
    This feature analyzes oscillator indicators (RSI, Stochastic, etc.) and
    identifies overbought/oversold conditions and divergences.
    """
    
    def __init__(self, 
                 name: str = "oscillator_state", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Oscillator state analysis"):
        """
        Initialize the oscillator state feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - oscillator: Name of the oscillator indicator (default: 'RSI')
                - overbought: Overbought threshold (default: 70)
                - oversold: Oversold threshold (default: 30)
                - check_divergence: Whether to check for divergence (default: True)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for oscillator state."""
        return {
            'oscillator': 'RSI',
            'overbought': 70,
            'oversold': 30,
            'check_divergence': True
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate oscillator state and identify conditions.
        
        Args:
            data: Dictionary containing price data and indicators
        
        Returns:
            Dictionary with oscillator state information
        """
        oscillator_key = self.params.get('oscillator', 'RSI')
        overbought = self.params.get('overbought', 70)
        oversold = self.params.get('oversold', 30)
        check_divergence = self.params.get('check_divergence', True)
        
        # Check if oscillator is available in data
        if oscillator_key not in data:
            return {
                'condition': 'neutral',  # 'overbought', 'oversold', or 'neutral'
                'value': np.nan,          # Current oscillator value
                'divergence': None,       # 'bullish', 'bearish', or None
                'signal': 0               # Trading signal: 1 = buy, -1 = sell, 0 = neutral
            }
        
        # Extract oscillator values
        oscillator = data[oscillator_key]
        
        # Extract latest value
        current_value = oscillator[-1] if isinstance(oscillator, (list, np.ndarray, pd.Series)) else oscillator
        
        # Determine condition
        if current_value >= overbought:
            condition = 'overbought'
        elif current_value <= oversold:
            condition = 'oversold'
        else:
            condition = 'neutral'
        
        # Check for divergence if requested
        divergence = None
        if check_divergence and isinstance(oscillator, (list, np.ndarray, pd.Series)) and len(oscillator) >= 5:
            if 'Close' in data and isinstance(data['Close'], (list, np.ndarray, pd.Series)) and len(data['Close']) >= 5:
                # Get recent oscillator and price data
                recent_oscillator = oscillator[-5:]
                recent_prices = data['Close'][-5:]
                
                # Check for bearish divergence: price making higher highs but oscillator making lower highs
                if recent_prices[-1] > max(recent_prices[:-1]) and recent_oscillator[-1] < max(recent_oscillator[:-1]):
                    divergence = 'bearish'
                
                # Check for bullish divergence: price making lower lows but oscillator making higher lows
                elif recent_prices[-1] < min(recent_prices[:-1]) and recent_oscillator[-1] > min(recent_oscillator[:-1]):
                    divergence = 'bullish'
        
        # Generate signal
        signal = 0
        if condition == 'overbought':
            signal = -1  # Sell signal
        elif condition == 'oversold':
            signal = 1   # Buy signal
        elif divergence == 'bullish':
            signal = 1   # Buy signal on bullish divergence
        elif divergence == 'bearish':
            signal = -1  # Sell signal on bearish divergence
        
        return {
            'condition': condition,
            'value': current_value,
            'divergence': divergence,
            'signal': signal
        }


@register_feature(category="technical")
class TrendStrengthFeature(Feature):
    """
    Trend Strength feature.
    
    This feature measures the strength and direction of the current trend
    using indicators like ADX, Aroon, or directional movement.
    """
    
    def __init__(self, 
                 name: str = "trend_strength", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Trend strength measurement"):
        """
        Initialize the trend strength feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - method: Method for trend strength calculation ('adx', 'aroon', 'slope')
                - threshold: Threshold for strong trend (default: 25 for ADX)
                - lookback: Lookback period for calculation (default: 14)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for trend strength."""
        return {
            'method': 'adx',
            'threshold': 25,
            'lookback': 14
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_methods = ['adx', 'aroon', 'slope']
        if self.params.get('method') not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trend strength and direction.
        
        Args:
            data: Dictionary containing price data and indicators
        
        Returns:
            Dictionary with trend information
        """
        method = self.params.get('method', 'adx')
        threshold = self.params.get('threshold', 25)
        
        if method == 'adx':
            return self._calculate_adx(data, threshold)
        elif method == 'aroon':
            return self._calculate_aroon(data, threshold)
        elif method == 'slope':
            return self._calculate_slope(data, threshold)
        
        # Fallback
        return {
            'strength': 0,    # Trend strength value (0-100)
            'direction': 0,   # Trend direction: 1 = up, -1 = down, 0 = neutral
            'is_strong': False # Whether the trend is considered strong
        }
    
    def _calculate_adx(self, data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Calculate trend strength using ADX."""
        # Check for required indicators
        if 'ADX' not in data or 'DI+' not in data or 'DI-' not in data:
            return {'strength': 0, 'direction': 0, 'is_strong': False}
        
        # Extract values
        adx = data['ADX'][-1] if isinstance(data['ADX'], (list, np.ndarray, pd.Series)) else data['ADX']
        di_plus = data['DI+'][-1] if isinstance(data['DI+'], (list, np.ndarray, pd.Series)) else data['DI+']
        di_minus = data['DI-'][-1] if isinstance(data['DI-'], (list, np.ndarray, pd.Series)) else data['DI-']
        
        # Determine direction
        if di_plus > di_minus:
            direction = 1  # Uptrend
        elif di_minus > di_plus:
            direction = -1  # Downtrend
        else:
            direction = 0  # Neutral
        
        # Determine if trend is strong
        is_strong = adx >= threshold
        
        return {
            'strength': adx,
            'direction': direction,
            'is_strong': is_strong,
            'di_plus': di_plus,
            'di_minus': di_minus
        }
    
    def _calculate_aroon(self, data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Calculate trend strength using Aroon indicators."""
        # Check for required indicators
        if 'Aroon_Up' not in data or 'Aroon_Down' not in data:
            return {'strength': 0, 'direction': 0, 'is_strong': False}
        
        # Extract values
        aroon_up = data['Aroon_Up'][-1] if isinstance(data['Aroon_Up'], (list, np.ndarray, pd.Series)) else data['Aroon_Up']
        aroon_down = data['Aroon_Down'][-1] if isinstance(data['Aroon_Down'], (list, np.ndarray, pd.Series)) else data['Aroon_Down']
        
        # Calculate Aroon Oscillator
        aroon_osc = aroon_up - aroon_down
        
        # Determine direction
        if aroon_up > aroon_down:
            direction = 1  # Uptrend
        elif aroon_down > aroon_up:
            direction = -1  # Downtrend
        else:
            direction = 0  # Neutral
        
        # Calculate strength (max Aroon value indicates strength)
        strength = max(aroon_up, aroon_down)
        
        # Determine if trend is strong
        is_strong = strength >= threshold
        
        return {
            'strength': strength,
            'direction': direction,
            'is_strong': is_strong,
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_osc
        }
    
    def _calculate_slope(self, data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Calculate trend strength using price slope."""
        lookback = self.params.get('lookback', 14)
        
        # Check for required price data
        if 'Close' not in data:
            return {'strength': 0, 'direction': 0, 'is_strong': False}
        
        prices = data['Close']
        
        # Ensure we have enough data
        if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < lookback:
            return {'strength': 0, 'direction': 0, 'is_strong': False}
        
        # Get the price range for calculation
        price_range = prices[-lookback:]
        
        # Calculate linear regression
        x = np.arange(lookback)
        slope, intercept = np.polyfit(x, price_range, 1)
        
        # Convert slope to percentage
        avg_price = np.mean(price_range)
        slope_pct = (slope / avg_price) * 100
        
        # Determine direction
        if slope > 0:
            direction = 1  # Uptrend
        elif slope < 0:
            direction = -1  # Downtrend
        else:
            direction = 0  # Neutral
        
        # Calculate R-squared (goodness of fit)
        y_pred = slope * x + intercept
        ss_tot = np.sum((price_range - np.mean(price_range)) ** 2)
        ss_res = np.sum((price_range - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate strength (combination of slope and fit)
        strength = abs(slope_pct) * r_squared * 100
        
        # Determine if trend is strong
        is_strong = strength >= threshold
        
        return {
            'strength': strength,
            'direction': direction,
            'is_strong': is_strong,
            'slope': slope,
            'slope_pct': slope_pct,
            'r_squared': r_squared
        }



@register_feature()
class SMA_Crossover(Feature):
    """Feature that detects crossovers between two SMAs."""
    def __init__(self, fast_window=10, slow_window=30, name=None):
        super().__init__(name or f"SMA_Crossover_{fast_window}_{slow_window}")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.prev_fast_sma = None
        self.prev_slow_sma = None
        self.prev_value = 0
        
    def calculate(self, bar_data, history=None):
        if history is None or len(history) < self.slow_window:
            self.value = 0
            return self.value
            
        # Calculate current SMAs
        prices = [bar['Close'] for bar in history[-self.slow_window:]] + [bar_data['Close']]
        fast_sma = sum(prices[-self.fast_window:]) / self.fast_window
        slow_sma = sum(prices[-self.slow_window:]) / self.slow_window
        
        # Calculate crossover value (positive for bullish, negative for bearish)
        if self.prev_fast_sma is not None and self.prev_slow_sma is not None:
            if self.prev_fast_sma <= self.prev_slow_sma and fast_sma > slow_sma:
                self.value = 1  # Bullish crossover
            elif self.prev_fast_sma >= self.prev_slow_sma and fast_sma < slow_sma:
                self.value = -1  # Bearish crossover
            else:
                # No crossover, but return relationship between SMAs
                self.value = 0.5 if fast_sma > slow_sma else -0.5
        else:
            self.value = 0.5 if fast_sma > slow_sma else -0.5
            
        # Update previous values
        self.prev_fast_sma = fast_sma
        self.prev_slow_sma = slow_sma
        self.prev_value = self.value
        
        return self.value
