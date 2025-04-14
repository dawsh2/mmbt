"""
Price Features Module

This module provides features derived directly from price data, such as returns,
normalized prices, and price patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque

from src.features.feature_base import Feature, StatefulFeature
from src.features.feature_registry import register_feature


@register_feature(category="price")
class ReturnFeature(Feature):
    """
    Calculate price returns over specified periods.
    
    This feature computes price returns (percentage change) over one or more time periods.
    """
    
    def __init__(self, 
                 name: str = "return", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Price returns over specified periods"):
        """
        Initialize the return feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - periods: List of periods to calculate returns for (default: [1])
                - price_key: Key to price data in the input dictionary (default: 'Close')
                - log_returns: Whether to compute log returns instead of simple returns (default: True)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for return calculation."""
        return {
            'periods': [1],
            'price_key': 'Close',
            'log_returns': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        if not isinstance(self.params.get('periods', []), (list, tuple)):
            raise ValueError("Parameter 'periods' must be a list or tuple")
        
        if not all(isinstance(p, int) and p > 0 for p in self.params.get('periods', [])):
            raise ValueError("All periods must be positive integers")
            
    def calculate(self, data: Dict[str, Any]) -> Dict[int, float]:
        """
        Calculate returns for each specified period.
        
        Args:
            data: Dictionary containing price history with keys like 'Open', 'High', 'Low', 'Close'
                 Expected format: {'Close': np.array or list of prices, ...}
        
        Returns:
            Dictionary mapping periods to calculated returns
        """
        price_key = self.params.get('price_key', 'Close')
        periods = self.params.get('periods', [1])
        log_returns = self.params.get('log_returns', True)
        
        # Extract price data
        if price_key not in data:
            raise KeyError(f"Price key '{price_key}' not found in data")
            
        prices = data[price_key]
        
        # Check if we have enough data
        if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) <= max(periods):
            return {period: np.nan for period in periods}
        
        # Calculate returns for each period
        result = {}
        current_price = prices[-1]
        
        for period in periods:
            if period < len(prices):
                previous_price = prices[-(period+1)]
                
                if log_returns:
                    # Calculate log return
                    if previous_price > 0 and current_price > 0:
                        result[period] = np.log(current_price / previous_price)
                    else:
                        result[period] = np.nan
                else:
                    # Calculate simple return
                    if previous_price != 0:
                        result[period] = (current_price - previous_price) / previous_price
                    else:
                        result[period] = np.nan
            else:
                result[period] = np.nan
                
        return result


@register_feature(category="price")
class NormalizedPriceFeature(Feature):
    """
    Normalize price data relative to a reference point.
    
    This feature normalizes prices using various methods such as z-score normalization,
    min-max scaling, or relative to a moving average.
    """
    
    def __init__(self, 
                 name: str = "normalized_price", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Normalized price data"):
        """
        Initialize the normalized price feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - method: Normalization method ('z-score', 'min-max', or 'relative') (default: 'z-score')
                - window: Window size for calculating statistics (default: 20)
                - price_key: Key to price data in the input dictionary (default: 'Close')
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for normalization."""
        return {
            'method': 'z-score',
            'window': 20,
            'price_key': 'Close'
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_methods = ['z-score', 'min-max', 'relative']
        if self.params.get('method') not in valid_methods:
            raise ValueError(f"Normalization method must be one of {valid_methods}")
            
        if not isinstance(self.params.get('window', 20), int) or self.params.get('window', 20) <= 0:
            raise ValueError("Window size must be a positive integer")
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """
        Calculate normalized price.
        
        Args:
            data: Dictionary containing price history
        
        Returns:
            Normalized price value
        """
        price_key = self.params.get('price_key', 'Close')
        method = self.params.get('method', 'z-score')
        window = self.params.get('window', 20)
        
        # Extract price data
        if price_key not in data:
            raise KeyError(f"Price key '{price_key}' not found in data")
            
        prices = data[price_key]
        
        # Check if we have enough data
        if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < window:
            return np.nan
        
        # Get the window of prices for calculation
        price_window = prices[-window:]
        current_price = prices[-1]
        
        # Calculate normalized price based on selected method
        if method == 'z-score':
            # Z-score normalization: (x - mean) / std
            mean = np.mean(price_window)
            std = np.std(price_window)
            if std == 0:
                return 0  # Avoid division by zero
            return (current_price - mean) / std
            
        elif method == 'min-max':
            # Min-max scaling: (x - min) / (max - min)
            min_price = np.min(price_window)
            max_price = np.max(price_window)
            if max_price == min_price:
                return 0.5  # Avoid division by zero
            return (current_price - min_price) / (max_price - min_price)
            
        elif method == 'relative':
            # Relative to moving average: (x / MA) - 1
            ma = np.mean(price_window)
            if ma == 0:
                return np.nan  # Avoid division by zero
            return (current_price / ma) - 1
            
        return np.nan  # Fallback (shouldn't be reached due to validation)


@register_feature(category="price")
class PricePatternFeature(Feature):
    """
    Detect specific price patterns in the data.
    
    This feature identifies common price patterns such as double tops/bottoms,
    head and shoulders, or trendline breaks.
    """
    
    def __init__(self, 
                 name: str = "price_pattern", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Price pattern detection"):
        """
        Initialize the price pattern feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - pattern: Pattern to detect ('double_top', 'double_bottom', 'head_shoulders', etc.)
                - window: Window size for pattern detection (default: 20)
                - threshold: Threshold for pattern confirmation (default: 0.03)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for pattern detection."""
        return {
            'pattern': 'double_top',
            'window': 20,
            'threshold': 0.03
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_patterns = ['double_top', 'double_bottom', 'head_shoulders', 'inverse_head_shoulders']
        if self.params.get('pattern') not in valid_patterns:
            raise ValueError(f"Pattern must be one of {valid_patterns}")
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect price patterns in the data.
        
        Args:
            data: Dictionary containing price history
        
        Returns:
            Dictionary with pattern detection results, including:
            - detected: Boolean indicating if pattern was detected
            - confidence: Confidence score for the detection (0.0-1.0)
            - details: Additional pattern-specific details
        """
        pattern = self.params.get('pattern', 'double_top')
        window = self.params.get('window', 20)
        threshold = self.params.get('threshold', 0.03)
        
        # Extract required price data
        if 'High' not in data or 'Low' not in data or 'Close' not in data:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
            
        highs = data['High']
        lows = data['Low']
        closes = data['Close']
        
        # Check if we have enough data
        if not all(isinstance(x, (list, np.ndarray, pd.Series)) for x in [highs, lows, closes]):
            return {'detected': False, 'confidence': 0.0, 'details': {}}
            
        if len(closes) < window:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        # Get the window of data for pattern detection
        window_highs = highs[-window:]
        window_lows = lows[-window:]
        window_closes = closes[-window:]
        
        # Implement pattern detection logic
        if pattern == 'double_top':
            return self._detect_double_top(window_highs, window_lows, window_closes, threshold)
        elif pattern == 'double_bottom':
            return self._detect_double_bottom(window_highs, window_lows, window_closes, threshold)
        elif pattern == 'head_shoulders':
            return self._detect_head_shoulders(window_highs, window_lows, window_closes, threshold)
        elif pattern == 'inverse_head_shoulders':
            return self._detect_inverse_head_shoulders(window_highs, window_lows, window_closes, threshold)
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}
    
    def _detect_double_top(self, highs, lows, closes, threshold):
        """Detect double top pattern."""
        # Simplified algorithm for double top detection:
        # 1. Find local peaks in the price data
        # 2. Check if there are two peaks of similar height
        # 3. Check if there's a significant valley between them
        
        if len(highs) < 10:  # Need enough data for a meaningful pattern
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        # Find local maxima (peaks)
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        # Need at least two peaks
        if len(peaks) < 2:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        # Check for two peaks of similar height
        detected = False
        confidence = 0.0
        details = {}
        
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                idx1, peak1 = peaks[i]
                idx2, peak2 = peaks[j]
                
                # Peaks should be separated by at least 3 bars
                if idx2 - idx1 < 3:
                    continue
                
                # Check if peaks are of similar height
                diff = abs(peak1 - peak2) / max(peak1, peak2)
                if diff < threshold:
                    # Check if there's a valley between peaks
                    valley = min(closes[idx1:idx2])
                    valley_depth = (min(peak1, peak2) - valley) / min(peak1, peak2)
                    
                    if valley_depth > threshold:
                        detected = True
                        confidence = 1.0 - diff  # Higher confidence for more similar peaks
                        details = {
                            'peak1_idx': idx1,
                            'peak1_value': peak1,
                            'peak2_idx': idx2,
                            'peak2_value': peak2,
                            'valley_value': valley,
                            'valley_depth': valley_depth
                        }
                        # Once detected, return immediately
                        return {'detected': detected, 'confidence': confidence, 'details': details}
        
        return {'detected': detected, 'confidence': confidence, 'details': details}
    
    def _detect_double_bottom(self, highs, lows, closes, threshold):
        """Detect double bottom pattern."""
        # Similar to double top but looking for troughs
        
        if len(lows) < 10:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        # Find local minima (troughs)
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                troughs.append((i, lows[i]))
        
        # Need at least two troughs
        if len(troughs) < 2:
            return {'detected': False, 'confidence': 0.0, 'details': {}}
        
        # Check for two troughs of similar depth
        detected = False
        confidence = 0.0
        details = {}
        
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                idx1, trough1 = troughs[i]
                idx2, trough2 = troughs[j]
                
                # Troughs should be separated by at least 3 bars
                if idx2 - idx1 < 3:
                    continue
                
                # Check if troughs are of similar depth
                diff = abs(trough1 - trough2) / min(trough1, trough2)
                if diff < threshold:
                    # Check if there's a peak between troughs
                    peak = max(closes[idx1:idx2])
                    peak_height = (peak - max(trough1, trough2)) / max(trough1, trough2)
                    
                    if peak_height > threshold:
                        detected = True
                        confidence = 1.0 - diff  # Higher confidence for more similar troughs
                        details = {
                            'trough1_idx': idx1,
                            'trough1_value': trough1,
                            'trough2_idx': idx2,
                            'trough2_value': trough2,
                            'peak_value': peak,
                            'peak_height': peak_height
                        }
                        # Once detected, return immediately
                        return {'detected': detected, 'confidence': confidence, 'details': details}
        
        return {'detected': detected, 'confidence': confidence, 'details': details}
    
    def _detect_head_shoulders(self, highs, lows, closes, threshold):
        """Detect head and shoulders pattern."""
        # Simplified algorithm:
        # 1. Find 3 peaks with the middle one higher
        # 2. Check the "neckline" formed by the troughs
        
        # Implementation omitted for brevity
        # This would involve more complex peak detection and pattern confirmation
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}
    
    def _detect_inverse_head_shoulders(self, highs, lows, closes, threshold):
        """Detect inverse head and shoulders pattern."""
        # Similar to regular head and shoulders but inverted
        
        # Implementation omitted for brevity
        
        return {'detected': False, 'confidence': 0.0, 'details': {}}


@register_feature(category="price")
class VolumeProfileFeature(StatefulFeature):
    """
    Calculate volume profile features based on price and volume data.
    
    This feature analyzes the distribution of volume across price levels.
    """
    
    def __init__(self, 
                 name: str = "volume_profile", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Volume profile analysis",
                 max_history: int = 100):
        """
        Initialize the volume profile feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - num_bins: Number of price bins for volume distribution (default: 10)
                - window: Window size for calculation (default: 20)
            description: Feature description
            max_history: Maximum history to maintain
        """
        super().__init__(name, params or self.default_params, description, max_history)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for volume profile."""
        return {
            'num_bins': 10,
            'window': 20
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate volume profile features.
        
        Args:
            data: Dictionary containing price and volume history
        
        Returns:
            Dictionary with volume profile metrics
        """
        num_bins = self.params.get('num_bins', 10)
        window = self.params.get('window', 20)
        
        # Check for required data
        if 'Close' not in data or 'Volume' not in data:
            return {'value_area': 0, 'poc': 0, 'distribution': []}
            
        closes = data['Close']
        volumes = data['Volume']
        
        # Check if we have enough data
        if not isinstance(closes, (list, np.ndarray, pd.Series)) or len(closes) < window:
            return {'value_area': 0, 'poc': 0, 'distribution': []}
            
        if not isinstance(volumes, (list, np.ndarray, pd.Series)) or len(volumes) < window:
            return {'value_area': 0, 'poc': 0, 'distribution': []}
        
        # Get the window of data
        window_closes = closes[-window:]
        window_volumes = volumes[-window:]
        
        # Create price bins
        min_price = min(window_closes)
        max_price = max(window_closes)
        
        if min_price == max_price:
            return {'value_area': 0, 'poc': min_price, 'distribution': []}
            
        bin_edges = np.linspace(min_price, max_price, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Distribute volume into price bins
        volume_distribution = np.zeros(num_bins)
        
        for i in range(window):
            price = window_closes[i]
            volume = window_volumes[i]
            
            # Find the bin index for this price
            bin_idx = np.searchsorted(bin_edges, price) - 1
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            elif bin_idx < 0:
                bin_idx = 0
                
            volume_distribution[bin_idx] += volume
        
        # Normalize the distribution
        total_volume = np.sum(volume_distribution)
        if total_volume > 0:
            normalized_distribution = volume_distribution / total_volume
        else:
            normalized_distribution = np.zeros(num_bins)
        
        # Find the Point of Control (POC) - the price level with the highest volume
        poc_idx = np.argmax(volume_distribution)
        poc = bin_centers[poc_idx]
        
        # Calculate the Value Area (70% of volume)
        sorted_idx = np.argsort(volume_distribution)[::-1]  # Sort bins by volume, descending
        cumulative_volume = 0
        value_area_bins = []
        
        for idx in sorted_idx:
            value_area_bins.append(idx)
            cumulative_volume += volume_distribution[idx]
            if cumulative_volume / total_volume >= 0.7:
                break
        
        # Find min and max prices in the value area
        min_va_idx = min(value_area_bins)
        max_va_idx = max(value_area_bins)
        value_area = [bin_edges[min_va_idx], bin_edges[max_va_idx + 1]]
        
        # Prepare results
        result = {
            'point_of_control': poc,
            'value_area': value_area,
            'distribution': list(zip(bin_centers, normalized_distribution))
        }
        
        # Update state
        self.state['poc'] = poc
        self.state['value_area'] = value_area
        
        return result


@register_feature(category="price")
class PriceDistanceFeature(Feature):
    """
    Calculate distance of price from various reference points.
    
    This feature measures how far the current price is from reference
    levels such as moving averages, support/resistance, or previous highs/lows.
    """
    
    def __init__(self, 
                 name: str = "price_distance", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Distance from reference levels"):
        """
        Initialize the price distance feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - reference: Reference type ('ma', 'level', 'high_low', 'bands')
                - period: Period for moving average if reference is 'ma' (default: 20)
                - levels: List of price levels if reference is 'level'
                - lookback: Lookback period for high/low if reference is 'high_low' (default: 20)
                - as_percentage: Whether to return distance as percentage (default: True)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for price distance."""
        return {
            'reference': 'ma',
            'period': 20,
            'as_percentage': True
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_references = ['ma', 'level', 'high_low', 'bands']
        if self.params.get('reference') not in valid_references:
            raise ValueError(f"Reference must be one of {valid_references}")
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate price distance from reference levels.
        
        Args:
            data: Dictionary containing price history and indicators
        
        Returns:
            Dictionary with distance measurements
        """
        reference = self.params.get('reference', 'ma')
        as_percentage = self.params.get('as_percentage', True)
        
        # Check for required data
        if 'Close' not in data:
            return {'distance': np.nan}
            
        prices = data['Close']
        current_price = prices[-1] if isinstance(prices, (list, np.ndarray, pd.Series)) else prices
        
        if reference == 'ma':
            period = self.params.get('period', 20)
            
            # Check if MA is provided in data
            ma_key = f'SMA_{period}'
            if ma_key in data:
                ma_value = data[ma_key][-1] if isinstance(data[ma_key], (list, np.ndarray, pd.Series)) else data[ma_key]
            else:
                # Calculate MA if not provided
                if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < period:
                    return {'distance': np.nan}
                ma_value = np.mean(prices[-period:])
            
            # Calculate distance
            distance = current_price - ma_value
            
            if as_percentage and ma_value != 0:
                distance = (distance / ma_value) * 100
            
            return {'distance': distance, 'reference': ma_value}
            
        elif reference == 'level':
            levels = self.params.get('levels', [])
            
            if not levels:
                return {'distance': np.nan}
                
            # Find the closest level
            closest_level = min(levels, key=lambda x: abs(x - current_price))
            distance = current_price - closest_level
            
            if as_percentage and closest_level != 0:
                distance = (distance / closest_level) * 100
                
            return {'distance': distance, 'reference': closest_level}
            
        elif reference == 'high_low':
            lookback = self.params.get('lookback', 20)
            
            if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < lookback:
                return {'distance': np.nan}
                
            # Get the price range
            price_range = prices[-lookback:]
            high = max(price_range)
            low = min(price_range)
            
            # Calculate distances from high and low
            distance_from_high = current_price - high
            distance_from_low = current_price - low
            
            if as_percentage:
                if high != 0:
                    distance_from_high = (distance_from_high / high) * 100
                if low != 0:
                    distance_from_low = (distance_from_low / low) * 100
                    
            # Calculate normalized position in range (0-100%)
            range_size = high - low
            if range_size > 0:
                position = ((current_price - low) / range_size) * 100
            else:
                position = 50  # If range is zero, return middle position
                
            return {
                'distance_from_high': distance_from_high, 
                'distance_from_low': distance_from_low,
                'position_in_range': position,
                'high': high,
                'low': low
            }
            
        elif reference == 'bands':
            # Check for Bollinger Bands in data
            if 'BB_upper' not in data or 'BB_lower' not in data:
                return {'distance': np.nan}
                
            upper = data['BB_upper'][-1] if isinstance(data['BB_upper'], (list, np.ndarray, pd.Series)) else data['BB_upper']
            lower = data['BB_lower'][-1] if isinstance(data['BB_lower'], (list, np.ndarray, pd.Series)) else data['BB_lower']
            
            # Calculate distances from bands
            distance_from_upper = current_price - upper
            distance_from_lower = current_price - lower
            
            if as_percentage:
                if upper != 0:
                    distance_from_upper = (distance_from_upper / upper) * 100
                if lower != 0:
                    distance_from_lower = (distance_from_lower / lower) * 100
            
            # Calculate normalized position in band range (0-100%)
            band_range = upper - lower
            if band_range > 0:
                position = ((current_price - lower) / band_range) * 100
            else:
                position = 50  # If range is zero, return middle position
                
            return {
                'distance_from_upper': distance_from_upper,
                'distance_from_lower': distance_from_lower,
                'position_in_bands': position,
                'upper': upper,
                'lower': lower
            }
            
        return {'distance': np.nan}  # Fallback
