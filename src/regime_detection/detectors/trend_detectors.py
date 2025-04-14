"""
Trend-based regime detectors.
"""

from collections import deque
import numpy as np

from src.regime_detection.detector_base import DetectorBase
from src.regime_detection.regime_type import RegimeType
from src.regime_detection.detector_registry import registry

@registry.register(category="trend")
class TrendStrengthRegimeDetector(DetectorBase):
    """
    Regime detector based on trend strength using the ADX indicator.
    
    This detector identifies trending and range-bound markets using the Average
    Directional Index (ADX) and directional movement indicators (+DI, -DI).
    """
    
    def __init__(self, name=None, config=None):
        """
        Initialize the trend strength detector.
        
        Args:
            name: Optional name for the detector
            config: Optional configuration dictionary with parameters:
                - adx_period: Period for ADX calculation (default: 14)
                - adx_threshold: Threshold for trend identification (default: 25)
        """
        super().__init__(name=name, config=config)
        
        # Extract parameters from config
        self.adx_period = self.config.get('adx_period', 14)
        self.adx_threshold = self.config.get('adx_threshold', 25)
        
        # Initialize data structures
        self.high_history = deque(maxlen=self.adx_period + 1)
        self.low_history = deque(maxlen=self.adx_period + 1)
        self.close_history = deque(maxlen=self.adx_period + 1)
        self.tr_history = deque(maxlen=self.adx_period)
        self.plus_dm_history = deque(maxlen=self.adx_period)
        self.minus_dm_history = deque(maxlen=self.adx_period)
        self.adx_history = deque(maxlen=self.adx_period)
    
    def detect_regime(self, bar):
        """
        Detect the current market regime based on ADX and directional movements.
        
        Args:
            bar: Bar data dictionary with 'High', 'Low', 'Close' keys
            
        Returns:
            RegimeType: The detected market regime
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
        """Reset the detector state."""
        super().reset()
        self.high_history.clear()
        self.low_history.clear()
        self.close_history.clear()
        self.tr_history.clear()
        self.plus_dm_history.clear()
        self.minus_dm_history.clear()
        self.adx_history.clear()
