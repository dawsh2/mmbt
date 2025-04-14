"""
Volatility-based regime detectors.
"""

from collections import deque
import numpy as np

from src.regime_detection.detector_base import DetectorBase
from src.regime_detection.regime_type import RegimeType
from src.regime_detection.detector_registry import registry

@registry.register(category="volatility")
class VolatilityRegimeDetector(DetectorBase):
    """
    Regime detector based on market volatility.
    
    This detector identifies volatile and low-volatility markets based on
    the standard deviation of returns over a specified lookback period.
    """
    
    def __init__(self, name=None, config=None):
        """
        Initialize the volatility detector.
        
        Args:
            name: Optional name for the detector
            config: Optional configuration dictionary with parameters:
                - lookback_period: Period for volatility calculation (default: 20)
                - volatility_threshold: Threshold for volatility regimes (default: 0.015)
        """
        super().__init__(name=name, config=config)
        
        # Extract parameters from config
        self.lookback_period = self.config.get('lookback_period', 20)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.015)
        
        # Initialize data structures
        self.close_history = deque(maxlen=self.lookback_period + 1)
        self.returns_history = deque(maxlen=self.lookback_period)
    
    def detect_regime(self, bar):
        """
        Detect the current market regime based on volatility.
        
        Args:
            bar: Bar data dictionary with 'Close' key
            
        Returns:
            RegimeType: The detected market regime
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
        """Reset the detector state."""
        super().reset()
        self.close_history.clear()
        self.returns_history.clear()
