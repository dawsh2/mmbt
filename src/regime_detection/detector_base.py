"""
Base class for market regime detectors.
"""

from abc import ABC, abstractmethod
from .regime_type import RegimeType

class DetectorBase(ABC):
    """
    Abstract base class for regime detection algorithms.
    
    This class defines the interface that all regime detectors must implement.
    Subclasses should override the detect_regime method to provide their specific
    regime detection logic.
    """
    
    def __init__(self, name=None, config=None):
        """
        Initialize the regime detector.
        
        Args:
            name: Optional name for the detector
            config: Optional configuration dictionary
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.current_regime = RegimeType.UNKNOWN
    
    @abstractmethod
    def detect_regime(self, bar_data):
        """
        Detect the current market regime based on bar data.
        
        Args:
            bar_data: Dictionary containing market data (OHLCV)
            
        Returns:
            RegimeType: The detected market regime
        """
        pass
    
    def reset(self):
        """
        Reset the detector's internal state.
        
        This method should be called when restarting analysis or backtesting.
        """
        self.current_regime = RegimeType.UNKNOWN
    
    def __str__(self):
        """Return a string representation of the detector."""
        return f"{self.name} (current regime: {self.current_regime})"
