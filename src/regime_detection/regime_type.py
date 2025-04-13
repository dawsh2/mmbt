"""
Enumeration of market regime types for regime detection.
"""

from enum import Enum, auto

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
    
    def __str__(self):
        """Return a readable string representation."""
        return self.name
    
    @classmethod
    def from_string(cls, regime_name):
        """Create a RegimeType from string name."""
        try:
            return cls[regime_name.upper()]
        except KeyError:
            return cls.UNKNOWN
