"""
Signals Compatibility Module

This module provides backward compatibility with code that imports from the 
old src.signals module by forwarding imports to the new signal_event module.
"""

# Import the new signal classes
from src.events.signal_event import SignalEvent

# Create compatibility aliases
Signal = SignalEvent

# Create SignalType enum for backward compatibility
class SignalType:
    """
    Signal type compatibility class to maintain backward compatibility.
    Maps to the constants in SignalEvent.
    """
    BUY = SignalEvent.BUY
    SELL = SignalEvent.SELL
    NEUTRAL = SignalEvent.NEUTRAL

    @classmethod
    def from_value(cls, value):
        """Convert a numeric value to a signal type."""
        if value == 1:
            return cls.BUY
        elif value == -1:
            return cls.SELL
        else:
            return cls.NEUTRAL
