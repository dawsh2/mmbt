"""
Event Utilities Module

Provides helper functions for working with events, including
unpacking/packing event data and creating standardized objects.
"""

from src.events.event_types import EventType
from src.signals import Signal, SignalType

# Bar Event Utilities
def unpack_bar_event(event):
    """Extract bar data from an event object.
    
    Args:
        event: Event object containing bar data
        
    Returns:
        tuple: (bar_dict, symbol, price, timestamp)
    """
    if not hasattr(event, 'data'):
        raise TypeError(f"Expected Event object with data attribute")
        
    bar_event = event.data
    if not hasattr(bar_event, 'bar'):
        raise TypeError(f"Expected BarEvent object in event.data")
        
    bar = bar_event.bar
    symbol = bar.get('symbol', 'unknown')
    price = bar.get('Close')
    timestamp = bar.get('timestamp')
    
    return bar, symbol, price, timestamp

# Signal Utilities
def create_signal(timestamp, signal_type, price, rule_id=None, confidence=1.0, symbol=None, metadata=None):
    """Create a standardized Signal object."""
    return Signal(
        timestamp=timestamp,
        signal_type=signal_type,
        price=price,
        rule_id=rule_id,
        confidence=confidence,
        symbol=symbol,
        metadata=metadata or {}
    )

def unpack_signal_event(event):
    """Extract signal data from an event object."""
    if not hasattr(event, 'data'):
        raise TypeError(f"Expected Event object with data attribute")
        
    signal = event.data
    if not hasattr(signal, 'symbol') or not hasattr(signal, 'signal_type'):
        raise TypeError(f"Expected Signal object in event.data")
        
    symbol = signal.symbol
    price = signal.price if hasattr(signal, 'price') else None
    signal_type = signal.signal_type
    
    return signal, symbol, price, signal_type

# Position Action Utilities
def create_position_action(action_type, symbol, **kwargs):
    """Create a standardized position action dictionary."""
    action = {
        'action': action_type,
        'symbol': symbol,
        **kwargs
    }
    return action
