"""
Event Utilities Module

Provides helper functions for working with events, including
unpacking/packing event data and creating standardized objects.
"""

from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent

# Update the unpack_bar_event function
def unpack_bar_event(event) -> Tuple[Dict[str, Any], str, float, datetime]:
    """Extract bar data from an event object.
    
    Args:
        event: Event object containing bar data
        
    Returns:
        tuple: (bar_dict, symbol, price, timestamp)
    """
    if not isinstance(event, Event):
        raise TypeError(f"Expected Event object, got {type(event)}")
    
    # If it's a BarEvent
    if isinstance(event, BarEvent):
        return event.data, event.get_symbol(), event.get_price(), event.get_timestamp()
    
    # If it's a standard event with bar data
    if event.event_type == EventType.BAR:
        bar_data = event.data
        if isinstance(bar_data, dict) and 'Close' in bar_data:
            symbol = bar_data.get('symbol', 'unknown')
            price = bar_data.get('Close')
            timestamp = bar_data.get('timestamp')
            return bar_data, symbol, price, timestamp
    
    raise TypeError(f"Could not extract bar data from event")

# Update the create_bar_event function
def create_bar_event(bar_data: Dict[str, Any], timestamp=None) -> BarEvent:
    """Create a standardized BarEvent object.
    
    Args:
        bar_data: Dictionary containing bar data
        timestamp: Optional explicit timestamp
        
    Returns:
        BarEvent object
    """
    return BarEvent(bar_data, timestamp)


# Signal Event Utilities
def create_signal(
    signal_type, 
    price: float, 
    timestamp=None, 
    symbol: str = None, 
    rule_id: str = None, 
    confidence: float = 1.0, 
    metadata: Optional[Dict[str, Any]] = None
) -> SignalEvent:
    """Create a standardized SignalEvent object.
    
    Args:
        signal_type: Type of signal (BUY, SELL, NEUTRAL)
        price: Price at signal generation
        timestamp: Optional signal timestamp (defaults to now)
        symbol: Optional instrument symbol (defaults to 'default')
        rule_id: Optional ID of the rule that generated the signal
        confidence: Optional confidence score (0-1)
        metadata: Optional additional signal metadata
        
    Returns:
        SignalEvent object
    """
    return SignalEvent(
        signal_type=signal_type,
        price=price,
        symbol=symbol or "default",
        rule_id=rule_id,
        confidence=confidence,
        metadata=metadata or {},
        timestamp=timestamp
    )


def create_signal_from_numeric(
    signal_value: int, 
    price: float, 
    timestamp=None, 
    symbol: str = None, 
    rule_id: str = None, 
    confidence: float = 1.0, 
    metadata: Optional[Dict[str, Any]] = None
) -> SignalEvent:
    """Create a SignalEvent from a numeric signal value (-1, 0, 1).
    
    Args:
        signal_value: Numeric signal value (-1, 0, 1)
        price: Price at signal generation
        timestamp: Optional signal timestamp
        symbol: Optional instrument symbol
        rule_id: Optional ID of the rule that generated the signal
        confidence: Optional confidence score (0-1)
        metadata: Optional additional signal metadata
        
    Returns:
        SignalEvent object
    """
    return SignalEvent.from_numeric(
        signal_value=signal_value,
        price=price,
        symbol=symbol,
        rule_id=rule_id,
        confidence=confidence,
        metadata=metadata,
        timestamp=timestamp
    )


def unpack_signal_event(event) -> Tuple[Any, str, float, Any]:
    """Extract signal data from an event object.
    
    Args:
        event: Event object containing signal data
        
    Returns:
        tuple: (signal, symbol, price, signal_type)
        
    Raises:
        TypeError: If event structure is not as expected
    """
    if not hasattr(event, 'data'):
        raise TypeError(f"Expected Event object with data attribute")
    
    # If it's already a SignalEvent, extract needed properties
    if isinstance(event, SignalEvent):
        return event, event.symbol, event.price, event.signal_type
        
    # If event.data is a SignalEvent
    if isinstance(event.data, SignalEvent):
        signal = event.data
        return signal, signal.symbol, signal.price, signal.signal_type
    
    # If event.data is a dictionary with signal data
    if isinstance(event.data, dict) and 'signal_type' in event.data:
        signal = event.data
        symbol = signal.get('symbol', 'default')
        price = signal.get('price')
        signal_type = signal.get('signal_type')
        return signal, symbol, price, signal_type
    
    raise TypeError(f"Expected SignalEvent or signal data dictionary in event.data")


# Position Action Utilities
def create_position_action(action_type: str, symbol: str, **kwargs) -> Dict[str, Any]:
    """Create a standardized position action dictionary.
    
    Args:
        action_type: Type of action ('entry', 'exit', 'modify')
        symbol: Instrument symbol
        **kwargs: Additional action parameters
        
    Returns:
        Position action dictionary
    """
    action = {
        'action': action_type,
        'symbol': symbol,
        **kwargs
    }
    return action


