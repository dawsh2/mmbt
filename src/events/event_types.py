"""
Event Types Module

This module defines the event types used in the trading system's event-driven architecture.
It provides the EventType enumeration and utility functions for event type operations.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Set




class EventType(Enum):
    """
    Enumeration of event types in the trading system.
    
    Each event type represents a specific kind of notification or action
    within the system that components can emit or listen for.
    """
    
    # Market data events
    BAR = auto()             # New price bar (e.g., 1-min, 1-hour, daily)
    TICK = auto()            # New tick data
    MARKET_OPEN = auto()     # Market opening
    MARKET_CLOSE = auto()    # Market closing
    
    # Signal events
    SIGNAL = auto()          # Trading signal from strategy
    
    # Order events
    ORDER = auto()           # Order request
    CANCEL = auto()          # Order cancellation
    MODIFY = auto()          # Order modification
    
    # Execution events
    FILL = auto()            # Order fill (complete execution)
    PARTIAL_FILL = auto()    # Partial order fill
    REJECT = auto()          # Order rejection
    
    # Portfolio events
    POSITION_OPENED = auto() # Position opened
    POSITION_CLOSED = auto() # Position closed
    POSITION_MODIFIED = auto() # Position modified
    
    # System events
    START = auto()           # System start
    STOP = auto()            # System stop
    PAUSE = auto()           # System pause
    RESUME = auto()          # System resume
    ERROR = auto()           # System error
    
    # Analysis events
    METRIC_CALCULATED = auto()  # Performance metric calculated
    ANALYSIS_COMPLETE = auto()  # Analysis process completed
    
    # Custom event type
    CUSTOM = auto()          # Custom event
    
    @classmethod
    def market_data_events(cls) -> Set['EventType']:
        """
        Get a set of all market data related event types.
        
        Returns:
            Set of market data event types
        """
        return {cls.BAR, cls.TICK, cls.MARKET_OPEN, cls.MARKET_CLOSE}
    
    @classmethod
    def order_events(cls) -> Set['EventType']:
        """
        Get a set of all order related event types.
        
        Returns:
            Set of order event types
        """
        return {cls.ORDER, cls.CANCEL, cls.MODIFY, cls.FILL, cls.PARTIAL_FILL, cls.REJECT}
    
    @classmethod
    def position_events(cls) -> Set['EventType']:
        """
        Get a set of all position related event types.
        
        Returns:
            Set of position event types
        """
        return {cls.POSITION_OPENED, cls.POSITION_CLOSED, cls.POSITION_MODIFIED}
    
    @classmethod
    def system_events(cls) -> Set['EventType']:
        """
        Get a set of all system related event types.
        
        Returns:
            Set of system event types
        """
        return {cls.START, cls.STOP, cls.PAUSE, cls.RESUME, cls.ERROR}
    
    @classmethod
    def from_string(cls, name: str) -> 'EventType':
        """
        Get event type from string name.
        
        Args:
            name: String name of event type (case insensitive)
            
        Returns:
            EventType enum value
            
        Raises:
            ValueError: If no matching event type found
        """
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"No event type with name: {name}")

        
class BarEvent:
    """
    Event wrapper for bar data.
    
    This class wraps raw bar data dictionaries to provide a consistent
    interface for strategy components that process market data.
    
    Components should always access data through the methods provided by this class
    rather than accessing the underlying dictionary directly.
    """
    def __init__(self, bar_data):
        self.bar = bar_data
        
    def get(self, key, default=None):
        """
        Get a value from the bar data.
        
        Args:
            key: Dictionary key to retrieve
            default: Default value if key is not found
            
        Returns:
            Value for the key or default
        """
        return self.bar.get(key, default)
        
    def get_symbol(self):
        """Get the instrument symbol."""
        return self.bar.get('symbol', 'default')
        
    def get_price(self):
        """Get the close price."""
        return self.bar.get('Close')
        
    def get_timestamp(self):
        """Get the bar timestamp."""
        return self.bar.get('timestamp')
        
    def __repr__(self):
        symbol = self.bar.get('symbol', 'unknown')
        timestamp = self.bar.get('timestamp', 'unknown')
        return f"BarEvent({symbol} @ {timestamp})"


# Event type categories with descriptions
EVENT_TYPE_DESCRIPTIONS = {
    EventType.BAR: "New price bar with OHLCV data",
    EventType.TICK: "Individual tick data with price and volume",
    EventType.MARKET_OPEN: "Market opening notification",
    EventType.MARKET_CLOSE: "Market closing notification",
    
    EventType.SIGNAL: "Trading signal generated by a strategy",
    
    EventType.ORDER: "Order request for execution",
    EventType.CANCEL: "Order cancellation request",
    EventType.MODIFY: "Order modification request",
    
    EventType.FILL: "Order completely filled",
    EventType.PARTIAL_FILL: "Order partially filled",
    EventType.REJECT: "Order rejected by broker or exchange",
    
    EventType.POSITION_OPENED: "New position opened",
    EventType.POSITION_CLOSED: "Existing position closed",
    EventType.POSITION_MODIFIED: "Position size or parameters modified",
    
    EventType.START: "System or component start",
    EventType.STOP: "System or component stop",
    EventType.PAUSE: "System or component pause",
    EventType.RESUME: "System or component resume",
    EventType.ERROR: "System or component error",
    
    EventType.METRIC_CALCULATED: "Performance metric calculation completed",
    EventType.ANALYSIS_COMPLETE: "Analysis process completed",
    
    EventType.CUSTOM: "Custom event type"
}


def get_event_description(event_type: EventType) -> str:
    """
    Get description for an event type.
    
    Args:
        event_type: Event type
        
    Returns:
        Description of the event type
    """
    return EVENT_TYPE_DESCRIPTIONS.get(event_type, "No description available")


def get_all_event_types_with_descriptions() -> Dict[str, str]:
    """
    Get dictionary of all event types with descriptions.
    
    Returns:
        Dictionary mapping event type names to descriptions
    """
    return {event_type.name: get_event_description(event_type) 
            for event_type in EventType}


# Example usage
if __name__ == "__main__":
    # Print all event types with descriptions
    for event_type, description in get_all_event_types_with_descriptions().items():
        print(f"{event_type}: {description}")
        
    # Get event type from string
    bar_event = EventType.from_string("BAR")
    print(f"Event type from string 'BAR': {bar_event}")
    
    # Get market data events
    market_data_events = EventType.market_data_events()
    print(f"Market data events: {[e.name for e in market_data_events]}")
