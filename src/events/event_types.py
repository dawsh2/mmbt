"""
Event Types Module

This module defines the event types used in the trading system's event-driven architecture.
It provides the EventType enumeration and utility functions for event type operations.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Set



class EventType(Enum):
    # Market data events
    BAR = auto()
    TICK = auto()
    MARKET_OPEN = auto()
    MARKET_CLOSE = auto()
    
    # Signal events
    SIGNAL = auto()
    
    # Order events
    ORDER = auto()
    CANCEL = auto()
    MODIFY = auto()
    
    # Execution events
    FILL = auto()
    PARTIAL_FILL = auto()
    REJECT = auto()
    
    # Position events
    POSITION_ACTION = auto()  # New event type for position actions
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()
    POSITION_STOPPED = auto()  # For stop-loss/take-profit events
    
    # Portfolio events
    PORTFOLIO_UPDATE = auto()  # For general portfolio state updates
    EQUITY_UPDATE = auto()     # For equity/PnL updates
    MARGIN_UPDATE = auto()     # For margin requirement updates
    
    # System events
    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    ERROR = auto()
    
    # Analysis events
    METRIC_CALCULATED = auto()
    ANALYSIS_COMPLETE = auto()
    
    # Custom event type
    CUSTOM = auto()
    
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

class BarEvent(Event):
    """Specialized event for bar data."""
    
    def __init__(self, bar_data):
        """Initialize with bar data."""
        super().__init__(EventType.BAR, bar_data)

    def timestamp(self):
        """Get the bar timestamp."""
        return self.bar.get('timestamp')
    
    @property
    def open(self):
        """Get the opening price."""
        return self.data.get('Open')
    
    @property
    def high(self):
        """Get the high price."""
        return self.data.get('High')
    
    @property
    def low(self):
        """Get the low price."""
        return self.data.get('Low')
    
    @property
    def close(self):
        """Get the closing price."""
        return self.data.get('Close')
    
    @property
    def volume(self):
        """Get the volume."""
        return self.data.get('Volume')
    
    @property
    def symbol(self):
        """Get the symbol."""
        return self.data.get('symbol')
        


class BarEvent(Event):
    """Event specifically for market data bars."""
    
    def __init__(self, bar_data, timestamp=None):
        """
        Initialize a bar event.
        
        Args:
            bar_data: Dictionary containing OHLCV data
            timestamp: Optional explicit timestamp (defaults to bar_data's timestamp)
        """
        # Use bar's timestamp if not explicitly provided
        if timestamp is None and isinstance(bar_data, dict):
            timestamp = bar_data.get('timestamp')
            
        super().__init__(EventType.BAR, bar_data, timestamp)
    
    def get_symbol(self):
        """Get the instrument symbol."""
        return self.get('symbol', 'default')
    
    def get_price(self):
        """Get the close price."""
        return self.get('Close')
    
    def get_timestamp(self):
        """Get the bar timestamp."""
        return self.get('timestamp', self.timestamp)
    
    def get_open(self):
        """Get the opening price."""
        return self.get('Open')
    
    def get_high(self):
        """Get the high price."""
        return self.get('High')
    
    def get_low(self):
        """Get the low price."""
        return self.get('Low')
    
    def get_volume(self):
        """Get the volume."""
        return self.get('Volume')
    
    def __repr__(self):
        """String representation of the bar event."""
        symbol = self.get_symbol()
        timestamp = self.get_timestamp()
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
