"""
Events module initialization.
This module provides the event-driven architecture components for the trading system.
"""

from src.events.event_bus import Event, EventBus
try:
    # Primary imports - this might be where the issue is
    from src.events.event_types import EventType
except ImportError:
    # Alternative import path that might work depending on your project structure
    try:
        from events.event_types import EventType
    except ImportError:
        # Define a minimal EventType for basic functionality if it can't be imported
        from enum import Enum, auto
        
        class EventType(Enum):
            BAR = auto()
            SIGNAL = auto()
            ORDER = auto()
            FILL = auto()
            CUSTOM = auto()
            
            @classmethod
            def market_data_events(cls):
                return {cls.BAR}

# Import event handlers if they exist
try:
    from src.events.event_handlers import (
        EventHandler, 
        FunctionEventHandler,
        LoggingHandler,
        FilterHandler
    )
except ImportError:
    pass

# Import event emitters if they exist
try:
    from src.events.event_emitters import (
        EventEmitter, 
        MarketDataEmitter,
        SignalEmitter,
        OrderEmitter
    )
except ImportError:
    pass

# Make common classes available directly from events module
__all__ = [
    'Event',
    'EventBus',
    'EventType'
]
