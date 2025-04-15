"""
Events module initialization.
This module provides the event-driven architecture components for the trading system.
"""

# Import core event components
from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType

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
