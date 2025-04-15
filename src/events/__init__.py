"""
Events module initialization.
This module provides the event-driven architecture components for the trading system.
"""

# Import core event components
from src.events.event_bus import Event, EventBus, EventCacheManager
from src.events.event_types import EventType, BarEvent
from src.events.event_manager import EventManager

# Import event handlers
from src.events.event_handlers import (
    EventHandler, 
    FunctionEventHandler,
    LoggingHandler,
    FilterHandler,
    DebounceHandler,
    AsyncEventHandler,
    EventHandlerGroup,
    CompositeHandler,
    MarketDataHandler,
    SignalHandler,
    OrderHandler,
    FillHandler
)

# Import event emitters
from src.events.event_emitters import (
    EventEmitter, 
    MarketDataEmitter,
    SignalEmitter,
    OrderEmitter,
    FillEmitter,
    PortfolioEmitter,
    SystemEmitter
)

# Make common classes available directly from events module
__all__ = [
    # Manager
    'EventsManager',
    
    # Core components
    'Event',
    'EventBus',
    'EventType',
    'EventCacheManager',
    'BarEvent',
    
    # Handlers
    'EventHandler',
    'FunctionEventHandler',
    'LoggingHandler',
    'FilterHandler',
    'DebounceHandler',
    'AsyncEventHandler',
    'EventHandlerGroup',
    'CompositeHandler',
    'MarketDataHandler',
    'SignalHandler',
    'OrderHandler',
    'FillHandler',
    
    # Emitters
    'EventEmitter',
    'MarketDataEmitter',
    'SignalEmitter',
    'OrderEmitter',
    'FillEmitter',
    'PortfolioEmitter',
    'SystemEmitter'
]
