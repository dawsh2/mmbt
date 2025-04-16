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

# Import event utilities
from src.events.event_utils import (
    unpack_bar_event,
    create_signal,
    unpack_signal_event,
    create_position_action
)

# Make common classes available directly from events module
__all__ = [
    # Manager
    'EventManager',
    
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
    'SystemEmitter',
    
    # Utilities
    'unpack_bar_event',
    'create_signal',
    'unpack_signal_event',
    'create_position_action'
]
