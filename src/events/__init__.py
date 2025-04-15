"""
Events module initialization.
This module provides the event-driven architecture components for the trading system.
"""

# Import core event components
from src.events.event_bus import Event, EventBus, EventCacheManager
from src.events.event_types import EventType

# Import event handlers
from src.events.event_handlers import (
    EventHandler, 
    FunctionEventHandler,
    LoggingHandler,
    FilterHandler,
    MarketDataHandler,
    SignalHandler,
    OrderHandler,
    FillHandler
)

# Import additional handlers from our new implementation
from src.events.missing_handlers import (
    DebounceHandler,
    AsyncEventHandler,
    EventHandlerGroup,
    CompositeHandler
)

# Import event emitters
from src.events.event_emitters import (
    EventEmitter, 
    MarketDataEmitter,
    SignalEmitter,
    OrderEmitter,
    FillEmitter
)

# Import additional emitters from our new implementation
from src.events.missing_emitters import (
    PortfolioEmitter,
    SystemEmitter,
    AnalysisEmitter
)

# Make common classes available directly from events module
__all__ = [
    # Core components
    'Event',
    'EventBus',
    'EventType',
    'EventCacheManager',
    
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
    'AnalysisEmitter'
]
