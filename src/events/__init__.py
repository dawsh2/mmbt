"""
Event System Package

This package provides the event-driven infrastructure for the trading system.
It includes event definitions, the event bus, handlers, and utilities for event-based communication.
"""

# Core event components
from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent
from src.events.event_bus import EventBus
from src.events.signal_event import SignalEvent

# Event handlers
from src.events.event_handlers import (
    EventHandler,
    FunctionEventHandler,
    LoggingHandler,
    DebounceHandler,
    FilterHandler,
    AsyncEventHandler,
    CompositeHandler,
    EventHandlerGroup,
    MarketDataHandler,
    SignalHandler,
    OrderHandler,
    FillHandler,
    EventProcessor
)

# Event emitters
from src.events.event_emitters import (
    EventEmitter,
    MarketDataEmitter,
    SignalEmitter,
    OrderEmitter,
    FillEmitter,
    PortfolioEmitter,
    SystemEmitter
)

# Portfolio events
from src.events.portfolio_events import (
    PositionActionEvent,
    PortfolioUpdateEvent,
    PositionOpenedEvent,
    PositionClosedEvent
)

# Event utilities and validation
from src.events.event_schema import (
    EventSchema,
    validate_event_data,
    validate_signal_event,
    validate_bar_event,
    get_schema_documentation,
    BAR_SCHEMA,
    SIGNAL_SCHEMA,
    ORDER_SCHEMA,
    FILL_SCHEMA
)

from src.events.event_utils import (
    unpack_bar_event,
    create_bar_event,
    create_signal,
    create_signal_from_numeric,
    unpack_signal_event,
    create_position_action,
    create_error_event,
    MetricsCollector,
    EventValidator,
    ErrorHandler
)

# Optional component if present in your codebase
from src.events.event_manager import EventManager

# Version info
__version__ = '0.1.0'
