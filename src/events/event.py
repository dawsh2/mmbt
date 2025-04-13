"""
Event system for the trading application.

This module provides an event-driven architecture for communication
between different components of the trading system.
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime
import uuid
import weakref
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of event types in the trading system."""
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
    
    # Portfolio events
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()
    
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


class Event:
    """
    Base class for all events in the trading system.
    
    Each event has a type, timestamp, and optional data payload.
    """
    
    def __init__(self, event_type: EventType, data: Any = None, 
                 timestamp: Optional[datetime] = None):
        """
        Initialize an event.
        
        Args:
            event_type: Type of the event
            data: Optional data payload
            timestamp: Event timestamp (defaults to current time)
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.id = str(uuid.uuid4())  # Unique event ID
        
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event({self.event_type.name}, {self.timestamp}, {self.id})"
        
    def __repr__(self) -> str:
        """Detailed representation of the event."""
        return f"Event(type={self.event_type.name}, id={self.id}, timestamp={self.timestamp}, data={repr(self.data)})"


class EventHandler:
    """
    Base class for event handlers.
    
    Event handlers process events of specific types.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]]):
        """
        Initialize an event handler.
        
        Args:
            event_types: Event type(s) this handler processes
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]
            
        self.event_types = set(event_types)
        
    def can_handle(self, event_type: EventType) -> bool:
        """
        Check if this handler can process events of the given type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            True if this handler can process the event type, False otherwise
        """
        return event_type in self.event_types
        
    def handle(self, event: Event) -> None:
        """
        Process an event.
        
        Args:
            event: Event to process
        """
        if not self.can_handle(event.event_type):
            logger.warning(f"Handler {self} cannot handle event type {event.event_type}")
            return
            
        try:
            self._process_event(event)
        except Exception as e:
            logger.error(f"Error processing event {event}: {str(e)}")
            
    def _process_event(self, event: Event) -> None:
        """
        Internal method to process an event.
        
        Args:
            event: Event to process
        """
        raise NotImplementedError("Subclasses must implement _process_event")


class FunctionEventHandler(EventHandler):
    """
    Event handler that delegates processing to a function.
    
    This allows for simple function-based event handlers without
    needing to create full handler classes.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]], 
                 handler_func: Callable[[Event], None]):
        """
        Initialize a function-based event handler.
        
        Args:
            event_types: Event type(s) this handler processes
            handler_func: Function to call for event processing
        """
        super().__init__(event_types)
        self.handler_func = handler_func
        
    def _process_event(self, event: Event) -> None:
        """
        Process an event by calling the handler function.
        
        Args:
            event: Event to process
        """
        self.handler_func(event)


class EventBus:
    """
    Central event bus for routing events between system components.
    
    The event bus maintains a registry of handlers for different event types
    and dispatches events to the appropriate handlers.
    """
    
    def __init__(self, async_mode: bool = False):
        """
        Initialize the event bus.
        
        Args:
            async_mode: Whether to dispatch events asynchronously
        """
        self.handlers: Dict[EventType, List[weakref.ReferenceType]] = {
            event_type: [] for event_type in EventType
        }
        self.async_mode = async_mode
        self.event_history: List[Event] = []
        self.max_history = 1000  # Maximum number of events to keep in history
        
    def register(self, event_type: EventType, handler: Union[EventHandler, Callable]) -> None:
        """
        Register a handler for an event type.
        
        Args:
            event_type: Event type to register for
            handler: Handler to register (EventHandler or callable)
        """
        # Convert callable to FunctionEventHandler if needed
        if callable(handler) and not isinstance(handler, EventHandler):
            handler = FunctionEventHandler(event_type, handler)
            
        if not isinstance(handler, EventHandler):
            raise TypeError("Handler must be an EventHandler instance or callable")
            
        # Store weak reference to handler
        self.handlers[event_type].append(weakref.ref(handler))
        logger.debug(f"Registered handler {handler} for event type {event_type.name}")
        
    def unregister(self, event_type: EventType, handler: Union[EventHandler, Callable]) -> bool:
        """
        Unregister a handler for an event type.
        
        Args:
            event_type: Event type to unregister from
            handler: Handler to unregister
            
        Returns:
            True if handler was unregistered, False if not found
        """
        # Find and remove the handler reference
        handler_refs = self.handlers[event_type]
        for i, ref in enumerate(handler_refs):
            if ref() is handler:
                handler_refs.pop(i)
                logger.debug(f"Unregistered handler {handler} for event type {event_type.name}")
                return True
                
        return False
        
    def emit(self, event: Event) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event: Event to emit
        """
        # Record event in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
            
        # Get handlers for this event type
        handler_refs = self.handlers[event.event_type]
        
        # Add handlers for CUSTOM events if applicable
        if event.event_type != EventType.CUSTOM:
            handler_refs = handler_refs + self.handlers[EventType.CUSTOM]
            
        # Dispatch to handlers
        for ref in handler_refs:
            handler = ref()
            if handler is None:
                # Remove dead references
                handler_refs.remove(ref)
                continue
                
            if self.async_mode:
                # In a real implementation, this would dispatch to a thread or task queue
                # This is a placeholder for demonstration
                logger.debug(f"Async dispatch of {event} to {handler}")
                handler.handle(event)
            else:
                handler.handle(event)
                
        logger.debug(f"Emitted event {event} to {len(handler_refs)} handlers")
        
    def emit_all(self, events: List[Event]) -> None:
        """
        Emit multiple events to all registered handlers.
        
        Args:
            events: List of events to emit
        """
        for event in events:
            self.emit(event)
            
    def clear_history(self) -> None:
        """Clear the event history."""
        self.event_history = []
        
    def get_history(self, event_type: Optional[EventType] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Event]:
        """
        Get event history, optionally filtered by type and time range.
        
        Args:
            event_type: Optional event type to filter by
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of events matching the filters
        """
        filtered = self.event_history
        
        if event_type is not None:
            filtered = [e for e in filtered if e.event_type == event_type]
            
        if start_time is not None:
            filtered = [e for e in filtered if e.timestamp >= start_time]
            
        if end_time is not None:
            filtered = [e for e in filtered if e.timestamp <= end_time]
            
        return filtered
        
    def reset(self) -> None:
        """Reset the event bus state."""
        self.clear_history()
        

class EventEmitter:
    """
    Mixin class for components that emit events.
    
    This provides a standard interface for components to emit events
    to the event bus.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the event emitter.
        
        Args:
            event_bus: Event bus to emit events to
        """
        self.event_bus = event_bus
        
    def emit(self, event_type: EventType, data: Any = None) -> Event:
        """
        Create and emit an event.
        
        Args:
            event_type: Type of event to emit
            data: Optional data payload
            
        Returns:
            The emitted event
        """
        event = Event(event_type, data)
        self.event_bus.emit(event)
        return event
        
    def emit_event(self, event: Event) -> None:
        """
        Emit an existing event.
        
        Args:
            event: Event to emit
        """
        self.event_bus.emit(event)


# Example event handlers for different components
class MarketDataHandler(EventHandler):
    """Event handler for market data events."""
    
    def __init__(self, strategy):
        """
        Initialize the market data handler.
        
        Args:
            strategy: Strategy instance to forward data to
        """
        super().__init__([EventType.BAR, EventType.TICK])
        self.strategy = strategy
        
    def _process_event(self, event: Event) -> None:
        """
        Process market data events.
        
        Args:
            event: Market data event
        """
        if event.event_type == EventType.BAR:
            self.strategy.on_bar(event.data)
        elif event.event_type == EventType.TICK:
            self.strategy.on_tick(event.data)


class SignalHandler(EventHandler):
    """Event handler for signal events."""
    
    def __init__(self, portfolio_manager):
        """
        Initialize the signal handler.
        
        Args:
            portfolio_manager: Portfolio manager to forward signals to
        """
        super().__init__([EventType.SIGNAL])
        self.portfolio_manager = portfolio_manager
        
    def _process_event(self, event: Event) -> None:
        """
        Process signal events.
        
        Args:
            event: Signal event
        """
        self.portfolio_manager.on_signal(event.data)


class OrderHandler(EventHandler):
    """Event handler for order events."""
    
    def __init__(self, execution_engine):
        """
        Initialize the order handler.
        
        Args:
            execution_engine: Execution engine to forward orders to
        """
        super().__init__([EventType.ORDER, EventType.CANCEL, EventType.MODIFY])
        self.execution_engine = execution_engine
        
    def _process_event(self, event: Event) -> None:
        """
        Process order events.
        
        Args:
            event: Order event
        """
        if event.event_type == EventType.ORDER:
            self.execution_engine.place_order(event.data)
        elif event.event_type == EventType.CANCEL:
            self.execution_engine.cancel_order(event.data)
        elif event.event_type == EventType.MODIFY:
            self.execution_engine.modify_order(event.data)
