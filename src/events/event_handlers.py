"""
Event Handlers Module

This module defines handlers for processing events in the trading system.
It includes base handler classes and utility functions for event handling.
"""

import logging
import time
import weakref
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable, Set, Type, TypeVar
from functools import wraps

from .event_types import EventType
from .event_bus import Event

# Set up logging
logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """
    Base class for all event handlers.
    
    Event handlers process specific types of events and can be registered
    with the event bus to receive events of those types.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]]):
        """
        Initialize event handler.
        
        Args:
            event_types: Event type(s) this handler processes
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]
            
        self.event_types = set(event_types)
        self.enabled = True
        
    def can_handle(self, event_type: EventType) -> bool:
        """
        Check if this handler can process events of the given type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            True if this handler can process the event type, False otherwise
        """
        return event_type in self.event_types and self.enabled
    
    def handle(self, event: Event) -> None:
        """
        Process an event.
        
        Args:
            event: Event to process
        """
        if not self.enabled:
            return
            
        try:
            self._process_event(event)
        except Exception as e:
            logger.error(f"Error processing event: {str(e)}", exc_info=True)
    
    @abstractmethod
    def _process_event(self, event: Event) -> None:
        """
        Internal method to process an event.
        
        This method must be implemented by subclasses.
        
        Args:
            event: Event to process
        """
        pass
    
    def enable(self) -> None:
        """Enable the handler."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the handler."""
        self.enabled = False


class FunctionEventHandler(EventHandler):
    """
    Event handler that delegates processing to a function.
    
    This handler calls a specified function to process events.
    It's useful for creating handlers on-the-fly without
    defining a new handler class.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]],
                handler_func: Callable[[Event], None]):
        """
        Initialize function-based event handler.
        
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


class EventHandlerGroup:
    """
    Group of event handlers that can be managed together.
    
    This class is useful for organizing related handlers and
    enabling/disabling them as a group.
    """
    
    def __init__(self, name: str, handlers: Optional[List[EventHandler]] = None):
        """
        Initialize event handler group.
        
        Args:
            name: Name for the group
            handlers: Optional list of handlers to add initially
        """
        self.name = name
        self.handlers = []
        
        if handlers:
            for handler in handlers:
                self.add_handler(handler)
    
    def add_handler(self, handler: EventHandler) -> None:
        """
        Add a handler to the group.
        
        Args:
            handler: Event handler to add
        """
        self.handlers.append(handler)
    
    def remove_handler(self, handler: EventHandler) -> bool:
        """
        Remove a handler from the group.
        
        Args:
            handler: Event handler to remove
            
        Returns:
            True if handler was removed, False if not found
        """
        try:
            self.handlers.remove(handler)
            return True
        except ValueError:
            return False
    
    def enable_all(self) -> None:
        """Enable all handlers in the group."""
        for handler in self.handlers:
            handler.enable()
    
    def disable_all(self) -> None:
        """Disable all handlers in the group."""
        for handler in self.handlers:
            handler.disable()
    
    def get_handlers(self) -> List[EventHandler]:
        """
        Get all handlers in the group.
        
        Returns:
            List of event handlers
        """
        return self.handlers.copy()


class LoggingHandler(EventHandler):
    """
    Event handler that logs events.
    
    This handler logs events at specified logging levels
    based on event type.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]],
                log_level: int = logging.INFO):
        """
        Initialize logging handler.
        
        Args:
            event_types: Event type(s) to log
            log_level: Default logging level
        """
        super().__init__(event_types)
        self.log_level = log_level
        self.event_log_levels = {}  # Override log level for specific event types
    
    def set_event_log_level(self, event_type: EventType, log_level: int) -> None:
        """
        Set logging level for a specific event type.
        
        Args:
            event_type: Event type to set level for
            log_level: Logging level to use
        """
        self.event_log_levels[event_type] = log_level
    
    def _process_event(self, event: Event) -> None:
        """
        Log an event at the appropriate level.
        
        Args:
            event: Event to log
        """
        # Get log level for this event type
        log_level = self.event_log_levels.get(event.event_type, self.log_level)
        
        # Format event data for logging
        event_data = str(event.data) if event.data else "No data"
        if isinstance(event.data, dict):
            # More readable format for dict data
            event_data = ", ".join(f"{k}={v}" for k, v in event.data.items())
        
        # Log the event
        logger.log(log_level, f"Event: {event.event_type.name}, ID: {event.id}, Data: {event_data}")


class DebounceHandler(EventHandler):
    """
    Event handler that debounces events before processing.
    
    This handler prevents processing the same event type too frequently
    by enforcing a minimum time between processing events of the same type.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]],
                handler: EventHandler, debounce_seconds: float = 0.1):
        """
        Initialize debounce handler.
        
        Args:
            event_types: Event type(s) this handler processes
            handler: Handler to delegate to after debouncing
            debounce_seconds: Minimum seconds between events of same type
        """
        super().__init__(event_types)
        self.handler = handler
        self.debounce_seconds = debounce_seconds
        self.last_processed = {}  # Mapping of event type to timestamp
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event with debouncing.
        
        Args:
            event: Event to process
        """
        current_time = time.time()
        last_time = self.last_processed.get(event.event_type, 0)
        
        # Check if enough time has passed since last event of this type
        if current_time - last_time >= self.debounce_seconds:
            # Update last processed time
            self.last_processed[event.event_type] = current_time
            
            # Delegate to wrapped handler
            self.handler.handle(event)


class FilterHandler(EventHandler):
    """
    Event handler that filters events based on criteria.
    
    This handler only processes events that pass its filter criteria.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]],
                handler: EventHandler, filter_func: Callable[[Event], bool]):
        """
        Initialize filter handler.
        
        Args:
            event_types: Event type(s) this handler processes
            handler: Handler to delegate to if event passes filter
            filter_func: Function that returns True if event should be processed
        """
        super().__init__(event_types)
        self.handler = handler
        self.filter_func = filter_func
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event if it passes the filter.
        
        Args:
            event: Event to process
        """
        # Only process if filter passes
        if self.filter_func(event):
            self.handler.handle(event)


class AsyncEventHandler(EventHandler):
    """
    Event handler that processes events asynchronously.
    
    This handler processes events in a separate thread to avoid
    blocking the event bus.
    
    Note: This is a simplified example. In a real system, you might want to
    use a proper async framework or thread pool.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]],
                handler: EventHandler):
        """
        Initialize async event handler.
        
        Args:
            event_types: Event type(s) this handler processes
            handler: Handler to delegate to asynchronously
        """
        super().__init__(event_types)
        self.handler = handler
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event asynchronously.
        
        Args:
            event: Event to process
        """
        import threading
        
        # Create and start a thread for processing
        thread = threading.Thread(
            target=self.handler.handle,
            args=(event,),
            daemon=True
        )
        thread.start()


# Handler for market data events
class MarketDataHandler(EventHandler):
    """
    Handler for market data events.
    
    This handler processes bar and tick events, typically
    forwarding them to strategies for signal generation.
    """
    
    def __init__(self, strategy):
        """
        Initialize market data handler.
        
        Args:
            strategy: Strategy to forward data to
        """
        super().__init__([EventType.BAR, EventType.TICK])
        self.strategy = strategy
    
    def _process_event(self, event: Event) -> None:
        """
        Process a market data event.
        
        Args:
            event: Event to process
        """
        if event.event_type == EventType.BAR:
            self.strategy.on_bar(event)
        elif event.event_type == EventType.TICK:
            self.strategy.on_tick(event)


# Handler for signal events
class SignalHandler(EventHandler):
    """
    Handler for signal events.
    
    This handler processes signal events from strategies
    and forwards them to portfolio manager for order generation.
    """
    
    def __init__(self, portfolio_manager):
        """
        Initialize signal handler.
        
        Args:
            portfolio_manager: Portfolio manager to forward signals to
        """
        super().__init__([EventType.SIGNAL])
        self.portfolio_manager = portfolio_manager
    
    def _process_event(self, event: Event) -> None:
        """
        Process a signal event.
        
        Args:
            event: Event to process
        """
        self.portfolio_manager.on_signal(event)


# Handler for order events
class OrderHandler(EventHandler):
    """
    Handler for order events.
    
    This handler processes order, cancel, and modify events
    and forwards them to execution engine.
    """
    
    def __init__(self, execution_engine):
        """
        Initialize order handler.
        
        Args:
            execution_engine: Execution engine to forward orders to
        """
        super().__init__([EventType.ORDER, EventType.CANCEL, EventType.MODIFY])
        self.execution_engine = execution_engine
    
    def _process_event(self, event: Event) -> None:
        """
        Process an order event.
        
        Args:
            event: Event to process
        """
        if event.event_type == EventType.ORDER:
            self.execution_engine.place_order(event.data)
        elif event.event_type == EventType.CANCEL:
            self.execution_engine.cancel_order(event.data)
        elif event.event_type == EventType.MODIFY:
            self.execution_engine.modify_order(event.data)


# Handler for fill events
class FillHandler(EventHandler):
    """
    Handler for fill events.
    
    This handler processes fill events from execution engine
    and updates portfolio positions accordingly.
    """
    
    def __init__(self, portfolio_manager):
        """
        Initialize fill handler.
        
        Args:
            portfolio_manager: Portfolio manager to update
        """
        super().__init__([EventType.FILL, EventType.PARTIAL_FILL])
        self.portfolio_manager = portfolio_manager
    
    def _process_event(self, event: Event) -> None:
        """
        Process a fill event.
        
        Args:
            event: Event to process
        """
        if event.event_type in (EventType.FILL, EventType.PARTIAL_FILL):
            self.portfolio_manager.on_fill(event.data)


# Example usage
if __name__ == "__main__":
    from event_bus import EventBus, Event
    
    # Create event bus
    event_bus = EventBus()
    
    # Create logging handler for all events
    logging_handler = LoggingHandler([
        EventType.BAR,
        EventType.SIGNAL,
        EventType.ORDER,
        EventType.FILL
    ])
    
    # Set specific log levels for some events
    logging_handler.set_event_log_level(EventType.BAR, logging.DEBUG)
    logging_handler.set_event_log_level(EventType.SIGNAL, logging.INFO)
    
    # Register handler
    for event_type in logging_handler.event_types:
        event_bus.register(event_type, logging_handler)
    
    # Create and emit events
    bar_event = Event(
        event_type=EventType.BAR,
        data={
            "symbol": "AAPL",
            "open": 150.0,
            "high": 151.5,
            "low": 149.5,
            "close": 151.0,
            "volume": 1000000
        }
    )
    
    signal_event = Event(
        event_type=EventType.SIGNAL,
        data={
            "symbol": "AAPL",
            "direction": 1,
            "strength": 0.8
        }
    )
    
    # Emit events
    event_bus.emit(bar_event)
    event_bus.emit(signal_event)
