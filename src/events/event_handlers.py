"""
Event Handlers Module

This module defines handlers for processing events in the trading system.
It includes base handler classes and specialized implementations for event handling.
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable, Set, Type, TypeVar
from functools import wraps

from src.events.event_types import EventType
from src.events.event_bus import Event

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
        
        if not self.can_handle(event.event_type):
            logger.warning(f"Handler {self} cannot handle event type {event.event_type}")
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
        
    def __str__(self) -> str:
        """String representation of the handler."""
        return f"{self.__class__.__name__}(event_types={[et.name for et in self.event_types]})"


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
        else:
            logger.debug(f"Debounced event {event.event_type} (too soon after previous)")


class FilterHandler(EventHandler):
    """
    Event handler that filters events based on criteria.
    
    This handler only processes events that pass its filter criteria.
    It's useful for filtering out events that don't meet certain conditions.
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
        else:
            logger.debug(f"Event {event.id} filtered out")


class AsyncEventHandler(EventHandler):
    """
    Event handler that processes events asynchronously.
    
    This handler processes events in a separate thread to avoid
    blocking the event bus. It's useful for handlers that may take
    significant time to process an event.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]],
                handler: EventHandler, max_workers: int = 5):
        """
        Initialize async event handler.
        
        Args:
            event_types: Event type(s) this handler processes
            handler: Handler to delegate to asynchronously
            max_workers: Maximum number of worker threads
        """
        super().__init__(event_types)
        self.handler = handler
        self.max_workers = max_workers
        self.active_workers = 0
        self.lock = threading.Lock()
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event asynchronously.
        
        Args:
            event: Event to process
        """
        # Check if we've reached the maximum number of workers
        with self.lock:
            if self.active_workers >= self.max_workers:
                logger.warning(f"Max workers ({self.max_workers}) reached, processing synchronously")
                self.handler.handle(event)
                return
            self.active_workers += 1
        
        # Process in a separate thread
        thread = threading.Thread(
            target=self._worker,
            args=(event,),
            daemon=True  # Make thread a daemon so it doesn't block program exit
        )
        thread.start()
    
    def _worker(self, event: Event) -> None:
        """
        Worker thread that processes an event and updates active worker count.
        
        Args:
            event: Event to process
        """
        try:
            self.handler.handle(event)
        except Exception as e:
            logger.error(f"Error in async handler: {e}", exc_info=True)
        finally:
            with self.lock:
                self.active_workers -= 1


class CompositeHandler(EventHandler):
    """
    Event handler that delegates to multiple handlers.
    
    This handler allows multiple handlers to process the same event.
    It's useful for implementing multiple independent reactions to an event.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]], 
                handlers: List[EventHandler]):
        """
        Initialize composite handler.
        
        Args:
            event_types: Event type(s) this handler processes
            handlers: List of handlers to delegate to
        """
        super().__init__(event_types)
        self.handlers = handlers
    
    def _process_event(self, event: Event) -> None:
        """
        Process an event by delegating to all handlers.
        
        Args:
            event: Event to process
        """
        for handler in self.handlers:
            try:
                if handler.can_handle(event.event_type):
                    handler.handle(event)
            except Exception as e:
                logger.error(f"Error in composite handler delegate: {e}", exc_info=True)


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
    
    def register_all(self, event_bus) -> None:
        """
        Register all handlers in the group with an event bus.
        
        Args:
            event_bus: Event bus to register handlers with
        """
        for handler in self.handlers:
            for event_type in handler.event_types:
                event_bus.register(event_type, handler)
    
    def unregister_all(self, event_bus) -> None:
        """
        Unregister all handlers in the group from an event bus.
        
        Args:
            event_bus: Event bus to unregister handlers from
        """
        for handler in self.handlers:
            for event_type in handler.event_types:
                event_bus.unregister(event_type, handler)


# Domain-specific event handlers

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
            if hasattr(self.strategy, 'on_bar'):
                self.strategy.on_bar(event)
            else:
                logger.warning("Strategy does not have on_bar method")
                
        elif event.event_type == EventType.TICK:
            if hasattr(self.strategy, 'on_tick'):
                self.strategy.on_tick(event)
            else:
                logger.warning("Strategy does not have on_tick method")




                
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
        if hasattr(self.portfolio_manager, 'on_signal'):
            self.portfolio_manager.on_signal(event)
        else:
            logger.warning("Portfolio manager does not have on_signal method")


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
            if hasattr(self.execution_engine, 'place_order'):
                self.execution_engine.place_order(event.data)
            elif hasattr(self.execution_engine, 'on_order'):
                self.execution_engine.on_order(event)
            else:
                logger.warning("Execution engine does not have order handling methods")
                
        elif event.event_type == EventType.CANCEL:
            if hasattr(self.execution_engine, 'cancel_order'):
                self.execution_engine.cancel_order(event.data)
            else:
                logger.warning("Execution engine does not have cancel_order method")
                
        elif event.event_type == EventType.MODIFY:
            if hasattr(self.execution_engine, 'modify_order'):
                self.execution_engine.modify_order(event.data)
            else:
                logger.warning("Execution engine does not have modify_order method")


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
            if hasattr(self.portfolio_manager, 'on_fill'):
                self.portfolio_manager.on_fill(event.data)
            else:
                logger.warning("Portfolio manager does not have on_fill method")


class EventProcessor:
    """
    Interface for components that process events.
    
    All components in the system that need to handle events should implement
    this interface or inherit from a class that implements it.
    """
    
    def on_event(self, event):
        """
        Process any event based on its type.
        
        Args:
            event: Event to process
            
        Returns:
            Result of event processing (varies by event type)
        """
        # Dispatch based on event type
        if event.event_type == EventType.BAR:
            return self.on_bar(event)
        elif event.event_type == EventType.SIGNAL:
            return self.on_signal(event)
        elif event.event_type == EventType.ORDER:
            return self.on_order(event)
        elif event.event_type == EventType.FILL:
            return self.on_fill(event)
        else:
            return None
    
    def on_bar(self, event):
        """
        Process a bar event (market data).
        
        Args:
            event: Event with a BarEvent data payload
            
        Returns:
            Depends on the component
        """
        raise NotImplementedError("Subclasses must implement on_bar")
    
    def on_signal(self, event):
        """
        Process a signal event.
        
        Args:
            event: Event with a Signal data payload
            
        Returns:
            Depends on the component
        """
        raise NotImplementedError("Subclasses must implement on_signal")
    
    def on_order(self, event):
        """
        Process an order event.
        
        Args:
            event: Event with an Order data payload
            
        Returns:
            Depends on the component
        """
        raise NotImplementedError("Subclasses must implement on_order")
    
    def on_fill(self, event):
        """
        Process a fill event.
        
        Args:
            event: Event with a Fill data payload
            
        Returns:
            Depends on the component
        """
        raise NotImplementedError("Subclasses must implement on_fill")                
