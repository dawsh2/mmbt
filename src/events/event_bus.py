"""
Event Bus Module

This module provides the event bus infrastructure for the trading system.
It includes the Event class, EventBus class, and related utilities for
event-driven communication between system components.
"""

import uuid
import datetime
import threading
import queue
import weakref
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Set, Type

from src.events.event_base import Event 
from src.events.event_types import EventType
from src.events.event_utils import EventValidator

# Set up logging
logger = logging.getLogger(__name__)


    
class EventBus:
    """
    Central event bus for routing events between system components.
    
    The event bus maintains a registry of handlers for different event types
    and dispatches events to the appropriate handlers when they are emitted.
    """

    def __init__(self, async_mode: bool = False, validate_events: bool = False):
        """
        Initialize event bus.

        Args:
            async_mode: Whether to dispatch events asynchronously
            validate_events: Whether to validate events before processing
        """
        self.handlers = {event_type: [] for event_type in EventType}
        self.async_mode = async_mode
        self.history = []
        self.max_history_size = 1000

        # For async mode
        self.event_queue = queue.Queue() if async_mode else None
        self.dispatch_thread = None
        self.running = False

        # Initialize metrics
        self.metrics = {
            'events_emitted': {event_type: 0 for event_type in EventType},
            'events_processed': {event_type: 0 for event_type in EventType},
            'event_processing_time': {event_type: [] for event_type in EventType},
            'errors': 0,
            'last_event_time': None,
            'event_rate': 0  # events/second
        }
        self.metrics_start_time = datetime.datetime.now()
        self.metrics_last_update = self.metrics_start_time

        # Set validation flag
        self.validate_events = validate_events
        if validate_events:
            self.validator = EventValidator()

    
    def register(self, event_type: EventType, handler) -> None:
        """
        Register a handler for an event type.
        
        Args:
            event_type: Event type to register for
            handler: Handler to register
        """
        # Use weak reference to avoid circular references
        self.handlers[event_type].append(weakref.ref(handler))
    
    def unregister(self, event_type: EventType, handler) -> bool:
        """
        Unregister a handler for an event type.
        
        Args:
            event_type: Event type to unregister from
            handler: Handler to unregister
            
        Returns:
            True if handler was unregistered, False if not found
        """
        # Find the weak reference for this handler
        handler_refs = self.handlers[event_type]
        for i, handler_ref in enumerate(handler_refs):
            # Get the handler from the weak reference
            h = handler_ref()
            
            # If the reference is dead or matches our handler, remove it
            if h is None or h is handler:
                handler_refs.pop(i)
                return True
        
        return False

    def emit(self, event):
        """Emit an event to registered handlers."""
        # Validate event if enabled
        if self.validate_events:
            try:
                self.validator.validate(event)
            except ValueError as e:
                logger.error(f"Event validation failed: {str(e)}")
                # Create and emit an error event instead
                error_event = create_error_event(
                    source="EventBus",
                    message=str(e),
                    error_type="ValidationError",
                    original_event=event
                )
                self._dispatch_event(error_event)
                return

        # Add to history
        self._add_to_history(event)
        
        # Dispatch directly or queue for async dispatch
        if self.async_mode:
            self.event_queue.put(event)
        else:
            self._dispatch_event(event)
    
    def emit_all(self, events: List[Event]) -> None:
        """
        Emit multiple events to all registered handlers.
        
        Args:
            events: List of events to emit
        """
        for event in events:
            self.emit(event)
    
    def _dispatch_event(self, event: Event) -> None:
        """
        Dispatch an event to all registered handlers.
        
        Args:
            event: Event to dispatch
        """
        # Get handlers for this event type
        handler_refs = self.handlers[event.event_type]
        
        # Remove dead references
        self.handlers[event.event_type] = [
            ref for ref in handler_refs if ref() is not None
        ]
        
        # Dispatch to each handler
        for handler_ref in self.handlers[event.event_type]:
            handler = handler_ref()
            if handler is not None:
                try:
                    if hasattr(handler, 'handle'):
                        handler.handle(event)
                    elif callable(handler):
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in handler: {str(e)}", exc_info=True)
    
    def _add_to_history(self, event: Event) -> None:
        """
        Add an event to the history.
        
        Args:
            event: Event to add
        """
        self.history.append(event)
        
        # Remove oldest events if history exceeds max size
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
    
    def clear_history(self) -> None:
        """Clear the event history."""
        self.history = []
    
    def get_history(self, event_type: Optional[EventType] = None,
                   start_time: Optional[datetime.datetime] = None,
                   end_time: Optional[datetime.datetime] = None) -> List[Event]:
        """
        Get event history, optionally filtered.
        
        Args:
            event_type: Optional event type to filter by
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of events matching the filters
        """
        # Start with full history
        filtered_history = self.history
        
        # Filter by event type
        if event_type is not None:
            filtered_history = [
                event for event in filtered_history 
                if event.event_type == event_type
            ]
        
        # Filter by start time
        if start_time is not None:
            filtered_history = [
                event for event in filtered_history 
                if event.timestamp >= start_time
            ]
        
        # Filter by end time
        if end_time is not None:
            filtered_history = [
                event for event in filtered_history 
                if event.timestamp <= end_time
            ]
        
        return filtered_history
    
    def start_dispatch_thread(self) -> None:
        """Start the async dispatch thread."""
        if self.dispatch_thread is not None and self.dispatch_thread.is_alive():
            return
            
        self.running = True
        self.dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True
        )
        self.dispatch_thread.start()
    
    def stop_dispatch_thread(self) -> None:
        """Stop the async dispatch thread."""
        self.running = False
        if self.dispatch_thread is not None:
            self.dispatch_thread.join(timeout=1.0)
            self.dispatch_thread = None
    
    def _dispatch_loop(self) -> None:
        """Main loop for async event dispatching."""
        while self.running:
            try:
                # Get an event from the queue (timeout so we can check running flag)
                event = self.event_queue.get(timeout=0.1)
                self._dispatch_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                # Queue empty, just continue
                pass
            except Exception as e:
                logger.error(f"Error in dispatch loop: {str(e)}", exc_info=True)


def _update_metrics(self, event_type, processing_time=None):
    """Update event metrics."""
    # Count emitted event
    self.metrics['events_emitted'][event_type] += 1
    
    # Update timing
    now = datetime.datetime.now()
    self.metrics['last_event_time'] = now
    
    # Update event rate (every 10 events)
    total_events = sum(self.metrics['events_emitted'].values())
    if total_events % 10 == 0:
        elapsed = (now - self.metrics_start_time).total_seconds()
        if elapsed > 0:
            self.metrics['event_rate'] = total_events / elapsed
    
    # Add processing time if provided
    if processing_time is not None:
        self.metrics['events_processed'][event_type] += 1
        self.metrics['event_processing_time'][event_type].append(processing_time)

    def get_metrics(self):
        """Get the current event metrics."""
        # Calculate average processing times
        avg_processing_times = {}
        for event_type, times in self.metrics['event_processing_time'].items():
            if times:
                avg_processing_times[event_type] = sum(times) / len(times)
            else:
                avg_processing_times[event_type] = 0

        # Return compiled metrics
        return {
            'events_emitted': dict(self.metrics['events_emitted']),
            'events_processed': dict(self.metrics['events_processed']),
            'avg_processing_time': avg_processing_times,
            'event_rate': self.metrics['event_rate'],
            'run_time': (datetime.datetime.now() - self.metrics_start_time).total_seconds(),
            'errors': self.metrics['errors']
        }                
    
    def reset(self) -> None:
        """Reset the event bus state."""
        # Stop async dispatch if active
        if self.async_mode:
            self.stop_dispatch_thread()
        
        # Clear handlers and history
        self.handlers = {event_type: [] for event_type in EventType}
        self.clear_history()
        
        # Restart async dispatch if needed
        if self.async_mode:
            self.event_queue = queue.Queue()
            self.start_dispatch_thread()





class EventCacheManager:
    """
    Manager for caching events for improved performance.
    
    This class maintains caches of events by type to prevent
    regenerating the same events repeatedly.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Initialize event cache manager.
        
        Args:
            max_cache_size: Maximum size of each event type cache
        """
        self.caches = {event_type: {} for event_type in EventType}
        self.max_cache_size = max_cache_size
    
    def get_cached_event(self, event_type: EventType, key: str) -> Optional[Event]:
        """
        Get cached event if available.
        
        Args:
            event_type: Type of event
            key: Cache key for the event
            
        Returns:
            Cached event or None if not found
        """
        return self.caches[event_type].get(key)
    
    def cache_event(self, event: Event, key: str) -> None:
        """
        Cache an event.
        
        Args:
            event: Event to cache
            key: Cache key for the event
        """
        cache = self.caches[event.event_type]
        
        # Add to cache
        cache[key] = event
        
        # Enforce max cache size
        if len(cache) > self.max_cache_size:
            # Remove oldest keys first (we'll use a simple approach here)
            for k in list(cache.keys())[:1]:
                del cache[k]
    
    def clear_cache(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear event cache.
        
        Args:
            event_type: Optional event type to clear (None for all)
        """
        if event_type is None:
            # Clear all caches
            self.caches = {event_type: {} for event_type in EventType}
        else:
            # Clear specific cache
            self.caches[event_type] = {}            


# Example usage
if __name__ == "__main__":
    # Define a simple handler
    class SimpleHandler:
        def handle(self, event):
            print(f"Handling event: {event}")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handler
    handler = SimpleHandler()
    
    # Register handler
    event_bus.register(EventType.BAR, handler)
    
    # Create and emit an event
    event = Event(
        event_type=EventType.BAR,
        data={"symbol": "AAPL", "close": 150.75},
        timestamp=datetime.datetime.now()
    )
    event_bus.emit(event)
    
    # Create an event cache manager
    cache_manager = EventCacheManager()
    
    # Cache an event
    cache_manager.cache_event(event, key="AAPL_2023-06-15")
    
    # Retrieve cached event
    cached_event = cache_manager.get_cached_event(EventType.BAR, key="AAPL_2023-06-15")
    print(f"Cached event: {cached_event}")


    
