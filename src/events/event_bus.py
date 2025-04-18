


"""
Event Bus Module

This module provides the event bus infrastructure for the trading system.
Modified to preserve object references without serialization.
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

# src/events/event_bus.py

class EventBus:
    """
    Central event bus for routing events between system components.
    """

    def __init__(self, async_mode=False, validate_events=False):
        """Initialize event bus."""
        self.handlers = {}  # EventType -> list of handlers
        self.event_counts = {}  # For tracking event statistics


    def register(self, event_type, handler):
        """Register a handler for an event type."""
        # Validate that handler is callable
        if not callable(handler):
            logger.error(f"Cannot register non-callable handler: {handler}")
            return

        # Initialize handler list for this event type if not exists
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        # Add handler to the list
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.name if hasattr(event_type, 'name') else event_type}")        
        

    def unregister(self, event_type, handler):
        """Explicitly unregister a handler"""
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)


    def emit(self, event):
        """Emit an event with proper counting."""
        # Initialize event_counts if not present
        if not hasattr(self, 'event_counts'):
            self.event_counts = {}

        # Track event
        event_type = event.event_type
        if event_type in self.event_counts:
            self.event_counts[event_type] += 1
        else:
            self.event_counts[event_type] = 1

        # Log emission for debugging
        if hasattr(event_type, 'name'):
            logger.debug(f"Emitting {event_type.name} event")

        # Process handlers
        if event_type in self.handlers:
            # Create a copy of handlers to prevent modification during iteration
            handlers_copy = list(self.handlers[event_type])
            for handler in handlers_copy:
                try:
                    # Validate handler is callable
                    if not callable(handler):
                        logger.error(f"Handler is not callable: {handler}")
                        continue

                    # Call the handler with the event
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in handler: {e}", exc_info=True)



    #     """
    #     Register a handler for an event type.

    #     Args:
    #         event_type: Event type to register for
    #         handler: Handler to register
    #     """
    #     # Initialize handler list for this event type if not exists
    #     if event_type not in self.handlers:
    #         self.handlers[event_type] = []
            
    #     # For instance methods, store the instance and method name separately
    #     if hasattr(handler, '__self__') and hasattr(handler, '__func__'):
    #         # This is an instance method
    #         instance = handler.__self__
    #         method_name = handler.__func__.__name__

    #         # Store as tuple (instance_ref, method_name)
    #         self.handlers[event_type].append((weakref.ref(instance), method_name))
    #     else:
    #         # Standard function or callable object
    #         self.handlers[event_type].append(weakref.ref(handler))

    def unregister(self, event_type: EventType, handler) -> bool:
        """
        Unregister a handler for an event type.
        
        Args:
            event_type: Event type to unregister from
            handler: Handler to unregister
            
        Returns:
            True if handler was unregistered, False if not found
        """
        if event_type not in self.handlers:
            return False
            
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

    # def emit(self, event: Event) -> None:
    #     """
    #     Emit an event to registered handlers.
        
    #     This method preserves the original event object reference
    #     without serialization or creating copies.
        
    #     Args:
    #         event: Event object to emit
    #     """
    #     # Validate event if enabled
    #     if self.validate_events:
    #         try:
    #             self.validator.validate(event)
    #         except ValueError as e:
    #             logger.error(f"Event validation failed: {str(e)}")
    #             # Create and emit an error event instead but don't validate it
    #             error_event = Event(
    #                 EventType.ERROR,
    #                 {
    #                     'source': "EventBus",
    #                     'message': str(e),
    #                     'error_type': "ValidationError",
    #                     'original_event_id': event.id,
    #                     'original_event_type': event.event_type.name
    #                 },
    #                 datetime.datetime.now()
    #             )
    #             self._dispatch_event(error_event)
    #             return

    #     # Add to history - store original reference
    #     self._add_to_history(event)
        
    #     # Dispatch directly or queue for async dispatch
    #     if self.async_mode:
    #         self.event_queue.put(event)  # Original event reference is preserved
    #     else:
    #         self._dispatch_event(event)  # Pass reference directly to handlers
    
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
            event: Original event object (not a copy)
        """
        # Get the event type
        current_event_type = event.event_type

        # Skip if no handlers for this event type
        if current_event_type not in self.handlers:
            return

        # Get handlers for this event type
        handlers = self.handlers[current_event_type]

        # Process all handlers with the original event object
        for handler in handlers:
            try:
                # Directly call the handler with the original event
                if hasattr(handler, 'handle'):
                    handler.handle(event)  # Pass original event reference
                elif callable(handler):
                    handler(event)  # Pass original event reference
                else:
                    logger.warning(f"Unsupported handler type: {type(handler)}")
            except Exception as e:
                logger.error(f"Error in handler: {str(e)}", exc_info=True)
            
 
    def _add_to_history(self, event: Event) -> None:
        """
        Add an event to the history.
        
        Args:
            event: Event to add (original reference)
        """
        self.history.append(event)  # Store reference, not a copy
        
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
            List of events matching the filters (original references)
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
        if not self.async_mode:
            logger.warning("Cannot start dispatch thread when async_mode is False")
            return
            
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
        if not self.async_mode:
            return
            
        self.running = False
        if self.dispatch_thread is not None:
            self.dispatch_thread.join(timeout=1.0)
            self.dispatch_thread = None
    
    def _dispatch_loop(self) -> None:
        """
        Main loop for async event dispatching.
        
        Passes event references directly from the queue to handlers
        without serialization or copying.
        """
        while self.running:
            try:
                # Get an event from the queue (timeout so we can check running flag)
                event = self.event_queue.get(timeout=0.1)
                
                # Dispatch the original event reference
                self._dispatch_event(event)
                
                self.event_queue.task_done()
            except queue.Empty:
                # Queue empty, just continue
                pass
            except Exception as e:
                logger.error(f"Error in dispatch loop: {str(e)}", exc_info=True)
                self.metrics['errors'] += 1
                
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

    # Fix for src/events/event_bus.py

    def reset(self):
        """Reset the event bus state."""
        # Clear handlers and history
        self.handlers = {}
        if hasattr(self, 'history'):
            self.history = []

        # Reset metrics
        self.event_counts = {}

        # Reset any other state
        if hasattr(self, 'metrics'):
            self.metrics = {
                'events_emitted': {},
                'events_processed': {},
                'event_processing_time': {},
                'errors': 0,
                'last_event_time': None,
                'event_rate': 0
            }
    
 
class EventCacheManager:
    """
    Manager for caching events for improved performance.
    
    This class maintains caches of events by type to prevent
    regenerating the same events repeatedly. It preserves object
    references to ensure type safety throughout the system.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Initialize event cache manager.
        
        Args:
            max_cache_size: Maximum size of each event type cache
        """
        self.caches = {event_type: {} for event_type in EventType}
        self.max_cache_size = max_cache_size
        self.access_timestamps = {event_type: {} for event_type in EventType}
    
    def get_cached_event(self, event_type: EventType, key: str) -> Optional[Event]:
        """
        Get cached event if available.
        
        Args:
            event_type: Type of event
            key: Cache key for the event
            
        Returns:
            Cached event (original reference) or None if not found
        """
        if key in self.caches[event_type]:
            # Update access timestamp
            self.access_timestamps[event_type][key] = datetime.datetime.now()
            return self.caches[event_type][key]
        return None
    
    def cache_event(self, event: Event, key: str) -> None:
        """
        Cache an event.
        
        Args:
            event: Event to cache (stores original reference)
            key: Cache key for the event
        """
        cache = self.caches[event.event_type]
        
        # Add to cache (storing reference, not copy)
        cache[key] = event
        self.access_timestamps[event.event_type][key] = datetime.datetime.now()
        
        # Enforce max cache size
        if len(cache) > self.max_cache_size:
            self._prune_cache(event.event_type)
    
    def _prune_cache(self, event_type: EventType) -> None:
        """
        Prune cache to enforce maximum size.
        
        Uses least recently used (LRU) strategy.
        
        Args:
            event_type: Event type cache to prune
        """
        cache = self.caches[event_type]
        timestamps = self.access_timestamps[event_type]
        
        # Sort keys by access time
        sorted_keys = sorted(timestamps.keys(), key=lambda k: timestamps[k])
        
        # Remove oldest entries until we're back to max size
        keys_to_remove = sorted_keys[:len(cache) - self.max_cache_size]
        for key in keys_to_remove:
            del cache[key]
            del timestamps[key]
    
    def clear_cache(self, event_type: Optional[EventType] = None) -> None:
        """
        Clear event cache.
        
        Args:
            event_type: Optional event type to clear (None for all)
        """
        if event_type is None:
            # Clear all caches
            self.caches = {event_type: {} for event_type in EventType}
            self.access_timestamps = {event_type: {} for event_type in EventType}
        else:
            # Clear specific cache
            self.caches[event_type] = {}
            self.access_timestamps[event_type] = {}

