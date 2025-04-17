#!/usr/bin/env python3
"""
Event System Test

This script tests whether the event system correctly registers handlers
and dispatches events between components.
"""

import logging
import datetime
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the codebase
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType

# Simple handler classes for testing
class TestHandler:
    """Test handler to verify event reception."""
    
    def __init__(self, name):
        self.name = name
        self.events_received = 0
        self.last_event = None
        logger.info(f"Created handler: {name}")
    
    def __call__(self, event):
        """Called when an event is received."""
        self.events_received += 1
        self.last_event = event
        logger.info(f"Handler {self.name} received event #{self.events_received}")

def test_handler_registration():
    """Test that handlers are correctly registered with the event bus."""
    logger.info("=== Testing Handler Registration ===")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handler
    handler = TestHandler("TestHandler")
    
    # Register handler
    event_bus.register(EventType.BAR, handler)
    
    # Verify handler is stored correctly
    if EventType.BAR not in event_bus.handlers:
        logger.error("Event type not found in handlers dict")
        return False
    
    handlers = event_bus.handlers[EventType.BAR]
    if len(handlers) == 0:
        logger.error("No handlers registered for BAR events")
        return False
    
    if handler not in handlers:
        logger.error("Handler not stored correctly in handlers list")
        return False
    
    logger.info("Handler registration test PASSED")
    return True

def test_event_dispatch():
    """Test that events are correctly dispatched to handlers."""
    logger.info("=== Testing Event Dispatch ===")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handler
    handler = TestHandler("DispatchTestHandler")
    
    # Register handler
    event_bus.register(EventType.BAR, handler)
    
    # Create event
    event = Event(
        event_type=EventType.BAR,
        data={"symbol": "SPY", "price": 400.0},
        timestamp=datetime.datetime.now()
    )
    
    # Emit event
    event_bus.emit(event)
    
    # Verify handler received event
    if handler.events_received != 1:
        logger.error(f"Handler received {handler.events_received} events, expected 1")
        return False
    
    if handler.last_event != event:
        logger.error("Handler received wrong event")
        return False
    
    logger.info("Event dispatch test PASSED")
    return True

def test_handler_persistence():
    """Test that handlers persist and aren't garbage collected."""
    logger.info("=== Testing Handler Persistence ===")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handler
    handler = TestHandler("PersistenceTestHandler")
    
    # Register handler
    event_bus.register(EventType.BAR, handler)
    
    # Create event
    event = Event(
        event_type=EventType.BAR,
        data={"symbol": "SPY", "price": 400.0},
        timestamp=datetime.datetime.now()
    )
    
    # Emit event
    event_bus.emit(event)
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Emit another event
    event2 = Event(
        event_type=EventType.BAR,
        data={"symbol": "SPY", "price": 401.0},
        timestamp=datetime.datetime.now()
    )
    event_bus.emit(event2)
    
    # Verify handler received both events
    if handler.events_received != 2:
        logger.error(f"Handler received {handler.events_received} events, expected 2")
        return False
    
    logger.info("Handler persistence test PASSED")
    return True

def test_multiple_handlers():
    """Test that multiple handlers receive the same event."""
    logger.info("=== Testing Multiple Handlers ===")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handlers
    handler1 = TestHandler("Handler1")
    handler2 = TestHandler("Handler2")
    
    # Register handlers
    event_bus.register(EventType.BAR, handler1)
    event_bus.register(EventType.BAR, handler2)
    
    # Create event
    event = Event(
        event_type=EventType.BAR,
        data={"symbol": "SPY", "price": 400.0},
        timestamp=datetime.datetime.now()
    )
    
    # Emit event
    event_bus.emit(event)
    
    # Verify both handlers received event
    if handler1.events_received != 1:
        logger.error(f"Handler1 received {handler1.events_received} events, expected 1")
        return False
    
    if handler2.events_received != 1:
        logger.error(f"Handler2 received {handler2.events_received} events, expected 1")
        return False
    
    logger.info("Multiple handlers test PASSED")
    return True

if __name__ == "__main__":
    # Run tests
    tests = [
        test_handler_registration,
        test_event_dispatch,
        test_handler_persistence,
        test_multiple_handlers
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
                logger.error(f"Test failed: {test.__name__}")
        except Exception as e:
            all_passed = False
            logger.error(f"Error in test {test.__name__}: {e}", exc_info=True)
    
    if all_passed:
        logger.info("All tests PASSED!")
        sys.exit(0)
    else:
        logger.error("Some tests FAILED!")
        sys.exit(1)
