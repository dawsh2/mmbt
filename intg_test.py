#!/usr/bin/env python3
"""
Self-contained test script for event system object preservation.

This script contains all the necessary classes to demonstrate the problem and solution
without modifying any existing files. You can run this directly to verify the approach.
"""

import uuid
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum, auto
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------- Event System Implementation ---------

class EventType(Enum):
    """Event types enum for testing."""
    BAR = auto()
    SIGNAL = auto()
    ORDER = auto()
    POSITION_ACTION = auto()


class Event:
    """
    Base class for all events in the trading system.
    Modified to preserve object references.
    """
    def __init__(self, event_type, data=None, timestamp=None):
        """Initialize an event."""
        self.event_type = event_type
        self._data = data  # Store reference to original object
        self.timestamp = timestamp or datetime.datetime.now()
        self.id = str(uuid.uuid4())
    
    @property
    def data(self):
        """Get the event data (original reference)."""
        return self._data
    
    def get(self, key, default=None):
        """
        Get a value from the event data.
        
        Supports multiple access patterns:
        1. Dictionary access if data is a dict
        2. Attribute access if data has the attribute
        3. Method access if data has a compatible get() method
        """
        # Handle dictionary-like data
        if isinstance(self._data, dict) and key in self._data:
            return self._data[key]
        
        # Handle object attributes
        if hasattr(self._data, key):
            return getattr(self._data, key)
        
        # Handle objects with get method
        if hasattr(self._data, 'get') and callable(self._data.get):
            try:
                return self._data.get(key, default)
            except Exception:
                pass
        
        return default
    
    def __str__(self):
        """String representation of the event."""
        return f"Event(type={self.event_type}, id={self.id}, timestamp={self.timestamp})"


class SignalEvent:
    """Test implementation of a signal event."""
    # Signal constants
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    def __init__(self, signal_value, price, symbol="default", rule_id=None, 
                 metadata=None, timestamp=None):
        """Initialize a signal event."""
        self.signal_value = signal_value
        self.price = price
        self.symbol = symbol
        self.rule_id = rule_id
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.datetime.now()
    
    def get_signal_value(self):
        """Get the signal value."""
        return self.signal_value
    
    def get_symbol(self):
        """Get the symbol."""
        return self.symbol
    
    def get_price(self):
        """Get the price."""
        return self.price
    
    def get_rule_id(self):
        """Get the rule ID."""
        return self.rule_id
    
    def get_metadata(self):
        """Get the metadata."""
        return self.metadata
    
    def __str__(self):
        """String representation."""
        signal_name = "BUY" if self.signal_value == self.BUY else \
                      "SELL" if self.signal_value == self.SELL else "NEUTRAL"
        return f"SignalEvent({signal_name}, {self.symbol}, price={self.price})"


class PositionActionEvent:
    """Test implementation of a position action event."""
    
    def __init__(self, action_type, **kwargs):
        """Initialize position action event."""
        self.action_type = action_type
        self.data = {'action_type': action_type, **kwargs}
    
    def get_action_type(self):
        """Get the action type."""
        return self.action_type
    
    def get(self, key, default=None):
        """Get a value from the data."""
        return self.data.get(key, default)
    
    def __str__(self):
        """String representation."""
        return f"PositionActionEvent(type={self.action_type})"


class EventBus:
    """
    Central event bus for routing events between system components.
    Modified to preserve object references.
    """
    def __init__(self):
        """Initialize event bus."""
        self.handlers = {}
        self.history = []
    
    def register(self, event_type, handler):
        """Register a handler for an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def emit(self, event):
        """
        Emit an event to registered handlers.
        Preserves original event reference.
        """
        # Record event in history
        self.history.append(event)
        
        # Get handlers for this event type
        handlers = self.handlers.get(event.event_type, [])
        
        # Process all handlers with the original event object
        for handler in handlers:
            try:
                if hasattr(handler, 'handle'):
                    handler.handle(event)
                elif callable(handler):
                    handler(event)
            except Exception as e:
                logger.error(f"Error in handler: {str(e)}")


class EventHandler(ABC):
    """
    Base class for all event handlers.
    Modified to preserve object references.
    """
    def __init__(self, event_types):
        """Initialize event handler."""
        if isinstance(event_types, EventType):
            event_types = {event_types}
        elif isinstance(event_types, list):
            event_types = set(event_types)
            
        self.event_types = event_types
        self.enabled = True
    
    def can_handle(self, event_type):
        """Check if this handler can process the event type."""
        return event_type in self.event_types and self.enabled
    
    def handle(self, event):
        """Process an event."""
        if not self.enabled or not self.can_handle(event.event_type):
            return
            
        try:
            self._process_event(event)
        except Exception as e:
            logger.error(f"Error in handler: {str(e)}")
    
    @abstractmethod
    def _process_event(self, event):
        """Internal method to process an event."""
        pass


# --------- Application-specific Components ---------

class TestHandler(EventHandler):
    """Test handler that verifies object type preservation."""
    
    def __init__(self, event_types):
        """Initialize with tracking."""
        super().__init__(event_types)
        self.handled_events = []
        
    def _process_event(self, event):
        """Process an event with type checking."""
        # Save the event for inspection
        self.handled_events.append(event)
        
        # Type checking for signal events
        if event.event_type == EventType.SIGNAL:
            if not isinstance(event.data, SignalEvent):
                logger.error(f"Expected SignalEvent, got {type(event.data)}")
                return
                
            # Access signal properties to verify it works
            signal = event.data
            logger.info(f"Received signal: {signal.get_symbol()} {signal.get_signal_value()} @ {signal.get_price()}")


class PositionManager:
    """Test position manager that processes signals."""
    
    def __init__(self, event_bus=None):
        """Initialize position manager."""
        self.event_bus = event_bus
        self.signals = []
        self.actions = []
    
    def on_signal(self, event_or_signal):
        """Process a signal with proper type checking."""
        # Extract the SignalEvent with proper type checking
        signal = None
        
        # Case 1: Argument is an Event with a SignalEvent in data
        if isinstance(event_or_signal, Event):
            if hasattr(event_or_signal, 'data') and isinstance(event_or_signal.data, SignalEvent):
                signal = event_or_signal.data
            else:
                logger.error(f"Expected Event with SignalEvent in data, got {type(event_or_signal.data)}")
                return []
        # Case 2: Argument is a SignalEvent directly
        elif isinstance(event_or_signal, SignalEvent):
            signal = event_or_signal
        else:
            logger.error(f"Expected Event or SignalEvent, got {type(event_or_signal)}")
            return []
                
        # Store signal for tracking
        self.signals.append(signal)
        
        # Generate a position action based on the signal
        action = self._create_action_from_signal(signal)
        self.actions.append(action)
        
        # Emit position action event if we have an event bus
        if self.event_bus:
            action_event = Event(EventType.POSITION_ACTION, action)
            self.event_bus.emit(action_event)
            
        logger.info(f"Generated action for signal: {signal}")
        return [action]
    
    def _create_action_from_signal(self, signal):
        """Create a position action from a signal."""
        if signal.get_signal_value() == SignalEvent.BUY:
            return PositionActionEvent(
                action_type='entry',
                symbol=signal.get_symbol(),
                direction=1,
                price=signal.get_price()
            )
        elif signal.get_signal_value() == SignalEvent.SELL:
            return PositionActionEvent(
                action_type='entry',
                symbol=signal.get_symbol(),
                direction=-1,
                price=signal.get_price()
            )
        else:
            return PositionActionEvent(
                action_type='none',
                symbol=signal.get_symbol()
            )


# --------- Test Code ---------

def test_signal_reference_preservation():
    """Test that SignalEvent references are preserved through the event system."""
    print("\n=== Testing Signal Reference Preservation ===")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handlers
    test_handler = TestHandler([EventType.SIGNAL])
    position_manager = PositionManager(event_bus)
    
    # Register handlers
    event_bus.register(EventType.SIGNAL, test_handler)
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)
    
    # Create a signal
    signal = SignalEvent(
        signal_value=SignalEvent.BUY,
        price=100.0,
        symbol="AAPL",
        rule_id="test_rule",
        metadata={"confidence": 0.8}
    )
    
    # Store original object ID
    original_id = id(signal)
    print(f"Original signal ID: {original_id}")
    
    # Create and emit event
    event = Event(EventType.SIGNAL, signal)
    event_bus.emit(event)
    
    # Verify test handler received the event
    if not test_handler.handled_events:
        print("❌ Test handler did not receive any events")
        return False
    
    received_event = test_handler.handled_events[0]
    received_signal = received_event.data
    
    # Verify signal preserved its type
    if not isinstance(received_signal, SignalEvent):
        print(f"❌ Signal lost its type, got {type(received_signal)}")
        return False
    print(f"✅ Signal preserved its type: {type(received_signal)}")
    
    # Verify signal preserved its identity (same object)
    received_id = id(received_signal)
    if received_id != original_id:
        print(f"❌ Signal reference changed, got ID {received_id}")
        return False
    print(f"✅ Signal preserved its identity, ID: {received_id}")
    
    # Verify position manager received and processed the signal
    if not position_manager.signals:
        print("❌ Position manager did not receive any signals")
        return False
    
    processed_signal = position_manager.signals[0]
    processed_id = id(processed_signal)
    
    if processed_id != original_id:
        print(f"❌ Position manager received a different signal, ID: {processed_id}")
        return False
    print(f"✅ Position manager received the same signal, ID: {processed_id}")
    
    # Verify we can access signal properties
    try:
        symbol = processed_signal.get_symbol()
        value = processed_signal.get_signal_value()
        price = processed_signal.get_price()
        print(f"✅ Successfully accessed signal properties: {symbol} {value} @ {price}")
    except Exception as e:
        print(f"❌ Error accessing signal properties: {e}")
        return False
    
    # Verify action generation
    if not position_manager.actions:
        print("❌ Position manager did not generate any actions")
        return False
    
    action = position_manager.actions[0]
    if not isinstance(action, PositionActionEvent):
        print(f"❌ Generated action is not a PositionActionEvent, got {type(action)}")
        return False
    
    print(f"✅ Successfully generated action: {action.get_action_type()}")
    
    return True


def test_direct_signal_handling():
    """Test that PositionManager can handle SignalEvent objects directly."""
    print("\n=== Testing Direct Signal Handling ===")
    
    # Create position manager without event bus
    position_manager = PositionManager()
    
    # Create a signal
    signal = SignalEvent(
        signal_value=SignalEvent.SELL,
        price=105.0,
        symbol="MSFT",
        rule_id="test_rule"
    )
    
    # Process signal directly
    actions = position_manager.on_signal(signal)
    
    # Verify signal was processed
    if not position_manager.signals:
        print("❌ Position manager did not store the signal")
        return False
    
    if not actions:
        print("❌ Position manager did not generate any actions")
        return False
    
    processed_signal = position_manager.signals[0]
    if not isinstance(processed_signal, SignalEvent):
        print(f"❌ Processed signal is not a SignalEvent, got {type(processed_signal)}")
        return False
    
    print(f"✅ Successfully processed direct signal: {processed_signal}")
    
    action = actions[0]
    if action.get_action_type() != 'entry' or action.get('direction') != -1:
        print(f"❌ Incorrect action generated: {action}")
        return False
    
    print(f"✅ Successfully generated correct action: {action}")
    
    return True


def run_tests():
    """Run all tests."""
    tests = [
        test_signal_reference_preservation,
        test_direct_signal_handling
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n=== Test Results ===")
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASSED" if result else "FAILED"
        print(f"Test {i+1}: {test.__name__} - {status}")
    
    if all(results):
        print("\n✅ All tests passed! The solution preserves object references correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs for details.")


if __name__ == "__main__":
    run_tests()
