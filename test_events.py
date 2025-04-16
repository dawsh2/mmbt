"""
Minimal Test for Event System

This is a simplified test file that just tests the basic event system functionality.
"""

import unittest
import datetime

# Only import the event-related modules
from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent
from src.events.event_bus import EventBus
from src.events.signal_event import SignalEvent
from src.events.event_utils import get_event_timestamp, get_event_symbol


class TestEventSystem(unittest.TestCase):
    """Basic tests for the event system."""
    
    def test_event_creation(self):
        """Test basic Event creation."""
        event = Event(EventType.BAR, {"test": True})
        self.assertEqual(event.event_type, EventType.BAR)
        self.assertEqual(event.data, {"test": True})
        self.assertIsNotNone(event.timestamp)
        self.assertIsNotNone(event.id)
    
    def test_bar_event(self):
        """Test BarEvent creation and accessors."""
        bar_data = {
            "timestamp": datetime.datetime.now(),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1000,
            "symbol": "TEST"
        }
        
        bar_event = BarEvent(bar_data)
        self.assertEqual(bar_event.get_symbol(), "TEST")
        self.assertEqual(bar_event.get_price(), 100.5)
        self.assertEqual(bar_event.get_open(), 100.0)
        self.assertEqual(bar_event.get_high(), 101.0)
        self.assertEqual(bar_event.get_low(), 99.0)
        self.assertEqual(bar_event.get_volume(), 1000)
    
    def test_event_utils(self):
        """Test event utilities functions."""
        # Create events
        timestamp = datetime.datetime.now()
        bar_data = {
            "timestamp": timestamp,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1000,
            "symbol": "TEST"
        }
        
        bar_event = BarEvent(bar_data)
        event = Event(EventType.BAR, bar_event, timestamp)
        
        # Test get_event_timestamp
        extracted_timestamp = get_event_timestamp(event)
        self.assertEqual(extracted_timestamp, timestamp)
        
        # Test get_event_symbol
        extracted_symbol = get_event_symbol(event)
        self.assertEqual(extracted_symbol, "TEST")
    
    def test_event_bus(self):
        """Test EventBus functionality."""
        # Create event bus
        event_bus = EventBus()
        
        # Create event handler
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        # Register handler
        event_bus.register(EventType.BAR, handler)
        
        # Create and emit event
        event = Event(EventType.BAR, {"test": True})
        event_bus.emit(event)
        
        # Check handler was called
        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0], event)


if __name__ == "__main__":
    unittest.main()
