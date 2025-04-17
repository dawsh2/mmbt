#!/usr/bin/env python3
"""
Event System Test Script

This script tests the event system to verify that event handlers are properly
registered and events are correctly dispatched.
"""

import logging
import datetime
import time
import random
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the project
try:
    from src.events.event_base import Event
    from src.events.event_bus import EventBus
    from src.events.event_types import EventType
    from src.events.signal_event import SignalEvent
except ImportError as e:
    logger.error(f"Import error: {e}. Make sure you're running from the project root directory.")
    sys.exit(1)

class TestHandler:
    """Test handler for events."""
    
    def __init__(self, name):
        self.name = name
        self.events_received = 0
        self.last_event = None
    
    def __call__(self, event):
        """Handle an event."""
        self.events_received += 1
        self.last_event = event
        logger.info(f"Handler '{self.name}' received event #{self.events_received}: {event.event_type}")
    
    def reset(self):
        """Reset the handler."""
        self.events_received = 0
        self.last_event = None

def test_event_registration_and_dispatch():
    """Test event registration and dispatching."""
    logger.info("--- Testing event registration and dispatching ---")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create handlers
    bar_handler = TestHandler("BarHandler")
    signal_handler = TestHandler("SignalHandler")
    multi_handler = TestHandler("MultiHandler")
    
    # Register handlers
    event_bus.register(EventType.BAR, bar_handler)
    event_bus.register(EventType.SIGNAL, signal_handler)
    
    # Register multi_handler for multiple event types
    event_bus.register(EventType.BAR, multi_handler)
    event_bus.register(EventType.SIGNAL, multi_handler)
    
    # Emit bar event
    bar_event = Event(EventType.BAR, {"symbol": "SPY", "price": 400.0})
    event_bus.emit(bar_event)
    
    # Emit signal event
    signal_event = Event(EventType.SIGNAL, 
                         SignalEvent(signal_value=1, price=400.0, symbol="SPY"))
    event_bus.emit(signal_event)
    
    # Check results
    assert bar_handler.events_received == 1, f"Bar handler received {bar_handler.events_received} events, expected 1"
    assert signal_handler.events_received == 1, f"Signal handler received {signal_handler.events_received} events, expected 1"
    assert multi_handler.events_received == 2, f"Multi handler received {multi_handler.events_received} events, expected 2"
    
    logger.info("All assertions passed")
    return True

def test_handler_persistence():
    """Test that handlers persist and aren't garbage collected."""
    logger.info("--- Testing handler persistence ---")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create and register handler
    handler = TestHandler("PersistenceHandler")
    event_bus.register(EventType.BAR, handler)
    
    # Emit event
    event = Event(EventType.BAR, {"symbol": "SPY", "price": 400.0})
    event_bus.emit(event)
    
    # Check handler was called
    assert handler.events_received == 1, f"Handler received {handler.events_received} events, expected 1"
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Emit another event
    event = Event(EventType.BAR, {"symbol": "SPY", "price": 401.0})
    event_bus.emit(event)
    
    # Check handler was called again
    assert handler.events_received == 2, f"Handler received {handler.events_received} events, expected 2"
    
    logger.info("All assertions passed")
    return True

def test_event_pipeline():
    """Test a complete event pipeline with multiple components."""
    logger.info("--- Testing event pipeline ---")
    
    # Create event bus
    event_bus = EventBus()
    
    # Track events through the pipeline
    events_flow = {
        'bar_events': 0,
        'signal_events': 0,
        'order_events': 0,
        'fill_events': 0
    }
    
    # Create handlers
    class StrategyHandler:
        def __call__(self, event):
            # Process bar event and generate signal
            nonlocal events_flow
            events_flow['bar_events'] += 1
            
            # Create signal event
            if random.random() > 0.5:  # 50% chance to generate signal
                signal_value = 1 if random.random() > 0.5 else -1
                signal = SignalEvent(
                    signal_value=signal_value,
                    price=100.0,
                    symbol="SPY"
                )
                signal_event = Event(EventType.SIGNAL, signal)
                event_bus.emit(signal_event)
                logger.info(f"Strategy generated {'BUY' if signal_value > 0 else 'SELL'} signal")
    
    class PositionManagerHandler:
        def __call__(self, event):
            # Process signal event and generate order
            nonlocal events_flow
            events_flow['signal_events'] += 1
            
            # Create order event
            order_event = Event(EventType.ORDER, {
                'symbol': 'SPY',
                'direction': event.data.get_signal_value(),
                'quantity': 100,
                'price': event.data.get_price()
            })
            event_bus.emit(order_event)
            logger.info("Position manager generated order")
    
    class ExecutionHandler:
        def __call__(self, event):
            # Process order event and generate fill
            nonlocal events_flow
            events_flow['order_events'] += 1
            
            # Create fill event
            fill_event = Event(EventType.FILL, {
                'symbol': 'SPY',
                'direction': event.data['direction'],
                'quantity': event.data['quantity'],
                'price': event.data['price'],
                'commission': 1.0
            })
            event_bus.emit(fill_event)
            logger.info("Execution engine generated fill")
    
    class PortfolioHandler:
        def __call__(self, event):
            # Process fill event
            nonlocal events_flow
            events_flow['fill_events'] += 1
            logger.info("Portfolio processed fill")
    
    # Register handlers
    strategy = StrategyHandler()
    position_manager = PositionManagerHandler()
    execution = ExecutionHandler()
    portfolio = PortfolioHandler()
    
    event_bus.register(EventType.BAR, strategy)
    event_bus.register(EventType.SIGNAL, position_manager)
    event_bus.register(EventType.ORDER, execution)
    event_bus.register(EventType.FILL, portfolio)
    
    # Emit 10 bar events
    for i in range(10):
        bar_event = Event(EventType.BAR, {
            'symbol': 'SPY',
            'timestamp': datetime.datetime.now(),
            'Open': 100.0 + i,
            'High': 101.0 + i,
            'Low': 99.0 + i,
            'Close': 100.5 + i,
            'Volume': 1000000
        })
        event_bus.emit(bar_event)
        
        # Give time for events to propagate
        time.sleep(0.05)
    
    # Verify events flow
    logger.info(f"Events flow: {events_flow}")
    
    # Check that some events made it through the pipeline
    assert events_flow['bar_events'] == 10, f"Expected 10 bar events, got {events_flow['bar_events']}"
    assert events_flow['signal_events'] > 0, "No signal events were processed"
    assert events_flow['order_events'] > 0, "No order events were processed"
    assert events_flow['fill_events'] > 0, "No fill events were processed"
    
    # Verify events are consistent in the pipeline
    assert events_flow['signal_events'] <= events_flow['bar_events'], "More signal events than bar events"
    assert events_flow['order_events'] <= events_flow['signal_events'], "More order events than signal events"
    assert events_flow['fill_events'] <= events_flow['order_events'], "More fill events than order events"
    
    logger.info("All assertions passed")
    return True

if __name__ == "__main__":
    try:
        test_event_registration_and_dispatch()
        test_handler_persistence()
        test_event_pipeline()
        logger.info("All tests passed!")
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
