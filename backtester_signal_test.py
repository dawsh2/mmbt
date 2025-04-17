import logging
import datetime
import uuid

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('backtester_signal_test')

# Import event and backtester components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, OrderEvent
from src.events.signal_event import SignalEvent

# Manual backtester signal handler (a simplified version of what should happen)
def backtester_signal_handler(event):
    logger.info("Signal handler called")
    
    # Extract signal from event
    if not hasattr(event, 'data'):
        logger.error("Event has no data attribute")
        return None
    
    signal = event.data
    if not isinstance(signal, SignalEvent):
        logger.error(f"Expected SignalEvent, got {type(signal)}")
        return None
    
    logger.info(f"Processing signal: {signal.get_signal_name()} for {signal.get_symbol()}")
    
    # Skip neutral signals
    if signal.get_signal_value() == SignalEvent.NEUTRAL:
        logger.info("Neutral signal - skipping")
        return None
    
    # Create order
    try:
        order = OrderEvent(
            symbol=signal.get_symbol(),
            direction=signal.get_signal_value(),
            quantity=100,  # Fixed size for testing
            price=signal.get_price(),
            order_type="MARKET",
            timestamp=signal.timestamp or datetime.datetime.now()
        )
        
        logger.info(f"Created order: {order}")
        return order
    except Exception as e:
        logger.error(f"Error creating order: {e}", exc_info=True)
        return None

# Test function
def test_backtester_signal_handling():
    # Create test signal
    test_signal = SignalEvent(
        signal_value=SignalEvent.BUY,
        price=520.76,
        symbol="SPY",
        timestamp=datetime.datetime.now()
    )
    
    # Create event
    event = Event(EventType.SIGNAL, test_signal)
    
    # Process directly
    result = backtester_signal_handler(event)
    
    # Check result
    logger.info(f"Result: {result}")
    
    return result

# Run the test
if __name__ == "__main__":
    order = test_backtester_signal_handling()
