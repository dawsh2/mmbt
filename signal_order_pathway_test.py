import datetime
import logging
import uuid

# Configure more detailed logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_script')

# Import key components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent, OrderEvent
from src.events.signal_event import SignalEvent
from src.rules.crossover_rules import SMACrossoverRule

# Create a standalone test of the signal-to-order pathway
def test_signal_to_order_pathway():
    # 1. Create a simple event bus
    event_bus = EventBus()
    
    # 2. Set up trackers for events
    signal_events = []
    order_events = []
    
    # 3. Create simple handler functions with detailed logging
    def signal_handler(event):
        logger.debug(f"Signal handler received: {event}")
        signal_events.append(event)
        
        if not hasattr(event, 'data') or not isinstance(event.data, SignalEvent):
            logger.error(f"Event data is not a SignalEvent: {type(event.data) if hasattr(event, 'data') else 'None'}")
            return
            
        signal = event.data
        logger.debug(f"Processing signal: {signal.get_signal_name()} for {signal.get_symbol()}")
        
        # Skip neutral signals
        if signal.get_signal_value() == SignalEvent.NEUTRAL:
            logger.debug("Skipping neutral signal")
            return
        
        # Create order directly
        try:
            order = OrderEvent(
                symbol=signal.get_symbol(),
                direction=signal.get_signal_value(),
                quantity=100,  # Fixed size for testing
                price=signal.get_price(),
                order_type="MARKET",
                timestamp=signal.timestamp or datetime.datetime.now()
            )
            
            logger.debug(f"Created order: {order}")
            
            # Emit order event
            event_bus.emit(Event(EventType.ORDER, order))
            logger.debug("Emitted order event")
        except Exception as e:
            logger.error(f"Error creating order: {e}", exc_info=True)
    
    def order_handler(event):
        logger.debug(f"Order handler received: {event}")
        order_events.append(event)
    
    # 4. Register handlers
    event_bus.register(EventType.SIGNAL, signal_handler)
    event_bus.register(EventType.ORDER, order_handler)
    
    # 5. Create and emit test signal
    test_signal = SignalEvent(
        signal_value=SignalEvent.BUY,
        price=520.76,
        symbol="SPY",
        timestamp=datetime.datetime.now()
    )
    
    logger.info("Emitting test signal event")
    event_bus.emit(Event(EventType.SIGNAL, test_signal))
    
    # 6. Report results
    logger.info(f"Signals received: {len(signal_events)}")
    logger.info(f"Orders created: {len(order_events)}")
    
    return signal_events, order_events

# Run the test
if __name__ == "__main__":
    signals, orders = test_signal_to_order_pathway()
