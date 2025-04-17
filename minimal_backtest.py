import datetime
import logging
import uuid

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('minimal_backtest')

# Import essential components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent, OrderEvent
from src.events.signal_event import SignalEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule

# Simplified execution engine 
class MinimalExecutionEngine:
    def __init__(self):
        self.orders = []
        self.trades = []
        logger.info("MinimalExecutionEngine initialized")
    
    def on_order(self, event):
        """Process order events."""
        logger.info(f"Order received: {event}")
        
        # Extract order data
        order = event.data
        
        # Just store the order for tracking
        self.orders.append(order)
        
        # Log order details
        if hasattr(order, 'get_symbol'):
            symbol = order.get_symbol()
            quantity = order.get_quantity()
            direction = order.get_direction()
            price = order.get_price()
            logger.info(f"Order stored: {symbol} {direction * quantity} @ {price}")
        else:
            logger.warning(f"Order doesn't have expected methods: {order}")

# Simple strategy
class MinimalStrategy:
    def __init__(self, rule, event_bus):
        self.rule = rule
        self.event_bus = event_bus
        rule.set_event_bus(event_bus)
        logger.info("MinimalStrategy initialized")
    
    def on_bar(self, event):
        """Process bar events and delegate to rule."""
        # Just log and delegate
        logger.debug(f"Strategy received bar event: {event}")
        return self.rule.on_bar(event)

# Direct signal handler function
def direct_signal_handler(event, event_bus, execution_engine):
    """Handle signals directly and create orders."""
    logger.info(f"Signal handler received event: {event}")
    
    # Extract signal
    if not hasattr(event, 'data') or not isinstance(event.data, SignalEvent):
        logger.error(f"Expected SignalEvent in event data, got {type(event.data) if hasattr(event, 'data') else 'None'}")
        return
    
    signal = event.data
    logger.info(f"Processing signal: {signal.get_signal_name()} for {signal.get_symbol()} @ {signal.get_price()}")
    
    # Skip neutral signals
    if signal.get_signal_value() == SignalEvent.NEUTRAL:
        logger.info("Skipping neutral signal")
        return
    
    # Generate order
    try:
        # Fixed position size for testing
        position_size = 100
        
        # Create order
        order = OrderEvent(
            symbol=signal.get_symbol(),
            direction=signal.get_signal_value(),
            quantity=position_size,
            price=signal.get_price(),
            order_type="MARKET",
            timestamp=signal.timestamp
        )
        
        logger.info(f"Created order: {order}")
        
        # Emit order event
        event_bus.emit(Event(EventType.ORDER, order))
        logger.info("Order event emitted")
        
    except Exception as e:
        logger.error(f"Error creating order: {e}", exc_info=True)

# Run a minimal backtest focusing on signal-to-order conversion
def run_minimal_backtest():
    # Create event bus
    event_bus = EventBus()
    
    # Create execution engine
    execution_engine = MinimalExecutionEngine()
    
    # Create rule
    rule = SMACrossoverRule(
        name="sma_test",
        params={"fast_window": 5, "slow_window": 15}
    )
    
    # Create strategy
    strategy = MinimalStrategy(rule, event_bus)
    
    # Register event handlers
    event_bus.register(EventType.BAR, lambda e: strategy.on_bar(e))
    event_bus.register(EventType.SIGNAL, lambda e: direct_signal_handler(e, event_bus, execution_engine))
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    
    # Create data handler
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source)
    
    # Load data
    data_handler.load_data(
        symbols=["SPY"],
        start_date=datetime.datetime(2024, 3, 26),
        end_date=datetime.datetime(2024, 3, 27),
        timeframe="1m"
    )
    
    # Log event bus handlers
    logger.info(f"Event handlers: {event_bus.handlers}")
    
    # Process bars
    logger.info("Starting backtest")
    for bar in data_handler.iter_train():
        # Convert to BarEvent if needed
        if not isinstance(bar, BarEvent):
            bar_event = BarEvent(bar)
        else:
            bar_event = bar
        
        # Create and emit bar event
        event = Event(EventType.BAR, bar_event)
        event_bus.emit(event)
    
    # Report results
    logger.info(f"Backtest complete. Orders created: {len(execution_engine.orders)}")
    
    return execution_engine.orders

# Run the test
if __name__ == "__main__":
    orders = run_minimal_backtest()
