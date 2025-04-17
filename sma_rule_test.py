import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.events.event_emitters import SignalEmitter
from src.events.signal_event import SignalEvent

# Create event bus
event_bus = EventBus(async_mode=False)

# Create a data source and handler
data_dir = "./data"
data_source = CSVDataSource(data_dir)
data_handler = DataHandler(data_source)
data_handler.set_event_bus(event_bus)

# Create our rule instance
sma_rule = SMACrossoverRule(
    name="sma_crossover",
    params={
        "fast_window": 5,
        "slow_window": 15
    },
    description="SMA Crossover (5,15)"
)

# Create signal emitter
signal_emitter = SignalEmitter(event_bus)

# Create a wrapper that connects the rule's output to the event bus
def rule_bar_handler(event):
    # Call the rule's on_bar method to get a signal (if any)
    signal = sma_rule.on_bar(event)
    
    # If a signal was generated, emit it
    if signal is not None:
        logger.debug(f"Rule generated signal, emitting: {signal}")
        
        # Ensure signal is a proper SignalEvent
        if isinstance(signal, dict):
            # Convert dictionary to SignalEvent if needed
            logger.warning("Rule returned dict instead of SignalEvent, converting")
            signal = SignalEvent(
                signal_value=signal.get('signal_type', 0),
                price=signal.get('price', 0),
                symbol=signal.get('symbol', 'default'),
                rule_id=signal.get('rule_id', sma_rule.name),
                metadata=signal.get('metadata', {}),
                timestamp=signal.get('timestamp', datetime.datetime.now())
            )
        
        # Now emit the signal event
        try:
            signal_event = Event(EventType.SIGNAL, signal)
            event_bus.emit(signal_event)
        except Exception as e:
            logger.error(f"Error emitting signal: {e}")

# Signal handler to log and count signals
signal_count = 0

def handle_signal(event):
    global signal_count
    
    if not hasattr(event, 'data'):
        logger.error(f"Event has no data attribute: {event}")
        return
        
    signal = event.data
    
    # Handle different signal object types
    if isinstance(signal, SignalEvent):
        # Use SignalEvent methods
        direction = "BUY" if signal.get_signal_value() == 1 else "SELL"
        price = signal.get_price()
        symbol = signal.get_symbol()
        timestamp = signal.timestamp
    elif isinstance(signal, dict):
        # Handle dictionary signal (legacy format)
        direction = "BUY" if signal.get('signal_type', 0) == 1 else "SELL"
        price = signal.get('price', 0)
        symbol = signal.get('symbol', 'unknown')
        timestamp = signal.get('timestamp', datetime.datetime.now())
    else:
        logger.error(f"Unknown signal type: {type(signal)}")
        return
    
    logger.info(f"SIGNAL: {direction} {symbol} @ {price:.2f}, Time: {timestamp}")
    signal_count += 1

# Register our handlers
event_bus.register(EventType.BAR, rule_bar_handler)  # Use our wrapper that emits signals
event_bus.register(EventType.SIGNAL, handle_signal)  # Log signals

# Configure the data load parameters
symbols = ["SPY"]
start_date = datetime.datetime(2024, 3, 26)
end_date = datetime.datetime(2025, 4, 2)
timeframe = "1m"

# Load data
logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
data_handler.load_data(symbols=symbols, 
                      start_date=start_date,
                      end_date=end_date,
                      timeframe=timeframe)

# Process bars
logger.info("Processing bars to find trading signals...")
max_bars = 1000  # Process a limited number of bars for this test

for i, bar_event in enumerate(data_handler.iter_train()):
    if i >= max_bars:
        break
    
    # Create and emit bar event
    bar_event_obj = Event(EventType.BAR, bar_event)
    event_bus.emit(bar_event_obj)

logger.info(f"Processed {min(max_bars, i+1)} bars and generated {signal_count} signals")
logger.info("Test completed successfully!")
