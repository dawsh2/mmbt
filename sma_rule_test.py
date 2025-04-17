"""
Test script for the algorithmic trading system with native event emission.

This script demonstrates the proper use of the system's event handling and rule implementation
without relying on wrapper code. It uses the system's native components to:
1. Load market data
2. Process bars through trading rules
3. Generate signals
4. Emit signals to the event bus
5. Handle signals through registered handlers
"""

import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule

# Create event bus
event_bus = EventBus(async_mode=False)

# Signal handler to log and count signals
signal_count = 0

def handle_signal(event):
    """Handler for signal events."""
    global signal_count
    signal = event.data
    
    direction = "BUY" if signal.get_signal_value() == 1 else "SELL" 
    symbol = signal.get_symbol()
    price = signal.get_price()
    timestamp = signal.timestamp if hasattr(signal, 'timestamp') else datetime.datetime.now()
    
    logger.info(f"SIGNAL: {direction} {symbol} @ {price:.2f}, Time: {timestamp}")
    signal_count += 1

# Create data source and handler
data_dir = "./data"
data_source = CSVDataSource(data_dir)
data_handler = DataHandler(data_source, event_bus=event_bus)

# Create our rule
sma_rule = SMACrossoverRule(
    name="sma_crossover",
    params={
        "fast_window": 5,
        "slow_window": 15
    },
    description="SMA Crossover (5,15)",
    event_bus=event_bus  # Pass event bus directly to rule
)

# Register handlers
event_bus.register(EventType.SIGNAL, handle_signal)
event_bus.register(EventType.BAR, sma_rule.on_bar)

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

for i, bar_event in enumerate(data_handler.iter_train(use_bar_events=True)):
    if i >= max_bars:
        break
    
    # Log bar information (periodically to reduce log volume)
    if i % 50 == 0:
        logger.debug(f"Processing bar {i}: {bar_event.get_symbol()} @ {bar_event.get_timestamp()}, Close: {bar_event.get_price():.2f}")
    
    # Create and emit bar event
    bar_event_obj = Event(EventType.BAR, bar_event)
    event_bus.emit(bar_event_obj)
    
    # Log progress every 100 bars
    if i % 100 == 0 and i > 0:
        logger.info(f"Processed {i} bars, signals generated: {signal_count}")

logger.info(f"Processed {min(max_bars, i+1)} bars and generated {signal_count} signals")
logger.info("Test completed successfully!")
