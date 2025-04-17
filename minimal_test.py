import datetime
import os
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our custom components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.data.data_handler import DataHandler, CSVDataSource

# Create event bus
event_bus = EventBus(async_mode=False, validate_events=False)

# Create a simple handler to log bar events
def log_bar_event(event):
    if event.event_type == EventType.BAR:
        bar = event.data
        logger.info(f"Bar: {bar.get_symbol()} @ {bar.get_timestamp()}, " 
                   f"O: {bar.get_open():.2f}, H: {bar.get_high():.2f}, "
                   f"L: {bar.get_low():.2f}, C: {bar.get_price():.2f}, "
                   f"V: {bar.get_volume()}")

# Register the handler
event_bus.register(EventType.BAR, log_bar_event)

# Create data source and handler
data_dir = "./data"  # Adjust this to your data directory
data_source = CSVDataSource(data_dir)
data_handler = DataHandler(data_source)
data_handler.set_event_bus(event_bus)

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

# Process a few bars and emit events
logger.info("Processing the first 5 bars:")
for i, bar_event in enumerate(data_handler.iter_train()):
    if i >= 5:
        break
    # Emit the bar event
    event = Event(EventType.BAR, bar_event)
    event_bus.emit(event)

logger.info("Minimal test completed successfully!")
