#!/usr/bin/env python3
# run_backtest.py - Basic run script for the algorithmic trading system

import datetime
import os
import pandas as pd
import numpy as np

# Import from the working event system
from src.events.event_bus import EventBus, Event
from src.events.event_types import EventType
from src.events.event_handlers import LoggingHandler
from src.events.event_emitters import MarketDataEmitter

# Import other components as needed
from src.config import ConfigManager  # Import the config manager
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource


def main():
    print("Starting the trading system setup...")
    
    # 1. Initialize the event system
    print("Initializing event system")
    event_bus = EventBus()
    
    # Add a logging handler to see events
    logging_handler = LoggingHandler([EventType.BAR, EventType.MARKET_OPEN, EventType.MARKET_CLOSE])
    for event_type in [EventType.BAR, EventType.MARKET_OPEN, EventType.MARKET_CLOSE]:
        event_bus.register(event_type, logging_handler)
    
    # 2. Create and configure the config manager
    print("Loading configuration")
    config = ConfigManager()
    
    # Set default configuration values
    config.set('backtester.initial_capital', 100000)
    config.set('backtester.market_simulation.slippage_model', 'fixed')
    config.set('backtester.market_simulation.slippage_bps', 5)
    
    # 3. Create synthetic data if it doesn't exist
    print("Checking for test data")
    symbol = "SYNTHETIC"
    timeframe = "1d"
    filename = f"{symbol}_{timeframe}.csv"
    
    if not os.path.exists(filename):
        print(f"Creating synthetic data file: {filename}")
        # Create a synthetic dataset
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        # Create a price series with some randomness
        base_price = 100
        prices = [base_price]
        for i in range(1, len(dates)):
            # Random daily change between -1% and +1%
            daily_change = np.random.normal(0.0005, 0.01)  
            new_price = prices[-1] * (1 + daily_change)
            prices.append(new_price)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
            'Close': prices,
            'Volume': [int(np.random.normal(100000, 20000)) for _ in prices]
        })
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Created synthetic data with {len(df)} bars")
    
    # 4. Set up data sources and handler
    print("Setting up data handler")
    data_source = CSVDataSource(".")  # Look for CSV files in the current directory
    data_handler = DataHandler(data_source)
    
    # 5. Load market data
    print(f"Loading data for symbol: {symbol}, timeframe: {timeframe}")
    try:
        start_date = datetime.datetime(2022, 1, 1)
        end_date = datetime.datetime(2022, 12, 31)
        
        # Load data using the DataHandler - NOTE: it takes a list of symbols
        data_handler.load_data(
            symbols=[symbol],  # DataHandler expects a list
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        print(f"Successfully loaded data")
        
        # Emit events using the iterator pattern
        print("\nEmitting market events through the event system:")
        market_data_emitter = MarketDataEmitter(event_bus)
        
        # Emit market open event
        print("Emitting MARKET_OPEN event")
        market_data_emitter.emit_market_open()
        
        # Process bars from the training data
        count = 0
        for bar in data_handler.iter_train():
            if count < 5:  # Just a few for demo
                print(f"Emitting BAR event for {bar['timestamp']}")
                market_data_emitter.emit_bar(bar)
                count += 1
            else:
                break
                
        # Emit market close event
        print("Emitting MARKET_CLOSE event")
        market_data_emitter.emit_market_close()
            
        print("\nEvents emitted successfully! Check event bus history:")
        print(f"Event history length: {len(event_bus.history)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\nEvent system test complete.")
    
    return {
        "status": "setup_complete", 
        "data_loaded": True,
        "events_emitted": len(event_bus.history)
    }


if __name__ == "__main__":
    try:
        result = main()
        print("\nScript executed successfully!")
        print(f"Status: {result['status']}")
        print(f"Events emitted: {result['events_emitted']}")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
