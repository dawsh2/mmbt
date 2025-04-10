# data_handler.py
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os
from typing import List, Dict, Any

from events import EventQueue, MarketEvent

class EventDrivenDataHandler(ABC):
    def __init__(self, event_queue, symbols):
        self.event_queue = event_queue
        self.symbols = symbols
        self.continue_backtest = True
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.current_index = 0
    
    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """Return the latest N bars for a symbol"""
        raise NotImplementedError("Implement in subclass")
    
    @abstractmethod
    def update_bars(self):
        """Push latest bars to the event queue as MarketEvents"""
        raise NotImplementedError("Implement in subclass")

class CSVDataHandler(EventDrivenDataHandler):
    def __init__(self, event_queue, csv_dir, symbol_list):
        super().__init__(event_queue, symbol_list)
        self.csv_dir = csv_dir
        self._load_csv_files()
        self._create_timeline()
    
    def _load_csv_files(self):
        """Load all CSV files for the symbols"""
        for symbol in self.symbols:
            # Load CSV
            file_path = os.path.join(self.csv_dir, f"{symbol}.csv")
            self.symbol_data[symbol] = pd.read_csv(
                file_path, 
                header=0, 
                index_col='Date',  # Adjust if your timestamp column has a different name
                parse_dates=True
            )
            
            # Sort by date and add calculated fields
            self.symbol_data[symbol].sort_index(inplace=True)
            
            # Add log returns 
            self.symbol_data[symbol]['LogReturn'] = np.log(
                self.symbol_data[symbol]['Close'] / 
                self.symbol_data[symbol]['Close'].shift(1)
            ).fillna(0)
            
            # Initialize container for the latest data
            self.latest_symbol_data[symbol] = []
    
    def _create_timeline(self):
        """Create a unified timeline for all symbols"""
        # Collect all dates across all symbols
        all_dates = set()
        for symbol in self.symbols:
            all_dates.update(self.symbol_data[symbol].index)
        
        # Sort dates to create timeline
        self.timeline = sorted(all_dates)
        self.current_index = 0
    
    def get_latest_bars(self, symbol, N=1):
        """Return the latest N bars for a symbol"""
        try:
            bars = self.latest_symbol_data[symbol]
        except KeyError:
            print(f"Symbol {symbol} not found in data")
            return None
        
        # Return N bars or all if less than N
        result = bars[-N:] if len(bars) >= N else bars
        return pd.DataFrame(result)
    
    def update_bars(self):
        """Push the latest bar data to the event queue"""
        # Check if we've reached the end of the data
        if self.current_index >= len(self.timeline):
            self.continue_backtest = False
            return
        
        # Get current datetime
        current_date = self.timeline[self.current_index]
        
        # Update the latest bars for each symbol
        for symbol in self.symbols:
            # Check if this symbol has data for the current date
            symbol_data = self.symbol_data[symbol]
            if current_date in symbol_data.index:
                # Get the bar data for this date
                bar = symbol_data.loc[current_date].to_dict()
                
                # Add datetime info
                bar['datetime'] = current_date
                
                # Add to list of latest data
                self.latest_symbol_data[symbol].append(bar)
                
                # Create and add MarketEvent to the queue
                market_event = MarketEvent(
                    symbol=symbol,
                    bar_data=bar
                )
                self.event_queue.put(market_event)
        
        # Move to next bar
        self.current_index += 1
