"""
Data handling module for the trading system.

This module provides components for data loading, preprocessing, 
and management from various sources.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import os
import logging
from src.events.event_types import BarEvent, Event, EventType


# Set up logging
logger = logging.getLogger(__name__)



class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    DataSource classes are responsible for fetching data from a specific source
    (e.g., CSV files, APIs, databases) and converting it into a standardized format.
    """
    
    @abstractmethod
    def get_data(self, symbols: List[str], start_date: datetime, 
                 end_date: datetime, timeframe: str) -> pd.DataFrame:
        """
        Retrieve data for the specified symbols and date range.
        
        Args:
            symbols: List of symbol identifiers to fetch
            start_date: Start date for the data
            end_date: End date for the data
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
            
        Returns:
            DataFrame containing the requested data
        """
        pass
    
    @abstractmethod
    def is_available(self, symbol: str, start_date: datetime, 
                    end_date: datetime, timeframe: str) -> bool:
        """
        Check if data is available for the specified parameters.
        
        Args:
            symbol: Symbol identifier to check
            start_date: Start date to check
            end_date: End date to check
            timeframe: Data timeframe
            
        Returns:
            True if data is available, False otherwise
        """
        pass


class CSVDataSource(DataSource):
    """
    Data source that loads data from CSV files.
    
    This class implements loading from a directory of CSV files
    with standardized naming conventions.
    """
    
    def __init__(self, base_dir: str, filename_template: str = "{symbol}_{timeframe}.csv", 
                 date_format: str = "%Y-%m-%d"):
        """
        Initialize the CSV data source.
        
        Args:
            base_dir: Base directory containing CSV files
            filename_template: Template for CSV filenames
            date_format: Date format string for parsing dates
        """
        self.base_dir = base_dir
        self.filename_template = filename_template
        self.date_format = date_format
        
        # Cache of loaded data
        self.data_cache = {}

    def get_data(self, symbols: List[str], start_date: datetime, 
             end_date: datetime, timeframe: str) -> pd.DataFrame:
        """
        Load data from CSV files for the specified symbols and date range.

        Args:
            symbols: List of symbol identifiers to fetch
            start_date: Start date for the data
            end_date: End date for the data
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')

        Returns:
            DataFrame containing the requested data
        """
        combined_data = []

        # Handle both single string and list inputs for symbols
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            # Check if data is already in cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
            else:
                # Construct filename from template
                filename = self.filename_template.format(symbol=symbol, timeframe=timeframe)
                filepath = os.path.join(self.base_dir, filename)

                if not os.path.exists(filepath):
                    logger.warning(f"Data file not found: {filepath}")
                    continue

                try:
                    # Load data from CSV with proper timestamp handling
                    df = pd.read_csv(filepath)

                    # Handle the timestamp with timezone information
                    if 'timestamp' in df.columns:
                        # Use utc=True to standardize timezone handling
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

                        # Convert to naive datetime by removing timezone information
                        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

                    # Store in cache
                    self.data_cache[cache_key] = df
                except Exception as e:
                    logger.error(f"Error loading data from {filepath}: {str(e)}")
                    continue

            # Ensure start_date and end_date are naive datetime objects if timestamps are naive
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            # Filter by date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            filtered_df = df.loc[mask].copy()

            # Add symbol column if not already present
            if 'symbol' not in filtered_df.columns:
                filtered_df['symbol'] = symbol

            combined_data.append(filtered_df)

        if not combined_data:
            logger.warning(f"No data found for symbols {symbols}")
            return pd.DataFrame()

        # Combine data from all symbols
        result = pd.concat(combined_data, ignore_index=True)

        # Ensure sorted by timestamp
        result = result.sort_values('timestamp')

        return result

    def is_available(self, symbol: str, start_date: datetime, 
                end_date: datetime, timeframe: str) -> bool:
        """
        Check if data is available in CSV files for the specified parameters.

        Args:
            symbol: Symbol identifier to check
            start_date: Start date to check
            end_date: End date to check
            timeframe: Data timeframe

        Returns:
            True if data is available, False otherwise
        """
        # Construct filename from template
        filename = self.filename_template.format(symbol=symbol, timeframe=timeframe)
        filepath = os.path.join(self.base_dir, filename)

        if not os.path.exists(filepath):
            return False

        # Check if file contains data for the date range
        try:
            df = pd.read_csv(filepath)

            # Handle the timestamp with timezone information
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            if df.empty:
                return False

            # Ensure start_date and end_date are naive datetime objects
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            file_start = df['timestamp'].min()
            file_end = df['timestamp'].max()

            # Check if requested date range is available
            if file_start <= start_date and file_end >= end_date:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking data availability: {str(e)}")
            return False
 


class DataHandler:
    """
    Main class for handling data operations in the trading system.
    
    This class orchestrates data loading, processing, and management.
    It serves as the primary interface for strategies to access data.
    """
    
    def __init__(self, data_source, train_fraction: float = 0.8, event_bus = None):
        """
        Initialize the data handler.
        
        Args:
            data_source: DataSource instance for loading data
            train_fraction: Fraction of data to use for training (vs testing)
            event_bus: Optional event bus for emitting events
        """
        self.data_source = data_source
        self.train_fraction = train_fraction
        self.event_bus = event_bus
        self.full_data = None
        self.train_data = None
        self.test_data = None
        self.current_train_index = 0
        self.current_test_index = 0
    
    def set_event_bus(self, event_bus) -> None:
        """
        Set the event bus for emitting events.
        
        Args:
            event_bus: Event bus instance
        """
        self.event_bus = event_bus
    
    def load_data(self, symbols: List[str], start_date: datetime, 
                 end_date: datetime, timeframe: str) -> None:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
        """
        try:
            # Get data for all symbols at once
            self.full_data = self.data_source.get_data(symbols, start_date, end_date, timeframe)
            
            if self.full_data.empty:
                logger.warning("No data loaded.")
                return
                
            # Create train/test split
            split_idx = int(len(self.full_data) * self.train_fraction)
            
            # Determine if we need to ensure the split is on a day boundary
            timestamps = self.full_data['timestamp'].values
            split_timestamp = timestamps[split_idx]
            
            # Adjust split to end of day if using daily or larger timeframe
            if timeframe.endswith('d') or timeframe.endswith('w') or timeframe.endswith('m'):
                next_day = (pd.Timestamp(split_timestamp) + pd.Timedelta(days=1)).floor('D')
                # Find the closest index to this timestamp
                closest_idx = self.full_data['timestamp'].searchsorted(next_day)
                split_idx = closest_idx if closest_idx < len(self.full_data) else split_idx
                
            # Create the splits
            self.train_data = self.full_data.iloc[:split_idx].copy()
            self.test_data = self.full_data.iloc[split_idx:].copy()
            
            # Reset indices
            self.reset()
            
            logger.info(f"Loaded {len(self.full_data)} bars from {start_date} to {end_date}")
            logger.info(f"Training data: {len(self.train_data)} bars")
            logger.info(f"Testing data: {len(self.test_data)} bars")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_next_train_bar(self) -> Optional[Dict[str, Any]]:
        """
        Get the next bar from the training data.
        
        Returns:
            Dict containing bar data or None if no more data
        """
        if self.train_data is None or self.current_train_index >= len(self.train_data):
            return None
            
        bar = self.train_data.iloc[self.current_train_index].to_dict()
        self.current_train_index += 1
            
        return bar
    
    def get_next_test_bar(self) -> Optional[Dict[str, Any]]:
        """
        Get the next bar from the testing data.
        
        Returns:
            Dict containing bar data or None if no more data
        """
        if self.test_data is None or self.current_test_index >= len(self.test_data):
            return None
            
        bar = self.test_data.iloc[self.current_test_index].to_dict()
        self.current_test_index += 1
            
        return bar
    
    def get_next_train_bar_event(self) -> Optional[BarEvent]:
        """
        Get the next bar from the training data as a BarEvent.
        
        Returns:
            BarEvent object or None if no more data
        """
        bar_dict = self.get_next_train_bar()
        if bar_dict is None:
            return None
        
        # Convert to BarEvent
        return BarEvent(bar_dict)
    
    def get_next_test_bar_event(self) -> Optional[BarEvent]:
        """
        Get the next bar from the testing data as a BarEvent.
        
        Returns:
            BarEvent object or None if no more data
        """
        bar_dict = self.get_next_test_bar()
        if bar_dict is None:
            return None
        
        # Convert to BarEvent
        return BarEvent(bar_dict)
    
    def reset_train(self) -> None:
        """Reset the training data iterator."""
        self.current_train_index = 0
    
    def reset_test(self) -> None:
        """Reset the testing data iterator."""
        self.current_test_index = 0
    
    def reset(self) -> None:
        """Reset both training and testing data iterators."""
        self.reset_train()
        self.reset_test()
    
    def iter_train(self, use_bar_events: bool = True):
        """
        Iterator for training data.
        
        Args:
            use_bar_events: If True, yield BarEvent objects instead of dictionaries
            
        Yields:
            Dict containing bar data or BarEvent object
        """
        self.reset_train()
        while True:
            if use_bar_events:
                bar = self.get_next_train_bar_event()
            else:
                bar = self.get_next_train_bar()
            
            if bar is None:
                break
            
            yield bar
    
    def iter_test(self, use_bar_events: bool = True):
        """
        Iterator for testing data.
        
        Args:
            use_bar_events: If True, yield BarEvent objects instead of dictionaries
            
        Yields:
            Dict containing bar data or BarEvent object
        """
        self.reset_test()
        while True:
            if use_bar_events:
                bar = self.get_next_test_bar_event()
            else:
                bar = self.get_next_test_bar()
            
            if bar is None:
                break
            
            yield bar
    
    def create_bar_event(self, bar_data: Dict[str, Any]) -> BarEvent:
        """
        Create a standardized BarEvent from bar data.
        
        Args:
            bar_data: Dictionary containing OHLCV data
            
        Returns:
            BarEvent object
        """
        return BarEvent(bar_data)
    
    def emit_bar_event(self, bar_data: Union[Dict[str, Any], BarEvent]) -> None:
        """
        Create and emit a bar event.
        
        Args:
            bar_data: Dictionary with OHLCV data or BarEvent
        """
        if self.event_bus is None:
            logger.warning("No event bus set. Cannot emit bar event.")
            return
        
        # Convert to BarEvent if necessary
        if not isinstance(bar_data, BarEvent):
            bar_event = self.create_bar_event(bar_data)
        else:
            bar_event = bar_data
        
        # Create and emit the event
        event = Event(EventType.BAR, bar_event)
        self.event_bus.emit(event)
    
    def process_bar(self, bar_data: Dict[str, Any]) -> None:
        """
        Process a bar of market data, emitting an event if possible.
        
        Args:
            bar_data: Dictionary with OHLCV data
        """
        # Create BarEvent
        bar_event = self.create_bar_event(bar_data)
        
        # Emit event if we have an event bus
        if self.event_bus is not None:
            self.emit_bar_event(bar_event)
    
    def emit_all_bars(self, use_train: bool = True) -> int:
        """
        Emit bar events for all bars in the specified dataset.
        
        Args:
            use_train: If True, use training data; otherwise use testing data
            
        Returns:
            Number of events emitted
        """
        if self.event_bus is None:
            logger.warning("No event bus provided. Cannot emit bar events.")
            return 0
        
        count = 0
        iterator = self.iter_train(use_bar_events=True) if use_train else self.iter_test(use_bar_events=True)
        
        # Emit events
        for bar_event in iterator:
            self.emit_bar_event(bar_event)
            count += 1
        
        return count
    
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Get all data for a specific symbol.
        
        Args:
            symbol: Symbol to retrieve data for
            
        Returns:
            DataFrame containing data for the symbol
        """
        if self.full_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if 'symbol' not in self.full_data.columns:
            # If only one symbol was loaded
            return self.full_data.copy()
            
        return self.full_data[self.full_data['symbol'] == symbol].copy()
