"""
Data handling module for the trading system.

This module provides components for data loading, preprocessing, 
and management from various sources.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import os
import glob


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


    # def get_data(self, symbol, start_date=None, end_date=None, timeframe=None):
    #     """Get data for the specified symbol and date range."""
    #     # Filter by date if requested
    #     if start_date or end_date:
    #         df = self.data.copy()

    #         if start_date:
    #             # Make start_date timezone-aware if the timestamps are
    #             if 'timestamp' in df.columns and df['timestamp'].dt.tz is not None:
    #                 import pytz
    #                 if start_date.tzinfo is None:
    #                     # Assume UTC if timestamp has timezone but start_date doesn't
    #                     start_date = pytz.utc.localize(start_date)
    #             df = df[df['timestamp'] >= start_date]

    #         if end_date:
    #             # Make end_date timezone-aware if the timestamps are
    #             if 'timestamp' in df.columns and df['timestamp'].dt.tz is not None:
    #                 import pytz
    #                 if end_date.tzinfo is None:
    #                     # Assume UTC if timestamp has timezone but end_date doesn't
    #                     end_date = pytz.utc.localize(end_date)
    #             df = df[df['timestamp'] <= end_date]

    #         return df

    #     return self.data
        
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
                    raise FileNotFoundError(f"Data file not found: {filepath}")
                
                # Load data from CSV
                df = pd.read_csv(filepath, parse_dates=['timestamp'])
                
                # Store in cache
                self.data_cache[cache_key] = df
            
            # Filter by date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            filtered_df = df.loc[mask].copy()
            
            # Add symbol column if not already present
            if 'symbol' not in filtered_df.columns:
                filtered_df['symbol'] = symbol
                
            combined_data.append(filtered_df)
        
        if not combined_data:
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
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            
            if df.empty:
                return False
                
            file_start = df['timestamp'].min()
            file_end = df['timestamp'].max()
            
            # Check if requested date range is available
            if file_start <= start_date and file_end >= end_date:
                return True
                
            return False
            
        except Exception:
            return False


class DataHandler:
    """
    Main class for handling data operations in the trading system.
    
    This class orchestrates data loading, processing, and management.
    It serves as the primary interface for strategies to access data.
    """
    
    def __init__(self, data_source: DataSource, train_fraction: float = 0.8):
        """
        Initialize the data handler.
        
        Args:
            data_source: DataSource instance for loading data
            train_fraction: Fraction of data to use for training (vs testing)
        """
        self.data_source = data_source
        self.train_fraction = train_fraction
        self.full_data = None
        self.train_data = None
        self.test_data = None
        self.current_train_index = 0
        self.current_test_index = 0


    def load_data(self, symbols: List[str], start_date: datetime, end_date: datetime, timeframe: str):
        """
        Load data for multiple symbols.

        Args:
            symbols: List of symbols to load
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (e.g., '1d', '1h', '5m')
        """
        all_data = []

        for symbol in symbols:
            # Get data for a single symbol
            symbol_data = self.data_source.get_data(symbol, start_date, end_date, timeframe)
            all_data.append(symbol_data)

        # Combine data from all symbols
        if all_data:
            self.full_data = pd.concat(all_data, ignore_index=True)
        else:
            self.full_data = pd.DataFrame()

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
        
        print(f"Loaded {len(self.full_data)} bars from {start_date} to {end_date}")
        print(f"Training data: {len(self.train_data)} bars")
        print(f"Testing data: {len(self.test_data)} bars")
        
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
        
        # Add end-of-day flag if appropriate
        if self.current_train_index < len(self.train_data):
            next_bar = self.train_data.iloc[self.current_train_index]
            bar['is_eod'] = next_bar['timestamp'].date() != pd.Timestamp(bar['timestamp']).date()
        else:
            bar['is_eod'] = True
            
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
        
        # Add end-of-day flag if appropriate
        if self.current_test_index < len(self.test_data):
            next_bar = self.test_data.iloc[self.current_test_index]
            bar['is_eod'] = next_bar['timestamp'].date() != pd.Timestamp(bar['timestamp']).date()
        else:
            bar['is_eod'] = True
            
        return bar
    
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
        
    def iter_train(self):
        """
        Iterator for training data.
        
        Yields:
            Dict containing bar data
        """
        self.reset_train()
        while True:
            bar = self.get_next_train_bar()
            if bar is None:
                break
            yield bar
            
    def iter_test(self):
        """
        Iterator for testing data.
        
        Yields:
            Dict containing bar data
        """
        self.reset_test()
        while True:
            bar = self.get_next_test_bar()
            if bar is None:
                break
            yield bar
            
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
        
    def get_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get data for a specific date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            
        Returns:
            DataFrame containing data for the date range
        """
        if self.full_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        mask = (self.full_data['timestamp'] >= start_date) & (self.full_data['timestamp'] <= end_date)
        return self.full_data.loc[mask].copy()

    def set_event_bus(self, event_bus):
        """Set the event bus for emitting bar events."""
        self.event_bus = event_bus

    def get_next_bar_event(self, is_training=True):
        """
        Get the next bar from the specified dataset as a BarEvent.

        Args:
            is_training: If True, get from training data; otherwise testing data

        Returns:
            BarEvent object or None if no more data
        """
        # Get the next bar dictionary
        if is_training:
            bar_dict = self.get_next_train_bar()
        else:
            bar_dict = self.get_next_test_bar()

        # Return None if no more data
        if bar_dict is None:
            return None

        # Convert to BarEvent
        return BarEvent(bar_dict)

    def emit_bar_event(self, bar_data):
        """
        Convert a bar to a BarEvent and emit it.

        Args:
            bar_data: Dictionary with OHLCV data or BarEvent object
        """
        if self.event_bus is None:
            logger.warning("No event bus set. Cannot emit bar event.")
            return

        # Convert to BarEvent if necessary
        if not isinstance(bar_data, BarEvent):
            bar_event = BarEvent(bar_data)
        else:
            bar_event = bar_data

        # Create and emit the event
        event = Event(EventType.BAR, bar_event)
        self.event_bus.emit(event)

    def process_data(self, emit_events=False):
        """
        Process all data, optionally emitting events.

        Args:
            emit_events: If True, emit bar events for each bar
        """
        # Skip if no event bus and events should be emitted
        if emit_events and self.event_bus is None:
            logger.warning("No event bus set. Cannot emit events.")
            return

        # Reset indices
        self.reset()

        # Process training data
        while True:
            bar = self.get_next_train_bar()
            if bar is None:
                break

            # Emit event if requested
            if emit_events:
                self.emit_bar_event(bar)

        # Reset for next use
        self.reset()

    def set_event_bus(self, event_bus):
        """
        Set the event bus for emitting events.

        Args:
            event_bus: Event bus instance
        """
        self.event_bus = event_bus

    def get_bar_event(self, bar_data):
        """
        Convert raw bar data to a BarEvent.

        Args:
            bar_data: Dictionary containing OHLCV data

        Returns:
            BarEvent object
        """
        return BarEvent(bar_data)

    def emit_bar_event(self, bar_data):
        """
        Create and emit a bar event.

        Args:
            bar_data: Dictionary with OHLCV data or BarEvent
        """
        if not hasattr(self, 'event_bus') or self.event_bus is None:
            logger.warning("No event bus set. Cannot emit bar event.")
            return

        # Convert to BarEvent if necessary
        if not isinstance(bar_data, BarEvent):
            bar_event = self.get_bar_event(bar_data)
        else:
            bar_event = bar_data

        # Create and emit the event
        event = Event(EventType.BAR, bar_event)
        self.event_bus.emit(event)

        return bar_event

    def process_bar(self, bar_data):
        """
        Process a bar of market data, potentially emitting an event.

        Args:
            bar_data: Dictionary with OHLCV data

        Returns:
            BarEvent if emitted, None otherwise
        """
        # Create BarEvent
        bar_event = self.get_bar_event(bar_data)

        # Update current indices
        if 'symbol' in bar_data:
            symbol = bar_data['symbol']
            # Store latest bar (implementation dependent)

        # Emit event if we have an event bus
        if hasattr(self, 'event_bus') and self.event_bus is not None:
            return self.emit_bar_event(bar_event)

        return bar_event

    def iter_bars_as_events(self, use_training=True):
        """
        Iterate through bars, returning them as BarEvent objects.

        Args:
            use_training: If True, use training data; otherwise testing data

        Yields:
            BarEvent objects
        """
        # Reset pointers
        if use_training:
            self.reset_train()
            iterator = self.iter_train
        else:
            self.reset_test()
            iterator = self.iter_test

        # Iterate through data
        for bar in iterator():
            yield self.get_bar_event(bar)

    def emit_all_bars(self, use_training=True):
        """
        Emit events for all bars in the dataset.

        Args:
            use_training: If True, use training data; otherwise testing data

        Returns:
            Number of bars emitted
        """
        if not hasattr(self, 'event_bus') or self.event_bus is None:
            logger.warning("No event bus set. Cannot emit events.")
            return 0

        count = 0
        for bar_event in self.iter_bars_as_events(use_training):
            self.emit_bar_event(bar_event)
            count += 1

        return count


class DataTransformer:
    """
    Class for transforming and preprocessing data.
    
    This includes normalization, feature engineering, and other
    data preparation steps.
    """
    
    def __init__(self):
        """Initialize the data transformer."""
        self.transformations = []
        
    def add_transformation(self, transformation_func):
        """
        Add a transformation function to the pipeline.
        
        Args:
            transformation_func: Function that transforms a DataFrame
        """
        self.transformations.append(transformation_func)
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        
        for transformation in self.transformations:
            result = transformation(result)
            
        return result


class DataConnector:
    """
    Class for connecting to external data sources like APIs or databases.
    
    This provides an interface for fetching data from various external sources.
    """
    
    def __init__(self, api_key: Optional[str] = None, connection_params: Optional[Dict] = None):
        """
        Initialize the data connector.
        
        Args:
            api_key: Optional API key for authentication
            connection_params: Optional connection parameters
        """
        self.api_key = api_key
        self.connection_params = connection_params or {}
        self.connection = None
        
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        # This would be implemented for specific data sources
        # For example, connecting to a database or API
        self.connection = True  # Placeholder
        return True
        
    def disconnect(self) -> None:
        """Close connection to the data source."""
        self.connection = None
        
    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data using the specified query.
        
        Args:
            query: Query string (format depends on the data source)
            
        Returns:
            DataFrame containing the fetched data
        """
        # This would be implemented for specific data sources
        # For example, executing a SQL query or API request
        return pd.DataFrame()  # Placeholder
