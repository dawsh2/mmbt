"""
Data Sources Module

This module provides interfaces and implementations for different data sources
such as CSV files, databases, APIs, and real-time market data feeds.
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator


class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: Optional[datetime.datetime] = None, 
                end_date: Optional[datetime.datetime] = None, timeframe: str = '1d') -> pd.DataFrame:
        """
        Retrieve data for a symbol within the specified date range.
        
        Args:
            symbol: The instrument symbol
            start_date: Optional start date for data
            end_date: Optional end date for data
            timeframe: Data timeframe/resolution (e.g., '1m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of symbol strings
        """
        pass


class CSVDataSource(DataSource):
    """Data source for CSV files."""
    
    def __init__(self, data_dir: str, filename_pattern: str = "{symbol}_{timeframe}.csv",
                 date_format: str = "%Y-%m-%d", datetime_format: str = "%Y-%m-%d %H:%M:%S",
                 date_column: str = "timestamp"):
        """
        Initialize CSV data source.
        
        Args:
            data_dir: Directory containing CSV files
            filename_pattern: Pattern for CSV filenames with {symbol} and {timeframe} placeholders
            date_format: Format for date strings
            datetime_format: Format for datetime strings
            date_column: Column name for date/datetime
        """
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern
        self.date_format = date_format
        self.datetime_format = datetime_format
        self.date_column = date_column
        
        # Cached file list for faster symbol lookup
        self._file_list = None
        
    def _get_file_list(self) -> List[str]:
        """Get list of CSV files in the data directory."""
        if self._file_list is None:
            if os.path.exists(self.data_dir):
                self._file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            else:
                self._file_list = []
        return self._file_list
    
    def _get_filename(self, symbol: str, timeframe: str) -> str:
        """Get filename for a symbol and timeframe."""
        return os.path.join(self.data_dir, 
                            self.filename_pattern.format(symbol=symbol, timeframe=timeframe))
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols from CSV files."""
        symbols = set()
        for filename in self._get_file_list():
            # Try to extract symbol from filename based on pattern
            # This is a simplistic approach - might need more sophisticated parsing
            parts = os.path.splitext(filename)[0].split('_')
            if len(parts) > 0:
                symbols.add(parts[0])
        return sorted(list(symbols))
    
    def get_data(self, symbol: str, start_date: Optional[datetime.datetime] = None, 
                end_date: Optional[datetime.datetime] = None, timeframe: str = '1d') -> pd.DataFrame:
        """Get data for a symbol from CSV file."""
        filename = self._get_filename(symbol, timeframe)
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        # Load data from CSV
        df = pd.read_csv(filename)
        
        # Convert date/datetime column
        if self.date_column in df.columns:
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            except:
                # If standard parsing fails, try specific formats
                try:
                    df[self.date_column] = pd.to_datetime(df[self.date_column], 
                                                         format=self.datetime_format)
                except:
                    try:
                        df[self.date_column] = pd.to_datetime(df[self.date_column], 
                                                             format=self.date_format)
                    except Exception as e:
                        raise ValueError(f"Could not parse date column: {str(e)}")
        
        # Filter by date range if provided
        if start_date is not None:
            df = df[df[self.date_column] >= start_date]
        if end_date is not None:
            df = df[df[self.date_column] <= end_date]
            
        # Ensure consistent column naming
        column_mapping = {
            'Date': 'timestamp',
            'Timestamp': 'timestamp',
            'Time': 'timestamp',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adj_Close'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Set index to timestamp but keep timestamp as column
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp').reset_index()
            
        return df
    
    def get_latest_date(self, symbol: str, timeframe: str = '1d') -> Optional[datetime.datetime]:
        """Get the latest available date for a symbol."""
        try:
            df = self.get_data(symbol, timeframe=timeframe)
            if len(df) > 0 and 'timestamp' in df.columns:
                return df['timestamp'].max()
        except Exception:
            pass
        return None


class SQLiteDataSource(DataSource):
    """Data source for SQLite database."""
    
    def __init__(self, db_path: str, table_pattern: str = "{symbol}_{timeframe}"):
        """
        Initialize SQLite data source.
        
        Args:
            db_path: Path to SQLite database file
            table_pattern: Pattern for table names with {symbol} and {timeframe} placeholders
        """
        self.db_path = db_path
        self.table_pattern = table_pattern
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection."""
        return sqlite3.connect(self.db_path)
    
    def _get_table_name(self, symbol: str, timeframe: str) -> str:
        """Get table name for a symbol and timeframe."""
        return self.table_pattern.format(symbol=symbol, timeframe=timeframe)
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols from database tables."""
        symbols = set()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Extract symbols from table names based on pattern
            for table in tables:
                parts = table.split('_')
                if len(parts) > 0:
                    symbols.add(parts[0])
                    
            conn.close()
        except Exception as e:
            print(f"Error getting symbols from database: {str(e)}")
            
        return sorted(list(symbols))
    
    def get_data(self, symbol: str, start_date: Optional[datetime.datetime] = None, 
                end_date: Optional[datetime.datetime] = None, timeframe: str = '1d') -> pd.DataFrame:
        """Get data for a symbol from database."""
        table_name = self._get_table_name(symbol, timeframe)
        
        try:
            conn = self._get_connection()
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            params = []
            
            if start_date is not None or end_date is not None:
                query += " WHERE"
                
                if start_date is not None:
                    query += " timestamp >= ?"
                    params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
                    
                    if end_date is not None:
                        query += " AND"
                        
                if end_date is not None:
                    query += " timestamp <= ?"
                    params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))
                    
            query += " ORDER BY timestamp"
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            conn.close()
            
            # Check if data was found
            if len(df) == 0:
                print(f"No data found for {symbol} in table {table_name}")
                
            return df
            
        except Exception as e:
            raise ValueError(f"Error retrieving data for {symbol}: {str(e)}")


class JSONDataSource(DataSource):
    """Data source for JSON files."""
    
    def __init__(self, data_dir: str, filename_pattern: str = "{symbol}_{timeframe}.json"):
        """
        Initialize JSON data source.
        
        Args:
            data_dir: Directory containing JSON files
            filename_pattern: Pattern for JSON filenames with {symbol} and {timeframe} placeholders
        """
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern
        
        # Cached file list for faster symbol lookup
        self._file_list = None
        
    def _get_file_list(self) -> List[str]:
        """Get list of JSON files in the data directory."""
        if self._file_list is None:
            if os.path.exists(self.data_dir):
                self._file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            else:
                self._file_list = []
        return self._file_list
    
    def _get_filename(self, symbol: str, timeframe: str) -> str:
        """Get filename for a symbol and timeframe."""
        return os.path.join(self.data_dir, 
                            self.filename_pattern.format(symbol=symbol, timeframe=timeframe))
    
    def get_symbols(self) -> List[str]:
        """Get list of available symbols from JSON files."""
        symbols = set()
        for filename in self._get_file_list():
            # Extract symbol from filename based on pattern
            parts = os.path.splitext(filename)[0].split('_')
            if len(parts) > 0:
                symbols.add(parts[0])
        return sorted(list(symbols))
    
    def get_data(self, symbol: str, start_date: Optional[datetime.datetime] = None, 
                end_date: Optional[datetime.datetime] = None, timeframe: str = '1d') -> pd.DataFrame:
        """Get data for a symbol from JSON file."""
        filename = self._get_filename(symbol, timeframe)
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        # Load data from JSON
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            raise ValueError(f"Unexpected JSON format in {filename}")
            
        # Convert date/datetime column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range if provided
        if start_date is not None and 'timestamp' in df.columns:
            df = df[df['timestamp'] >= start_date]
        if end_date is not None and 'timestamp' in df.columns:
            df = df[df['timestamp'] <= end_date]
            
        # Ensure required columns exist
        required_columns = ['timestamp', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df


class DataSourceRegistry:
    """Registry for data sources."""
    
    _sources = {}
    
    @classmethod
    def register(cls, name: str, source: DataSource) -> None:
        """
        Register a data source.
        
        Args:
            name: Name for the data source
            source: DataSource instance
        """
        cls._sources[name] = source
        
    @classmethod
    def get(cls, name: str) -> DataSource:
        """
        Get a registered data source.
        
        Args:
            name: Name of the data source
            
        Returns:
            DataSource instance
            
        Raises:
            ValueError: If data source is not registered
        """
        if name not in cls._sources:
            raise ValueError(f"Data source not registered: {name}")
        return cls._sources[name]
    
    @classmethod
    def list_sources(cls) -> List[str]:
        """
        Get list of registered data sources.
        
        Returns:
            List of data source names
        """
        return list(cls._sources.keys())


# Cache for data to avoid repeated disk reads
class DataCache:
    """Cache for market data to improve performance."""
    
    _cache = {}
    _max_size = 50  # Maximum number of datasets to cache
    
    @classmethod
    def set_max_size(cls, size: int) -> None:
        """Set maximum cache size."""
        cls._max_size = size
        cls._enforce_max_size()
        
    @classmethod
    def get(cls, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available."""
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, data: pd.DataFrame) -> None:
        """Store data in cache."""
        cls._cache[key] = data
        cls._enforce_max_size()
        
    @classmethod
    def clear(cls) -> None:
        """Clear the cache."""
        cls._cache.clear()
        
    @classmethod
    def _enforce_max_size(cls) -> None:
        """Enforce maximum cache size by removing oldest entries."""
        if len(cls._cache) > cls._max_size:
            # Remove oldest entries
            keys_to_remove = sorted(cls._cache.keys())[:len(cls._cache) - cls._max_size]
            for key in keys_to_remove:
                del cls._cache[key]


# Example usage
if __name__ == "__main__":
    # Create CSV data source
    csv_source = CSVDataSource("data/csv")
    
    # Register data source
    DataSourceRegistry.register("csv", csv_source)
    
    # Get list of symbols
    symbols = csv_source.get_symbols()
    print(f"Available symbols: {symbols}")
    
    # Get data for a symbol
    if symbols:
        symbol = symbols[0]
        data = csv_source.get_data(symbol)
        print(f"Data for {symbol}:")
        print(data.head())
