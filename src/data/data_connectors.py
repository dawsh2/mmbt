"""
Data Connectors Module

This module provides connectors for various external data sources like
APIs, WebSockets, and databases. These connectors handle authentication,
rate limiting, and data format conversion.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime
import os
import logging
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

# Alias DataConnector to DatabaseConnector for backward compatibility
# This will resolve the import error
class DataConnector(ABC):
    """
    Base class for data source connections.
    
    This is a generic connector class that can be subclassed for specific data sources.
    """
    
    def __init__(self, connection_params: Optional[Dict] = None):
        """
        Initialize the data connector.
        
        Args:
            connection_params: Optional connection parameters
        """
        self.connection_params = connection_params or {}
        self.connection = None
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
        
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
        
    @abstractmethod
    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data using the specified query.
        
        Args:
            query: Query string (format depends on the data source)
            
        Returns:
            DataFrame containing the fetched data
        """
        pass


class DatabaseConnector(DataConnector):
    """Base class for database connectors."""
    
    def __init__(self, db_path: str, table_prefix: str = ''):
        """
        Initialize database connector.
        
        Args:
            db_path: Path to database file or connection string
            table_prefix: Prefix for table names
        """
        super().__init__({'db_path': db_path, 'table_prefix': table_prefix})
        self.db_path = db_path
        self.table_prefix = table_prefix
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        self.connection = None
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        # This is a basic implementation that subclasses should override
        raise NotImplementedError("Subclasses must implement fetch_data method")
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, symbol: str, timeframe: str = '1d') -> bool:
        """
        Save market data to database.
        
        Args:
            data: DataFrame with market data
            symbol: Instrument symbol
            timeframe: Data timeframe
            
        Returns:
            True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load_data(self, symbol: str, timeframe: str = '1d', 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load market data from database.
        
        Args:
            symbol: Instrument symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with market data
        """
        pass


class SQLiteConnector(DatabaseConnector):
    """Connector for SQLite database."""
    
    def __init__(self, db_path: str, table_prefix: str = ''):
        """
        Initialize SQLite connector.
        
        Args:
            db_path: Path to SQLite database file
            table_prefix: Prefix for table names
        """
        super().__init__(db_path, table_prefix)
        self.conn = None
        
    def connect(self) -> bool:
        """
        Connect to SQLite database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            import sqlite3
            self.conn = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {str(e)}")
            return False
        
    def disconnect(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def _get_table_name(self, symbol: str, timeframe: str) -> str:
        """
        Get table name for a symbol and timeframe.
        
        Args:
            symbol: Instrument symbol
            timeframe: Data timeframe
            
        Returns:
            Table name
        """
        # Sanitize symbol for SQL
        safe_symbol = symbol.replace('-', '_').replace('.', '_')
        return f"{self.table_prefix}{safe_symbol}_{timeframe}"
    
    def _ensure_table_exists(self, table_name: str) -> bool:
        """
        Ensure market data table exists.
        
        Args:
            table_name: Table name
            
        Returns:
            True if table exists or was created, False on error
        """
        if not self.conn:
            if not self.connect():
                return False
            
        try:
            cursor = self.conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TEXT PRIMARY KEY,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume REAL,
                    Adj_Close REAL
                )
            """)
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {str(e)}")
            return False
    
    def save_data(self, data: pd.DataFrame, symbol: str, timeframe: str = '1d') -> bool:
        """
        Save market data to SQLite database.
        
        Args:
            data: DataFrame with market data
            symbol: Instrument symbol
            timeframe: Data timeframe
            
        Returns:
            True if save was successful, False otherwise
        """
        if not self.conn and not self.connect():
            return False
            
        table_name = self._get_table_name(symbol, timeframe)
        if not self._ensure_table_exists(table_name):
            return False
        
        try:
            # Prepare data for insertion
            df = data.copy()
            
            # Ensure timestamp is a string
            if 'timestamp' in df.columns and not isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
            # Drop any extra columns
            columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
            for col in columns:
                if col not in df.columns:
                    df[col] = None
                    
            df = df[columns]
            
            # Save to database
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving data to SQLite: {str(e)}")
            return False
        
    def load_data(self, symbol: str, timeframe: str = '1d', 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load market data from SQLite database.
        
        Args:
            symbol: Instrument symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with market data
        """
        if not self.conn and not self.connect():
            return pd.DataFrame()
            
        table_name = self._get_table_name(symbol, timeframe)
        
        try:
            # Check if table exists
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                return pd.DataFrame()
                
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
            
            # Execute query
            df = pd.read_sql_query(query, self.conn, params=params)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
        except Exception as e:
            logger.error(f"Error loading data from SQLite: {str(e)}")
            return pd.DataFrame()
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        if not self.conn and not self.connect():
            return pd.DataFrame()
            
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()


class APIConnector(DataConnector):
    """Base class for API connectors."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 base_url: str = "", rate_limit: int = 60, rate_limit_period: int = 60):
        """
        Initialize API connector.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            base_url: Base URL for API requests
            rate_limit: Number of requests allowed in rate_limit_period
            rate_limit_period: Period in seconds for rate limiting
        """
        super().__init__({
            'api_key': api_key,
            'api_secret': api_secret,
            'base_url': base_url,
            'rate_limit': rate_limit,
            'rate_limit_period': rate_limit_period
        })
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.rate_limit_period = rate_limit_period
        
        # Rate limiting state
        self.request_timestamps = []
        
        # Session for HTTP requests
        try:
            import requests
            self.session = requests.Session()
        except ImportError:
            logger.warning("Requests module not found. API requests will not be available.")
            self.session = None
    
    def connect(self) -> bool:
        """
        Establish connection to the API.
        
        Returns:
            True if connection test successful, False otherwise
        """
        if self.session is None:
            return False
            
        try:
            # Test connection with a simple request
            response = self.session.get(self.base_url)
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Error connecting to API: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Close API connection."""
        if self.session:
            self.session.close()
    
    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data using the specified query.
        
        Args:
            query: API endpoint or query string
            
        Returns:
            DataFrame containing the fetched data
        """
        if self.session is None:
            return pd.DataFrame()
            
        try:
            self._wait_for_rate_limit()
            response = self.session.get(f"{self.base_url}/{query}", headers=self._get_headers())
            data = self._handle_response(response)
            
            # Convert to DataFrame if not already
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data from API: {str(e)}")
            return pd.DataFrame()
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to comply with rate limits."""
        current_time = time.time()
        
        # Remove timestamps outside the rate limit period
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts <= self.rate_limit_period]
        
        # If we've hit the rate limit, wait
        if len(self.request_timestamps) >= self.rate_limit:
            oldest_timestamp = min(self.request_timestamps)
            wait_time = self.rate_limit_period - (current_time - oldest_timestamp)
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                
        # Add current request to timestamps
        self.request_timestamps.append(time.time())
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'PythonTradingSystem/1.0'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        return headers
    
    def _handle_response(self, response) -> Dict[str, Any]:
        """
        Handle API response.
        
        Args:
            response: Response object from requests
            
        Returns:
            Parsed JSON data
            
        Raises:
            Exception: If response status code indicates failure
        """
        if response.status_code in (200, 201):
            return response.json()
        else:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            response.raise_for_status()
            
    def get_historical_data(self, symbol: str, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None, 
                           timeframe: str = '1d') -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe/interval
            
        Returns:
            DataFrame with market data
        """
        # This is a placeholder that should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_historical_data method")


# Create shorter name aliases for backward compatibility
MarketDataAPI = APIConnector
DatabaseAPI = DatabaseConnector
