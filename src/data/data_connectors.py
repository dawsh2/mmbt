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
import websocket
import threading
import os
import sqlite3
import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging

# Set up logging
logger = logging.getLogger(__name__)


class APIConnector(ABC):
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
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.rate_limit_period = rate_limit_period
        
        # Rate limiting state
        self.request_timestamps = []
        
        # Session for HTTP requests
        self.session = requests.Session()
        
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
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
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
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None, 
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
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Latest price
        """
        pass


class AlphaVantageConnector(APIConnector):
    """Connector for Alpha Vantage API."""
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage connector.
        
        Args:
            api_key: Alpha Vantage API key
        """
        super().__init__(
            api_key=api_key,
            base_url="https://www.alphavantage.co/query",
            rate_limit=5,          # Alpha Vantage free tier: 5 requests per minute
            rate_limit_period=60
        )
        
    def get_historical_data(self, symbol: str, start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None, 
                          timeframe: str = '1d') -> pd.DataFrame:
        """
        Get historical market data from Alpha Vantage.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe ('1d' for daily, '1min', '5min', etc.)
            
        Returns:
            DataFrame with market data
        """
        self._wait_for_rate_limit()
        
        # Map timeframe to Alpha Vantage function and interval
        if timeframe == '1d':
            function = "TIME_SERIES_DAILY_ADJUSTED"
            interval_param = None
            time_series_key = "Time Series (Daily)"
        elif timeframe.endswith('min'):
            function = "TIME_SERIES_INTRADAY"
            interval_param = timeframe
            time_series_key = f"Time Series ({timeframe})"
        elif timeframe == '1w':
            function = "TIME_SERIES_WEEKLY_ADJUSTED"
            interval_param = None
            time_series_key = "Weekly Adjusted Time Series"
        elif timeframe == '1mo':
            function = "TIME_SERIES_MONTHLY_ADJUSTED"
            interval_param = None
            time_series_key = "Monthly Adjusted Time Series"
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        # Build params
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        if interval_param:
            params['interval'] = interval_param
            
        # Make API request
        response = self.session.get(self.base_url, params=params)
        data = self._handle_response(response)
        
        # Check for error message
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
        # Parse response into DataFrame
        if time_series_key not in data:
            raise ValueError(f"Unexpected response format: {data.keys()}")
            
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Rename columns to standard format
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume',
            '5. adjusted close': 'Adj_Close',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividend',
            '8. split coefficient': 'Split'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Convert index to datetime and reset
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        # Convert columns to numeric
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col])
                
        # Filter by date range if provided
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
            
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Latest price
        """
        self._wait_for_rate_limit()
        
        # Make API request for global quote
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = self.session.get(self.base_url, params=params)
        data = self._handle_response(response)
        
        # Check for error message
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
            
        # Parse response
        if 'Global Quote' not in data:
            raise ValueError(f"Unexpected response format: {data.keys()}")
            
        quote = data['Global Quote']
        if '05. price' not in quote:
            raise ValueError(f"Price not found in response: {quote.keys()}")
            
        return float(quote['05. price'])


class YahooFinanceConnector(APIConnector):
    """Connector for Yahoo Finance API (unofficial)."""
    
    def __init__(self):
        """Initialize Yahoo Finance connector."""
        super().__init__(
            base_url="https://query1.finance.yahoo.com/v8/finance/chart",
            rate_limit=2000,  # Yahoo allows ~2000 requests/hour without authentication
            rate_limit_period=3600
        )
        
    def get_historical_data(self, symbol: str, start_date: Optional[datetime.datetime] = None,
                          end_date: Optional[datetime.datetime] = None, 
                          timeframe: str = '1d') -> pd.DataFrame:
        """
        Get historical market data from Yahoo Finance.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with market data
        """
        self._wait_for_rate_limit()
        
        # Convert timeframe to Yahoo interval
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '60m',
            '1d': '1d',
            '1w': '1wk',
            '1mo': '1mo'
        }
        
        if timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        interval = interval_map[timeframe]
        
        # Convert dates to Unix timestamps
        period1 = int(start_date.timestamp()) if start_date else 0
        period2 = int(end_date.timestamp()) if end_date else int(time.time())
        
        # Build URL
        url = f"{self.base_url}/{symbol}"
        
        # Build params
        params = {
            'period1': period1,
            'period2': period2,
            'interval': interval,
            'includePrePost': 'false',
            'events': 'div,split'
        }
        
        # Make API request
        response = self.session.get(url, params=params)
        data = self._handle_response(response)
        
        # Check for error
        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            error = data.get('chart', {}).get('error', {})
            if error:
                raise ValueError(f"Yahoo Finance API error: {error}")
            else:
                raise ValueError(f"No data found for {symbol}")
                
        # Parse response
        chart_data = data['chart']['result'][0]
        
        # Get timestamp data
        timestamps = chart_data['timestamp']
        
        # Get OHLCV data
        ohlcv = chart_data['indicators']['quote'][0]
        
        # Get adjusted close if available
        adjclose = None
        if 'adjclose' in chart_data['indicators']:
            adjclose = chart_data['indicators']['adjclose'][0]['adjclose']
            
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'Open': ohlcv['open'],
            'High': ohlcv['high'],
            'Low': ohlcv['low'],
            'Close': ohlcv['close'],
            'Volume': ohlcv['volume']
        })
        
        if adjclose is not None:
            df['Adj_Close'] = adjclose
            
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Latest price
        """
        # Get the last day of data and return the close price
        df = self.get_historical_data(
            symbol=symbol,
            start_date=datetime.datetime.now() - datetime.timedelta(days=1),
            timeframe='1d'
        )
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
            
        return df.iloc[-1]['Close']


class WebSocketClient(threading.Thread):
    """Base class for WebSocket clients."""
    
    def __init__(self, url: str, on_message: Callable[[str], None], 
                on_error: Optional[Callable[[Exception], None]] = None,
                reconnect_interval: int = 5, heartbeat_interval: int = 30):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket URL
            on_message: Callback for message handling
            on_error: Callback for error handling
            reconnect_interval: Seconds to wait before reconnecting
            heartbeat_interval: Seconds between heartbeat messages
        """
        super().__init__()
        self.url = url
        self.on_message = on_message
        self.on_error = on_error or (lambda e: logger.error(f"WebSocket error: {str(e)}"))
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval
        
        self.ws = None
        self.running = False
        self.last_heartbeat = 0
        
    def run(self):
        """Run the WebSocket client thread."""
        self.running = True
        
        while self.running:
            try:
                # Connect to WebSocket
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_message=lambda ws, msg: self.on_message(msg),
                    on_error=lambda ws, error: self.on_error(error),
                    on_close=lambda ws, close_status_code, close_msg: self._on_close(close_status_code, close_msg),
                    on_open=lambda ws: self._on_open()
                )
                
                # Start heartbeat thread
                if self.heartbeat_interval > 0:
                    threading.Thread(target=self._heartbeat).start()
                
                # Start WebSocket connection
                self.ws.run_forever()
                
                # If we get here, the connection was closed
                if not self.running:
                    break
                    
                logger.info(f"WebSocket disconnected. Reconnecting in {self.reconnect_interval} seconds.")
                time.sleep(self.reconnect_interval)
                
            except Exception as e:
                self.on_error(e)
                logger.error(f"WebSocket error: {str(e)}. Reconnecting in {self.reconnect_interval} seconds.")
                time.sleep(self.reconnect_interval)
    
    def _on_open(self):
        """Called when WebSocket connection is established."""
        logger.info("WebSocket connection established")
        self.last_heartbeat = time.time()
    
    def _on_close(self, close_status_code, close_msg):
        """Called when WebSocket connection is closed."""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def _heartbeat(self):
        """Send heartbeat messages to keep connection alive."""
        while self.running and self.ws.sock and self.ws.sock.connected:
            if time.time() - self.last_heartbeat > self.heartbeat_interval:
                try:
                    self.send_heartbeat()
                    self.last_heartbeat = time.time()
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {str(e)}")
            time.sleep(1)
    
    def send_heartbeat(self):
        """Send heartbeat message (override in subclasses)."""
        self.ws.send('{"type":"ping"}')
    
    def send(self, message: str):
        """
        Send a message to the WebSocket.
        
        Args:
            message: Message to send
        """
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(message)
        else:
            raise ConnectionError("WebSocket not connected")
    
    def subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols (override in subclasses).
        
        Args:
            symbols: List of symbols to subscribe to
        """
        pass
    
    def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols (override in subclasses).
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        pass
    
    def stop(self):
        """Stop the WebSocket client."""
        self.running = False
        if self.ws:
            self.ws.close()


class MarketDataWebSocketClient(WebSocketClient):
    """WebSocket client for market data."""
    
    def __init__(self, url: str, api_key: Optional[str] = None, 
                on_quote: Callable[[Dict[str, Any]], None] = None,
                on_trade: Callable[[Dict[str, Any]], None] = None,
                on_bar: Callable[[Dict[str, Any]], None] = None):
        """
        Initialize market data WebSocket client.
        
        Args:
            url: WebSocket URL
            api_key: API key for authentication
            on_quote: Callback for quote messages
            on_trade: Callback for trade messages
            on_bar: Callback for bar messages
        """
        self.api_key = api_key
        self.on_quote = on_quote or (lambda q: None)
        self.on_trade = on_trade or (lambda t: None)
        self.on_bar = on_bar or (lambda b: None)
        
        super().__init__(
            url=url,
            on_message=self._handle_message,
            reconnect_interval=5,
            heartbeat_interval=30
        )
        
        self.subscriptions = set()
    
    def _handle_message(self, message: str):
        """
        Handle incoming WebSocket messages.
        
        Args:
            message: JSON message string
        """
        try:
            data = json.loads(message)
            
            if 'type' in data:
                if data['type'] == 'quote':
                    self.on_quote(data)
                elif data['type'] == 'trade':
                    self.on_trade(data)
                elif data['type'] == 'bar':
                    self.on_bar(data)
                elif data['type'] == 'pong':
                    # Heartbeat response
                    pass
                elif data['type'] == 'error':
                    logger.error(f"WebSocket error: {data.get('message', 'Unknown error')}")
                else:
                    logger.warning(f"Unknown message type: {data['type']}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    def _on_open(self):
        """Send authentication message after connection is established."""
        super()._on_open()
        
        # Authenticate if API key is provided
        if self.api_key:
            auth_msg = json.dumps({
                "type": "auth",
                "key": self.api_key
            })
            self.send(auth_msg)
        
        # Resubscribe to previous subscriptions
        if self.subscriptions:
            self.subscribe(list(self.subscriptions))
    
    def subscribe(self, symbols: List[str]):
        """
        Subscribe to symbols.
        
        Args:
            symbols: List of symbols to subscribe to
        """
        subscription_msg = json.dumps({
            "type": "subscribe",
            "symbols": symbols
        })
        self.send(subscription_msg)
        
        # Update subscriptions set
        self.subscriptions.update(symbols)
    
    def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
        """
        unsubscription_msg = json.dumps({
            "type": "unsubscribe",
            "symbols": symbols
        })
        self.send(unsubscription_msg)
        
        # Update subscriptions set
        self.subscriptions.difference_update(symbols)


class DatabaseConnector:
    """Base class for database connectors."""
    
    def __init__(self, db_path: str, table_prefix: str = ''):
        """
        Initialize database connector.
        
        Args:
            db_path: Path to database file or connection string
            table_prefix: Prefix for table names
        """
        self.db_path = db_path
        self.table_prefix = table_prefix
    
    @abstractmethod
    def connect(self):
        """Connect to the database."""
        pass
    
    @abstractmethod
    def close(self):
        """Close database connection."""
        pass
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, symbol: str, timeframe: str = '1d'):
        """
        Save market data to database.
        
        Args:
            data: DataFrame with market data
            symbol: Instrument symbol
            timeframe: Data timeframe
        """
        pass
    
    @abstractmethod
    def load_data(self, symbol: str, timeframe: str = '1d', 
                start_date: Optional[datetime.datetime] = None,
                end_date: Optional[datetime.datetime] = None) -> pd.DataFrame:
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
        
    def connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        
    def close(self):
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
    
    def _ensure_table_exists(self, table_name: str):
        """
        Ensure market data table exists.
        
        Args:
            table_name: Table name
        """
        if not self.conn:
            self.connect()
            
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
    
    def save_data(self, data: pd.DataFrame, symbol: str, timeframe: str = '1d'):
        """
        Save market data to SQLite database.
        
        Args:
            data: DataFrame with market data
            symbol: Instrument symbol
            timeframe: Data timeframe
        """
        if not self.conn:
            self.connect()
            
        table_name = self._get_table_name(symbol, timeframe)
        self._ensure_table_exists(table_name)
        
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
        
    def load_data(self, symbol: str, timeframe: str = '1d', 
                start_date: Optional[datetime.datetime] = None,
                end_date: Optional[datetime.datetime] = None) -> pd.DataFrame:
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
        if not self.conn:
            self.connect()
            
        table_name = self._get_table_name(symbol, timeframe)
        
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


# Cache for API responses to reduce API calls
class APIResponseCache:
    """Cache for API responses to reduce API calls."""
    
    _cache = {}
    _max_age = 300  # Maximum age in seconds
    _max_size = 100  # Maximum number of cached responses
    
    @classmethod
    def set_max_age(cls, seconds: int):
        """Set maximum cache age in seconds."""
        cls._max_age = seconds
        
    @classmethod
    def set_max_size(cls, size: int):
        """Set maximum cache size."""
        cls._max_size = size
        cls._enforce_max_size()
        
    @classmethod
    def get(cls, key: str) -> Optional[Tuple[Any, float]]:
        """
        Get cached response if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (response, timestamp) or None if not found or expired
        """
        if key in cls._cache:
            response, timestamp = cls._cache[key]
            if time.time() - timestamp <= cls._max_age:
                return response, timestamp
            else:
                # Remove expired entry
                del cls._cache[key]
                
        return None
        
    @classmethod
    def set(cls, key: str, response: Any):
        """
        Cache an API response.
        
        Args:
            key: Cache key
            response: Response data
        """
        cls._cache[key] = (response, time.time())
        cls._enforce_max_size()
        
    @classmethod
    def clear(cls):
        """Clear the cache."""
        cls._cache.clear()
        
    @classmethod
    def _enforce_max_size(cls):
        """Enforce maximum cache size by removing oldest entries."""
        if len(cls._cache) > cls._max_size:
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(cls._cache.items(), key=lambda x: x[1][1])
            
            # Remove oldest entries
            entries_to_remove = len(cls._cache) - cls._max_size
            for i in range(entries_to_remove):
                del cls._cache[sorted_entries[i][0]]


# Example usage
if __name__ == "__main__":
    # Example: connecting to Alpha Vantage
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
    connector = AlphaVantageConnector(api_key=api_key)
    
    try:
        # Get historical data
        df = connector.get_historical_data(
            symbol="AAPL",
            start_date=datetime.datetime.now() - datetime.timedelta(days=30),
            timeframe="1d"
        )
        
        print(f"Retrieved {len(df)} rows of AAPL data:")
        print(df.head())
        
        # Get latest price
        price = connector.get_latest_price("AAPL")
        print(f"Latest AAPL price: ${price:.2f}")
        
        # Save to SQLite database
        db = SQLiteConnector("market_data.db")
        db.connect()
        db.save_data(df, "AAPL", "1d")
        print("Data saved to database.")
        
        # Retrieve from database
        data = db.load_data("AAPL", "1d")
        print(f"Retrieved {len(data)} rows from database.")
        
        db.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
