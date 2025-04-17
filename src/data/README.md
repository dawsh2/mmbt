# Data Module

Data Module Initialization

This module provides components for market data acquisition, preprocessing, and management.

## Contents

- [data_connectors](#data_connectors)
- [data_handler](#data_handler)
- [data_sources](#data_sources)
- [data_transformer](#data_transformer)

## data_connectors

Data Connectors Module

This module provides connectors for various external data sources like
APIs, WebSockets, and databases. These connectors handle authentication,
rate limiting, and data format conversion.

### Classes

#### `DataConnector`

Base class for data source connections.

This is a generic connector class that can be subclassed for specific data sources.

##### Methods

###### `__init__(connection_params=None)`

Initialize the data connector.

Args:
    connection_params: Optional connection parameters

###### `connect()`

*Returns:* `bool`

Establish connection to the data source.

Returns:
    True if connection successful, False otherwise

###### `disconnect()`

*Returns:* `None`

Close connection to the data source.

###### `fetch_data(query)`

*Returns:* `pd.DataFrame`

Fetch data using the specified query.

Args:
    query: Query string (format depends on the data source)
    
Returns:
    DataFrame containing the fetched data

#### `DatabaseConnector`

Base class for database connectors.

##### Methods

###### `__init__(db_path, table_prefix='')`

Initialize database connector.

Args:
    db_path: Path to database file or connection string
    table_prefix: Prefix for table names

###### `connect()`

*Returns:* `bool`

Connect to the database.

Returns:
    True if connection successful, False otherwise

###### `disconnect()`

*Returns:* `None`

Close database connection.

###### `fetch_data(query)`

*Returns:* `pd.DataFrame`

Execute a query and return results as DataFrame.

Args:
    query: SQL query string
    
Returns:
    DataFrame with query results

###### `save_data(data, symbol, timeframe='1d')`

*Returns:* `bool`

Save market data to database.

Args:
    data: DataFrame with market data
    symbol: Instrument symbol
    timeframe: Data timeframe
    
Returns:
    True if save was successful, False otherwise

###### `load_data(symbol, timeframe='1d', start_date=None, end_date=None)`

*Returns:* `pd.DataFrame`

Load market data from database.

Args:
    symbol: Instrument symbol
    timeframe: Data timeframe
    start_date: Start date for data
    end_date: End date for data
    
Returns:
    DataFrame with market data

#### `SQLiteConnector`

Connector for SQLite database.

##### Methods

###### `__init__(db_path, table_prefix='')`

Initialize SQLite connector.

Args:
    db_path: Path to SQLite database file
    table_prefix: Prefix for table names

###### `connect()`

*Returns:* `bool`

Connect to SQLite database.

Returns:
    True if connection successful, False otherwise

###### `disconnect()`

*Returns:* `None`

Close database connection.

###### `_get_table_name(symbol, timeframe)`

*Returns:* `str`

Get table name for a symbol and timeframe.

Args:
    symbol: Instrument symbol
    timeframe: Data timeframe
    
Returns:
    Table name

###### `_ensure_table_exists(table_name)`

*Returns:* `bool`

Ensure market data table exists.

Args:
    table_name: Table name
    
Returns:
    True if table exists or was created, False on error

###### `save_data(data, symbol, timeframe='1d')`

*Returns:* `bool`

Save market data to SQLite database.

Args:
    data: DataFrame with market data
    symbol: Instrument symbol
    timeframe: Data timeframe
    
Returns:
    True if save was successful, False otherwise

###### `load_data(symbol, timeframe='1d', start_date=None, end_date=None)`

*Returns:* `pd.DataFrame`

Load market data from SQLite database.

Args:
    symbol: Instrument symbol
    timeframe: Data timeframe
    start_date: Start date for data
    end_date: End date for data
    
Returns:
    DataFrame with market data

###### `fetch_data(query)`

*Returns:* `pd.DataFrame`

Execute a query and return results as DataFrame.

Args:
    query: SQL query string
    
Returns:
    DataFrame with query results

#### `APIConnector`

Base class for API connectors.

##### Methods

###### `__init__(api_key=None, api_secret=None, base_url='', rate_limit=60, rate_limit_period=60)`

Initialize API connector.

Args:
    api_key: API key for authentication
    api_secret: API secret for authentication
    base_url: Base URL for API requests
    rate_limit: Number of requests allowed in rate_limit_period
    rate_limit_period: Period in seconds for rate limiting

###### `connect()`

*Returns:* `bool`

Establish connection to the API.

Returns:
    True if connection test successful, False otherwise

###### `disconnect()`

*Returns:* `None`

Close API connection.

###### `fetch_data(query)`

*Returns:* `pd.DataFrame`

Fetch data using the specified query.

Args:
    query: API endpoint or query string
    
Returns:
    DataFrame containing the fetched data

###### `_wait_for_rate_limit()`

*Returns:* `None`

Wait if necessary to comply with rate limits.

###### `_get_headers()`

*Returns:* `Dict[str, str]`

Get headers for API requests.

###### `_handle_response(response)`

*Returns:* `Dict[str, Any]`

Handle API response.

Args:
    response: Response object from requests
    
Returns:
    Parsed JSON data
    
Raises:
    Exception: If response status code indicates failure

###### `get_historical_data(symbol, start_date=None, end_date=None, timeframe='1d')`

*Returns:* `pd.DataFrame`

Get historical market data.

Args:
    symbol: Instrument symbol
    start_date: Start date for data
    end_date: End date for data
    timeframe: Data timeframe/interval
    
Returns:
    DataFrame with market data

## data_handler

Data handling module for the trading system.

This module provides components for data loading, preprocessing, 
and management from various sources.

### Classes

#### `DataSource`

Abstract base class for data sources.

DataSource classes are responsible for fetching data from a specific source
(e.g., CSV files, APIs, databases) and converting it into a standardized format.

##### Methods

###### `get_data(symbols, start_date, end_date, timeframe)`

*Returns:* `pd.DataFrame`

Retrieve data for the specified symbols and date range.

Args:
    symbols: List of symbol identifiers to fetch
    start_date: Start date for the data
    end_date: End date for the data
    timeframe: Data timeframe (e.g., '1d', '1h', '5m')
    
Returns:
    DataFrame containing the requested data

###### `is_available(symbol, start_date, end_date, timeframe)`

*Returns:* `bool`

Check if data is available for the specified parameters.

Args:
    symbol: Symbol identifier to check
    start_date: Start date to check
    end_date: End date to check
    timeframe: Data timeframe
    
Returns:
    True if data is available, False otherwise

#### `CSVDataSource`

Data source that loads data from CSV files.

This class implements loading from a directory of CSV files
with standardized naming conventions.

##### Methods

###### `__init__(base_dir, filename_template='{symbol}_{timeframe}.csv', date_format='%Y-%m-%d')`

Initialize the CSV data source.

Args:
    base_dir: Base directory containing CSV files
    filename_template: Template for CSV filenames
    date_format: Date format string for parsing dates

###### `get_data(symbols, start_date, end_date, timeframe)`

*Returns:* `pd.DataFrame`

Load data from CSV files for the specified symbols and date range.

Args:
    symbols: List of symbol identifiers to fetch
    start_date: Start date for the data
    end_date: End date for the data
    timeframe: Data timeframe (e.g., '1d', '1h', '5m')

Returns:
    DataFrame containing the requested data

###### `is_available(symbol, start_date, end_date, timeframe)`

*Returns:* `bool`

Check if data is available in CSV files for the specified parameters.

Args:
    symbol: Symbol identifier to check
    start_date: Start date to check
    end_date: End date to check
    timeframe: Data timeframe

Returns:
    True if data is available, False otherwise

#### `DataHandler`

Main class for handling data operations in the trading system.

This class orchestrates data loading, processing, and management.
It serves as the primary interface for strategies to access data.

##### Methods

###### `__init__(data_source, train_fraction=0.8, event_bus=None)`

Initialize the data handler.

Args:
    data_source: DataSource instance for loading data
    train_fraction: Fraction of data to use for training (vs testing)
    event_bus: Optional event bus for emitting events

###### `set_event_bus(event_bus)`

*Returns:* `None`

Set the event bus for emitting events.

Args:
    event_bus: Event bus instance

###### `load_data(symbols, start_date, end_date, timeframe)`

*Returns:* `None`

Load data for multiple symbols.

Args:
    symbols: List of symbols to load
    start_date: Start date for data
    end_date: End date for data
    timeframe: Data timeframe (e.g., '1d', '1h', '5m')

###### `get_next_train_bar()`

*Returns:* `Optional[Dict[str, Any]]`

Get the next bar from the training data.

Returns:
    Dict containing bar data or None if no more data

###### `get_next_test_bar()`

*Returns:* `Optional[Dict[str, Any]]`

Get the next bar from the testing data.

Returns:
    Dict containing bar data or None if no more data

###### `get_next_train_bar_event()`

*Returns:* `Optional[BarEvent]`

Get the next bar from the training data as a BarEvent.

Returns:
    BarEvent object or None if no more data

###### `get_next_test_bar_event()`

*Returns:* `Optional[BarEvent]`

Get the next bar from the testing data as a BarEvent.

Returns:
    BarEvent object or None if no more data

###### `reset_train()`

*Returns:* `None`

Reset the training data iterator.

###### `reset_test()`

*Returns:* `None`

Reset the testing data iterator.

###### `reset()`

*Returns:* `None`

Reset both training and testing data iterators.

###### `iter_train(use_bar_events=True)`

Iterator for training data.

Args:
    use_bar_events: If True, yield BarEvent objects instead of dictionaries
    
Yields:
    Dict containing bar data or BarEvent object

###### `iter_test(use_bar_events=True)`

Iterator for testing data.

Args:
    use_bar_events: If True, yield BarEvent objects instead of dictionaries
    
Yields:
    Dict containing bar data or BarEvent object

###### `create_bar_event(bar_data)`

*Returns:* `BarEvent`

Create a standardized BarEvent from bar data.

Args:
    bar_data: Dictionary containing OHLCV data
    
Returns:
    BarEvent object

###### `emit_bar_event(bar_data)`

*Returns:* `None`

Create and emit a bar event.

Args:
    bar_data: Dictionary with OHLCV data or BarEvent

###### `process_bar(bar_data)`

*Returns:* `None`

Process a bar of market data, emitting an event if possible.

Args:
    bar_data: Dictionary with OHLCV data

###### `emit_all_bars(use_train=True)`

*Returns:* `int`

Emit bar events for all bars in the specified dataset.

Args:
    use_train: If True, use training data; otherwise use testing data
    
Returns:
    Number of events emitted

###### `get_symbol_data(symbol)`

*Returns:* `pd.DataFrame`

Get all data for a specific symbol.

Args:
    symbol: Symbol to retrieve data for
    
Returns:
    DataFrame containing data for the symbol

## data_sources

Data Sources Module

This module provides interfaces and implementations for different data sources
such as CSV files, databases, APIs, and real-time market data feeds.

### Classes

#### `DataSource`

Abstract base class for all data sources.

##### Methods

###### `get_data(symbol, start_date=None, end_date=None, timeframe='1d')`

*Returns:* `pd.DataFrame`

Retrieve data for a symbol within the specified date range.

Args:
    symbol: The instrument symbol
    start_date: Optional start date for data
    end_date: Optional end date for data
    timeframe: Data timeframe/resolution (e.g., '1m', '1h', '1d')
    
Returns:
    DataFrame with OHLCV data

###### `get_symbols()`

*Returns:* `List[str]`

Get list of available symbols.

Returns:
    List of symbol strings

#### `CSVDataSource`

Data source for CSV files.

##### Methods

###### `__init__(data_dir, filename_pattern='{symbol}_{timeframe}.csv', date_format='%Y-%m-%d', datetime_format='%Y-%m-%d %H:%M:%S', date_column='timestamp')`

Initialize CSV data source.

Args:
    data_dir: Directory containing CSV files
    filename_pattern: Pattern for CSV filenames with {symbol} and {timeframe} placeholders
    date_format: Format for date strings
    datetime_format: Format for datetime strings
    date_column: Column name for date/datetime

###### `_get_file_list()`

*Returns:* `List[str]`

Get list of CSV files in the data directory.

###### `_get_filename(symbol, timeframe)`

*Returns:* `str`

Get filename for a symbol and timeframe.

###### `get_symbols()`

*Returns:* `List[str]`

Get list of available symbols from CSV files.

###### `get_data(symbol, start_date=None, end_date=None, timeframe='1d')`

*Returns:* `pd.DataFrame`

Get data for a symbol from CSV file.

###### `get_latest_date(symbol, timeframe='1d')`

*Returns:* `Optional[datetime.datetime]`

Get the latest available date for a symbol.

#### `SQLiteDataSource`

Data source for SQLite database.

##### Methods

###### `__init__(db_path, table_pattern='{symbol}_{timeframe}')`

Initialize SQLite data source.

Args:
    db_path: Path to SQLite database file
    table_pattern: Pattern for table names with {symbol} and {timeframe} placeholders

###### `_get_connection()`

*Returns:* `sqlite3.Connection`

Get SQLite connection.

###### `_get_table_name(symbol, timeframe)`

*Returns:* `str`

Get table name for a symbol and timeframe.

###### `get_symbols()`

*Returns:* `List[str]`

Get list of available symbols from database tables.

###### `get_data(symbol, start_date=None, end_date=None, timeframe='1d')`

*Returns:* `pd.DataFrame`

Get data for a symbol from database.

#### `JSONDataSource`

Data source for JSON files.

##### Methods

###### `__init__(data_dir, filename_pattern='{symbol}_{timeframe}.json')`

Initialize JSON data source.

Args:
    data_dir: Directory containing JSON files
    filename_pattern: Pattern for JSON filenames with {symbol} and {timeframe} placeholders

###### `_get_file_list()`

*Returns:* `List[str]`

Get list of JSON files in the data directory.

###### `_get_filename(symbol, timeframe)`

*Returns:* `str`

Get filename for a symbol and timeframe.

###### `get_symbols()`

*Returns:* `List[str]`

Get list of available symbols from JSON files.

###### `get_data(symbol, start_date=None, end_date=None, timeframe='1d')`

*Returns:* `pd.DataFrame`

Get data for a symbol from JSON file.

#### `DataSourceRegistry`

Registry for data sources.

##### Methods

###### `register(cls, name, source)`

*Returns:* `None`

Register a data source.

Args:
    name: Name for the data source
    source: DataSource instance

###### `get(cls, name)`

*Returns:* `DataSource`

Get a registered data source.

Args:
    name: Name of the data source
    
Returns:
    DataSource instance
    
Raises:
    ValueError: If data source is not registered

###### `list_sources(cls)`

*Returns:* `List[str]`

Get list of registered data sources.

Returns:
    List of data source names

#### `DataCache`

Cache for market data to improve performance.

##### Methods

###### `set_max_size(cls, size)`

*Returns:* `None`

Set maximum cache size.

###### `get(cls, key)`

*Returns:* `Optional[pd.DataFrame]`

Get data from cache if available.

###### `set(cls, key, data)`

*Returns:* `None`

Store data in cache.

###### `clear(cls)`

*Returns:* `None`

Clear the cache.

###### `_enforce_max_size(cls)`

*Returns:* `None`

Enforce maximum cache size by removing oldest entries.

## data_transformer

Data Transformers Module

This module provides transformers for preprocessing market data before feeding
it to strategies and indicators. Transformers can handle operations like:
- Resampling to different timeframes
- Adjusting for splits and dividends
- Filling missing values
- Normalizing data
- Feature engineering

### Classes

#### `DataTransformer`

Base class for all data transformers.

##### Methods

###### `transform(data)`

*Returns:* `pd.DataFrame`

Transform input data.

Args:
    data: Input DataFrame
    
Returns:
    Transformed DataFrame

#### `ResampleTransformer`

Transformer for resampling time series data to different frequencies.

##### Methods

###### `__init__(timeframe='1h', aggregation=None)`

Initialize resampler.

Args:
    timeframe: Target timeframe/frequency (e.g., '1min', '5min', '1H', '1D')
    aggregation: Custom aggregation rules. Default uses OHLCV rules.

###### `transform(data)`

*Returns:* `pd.DataFrame`

Resample data to target timeframe.

Args:
    data: Input DataFrame with timestamp index or column
    
Returns:
    Resampled DataFrame

#### `MissingValueHandler`

Transformer for handling missing values in data.

##### Methods

###### `__init__(method='ffill', columns=None)`

Initialize missing value handler.

Args:
    method: Method for handling missing values ('ffill', 'bfill', 'zero', 'mean', 'median')
    columns: Specific columns to apply the handling to (None for all columns)

###### `transform(data)`

*Returns:* `pd.DataFrame`

Fill missing values in data.

Args:
    data: Input DataFrame
    
Returns:
    DataFrame with missing values handled

#### `AdjustedCloseTransformer`

Transformer for adjusting OHLC data using Adjusted Close.

##### Methods

###### `transform(data)`

*Returns:* `pd.DataFrame`

Adjust OHLC data using Adjusted Close.

Args:
    data: Input DataFrame with OHLC and Adj_Close columns
    
Returns:
    DataFrame with adjusted OHLC values

#### `ReturnCalculator`

Transformer for calculating returns from price data.

##### Methods

###### `__init__(periods, price_col='Close', log_returns=False)`

Initialize return calculator.

Args:
    periods: List of periods to calculate returns for
    price_col: Column to use for price data
    log_returns: Whether to calculate log returns (True) or simple returns (False)

###### `transform(data)`

*Returns:* `pd.DataFrame`

Calculate returns and add as new columns.

Args:
    data: Input DataFrame with price data
    
Returns:
    DataFrame with additional return columns

#### `NormalizationTransformer`

Transformer for normalizing price data.

##### Methods

###### `__init__(method='z-score', window=20, columns=None)`

Initialize normalizer.

Args:
    method: Normalization method ('z-score', 'min-max', 'decimal-scaling')
    window: Window size for rolling normalization (0 for full series)
    columns: Columns to normalize (None for all numeric columns)

###### `transform(data)`

*Returns:* `pd.DataFrame`

Normalize data.

Args:
    data: Input DataFrame
    
Returns:
    Normalized DataFrame

#### `FeatureEngineeringTransformer`

Transformer for engineering common technical features.

##### Methods

###### `__init__(features, params=None)`

Initialize feature engineer.

Args:
    features: List of features to engineer ('ma', 'ema', 'rsi', 'bbands', etc.)
    params: Dictionary of parameters for features

###### `transform(data)`

*Returns:* `pd.DataFrame`

Add engineered features to data.

Args:
    data: Input DataFrame
    
Returns:
    DataFrame with additional feature columns

#### `TransformerPipeline`

Pipeline for applying multiple transformers in sequence.

##### Methods

###### `__init__(transformers)`

Initialize transformer pipeline.

Args:
    transformers: List of transformers to apply in sequence

###### `transform(data)`

*Returns:* `pd.DataFrame`

Apply all transformers in sequence.

Args:
    data: Input DataFrame
    
Returns:
    Transformed DataFrame
