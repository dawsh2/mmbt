# Data Module

The Data module is responsible for handling all aspects of market data in the trading system, including fetching, transforming, storing, and providing data to other system components.

## Overview

This module provides a layered approach to data handling:

1. **Data Sources** - Interfaces for retrieving data from various origins (CSV files, APIs, databases)
2. **Data Transformers** - Components for preprocessing and transforming raw data
3. **Data Handlers** - Core components that manage data flow and provide a unified interface to the rest of the system
4. **Data Connectors** - Components for connecting to external data services

## Core Components

### Data Sources

The `DataSource` abstract base class defines the interface for all data sources. Implementations include:

- **CSVDataSource** - For loading data from CSV files
- **SQLiteDataSource** - For retrieving data from SQLite databases
- **JSONDataSource** - For loading data from JSON files
- **DataSourceRegistry** - For managing and accessing multiple data sources

```python
# Example: Creating and using a CSV data source
from data.data_sources import CSVDataSource

# Create data source
csv_source = CSVDataSource("data/csv", filename_pattern="{symbol}_{timeframe}.csv")

# Get available symbols
symbols = csv_source.get_symbols()

# Get data for a symbol
data = csv_source.get_data(
    symbol="AAPL",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    timeframe="1d"
)
```

### Data Transformers

The `DataTransformer` abstract base class defines the interface for components that transform raw market data:

- **ResampleTransformer** - For changing the timeframe of data
- **MissingValueHandler** - For handling missing or invalid data points
- **AdjustedCloseTransformer** - For adjusting OHLC data using adjusted close prices
- **ReturnCalculator** - For calculating returns from price data
- **NormalizationTransformer** - For normalizing price data
- **FeatureEngineeringTransformer** - For adding technical indicators and features
- **TransformerPipeline** - For applying multiple transformers in sequence

```python
# Example: Creating a transformer pipeline
from data.data_transformers import (
    ResampleTransformer, 
    MissingValueHandler, 
    FeatureEngineeringTransformer,
    TransformerPipeline
)

# Create transformer pipeline
pipeline = TransformerPipeline([
    MissingValueHandler(method='ffill'),
    ResampleTransformer(timeframe='1h'),
    FeatureEngineeringTransformer(
        features=['ma', 'rsi', 'bbands'],
        params={'ma_periods': [10, 20, 50]}
    )
])

# Transform data
transformed_data = pipeline.transform(data)
```

### Data Handler

The `DataHandler` class is the main interface between data sources and the rest of the system:

- Manages loading data from sources
- Handles data splitting for train/test
- Implements iteration through data bars
- Provides methods for retrieving specific data subsets

```python
# Example: Using DataHandler
from data.data_handler import DataHandler
from data.data_sources import CSVDataSource

# Create data source and handler
data_source = CSVDataSource("data/csv")
handler = DataHandler(data_source, train_fraction=0.8)

# Load data
handler.load_data(
    symbols=["AAPL", "MSFT"],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2022, 12, 31),
    timeframe="1d"
)

# Iterate through training data
for bar in handler.iter_train():
    # Process bar...
    pass

# Iterate through testing data
for bar in handler.iter_test():
    # Process bar...
    pass

# Get data for specific symbol
apple_data = handler.get_symbol_data("AAPL")
```

### Data Connectors

The `DataConnector` classes provide interfaces to external data services and APIs:

- **APIConnector** - Base class for API connectors
- **AlphaVantageConnector** - For accessing Alpha Vantage API
- **YahooFinanceConnector** - For accessing Yahoo Finance data
- **WebSocketClient** - Base class for real-time data streams
- **MarketDataWebSocketClient** - For streaming market data
- **DatabaseConnector** - Base class for database connections
- **SQLiteConnector** - For SQLite database connections

```python
# Example: Using an API connector
from data.data_connectors import AlphaVantageConnector

# Create connector with API key
connector = AlphaVantageConnector(api_key="YOUR_API_KEY")

# Get historical data
data = connector.get_historical_data(
    symbol="AAPL",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    timeframe="1d"
)

# Get latest price
price = connector.get_latest_price("AAPL")
```

## Event Integration

The Data module integrates with the Events module, allowing market data to be emitted as events through the system:

- `CSVDataHandler` implements the `EventEmitter` interface
- Market data bars are emitted as `BAR` events
- WebSocket connectors emit real-time market data events

```python
# Example: Using a data handler as an event emitter
from data.data_handler import DataHandler
from events.event_bus import EventBus
from events.event_types import EventType

# Create components
event_bus = EventBus()
data_handler = DataHandler(data_source, event_bus=event_bus)

# Register a handler for bar events
event_bus.register(EventType.BAR, bar_handler)

# Load and process data
data_handler.load_data(symbols=["AAPL"], start_date=start_date, end_date=end_date)

# Process data bars as events
for bar in data_handler.iter_train():
    # Bar events are automatically emitted to the event bus
    pass
```

## Caching

The module implements caching mechanisms to improve performance:

- **DataCache** - For caching market data to avoid repeated disk reads
- **APIResponseCache** - For caching API responses to reduce API calls

```python
# Example: Setting up data cache
from data.data_sources import DataCache

# Configure cache size
DataCache.set_max_size(50)  # Cache up to 50 datasets

# Clear cache
DataCache.clear()
```

## Best Practices

1. **Use the highest-level interface** - In most cases, use `DataHandler` rather than working directly with sources
2. **Apply transformations consistently** - Create standard transformation pipelines for your data
3. **Implement caching** - Use caching for better performance with large datasets
4. **Handle missing values** - Always handle missing values properly to avoid issues in analysis
5. **Emit events efficiently** - When integrating with the event system, be mindful of event volume

## Examples

### Creating a Custom Data Source

```python
from data.data_sources import DataSource
import pandas as pd
from datetime import datetime

class MyCustomDataSource(DataSource):
    def __init__(self, api_key):
        self.api_key = api_key
        
    def get_data(self, symbol, start_date=None, end_date=None, timeframe="1d"):
        # Custom logic to fetch data from your source
        # ...
        
        # Return as pandas DataFrame with standardized format
        return df
        
    def is_available(self, symbol, start_date, end_date, timeframe):
        # Check if data is available
        # ...
        return True
```

### Building a Data Pipeline

```python
from data.data_sources import CSVDataSource
from data.data_transformers import MissingValueHandler, FeatureEngineeringTransformer, TransformerPipeline
from data.data_handler import DataHandler

# Create data source
source = CSVDataSource("data/csv")

# Create transformer pipeline
transformer = TransformerPipeline([
    MissingValueHandler(method="ffill"),
    FeatureEngineeringTransformer(
        features=["ma", "rsi", "bbands", "macd"],
        params={
            "ma_periods": [10, 20, 50, 200],
            "rsi_period": 14,
            "bb_period": 20
        }
    )
])

# Create data handler
handler = DataHandler(source)

# Load and transform data
handler.load_data(symbols=["AAPL"], start_date=start_date, end_date=end_date)
transformed_data = transformer.transform(handler.get_symbol_data("AAPL"))
```

### Implementing Real-time Data Streaming

```python
from data.data_connectors import MarketDataWebSocketClient
from events.event_bus import EventBus
from events.event_types import EventType

# Create event bus
event_bus = EventBus()

# Define event handlers
def on_quote(quote_data):
    # Process quote data
    event_bus.emit(EventType.TICK, quote_data)

def on_bar(bar_data):
    # Process bar data
    event_bus.emit(EventType.BAR, bar_data)

# Create WebSocket client
client = MarketDataWebSocketClient(
    url="wss://example.com/market-data",
    api_key="YOUR_API_KEY",
    on_quote=on_quote,
    on_bar=on_bar
)

# Start client in background
client.start()

# Subscribe to symbols
client.subscribe(["AAPL", "MSFT", "GOOG"])
```