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

## Important Notes on File Naming and Data Loading

### File Naming Conventions

The CSVDataSource expects files to follow a specific naming convention by default:

```
{symbol}_{timeframe}.csv
```

For example:
- `AAPL_1d.csv` - Daily AAPL data
- `MSFT_1h.csv` - Hourly MSFT data
- `BTC_1m.csv` - Minute BTC data

You can customize this pattern using the `filename_pattern` parameter when initializing the CSVDataSource:

```python
csv_source = CSVDataSource("data/csv", filename_pattern="{symbol}-{timeframe}.csv")
```

If your files don't follow a specific pattern, you might need to rename them or create a custom DataSource.

### Symbol Parameter Format

When using the `load_data` method in the DataHandler, always pass the symbols as a list, even if you only have one symbol:

```python
# CORRECT - symbols as a list
handler.load_data(symbols=["AAPL"], start_date=start_date, end_date=end_date)

# INCORRECT - will cause errors
handler.load_data(symbols="AAPL", start_date=start_date, end_date=end_date)
```

### Timezone Handling

#### Timezone-Aware vs Timezone-Naive Datetimes

A common source of errors is mismatched timezone information between your data and filtering parameters. 

If your data contains timezone-aware datetimes, your filter dates must also be timezone-aware:

```python
# For timezone-aware data
import pandas as pd

# Create timezone-aware timestamps
start_date = pd.Timestamp('2020-01-01').tz_localize('UTC')
end_date = pd.Timestamp('2022-12-31').tz_localize('UTC')

data_handler.load_data(
    symbols=["AAPL"],
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)
```

You can check if your data has timezone information:

```python
# Load a small sample first
data = data_handler.data_source.get_data(symbol, None, None, timeframe)
first_date = data['Date'].iloc[0]

# Check if it has timezone info
if hasattr(first_date, 'tzinfo') and first_date.tzinfo is not None:
    print("Data has timezone information")
    # Make your filter dates timezone-aware too
    import pytz
    start_date = start_date.tz_localize('UTC')
    end_date = end_date.tz_localize('UTC')
```

#### Recommended Approach

To avoid timezone issues, consider:

1. Standardizing to UTC timezone consistently throughout your system
2. Explicitly converting all datetimes to timezone-aware or timezone-naive format
3. Using pandas Timestamp objects for consistency rather than Python datetime

```python
# Standardizing approach
import pandas as pd
import pytz

# Convert string to timezone-aware timestamp
def to_utc_timestamp(date_string):
    if isinstance(date_string, str):
        timestamp = pd.to_datetime(date_string)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize('UTC')
        return timestamp.tz_convert('UTC')
    return date_string

# Use in data filtering
start_date = to_utc_timestamp('2020-01-01')
end_date = to_utc_timestamp('2022-12-31')
```

### CSV Format Requirements

CSVs should have the following columns:
- **Date** or **Timestamp**: The datetime column (required)
- **Open**: Opening price (required)
- **High**: High price (required)
- **Low**: Low price (required)
- **Close**: Closing price (required)
- **Volume**: Volume (optional)

The date/timestamp column name can be customized with the `date_column` parameter in CSVDataSource.

### Troubleshooting Data Loading

If you encounter data loading issues:

1. **Verify the file exists and follows the expected naming convention**:
   ```python
   import os
   expected_file = f"data/{symbol}_{timeframe}.csv"
   print(f"Looking for file: {expected_file}")
   print(f"File exists: {os.path.exists(expected_file)}")
   ```

2. **Check the data format by reading directly with pandas**:
   ```python
   import pandas as pd
   df = pd.read_csv(f"data/{symbol}_{timeframe}.csv")
   print(f"Columns: {df.columns.tolist()}")
   print(f"Date format: {type(df['Date'].iloc[0])}")
   ```

3. **Test timezone compatibility**:
   ```python
   # If you have timezone-aware dates in your CSV
   dates = pd.to_datetime(df['Date'])
   if any(d.tzinfo is not None for d in dates):
       print("CSV contains timezone-aware dates")
       # Make your filtering dates timezone-aware too
       start_date = pd.Timestamp('2020-01-01').tz_localize('UTC')
       end_date = pd.Timestamp('2022-12-31').tz_localize('UTC')
   ```

4. **Try loading without date filtering first**:
   ```python
   # Load without date filtering
   data_handler.load_data(
       symbols=[symbol],
       start_date=None,
       end_date=None,
       timeframe=timeframe
   )
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
6. **Be consistent with timezones** - Either use timezone-aware timestamps throughout or timezone-naive throughout
7. **Validate data format** - Check that your CSV files have the expected columns and formats
8. **Pass symbols as lists** - Always pass symbols as a list to the data_handler, even for a single symbol
