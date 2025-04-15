# Data Module

The Data module is responsible for handling all aspects of market data in the trading system, including fetching, transforming, storing, and providing data to other system components.

## Overview

This module provides a layered approach to data handling:

1. **Data Sources** - Interfaces for retrieving data from various origins (CSV files, APIs, databases)
2. **Data Transformers** - Components for preprocessing and transforming raw data
3. **Data Handlers** - Core components that manage data flow and provide a unified interface to the rest of the system
4. **Data Connectors** - Components for connecting to external data services
5. **Data Cache** - Optimizes performance by storing frequently accessed data

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

# Get data for a symbol - IMPORTANT: Note the difference between get_data and DataHandler.load_data
# This method takes a single symbol string, not a list
data = csv_source.get_data(
    symbol="AAPL",  # Single string, not a list
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    timeframe="1d"
)
```

### Data Handler

The `DataHandler` is the main class for orchestrating data operations. It provides:

- Loading data from various sources
- Splitting data into training and testing sets
- Iterating through data sequentially
- Filtering data by symbol or date range

```python
# Example: Using the DataHandler
from data.data_handler import DataHandler

# Create data handler with a data source
data_handler = DataHandler(csv_source)

# Load data - IMPORTANT: This takes a list of symbols
data_handler.load_data(
    symbols=["AAPL", "MSFT"],  # List of symbols
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    timeframe="1d"
)

# Iterate through training data
for bar in data_handler.iter_train():
    # Process each bar
    print(bar['timestamp'], bar['Close'])

# Get data for a specific symbol
aapl_data = data_handler.get_symbol_data("AAPL")

# Get data for a specific date range
range_data = data_handler.get_range(
    start_date=datetime(2022, 6, 1),
    end_date=datetime(2022, 6, 30)
)
```

### Data Transformers

The `DataTransformer` classes provide functionality to preprocess and transform market data:

- **ResampleTransformer** - Resamples data to different timeframes (e.g., 1m to 5m)
- **MissingValueHandler** - Fills in missing values using various methods
- **AdjustedCloseTransformer** - Adjusts OHLC data using Adjusted Close
- **ReturnCalculator** - Calculates returns from price data
- **NormalizationTransformer** - Normalizes data using different methods
- **FeatureEngineeringTransformer** - Creates technical indicators and derived features
- **TransformerPipeline** - Applies multiple transformers in sequence

```python
# Example: Using transformers
from data.data_transformer import (
    ResampleTransformer,
    FeatureEngineeringTransformer,
    TransformerPipeline
)

# Create a transformer pipeline
pipeline = TransformerPipeline([
    MissingValueHandler(method='ffill'),
    ResampleTransformer(timeframe='1h'),
    FeatureEngineeringTransformer(
        features=['ma', 'rsi', 'bbands'],
        params={'ma_periods': [10, 20, 50]}
    )
])

# Transform data
transformed_data = pipeline.transform(original_data)
```

### Data Connectors

The `DataConnector` classes facilitate connection to external data services:

- **APIConnector** - Base class for API connections
- **AlphaVantageConnector** - Connector for Alpha Vantage API
- **YahooFinanceConnector** - Connector for Yahoo Finance API
- **WebSocketClient** - Generic WebSocket client
- **MarketDataWebSocketClient** - WebSocket client for market data
- **DatabaseConnector** - Base class for database connections
- **SQLiteConnector** - Connector for SQLite databases

```python
# Example: Using an API connector
from data.data_connectors import AlphaVantageConnector

# Create connector
connector = AlphaVantageConnector(api_key="YOUR_API_KEY")

# Get historical data
historical_data = connector.get_historical_data(
    symbol="AAPL",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    timeframe="1d"
)

# Get latest price
current_price = connector.get_latest_price("AAPL")
```

### Data Cache

The `DataCache` optimizes performance by storing frequently accessed data:

```python
from data.data_sources import DataCache

# Configure cache
DataCache.set_max_size(100)  # Store up to 100 datasets

# Data is automatically cached when loaded through data sources
```

## Important Implementation Details

### Data Source vs. Data Handler Methods

**CRITICAL DIFFERENCE**: There's an important distinction between the methods in `DataSource` and `DataHandler`:

1. **DataSource.get_data()** takes a **single symbol string** parameter:
   ```python
   # CORRECT - single symbol string
   data_source.get_data(symbol="AAPL", start_date=start_date, end_date=end_date, timeframe="1d")
   ```

2. **DataHandler.load_data()** takes a **list of symbol strings**:
   ```python
   # CORRECT - list of symbols
   data_handler.load_data(symbols=["AAPL"], start_date=start_date, end_date=end_date, timeframe="1d")
   ```

This is a common source of errors. The error message `Data file not found: ./['SYNTHETIC']_1d.csv` indicates that a list is being passed where a string is expected, or vice versa.

### File Naming Conventions

The CSVDataSource expects files to follow a specific naming convention by default:

```
{symbol}_{timeframe}.csv
```

For example:
- `AAPL_1d.csv` - Daily AAPL data
- `MSFT_1h.csv` - Hourly MSFT data
- `BTC_1m.csv` - Minute BTC data

For testing with synthetic data, make sure your CSV file is named correctly:
```
SYNTHETIC_1d.csv
```

You can customize this pattern using the `filename_pattern` parameter when initializing the CSVDataSource:

```python
csv_source = CSVDataSource("data/csv", filename_pattern="{symbol}-{timeframe}.csv")
```

### Full Working Example

Here's a complete working example showing how to:
1. Create a synthetic dataset
2. Save it with the correct filename
3. Load it properly

```python
import pandas as pd
import os
from datetime import datetime, timedelta
from src.data.data_sources import CSVDataSource
from src.data.data_handler import DataHandler

# 1. Create a synthetic dataset
dates = pd.date_range(start='2022-01-01', end='2022-01-31', freq='D')
prices = list(range(100, 100 + len(dates)))

data = {
    'timestamp': dates,
    'Open': prices,
    'High': [p + 1 for p in prices],
    'Low': [p - 1 for p in prices],
    'Close': [p + 0.5 for p in prices],
    'Volume': [10000 for _ in prices]
}

df = pd.DataFrame(data)

# 2. Save it with the CORRECT filename pattern - VERY IMPORTANT
symbol = "SYNTHETIC"
timeframe = "1d"
filename = f"{symbol}_{timeframe}.csv"
df.to_csv(filename, index=False)
print(f"Saved data to {os.path.abspath(filename)}")

# 3. Create a data source pointing to the current directory
data_source = CSVDataSource(".")  # "." means current directory

# 4. Directly test the data source's get_data method
try:
    # Notice this uses a single symbol string, not a list
    test_data = data_source.get_data(
        symbol=symbol,  # Single string here
        start_date=dates[0],
        end_date=dates[-1],
        timeframe=timeframe
    )
    print(f"Successfully loaded data directly from data source: {len(test_data)} rows")
except Exception as e:
    print(f"Error loading data directly from data source: {str(e)}")

# 5. Create a data handler
data_handler = DataHandler(data_source)

# 6. Now properly load the data using load_data with a symbol LIST
try:
    data_handler.load_data(
        symbols=[symbol],  # LIST with one symbol
        start_date=dates[0],
        end_date=dates[-1],
        timeframe=timeframe
    )
    print(f"Successfully loaded data via data handler: {len(data_handler.full_data)} rows")
except Exception as e:
    print(f"Error loading data via data handler: {str(e)}")
```

## Troubleshooting Data Loading Issues

### Common Errors and Solutions

1. **FileNotFoundError: Data file not found: ./['SYNTHETIC']_1d.csv**
   - **Cause**: Passing a list of symbols to `data_source.get_data()` instead of a single string
   - **Solution**: Use a single string with `data_source.get_data()` or use `data_handler.load_data()` with a list

2. **TypeError: DataHandler.load_data() got an unexpected keyword argument 'filename'**
   - **Cause**: The `DataHandler.load_data()` method doesn't accept a filename parameter
   - **Solution**: DataHandler loads files based on the symbol and timeframe parameters. Remove the filename parameter.

3. **TypeError: CSVDataSource.get_data() missing 1 required positional argument: 'timeframe'**
   - **Cause**: The timeframe parameter is required but missing
   - **Solution**: Always include the timeframe parameter (e.g., '1d', '1h', '5m')

### Data File Format Requirements

CSVs should have the following columns:
- **timestamp** or **Date**: The datetime column (required)
- **Open**: Opening price (required)
- **High**: High price (required)
- **Low**: Low price (required)
- **Close**: Closing price (required)
- **Volume**: Volume (optional)

Here's a sample CSV file format:

```
timestamp,Open,High,Low,Close,Volume
2022-01-01,100.0,101.0,99.0,100.5,10000
2022-01-02,101.0,102.0,100.0,101.5,12000
```

### Step-by-Step Debugging Guide

If you're having issues loading data:

1. **Check if the file exists with the expected name**:
   ```python
   import os
   symbol = "SYNTHETIC"
   timeframe = "1d"
   expected_file = f"{symbol}_{timeframe}.csv"
   print(f"Looking for file: {expected_file}")
   print(f"File exists: {os.path.exists(expected_file)}")
   ```

2. **Look at the internal implementation** - The most common error is confusion between single string and list of symbols:
   ```python
   # DataSource.get_data() expects a single symbol string:
   data = data_source.get_data(symbol="AAPL", ...)  # CORRECT
   
   # DataHandler.load_data() expects a list of symbols:
   data_handler.load_data(symbols=["AAPL"], ...)  # CORRECT
   ```

3. **Check the columns in your CSV file**:
   ```python
   import pandas as pd
   df = pd.read_csv("SYNTHETIC_1d.csv")
   print(f"Columns in CSV: {df.columns.tolist()}")
   print(f"First few rows:\n{df.head()}")
   ```

4. **Verify the date format**:
   ```python
   df = pd.read_csv("SYNTHETIC_1d.csv")
   if 'timestamp' in df.columns:
       print(f"First timestamp: {df['timestamp'].iloc[0]}")
       # Convert to datetime if needed
       df['timestamp'] = pd.to_datetime(df['timestamp'])
       df.to_csv("SYNTHETIC_1d.csv", index=False)  # Save back
   ```

## Best Practices

1. **Understand the API differences** - Be clear about when to use a single symbol string vs. a list of symbols
2. **Use consistent file naming** - Follow the "{symbol}_{timeframe}.csv" convention
3. **Check file paths** - Verify that files are in the directory you expect
4. **Use absolute paths when in doubt** - Replace relative paths like "." with full absolute paths
5. **Parse dates consistently** - Ensure timestamps are properly parsed as datetime objects
6. **Validate CSV columns** - Make sure your CSV has all required columns (timestamp, OHLC)
7. **Keep the timeframe consistent** - Use the same timeframe string in filenames and method calls
8. **Use the transformer pipeline** - For consistent data preprocessing across different datasets
9. **Implement caching** - For datasets that are accessed frequently
10. **Handle missing data** - Use appropriate filling strategies for your specific use case
