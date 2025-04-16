# Data Module

The Data module handles all aspects of market data, including fetching, transforming, storing, and providing data to other system components. It supports various data sources (CSV, databases, APIs) and provides a unified interface for accessing market data.

## Core Components

```
src/data/
├── __init__.py               # Package exports
├── data_handler.py           # DataHandler orchestration
├── data_sources.py           # Data source implementations
├── data_transformer.py       # Data transformation tools
└── data_connectors.py        # External data service connectors
```

## Key Classes

- `DataSource`: Interface for retrieving data from various origins
- `DataHandler`: Core component that manages data flow
- `DataTransformer`: Components for preprocessing and transforming raw data
- Various implementations (`CSVDataSource`, `SQLiteDataSource`, etc.)

## Basic Usage

```python
from src.data.data_sources import CSVDataSource
from src.data.data_handler import DataHandler
import datetime

# Create data source
data_source = CSVDataSource("data/csv")

# Create data handler
data_handler = DataHandler(data_source)

# Load data
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2022, 12, 31)
data_handler.load_data(
    symbols=["AAPL", "MSFT"],  # Note: load_data takes a list of symbols
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)

# Iterate through training data (yields dictionaries)
for bar in data_handler.iter_train():
    print(f"Date: {bar['timestamp']}, Close: {bar['Close']}")

# Access specific data
aapl_data = data_handler.get_symbol_data("AAPL")  # Returns DataFrame
```

## Important Interface Details

### DataSource Interface

DataSources provide raw data from specific origins:

```python
# Getting data for a specific symbol (NOT a list)
data = data_source.get_data(
    symbol="AAPL",            # Single string, not a list
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)
```

### DataHandler Interface

DataHandler orchestrates loading and providing data:

```python
# Loading data (takes a LIST of symbols)
data_handler.load_data(
    symbols=["AAPL", "MSFT"],  # List of symbols
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
    # NOTE: The load_data method does NOT accept a 'filename' parameter
    # It uses the data_source to find files based on symbols and timeframe
)

# Accessing data (returns DataFrame)
train_df = data_handler.train_data
test_df = data_handler.test_data

# Iterating through data (yields dictionaries)
for bar in data_handler.iter_train():
    # Each bar is a dictionary with OHLCV data
    process_bar(bar)
```

### Working with Custom Filenames

If you need to use a specific filename that doesn't follow the standard naming convention:

```python
# Option 1: Use a custom DataSource implementation
class CustomFileDataSource(CSVDataSource):
    def __init__(self, directory, filename_map=None):
        super().__init__(directory)
        self.filename_map = filename_map or {}
    
    def get_data(self, symbol, start_date, end_date, timeframe):
        # Check if we have a custom filename for this symbol/timeframe
        if symbol in self.filename_map:
            custom_filename = self.filename_map[symbol]
            file_path = os.path.join(self.directory, custom_filename)
            # Load from custom file
            return self._load_from_file(file_path, start_date, end_date)
        
        # Fall back to standard behavior
        return super().get_data(symbol, start_date, end_date, timeframe)

# Usage:
custom_source = CustomFileDataSource(
    directory=".",
    filename_map={"SYNTHETIC": "custom_synthetic_file.csv"}
)
data_handler = DataHandler(custom_source)
```

**Option 2**: Rename your file to follow the naming convention expected by CSVDataSource:

The CSVDataSource expects files to follow this naming convention:
```
{symbol}_{timeframe}.csv
```

For example:
- `AAPL_1d.csv` - Daily AAPL data
- `MSFT_1h.csv` - Hourly MSFT data

## CSV File Format

CSVs should have the following columns:
- **timestamp**: The datetime column (required)
- **Open**: Opening price (required)
- **High**: High price (required)
- **Low**: Low price (required)
- **Close**: Closing price (required)
- **Volume**: Volume (optional)
- **symbol**: The symbol/ticker (optional - if not present, the symbol from the filename is used)

## Integration with Event System

To emit market data events using the data handler:

```python
from src.events.event_bus import EventBus, Event
from src.events.event_types import EventType, BarEvent
from src.events.event_emitters import MarketDataEmitter

# Create event bus and emitter
event_bus = EventBus()
market_data_emitter = MarketDataEmitter(event_bus)

# Process bars from the data handler
for bar in data_handler.iter_train():
    # Create a BarEvent object
    bar_event = BarEvent(bar)
    
    # Create and emit an event with the bar event
    event = Event(EventType.BAR, bar_event)
    event_bus.emit(event)
```

## Bar Data Standardization

The system uses a standardized `BarEvent` class from the events module to encapsulate bar data consistently across all components.

### Using BarEvent Objects

```python
from src.events.event_types import BarEvent
from src.events.event_bus import Event
from src.events.event_types import EventType

# Create a BarEvent from dictionary data
bar_data = {
    "timestamp": datetime.now(),
    "Open": 100.0,
    "High": 101.0,
    "Low": 99.0,
    "Close": 100.5,
    "Volume": 1000,
    "symbol": "AAPL"
}
bar_event = BarEvent(bar_data)

# Create and emit a BAR event
event = Event(EventType.BAR, bar_event)
event_bus.emit(event)

# Accessing bar data in event handlers
def on_bar(self, event):
    bar_event = event.data
    if isinstance(bar_event, BarEvent):
        bar_data = bar_event.bar
        symbol = bar_event.get_symbol()
        close_price = bar_event.get_price()
        timestamp = bar_event.get_timestamp()
        # Process the bar data...
```

## Testing with Synthetic Data

When testing with synthetic data, make sure to:

1. Save the synthetic data to a file that follows the naming convention
2. Use the appropriate symbol name when loading the data

```python
# Generate synthetic data
synthetic_df = create_synthetic_data(symbol="SYNTHETIC", timeframe="1d")

# Save to a file following the naming convention: SYNTHETIC_1d.csv
synthetic_df.to_csv("SYNTHETIC_1d.csv", index=False)

# Then load using DataHandler
data_handler = DataHandler(CSVDataSource("."))
data_handler.load_data(
    symbols=["SYNTHETIC"],
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)
```

## Best Practices

1. **Follow the naming convention** for data files to work seamlessly with CSVDataSource
2. **Always call reset()** on the data handler before reusing it for a different dataset
3. **Provide complete date ranges** when loading data to avoid missing important bars
4. **Check that required columns exist** in your data files
5. **Use proper data types** in CSV files (especially for timestamp columns)
6. **Handle dates consistently** across your application
7. **Create a custom DataSource** if you need special file handling
8. **Use try/except** when loading data to handle potential file errors gracefully
