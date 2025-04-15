## Data Module

The Data module handles all aspects of market data, including fetching, transforming, storing, and providing data to other system components. It supports various data sources (CSV, databases, APIs) and provides a unified interface for accessing market data.

### Core Components

```
src/data/
├── __init__.py               # Package exports
├── data_handler.py           # DataHandler orchestration
├── data_sources.py           # Data source implementations
├── data_transformer.py       # Data transformation tools
└── data_connectors.py        # External data service connectors
```

### Key Classes

- `DataSource`: Interface for retrieving data from various origins
- `DataHandler`: Core component that manages data flow
- `DataTransformer`: Components for preprocessing and transforming raw data
- Various implementations (`CSVDataSource`, `SQLiteDataSource`, etc.)

### Basic Usage

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

### Important Interface Details

#### DataSource Interface

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

#### DataHandler Interface

DataHandler orchestrates loading and providing data:

```python
# Loading data (takes a LIST of symbols)
data_handler.load_data(
    symbols=["AAPL", "MSFT"],  # List of symbols
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)

# Accessing data (returns DataFrame)
train_df = data_handler.train_data
test_df = data_handler.test_data

# Iterating through data (yields dictionaries)
for bar in data_handler.iter_train():
    # Each bar is a dictionary with OHLCV data
    process_bar(bar)
```

### Integration with Event System

To emit market data events using the data handler:

```python
from src.events.event_bus import EventBus
from src.events.event_emitters import MarketDataEmitter

# Create event bus and emitter
event_bus = EventBus()
market_data_emitter = MarketDataEmitter(event_bus)

# Process bars from the data handler
for bar in data_handler.iter_train():
    # Each bar is a dictionary ready for emission
    market_data_emitter.emit_bar(bar)
```

### Common File Format

CSVs should have the following columns:
- **timestamp**: The datetime column (required)
- **Open**: Opening price (required)
- **High**: High price (required)
- **Low**: Low price (required)
- **Close**: Closing price (required)
- **Volume**: Volume (optional)

### File Naming Convention

The CSVDataSource expects files to follow a specific naming convention by default:
```
{symbol}_{timeframe}.csv
```

For example:
- `AAPL_1d.csv` - Daily AAPL data
- `MSFT_1h.csv` - Hourly MSFT data
