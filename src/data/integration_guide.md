# Data Module Integration Guide

This guide explains how to properly integrate the Data Module with other components of the trading system, particularly the Event System.

## Understanding the Data Flow

The Data Module provides a structured way to load and process market data:

1. A `DataSource` loads raw data from files, databases, or APIs
2. The `DataHandler` manages the data and splits it into training/testing sets
3. Components access the data via iterators or direct query methods

## Key Interfaces

### DataSource

The `DataSource` interface provides methods to fetch data from specific sources:

```python
class DataSource(ABC):
    @abstractmethod
    def get_data(self, symbol: str, start_date=None, end_date=None, timeframe=None) -> pd.DataFrame:
        """Get data for a specific symbol within the date range."""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
```

**Important Note**: The `get_data()` method takes a single symbol as a string parameter, not a list.

### DataHandler

The `DataHandler` orchestrates data loading and provides access methods:

```python
class DataHandler:
    def __init__(self, data_source, train_fraction=0.8):
        self.data_source = data_source
        self.train_fraction = train_fraction
        self.full_data = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self, symbols, start_date, end_date, timeframe):
        """Load data for multiple symbols."""
        # This method accepts a list of symbols
```

The DataHandler offers several methods to access the data:

- `get_next_train_bar()`: Get the next bar from training data as a dictionary
- `get_next_test_bar()`: Get the next bar from testing data as a dictionary
- `iter_train()`: Iterator over training bars
- `iter_test()`: Iterator over testing bars

## Integration with Event System

### Emitting Market Data Events

To emit market data events from the DataHandler:

```python
# Create event bus and market data emitter
event_bus = EventBus()
market_data_emitter = MarketDataEmitter(event_bus)

# Load data using the DataHandler
data_handler.load_data(
    symbols=["AAPL"],
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)

# Process bars from the training data
for bar in data_handler.iter_train():
    # Each bar is already a dictionary with OHLCV data
    market_data_emitter.emit_bar(bar)
```

Note that the `iter_train()` and `iter_test()` methods yield each bar as a dictionary, which is the format expected by the `emit_bar()` method.

### Handling DataFrame Access

The DataHandler stores data as pandas DataFrames, but converts them to dictionaries when accessed through iterators:

```python
# DataHandler.get_next_train_bar() method
def get_next_train_bar(self):
    if self.train_data is None or self.current_train_index >= len(self.train_data):
        return None
        
    # Convert DataFrame row to dictionary
    bar = self.train_data.iloc[self.current_train_index].to_dict()
    self.current_train_index += 1
    
    return bar
```

## Common Integration Mistakes

1. **Passing list to DataSource.get_data()**:
   ```python
   # INCORRECT
   data = data_source.get_data(symbols=["AAPL"])
   
   # CORRECT
   data = data_source.get_data(symbol="AAPL")
   ```

2. **Passing string to DataHandler.load_data()**:
   ```python
   # INCORRECT
   data_handler.load_data(symbols="AAPL")
   
   # CORRECT
   data_handler.load_data(symbols=["AAPL"])
   ```

3. **Directly iterating full_data**:
   ```python
   # INCORRECT - full_data is a DataFrame
   for bar in data_handler.full_data:
       market_data_emitter.emit_bar(bar)
   
   # CORRECT - use iterator methods
   for bar in data_handler.iter_train():
       market_data_emitter.emit_bar(bar)
   ```

## Full Integration Example

```python
from src.events.event_bus import EventBus
from src.events.event_types import EventType
from src.events.event_emitters import MarketDataEmitter
from src.data.data_sources import CSVDataSource
from src.data.data_handler import DataHandler
import datetime

# 1. Initialize event system
event_bus = EventBus()
market_data_emitter = MarketDataEmitter(event_bus)

# 2. Set up data source and handler
data_source = CSVDataSource("data/")
data_handler = DataHandler(data_source)

# 3. Load market data
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2022, 12, 31)
data_handler.load_data(
    symbols=["AAPL"],
    start_date=start_date,
    end_date=end_date,
    timeframe="1d"
)

# 4. Emit market open event
market_data_emitter.emit_market_open()

# 5. Process bars and emit events
for bar in data_handler.iter_train():
    market_data_emitter.emit_bar(bar)

# 6. Emit market close event
market_data_emitter.emit_market_close()
```

This integration approach ensures that data flows correctly from your Data Module to the Event System and the rest of your trading system components.