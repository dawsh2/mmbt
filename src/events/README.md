# Events Module

The Events module provides a robust event-driven architecture for the backtesting system. It enables decoupled communication between components through a central event bus, ensuring flexibility and maintainability in the trading system architecture.

## Core Concepts

The events module is built around these key concepts:

1. **Events**: Self-contained messages that include a type, data payload, and timestamp
2. **Event Types**: Enumeration of supported event categories in the system
3. **Event Bus**: Central message broker that routes events between components
4. **Event Handlers**: Components that process specific types of events
5. **Event Emitters**: Components that generate and publish events

## Architecture Overview

```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐
│  Event Emitters │────▶│   Event Bus   │────▶│ Event Handlers │
└─────────────────┘     └───────────────┘     └────────────────┘
   (Producers)             (Broker)            (Consumers)
```

This architecture enables:

- **Loose Coupling**: Components communicate without direct dependencies
- **Flexibility**: Easy to add, remove, or modify components
- **Centralization**: All messages flow through a common bus
- **Standardization**: Well-defined event types with consistent structures

## Core Components

### EventType

The `EventType` enum defines all supported event categories in the system:

```python
from src.events.event_types import EventType

# Market data events
event_type = EventType.BAR
event_type = EventType.TICK
event_type = EventType.MARKET_OPEN
event_type = EventType.MARKET_CLOSE

# Signal events
event_type = EventType.SIGNAL

# Order events
event_type = EventType.ORDER
event_type = EventType.CANCEL
event_type = EventType.MODIFY

# Execution events
event_type = EventType.FILL
event_type = EventType.PARTIAL_FILL
event_type = EventType.REJECT

# Portfolio events
event_type = EventType.POSITION_OPENED
event_type = EventType.POSITION_CLOSED
event_type = EventType.POSITION_MODIFIED

# System events
event_type = EventType.START
event_type = EventType.STOP
event_type = EventType.PAUSE
event_type = EventType.RESUME
event_type = EventType.ERROR

# Analysis events
event_type = EventType.METRIC_CALCULATED
event_type = EventType.ANALYSIS_COMPLETE

# Custom events
event_type = EventType.CUSTOM
```

You can group event types by category:
```python
# Get all market data events
market_events = EventType.market_data_events()  # Returns {BAR, TICK, MARKET_OPEN, MARKET_CLOSE}

# Get all order-related events
order_events = EventType.order_events()  # Returns {ORDER, CANCEL, MODIFY, FILL, PARTIAL_FILL, REJECT}
```

### Event

The `Event` class represents a single event in the system:

```python
from src.events.event_bus import Event
from src.events.event_types import EventType
from datetime import datetime

# Create an event
event = Event(
    event_type=EventType.BAR,
    data={"symbol": "AAPL", "close": 150.75},
    timestamp=datetime.now()
)

# Access event properties
print(f"Event type: {event.event_type}")
print(f"Event ID: {event.id}")
print(f"Event timestamp: {event.timestamp}")
print(f"Event data: {event.data}")
```

### EventBus

The `EventBus` serves as the central hub for all event communication:

```python
from src.events.event_bus import EventBus
from src.events.event_types import EventType

# Create an event bus
event_bus = EventBus()

# Create and emit an event
event = Event(EventType.BAR, {"symbol": "AAPL", "close": 150.75})
event_bus.emit(event)

# Get event history
market_data_events = event_bus.get_history(EventType.BAR)
```

### EventHandler

`EventHandler` is the base class for all components that process events:

```python
from src.events.event_handlers import EventHandler
from src.events.event_types import EventType

class MySignalHandler(EventHandler):
    def __init__(self, portfolio_manager):
        # Initialize with the event types to handle
        super().__init__([EventType.SIGNAL])
        self.portfolio_manager = portfolio_manager
    
    def _process_event(self, event):
        # Process the event
        signal = event.data
        self.portfolio_manager.on_signal(signal)
```

### EventEmitter

`EventEmitter` is a mixin class for components that emit events:

```python
from src.events.event_emitters import EventEmitter
from src.events.event_types import EventType
from datetime import datetime

class Strategy(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def generate_signal(self, symbol, direction):
        # Create signal data
        signal = {
            "symbol": symbol,
            "direction": direction,
            "timestamp": datetime.now()
        }
        
        # Emit signal event
        self.emit(EventType.SIGNAL, signal)
```

## Specialized Event Handlers

The module provides several specialized event handlers:

### FunctionEventHandler

Handler that delegates processing to a function:

```python
from src.events.event_handlers import FunctionEventHandler

# Create a function handler
def handle_market_data(event):
    bar_data = event.data
    print(f"Processing bar data: {bar_data['symbol']}")

market_data_handler = FunctionEventHandler(EventType.BAR, handle_market_data)
event_bus.register(EventType.BAR, market_data_handler)
```

### LoggingHandler

Handler that logs events at specified levels:

```python
from src.events.event_handlers import LoggingHandler
import logging

# Create a logging handler for specific event types
logging_handler = LoggingHandler([EventType.BAR, EventType.SIGNAL])

# Set different log levels for different event types
logging_handler.set_event_log_level(EventType.BAR, logging.DEBUG)
logging_handler.set_event_log_level(EventType.SIGNAL, logging.INFO)

# Register with event bus
event_bus.register(EventType.BAR, logging_handler)
event_bus.register(EventType.SIGNAL, logging_handler)
```

### FilterHandler

Handler that only processes events matching specific criteria:

```python
from src.events.event_handlers import FilterHandler

# Create the base handler
base_handler = MySignalHandler(portfolio_manager)

# Create filter function
def high_confidence_filter(event):
    return event.data.get('confidence', 0) >= 0.7

# Create filtered handler
filtered_handler = FilterHandler(
    [EventType.SIGNAL], 
    base_handler, 
    high_confidence_filter
)

# Register the filtered handler
event_bus.register(EventType.SIGNAL, filtered_handler)
```

### DebounceHandler

Handler that prevents processing the same event type too frequently:

```python
from src.events.event_handlers import DebounceHandler

# Create base handler
base_handler = MySignalHandler(portfolio_manager)

# Create debounced handler (min 100ms between events)
debounced_handler = DebounceHandler(
    [EventType.SIGNAL],
    base_handler,
    0.1  # 100 milliseconds
)

# Register with event bus
event_bus.register(EventType.SIGNAL, debounced_handler)
```

## Specialized Event Emitters

The module provides specialized event emitters for different event types:

### MarketDataEmitter

```python
from src.events.event_emitters import MarketDataEmitter

# Create a market data emitter
market_data_emitter = MarketDataEmitter(event_bus)

# Emit a bar event
market_data_emitter.emit_bar({
    "symbol": "AAPL",
    "open": 150.0,
    "high": 151.5,
    "low": 149.5,
    "close": 151.0,
    "volume": 1000000
})

# Emit market open/close events
market_data_emitter.emit_market_open()
market_data_emitter.emit_market_close()
```

### SignalEmitter

```python
from src.events.event_emitters import SignalEmitter

# Create a signal emitter
signal_emitter = SignalEmitter(event_bus)

# Emit a signal
signal_emitter.emit_signal(
    symbol="AAPL",
    signal_type="BUY",
    price=151.0,
    confidence=0.8,
    rule_id="sma_crossover"
)
```

### OrderEmitter

```python
from src.events.event_emitters import OrderEmitter

# Create an order emitter
order_emitter = OrderEmitter(event_bus)

# Emit an order
order_emitter.emit_order(
    symbol="AAPL",
    order_type="MARKET",
    quantity=100,
    direction=1,  # Buy
    price=151.0
)

# Emit a cancel order
order_emitter.emit_cancel("order-123", reason="Strategy change")
```

### FillEmitter

```python
from src.events.event_emitters import FillEmitter

# Create a fill emitter
fill_emitter = FillEmitter(event_bus)

# Emit a fill event
fill_emitter.emit_fill(
    order_id="order-123",
    symbol="AAPL",
    quantity=100,
    price=151.25,
    direction=1,
    transaction_cost=7.50
)
```

## Domain-Specific Handlers

The module includes handlers for specific trading system components:

### MarketDataHandler

Handles market data events and forwards them to strategies:

```python
from src.events.event_handlers import MarketDataHandler

# Create market data handler
market_data_handler = MarketDataHandler(strategy)

# Register with event bus
event_bus.register(EventType.BAR, market_data_handler)
event_bus.register(EventType.TICK, market_data_handler)
```

### SignalHandler

Handles signal events and forwards them to portfolio management:

```python
from src.events.event_handlers import SignalHandler

# Create signal handler
signal_handler = SignalHandler(portfolio_manager)

# Register with event bus
event_bus.register(EventType.SIGNAL, signal_handler)
```

### OrderHandler

Handles order events and forwards them to execution engine:

```python
from src.events.event_handlers import OrderHandler

# Create order handler
order_handler = OrderHandler(execution_engine)

# Register with event bus
event_bus.register(EventType.ORDER, order_handler)
event_bus.register(EventType.CANCEL, order_handler)
event_bus.register(EventType.MODIFY, order_handler)
```

### FillHandler

Handles fill events and updates portfolio positions:

```python
from src.events.event_handlers import FillHandler

# Create fill handler
fill_handler = FillHandler(portfolio_manager)

# Register with event bus
event_bus.register(EventType.FILL, fill_handler)
event_bus.register(EventType.PARTIAL_FILL, fill_handler)
```

## Integration Examples

### Basic Trading System Flow

```python
from src.events.event_bus import EventBus
from src.events.event_handlers import MarketDataHandler, SignalHandler, OrderHandler, FillHandler
from src.events.event_emitters import MarketDataEmitter

# Create event bus
event_bus = EventBus()

# Create components
strategy = Strategy()
portfolio_manager = PortfolioManager()
execution_engine = ExecutionEngine()

# Create handlers
market_data_handler = MarketDataHandler(strategy)
signal_handler = SignalHandler(portfolio_manager)
order_handler = OrderHandler(execution_engine)
fill_handler = FillHandler(portfolio_manager)

# Register handlers with event bus
event_bus.register(EventType.BAR, market_data_handler)
event_bus.register(EventType.SIGNAL, signal_handler)
event_bus.register(EventType.ORDER, order_handler)
event_bus.register(EventType.FILL, fill_handler)

# Create market data emitter
market_data_emitter = MarketDataEmitter(event_bus)

# Simulate a bar event
market_data_emitter.emit_bar({
    "symbol": "AAPL",
    "open": 150.0,
    "high": 151.5,
    "low": 149.5,
    "close": 151.0,
    "volume": 1000000
})

# The event flows through the system:
# 1. Bar event emitted
# 2. MarketDataHandler processes it and calls strategy.on_bar()
# 3. Strategy generates a signal event
# 4. SignalHandler processes it and creates an order
# 5. OrderHandler processes the order event
# 6. ExecutionEngine executes the order and emits a fill event
# 7. FillHandler updates portfolio positions
```

### Integration with Backtester

```python
from src.events.event_bus import EventBus
from src.engine.backtester import Backtester
from src.strategies import WeightedStrategy

# Create event bus
event_bus = EventBus()

# Create components with shared event bus
strategy = WeightedStrategy(components=rule_objects, weights=[0.4, 0.3, 0.3])
backtester = Backtester(config, data_handler, strategy)

# Run backtest (automatically uses event bus for communication)
results = backtester.run()
```

## Best Practices

1. **Register handlers early**: Register all event handlers at initialization time before emitting any events.

2. **Use weak references**: The EventBus uses weak references to handlers to prevent memory leaks. Make sure you maintain a reference to your handlers elsewhere if they need to persist.

3. **Keep handlers focused**: Each handler should focus on a specific task. Use composition to build complex handlers from simpler ones.

4. **Error handling**: Implement proper error handling in event handlers to prevent exceptions from halting the entire system.

5. **Event history**: Use event history for debugging and analysis, but be mindful of memory usage with large backtests.

6. **Consider thread safety**: If using multiple threads, be aware of potential race conditions in event handling.

7. **Event schemas**: Use consistent data structures for each event type to prevent downstream processing errors.

## Event Type Reference

### Market Data Events
- `BAR`: New price bar with OHLCV data
- `TICK`: Individual tick data with price and volume
- `MARKET_OPEN`: Market opening notification
- `MARKET_CLOSE`: Market closing notification

### Signal Events
- `SIGNAL`: Trading signal generated by a strategy

### Order Events
- `ORDER`: Order request for execution
- `CANCEL`: Order cancellation request
- `MODIFY`: Order modification request

### Execution Events
- `FILL`: Order completely filled
- `PARTIAL_FILL`: Order partially filled
- `REJECT`: Order rejected by broker or exchange

### Portfolio Events
- `POSITION_OPENED`: New position opened
- `POSITION_CLOSED`: Existing position closed
- `POSITION_MODIFIED`: Position size or parameters modified

### System Events
- `START`: System or component start
- `STOP`: System or component stop
- `PAUSE`: System or component pause
- `RESUME`: System or component resume
- `ERROR`: System or component error

### Analysis Events
- `METRIC_CALCULATED`: Performance metric calculation completed
- `ANALYSIS_COMPLETE`: Analysis process completed

### Custom Events
- `CUSTOM`: Custom event type for user-defined events
