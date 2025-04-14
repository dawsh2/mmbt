# Events Module

The Events module implements a robust event-driven architecture for the trading system. It provides a way to decouple system components while enabling efficient communication between them through a central event bus.

## Overview

This module enables:

1. **Loose Coupling** - Components communicate without direct dependencies
2. **Flexible Architecture** - Easy to add, remove, or modify components
3. **Centralized Communication** - All system messages flow through a common bus
4. **Standardized Messaging** - Well-defined event types with consistent structures
5. **Robust Error Handling** - Isolated error handling at the event level
6. **Asynchronous Processing** - Optional async event processing

## Core Components

### Event Types

The `EventType` enum defines all supported event types within the system:

```python
from events.event_types import EventType

# Example event types
market_data_event = EventType.BAR
signal_event = EventType.SIGNAL
order_event = EventType.ORDER
fill_event = EventType.FILL

# Get all event types in a category
market_data_events = EventType.market_data_events()  # Returns set of market data events
```

Categories of event types include:

- **Market Data Events**: `BAR`, `TICK`, `MARKET_OPEN`, `MARKET_CLOSE`
- **Signal Events**: `SIGNAL`
- **Order Events**: `ORDER`, `CANCEL`, `MODIFY`
- **Execution Events**: `FILL`, `PARTIAL_FILL`, `REJECT`
- **Portfolio Events**: `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_MODIFIED`
- **System Events**: `START`, `STOP`, `PAUSE`, `RESUME`, `ERROR`
- **Analysis Events**: `METRIC_CALCULATED`, `ANALYSIS_COMPLETE`
- **Custom Events**: `CUSTOM`

### Event

The `Event` class represents a single event in the system:

```python
from events.event_bus import Event
from events.event_types import EventType
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

# Convert to dictionary
event_dict = event.to_dict()

# Create from dictionary
reconstructed_event = Event.from_dict(event_dict)
```

Each event has:
- A unique event type
- A timestamp
- A unique ID
- Optional data payload

### Event Handlers

`EventHandler` is the base class for all components that process events:

```python
from events.event_handlers import EventHandler
from events.event_types import EventType
from events.event_bus import Event

class MySignalHandler(EventHandler):
    def __init__(self, portfolio_manager):
        # Initialize with the event types to handle
        super().__init__([EventType.SIGNAL])
        self.portfolio_manager = portfolio_manager
    
    def _process_event(self, event):
        # This method is called for each event
        signal = event.data
        self.portfolio_manager.on_signal(signal)
```

Special handler types include:

- **FunctionEventHandler**: Uses a function as a handler
- **LoggingHandler**: Logs events at specified levels
- **DebounceHandler**: Prevents handling too many events of the same type
- **FilterHandler**: Only handles events that pass a filter
- **AsyncEventHandler**: Processes events asynchronously

```python
from events.event_handlers import FunctionEventHandler

# Create a function-based handler
def handle_signal(event):
    signal = event.data
    print(f"Processing signal for {signal['symbol']}")

signal_handler = FunctionEventHandler(EventType.SIGNAL, handle_signal)
```

### Event Bus

The `EventBus` is the central hub for all event communication:

```python
from events.event_bus import EventBus, Event
from events.event_types import EventType
from events.event_handlers import EventHandler

# Create an event bus
event_bus = EventBus()

# Create and register a handler
handler = MySignalHandler(portfolio_manager)
event_bus.register(EventType.SIGNAL, handler)

# Create and emit an event
event = Event(EventType.SIGNAL, {"symbol": "AAPL", "direction": 1})
event_bus.emit(event)

# Get event history
signal_events = event_bus.get_history(EventType.SIGNAL)
```

Key features:
- Register handlers for specific event types
- Emit events to registered handlers
- Maintain event history
- Support for async event processing
- Weak references to prevent memory leaks

### Event Emitters

`EventEmitter` is a mixin class for components that emit events:

```python
from events.event_emitters import EventEmitter
from events.event_types import EventType
from events.event_bus import EventBus

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

Specialized emitters include:

- **MarketDataEmitter**: For emitting market data events
- **SignalEmitter**: For emitting trading signals
- **OrderEmitter**: For emitting order-related events
- **FillEmitter**: For emitting fill-related events
- **PortfolioEmitter**: For emitting portfolio events
- **SystemEmitter**: For emitting system events

```python
from events.event_emitters import MarketDataEmitter

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
```

### Event Schemas

The `schema` module defines data schemas for different event types:

```python
from events.schema import validate_event_data, get_schema_documentation

# Get documentation for an event schema
print(get_schema_documentation('SIGNAL'))

# Validate event data against schema
signal_data = {
    'timestamp': datetime.now(),
    'signal_type': 'BUY',
    'price': 150.75,
    'symbol': 'AAPL',
    'confidence': 0.8
}

validated_data = validate_event_data('SIGNAL', signal_data)
```

Schemas are available for:
- `BAR` events
- `SIGNAL` events
- `ORDER` events
- `FILL` events
- `MARKET_CLOSE` events
- And more

## Event Groups and Hierarchies

Components can be organized in event handler groups:

```python
from events.event_handlers import EventHandlerGroup

# Create a group of related handlers
strategy_handlers = EventHandlerGroup("strategy_handlers", [
    signal_handler,
    order_handler,
    market_data_handler
])

# Enable or disable all handlers in the group
strategy_handlers.disable_all()
strategy_handlers.enable_all()
```

## Asynchronous Event Processing

The event bus supports asynchronous event processing:

```python
from events.event_bus import EventBus

# Create async event bus
async_event_bus = EventBus(async_mode=True)

# Register handlers and emit events as usual
async_event_bus.register(EventType.BAR, handler)
async_event_bus.emit(event)

# Stop async processing when done
async_event_bus.stop_dispatch_thread()
```

## Event Monitoring and Caching

The module includes tools for monitoring and caching events:

```python
from events.event_handlers import LoggingHandler
from events.event_bus import EventCacheManager
import logging

# Create a logging handler for all events
logging_handler = LoggingHandler(list(EventType), logging.INFO)
for event_type in EventType:
    event_bus.register(event_type, logging_handler)

# Create event cache manager
cache_manager = EventCacheManager(max_cache_size=100)

# Cache an event for reuse
cache_manager.cache_event(event, key="AAPL_2023-06-15")

# Retrieve cached event
cached_event = cache_manager.get_cached_event(EventType.BAR, key="AAPL_2023-06-15")
```

## Integration Examples

### Basic Event Flow

```python
from events.event_bus import EventBus, Event
from events.event_types import EventType
from events.event_handlers import FunctionEventHandler

# Create event bus
event_bus = EventBus()

# Define handlers
def handle_bar(event):
    bar_data = event.data
    print(f"Processing bar: {bar_data['symbol']} @ {bar_data['close']}")

def handle_signal(event):
    signal = event.data
    print(f"Processing signal: {signal['symbol']} {signal['direction']}")

def handle_order(event):
    order = event.data
    print(f"Processing order: {order['symbol']} {order['quantity']} @ {order['price']}")

# Register handlers
event_bus.register(EventType.BAR, FunctionEventHandler(EventType.BAR, handle_bar))
event_bus.register(EventType.SIGNAL, FunctionEventHandler(EventType.SIGNAL, handle_signal))
event_bus.register(EventType.ORDER, FunctionEventHandler(EventType.ORDER, handle_order))

# Emit events
event_bus.emit(Event(EventType.BAR, {"symbol": "AAPL", "close": 150.0}))
event_bus.emit(Event(EventType.SIGNAL, {"symbol": "AAPL", "direction": 1}))
event_bus.emit(Event(EventType.ORDER, {"symbol": "AAPL", "quantity": 100, "price": 150.0}))
```

### Complete Trading System Event Flow

```python
from events.event_bus import EventBus
from events.event_types import EventType
from events.event_handlers import EventHandler
from events.event_emitters import EventEmitter

# Create event bus
event_bus = EventBus()

# Create components
class DataHandler(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def process_bar(self, bar_data):
        # Emit bar event
        self.emit(EventType.BAR, bar_data)
        
class Strategy(EventHandler, EventEmitter):
    def __init__(self, event_bus):
        EventHandler.__init__(self, [EventType.BAR])
        EventEmitter.__init__(self, event_bus)
        
    def _process_event(self, event):
        if event.event_type == EventType.BAR:
            bar_data = event.data
            
            # Simple strategy: Buy when price increases
            if bar_data["close"] > bar_data["open"]:
                signal = {
                    "symbol": bar_data["symbol"],
                    "direction": 1,  # Buy
                    "price": bar_data["close"]
                }
                
                # Emit signal event
                self.emit(EventType.SIGNAL, signal)
                
class PortfolioManager(EventHandler, EventEmitter):
    def __init__(self, event_bus):
        EventHandler.__init__(self, [EventType.SIGNAL, EventType.FILL])
        EventEmitter.__init__(self, event_bus)
        
    def _process_event(self, event):
        if event.event_type == EventType.SIGNAL:
            signal = event.data
            
            # Convert signal to order
            order = {
                "symbol": signal["symbol"],
                "direction": signal["direction"],
                "quantity": 100,
                "price": signal["price"],
                "order_type": "MARKET"
            }
            
            # Emit order event
            self.emit(EventType.ORDER, order)
            
        elif event.event_type == EventType.FILL:
            fill = event.data
            print(f"Position opened: {fill['symbol']} {fill['quantity']} @ {fill['price']}")
            
class ExecutionHandler(EventHandler, EventEmitter):
    def __init__(self, event_bus):
        EventHandler.__init__(self, [EventType.ORDER])
        EventEmitter.__init__(self, event_bus)
        
    def _process_event(self, event):
        if event.event_type == EventType.ORDER:
            order = event.data
            
            # Process order to fill
            fill = {
                "symbol": order["symbol"],
                "quantity": order["quantity"],
                "price": order["price"],
                "direction": order["direction"],
                "timestamp": datetime.now()
            }
            
            # Emit fill event
            self.emit(EventType.FILL, fill)

# Create and connect components
data_handler = DataHandler(event_bus)
strategy = Strategy(event_bus)
portfolio = PortfolioManager(event_bus)
execution = ExecutionHandler(event_bus)

# Register handlers
event_bus.register(EventType.BAR, strategy)
event_bus.register(EventType.SIGNAL, portfolio)
event_bus.register(EventType.ORDER, execution)
event_bus.register(EventType.FILL, portfolio)

# Process a bar
data_handler.process_bar({
    "symbol": "AAPL",
    "open": 150.0,
    "high": 152.5,
    "low": 149.5,
    "close": 152.0,
    "volume": 1000000
})
```

### Custom Event Filtering

```python
from events.event_bus import EventBus, Event
from events.event_types import EventType
from events.event_handlers import EventHandler, FilterHandler

# Create event bus
event_bus = EventBus()

# Create a base handler
class SignalProcessor(EventHandler):
    def __init__(self):
        super().__init__([EventType.SIGNAL])
        
    def _process_event(self, event):
        signal = event.data
        print(f"Processing signal: {signal['symbol']} {signal['direction']}")

# Create filter function
def high_confidence_filter(event):
    signal = event.data
    return signal.get('confidence', 0) >= 0.7

# Create filtered handler
signal_processor = SignalProcessor()
filtered_handler = FilterHandler(
    [EventType.SIGNAL], 
    signal_processor, 
    high_confidence_filter
)

# Register the filtered handler
event_bus.register(EventType.SIGNAL, filtered_handler)

# Emit signals
event_bus.emit(Event(EventType.SIGNAL, {
    "symbol": "AAPL", 
    "direction": 1, 
    "confidence": 0.8
}))  # Will be processed

event_bus.emit(Event(EventType.SIGNAL, {
    "symbol": "MSFT", 
    "direction": 1, 
    "confidence": 0.5
}))  # Will be filtered out
```

## Performance Considerations

1. **Event Volume**: Be mindful of event frequency and volume, particularly for high-frequency data
2. **Handler Performance**: Keep handlers efficient to avoid bottlenecks
3. **Weak References**: The event bus uses weak references to avoid memory leaks
4. **Async Processing**: Use async mode for high-volume event processing
5. **Event Caching**: Use event caching for frequently reused events
6. **Selective History**: Use selective history to avoid memory issues

## Best Practices

1. **Consistent Event Structure**: Maintain consistent data structures for each event type
2. **Validate Event Data**: Use schema validation for important events
3. **Handle Errors Gracefully**: Implement proper error handling in event handlers
4. **Document Event Types**: Document the purpose and data structure of custom event types
5. **Use Event Groups**: Organize related handlers into groups for better management
6. **Be Cautious with Async**: Async processing adds complexity; use only when needed
7. **Monitor Event Flow**: Log or monitor events during development and testing
8. **Follow Event Hierarchy**: Use appropriate event types for specific domains

## Debugging Tools

The module includes tools for debugging event flow:

```python
from events.event_handlers import LoggingHandler
import logging

# Create a logging handler for all events
logging_handler = LoggingHandler(list(EventType))

# Set specific log levels for different event types
logging_handler.set_event_log_level(EventType.BAR, logging.DEBUG)
logging_handler.set_event_log_level(EventType.SIGNAL, logging.INFO)
logging_handler.set_event_log_level(EventType.ORDER, logging.WARNING)

# Register handler for all event types
for event_type in EventType:
    event_bus.register(event_type, logging_handler)
```

## Event Schema Documentation

Each event type has a documented schema that describes its expected data structure:

### BAR Event
- `timestamp`: Bar timestamp (required)
- `Open`: Opening price (required)
- `High`: High price (required)
- `Low`: Low price (required)
- `Close`: Closing price (required)
- `Volume`: Volume (optional)
- `is_eod`: Whether this is end of day (optional)
- `symbol`: Instrument symbol (optional)

### SIGNAL Event
- `timestamp`: Signal timestamp (required)
- `signal_type`: Signal type (BUY, SELL, NEUTRAL) (required)
- `price`: Price at signal generation (required)
- `rule_id`: ID of rule that generated the signal (optional)
- `confidence`: Confidence score (0-1) (optional)
- `metadata`: Additional signal metadata (optional)
- `symbol`: Instrument symbol (optional)

### ORDER Event
- `timestamp`: Order timestamp (required)
- `symbol`: Instrument symbol (required)
- `order_type`: Order type (MARKET, LIMIT, etc.) (required)
- `quantity`: Order quantity (required)
- `direction`: Order direction (1 for buy, -1 for sell) (required)
- `price`: Limit price (for LIMIT orders) (optional)
- `order_id`: Unique order ID (optional)

### FILL Event
- `timestamp`: Fill timestamp (required)
- `symbol`: Instrument symbol (required)
- `quantity`: Filled quantity (required)
- `price`: Fill price (required)
- `direction`: Direction (1 for buy, -1 for sell) (required)
- `order_id`: Original order ID (optional)
- `transaction_cost`: Transaction cost (optional)
