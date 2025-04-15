# Event-Driven Architecture in the Trading System

This document provides a comprehensive overview of the event-driven architecture implemented in our trading system, covering the core components, interaction patterns, and best practices.

## Overview

The event-driven architecture allows components to communicate without direct dependencies, promoting loose coupling and maintainability. Key benefits include:

- **Decoupling**: Components interact only through events, not direct method calls
- **Extensibility**: Easy to add new components without modifying existing ones
- **Testability**: Components can be tested in isolation by simulating events
- **Flexibility**: Event flow can be reconfigured without changing component logic

## Core Components

### Event

The `Event` class represents a discrete notification or action within the system:

```python
class Event:
    def __init__(self, event_type, data=None, timestamp=None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.id = str(uuid.uuid4())
```

Events contain:
- **Type**: The category of the event (e.g., BAR, SIGNAL, ORDER)
- **Data**: The payload associated with the event
- **Timestamp**: When the event occurred
- **ID**: Unique identifier for tracking and debugging

### EventBus

The `EventBus` is the central messaging system that routes events from producers to consumers:

```python
class EventBus:
    def __init__(self, async_mode=False):
        self.handlers = {event_type: [] for event_type in EventType}
        self.async_mode = async_mode
        self.history = []
        # ...
        
    def register(self, event_type, handler):
        # Register handler for event type...
        
    def emit(self, event):
        # Deliver event to registered handlers...
```

The EventBus provides:
- **Registration**: Components register to receive specific event types
- **Emission**: Components emit events to be delivered to interested handlers
- **History**: Records events for analysis and debugging
- **Async Processing**: Optional asynchronous event processing

### EventType

The `EventType` enumeration defines standardized event categories:

```python
class EventType(Enum):
    # Market data events
    BAR = auto()
    TICK = auto()
    MARKET_OPEN = auto()
    MARKET_CLOSE = auto()
    
    # Signal events
    SIGNAL = auto()
    
    # Order events
    ORDER = auto()
    CANCEL = auto()
    MODIFY = auto()
    
    # And many more...
```

These standardized types ensure consistent communication across components.

### EventHandler

The `EventHandler` is the base class for components that process events:

```python
class EventHandler(ABC):
    def __init__(self, event_types):
        self.event_types = set(event_types)
        self.enabled = True
        
    def handle(self, event):
        # Process the event...
        
    @abstractmethod
    def _process_event(self, event):
        # Implemented by subclasses
        pass
```

Handlers provide:
- **Type Filtering**: Only process relevant event types
- **Error Handling**: Protect against exceptions in event processing
- **Enable/Disable**: Can be temporarily disabled when needed

### EventEmitter

The `EventEmitter` is the base class for components that generate events:

```python
class EventEmitter:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        
    def emit(self, event_type, data=None):
        # Create and emit an event...
        
    def emit_event(self, event):
        # Emit an existing event...
```

Emitters provide:
- **Event Creation**: Convenience methods to create events
- **Event Dispatch**: Send events to the event bus for distribution

## Event Flow

The typical event flow in the system follows this pattern:

1. **Components register interest** in specific event types:
   ```python
   event_bus.register(EventType.BAR, strategy_handler)
   ```

2. **Events are emitted** by components when something notable occurs:
   ```python
   market_data_emitter.emit_bar(bar_data)
   ```

3. **The event bus routes events** to registered handlers:
   ```python
   # Inside EventBus._dispatch_event
   for handler_ref in self.handlers[event.event_type]:
       handler = handler_ref()
       if handler is not None:
           handler.handle(event)
   ```

4. **Handlers process events** and may emit new events in response:
   ```python
   # Inside Strategy._process_event
   if event.event_type == EventType.BAR:
       signal = self.generate_signal(event.data)
       if signal:
           self.signal_emitter.emit_signal(**signal)
   ```

This creates a chain of event propagation through the system.

## Specialized Handlers and Emitters

### Handlers

The system includes several specialized event handlers:

#### LoggingHandler

Logs events at configurable levels:

```python
logging_handler = LoggingHandler([EventType.BAR, EventType.SIGNAL])
logging_handler.set_event_log_level(EventType.BAR, logging.DEBUG)
logging_handler.set_event_log_level(EventType.SIGNAL, logging.INFO)
```

#### FunctionEventHandler

Delegates processing to a function:

```python
def process_signal(event):
    # Process signal event...
    
signal_handler = FunctionEventHandler(EventType.SIGNAL, process_signal)
```

#### FilterHandler

Only processes events that meet specific criteria:

```python
def high_confidence_filter(event):
    return event.data.get('confidence', 0) >= 0.7
    
filtered_handler = FilterHandler([EventType.SIGNAL], base_handler, high_confidence_filter)
```

#### DebounceHandler

Prevents processing events too frequently:

```python
debounced_handler = DebounceHandler([EventType.TICK], base_handler, 0.1)  # 100ms minimum
```

#### AsyncEventHandler

Processes events in a separate thread:

```python
async_handler = AsyncEventHandler([EventType.BAR], base_handler, max_workers=5)
```

#### CompositeHandler

Delegates to multiple handlers:

```python
composite = CompositeHandler([EventType.SIGNAL], [handler1, handler2, handler3])
```

### Emitters

The system includes several specialized event emitters:

#### MarketDataEmitter

Emits market data events:

```python
market_data_emitter = MarketDataEmitter(event_bus)
market_data_emitter.emit_bar(bar_data)
market_data_emitter.emit_market_open()
```

#### SignalEmitter

Emits trading signals:

```python
signal_emitter = SignalEmitter(event_bus)
signal_emitter.emit_signal(
    symbol="AAPL",
    signal_type="BUY",
    price=151.0,
    confidence=0.8,
    rule_id="sma_crossover"
)
```

#### OrderEmitter

Emits order-related events:

```python
order_emitter = OrderEmitter(event_bus)
order_emitter.emit_order(
    symbol="AAPL",
    order_type="MARKET",
    quantity=100,
    direction=1,
    price=151.0
)
```

## Event Groups and Hierarchies

Events can be organized into logical groups for better management:

### EventHandlerGroup

Manages collections of related handlers:

```python
# Create a group of related handlers
trading_handlers = EventHandlerGroup("trading_handlers", [
    bar_handler,
    signal_handler,
    order_handler,
    fill_handler
])

# Register all handlers at once
trading_handlers.register_all(event_bus)

# Enable/disable as a group
trading_handlers.disable_all()
```

## Event Schemas

The system includes schema validation for event data:

```python
from src.events.schema import validate_event_data

# Validate bar data
bar_data = {
    'timestamp': datetime.now(),
    'Open': 100.5,
    'High': 101.2,
    'Low': 99.8,
    'Close': 100.9,
    'Volume': 5000,
    'symbol': 'AAPL'
}

validated_data = validate_event_data('BAR', bar_data)
```

Schemas ensure data consistency across the system.

## Best Practices

### 1. Use Appropriate Event Types

Choose the most specific event type for your needs:

```python
# GOOD: Specific event type
event_bus.emit(EventType.MARKET_OPEN, data)

# BAD: Generic type that requires checking data
event_bus.emit(EventType.CUSTOM, {"type": "market_open", "data": data})
```

### 2. Keep Event Data Clean and Minimal

Include only what's necessary in event data:

```python
# GOOD: Clean, focused data
event_data = {
    'symbol': 'AAPL',
    'price': 150.75,
    'timestamp': datetime.now()
}

# BAD: Including unnecessary data
event_data = {
    'symbol': 'AAPL',
    'price': 150.75,
    'timestamp': datetime.now(),
    'strategy_instance': strategy,  # Objects that could create circular references
    'full_history': [...],  # Excessive data
    'internal_state': {...}  # Implementation details
}
```

### 3. Handle Events Efficiently

Minimize processing in event handlers:

```python
# GOOD: Quick processing, defer heavy work
def handle_bar(event):
    # Quick check if we're interested
    if event.data['symbol'] not in watch_list:
        return
        
    # Queue heavy processing for later
    analysis_queue.put(event.data)

# BAD: Heavy processing in handler
def handle_bar(event):
    # Long-running analysis in handler
    result = perform_complex_analysis(event.data)
    update_database(result)
    generate_report(result)
```

### 4. Use Strong Typing for Event Data

Validate event data to catch errors early:

```python
# GOOD: Validate against a schema
from src.events.schema import validate_event_data
validated = validate_event_data('SIGNAL', signal_data)

# BAD: Assume data structure
def process_signal(signal_data):
    # May raise KeyError or TypeError
    direction = 1 if signal_data['signal_type'] == 'BUY' else -1
    quantity = signal_data['quantity'] * 2
```

### 5. Implement Error Handling

Protect the event loop from handler exceptions:

```python
# GOOD: Catch exceptions in handler
def _process_event(self, event):
    try:
        result = self.strategy.process_bar(event.data)
        if result:
            self.emit_signal(result)
    except Exception as e:
        logger.error(f"Error processing bar: {e}", exc_info=True)
        # Continue operation despite error

# BAD: Unhandled exceptions
def _process_event(self, event):
    # Will crash if strategy.process_bar raises an exception
    result = self.strategy.process_bar(event.data)
    if result:
        self.emit_signal(result)
```

## Example Component Integration

### Strategy Component

```python
class Strategy(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        self.signal_emitter = SignalEmitter(event_bus)
        
    def on_bar(self, bar_data):
        # Process bar data
        signal = self.generate_signal(bar_data)
        
        # Emit signal if generated
        if signal:
            self.signal_emitter.emit_signal(**signal)
            
    def generate_signal(self, bar_data):
        # Implementation specific to strategy
        # Return signal parameters or None
        pass
```

### Portfolio Manager Component

```python
class PortfolioManager:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.order_emitter = OrderEmitter(event_bus)
        
    def on_signal(self, signal_data):
        # Convert signal to order
        order = self.create_order(signal_data)
        
        # Emit order
        self.order_emitter.emit_order(**order)
        
    def create_order(self, signal_data):
        # Implementation specific to portfolio management
        # Return order parameters
        pass
```

## Conclusion

The event-driven architecture is a powerful pattern for building flexible, maintainable trading systems. By following the principles and best practices outlined in this document, you can create robust components that communicate effectively while remaining loosely coupled.