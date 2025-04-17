# Event System Initialization Guide

This guide provides step-by-step instructions for setting up the enhanced event system with proper object reference preservation. It covers initialization, handler registration, and best practices.

## 1. Basic Event System Setup

```python
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType
from src.events.event_handlers import (
    EventHandler, 
    LoggingHandler, 
    FunctionEventHandler
)

# Create the event bus
event_bus = EventBus(
    async_mode=False,      # Set to True for asynchronous event processing
    validate_events=False  # Set to True to validate events before processing
)

# Create event cache (optional)
from src.events.event_bus import EventCacheManager
event_cache = EventCacheManager(max_cache_size=100)

# Create and register handlers
logger_handler = LoggingHandler(
    event_types=[EventType.BAR, EventType.SIGNAL, EventType.ORDER],
    log_level=logging.INFO
)

# Register handler with the bus
event_bus.register(EventType.BAR, logger_handler)
event_bus.register(EventType.SIGNAL, logger_handler)
event_bus.register(EventType.ORDER, logger_handler)

# Alternative shorthand for multiple registrations
for event_type in logger_handler.event_types:
    event_bus.register(event_type, logger_handler)
```

## 2. Trading System Component Integration

```python
# Create trading system components
from src.strategies.strategy_base import Strategy
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine
from src.events.event_handlers import (
    MarketDataHandler,
    SignalHandler,
    OrderHandler,
    FillHandler
)

# Create components
strategy = Strategy("my_strategy")
position_manager = PositionManager(portfolio)
execution_engine = ExecutionEngine()

# Create specialized handlers
market_data_handler = MarketDataHandler(strategy)
signal_handler = SignalHandler(position_manager)
order_handler = OrderHandler(execution_engine)
fill_handler = FillHandler(position_manager)

# Register with event bus
event_bus.register(EventType.BAR, market_data_handler)
event_bus.register(EventType.SIGNAL, signal_handler)
event_bus.register(EventType.ORDER, order_handler)
event_bus.register(EventType.FILL, fill_handler)
```

## 3. Using Handler Groups

```python
from src.events.event_handlers import EventHandlerGroup

# Create a handler group
system_handlers = EventHandlerGroup(
    name="system_handlers",
    handlers=[logger_handler, market_data_handler, signal_handler]
)

# Register all handlers in group
system_handlers.register_all(event_bus)

# Enable/disable all handlers in group
system_handlers.disable_all()  # Temporarily disable
system_handlers.enable_all()   # Re-enable
```

## 4. Creating Custom Event Handlers

```python
class CustomHandler(EventHandler):
    """Custom handler for specific business logic."""
    
    def __init__(self, event_types):
        """Initialize custom handler."""
        super().__init__(event_types)
        self.processed_events = []
    
    def _process_event(self, event):
        """
        Process an event with custom business logic.
        
        Args:
            event: Original event reference
        """
        # Custom logic here
        if event.event_type == EventType.SIGNAL:
            # Access specific object methods safely
            if hasattr(event.data, 'get_signal_value'):
                signal_value = event.data.get_signal_value()
                print(f"Processing signal with value: {signal_value}")
        
        # Store for tracking
        self.processed_events.append(event)
```

## 5. Function-Based Handlers

```python
# Create a handler using a function
def process_market_data(event):
    """Process market data events."""
    print(f"Got market data: {event.event_type}, timestamp: {event.timestamp}")
    # Access the original object's methods
    if hasattr(event.data, 'get_symbol'):
        symbol = event.data.get_symbol()
        print(f"Symbol: {symbol}")

# Register function handler
market_data_func_handler = FunctionEventHandler(
    event_types=[EventType.BAR, EventType.TICK],
    handler_func=process_market_data
)

event_bus.register(EventType.BAR, market_data_func_handler)
```

## 6. Advanced Handler Configurations

```python
# Debounce handler (limits frequency of event processing)
from src.events.event_handlers import DebounceHandler

debounced_handler = DebounceHandler(
    event_types=[EventType.SIGNAL],
    handler=signal_handler,
    debounce_seconds=0.5  # Minimum time between processing events
)

# Filter handler (only processes events that pass a filter)
from src.events.event_handlers import FilterHandler

def filter_func(event):
    """Only process events for specific symbols."""
    if event.event_type == EventType.BAR:
        symbol = event.data.get_symbol() if hasattr(event.data, 'get_symbol') else event.get('symbol')
        return symbol in ['AAPL', 'MSFT', 'GOOGL']
    return True

filtered_handler = FilterHandler(
    event_types=[EventType.BAR],
    handler=market_data_handler,
    filter_func=filter_func
)

# Asynchronous handler (processes events in a separate thread)
from src.events.event_handlers import AsyncEventHandler

async_handler = AsyncEventHandler(
    event_types=[EventType.SIGNAL],
    handler=signal_handler,
    max_workers=5  # Maximum number of concurrent worker threads
)

# Composite handler (delegates to multiple handlers)
from src.events.event_handlers import CompositeHandler

composite_handler = CompositeHandler(
    event_types=[EventType.SIGNAL],
    handlers=[signal_handler, logger_handler]
)
```

## 7. Object Type Preservation Best Practices

### Emitting Events

```python
# Creating and emitting events with object references
from src.events.signal_event import SignalEvent

# Create a SignalEvent
signal = SignalEvent(
    signal_value=SignalEvent.BUY,
    price=100.0,
    symbol="AAPL",
    rule_id="my_rule",
    metadata={"confidence": 0.8}
)

# Create an event referencing the signal
event = Event(EventType.SIGNAL, signal)

# Emit the event (object reference is preserved)
event_bus.emit(event)
```

### Handling Events

```python
def process_signal_event(event):
    """Process a signal event with proper type checking."""
    # Extract the signal with type checking
    if hasattr(event, 'data') and event.data is not None:
        if hasattr(event.data, 'get_signal_value'):
            # It's a SignalEvent, access its methods
            signal_value = event.data.get_signal_value()
            symbol = event.data.get_symbol()
            price = event.data.get_price()
            
            # Process based on signal value
            if signal_value == SignalEvent.BUY:
                print(f"Buy signal for {symbol} at {price}")
            elif signal_value == SignalEvent.SELL:
                print(f"Sell signal for {symbol} at {price}")
        else:
            # Not a SignalEvent
            print(f"Received non-SignalEvent data: {type(event.data)}")
```

### Direct Object Handling

```python
# Position manager that can handle both Event-wrapped signals and direct SignalEvents
def on_signal(self, event_or_signal):
    """
    Process a signal with proper type checking.
    
    Args:
        event_or_signal: Either Event containing SignalEvent or SignalEvent directly
    """
    signal = None
    
    # Case 1: Argument is an Event with SignalEvent in data
    if isinstance(event_or_signal, Event):
        if hasattr(event_or_signal, 'data') and isinstance(event_or_signal.data, SignalEvent):
            signal = event_or_signal.data
        else:
            print(f"Expected Event with SignalEvent, got {type(event_or_signal.data)}")
            return
    # Case 2: Argument is SignalEvent directly
    elif isinstance(event_or_signal, SignalEvent):
        signal = event_or_signal
    else:
        print(f"Expected Event or SignalEvent, got {type(event_or_signal)}")
        return
    
    # Now we have a SignalEvent, process it
    print(f"Processing signal: {signal.get_symbol()} {signal.get_signal_value()}")
```

## 8. Testing Event Type Preservation

```python
def test_event_type_preservation():
    """Test that object references are preserved through the event system."""
    # Create a signal
    signal = SignalEvent(
        signal_value=SignalEvent.BUY,
        price=100.0,
        symbol="AAPL"
    )
    
    # Store original object ID
    original_id = id(signal)
    
    # Create and emit an event
    event = Event(EventType.SIGNAL, signal)
    event_bus.emit(event)
    
    # Extract signal from handler (assuming handler stores processed signals)
    # This verifies the same object made it through the system
    handler_signal = signal_handler.last_processed_signal
    
    if handler_signal is None:
        print("Signal not processed")
    elif id(handler_signal) == original_id:
        print(f"✓ Signal reference preserved (ID: {original_id})")
    else:
        print(f"✗ Signal reference changed: {original_id} -> {id(handler_signal)}")
```

## 9. Shutdown and Cleanup

```python
# Clean shutdown
def shutdown_event_system():
    """Perform a clean shutdown of the event system."""
    # Stop async processing if used
    if event_bus.async_mode:
        event_bus.stop_dispatch_thread()
    
    # Clear caches
    event_cache.clear_cache()
    
    # Unregister handlers (optional)
    system_handlers.unregister_all(event_bus)
    
    print("Event system shutdown complete")
```

## 10. Common Pitfalls to Avoid

1. **Creating Copies**: Never create copies of objects when passing through the event system
2. **Event Serialization**: Avoid serializing events when using persistence or remote communication
3. **Type Checking**: Always perform proper type checking when accessing specialized methods
4. **Event Factory Methods**: Prefer factory methods that ensure proper object creation
5. **Circular References**: Be careful with circular references between events and handlers


# OUTDATED/OLD GUIDE BELOW:

# Event Handling Guide for Trading System

This guide explains how event handling works in the trading system, focusing on the correct patterns for working with events in different components.

## Core Concepts

- **Event Objects**: Containers for data with event type and payload
- **Event Bus**: Central component that routes events to handlers
- **Event Handlers**: Components that process specific event types
- **Event Emitters**: Components that generate events

## Event Flow

The standard event flow in the system is:

1. **Market Data** → **BAR Events** → **Strategy**
2. **Strategy** → **SIGNAL Events** → **Position Manager**
3. **Position Manager** → **ORDER Events** → **Execution Engine**
4. **Execution Engine** → **FILL Events** → **Portfolio**

## Working with Events

### Proper Event Handling Pattern

When developing components that handle events:

```python
def on_bar(self, event):
    """
    Process a bar event.
    
    Args:
        event: Event object containing bar data
    """
    # Extract data from the event
    bar_data = event.data
    
    # Now process the data
    # ...
    
    # Optionally emit new events
    # ...
```

### Rule Components

Rules are a special case that can work with both events and raw data:

```python
# Rules implement generate_signal() which works with raw data
def generate_signal(self, data):
    """
    Generate a signal from bar data.
    
    Args:
        data: Dictionary containing bar data
    """
    # Process data and generate signal
    # ...
    return signal

# The Rule.on_bar() method handles event extraction automatically
def on_bar(self, event_or_data):
    """
    Process a bar event or raw data.
    
    Args:
        event_or_data: Event object or dictionary
    """
    # Extract data if it's an Event object
    if hasattr(event_or_data, 'data'):
        data = event_or_data.data
    else:
        data = event_or_data
        
    # Call the rule's generate_signal method
    return self.generate_signal(data)
```

### Emitting Events

When emitting events:

```python
from src.events import Event, EventType

# Create an event with data
event = Event(EventType.SIGNAL, {
    'timestamp': datetime.now(),
    'signal_type': 'BUY',
    'price': 150.75
})

# Emit the event
event_bus.emit(event)
```

## Component Responsibilities

### Strategies

Strategies receive BAR events and emit SIGNAL events:

```python
def on_bar(self, event):
    bar_data = event.data
    
    # Process bar data with rules
    for rule in self.rules:
        signal = rule.on_bar(event)  # Pass the event object
        if signal and signal.signal_type != SignalType.NEUTRAL:
            # Create and emit a signal event
            signal_event = Event(EventType.SIGNAL, signal)
            self.event_bus.emit(signal_event)
```

### Position Manager

Position Manager receives SIGNAL events and emits ORDER events:

```python
def on_signal(self, event):
    signal = event.data
    
    # Process signal and determine position action
    # ...
    
    # Create order if needed
    if action == 'entry':
        order_data = {
            'symbol': signal.symbol,
            'direction': direction,
            'quantity': size,
            'price': signal.price
        }
        
        # Emit order event
        order_event = Event(EventType.ORDER, order_data)
        self.event_bus.emit(order_event)
```

### Execution Engine

Execution Engine receives ORDER events and emits FILL events:

```python
def on_order(self, event):
    order_data = event.data
    
    # Process order
    # ...
    
    # Create fill data
    fill_data = {
        'symbol': order_data['symbol'],
        'direction': order_data['direction'],
        'quantity': order_data['quantity'],
        'fill_price': execution_price,
        'timestamp': datetime.now()
    }
    
    # Emit fill event
    fill_event = Event(EventType.FILL, fill_data)
    self.event_bus.emit(fill_event)
```

## Event Registration

Components register for events using the Event Bus:

```python
# Register strategy for BAR events
event_bus.register(EventType.BAR, strategy.on_bar)

# Register position manager for SIGNAL events
event_bus.register(EventType.SIGNAL, position_manager.on_signal)

# Register execution engine for ORDER events
event_bus.register(EventType.ORDER, execution_engine.on_order)

# Use specialized FillHandler for FILL events
fill_handler = FillHandler(position_manager)
event_bus.register(EventType.FILL, fill_handler)
```

## Testing with Events

When testing components that handle events:

```python
# Create test data
bar_data = {
    'timestamp': datetime.now(),
    'Open': 100.0,
    'High': 101.0,
    'Low': 99.0,
    'Close': 100.5,
    'Volume': 1000
}

# Create event object
bar_event = Event(EventType.BAR, bar_data)

# Test component with event
result = component.on_bar(bar_event)  # Pass event, not just data
```

## Common Mistakes to Avoid

1. **Passing raw data instead of events**: Always pass Event objects to handlers, not just the data

2. **Not extracting data from events**: Remember to extract data from the event with `event.data`

3. **Missing event handlers**: Ensure handlers are registered for all relevant event types

4. **Incorrect handler signatures**: Handlers must accept an event parameter and extract data from it

5. **Bypassing the event system**: Always use the event system for component communication

## Best Practices

1. **Extract data in handlers**: Always extract data from events using `event.data`

2. **Rules implement generate_signal**: Rules should implement the `generate_signal(data)` method

3. **Use the Rule.on_bar method**: This method handles both event objects and raw data

4. **Test with Event objects**: When testing, pass proper Event objects to handlers

5. **Use EventManager**: Let it handle the complexities of event routing