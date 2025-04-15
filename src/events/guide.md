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