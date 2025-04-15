# Event System Documentation

## Overview

The event system provides a robust event-driven architecture for the backtesting system. It enables decoupled communication between components through a central event bus, ensuring flexibility and maintainability in the trading system architecture.

## Core Components

### Event Types

The system defines standard event types in `EventType` enumeration, including:

- **Market Data Events**: `BAR`, `TICK`, `MARKET_OPEN`, `MARKET_CLOSE`
- **Signal Events**: `SIGNAL`
- **Order Events**: `ORDER`, `CANCEL`, `MODIFY`
- **Execution Events**: `FILL`, `PARTIAL_FILL`, `REJECT`
- **Portfolio Events**: `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_MODIFIED`
- **System Events**: `START`, `STOP`, `PAUSE`, `RESUME`, `ERROR`
- **Analysis Events**: `METRIC_CALCULATED`, `ANALYSIS_COMPLETE`

### Event Handlers

Event handlers process specific types of events. The system includes specialized handlers:

- **MarketDataHandler**: Processes market data events
- **SignalHandler**: Processes signal events
- **OrderHandler**: Processes order events
- **FillHandler**: Processes fill events to update the portfolio

## Event Flow

The typical event flow in the system is:

1. `BAR` events → Strategy → `SIGNAL` events
2. `SIGNAL` events → Position Manager → `ORDER` events
3. `ORDER` events → Execution Engine → `FILL` events
4. `FILL` events → Fill Handler → Position Manager

## Setting Up Event Handlers

To properly connect components through the event system, you need to register the appropriate handlers for each event type:

```python
# Strategy handles bar events and emits signals
event_bus.register(EventType.BAR, lambda event: event_bus.emit(
    Event(EventType.SIGNAL, strategy.on_bar(BarEvent(event.data)))
))

# Position manager handles signal events and emits orders
event_bus.register(EventType.SIGNAL, position_manager.on_signal)

# Execution engine handles order events
event_bus.register(EventType.ORDER, execution_engine.on_order)

# Create and register a FillHandler to handle fill events
from events.event_handlers import FillHandler
fill_handler = FillHandler(position_manager)
event_bus.register(EventType.FILL, fill_handler)
event_bus.register(EventType.PARTIAL_FILL, fill_handler)
```

## Important Notes

- **Do not register the Portfolio directly**: The Portfolio class doesn't have event handler methods. Instead, use the Position Manager through the FillHandler.
- **Use specialized handlers**: For each event type, use the appropriate specialized handler class rather than attempting to register components directly.
- **Component relationships**: The Position Manager updates the Portfolio based on fill information, so the FillHandler should be connected to the Position Manager, not directly to the Portfolio.

## Common Pitfalls

1. **Incorrect handler registration**: Registering components that don't implement the necessary handler methods (e.g., trying to register Portfolio or ExecutionEngine for FILL events)
   
   ```python
   # INCORRECT
   event_bus.register(EventType.FILL, portfolio.on_fill)  # Portfolio has no on_fill method
   event_bus.register(EventType.FILL, execution_engine.on_fill)  # ExecutionEngine has no on_fill method
   
   # CORRECT
   fill_handler = FillHandler(position_manager)
   event_bus.register(EventType.FILL, fill_handler)
   ```

2. **Missing handler registration**: Not registering any handler for an important event type

3. **Wrong component relationships**: Using the wrong components to handle certain event types

## Event Handler Implementation

If you need to create a custom event handler, you should extend the EventHandler base class:

```python
from events.event_handlers import EventHandler
from events.event_types import EventType

class MyCustomHandler(EventHandler):
    def __init__(self, target_component):
        super().__init__([EventType.CUSTOM])  # Register for CUSTOM events
        self.target_component = target_component
    
    def _process_event(self, event):
        # Process the event and update the target component
        self.target_component.process_custom_event(event.data)
```
