# Fill Event Handling

## Overview

In an event-driven trading system, fill events (both `FILL` and `PARTIAL_FILL`) need to be properly handled to update positions and portfolio state. This document explains the correct way to handle fill events in the trading system.

## The Fill Event Flow

Fill events represent trade executions and follow this flow:

1. The Execution Engine executes orders and generates fill events
2. Fill events are processed by a specialized `FillHandler`
3. The `FillHandler` updates positions via the Position Manager
4. The Position Manager updates the Portfolio's state

## Using the FillHandler

The system includes a specialized `FillHandler` class specifically designed to process fill events:

```python
from src.events.event_handlers import FillHandler
from src.events.event_types import EventType

# Create a FillHandler that will update the position manager
fill_handler = FillHandler(position_manager)

# Register the handler with the event bus for both types of fill events
event_bus.register(EventType.FILL, fill_handler)
event_bus.register(EventType.PARTIAL_FILL, fill_handler)
```

## Common Mistakes to Avoid

### 1. Registering the wrong component

The most common mistake is trying to register components that don't have the necessary handler methods:

```python
# INCORRECT - Portfolio doesn't have an on_fill method
event_bus.register(EventType.FILL, portfolio.on_fill)

# INCORRECT - ExecutionEngine doesn't have an on_fill method
event_bus.register(EventType.FILL, execution_engine.on_fill)
```

### 2. Using direct updates instead of the event system

Another mistake is bypassing the event system for fills:

```python
# INCORRECT - Direct update bypasses the event system
def execute_order(order):
    # ... execution logic ...
    portfolio.update_position(order.symbol, order.quantity, fill_price)
```

### 3. Missing handler registration

Not registering any handler for fill events means that fills won't update the portfolio:

```python
# INCOMPLETE - Fill events are generated but not handled
event_bus.register(EventType.BAR, lambda event: ...)
event_bus.register(EventType.SIGNAL, position_manager.on_signal)
event_bus.register(EventType.ORDER, execution_engine.on_order)
# Missing fill event handler registration!
```

### 4. Incorrect handler method signatures

Handlers must accept an event parameter and extract data from it:

```python
# INCORRECT - Handler expects data directly
def handle_fill(fill_data):
    # This will fail because it expects data, not an event!
    # ...

# CORRECT - Handler expects an event and extracts data
def handle_fill(event):
    fill_data = event.data
    # Process fill data...
```

## FillHandler Implementation

The `FillHandler` handles fill events by forwarding them to the Position Manager:

```python
class FillHandler(EventHandler):
    """
    Handler for fill events.
    
    This handler processes fill events from execution engine
    and updates portfolio positions accordingly.
    """
    
    def __init__(self, portfolio_manager):
        """
        Initialize fill handler.
        
        Args:
            portfolio_manager: Portfolio manager to update
        """
        super().__init__([EventType.FILL, EventType.PARTIAL_FILL])
        self.portfolio_manager = portfolio_manager
    
    def _process_event(self, event):
        """
        Process a fill event.
        
        Args:
            event: Event object containing fill data
        """
        if hasattr(self.portfolio_manager, 'on_fill'):
            self.portfolio_manager.on_fill(event)
        else:
            logger.warning("Portfolio manager does not have on_fill method")
```

## Position Manager's Fill Handler

The Position Manager should implement an `on_fill` method that extracts data from the event:

```python
def on_fill(self, event):
    """
    Handle fill events.
    
    Args:
        event: Fill event object
    """
    # Extract fill data from the event
    fill_data = event.data
    
    symbol = fill_data.get('symbol')
    direction = fill_data.get('direction')
    quantity = fill_data.get('quantity')
    fill_price = fill_data.get('fill_price')
    timestamp = fill_data.get('timestamp')
    
    # Update the portfolio based on the fill
    if fill_data.get('action') == 'open':
        # Open a new position
        self.portfolio.open_position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp
        )
    elif fill_data.get('action') == 'close':
        # Close an existing position
        position_id = fill_data.get('position_id')
        self.portfolio.close_position(
            position_id=position_id,
            exit_price=fill_price,
            exit_time=timestamp
        )
```

## Setting Up All Event Handlers

In your main script, you should register handlers for all event types:

```python
# Strategy handles bar events and emits signals
event_bus.register(EventType.BAR, strategy.on_bar)

# Position manager handles signal events and emits orders
event_bus.register(EventType.SIGNAL, position_manager.on_signal)

# Execution engine handles order events
event_bus.register(EventType.ORDER, execution_engine.on_order)

# IMPORTANT: Use FillHandler to handle fill events
fill_handler = FillHandler(position_manager)
event_bus.register(EventType.FILL, fill_handler)
event_bus.register(EventType.PARTIAL_FILL, fill_handler)
```

## Event Manager Approach

For even more streamlined setup, use the EventManager to handle all registrations:

```python
# Create event manager
event_manager = EventManager(
    event_bus=event_bus,
    strategy=strategy,
    position_manager=position_manager,
    execution_engine=execution_engine,
    portfolio=portfolio
)

# Initialize the system (registers all handlers automatically)
event_manager.initialize()
```

This approach ensures all the right handlers are connected to the right event types, with proper data flow through the system.

## Best Practices

1. **Use EventHandler Base Classes**: Inherit from EventHandler to ensure proper event processing

2. **Extract Data Properly**: Always extract data from event objects in your handlers

3. **Register All Necessary Handlers**: Ensure handlers are registered for all relevant event types

4. **Use EventManager When Possible**: Let it handle the complexities of event routing

5. **Implement Proper Handler Signatures**: All handlers should accept event objects, not raw data

6. **Check Event Data**: Validate event data before processing to prevent errors

7. **Log Fill Events**: For debugging, set up logging for fill events

8. **Handle Partial Fills**: For live trading, ensure proper handling of partial fills

9. **Process Events in Order**: Maintain the correct event flow: BAR → SIGNAL → ORDER → FILL

10. **Test Event Flow**: Verify the full event flow with simple rules like AlwaysBuyRule