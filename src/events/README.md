# Event System

The Event System provides a robust event-driven architecture for the trading platform, enabling decoupled communication between components through a central event bus.

## Core Components

### Event Types

The system defines standard event types in the `EventType` enumeration:

- **Market Data Events**: `BAR`, `TICK`, `MARKET_OPEN`, `MARKET_CLOSE`
- **Signal Events**: `SIGNAL`
- **Order Events**: `ORDER`, `CANCEL`, `MODIFY`
- **Execution Events**: `FILL`, `PARTIAL_FILL`, `REJECT`
- **Portfolio Events**: `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_MODIFIED`
- **System Events**: `START`, `STOP`, `PAUSE`, `RESUME`, `ERROR`
- **Analysis Events**: `METRIC_CALCULATED`, `ANALYSIS_COMPLETE`

### Event Bus

The `EventBus` acts as a central message broker that:
- Maintains a registry of handlers for event types
- Dispatches events to registered handlers
- Supports both synchronous and asynchronous processing
- Provides event history tracking and filtering

### Event Handlers

The system includes various event handlers:

- **EventHandler (Base Class)**: Abstract base class for all event handlers
- **FunctionEventHandler**: Delegates to a specified callback function
- **LoggingHandler**: Logs events at configurable levels
- **FilterHandler**: Filters events based on criteria
- **DebounceHandler**: Prevents processing events too frequently
- **AsyncEventHandler**: Processes events in a separate thread
- **CompositeHandler**: Delegates to multiple handlers

### Event Emitters

Event emitters generate standardized events:

- **EventEmitter (Base Class)**: Mixin for components that emit events
- **MarketDataEmitter**: Emits bar, tick, and market open/close events
- **SignalEmitter**: Emits trading signals from strategies
- **OrderEmitter**: Emits order, cancel, and modify events
- **FillEmitter**: Emits fill and partial fill events
- **PortfolioEmitter**: Emits position-related events
- **SystemEmitter**: Emits system-related events

### Event Manager

The `EventManager` orchestrates event flow between components:
- Sets up and manages event handlers
- Facilitates proper event transformation between components
- Processes market data and creates appropriate events
- Maintains system state and status metrics

## Event Flow

The typical event flow in the system is:

1. `BAR` events → Strategy → `SIGNAL` events
2. `SIGNAL` events → Position Manager → `ORDER` events
3. `ORDER` events → Execution Engine → `FILL` events
4. `FILL` events → Position Manager / Portfolio

## Setting Up Event Handlers

To connect components through the event system:

```python
# Create event bus
event_bus = EventBus()

# Create event manager
event_manager = EventManager(
    event_bus=event_bus,
    strategy=strategy,
    position_manager=position_manager,
    execution_engine=execution_engine,
    portfolio=portfolio
)

# Initialize the system (registers handlers automatically)
event_manager.initialize()
```

## Processing Market Data

To process market data through the event-driven system:

```python
# Initialize the system
event_manager.initialize()

# Emit market open event
market_open_event = Event(EventType.MARKET_OPEN, {'timestamp': datetime.now()})
event_bus.emit(market_open_event)

# Process each bar of data
for bar in data_handler.iter_train():
    event_manager.process_market_data(bar)
    
# Emit market close event
market_close_event = Event(EventType.MARKET_CLOSE, {'timestamp': datetime.now()})
event_bus.emit(market_close_event)
```

## Important Notes

- **Use EventManager**: Always use the EventManager to set up event handlers rather than manually connecting components.
- **Correct Event Transformation**: The EventManager ensures proper transformation of events between components.
- **Proper Event Flow**: Follow the established event flow pattern to ensure consistent system behavior.
- **Error Handling**: All EventHandlers include built-in error handling to prevent exceptions from disrupting the event loop.

## Best Practices

1. **Use the EventManager**: Let it handle the complexities of event routing and transformation.
2. **Emit Standardized Events**: Use the provided emitter classes to ensure consistent event format.
3. **Handle Events Appropriately**: Process events in ways that maintain the integrity of the event flow.
4. **Add Context to Events**: Include relevant metadata in events to provide context for handlers.
5. **Monitor Event Flow**: Use the LoggingHandler to track events for debugging and analysis.
