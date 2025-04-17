# Events Module

Event System Package

This package provides the event-driven infrastructure for the trading system.
It includes event definitions, the event bus, handlers, and utilities for event-based communication.

## Contents

- [event_base](#event_base)
- [event_bus](#event_bus)
- [event_emitters](#event_emitters)
- [event_handlers](#event_handlers)
- [event_manager](#event_manager)
- [event_schema](#event_schema)
- [event_types](#event_types)
- [event_utils](#event_utils)
- [portfolio_events](#portfolio_events)
- [signal_event](#signal_event)

## event_base

Event Base Module

This module provides the base Event class used throughout the event system.

### Classes

#### `Event`

Base class for all events in the trading system.

Events contain a type, timestamp, unique ID, and data payload.

##### Methods

###### `__init__(event_type, data=None, timestamp=None)`

Initialize an event.

Args:
    event_type: Type of the event
    data: Optional data payload
    timestamp: Event timestamp (defaults to current time)

###### `get(key, default=None)`

Get a value from the event data.

Args:
    key: Dictionary key to retrieve
    default: Default value if key is not found
    
Returns:
    Value for the key or default

*Returns:* Value for the key or default

###### `__str__()`

*Returns:* `str`

String representation of the event.

## event_bus

Event Bus Module

This module provides the event bus infrastructure for the trading system.
It includes the Event class, EventBus class, and related utilities for
event-driven communication between system components.

### Functions

#### `_update_metrics(event_type, processing_time=None)`

Update event metrics.

### Classes

#### `EventBus`

Central event bus for routing events between system components.

The event bus maintains a registry of handlers for different event types
and dispatches events to the appropriate handlers when they are emitted.

##### Methods

###### `__init__(async_mode=False)`

Initialize event bus.

Args:
    async_mode: Whether to dispatch events asynchronously

###### `register(event_type, handler)`

*Returns:* `None`

Register a handler for an event type.

Args:
    event_type: Event type to register for
    handler: Handler to register

###### `unregister(event_type, handler)`

*Returns:* `bool`

Unregister a handler for an event type.

Args:
    event_type: Event type to unregister from
    handler: Handler to unregister
    
Returns:
    True if handler was unregistered, False if not found

###### `emit(event)`

Emit an event to registered handlers.

###### `emit_all(events)`

*Returns:* `None`

Emit multiple events to all registered handlers.

Args:
    events: List of events to emit

###### `_dispatch_event(event)`

*Returns:* `None`

Dispatch an event to all registered handlers.

Args:
    event: Event to dispatch

###### `_add_to_history(event)`

*Returns:* `None`

Add an event to the history.

Args:
    event: Event to add

###### `clear_history()`

*Returns:* `None`

Clear the event history.

###### `get_history(event_type=None, start_time=None, end_time=None)`

*Returns:* `List[Event]`

Get event history, optionally filtered.

Args:
    event_type: Optional event type to filter by
    start_time: Optional start time for filtering
    end_time: Optional end time for filtering
    
Returns:
    List of events matching the filters

###### `start_dispatch_thread()`

*Returns:* `None`

Start the async dispatch thread.

###### `stop_dispatch_thread()`

*Returns:* `None`

Stop the async dispatch thread.

###### `_dispatch_loop()`

*Returns:* `None`

Main loop for async event dispatching.

#### `EventCacheManager`

Manager for caching events for improved performance.

This class maintains caches of events by type to prevent
regenerating the same events repeatedly.

##### Methods

###### `__init__(max_cache_size=100)`

Initialize event cache manager.

Args:
    max_cache_size: Maximum size of each event type cache

###### `get_cached_event(event_type, key)`

*Returns:* `Optional[Event]`

Get cached event if available.

Args:
    event_type: Type of event
    key: Cache key for the event
    
Returns:
    Cached event or None if not found

###### `cache_event(event, key)`

*Returns:* `None`

Cache an event.

Args:
    event: Event to cache
    key: Cache key for the event

###### `clear_cache(event_type=None)`

*Returns:* `None`

Clear event cache.

Args:
    event_type: Optional event type to clear (None for all)

## event_emitters

Event Emitters Module

This module defines event emitters for generating events in the trading system.
It provides the EventEmitter mixin class and specialized emitter implementations.

### Classes

#### `EventEmitter`

Mixin class for components that emit events.

This class provides a standard interface for emitting events
to the event bus. It can be mixed into any class that needs
to generate events.

##### Methods

###### `__init__(event_bus)`

Initialize event emitter.

Args:
    event_bus: Event bus to emit events to

###### `emit(event_type, event_object)`

*Returns:* `Event`

Create and emit an event.

Args:
    event_type: Type of event to emit
    event_object: Event object to emit
    
Returns:
    The emitted event

###### `emit_event(event)`

*Returns:* `None`

Emit an existing event.

Args:
    event: Event to emit

#### `MarketDataEmitter`

Event emitter for market data.

This class emits bar, tick, and market open/close events.

Attributes:
    default_symbol (str, optional): Default symbol to use when bar data doesn't include one

##### Methods

###### `__init__(event_bus, default_symbol=None)`

Initialize the MarketDataEmitter.

Args:
    event_bus (EventBus): Event bus to emit events to
    default_symbol (str, optional): Default symbol to use if not present in bar data

###### `emit_bar(bar_event)`

*Returns:* `Event`

Emit a bar event.

Args:
    bar_event: BarEvent object to emit
    
Returns:
    The emitted event

###### `emit_tick(tick_event)`

*Returns:* `Event`

Emit a tick event.

Args:
    tick_event: TickEvent object to emit
    
Returns:
    The emitted event

###### `emit_market_open(market_open_event)`

*Returns:* `Event`

Emit a market open event.

Args:
    market_open_event: MarketOpenEvent object to emit
    
Returns:
    The emitted event

###### `emit_market_close(market_close_event)`

*Returns:* `Event`

Emit a market close event.

Args:
    market_close_event: MarketCloseEvent object to emit
    
Returns:
    The emitted event

#### `SignalEmitter`

Event emitter for trading signals.

This class emits signal events from strategies.

##### Methods

###### `emit_signal(signal_event)`

*Returns:* `Event`

Emit a signal event.

Args:
    signal_event: SignalEvent object to emit
    
Returns:
    The emitted event

#### `OrderEmitter`

Event emitter for order-related events.

This class emits order, cancel, and modify events.

##### Methods

###### `emit_order(order_event)`

*Returns:* `Event`

Emit an order event.

Args:
    order_event: OrderEvent object to emit
    
Returns:
    The emitted event

###### `emit_cancel(cancel_event)`

*Returns:* `Event`

Emit a cancel order event.

Args:
    cancel_event: CancelOrderEvent object to emit
    
Returns:
    The emitted event

###### `emit_modify(modify_event)`

*Returns:* `Event`

Emit a modify order event.

Args:
    modify_event: ModifyOrderEvent object to emit
    
Returns:
    The emitted event

#### `FillEmitter`

Event emitter for fill-related events.

This class emits fill and partial fill events.

##### Methods

###### `emit_fill(fill_event)`

*Returns:* `Event`

Emit a fill event.

Args:
    fill_event: FillEvent object to emit
    
Returns:
    The emitted event

###### `emit_partial_fill(partial_fill_event)`

*Returns:* `Event`

Emit a partial fill event.

Args:
    partial_fill_event: PartialFillEvent object to emit
    
Returns:
    The emitted event

###### `emit_reject(reject_event)`

*Returns:* `Event`

Emit an order rejection event.

Args:
    reject_event: RejectEvent object to emit
    
Returns:
    The emitted event

#### `PortfolioEmitter`

Event emitter for portfolio-related events.

This class emits position opened, closed, and modified events.

##### Methods

###### `emit_position_opened(position_opened_event)`

*Returns:* `Event`

Emit a position opened event.

Args:
    position_opened_event: PositionOpenedEvent object to emit
    
Returns:
    The emitted event

###### `emit_position_closed(position_closed_event)`

*Returns:* `Event`

Emit a position closed event.

Args:
    position_closed_event: PositionClosedEvent object to emit
    
Returns:
    The emitted event

###### `emit_position_modified(position_modified_event)`

*Returns:* `Event`

Emit a position modified event.

Args:
    position_modified_event: PositionModifiedEvent object to emit
    
Returns:
    The emitted event

#### `SystemEmitter`

Event emitter for system-related events.

This class emits system start, stop, pause, resume, and error events.

##### Methods

###### `emit_start(start_event)`

*Returns:* `Event`

Emit a system start event.

Args:
    start_event: StartEvent object to emit
    
Returns:
    The emitted event

###### `emit_stop(stop_event)`

*Returns:* `Event`

Emit a system stop event.

Args:
    stop_event: StopEvent object to emit
    
Returns:
    The emitted event

###### `emit_error(error_event)`

*Returns:* `Event`

Emit a system error event.

Args:
    error_event: ErrorEvent object to emit
    
Returns:
    The emitted event

## event_handlers

Event Handlers Module

This module defines handlers for processing events in the trading system.
It includes base handler classes and specialized implementations for event handling.

### Classes

#### `EventHandler`

Base class for all event handlers.

Event handlers process specific types of events and can be registered
with the event bus to receive events of those types.

##### Methods

###### `__init__(event_types)`

Initialize event handler.

Args:
    event_types: Event type(s) this handler processes

###### `can_handle(event_type)`

*Returns:* `bool`

Check if this handler can process events of the given type.

Args:
    event_type: Event type to check
    
Returns:
    True if this handler can process the event type, False otherwise

###### `handle(event)`

*Returns:* `None`

Process an event.

Args:
    event: Event to process

###### `_process_event(event)`

*Returns:* `None`

Internal method to process an event.

This method must be implemented by subclasses.

Args:
    event: Event to process

###### `enable()`

*Returns:* `None`

Enable the handler.

###### `disable()`

*Returns:* `None`

Disable the handler.

###### `__str__()`

*Returns:* `str`

String representation of the handler.

#### `FunctionEventHandler`

Event handler that delegates processing to a function.

This handler calls a specified function to process events.
It's useful for creating handlers on-the-fly without
defining a new handler class.

##### Methods

###### `__init__(event_types, handler_func)`

Initialize function-based event handler.

Args:
    event_types: Event type(s) this handler processes
    handler_func: Function to call for event processing

###### `_process_event(event)`

*Returns:* `None`

Process an event by calling the handler function.

Args:
    event: Event to process

#### `LoggingHandler`

Event handler that logs events.

This handler logs events at specified logging levels
based on event type.

##### Methods

###### `__init__(event_types, log_level)`

Initialize logging handler.

Args:
    event_types: Event type(s) to log
    log_level: Default logging level

###### `set_event_log_level(event_type, log_level)`

*Returns:* `None`

Set logging level for a specific event type.

Args:
    event_type: Event type to set level for
    log_level: Logging level to use

###### `_process_event(event)`

*Returns:* `None`

Log an event at the appropriate level.

Args:
    event: Event to log

#### `DebounceHandler`

Event handler that debounces events before processing.

This handler prevents processing the same event type too frequently
by enforcing a minimum time between processing events of the same type.

##### Methods

###### `__init__(event_types, handler, debounce_seconds=0.1)`

Initialize debounce handler.

Args:
    event_types: Event type(s) this handler processes
    handler: Handler to delegate to after debouncing
    debounce_seconds: Minimum seconds between events of same type

###### `_process_event(event)`

*Returns:* `None`

Process an event with debouncing.

Args:
    event: Event to process

#### `FilterHandler`

Event handler that filters events based on criteria.

This handler only processes events that pass its filter criteria.
It's useful for filtering out events that don't meet certain conditions.

##### Methods

###### `__init__(event_types, handler, filter_func)`

Initialize filter handler.

Args:
    event_types: Event type(s) this handler processes
    handler: Handler to delegate to if event passes filter
    filter_func: Function that returns True if event should be processed

###### `_process_event(event)`

*Returns:* `None`

Process an event if it passes the filter.

Args:
    event: Event to process

#### `AsyncEventHandler`

Event handler that processes events asynchronously.

This handler processes events in a separate thread to avoid
blocking the event bus. It's useful for handlers that may take
significant time to process an event.

##### Methods

###### `__init__(event_types, handler, max_workers=5)`

Initialize async event handler.

Args:
    event_types: Event type(s) this handler processes
    handler: Handler to delegate to asynchronously
    max_workers: Maximum number of worker threads

###### `_process_event(event)`

*Returns:* `None`

Process an event asynchronously.

Args:
    event: Event to process

###### `_worker(event)`

*Returns:* `None`

Worker thread that processes an event and updates active worker count.

Args:
    event: Event to process

#### `CompositeHandler`

Event handler that delegates to multiple handlers.

This handler allows multiple handlers to process the same event.
It's useful for implementing multiple independent reactions to an event.

##### Methods

###### `__init__(event_types, handlers)`

Initialize composite handler.

Args:
    event_types: Event type(s) this handler processes
    handlers: List of handlers to delegate to

###### `_process_event(event)`

*Returns:* `None`

Process an event by delegating to all handlers.

Args:
    event: Event to process

#### `EventHandlerGroup`

Group of event handlers that can be managed together.

This class is useful for organizing related handlers and
enabling/disabling them as a group.

##### Methods

###### `__init__(name, handlers=None)`

Initialize event handler group.

Args:
    name: Name for the group
    handlers: Optional list of handlers to add initially

###### `add_handler(handler)`

*Returns:* `None`

Add a handler to the group.

Args:
    handler: Event handler to add

###### `remove_handler(handler)`

*Returns:* `bool`

Remove a handler from the group.

Args:
    handler: Event handler to remove
    
Returns:
    True if handler was removed, False if not found

###### `enable_all()`

*Returns:* `None`

Enable all handlers in the group.

###### `disable_all()`

*Returns:* `None`

Disable all handlers in the group.

###### `get_handlers()`

*Returns:* `List[EventHandler]`

Get all handlers in the group.

Returns:
    List of event handlers

###### `register_all(event_bus)`

*Returns:* `None`

Register all handlers in the group with an event bus.

Args:
    event_bus: Event bus to register handlers with

###### `unregister_all(event_bus)`

*Returns:* `None`

Unregister all handlers in the group from an event bus.

Args:
    event_bus: Event bus to unregister handlers from

#### `MarketDataHandler`

Handler for market data events.

This handler processes bar and tick events, typically
forwarding them to strategies for signal generation.

##### Methods

###### `__init__(strategy)`

Initialize market data handler.

Args:
    strategy: Strategy to forward data to

###### `_process_event(event)`

*Returns:* `None`

Process a market data event.

Args:
    event: Event to process

#### `SignalHandler`

Handler for signal events.

This handler processes signal events from strategies
and forwards them to portfolio manager for order generation.

##### Methods

###### `__init__(portfolio_manager)`

Initialize signal handler.

Args:
    portfolio_manager: Portfolio manager to forward signals to

###### `_process_event(event)`

*Returns:* `None`

Process a signal event.

Args:
    event: Event to process

#### `OrderHandler`

Handler for order events.

This handler processes order, cancel, and modify events
and forwards them to execution engine.

##### Methods

###### `__init__(execution_engine)`

Initialize order handler.

Args:
    execution_engine: Execution engine to forward orders to

###### `_process_event(event)`

*Returns:* `None`

Process an order event.

Args:
    event: Event to process

#### `FillHandler`

Handler for fill events.

This handler processes fill events from execution engine
and updates portfolio positions accordingly.

##### Methods

###### `__init__(portfolio_manager)`

Initialize fill handler.

Args:
    portfolio_manager: Portfolio manager to update

###### `_process_event(event)`

*Returns:* `None`

Process a fill event.

Args:
    event: Event to process

#### `EventProcessor`

Interface for components that process events.

All components in the system that need to handle events should implement
this interface or inherit from a class that implements it.

##### Methods

###### `on_event(event)`

Process any event based on its type.

Args:
    event: Event to process
    
Returns:
    Result of event processing (varies by event type)

*Returns:* Result of event processing (varies by event type)

###### `on_bar(event)`

Process a bar event (market data).

Args:
    event: Event with a BarEvent data payload
    
Returns:
    Depends on the component

*Returns:* Depends on the component

###### `on_signal(event)`

Process a signal event.

Args:
    event: Event with a Signal data payload
    
Returns:
    Depends on the component

*Returns:* Depends on the component

###### `on_order(event)`

Process an order event.

Args:
    event: Event with an Order data payload
    
Returns:
    Depends on the component

*Returns:* Depends on the component

###### `on_fill(event)`

Process a fill event.

Args:
    event: Event with a Fill data payload
    
Returns:
    Depends on the component

*Returns:* Depends on the component

## event_manager

### Classes

#### `EventManager`

Central manager for event flow between components in the trading system.
Ensures proper event handling and data transformation between components.

##### Methods

###### `__init__(event_bus, strategy, position_manager, execution_engine, portfolio=None)`

Initialize the events manager with all required components.

Args:
    event_bus: The central event bus for communication
    strategy: The trading strategy to use
    position_manager: The position manager
    execution_engine: The execution engine
    portfolio: Optional portfolio instance

###### `initialize()`

Register all event handlers and initialize components.

###### `_register_event_handlers()`

Register all necessary event handlers.

###### `_create_bar_handler()`

Create a handler for BAR events.

###### `_create_signal_handler()`

Create a handler for SIGNAL events.

###### `_create_order_handler()`

Create a handler for ORDER events.

###### `_create_fill_handler()`

Create a handler for FILL events.

###### `process_market_data(bar_data)`

Process a single bar of market data.

Args:
    bar_data: Dictionary or BarEvent containing market data

###### `get_status()`

Get the current status of the trading system.

Returns:
    dict: Status information

*Returns:* dict: Status information

###### `reset()`

Reset the events manager state.

## event_schema

Event Data Schema Documentation

This module defines the data schemas for different event types in the system.
It provides type definitions, validation utilities, and schema documentation
to ensure consistency across the event-driven architecture.

### Functions

#### `validate_event_data(event_type, data)`

*Returns:* `Dict[str, Any]`

Validate event data for a specific event type.

Args:
    event_type: Event type name
    data: Event data to validate
    
Returns:
    Validated data
    
Raises:
    ValueError: If data doesn't conform to schema or event type is unknown

#### `validate_signal_event(signal_event)`

*Returns:* `bool`

Validate a SignalEvent object against the SIGNAL schema.

Args:
    signal_event: SignalEvent object to validate
    
Returns:
    True if valid, raises ValueError otherwise

#### `validate_bar_event(bar_event)`

*Returns:* `bool`

Validate a BarEvent object.

Args:
    bar_event: BarEvent object to validate
    
Returns:
    True if valid, raises ValueError otherwise

#### `get_schema_documentation(event_type=None)`

*Returns:* `str`

Get documentation for event schemas.

Args:
    event_type: Optional event type to get documentation for
               If None, returns documentation for all event types
               
Returns:
    Documentation string

### Classes

#### `EventSchema`

Base class for event data schemas.

##### Methods

###### `__init__(schema_def)`

Initialize an event schema.

Args:
    schema_def: Dictionary defining the schema fields and their properties
        Each field has: type, required, description, validator (optional)

###### `validate(data)`

*Returns:* `Dict[str, Any]`

Validate event data against the schema.

Args:
    data: Event data to validate
    
Returns:
    Validated data (may convert types)
    
Raises:
    ValueError: If data doesn't conform to schema

## event_types

Event Types Module

This module defines the event types used in the trading system's event-driven architecture.
It provides the EventType enumeration and utility functions for event type operations.

### Functions

#### `get_event_description(event_type)`

*Returns:* `str`

Get description for an event type.

Args:
    event_type: Event type
    
Returns:
    Description of the event type

#### `get_all_event_types_with_descriptions()`

*Returns:* `Dict[str, str]`

Get dictionary of all event types with descriptions.

Returns:
    Dictionary mapping event type names to descriptions

### Classes

#### `EventType`

No docstring provided.

##### Methods

###### `market_data_events(cls)`

*Returns:* `Set['EventType']`

Get a set of all market data related event types.

Returns:
    Set of market data event types

###### `order_events(cls)`

*Returns:* `Set['EventType']`

Get a set of all order related event types.

Returns:
    Set of order event types

###### `position_events(cls)`

*Returns:* `Set['EventType']`

Get a set of all position related event types.

Returns:
    Set of position event types

###### `system_events(cls)`

*Returns:* `Set['EventType']`

Get a set of all system related event types.

Returns:
    Set of system event types

###### `from_string(cls, name)`

*Returns:* `'EventType'`

Get event type from string name.

Args:
    name: String name of event type (case insensitive)
    
Returns:
    EventType enum value
    
Raises:
    ValueError: If no matching event type found

#### `BarEvent`

Event specifically for market data bars.

##### Methods

###### `__init__(bar_data, timestamp=None)`

Initialize a bar event.

Args:
    bar_data: Dictionary containing OHLCV data
    timestamp: Optional explicit timestamp (defaults to bar_data's timestamp)

###### `get_symbol()`

*Returns:* `str`

Get the instrument symbol.

###### `get_price()`

*Returns:* `float`

Get the close price.

###### `get_timestamp()`

*Returns:* `datetime`

Get the bar timestamp.

###### `get_open()`

*Returns:* `float`

Get the opening price.

###### `get_high()`

*Returns:* `float`

Get the high price.

###### `get_low()`

*Returns:* `float`

Get the low price.

###### `get_volume()`

*Returns:* `float`

Get the volume.

###### `get_data()`

*Returns:* `Dict[str, Any]`

Get the complete bar data dictionary.

###### `__repr__()`

*Returns:* `str`

String representation of the bar event.

#### `OrderEvent`

Event specifically for trading orders.

##### Methods

###### `__init__(symbol, direction, quantity, price=None, order_type='MARKET', order_id=None, timestamp=None)`

Initialize an order event.

Args:
    symbol: Instrument symbol
    direction: Order direction (1 for buy, -1 for sell)
    quantity: Order quantity
    price: Order price (required for LIMIT and STOP_LIMIT orders)
    order_type: Order type (MARKET, LIMIT, STOP, STOP_LIMIT)
    order_id: Optional order ID (auto-generated if not provided)
    timestamp: Optional timestamp (defaults to now)

###### `get_symbol()`

*Returns:* `str`

Get the order symbol.

###### `get_direction()`

*Returns:* `int`

Get the order direction (1 for buy, -1 for sell).

###### `get_quantity()`

*Returns:* `float`

Get the order quantity.

###### `get_price()`

*Returns:* `Optional[float]`

Get the order price.

###### `get_order_type()`

*Returns:* `str`

Get the order type.

###### `get_order_id()`

*Returns:* `str`

Get the order ID.

###### `__str__()`

*Returns:* `str`

String representation of the order event.

#### `FillEvent`

Event specifically for order fills.

##### Methods

###### `__init__(symbol, quantity, price, direction, order_id=None, transaction_cost=0.0, timestamp=None)`

Initialize a fill event.

Args:
    symbol: Instrument symbol
    quantity: Filled quantity
    price: Fill price
    direction: Fill direction (1 for buy, -1 for sell)
    order_id: Optional order ID that was filled
    transaction_cost: Optional transaction cost
    timestamp: Optional timestamp (defaults to now)

###### `get_symbol()`

*Returns:* `str`

Get the fill symbol.

###### `get_quantity()`

*Returns:* `float`

Get the filled quantity.

###### `get_price()`

*Returns:* `float`

Get the fill price.

###### `get_direction()`

*Returns:* `int`

Get the fill direction (1 for buy, -1 for sell).

###### `get_order_id()`

*Returns:* `Optional[str]`

Get the original order ID.

###### `get_transaction_cost()`

*Returns:* `float`

Get the transaction cost.

###### `get_fill_value()`

*Returns:* `float`

Get the total value of the fill.

###### `__str__()`

*Returns:* `str`

String representation of the fill event.

#### `CancelOrderEvent`

Event for order cancellation requests.

##### Methods

###### `__init__(order_id, reason=None, timestamp=None)`

Initialize a cancel order event.

Args:
    order_id: ID of the order to cancel
    reason: Optional reason for cancellation
    timestamp: Optional timestamp (defaults to now)

###### `get_order_id()`

*Returns:* `str`

Get the order ID to cancel.

###### `get_reason()`

*Returns:* `Optional[str]`

Get the cancellation reason.

###### `__str__()`

*Returns:* `str`

String representation of the cancel event.

#### `PartialFillEvent`

Event for partial order fills.

##### Methods

###### `__init__(symbol, quantity, price, direction, remaining_quantity, order_id=None, transaction_cost=0.0, timestamp=None)`

Initialize a partial fill event.

Args:
    symbol: Instrument symbol
    quantity: Filled quantity
    price: Fill price
    direction: Fill direction (1 for buy, -1 for sell)
    remaining_quantity: Quantity remaining to be filled
    order_id: Optional order ID
    transaction_cost: Optional transaction cost
    timestamp: Optional timestamp (defaults to now)

###### `get_remaining_quantity()`

*Returns:* `float`

Get the remaining quantity to be filled.

###### `__str__()`

*Returns:* `str`

String representation of the partial fill event.

#### `RejectEvent`

Event for order rejections.

##### Methods

###### `__init__(order_id, reason, timestamp=None)`

Initialize a reject event.

Args:
    order_id: ID of the rejected order
    reason: Reason for rejection
    timestamp: Optional timestamp (defaults to now)

###### `get_order_id()`

*Returns:* `str`

Get the rejected order ID.

###### `get_reason()`

*Returns:* `str`

Get the rejection reason.

###### `__str__()`

*Returns:* `str`

String representation of the reject event.

## event_utils

Event Utilities Module

Provides helper functions for working with events, including
unpacking/packing event data and creating standardized objects.

### Functions

#### `unpack_bar_event(event)`

*Returns:* `Tuple[Dict[str, Any], str, float, datetime.datetime]`

Extract bar data from an event object.

Args:
    event: Event object containing bar data
    
Returns:
    tuple: (bar_dict, symbol, price, timestamp)

#### `get_event_timestamp(event)`

*Returns:* `Optional[datetime.datetime]`

Get the timestamp from an event.

Args:
    event: Event object
    
Returns:
    Event timestamp or None if not available

#### `get_event_symbol(event)`

*Returns:* `Optional[str]`

Get the symbol from an event.

Args:
    event: Event object
    
Returns:
    Symbol string or None if not available

#### `create_bar_event(bar_data, timestamp=None)`

*Returns:* `BarEvent`

Create a standardized BarEvent object.

Args:
    bar_data: Dictionary containing bar data
    timestamp: Optional explicit timestamp
    
Returns:
    BarEvent object

#### `create_signal(signal_type, price, timestamp=None, symbol=None, rule_id=None, confidence=1.0, metadata=None)`

*Returns:* `SignalEvent`

Create a standardized SignalEvent object.

DEPRECATED: Use SignalEvent constructor directly.

Args:
    signal_type: Type of signal (BUY, SELL, NEUTRAL)
    price: Price at signal generation
    timestamp: Optional signal timestamp (defaults to now)
    symbol: Optional instrument symbol (defaults to 'default')
    rule_id: Optional ID of the rule that generated the signal
    confidence: Optional confidence score (0-1)
    metadata: Optional additional signal metadata
    
Returns:
    SignalEvent object

#### `create_signal_from_numeric(signal_value, price, timestamp=None, symbol=None, rule_id=None, confidence=1.0, metadata=None)`

*Returns:* `SignalEvent`

Create a SignalEvent from a numeric signal value (-1, 0, 1).

DEPRECATED: Use SignalEvent constructor directly.

Args:
    signal_value: Numeric signal value (-1, 0, 1)
    price: Price at signal generation
    timestamp: Optional signal timestamp
    symbol: Optional instrument symbol
    rule_id: Optional ID of the rule that generated the signal
    confidence: Optional confidence score (0-1)
    metadata: Optional additional signal metadata
    
Returns:
    SignalEvent object

#### `unpack_signal_event(event)`

*Returns:* `Tuple[Any, str, float, Any]`

Extract signal data from an event object.

Args:
    event: Event object containing signal data
    
Returns:
    tuple: (signal, symbol, price, signal_type)
    
Raises:
    TypeError: If event structure is not as expected

#### `create_position_action(action_type, symbol)`

*Returns:* `Dict[str, Any]`

Create a standardized position action dictionary.

This function intentionally returns a dictionary as its purpose is to create
standardized dictionary structures for position actions.

Args:
    action_type: Type of action ('entry', 'exit', 'modify')
    symbol: Instrument symbol
    **kwargs: Additional action parameters
    
Returns:
    Position action dictionary

#### `create_error_event(source, message, error_type=None, original_event=None)`

Create a standardized error event.

This function creates an Event object for error reporting.

Args:
    source: Error source component
    message: Error message
    error_type: Optional error type
    original_event: Optional original event that caused the error
    
Returns:
    Event object with ERROR type

*Returns:* Event object with ERROR type

### Classes

#### `MetricsCollector`

Collects and aggregates metrics from all system components.

##### Methods

###### `__init__(components)`

Initialize metrics collector.

Args:
    components: Dictionary of system components to collect metrics from

###### `get_metrics()`

Get aggregated system metrics.

Returns:
    Dictionary of metrics from all components

*Returns:* Dictionary of metrics from all components

#### `EventValidator`

Validates events to ensure they conform to expected schemas.

##### Methods

###### `__init__()`

No docstring provided.

###### `validate(event)`

Validate an event against its schema.

Args:
    event: Event to validate
    
Returns:
    True if valid
    
Raises:
    ValueError: If event is invalid

*Returns:* True if valid

#### `ErrorHandler`

Centralized handler for system errors.

This component tracks errors, logs them appropriately, and can
perform actions like stopping the system or sending notifications.

##### Methods

###### `__init__(event_bus, log_level)`

No docstring provided.

###### `handle(event)`

Handle an error event.

## portfolio_events

Portfolio Event Classes

This module defines event classes related to portfolio management.

### Classes

#### `PositionActionEvent`

Event for position actions (entry, exit, modify).

##### Methods

###### `__init__(action_type)`

Initialize position action event.

Args:
    action_type: Type of action ('entry', 'exit', 'modify')
    **kwargs: Additional action parameters

###### `get_action_type()`

*Returns:* `str`

Get the action type.

###### `get_symbol()`

*Returns:* `str`

Get the symbol.

###### `get_direction()`

*Returns:* `int`

Get the direction.

#### `PortfolioUpdateEvent`

Event for portfolio state updates.

##### Methods

###### `__init__(portfolio_state, timestamp=None)`

Initialize portfolio update event.

Args:
    portfolio_state: Dictionary containing portfolio state
    timestamp: Event timestamp

###### `get_equity()`

*Returns:* `float`

Get the portfolio equity.

###### `get_cash()`

*Returns:* `float`

Get the portfolio cash.

###### `get_positions_count()`

*Returns:* `int`

Get the number of open positions.

#### `PositionOpenedEvent`

Event emitted when a position is opened.

##### Methods

###### `__init__(position_data, timestamp=None)`

Initialize position opened event.

Args:
    position_data: Dictionary containing position data
    timestamp: Event timestamp

###### `get_position_id()`

*Returns:* `str`

Get the position ID.

###### `get_symbol()`

*Returns:* `str`

Get the symbol.

###### `get_direction()`

*Returns:* `int`

Get the direction.

###### `get_quantity()`

*Returns:* `float`

Get the position quantity.

#### `PositionClosedEvent`

Event emitted when a position is closed.

##### Methods

###### `__init__(position_data, timestamp=None)`

Initialize position closed event.

Args:
    position_data: Dictionary containing position data
    timestamp: Event timestamp

###### `get_position_id()`

*Returns:* `str`

Get the position ID.

###### `get_symbol()`

*Returns:* `str`

Get the symbol.

###### `get_realized_pnl()`

*Returns:* `float`

Get the realized P&L.

## signal_event

Signal Event Module

This module defines the SignalEvent class that extends the base Event class
to represent trading signals directly as events rather than as separate domain objects.

### Classes

#### `SignalEvent`

Event class for trading signals.

##### Methods

###### `__init__(signal_value, price, symbol='default', rule_id=None, metadata=None, timestamp=None)`

Initialize a signal event.

Args:
    signal_value: Signal value (1 for buy, -1 for sell, 0 for neutral)
    price: Price at signal generation
    symbol: Instrument symbol
    rule_id: ID of the rule that generated the signal
    metadata: Additional signal metadata
    timestamp: Signal timestamp

###### `get_signal_value()`

*Returns:* `int`

Get the signal value.

###### `is_active()`

*Returns:* `bool`

Check if this is an active signal (not neutral).

###### `get_signal_name()`

*Returns:* `str`

Get the signal name (BUY, SELL, or NEUTRAL).

###### `get_price()`

*Returns:* `float`

Get the price at signal generation.

###### `get_symbol()`

*Returns:* `str`

Get the instrument symbol.

###### `get_rule_id()`

*Returns:* `Optional[str]`

Get the rule ID that generated the signal.

###### `get_metadata()`

*Returns:* `Dict[str, Any]`

Get the signal metadata.

###### `__str__()`

*Returns:* `str`

String representation of the signal event.
