## Event System

The Events module provides a robust event-driven architecture that enables decoupled communication between components through a central event bus. It defines standard event types and provides mechanisms for routing events from producers to consumers.

### Core Components

```
events/
├── __init__.py             # Package exports
├── event_bus.py            # Event and EventBus classes
├── event_types.py          # EventType enumeration
├── event_handlers.py       # Event handler classes
├── event_emitters.py       # Event emitter classes
└── schema.py               # Event data schemas
```

### Key Classes

- `EventBus`: Central message broker for routing events
- `Event`: Container for event data with type, payload, and timestamp
- `EventType`: Enumeration of event types (BAR, TICK, SIGNAL, etc.)
- `EventHandler`: Base class for components that process events
- `EventEmitter`: Base class for components that emit events

### Event Types

The system supports various event types organized by category:

```
Market Data Events
├── BAR
├── TICK
├── MARKET_OPEN
└── MARKET_CLOSE

Signal Events
└── SIGNAL

Order Events
├── ORDER
├── CANCEL
└── MODIFY

Execution Events
├── FILL
├── PARTIAL_FILL
└── REJECT

Portfolio Events
├── POSITION_OPENED
├── POSITION_CLOSED
└── POSITION_MODIFIED

System Events
├── START
├── STOP
├── PAUSE
├── RESUME
└── ERROR
```

### Event Handlers

Event handlers process events received from the event bus:

- `LoggingHandler`: Logs events at specified levels
- `FunctionEventHandler`: Delegates event processing to a function
- `FilterHandler`: Filters events based on criteria
- `DebounceHandler`: Prevents processing events too frequently
- `CompositeHandler`: Delegates to multiple handlers
- `AsyncEventHandler`: Processes events asynchronously
- `EventHandlerGroup`: Manages groups of handlers
- Domain-specific handlers: `MarketDataHandler`, `SignalHandler`, `OrderHandler`, `FillHandler`

### Event Emitters

Event emitters generate events and send them to the event bus:

- `MarketDataEmitter`: Emits bar, tick, and market events
- `SignalEmitter`: Emits signal events from strategies
- `OrderEmitter`: Emits order-related events
- `FillEmitter`: Emits fill-related events
- `PortfolioEmitter`: Emits portfolio-related events
- `SystemEmitter`: Emits system-related events

### Example Usage

```python
from src.events.event_bus import EventBus, Event
from src.events.event_types import EventType
from src.events.event_handlers import LoggingHandler
from src.events.event_emitters import MarketDataEmitter

# Create event bus
event_bus = EventBus()

# Create logging handler
logging_handler = LoggingHandler([EventType.BAR])
event_bus.register(EventType.BAR, logging_handler)

# Create market data emitter
market_data_emitter = MarketDataEmitter(event_bus)

# Emit bar event
market_data_emitter.emit_bar({
    "symbol": "AAPL",
    "timestamp": datetime.now(),
    "Open": 150.0,
    "High": 151.5,
    "Low": 149.5,
    "Close": 151.0,
    "Volume": 1000000
})
```
