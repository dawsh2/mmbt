# Events Module

The Events module provides the core event-driven infrastructure for the trading system. It enables loosely coupled communication between components through a centralized event bus.

## Core Components

### Event Base (`event_base.py`)

The foundation of the event system with the base Event class:

```python
class Event:
    """
    Base class for all events in the trading system.
    
    Events contain a type, timestamp, unique ID, and data payload.
    """
    
    def __init__(self, event_type, data=None, timestamp=None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.datetime.now()
        self.id = str(uuid.uuid4())
```

### Event Types (`event_types.py`)

Defines the enumeration of event types used throughout the system:

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
    
    # Execution events
    FILL = auto()
    PARTIAL_FILL = auto()
    REJECT = auto()
    
    # Position events
    POSITION_ACTION = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()
    POSITION_STOPPED = auto()
    
    # Portfolio events
    PORTFOLIO_UPDATE = auto()
    EQUITY_UPDATE = auto()
    MARGIN_UPDATE = auto()
    
    # System events
    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    ERROR = auto()
    
    # Analysis events
    METRIC_CALCULATED = auto()
    ANALYSIS_COMPLETE = auto()
    
    # Custom event type
    CUSTOM = auto()
```

Also includes the `BarEvent` class for market data representation:

```python
class BarEvent(Event):
    """Event specifically for market data bars."""
    
    def __init__(self, bar_data, timestamp=None):
        # Use bar's timestamp if not explicitly provided
        if timestamp is None and isinstance(bar_data, dict):
            timestamp = bar_data.get('timestamp')
            
        super().__init__(EventType.BAR, bar_data, timestamp)
```

### Event Bus (`event_bus.py`)

The central message broker that routes events between components:

```python
class EventBus:
    """
    Central event bus for routing events between system components.
    
    The event bus maintains a registry of handlers for different event types
    and dispatches events to the appropriate handlers when they are emitted.
    """
    
    def __init__(self, async_mode=False):
        self.handlers = {event_type: [] for event_type in EventType}
        self.async_mode = async_mode
        self.history = []
        self.max_history_size = 1000
        
        # For async mode
        self.event_queue = queue.Queue() if async_mode else None
        self.dispatch_thread = None
        self.running = False
```

### Event Handlers (`event_handlers.py`)

Components that process events of specific types:

```python
class EventHandler(ABC):
    """
    Base class for all event handlers.
    
    Event handlers process specific types of events and can be registered
    with the event bus to receive events of those types.
    """
    
    def __init__(self, event_types: Union[EventType, List[EventType]]):
        if isinstance(event_types, EventType):
            event_types = [event_types]
            
        self.event_types = set(event_types)
        self.enabled = True
```

Specialized handlers include:
- `LoggingHandler`: Logs events at different levels
- `DebounceHandler`: Prevents processing the same event type too frequently
- `FilterHandler`: Only processes events that meet specific criteria
- `AsyncEventHandler`: Processes events asynchronously
- `CompositeHandler`: Delegates to multiple handlers for the same event

### Event Emitters (`event_emitters.py`)

Components that generate and emit events:

```python
class EventEmitter:
    """
    Mixin class for components that emit events.
    
    This class provides a standard interface for emitting events
    to the event bus. It can be mixed into any class that needs
    to generate events.
    """
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
```

Domain-specific emitters include:
- `MarketDataEmitter`: For bar, tick, and market open/close events
- `SignalEmitter`: For trading signal events
- `OrderEmitter`: For order-related events
- `FillEmitter`: For execution fill events
- `PortfolioEmitter`: For position events
- `SystemEmitter`: For system control events

### Signal Event (`signal_event.py`)

Standardized signal event class:

```python
class SignalEvent(Event):
    """
    Event class for trading signals.
    """
    # Signal value constants
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    def __init__(self, signal_value, price, 
                 symbol="default", rule_id=None,
                 metadata=None, timestamp=None):
        # Validate signal value
        if signal_value not in (self.BUY, self.SELL, self.NEUTRAL):
            raise ValueError(f"Invalid signal value: {signal_value}")
            
        # Create signal data
        data = {
            'signal_value': signal_value,
            'price': price,
            'symbol': symbol,
            'rule_id': rule_id,
            'metadata': metadata or {},
        }
        
        # Initialize base Event
        super().__init__(EventType.SIGNAL, data, timestamp)
```

### Portfolio Events (`portfolio_events.py`)

Events related to portfolio and position management:

```python
class PositionOpenedEvent(Event):
    """Event emitted when a position is opened."""
    
    def __init__(self, position_data, timestamp=None):
        super().__init__(EventType.POSITION_OPENED, position_data, timestamp)
```

### Event Schema (`event_schema.py`)

Defines data schemas for different event types:

```python
class EventSchema:
    """Base class for event data schemas."""
    
    def __init__(self, schema_def):
        self.schema_def = schema_def
    
    def validate(self, data):
        """
        Validate event data against the schema.
        """
        validated = {}
        errors = []
        
        # Check required fields
        for field, field_def in self.schema_def.items():
            if field_def.get('required', False) and field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        # Validate fields
        # ...
```

### Event Utilities (`event_utils.py`)

Helper functions for working with events:

```python
def create_error_event(source, message, error_type=None, original_event=None):
    """Create a standardized error event."""
    error_data = {
        'source': source,
        'message': str(message),
        'error_type': error_type or type(message).__name__,
        'timestamp': datetime.datetime.now()
    }
    
    if original_event:
        error_data['original_event_id'] = original_event.id
        error_data['original_event_type'] = original_event.event_type.name
        
    return Event(EventType.ERROR, error_data)
```

## Recent Improvements

### Standardized Event Objects

All event handling now enforces the use of proper event objects rather than dictionaries:

```python
def emit_bar(self, bar_event) -> Event:
    """
    Emit a bar event.
    
    Args:
        bar_event: BarEvent object to emit
    """
    # Ensure bar_event is a BarEvent
    from src.events.event_types import BarEvent
    if not isinstance(bar_event, BarEvent):
        raise TypeError(f"Expected BarEvent object, got {type(bar_event).__name__}")
```

### Resolution of Circular Imports

Circular import issues have been resolved by:
- Moving the base `Event` class to its own module (`event_base.py`)
- Using strategic imports to prevent circularity
- Implementing proper dependency patterns

### Metrics Collection

The EventBus now collects metrics on event processing:
- Event counts by type
- Processing times
- Error rates
- Overall event throughput

### Error Handling Improvements

A centralized `ErrorHandler` class captures and manages errors:
- Tracks error counts by type
- Implements threshold-based system control
- Standardizes error reporting

## Usage Examples

### Registering Event Handlers

```python
event_bus = EventBus()
strategy = TradingStrategy()

# Register the strategy to receive BAR events
event_bus.register(EventType.BAR, strategy)
```

### Creating and Emitting Events

```python
# Create a signal event
signal = SignalEvent(
    signal_value=SignalEvent.BUY,
    price=100.50,
    symbol="AAPL",
    rule_id="sma_crossover"
)

# Emit the signal through an emitter
signal_emitter = SignalEmitter(event_bus)
signal_emitter.emit_signal(signal)
```

### Creating Custom Event Handlers

```python
class MyCustomHandler(EventHandler):
    def __init__(self):
        super().__init__([EventType.SIGNAL])
    
    def _process_event(self, event):
        # Custom processing logic
        signal = event.data
        print(f"Received signal: {signal.get_signal_name()}")
```

## Best Practices

1. **Use Proper Event Objects**: Always use the appropriate event classes (BarEvent, SignalEvent, etc.) rather than dictionaries.

2. **Handle Exceptions**: Implement proper exception handling in all event handlers to prevent errors from cascading through the system.

3. **Limit Event Payload Size**: Keep event data concise and relevant to prevent performance issues with large event payloads.

4. **Monitor Event Metrics**: Use the built-in metrics collection to monitor system performance and identify bottlenecks.

5. **Reset State Properly**: Ensure all handlers properly reset their state when the system is restarted or configurations change.

6. **Document Event Flow**: Maintain clear documentation of which components emit and handle which event types.

7. **Use Event Filtering**: Implement filtering in handlers to process only relevant events and reduce processing overhead.

8. **Validate Event Data**: Use the event schema validation system to ensure data integrity.


# !!!: POSSIBLY OUTDATED INFO BELOW:

# Events Module

The Events module provides the foundation for the trading system's event-driven architecture, enabling decoupled communication between components through standardized event objects.

## Event Flow Diagram:


┌───────────┐    BarEvent    ┌──────────┐    SignalEvent    ┌─────────────────┐
│  Data     │─────────────►  │ Strategy │──────────────────►│ Position        │
│  Handler  │                │          │                   │ Manager         │
└───────────┘                └──────────┘                   └─────────────────┘
                                                                     │
                                                                     │ PositionActionEvent
                                                                     │
                                                                     ▼
┌───────────┐    FillEvent    ┌──────────┐    OrderEvent    ┌─────────────────┐
│           │◄───────────────┤ Execution │◄─────────────────│ EventPortfolio  │
│ Analytics │                │ Engine    │                  │                 │
└───────────┘                └──────────┘                  └─────────────────┘
      ▲                                                           │
      │                                                           │
      └───────────────────────────────────────────────────────────┘
                           PortfolioUpdateEvent



The complete event flow in the system:

## Event Flow

The event flow in the system is managed by the EventManager, which:

1. Receives raw market data and wraps it in BarEvent objects
2. Routes bar events to Strategy components, which produce Signal events
3. Transforms raw Signal objects into standardized event format
4. Routes signal events to Position Manager, which produces Order events
5. Routes order events to Execution Engine, which produces Fill events
6. Routes fill events to Portfolio for position and equity tracking

The EventManager adds additional functionality:
- Maintains tracking information (price history, signal history, etc.)
- Handles data conversion between different formats
- Provides error handling and logging
- Ensures proper event sequencing

This event-driven approach decouples components and standardizes communication throughout the system.

## Core Components

### Event Types
taxonomy of events

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
    
    # Execution events
    FILL = auto()
    PARTIAL_FILL = auto()
    REJECT = auto()
    
    # Position events
    POSITION_ACTION = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()
    
    # Portfolio events
    PORTFOLIO_UPDATE = auto()
    EQUITY_UPDATE = auto()
    
    # System events
    START = auto()
    STOP = auto()
    ERROR = auto()
```

### Event Class

```python
class Event:
    def __init__(self, event_type: EventType, data: Any = None, 
                timestamp: Optional[datetime.datetime] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.datetime.now()
        self.id = str(uuid.uuid4())
```

### Event Bus

```python
class EventBus:
    def __init__(self, async_mode: bool = False):
        self.handlers = {event_type: [] for event_type in EventType}
        self.async_mode = async_mode
        self.history = []
        
    def register(self, event_type: EventType, handler) -> None:
        """Register a handler for an event type."""
        
    def unregister(self, event_type: EventType, handler) -> bool:
        """Unregister a handler for an event type."""
        
    def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        
    def get_history(self, event_type: Optional[EventType] = None,
                   start_time: Optional[datetime.datetime] = None,
                   end_time: Optional[datetime.datetime] = None) -> List[Event]:
        """Get event history, optionally filtered."""
```

### Event Handlers

```python
class EventHandler(ABC):
    def __init__(self, event_types: Union[EventType, List[EventType]]):
        self.event_types = set(event_types) if isinstance(event_types, list) else {event_types}
        self.enabled = True
        
    def handle(self, event: Event) -> None:
        """Process an event."""
        
    @abstractmethod
    def _process_event(self, event: Event) -> None:
        """Internal method to process an event."""
```

### Specialized Event Classes

#### BarEvent

```python
class BarEvent(Event):
    """Specialized event for bar data."""
    
    def __init__(self, bar_data):
        """Initialize with bar data."""
        super().__init__(EventType.BAR, bar_data)

    @property
    def timestamp(self):
        """Get the bar timestamp."""
        return self.bar.get('timestamp')

    @property
    def open(self):
        """Get the opening price."""
        return self.data.get('Open')
    
    @property
    def high(self):
        """Get the high price."""
        return self.data.get('High')
    
    @property
    def low(self):
        """Get the low price."""
        return self.data.get('Low')
    
    @property
    def close(self):
        """Get the closing price."""
        return self.data.get('Close')
    
    @property
    def volume(self):
        """Get the volume."""
        return self.data.get('Volume')
    
    @property
    def symbol(self):
        """Get the symbol."""
        return self.data.get('symbol')

## potentially include VWAP or trade_count 		
```

## Bar Data Standardization

The system uses a standardized `BarEvent` class from the events module to encapsulate bar data consistently across all components.

### Using BarEvent Objects

```python
from src.events.event_types import BarEvent
from src.events.event_bus import Event
from src.events.event_types import EventType

# Create a BarEvent from dictionary data
bar_data = {
    "timestamp": datetime.now(),
    "Open": 100.0,
    "High": 101.0,
    "Low": 99.0,
    "Close": 100.5,
    "Volume": 1000,
    "symbol": "AAPL"
}
bar_event = BarEvent(bar_data)

# Create and emit a BAR event
event = Event(EventType.BAR, bar_event)
event_bus.emit(event)

# Accessing bar data in event handlers
def on_bar(self, event):
    bar_event = event.data
    if isinstance(bar_event, BarEvent):
        bar_data = bar_event.bar
        symbol = bar_event.get_symbol()
        close_price = bar_event.get_price()
        timestamp = bar_event.get_timestamp()
        # Process the bar data...

#### SignalEvent

```python
class SignalEvent(Event):
    def __init__(self, signal_type, price, symbol="default", rule_id=None, 
                confidence=1.0, metadata=None, timestamp=None):
        # Create signal data
        data = {...}  # Signal data dictionary
        super().__init__(EventType.SIGNAL, data, timestamp)
    
    @property
    def signal_type(self):
        """Get the signal type."""
        
    @property
    def price(self):
        """Get the price at signal generation."""
        
    @property
    def direction(self):
        """Get the numeric direction of this signal."""
        
    def is_active(self):
        """Determine if this signal is actionable."""
```

#### PositionActionEvent

```python
class PositionActionEvent(Event):
    def __init__(self, action_type, **kwargs):
        data = {'action_type': action_type, **kwargs}
        super().__init__(EventType.POSITION_ACTION, data)
    
    @property
    def action_type(self):
        """Get the action type."""
        
    @property
    def symbol(self):
        """Get the symbol."""
        
    @property
    def direction(self):
        """Get the direction."""
```

#### PortfolioUpdateEvent

```python
class PortfolioUpdateEvent(Event):
    def __init__(self, portfolio_state, timestamp=None):
        super().__init__(EventType.PORTFOLIO_UPDATE, portfolio_state, timestamp)
    
    @property
    def equity(self):
        """Get the portfolio equity."""
        
    @property
    def cash(self):
        """Get the portfolio cash."""
```

## Event Utility Functions

```python
def create_signal_event(signal_type, price, symbol="default", rule_id=None, 
                       confidence=1.0, metadata=None, timestamp=None):
    """Create a standardized signal event."""
    
def unpack_bar_event(event):
    """Extract bar data from an event safely."""
    
def get_signal_direction(event):
    """Extract direction from a signal event."""
    
def create_position_action(action, **kwargs):
    """Create a standardized position action dictionary."""
```

## Integration Examples

### Creating a New Strategy

```python
class MyStrategy(StrategyBase):
    def generate_signals(self, event):
        # Extract bar data
        bar_data = extract_bar_data(event)
        
        # Strategy logic
        if some_condition:
            # Create and return signal event
            return SignalEvent(
                signal_type=SignalType.BUY,
                price=bar_data.get('Close'),
                symbol=bar_data.get('symbol'),
                rule_id=self.name
            )
        
        return None  # No signal
```

### Setting Up Event Flow

```python
# Create event bus
event_bus = EventBus()

# Create components
strategy = MyStrategy('my_strategy', event_bus)
position_manager = PositionManager(event_bus)
portfolio = EventPortfolio(100000, event_bus)

# Register event handlers
event_bus.register(EventType.BAR, strategy.on_bar)
event_bus.register(EventType.SIGNAL, position_manager.on_signal)
event_bus.register(EventType.POSITION_ACTION, portfolio.handle)

# Create and emit bar event
bar_data = {...}  # OHLCV data
bar_event = BarEvent(bar_data)
event = Event(EventType.BAR, bar_event)
event_bus.emit(event)
```

### Creating an Event Handler

```python
class MyAnalytics(EventHandler):
    def __init__(self):
        super().__init__([EventType.PORTFOLIO_UPDATE, EventType.POSITION_CLOSED])
        self.portfolio_history = []
        self.trade_history = []
    
    def _process_event(self, event):
        if event.event_type == EventType.PORTFOLIO_UPDATE:
            self.portfolio_history.append({
                'timestamp': event.timestamp,
                'equity': event.data.get('equity'),
                'cash': event.data.get('cash')
            })
        elif event.event_type == EventType.POSITION_CLOSED:
            self.trade_history.append({
                'symbol': event.data.get('symbol'),
                'pnl': event.data.get('realized_pnl')
            })
```

### Add Integration Examples as Needed Here


