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


