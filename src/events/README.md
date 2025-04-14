# Events Module Documentation

The Events module provides an event-driven architecture for communication between different components of the trading system. It enables loosely coupled components to interact through a central event bus.

## Core Concepts

**Event**: Base class for all events in the system, containing a type, timestamp, and data payload.  
**EventType**: Enumeration of different types of events (BAR, SIGNAL, ORDER, FILL, etc.).  
**EventHandler**: Base class for handlers that process specific types of events.  
**EventBus**: Central message bus that routes events between system components.  
**EventEmitter**: Mixin for components that emit events to the event bus.

## Basic Usage

```python
from events import EventBus, Event, EventType, EventHandler

# Create an event bus
event_bus = EventBus()

# Define a handler
class SignalHandler(EventHandler):
    def __init__(self):
        super().__init__([EventType.SIGNAL])
        
    def _process_event(self, event):
        signal = event.data
        print(f"Received signal: {signal}")

# Register the handler
signal_handler = SignalHandler()
event_bus.register(EventType.SIGNAL, signal_handler)

# Create and emit an event
signal_data = {"symbol": "AAPL", "direction": 1, "strength": 0.8}
event = Event(EventType.SIGNAL, signal_data)
event_bus.emit(event)
```

## API Reference

### EventType

Enumeration of event types in the trading system.

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
    
    # Portfolio events
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()
    
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

### Event

Base class for all events in the trading system.

**Constructor Parameters:**
- `event_type` (EventType): Type of the event
- `data` (Any, optional): Optional data payload
- `timestamp` (datetime, optional): Event timestamp (defaults to current time)

**Attributes:**
- `event_type` (EventType): Type of the event
- `data` (Any): Data payload
- `timestamp` (datetime): When the event occurred
- `id` (str): Unique event ID

**Example:**
```python
from events import Event, EventType
from datetime import datetime

# Create an event with custom timestamp
event = Event(
    event_type=EventType.BAR,
    data={"symbol": "AAPL", "close": 150.75},
    timestamp=datetime.now()
)
```

### EventHandler

Base class for event handlers.

**Constructor Parameters:**
- `event_types` (EventType|List[EventType]): Event type(s) this handler processes

**Methods:**
- `can_handle(event_type)`: Check if this handler can process events of the given type
  - `event_type` (EventType): Event type to check
  - Returns: True if this handler can process the event type, False otherwise

- `handle(event)`: Process an event
  - `event` (Event): Event to process

- `_process_event(event)`: Internal method to process an event (must be implemented by subclasses)
  - `event` (Event): Event to process

**Example:**
```python
from events import EventHandler, EventType, Event

class OrderHandler(EventHandler):
    def __init__(self):
        super().__init__([EventType.ORDER, EventType.CANCEL])
        
    def _process_event(self, event):
        if event.event_type == EventType.ORDER:
            print(f"Processing order: {event.data}")
        elif event.event_type == EventType.CANCEL:
            print(f"Processing cancellation: {event.data}")
```

### FunctionEventHandler

Event handler that delegates processing to a function.

**Constructor Parameters:**
- `event_types` (EventType|List[EventType]): Event type(s) this handler processes
- `handler_func` (Callable[[Event], None]): Function to call for event processing

**Example:**
```python
from events import FunctionEventHandler, EventType

def process_signal(event):
    signal = event.data
    print(f"Processing signal for {signal['symbol']}")

# Create a function-based handler
signal_handler = FunctionEventHandler(EventType.SIGNAL, process_signal)
```

### EventBus

Central event bus for routing events between system components.

**Constructor Parameters:**
- `async_mode` (bool, optional): Whether to dispatch events asynchronously (default: False)

**Methods:**
- `register(event_type, handler)`: Register a handler for an event type
  - `event_type` (EventType): Event type to register for
  - `handler` (EventHandler|Callable): Handler to register

- `unregister(event_type, handler)`: Unregister a handler for an event type
  - `event_type` (EventType): Event type to unregister from
  - `handler` (EventHandler|Callable): Handler to unregister
  - Returns: True if handler was unregistered, False if not found

- `emit(event)`: Emit an event to all registered handlers
  - `event` (Event): Event to emit

- `emit_all(events)`: Emit multiple events to all registered handlers
  - `events` (List[Event]): List of events to emit

- `clear_history()`: Clear the event history

- `get_history(event_type=None, start_time=None, end_time=None)`: Get event history, optionally filtered
  - `event_type` (EventType, optional): Optional event type to filter by
  - `start_time` (datetime, optional): Optional start time for filtering
  - `end_time` (datetime, optional): Optional end time for filtering
  - Returns: List of events matching the filters

- `reset()`: Reset the event bus state

**Example:**
```python
from events import EventBus, Event, EventType

# Create an event bus
event_bus = EventBus()

# Register handlers
event_bus.register(EventType.BAR, bar_handler)
event_bus.register(EventType.SIGNAL, signal_handler)

# Emit an event
event = Event(EventType.BAR, bar_data)
event_bus.emit(event)

# Get filtered history
bar_events = event_bus.get_history(EventType.BAR)
```

### EventEmitter

Mixin class for components that emit events.

**Constructor Parameters:**
- `event_bus` (EventBus): Event bus to emit events to

**Methods:**
- `emit(event_type, data=None)`: Create and emit an event
  - `event_type` (EventType): Type of event to emit
  - `data` (Any, optional): Optional data payload
  - Returns: The emitted event

- `emit_event(event)`: Emit an existing event
  - `event` (Event): Event to emit

**Example:**
```python
from events import EventEmitter, EventType, EventBus

class Strategy(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def generate_signal(self, symbol, direction):
        # Create signal data
        signal = {
            "symbol": symbol,
            "direction": direction,
            "timestamp": datetime.now()
        }
        
        # Emit signal event
        self.emit(EventType.SIGNAL, signal)
```

## Example Event Handlers

### MarketDataHandler

Event handler for market data events.

**Constructor Parameters:**
- `strategy`: Strategy instance to forward data to

**Example:**
```python
from events import EventHandler, EventType

class MarketDataHandler(EventHandler):
    def __init__(self, strategy):
        super().__init__([EventType.BAR, EventType.TICK])
        self.strategy = strategy
        
    def _process_event(self, event):
        if event.event_type == EventType.BAR:
            self.strategy.on_bar(event.data)
        elif event.event_type == EventType.TICK:
            self.strategy.on_tick(event.data)
```

### SignalHandler

Event handler for signal events.

**Constructor Parameters:**
- `portfolio_manager`: Portfolio manager to forward signals to

**Example:**
```python
from events import EventHandler, EventType

class SignalHandler(EventHandler):
    def __init__(self, portfolio_manager):
        super().__init__([EventType.SIGNAL])
        self.portfolio_manager = portfolio_manager
        
    def _process_event(self, event):
        self.portfolio_manager.on_signal(event.data)
```

### OrderHandler

Event handler for order events.

**Constructor Parameters:**
- `execution_engine`: Execution engine to forward orders to

**Example:**
```python
from events import EventHandler, EventType

class OrderHandler(EventHandler):
    def __init__(self, execution_engine):
        super().__init__([EventType.ORDER, EventType.CANCEL, EventType.MODIFY])
        self.execution_engine = execution_engine
        
    def _process_event(self, event):
        if event.event_type == EventType.ORDER:
            self.execution_engine.place_order(event.data)
        elif event.event_type == EventType.CANCEL:
            self.execution_engine.cancel_order(event.data)
        elif event.event_type == EventType.MODIFY:
            self.execution_engine.modify_order(event.data)
```

## Advanced Usage

### Creating a Complete Event-Driven System

```python
from events import EventBus, Event, EventType, EventHandler, EventEmitter

# Create event bus
event_bus = EventBus()

# Create components
class DataFeed(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def process_bar(self, bar_data):
        self.emit(EventType.BAR, bar_data)

class Strategy(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        self.positions = {}
        
    def on_bar(self, bar_data):
        symbol = bar_data["symbol"]
        
        # Simple strategy: Buy when price increases
        if symbol in self.positions:
            return
            
        if bar_data["close"] > bar_data["open"]:
            signal = {
                "symbol": symbol,
                "direction": 1,  # Buy
                "price": bar_data["close"]
            }
            self.emit(EventType.SIGNAL, signal)

class PortfolioManager(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def on_signal(self, signal):
        # Convert signal to order
        order = {
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "quantity": 100,
            "order_type": "MARKET"
        }
        self.emit(EventType.ORDER, order)

class ExecutionEngine(EventEmitter):
    def __init__(self, event_bus):
        super().__init__(event_bus)
        
    def place_order(self, order):
        # Simulate order execution
        fill = {
            "symbol": order["symbol"],
            "direction": order["direction"],
            "quantity": order["quantity"],
            "price": order.get("price", 100.0),  # Simulated price
            "timestamp": datetime.now()
        }
        self.emit(EventType.FILL, fill)

# Create handlers
class BarHandler(EventHandler):
    def __init__(self, strategy):
        super().__init__([EventType.BAR])
        self.strategy = strategy
        
    def _process_event(self, event):
        self.strategy.on_bar(event.data)

class SignalHandler(EventHandler):
    def __init__(self, portfolio_manager):
        super().__init__([EventType.SIGNAL])
        self.portfolio_manager = portfolio_manager
        
    def _process_event(self, event):
        self.portfolio_manager.on_signal(event.data)

class OrderHandler(EventHandler):
    def __init__(self, execution_engine):
        super().__init__([EventType.ORDER])
        self.execution_engine = execution_engine
        
    def _process_event(self, event):
        self.execution_engine.place_order(event.data)

class FillHandler(EventHandler):
    def __init__(self):
        super().__init__([EventType.FILL])
        
    def _process_event(self, event):
        fill = event.data
        print(f"Order filled: {fill['quantity']} shares of {fill['symbol']} at ${fill['price']}")

# Initialize components
data_feed = DataFeed(event_bus)
strategy = Strategy(event_bus)
portfolio_manager = PortfolioManager(event_bus)
execution_engine = ExecutionEngine(event_bus)

# Initialize handlers
bar_handler = BarHandler(strategy)
signal_handler = SignalHandler(portfolio_manager)
order_handler = OrderHandler(execution_engine)
fill_handler = FillHandler()

# Register handlers
event_bus.register(EventType.BAR, bar_handler)
event_bus.register(EventType.SIGNAL, signal_handler)
event_bus.register(EventType.ORDER, order_handler)
event_bus.register(EventType.FILL, fill_handler)

# Start system
data_feed.process_bar({
    "symbol": "AAPL",
    "open": 150.0,
    "high": 152.5,
    "low": 149.5,
    "close": 152.0,
    "volume": 1000000
})
```

### Asynchronous Event Processing

```python
import asyncio
from events import EventBus, Event, EventType, EventHandler

# Custom asynchronous event bus
class AsyncEventBus(EventBus):
    def __init__(self):
        super().__init__(async_mode=True)
        self.event_queue = asyncio.Queue()
        self.running = False
        
    async def emit_async(self, event):
        """Emit an event asynchronously."""
        await self.event_queue.put(event)
        
    async def process_events(self):
        """Process events from the queue."""
        self.running = True
        while self.running:
            try:
                event = await self.event_queue.get()
                # Dispatch event to handlers
                for handler_ref in self.handlers[event.event_type]:
                    handler = handler_ref()
                    if handler:
                        # Run handler in a task to prevent blocking
                        asyncio.create_task(self._async_handle(handler, event))
                self.event_queue.task_done()
            except Exception as e:
                print(f"Error processing event: {e}")
                
    async def _async_handle(self, handler, event):
        """Handle an event asynchronously."""
        try:
            await asyncio.to_thread(handler.handle, event)
        except Exception as e:
            print(f"Error in handler: {e}")
    
    def stop(self):
        """Stop processing events."""
        self.running = False

# Example usage
async def main():
    # Create async event bus
    event_bus = AsyncEventBus()
    
    # Start event processing
    process_task = asyncio.create_task(event_bus.process_events())
    
    # Create and emit events
    for i in range(10):
        event = Event(EventType.BAR, {"index": i})
        await event_bus.emit_async(event)
        await asyncio.sleep(0.1)
    
    # Wait for all events to be processed
    await event_bus.event_queue.join()
    
    # Stop event processing
    event_bus.stop()
    await process_task
    
# Run the async example
if __name__ == "__main__":
    asyncio.run(main())
```

### Event Monitoring and Analytics

```python
from events import EventHandler, EventType, EventBus
from collections import defaultdict
import threading
import time

class EventMonitor(EventHandler):
    """Monitor events for analytics and debugging."""
    
    def __init__(self, event_bus):
        super().__init__([event_type for event_type in EventType])
        self.event_bus = event_bus
        self.event_counts = defaultdict(int)
        self.event_timing = defaultdict(list)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Register with all event types
        for event_type in EventType:
            event_bus.register(event_type, self)
        
    def _process_event(self, event):
        with self.lock:
            # Count events by type
            self.event_counts[event.event_type] += 1
            
            # Record event timing
            elapsed = time.time() - self.start_time
            self.event_timing[event.event_type].append(elapsed)
    
    def get_summary(self):
        """Get event processing summary."""
        with self.lock:
            total_events = sum(self.event_counts.values())
            elapsed = time.time() - self.start_time
            
            summary = {
                "total_events": total_events,
                "events_per_second": total_events / elapsed if elapsed > 0 else 0,
                "event_counts": dict(self.event_counts),
                "elapsed_time": elapsed
            }
            
            # Calculate event frequencies
            frequencies = {}
            for event_type, timestamps in self.event_timing.items():
                if len(timestamps) >= 2:
                    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                    avg_interval = sum(intervals) / len(intervals)
                    frequencies[event_type.name] = 1 / avg_interval if avg_interval > 0 else 0
                    
            summary["event_frequencies"] = frequencies
            
            return summary
    
    def reset(self):
        """Reset monitoring stats."""
        with self.lock:
            self.event_counts.clear()
            self.event_timing.clear()
            self.start_time = time.time()

# Usage
event_bus = EventBus()
monitor = EventMonitor(event_bus)

# After some time
summary = monitor.get_summary()
print(f"Total events: {summary['total_events']}")
print(f"Events per second: {summary['events_per_second']:.2f}")
print("Event counts:")
for event_type, count in summary['event_counts'].items():
    print(f"  {event_type.name}: {count}")
```

### Creating Custom Event Types

```python
from events import Event, EventType, EventBus

# Add custom event types to the enum if needed
class CustomEventType(EventType):
    MARKET_DATA_READY = auto()
    ANALYSIS_STARTED = auto()
    MODEL_TRAINED = auto()
    BACKTEST_COMPLETED = auto()

# Create custom event classes for specialized data
class BacktestResultEvent(Event):
    def __init__(self, data, timestamp=None):
        super().__init__(CustomEventType.BACKTEST_COMPLETED, data, timestamp)
        
    @property
    def total_return(self):
        return self.data.get('total_return', 0)
        
    @property
    def sharpe_ratio(self):
        return self.data.get('sharpe_ratio', 0)
    
    @property
    def trade_count(self):
        return self.data.get('trade_count', 0)

# Usage
backtest_results = {
    'total_return': 0.155,  # 15.5%
    'sharpe_ratio': 1.2,
    'trade_count': 45,
    'win_rate': 0.65,
    'max_drawdown': 0.08
}

event = BacktestResultEvent(backtest_results)
print(f"Backtest return: {event.total_return * 100:.1f}%")
print(f"Sharpe ratio: {event.sharpe_ratio:.2f}")
```

## Best Practices

1. **Keep events decoupled and focused**: Each event should have a single, clear purpose

2. **Use typed data payloads**: Define clear structures for event data to ensure consistency

3. **Handle events defensively**: Handlers should gracefully handle unexpected data formats

4. **Avoid circular dependencies**: Be careful not to create circular event chains that cause infinite loops

5. **Monitor performance**: Watch for bottlenecks in event processing, especially for high-frequency events

6. **Consider event batching**: For high-volume events, consider batching to improve performance

7. **Maintain event history selectively**: Store only the events needed for audit or replay purposes

8. **Clean handler references**: Use weak references to avoid memory leaks when components are removed

9. **Test event flow**: Verify event propagation through the system works as expected

10. **Document event schemas**: Clearly define the expected structure and content of event data payloads