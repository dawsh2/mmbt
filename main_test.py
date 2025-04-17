import datetime

from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent


# Create a simple event bus (with validation disabled to simplify)
event_bus = EventBus(async_mode=False, validate_events=False)

# Create a simple handler function
def print_bar(event):
    bar_event = event.data
    print(f"Bar received: {bar_event.get_symbol()} @ {bar_event.get_timestamp()}")

# Register the handler
event_bus.register(EventType.BAR, print_bar)

# Create and emit a bar event
bar_data = {
    'timestamp': datetime.datetime.now(),
    'symbol': 'SPY',
    'Open': 521.23,
    'High': 521.32,
    'Low': 520.92,
    'Close': 521.01,
    'Volume': 342898
}
bar_event = BarEvent(bar_data)
event = Event(EventType.BAR, bar_event)
event_bus.emit(event)
