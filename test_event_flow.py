import datetime
from src.events.event_bus import EventBus, Event
from src.events.event_types import EventType, BarEvent
from src.events.event_manager import EventManager
from src.signals import Signal, SignalType
from src.position_management import PositionManager
from src.engine.execution_engine import ExecutionEngine
from src.strategies.weighted_strategy import WeightedStrategy


def test_event_flow():
    
    # Create event bus
    event_bus = EventBus()
    
    # Create a simple strategy - using WeightedStrategy with no components initially
    # This is just for testing the event flow
    strategy = WeightedStrategy(
        components=[],  # Empty list for testing
        name="TestStrategy"
    )
    
    # Create other necessary components
    position_manager = PositionManager()  # Uses default test portfolio
    execution_engine = ExecutionEngine(position_manager)
    
    # Create event manager
    event_manager = EventManager(
        event_bus=event_bus,
        strategy=strategy,
        position_manager=position_manager,
        execution_engine=execution_engine
    )
    
    # Initialize components
    event_manager.initialize()
    
    # Create a test bar
    bar_data = {
        'symbol': 'TEST',
        'timestamp': datetime.datetime.now(),
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.5,
        'Volume': 10000
    }
    
    # Process through the event manager - note that we pass the bar_data directly
    # The event manager will handle wrapping it properly
    event_manager.process_market_data(bar_data)
    
    # Check results
    status = event_manager.get_status()
    print(f"Event flow test status: {status}")
    
    # Check for specific events in history
    for event in event_bus.history:
        print(f"Event: {event.event_type.name}, Data: {event.data}")

    print("Event flow test completed successfully")


# In your test_event_flow.py

def test_signal_handling():
    """Test signal handling with standardized event pattern."""
    # Create position manager
    position_manager = PositionManager()
    
    # Create a signal
    signal = Signal(
        timestamp=datetime.datetime.now(),
        signal_type=SignalType.BUY,
        price=150.0,
        rule_id="test_rule",
        confidence=0.8,
        metadata={"test": True},
        symbol="TEST"
    )
    
    # Create signal event
    signal_event = Event(EventType.SIGNAL, signal)
    
    # Process the event
    position_manager.on_signal(signal_event)
    
    # Check position manager state
    assert hasattr(position_manager, 'signal_history'), "Position manager should have signal_history attribute"
    assert len(position_manager.signal_history) > 0, "Position manager should have recorded the signal"


if __name__ == "__main__":
    # Execute tests when script is run directly
    print("Running test_event_flow...")
    test_event_flow()
    
    print("\\nRunning test_signal_handling...")
    test_signal_handling()
    
    print("\\nAll tests completed")

