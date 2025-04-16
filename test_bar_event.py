# Create a test script to verify the BarEvent functionality
import datetime
from src.events.event_types import BarEvent

def test_bar_event():
    # Create a bar data dictionary
    bar_data = {
        'symbol': 'AAPL',
        'timestamp': datetime.datetime.now(),
        'Open': 150.0,
        'High': 151.0,
        'Low': 149.0,
        'Close': 150.5,
        'Volume': 1000000
    }
    
    # Create a BarEvent
    bar_event = BarEvent(bar_data)
    
    # Test the get method
    assert bar_event.get('symbol') == 'AAPL'
    assert bar_event.get('Close') == 150.5
    assert bar_event.get('nonexistent', 'default') == 'default'
    
    # Test other accessor methods
    assert bar_event.get_symbol() == 'AAPL'
    assert bar_event.get_price() == 150.5
    assert bar_event.get_timestamp() == bar_data['timestamp']
    
    print("All tests passed!")

if __name__ == "__main__":
    test_bar_event()
