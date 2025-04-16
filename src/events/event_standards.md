# Event Handling Standards

This document outlines the standard event interface for all components in the trading system.

## Event Structure

All events in the system follow a standard structure:

1. **Event Class**: An instance of `Event` with:
   - `event_type`: An `EventType` enum value
   - `data`: A standardized data payload (see below)
   - `timestamp`: The event timestamp
   - `id`: A unique event ID

2. **Data Payload**:
   - BAR events: `BarEvent` object with market data
   - SIGNAL events: `Signal` object with signal type, price, etc.
   - ORDER events: `Order` object or standardized dictionary
   - FILL events: `Fill` object or standardized dictionary

## Component Requirements

All components that handle events should:

1. **Implement standard interfaces**:
   - `EventProcessor` for components that process events
   - Event-specific methods (e.g., `on_bar`, `on_signal`) for components that only process specific events

2. **Use the standard data access methods**:
   - Access bar data through `BarEvent` methods, not directly from the underlying dictionary
   - Use standard field names and formats for all events

3. **Document event handling**:
   - Specify which event types the component processes
   - Specify which event types the component produces
   - Describe any unique data requirements or transformations

## Example Usage

```python
class SimpleStrategy(EventProcessor):
    """
    Example strategy showing standard event handling.
    
    Processes:
      - BAR events with BarEvent payloads
      
    Produces:
      - SIGNAL events with Signal payloads
    """
    
    def on_bar(self, event):
        """Process market data and generate signals."""
        bar_event = event.get_data()
        symbol = bar_event.get_symbol()
        close_price = bar_event.get_price()
        timestamp = bar_event.get_timestamp()
        
        # Generate signal logic...
        
        return signal