# Signal Migration Guide: Moving to a Fully Event-Driven Architecture

This guide outlines the transition from using separate `Signal` objects to a fully event-driven approach where signals are handled exclusively through the event system.

## Motivation

By treating signals solely as events rather than separate domain objects:

- We gain a more consistent architecture
- We reduce the number of object types to maintain
- We simplify component communication
- We make testing easier with a standardized interface

## Migration Steps

### 1. Standardize Signal Event Data Structure

Define a clear structure for signal event payloads:

```python
signal_data = {
    'timestamp': timestamp,  # When the signal was generated
    'signal_type': signal_type,  # SignalType enum value (BUY, SELL, NEUTRAL)
    'price': price,  # Price at signal generation
    'rule_id': rule_id,  # ID of the rule that generated the signal
    'confidence': confidence,  # Confidence score (0-1)
    'symbol': symbol,  # Instrument symbol
    'metadata': metadata or {}  # Additional signal data
}
```

### 2. Create Utility Functions for Working with Signal Events

Add helper functions for working with signal events:

```python
# In src/events/event_utilities.py

def create_signal_event(event_bus, timestamp, signal_type, price, rule_id=None, 
                       confidence=1.0, symbol=None, metadata=None):
    """Create and emit a standardized signal event."""
    signal_data = {
        'timestamp': timestamp,
        'signal_type': signal_type,
        'price': price,
        'rule_id': rule_id,
        'confidence': min(max(confidence, 0.0), 1.0),  # Ensure between 0 and 1
        'metadata': metadata or {},
        'symbol': symbol or 'default'
    }
    
    event = Event(EventType.SIGNAL, signal_data)
    if event_bus:
        event_bus.emit(event)
    return event

def get_signal_direction(signal_event):
    """Get the numeric direction from a signal event."""
    signal_data = signal_event.data
    signal_type = signal_data.get('signal_type')
    
    if hasattr(signal_type, 'value'):
        return signal_type.value
    return 0

def is_active_signal(signal_event):
    """Determine if a signal event is actionable (not neutral)."""
    signal_data = signal_event.data
    signal_type = signal_data.get('signal_type')
    return signal_type != SignalType.NEUTRAL
```

### 3. Update Event Emitters

Modify the `SignalEmitter` class to work with the new event structure:

```python
# In src/events/event_emitters.py

class SignalEmitter(EventEmitter):
    """Event emitter for trading signals."""
    
    def emit_signal(self, signal_type, price, 
                   rule_id=None, confidence=1.0, symbol=None, metadata=None):
        """Emit a signal event."""
        timestamp = metadata.get('timestamp') if metadata else None
        timestamp = timestamp or datetime.datetime.now()
        
        signal_data = {
            'timestamp': timestamp,
            'signal_type': signal_type,
            'price': price,
            'rule_id': rule_id,
            'confidence': confidence,
            'symbol': symbol or 'default',
            'metadata': metadata or {}
        }
            
        return self.emit(EventType.SIGNAL, signal_data)
```

### 4. Adapt Signal Processing for Events

Create a dedicated processor for signal events:

```python
# Create a new module: src/events/signal_processing.py

class SignalEventProcessor:
    """Process signal events through filtering and confidence scoring."""
    
    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or {}
        # Initialize filters, transforms, etc.
        
    def process_signal_event(self, signal_event, context=None):
        """Process a signal event through filtering and confidence scoring."""
        # Extract signal data
        signal_data = signal_event.data.copy()
        
        # Apply filters
        filtered_data = self._apply_filters(signal_data)
        
        # Apply confidence scoring
        if hasattr(self, 'confidence_scorer'):
            confidence = self.confidence_scorer.calculate_confidence(
                filtered_data, context)
            filtered_data['confidence'] = confidence
            filtered_data['metadata'] = filtered_data.get('metadata', {})
            filtered_data['metadata']['confidence_scored'] = True
        
        # Create a new event with the processed data
        processed_event = Event(
            event_type=EventType.SIGNAL,
            data=filtered_data,
            timestamp=signal_event.timestamp
        )
        processed_event.id = signal_event.id  # Preserve event ID
        
        return processed_event
```

### 5. Update Signal Collection and Router

Modify your existing collection classes to work with events:

```python
# Update src/events/signal_router.py

class SignalEventCollection:
    """A collection of signal events that provides consensus methods."""
    
    def __init__(self):
        """Initialize an empty signal collection."""
        self.signal_events = []
        
    def add(self, signal_event):
        """Add a signal event to the collection."""
        self.signal_events.append(signal_event)
        
    def get_weighted_consensus(self):
        """Get the weighted consensus signal type from all signals."""
        if not self.signal_events:
            return SignalType.NEUTRAL
            
        # Calculate the weighted average of signal values
        total_weight = 0
        weighted_sum = 0
        
        for event in self.signal_events:
            signal_data = event.data
            signal_type = signal_data.get('signal_type')
            weight = signal_data.get('confidence', 1.0)
            
            if hasattr(signal_type, 'value'):
                weighted_sum += signal_type.value * weight
                total_weight += weight
            
        if total_weight > 0:
            avg_value = weighted_sum / total_weight
        else:
            avg_value = 0
            
        # Convert to signal type
        if avg_value > 0.3:
            return SignalType.BUY
        elif avg_value < -0.3:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL
```

### 6. Update Strategy Components

Modify strategies to use the new event-based approach:

```python
class MyStrategy:
    def __init__(self, event_bus, signal_processor=None):
        self.event_bus = event_bus
        self.signal_processor = signal_processor
    
    def on_bar(self, event):
        # Extract data from event
        bar_data = event.data.bar
        
        # Process with strategy logic
        if some_condition:
            # Create and emit a signal event
            from src.events.event_utilities import create_signal_event
            
            signal_event = create_signal_event(
                event_bus=self.event_bus,
                timestamp=bar_data['timestamp'],
                signal_type=SignalType.BUY,
                price=bar_data['Close'],
                rule_id='my_strategy',
                confidence=0.8,
                symbol=bar_data['symbol']
            )
            
            # Process signal through filters if needed
            if self.signal_processor:
                processed_event = self.signal_processor.process_signal_event(signal_event)
                # No need to emit again as create_signal_event already did that
```

## Migration Checklist

- [ ] Create/update the SignalType enum in event_types.py
- [ ] Add utility functions for signal events in event_utilities.py
- [ ] Update SignalEmitter in event_emitters.py
- [ ] Create SignalEventProcessor for processing signal events
- [ ] Update SignalEventCollection and router
- [ ] Modify strategies to use the new event-based approach
- [ ] Update any signal handlers to extract data correctly from event.data
- [ ] Update tests to use the new event-based approach
- [ ] Ensure documentation reflects the new event-based signal approach

## Benefits of Pure Event-Based Architecture

1. **Consistency**: All system communications happen through events
2. **Simplicity**: No need to maintain separate object types
3. **Unified Interface**: All components work with Event objects in a standard way
4. **Easier Testing**: Just need to create Event objects for testing
5. **Reduced Complexity**: Fewer types to maintain and understand
6. **Better Traceability**: All signals flow through the event bus and can be logged/monitored

## Backward Compatibility

During the transition period, you may need to maintain backward compatibility. Consider using adapter functions that can convert between event-based signals and legacy Signal objects if needed.