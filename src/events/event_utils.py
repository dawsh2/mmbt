"""
Event Utilities Module

Provides helper functions for working with events, including
unpacking/packing event data and creating standardized objects.
"""

import datetime
import warnings
from typing import Dict, Any, Tuple, Optional

from src.events.event_types import EventType, BarEvent
from src.events.event_base import Event
from src.events.signal_event import SignalEvent

import logging
logger = logging.getLogger(__name__)

def unpack_bar_event(event) -> Tuple[Dict[str, Any], str, float, datetime.datetime]:
    """
    Extract bar data from an event object.
    
    Args:
        event: Event object containing bar data
        
    Returns:
        tuple: (bar_dict, symbol, price, timestamp)
    """
    if not isinstance(event, Event):
        raise TypeError(f"Expected Event object, got {type(event)}")
    
    # If it's a BarEvent
    if isinstance(event.data, BarEvent):
        return event.data.get_data(), event.data.get_symbol(), event.data.get_price(), event.data.get_timestamp()
    
    # If it's a standard event with bar data
    if event.event_type == EventType.BAR:
        bar_data = event.data
        if isinstance(bar_data, dict) and 'Close' in bar_data:
            warnings.warn(
                "Using dictionary for bar data is deprecated. Use BarEvent objects instead.",
                DeprecationWarning, stacklevel=2
            )
            symbol = bar_data.get('symbol', 'unknown')
            price = bar_data.get('Close')
            timestamp = bar_data.get('timestamp')
            return bar_data, symbol, price, timestamp
    
    raise TypeError(f"Could not extract bar data from event")


def get_event_timestamp(event) -> Optional[datetime.datetime]:
    """
    Get the timestamp from an event.
    
    Args:
        event: Event object
        
    Returns:
        Event timestamp or None if not available
    """
    # If it's an Event object, get timestamp from it
    if isinstance(event, Event):
        return event.timestamp
    
    # If it's a dict with timestamp
    if isinstance(event, dict) and 'timestamp' in event:
        return event['timestamp']
    
    # If it has a timestamp attribute
    if hasattr(event, 'timestamp'):
        return event.timestamp
    
    # If it's a BarEvent inside an Event
    if isinstance(event, Event) and isinstance(event.data, BarEvent):
        return event.data.get_timestamp()
    
    # If it's a SignalEvent inside an Event
    if isinstance(event, Event) and isinstance(event.data, SignalEvent):
        return event.data.timestamp
    
    return None


def get_event_symbol(event) -> Optional[str]:
    """
    Get the symbol from an event.
    
    Args:
        event: Event object
        
    Returns:
        Symbol string or None if not available
    """
    # If it's a dict with symbol
    if isinstance(event, dict) and 'symbol' in event:
        return event['symbol']
    
    # If it has a symbol attribute
    if hasattr(event, 'symbol'):
        return event.symbol
    
    # If it's a BarEvent inside an Event
    if isinstance(event, Event) and isinstance(event.data, BarEvent):
        return event.data.get_symbol()
    
    # If it's a SignalEvent inside an Event
    if isinstance(event, Event) and isinstance(event.data, SignalEvent):
        return event.data.get_symbol()
    
    # If it's a dict inside an Event
    if isinstance(event, Event) and isinstance(event.data, dict) and 'symbol' in event.data:
        return event.data['symbol']
    
    return None


def create_bar_event(bar_data: Dict[str, Any], timestamp=None) -> BarEvent:
    """
    Create a standardized BarEvent object.
    
    Args:
        bar_data: Dictionary containing bar data
        timestamp: Optional explicit timestamp
        
    Returns:
        BarEvent object
    """
    warnings.warn(
        "create_bar_event() is deprecated. Use BarEvent constructor directly.",
        DeprecationWarning, stacklevel=2
    )
    return BarEvent(bar_data, timestamp)


# Signal Event Utilities - DEPRECATED, KEPT FOR BACKWARD COMPATIBILITY
def create_signal(
    signal_type, 
    price: float, 
    timestamp=None, 
    symbol: str = None, 
    rule_id: str = None, 
    confidence: float = 1.0, 
    metadata: Optional[Dict[str, Any]] = None
) -> SignalEvent:
    """
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
    """
    warnings.warn(
        "create_signal() is deprecated. Use SignalEvent constructor directly.",
        DeprecationWarning, stacklevel=2
    )
    return SignalEvent(
        signal_value=signal_type.value if hasattr(signal_type, 'value') else signal_type,
        price=price,
        symbol=symbol or "default",
        rule_id=rule_id,
        confidence=confidence,
        metadata=metadata or {},
        timestamp=timestamp
    )


def create_signal_from_numeric(
    signal_value: int, 
    price: float, 
    timestamp=None, 
    symbol: str = None, 
    rule_id: str = None, 
    confidence: float = 1.0, 
    metadata: Optional[Dict[str, Any]] = None
) -> SignalEvent:
    """
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
    """
    warnings.warn(
        "create_signal_from_numeric() is deprecated. Use SignalEvent constructor directly.",
        DeprecationWarning, stacklevel=2
    )
    return SignalEvent(
        signal_value=signal_value,
        price=price,
        symbol=symbol or "default",
        rule_id=rule_id,
        confidence=confidence,
        metadata=metadata or {},
        timestamp=timestamp
    )


def unpack_signal_event(event) -> Tuple[Any, str, float, Any]:
    """
    Extract signal data from an event object.
    
    Args:
        event: Event object containing signal data
        
    Returns:
        tuple: (signal, symbol, price, signal_type)
        
    Raises:
        TypeError: If event structure is not as expected
    """
    if not hasattr(event, 'data'):
        raise TypeError(f"Expected Event object with data attribute")
    
    # If it's already a SignalEvent, extract needed properties
    if isinstance(event, SignalEvent):
        return event, event.get_symbol(), event.get_price(), event.get_signal_value()
        
    # If event.data is a SignalEvent
    if isinstance(event.data, SignalEvent):
        signal = event.data
        return signal, signal.get_symbol(), signal.get_price(), signal.get_signal_value()
    
    # If event.data is a dictionary with signal data - DEPRECATED
    if isinstance(event.data, dict) and 'signal_type' in event.data:
        warnings.warn(
            "Using dictionary for signal data is deprecated. Use SignalEvent objects instead.",
            DeprecationWarning, stacklevel=2
        )
        signal = event.data
        symbol = signal.get('symbol', 'default')
        price = signal.get('price')
        signal_type = signal.get('signal_type')
        return signal, symbol, price, signal_type
    
    raise TypeError(f"Expected SignalEvent or signal data dictionary in event.data")


# Position Action Utilities - These are INTENDED to return dictionaries, which is fine
def create_position_action(action_type: str, symbol: str, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized position action dictionary.
    
    This function intentionally returns a dictionary as its purpose is to create
    standardized dictionary structures for position actions.
    
    Args:
        action_type: Type of action ('entry', 'exit', 'modify')
        symbol: Instrument symbol
        **kwargs: Additional action parameters
        
    Returns:
        Position action dictionary
    """
    action = {
        'action': action_type,
        'symbol': symbol,
        **kwargs
    }
    return action


def create_error_event(source, message, error_type=None, original_event=None):
    """
    Create a standardized error event.
    
    This function creates an Event object for error reporting.
    
    Args:
        source: Error source component
        message: Error message
        error_type: Optional error type
        original_event: Optional original event that caused the error
        
    Returns:
        Event object with ERROR type
    """
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


class MetricsCollector:
    """
    Collects and aggregates metrics from all system components.
    """
    
    def __init__(self, components):
        """
        Initialize metrics collector.
        
        Args:
            components: Dictionary of system components to collect metrics from
        """
        self.components = components
        self.start_time = datetime.datetime.now()
        
    def get_metrics(self):
        """
        Get aggregated system metrics.
        
        Returns:
            Dictionary of metrics from all components
        """
        metrics = {
            'system': {
                'run_time': (datetime.datetime.now() - self.start_time).total_seconds(),
                'timestamp': datetime.datetime.now()
            }
        }
        
        # Collect metrics from each component
        for name, component in self.components.items():
            if hasattr(component, 'get_metrics'):
                metrics[name] = component.get_metrics()
        
        return metrics


class EventValidator:
    """
    Validates events to ensure they conform to expected schemas.
    """
    
    def __init__(self):
        # Import schemas
        from src.events.event_schema import EVENT_SCHEMAS
        self.schemas = EVENT_SCHEMAS
        
    def validate(self, event):
        """
        Validate an event against its schema.
        
        Args:
            event: Event to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If event is invalid
        """
        # Get schema for this event type
        event_type_name = event.event_type.name
        if event_type_name not in self.schemas:
            raise ValueError(f"No schema defined for event type: {event_type_name}")
            
        schema = self.schemas[event_type_name]
        
        # If event.data is a specialized event object
        if hasattr(event.data, '__dict__'):
            # Extract attributes from object
            data = {attr: getattr(event.data, attr) 
                   for attr in dir(event.data) 
                   if not attr.startswith('_') and not callable(getattr(event.data, attr))}
        elif isinstance(event.data, dict):
            data = event.data
        else:
            # If data is neither an object nor dict, we can't validate
            raise ValueError(f"Cannot validate event data of type: {type(event.data)}")
            
        # Validate against schema
        try:
            schema.validate(data)
            return True
        except ValueError as e:
            raise ValueError(f"Invalid {event_type_name} event: {str(e)}")    


class ErrorHandler:
    """
    Centralized handler for system errors.
    
    This component tracks errors, logs them appropriately, and can
    perform actions like stopping the system or sending notifications.
    """
    
    def __init__(self, event_bus, log_level=logging.ERROR):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        self.error_counts = {}
        self.error_threshold = 5  # Max errors of same type before system halt
        
        # Register with event bus
        self.event_bus.register(EventType.ERROR, self)
        
    def handle(self, event):
        """Handle an error event."""
        if event.event_type != EventType.ERROR:
            return
            
        error_data = event.data
        error_type = error_data.get('error_type', 'unknown')
        error_msg = error_data.get('message', 'No message')
        source = error_data.get('source', 'unknown')
        
        # Log the error
        self.logger.error(f"Error in {source}: {error_msg}")
        
        # Track error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Check if threshold exceeded
        if self.error_counts[error_type] >= self.error_threshold:
            self.logger.critical(
                f"Error threshold exceeded for {error_type}. System halting."
            )
            # Emit system halt event
            halt_event = Event(
                EventType.STOP, 
                {
                    'reason': f"Error threshold exceeded for {error_type}",
                    'emergency': True
                }
            )
            self.event_bus.emit(halt_event)
