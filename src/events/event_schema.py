"""
Event Data Schema Documentation

This module defines the data schemas for different event types in the system.
It provides type definitions, validation utilities, and schema documentation
to ensure consistency across the event-driven architecture.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Callable

from src.events.event_types import EventType
from src.events.signal_event import SignalEvent


class EventSchema:
    """Base class for event data schemas."""
    
    def __init__(self, schema_def: Dict[str, Dict[str, Any]]):
        """
        Initialize an event schema.
        
        Args:
            schema_def: Dictionary defining the schema fields and their properties
                Each field has: type, required, description, validator (optional)
        """
        self.schema_def = schema_def
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate event data against the schema.
        
        Args:
            data: Event data to validate
            
        Returns:
            Validated data (may convert types)
            
        Raises:
            ValueError: If data doesn't conform to schema
        """
        validated = {}
        errors = []
        
        # Check required fields
        for field, field_def in self.schema_def.items():
            if field_def.get('required', False) and field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        # Validate and convert fields
        for field, value in data.items():
            if field in self.schema_def:
                field_def = self.schema_def[field]
                expected_type = field_def.get('type')
                
                # Check type
                if expected_type and not isinstance(value, expected_type):
                    # Handle datetime conversion from string
                    if expected_type == datetime and isinstance(value, str):
                        try:
                            value = datetime.fromisoformat(value)
                        except ValueError:
                            errors.append(f"Field {field}: Cannot convert '{value}' to datetime")
                    else:
                        errors.append(f"Field {field}: Expected {expected_type.__name__}, got {type(value).__name__}")
                
                # Run custom validator if provided
                validator = field_def.get('validator')
                if validator and callable(validator):
                    try:
                        validator(value)
                    except Exception as e:
                        errors.append(f"Field {field} validation failed: {str(e)}")
                
                validated[field] = value
            else:
                # Include unspecified fields as-is
                validated[field] = value
        
        if errors:
            raise ValueError("\n".join(errors))
            
        return validated


# Schema Definitions

BAR_SCHEMA = EventSchema({
    'timestamp': {
        'type': datetime,
        'required': True,
        'description': 'Bar timestamp'
    },
    'Open': {
        'type': float,
        'required': True,
        'description': 'Opening price'
    },
    'High': {
        'type': float,
        'required': True,
        'description': 'High price'
    },
    'Low': {
        'type': float,
        'required': True,
        'description': 'Low price'
    },
    'Close': {
        'type': float,
        'required': True,
        'description': 'Closing price'
    },
    'Volume': {
        'type': float,
        'required': False,
        'description': 'Volume'
    },
    'is_eod': {
        'type': bool,
        'required': False,
        'description': 'Whether this is end of day'
    },
    'symbol': {
        'type': str,
        'required': False,
        'description': 'Instrument symbol'
    }
})

# Updated to match SignalEvent structure
SIGNAL_SCHEMA = EventSchema({
    'signal_type': {
        'required': True,
        'description': 'Signal type (BUY, SELL, NEUTRAL)'
    },
    'price': {
        'type': float,
        'required': True,
        'description': 'Price at signal generation'
    },
    'symbol': {
        'type': str,
        'required': False,  # Default value will be 'default'
        'description': 'Instrument symbol'
    },
    'rule_id': {
        'type': str,
        'required': False,
        'description': 'ID of rule that generated the signal'
    },
    'confidence': {
        'type': float,
        'required': False,
        'description': 'Confidence score (0-1)',
        'validator': lambda x: 0 <= x <= 1
    },
    'metadata': {
        'type': dict,
        'required': False,
        'description': 'Additional signal metadata'
    },
    'timestamp': {
        'type': datetime,
        'required': False,  # Will default to now if not provided
        'description': 'Signal timestamp'
    }
})

ORDER_SCHEMA = EventSchema({
    'timestamp': {
        'type': datetime,
        'required': True,
        'description': 'Order timestamp'
    },
    'symbol': {
        'type': str,
        'required': True,
        'description': 'Instrument symbol'
    },
    'order_type': {
        'type': str,
        'required': True,
        'description': 'Order type (MARKET, LIMIT, etc)',
        'validator': lambda x: x in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
    },
    'quantity': {
        'type': float,
        'required': True,
        'description': 'Order quantity'
    },
    'direction': {
        'type': int,
        'required': True,
        'description': 'Order direction (1 for buy, -1 for sell)',
        'validator': lambda x: x in [1, -1]
    },
    'price': {
        'type': float,
        'required': False,
        'description': 'Limit price (for LIMIT orders)'
    },
    'order_id': {
        'type': str,
        'required': False,
        'description': 'Unique order ID'
    }
})

FILL_SCHEMA = EventSchema({
    'timestamp': {
        'type': datetime,
        'required': True,
        'description': 'Fill timestamp'
    },
    'symbol': {
        'type': str,
        'required': True,
        'description': 'Instrument symbol'
    },
    'quantity': {
        'type': float,
        'required': True,
        'description': 'Filled quantity'
    },
    'price': {
        'type': float,
        'required': True,
        'description': 'Fill price'
    },
    'direction': {
        'type': int,
        'required': True,
        'description': 'Direction (1 for buy, -1 for sell)',
        'validator': lambda x: x in [1, -1]
    },
    'order_id': {
        'type': str,
        'required': False,
        'description': 'Original order ID'
    },
    'transaction_cost': {
        'type': float,
        'required': False,
        'description': 'Transaction cost'
    }
})

MARKET_CLOSE_SCHEMA = EventSchema({
    'timestamp': {
        'type': datetime,
        'required': True,
        'description': 'Market close timestamp'
    },
    'close_positions': {
        'type': bool,
        'required': False,
        'description': 'Whether to close positions'
    }
})

# Add a schema for BarEvent
BAR_EVENT_SCHEMA = EventSchema({
    'bar': {
        'type': dict,
        'required': True,
        'description': 'Raw bar data dictionary'
    }
})

# Schema registry for easy access
EVENT_SCHEMAS = {
    'BAR': BAR_SCHEMA,
    'BAR_EVENT': BAR_EVENT_SCHEMA,
    'SIGNAL': SIGNAL_SCHEMA,
    'ORDER': ORDER_SCHEMA,
    'FILL': FILL_SCHEMA,
    'MARKET_CLOSE': MARKET_CLOSE_SCHEMA
}


def validate_event_data(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate event data for a specific event type.
    
    Args:
        event_type: Event type name
        data: Event data to validate
        
    Returns:
        Validated data
        
    Raises:
        ValueError: If data doesn't conform to schema or event type is unknown
    """
    schema = EVENT_SCHEMAS.get(event_type.upper())
    if not schema:
        raise ValueError(f"Unknown event type: {event_type}")
    
    return schema.validate(data)


def validate_signal_event(signal_event: SignalEvent) -> bool:
    """
    Validate a SignalEvent object against the SIGNAL schema.
    
    Args:
        signal_event: SignalEvent object to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    try:
        # Extract data from SignalEvent to validate against schema
        signal_data = {
            'signal_type': signal_event.signal_type,
            'price': signal_event.price,
            'symbol': signal_event.symbol,
            'rule_id': signal_event.rule_id,
            'confidence': signal_event.confidence,
            'metadata': signal_event.metadata,
            'timestamp': signal_event.timestamp
        }
        
        # Validate against schema
        SIGNAL_SCHEMA.validate(signal_data)
        return True
    except ValueError as e:
        raise ValueError(f"Invalid SignalEvent: {str(e)}")


def validate_bar_event(bar_event) -> bool:
    """
    Validate a BarEvent object.
    
    Args:
        bar_event: BarEvent object to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    try:
        # Validate the underlying bar data
        BAR_SCHEMA.validate(bar_event.data)
        return True
    except ValueError as e:
        raise ValueError(f"Invalid BarEvent: {str(e)}")    


def get_schema_documentation(event_type: Optional[str] = None) -> str:
    """
    Get documentation for event schemas.
    
    Args:
        event_type: Optional event type to get documentation for
                   If None, returns documentation for all event types
                   
    Returns:
        Documentation string
    """
    if event_type:
        schema = EVENT_SCHEMAS.get(event_type.upper())
        if not schema:
            return f"Unknown event type: {event_type}"
        
        doc = f"{event_type} Event:\n"
        for field, field_def in schema.schema_def.items():
            req = "required" if field_def.get('required', False) else "optional"
            doc += f"  {field}: {field_def.get('description', '')} ({req})\n"
        return doc
    else:
        doc = "Event Schema Documentation:\n\n"
        for event_type, schema in EVENT_SCHEMAS.items():
            doc += f"{event_type} Event:\n"
            for field, field_def in schema.schema_def.items():
                req = "required" if field_def.get('required', False) else "optional"
                doc += f"  {field}: {field_def.get('description', '')} ({req})\n"
            doc += "\n"
        return doc


# Example usage
if __name__ == "__main__":
    # Example validation
    bar_data = {
        'timestamp': datetime.now(),
        'Open': 100.5,
        'High': 101.2,
        'Low': 99.8,
        'Close': 100.9,
        'Volume': 5000,
        'symbol': 'AAPL'
    }
    
    validated = validate_event_data('BAR', bar_data)
    print("Validated BAR data:", validated)
    
    # Print schema documentation
    print("\n" + get_schema_documentation('SIGNAL'))
