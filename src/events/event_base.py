"""
Event Base Module

This module provides the base Event class used throughout the event system.
Modified to preserve object references without serialization.
"""

import uuid
import datetime
from typing import Any, Optional

class Event:
    """
    Base class for all events in the trading system.

    Events contain a type, timestamp, unique ID, and data payload.
    The data payload preserves object types and is not serialized.
    """

    def __init__(self, event_type, data: Any = None, 
               timestamp: Optional[datetime.datetime] = None):
        """
        Initialize an event.
        
        Args:
            event_type: Type of the event
            data: Optional data payload (preserved as the original object type)
            timestamp: Event timestamp (defaults to current time)
        """
        self.event_type = event_type
        self._data = data  # Store reference to original object
        self.timestamp = timestamp or datetime.datetime.now()
        self.id = str(uuid.uuid4())
    
    @property
    def data(self):
        """
        Get the event data.
        
        Returns:
            The original data object without serialization
        """
        return self._data
    
    def get(self, key, default=None):
        """
        Get a value from the event data.
        
        Supports multiple access patterns:
        1. Dictionary access if data is a dict
        2. Attribute access if data has the attribute
        3. Method access if data has a compatible get() method
        
        Args:
            key: Dictionary key or attribute name to retrieve
            default: Default value if key is not found
            
        Returns:
            Value for the key or default
        """
        # Handle dictionary-like data
        if isinstance(self._data, dict) and key in self._data:
            return self._data[key]
        
        # Handle object attributes
        if hasattr(self._data, key):
            return getattr(self._data, key)
        
        # Handle objects with get method (like SignalEvent)
        if hasattr(self._data, 'get') and callable(self._data.get):
            try:
                return self._data.get(key, default)
            except Exception:
                pass
        
        return default
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event(type={self.event_type.name}, id={self.id}, timestamp={self.timestamp})"
