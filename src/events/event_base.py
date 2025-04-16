"""
Event Base Module

This module provides the base Event class used throughout the event system.
"""

import uuid
import datetime
from typing import Any, Optional

class Event:
    """
    Base class for all events in the trading system.

    Events contain a type, timestamp, unique ID, and data payload.
    """

    def __init__(self, event_type, data: Any = None, 
               timestamp: Optional[datetime.datetime] = None):
        """
        Initialize an event.
        
        Args:
            event_type: Type of the event
            data: Optional data payload
            timestamp: Event timestamp (defaults to current time)
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.datetime.now()
        self.id = str(uuid.uuid4())
    
    def get(self, key, default=None):
        """
        Get a value from the event data.
        
        Args:
            key: Dictionary key to retrieve
            default: Default value if key is not found
            
        Returns:
            Value for the key or default
        """
        if isinstance(self.data, dict):
            return self.data.get(key, default)
        return default
    
    def __str__(self) -> str:
        """String representation of the event."""
        return f"Event(type={self.event_type.name}, id={self.id}, timestamp={self.timestamp})"
