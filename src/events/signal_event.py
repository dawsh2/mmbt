"""
Signal Event Module

This module defines the SignalEvent class that extends the base Event class
to represent trading signals directly as events rather than as separate domain objects.
"""

import datetime
from typing import Dict, Any, Optional, Union
from enum import Enum

from src.events.event_bus import Event
from src.events.event_types import EventType


class SignalEvent(Event):
    """
    Event class for trading signals.
    """
    
    def __init__(self, signal_type: SignalType, price: float, 
                 symbol: str = "default", rule_id: Optional[str] = None,
                 confidence: float = 1.0, 
                 metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a signal event.
        
        Args:
            signal_type: Type of signal (BUY, SELL, NEUTRAL)
            price: Price at signal generation
            symbol: Instrument symbol
            rule_id: ID of the rule that generated the signal
            confidence: Signal confidence (0-1)
            metadata: Additional signal metadata
            timestamp: Signal timestamp
        """
        # Create signal data
        data = {
            'signal_type': signal_type,
            'price': price,
            'symbol': symbol,
            'rule_id': rule_id,
            'metadata': metadata or {},
        }
        
        # Initialize base Event
        super().__init__(EventType.SIGNAL, data, timestamp)
    
    def get_signal_type(self):
        """Get the signal type."""
        return self.get('signal_type')
    
    def get_price(self):
        """Get the price at signal generation."""
        return self.get('price')
    
    def get_symbol(self):
        """Get the instrument symbol."""
        return self.get('symbol', 'default')
    
    def get_rule_id(self):
        """Get the rule ID that generated the signal."""
        return self.get('rule_id')
    
    def get_metadata(self):
        """Get the signal metadata."""
        return self.get('metadata', {})
    
    def __str__(self):
        """String representation of the signal event."""
        signal_type = self.get_signal_type()
        return f"SignalEvent({signal_type}, {self.get_symbol()}, price={self.get_price()}, confidence={self.get_confidence():.2f})"



class SignalType(Enum):
    """Enumeration of different signal types."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    @property
    def is_active(self):
        """Check if this is an active signal (not NEUTRAL)."""
        return self != SignalType.NEUTRAL    
