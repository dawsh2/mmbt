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
    # Signal value constants
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    
    def __init__(self, signal_value: int, price: float, 
                 symbol: str = "default", rule_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a signal event.
        
        Args:
            signal_value: Signal value (1 for buy, -1 for sell, 0 for neutral)
            price: Price at signal generation
            symbol: Instrument symbol
            rule_id: ID of the rule that generated the signal
            metadata: Additional signal metadata
            timestamp: Signal timestamp
        """
        # Validate signal value
        if signal_value not in (self.BUY, self.SELL, self.NEUTRAL):
            raise ValueError(f"Invalid signal value: {signal_value}. Must be 1 (BUY), -1 (SELL), or 0 (NEUTRAL).")
            
        # Create signal data
        data = {
            'signal_value': signal_value,
            'price': price,
            'symbol': symbol,
            'rule_id': rule_id,
            'metadata': metadata or {},
        }
        
        # Initialize base Event
        super().__init__(EventType.SIGNAL, data, timestamp)
    
    def get_signal_value(self) -> int:
        """Get the signal value."""
        return self.get('signal_value', self.NEUTRAL)
    
    def is_active(self) -> bool:
        """Check if this is an active signal (not neutral)."""
        return self.get_signal_value() != self.NEUTRAL
    
    def get_signal_name(self) -> str:
        """Get the signal name (BUY, SELL, or NEUTRAL)."""
        value = self.get_signal_value()
        if value == self.BUY:
            return "BUY"
        elif value == self.SELL:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def get_price(self) -> float:
        """Get the price at signal generation."""
        return self.get('price')
    
    def get_symbol(self) -> str:
        """Get the instrument symbol."""
        return self.get('symbol', 'default')
    
    def get_rule_id(self) -> Optional[str]:
        """Get the rule ID that generated the signal."""
        return self.get('rule_id')
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get the signal metadata."""
        return self.get('metadata', {})

    def emit_signal(self, signal):
        """
        Emit a signal event.

        Args:
            signal: SignalEvent to emit
        """
        from src.events.event_types import EventType

        # Create an event to wrap the signal
        event = Event(EventType.SIGNAL, signal)

        # Emit the event
        self.emit(event)
    
    def __str__(self) -> str:
        """String representation of the signal event."""
        signal_name = self.get_signal_name()
        return f"SignalEvent({signal_name}, {self.get_symbol()}, price={self.get_price()})"
