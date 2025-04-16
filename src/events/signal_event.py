"""
Signal Event Module

This module defines the SignalEvent class that extends the base Event class
to represent trading signals directly as events rather than as separate domain objects.
"""

import datetime
from typing import Dict, Any, Optional, Union

from src.events.event_bus import Event
from src.events.event_types import EventType
from src.signals.signal_processing import SignalType


class SignalEvent(Event):
    """
    Event class for trading signals.
    
    This class extends the base Event class to directly represent trading signals,
    replacing the previous approach of wrapping Signal objects in Events.
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
            'confidence': min(max(confidence, 0.0), 1.0),  # Ensure between 0 and 1
            'metadata': metadata or {},
        }
        
        # Initialize base Event
        super().__init__(EventType.SIGNAL, data, timestamp or datetime.datetime.now())
    
    @property
    def signal_type(self) -> SignalType:
        """Get the signal type."""
        return self.data['signal_type']
    
    @property
    def price(self) -> float:
        """Get the price at signal generation."""
        return self.data['price']
    
    @property
    def symbol(self) -> str:
        """Get the instrument symbol."""
        return self.data['symbol']
    
    @property
    def rule_id(self) -> Optional[str]:
        """Get the rule ID that generated the signal."""
        return self.data['rule_id']
    
    @property
    def confidence(self) -> float:
        """Get the signal confidence."""
        return self.data['confidence']
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the signal metadata."""
        return self.data['metadata']
    
    @property
    def direction(self) -> int:
        """Get the numeric direction of this signal (1, -1, or 0)."""
        return self.signal_type.value if hasattr(self.signal_type, 'value') else 0
    
    def is_active(self) -> bool:
        """Determine if this signal is actionable (not neutral)."""
        return self.signal_type != SignalType.NEUTRAL
    
    def copy(self) -> 'SignalEvent':
        """Create a copy of the signal event."""
        return SignalEvent(
            signal_type=self.signal_type,
            price=self.price,
            symbol=self.symbol,
            rule_id=self.rule_id,
            confidence=self.confidence,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp
        )
    
    @classmethod
    def from_numeric(cls, signal_value: int, price: float,
                     symbol: str = "default", rule_id: Optional[str] = None, 
                     confidence: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None,
                     timestamp: Optional[datetime.datetime] = None) -> 'SignalEvent':
        """
        Create a SignalEvent from a numeric signal value (-1, 0, 1).
        
        Args:
            signal_value: Numeric signal value (-1, 0, 1)
            price: Price at signal generation
            symbol: Instrument symbol
            rule_id: ID of the rule that generated the signal
            confidence: Signal confidence (0-1)
            metadata: Additional signal metadata
            timestamp: Signal timestamp
            
        Returns:
            SignalEvent object
        """
        if signal_value == 1:
            signal_type = SignalType.BUY
        elif signal_value == -1:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
            
        return cls(
            signal_type=signal_type,
            price=price,
            symbol=symbol,
            rule_id=rule_id,
            confidence=confidence,
            metadata=metadata,
            timestamp=timestamp
        )
    
    @classmethod
    def from_signal(cls, signal) -> 'SignalEvent':
        """
        Create a SignalEvent from a legacy Signal object.
        
        This method provides backward compatibility during migration.
        
        Args:
            signal: Legacy Signal object
            
        Returns:
            SignalEvent object
        """
        return cls(
            signal_type=signal.signal_type,
            price=signal.price,
            symbol=getattr(signal, 'symbol', 'default'),
            rule_id=getattr(signal, 'rule_id', None),
            confidence=getattr(signal, 'confidence', 1.0),
            metadata=getattr(signal, 'metadata', {}).copy() if hasattr(signal, 'metadata') else {},
            timestamp=getattr(signal, 'timestamp', datetime.datetime.now())
        )
    
    def __str__(self) -> str:
        """String representation of the signal event."""
        return f"SignalEvent({self.signal_type}, {self.symbol}, price={self.price}, confidence={self.confidence:.2f})"
