"""
Event system module for the event-driven backtesting engine.
This module defines the core event classes and queue implementation.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import queue
from typing import Dict, Any, Optional


class EventType(Enum):
    """Event types in the backtesting system."""
    MARKET = 0
    SIGNAL = 1
    ORDER = 2
    FILL = 3


@dataclass
class Event:
    """Base event class."""
    type: EventType
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketEvent(Event):
    """Event for new market data."""
    symbol: str
    bar_data: Dict[str, Any]
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.MARKET


@dataclass
class SignalEvent(Event):
    """Event for trading signals."""
    symbol: str
    datetime: datetime
    signal_type: int  # 1: long, -1: short, 0: exit
    strength: float = 1.0
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Event for orders."""
    symbol: str
    order_type: str  # 'MKT' or 'LMT'
    quantity: int
    direction: int  # 1: long, -1: short
    price: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """Event for order fills."""
    symbol: str
    quantity: int
    direction: int
    fill_price: float
    commission: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.type = EventType.FILL


class EventQueue:
    """Queue for events."""
    def __init__(self):
        self._queue = queue.Queue()
    
    def put(self, event):
        """Add an event to the queue."""
        self._queue.put(event)
    
    def get(self):
        """Get the next event from the queue."""
        return self._queue.get()
    
    def empty(self):
        """Check if the queue is empty."""
        return self._queue.empty()
