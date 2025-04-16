"""
Event Types Module

This module defines the event types used in the trading system's event-driven architecture.
It provides the EventType enumeration and utility functions for event type operations.
"""

import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Set, Any

from src.events.event_base import Event

import logging 

class EventType(Enum):
    # Market data events
    BAR = auto()
    TICK = auto()
    MARKET_OPEN = auto()
    MARKET_CLOSE = auto()
    
    # Signal events
    SIGNAL = auto()
    
    # Order events
    ORDER = auto()
    CANCEL = auto()
    MODIFY = auto()
    
    # Execution events
    FILL = auto()
    PARTIAL_FILL = auto()
    REJECT = auto()
    
    # Position events
    POSITION_ACTION = auto()  # New event type for position actions
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_MODIFIED = auto()
    POSITION_STOPPED = auto()  # For stop-loss/take-profit events
    
    # Portfolio events
    PORTFOLIO_UPDATE = auto()  # For general portfolio state updates
    EQUITY_UPDATE = auto()     # For equity/PnL updates
    MARGIN_UPDATE = auto()     # For margin requirement updates
    
    # System events
    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    ERROR = auto()
    
    # Analysis events
    METRIC_CALCULATED = auto()
    ANALYSIS_COMPLETE = auto()
    
    # Custom event type
    CUSTOM = auto()
    
    @classmethod
    def market_data_events(cls) -> Set['EventType']:
        """
        Get a set of all market data related event types.
        
        Returns:
            Set of market data event types
        """
        return {cls.BAR, cls.TICK, cls.MARKET_OPEN, cls.MARKET_CLOSE}
    
    @classmethod
    def order_events(cls) -> Set['EventType']:
        """
        Get a set of all order related event types.
        
        Returns:
            Set of order event types
        """
        return {cls.ORDER, cls.CANCEL, cls.MODIFY, cls.FILL, cls.PARTIAL_FILL, cls.REJECT}
    
    @classmethod
    def position_events(cls) -> Set['EventType']:
        """
        Get a set of all position related event types.
        
        Returns:
            Set of position event types
        """
        return {cls.POSITION_OPENED, cls.POSITION_CLOSED, cls.POSITION_MODIFIED}
    
    @classmethod
    def system_events(cls) -> Set['EventType']:
        """
        Get a set of all system related event types.
        
        Returns:
            Set of system event types
        """
        return {cls.START, cls.STOP, cls.PAUSE, cls.RESUME, cls.ERROR}
    
    @classmethod
    def from_string(cls, name: str) -> 'EventType':
        """
        Get event type from string name.
        
        Args:
            name: String name of event type (case insensitive)
            
        Returns:
            EventType enum value
            
        Raises:
            ValueError: If no matching event type found
        """
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"No event type with name: {name}")



class BarEvent(Event):
    """Event specifically for market data bars."""
    
    def __init__(self, bar_data: Dict[str, Any], timestamp: Optional[datetime] = None):
        """
        Initialize a bar event.
        
        Args:
            bar_data: Dictionary containing OHLCV data
            timestamp: Optional explicit timestamp (defaults to bar_data's timestamp)
        """
        # Use bar's timestamp if not explicitly provided
        if timestamp is None and isinstance(bar_data, dict):
            timestamp = bar_data.get('timestamp')
            
        super().__init__(EventType.BAR, bar_data, timestamp)
    
    def get_symbol(self) -> str:
        """Get the instrument symbol."""
        return self.data.get('symbol', 'default')

    # refactor to get_close
    def get_price(self) -> float:
        """Get the close price."""
        return self.data.get('Close')
    
    def get_timestamp(self) -> datetime:
        """Get the bar timestamp."""
        return self.data.get('timestamp', self.timestamp)
    
    def get_open(self) -> float:
        """Get the opening price."""
        return self.data.get('Open')
    
    def get_high(self) -> float:
        """Get the high price."""
        return self.data.get('High')
    
    def get_low(self) -> float:
        """Get the low price."""
        return self.data.get('Low')
    
    def get_volume(self) -> float:
        """Get the volume."""
        return self.data.get('Volume')
    
    def get_data(self) -> Dict[str, Any]:
        """Get the complete bar data dictionary."""
        return self.data
    
    def __repr__(self) -> str:
        """String representation of the bar event."""
        symbol = self.get_symbol()
        timestamp = self.get_timestamp()
        return f"BarEvent({symbol} @ {timestamp}, O:{self.get_open():.2f}, H:{self.get_high():.2f}, L:{self.get_low():.2f}, C:{self.get_price():.2f})"        



# Event type categories with descriptions
EVENT_TYPE_DESCRIPTIONS = {
    EventType.BAR: "New price bar with OHLCV data",
    EventType.TICK: "Individual tick data with price and volume",
    EventType.MARKET_OPEN: "Market opening notification",
    EventType.MARKET_CLOSE: "Market closing notification",
    
    EventType.SIGNAL: "Trading signal generated by a strategy",
    
    EventType.ORDER: "Order request for execution",
    EventType.CANCEL: "Order cancellation request",
    EventType.MODIFY: "Order modification request",
    
    EventType.FILL: "Order completely filled",
    EventType.PARTIAL_FILL: "Order partially filled",
    EventType.REJECT: "Order rejected by broker or exchange",
    
    EventType.POSITION_OPENED: "New position opened",
    EventType.POSITION_CLOSED: "Existing position closed",
    EventType.POSITION_MODIFIED: "Position size or parameters modified",
    
    EventType.START: "System or component start",
    EventType.STOP: "System or component stop",
    EventType.PAUSE: "System or component pause",
    EventType.RESUME: "System or component resume",
    EventType.ERROR: "System or component error",
    
    EventType.METRIC_CALCULATED: "Performance metric calculation completed",
    EventType.ANALYSIS_COMPLETE: "Analysis process completed",
    
    EventType.CUSTOM: "Custom event type"
}


def get_event_description(event_type: EventType) -> str:
    """
    Get description for an event type.
    
    Args:
        event_type: Event type
        
    Returns:
        Description of the event type
    """
    return EVENT_TYPE_DESCRIPTIONS.get(event_type, "No description available")


def get_all_event_types_with_descriptions() -> Dict[str, str]:
    """
    Get dictionary of all event types with descriptions.
    
    Returns:
        Dictionary mapping event type names to descriptions
    """
    return {event_type.name: get_event_description(event_type) 
            for event_type in EventType}


class OrderEvent(Event):
    """Event specifically for trading orders."""
    
    # Order type constants
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    
    def __init__(self, symbol: str, direction: int, quantity: float, 
                 price: Optional[float] = None, order_type: str = "MARKET",
                 order_id: Optional[str] = None, timestamp: Optional[datetime.datetime] = None):
        """
        Initialize an order event.
        
        Args:
            symbol: Instrument symbol
            direction: Order direction (1 for buy, -1 for sell)
            quantity: Order quantity
            price: Order price (required for LIMIT and STOP_LIMIT orders)
            order_type: Order type (MARKET, LIMIT, STOP, STOP_LIMIT)
            order_id: Optional order ID (auto-generated if not provided)
            timestamp: Optional timestamp (defaults to now)
        """
        # Validate direction
        if direction not in (1, -1):
            raise ValueError(f"Invalid direction: {direction}. Must be 1 (buy) or -1 (sell).")
            
        # Validate order type
        if order_type not in (self.MARKET, self.LIMIT, self.STOP, self.STOP_LIMIT):
            raise ValueError(f"Invalid order type: {order_type}")
            
        # Ensure price is provided for LIMIT and STOP_LIMIT orders
        if order_type in (self.LIMIT, self.STOP_LIMIT) and price is None:
            raise ValueError(f"Price must be provided for {order_type} orders")
            
        # Generate order ID if not provided
        if order_id is None:
            order_id = str(uuid.uuid4())
            
        # Create order data
        data = {
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'order_id': order_id
        }
        
        # Initialize base Event
        super().__init__(EventType.ORDER, data, timestamp)
    
    def get_symbol(self) -> str:
        """Get the order symbol."""
        return self.data['symbol']
    
    def get_direction(self) -> int:
        """Get the order direction (1 for buy, -1 for sell)."""
        return self.data['direction']
    
    def get_quantity(self) -> float:
        """Get the order quantity."""
        return self.data['quantity']
    
    def get_price(self) -> Optional[float]:
        """Get the order price."""
        return self.data['price']
    
    def get_order_type(self) -> str:
        """Get the order type."""
        return self.data['order_type']
    
    def get_order_id(self) -> str:
        """Get the order ID."""
        return self.data['order_id']
    
    def __str__(self) -> str:
        """String representation of the order event."""
        direction = "BUY" if self.get_direction() == 1 else "SELL"
        price_str = f"@ {self.get_price()}" if self.get_price() is not None else "@ MARKET"
        return f"OrderEvent({direction} {self.get_quantity()} {self.get_symbol()} {price_str}, {self.get_order_type()}, ID: {self.get_order_id()})"


class FillEvent(Event):
    """Event specifically for order fills."""
    
    def __init__(self, symbol: str, quantity: float, price: float, direction: int,
                order_id: Optional[str] = None, transaction_cost: float = 0.0,
                timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a fill event.
        
        Args:
            symbol: Instrument symbol
            quantity: Filled quantity
            price: Fill price
            direction: Fill direction (1 for buy, -1 for sell)
            order_id: Optional order ID that was filled
            transaction_cost: Optional transaction cost
            timestamp: Optional timestamp (defaults to now)
        """
        # Validate direction
        if direction not in (1, -1):
            raise ValueError(f"Invalid direction: {direction}. Must be 1 (buy) or -1 (sell).")
            
        # Create fill data
        data = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'direction': direction,
            'order_id': order_id,
            'transaction_cost': transaction_cost
        }
        
        # Initialize base Event
        super().__init__(EventType.FILL, data, timestamp)
    
    def get_symbol(self) -> str:
        """Get the fill symbol."""
        return self.data['symbol']
    
    def get_quantity(self) -> float:
        """Get the filled quantity."""
        return self.data['quantity']
    
    def get_price(self) -> float:
        """Get the fill price."""
        return self.data['price']
    
    def get_direction(self) -> int:
        """Get the fill direction (1 for buy, -1 for sell)."""
        return self.data['direction']
    
    def get_order_id(self) -> Optional[str]:
        """Get the original order ID."""
        return self.data['order_id']
    
    def get_transaction_cost(self) -> float:
        """Get the transaction cost."""
        return self.data['transaction_cost']
    
    def get_fill_value(self) -> float:
        """Get the total value of the fill."""
        return self.get_quantity() * self.get_price()
    
    def __str__(self) -> str:
        """String representation of the fill event."""
        direction = "BUY" if self.get_direction() == 1 else "SELL"
        return f"FillEvent({direction} {self.get_quantity()} {self.get_symbol()} @ {self.get_price()}, cost={self.get_transaction_cost():.2f})"



class CancelOrderEvent(Event):
    """Event for order cancellation requests."""
    
    def __init__(self, order_id: str, reason: Optional[str] = None, 
                timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a cancel order event.
        
        Args:
            order_id: ID of the order to cancel
            reason: Optional reason for cancellation
            timestamp: Optional timestamp (defaults to now)
        """
        data = {
            'order_id': order_id,
            'reason': reason
        }
        
        super().__init__(EventType.CANCEL, data, timestamp)
    
    def get_order_id(self) -> str:
        """Get the order ID to cancel."""
        return self.data['order_id']
    
    def get_reason(self) -> Optional[str]:
        """Get the cancellation reason."""
        return self.data['reason']
    
    def __str__(self) -> str:
        """String representation of the cancel event."""
        reason_str = f", reason: {self.get_reason()}" if self.get_reason() else ""
        return f"CancelOrderEvent(order_id: {self.get_order_id()}{reason_str})"


class PartialFillEvent(FillEvent):
    """Event for partial order fills."""
    
    def __init__(self, symbol: str, quantity: float, price: float, direction: int,
                remaining_quantity: float, order_id: Optional[str] = None,
                transaction_cost: float = 0.0, timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a partial fill event.
        
        Args:
            symbol: Instrument symbol
            quantity: Filled quantity
            price: Fill price
            direction: Fill direction (1 for buy, -1 for sell)
            remaining_quantity: Quantity remaining to be filled
            order_id: Optional order ID
            transaction_cost: Optional transaction cost
            timestamp: Optional timestamp (defaults to now)
        """
        # Initialize parent FillEvent
        super().__init__(symbol, quantity, price, direction, order_id, transaction_cost, timestamp)
        
        # Add remaining quantity
        self.data['remaining_quantity'] = remaining_quantity
        
        # Override event type
        self.event_type = EventType.PARTIAL_FILL
    
    def get_remaining_quantity(self) -> float:
        """Get the remaining quantity to be filled."""
        return self.data['remaining_quantity']
    
    def __str__(self) -> str:
        """String representation of the partial fill event."""
        direction = "BUY" if self.get_direction() == 1 else "SELL"
        return f"PartialFillEvent({direction} {self.get_quantity()}/{self.get_quantity() + self.get_remaining_quantity()} {self.get_symbol()} @ {self.get_price()}, remaining: {self.get_remaining_quantity()})"


class RejectEvent(Event):
    """Event for order rejections."""
    
    def __init__(self, order_id: str, reason: str, timestamp: Optional[datetime.datetime] = None):
        """
        Initialize a reject event.
        
        Args:
            order_id: ID of the rejected order
            reason: Reason for rejection
            timestamp: Optional timestamp (defaults to now)
        """
        data = {
            'order_id': order_id,
            'reason': reason
        }
        
        super().__init__(EventType.REJECT, data, timestamp)
    
    def get_order_id(self) -> str:
        """Get the rejected order ID."""
        return self.data['order_id']
    
    def get_reason(self) -> str:
        """Get the rejection reason."""
        return self.data['reason']
    
    def __str__(self) -> str:
        """String representation of the reject event."""
        return f"RejectEvent(order_id: {self.get_order_id()}, reason: {self.get_reason()})"
    

# Example usage
if __name__ == "__main__":
    # Print all event types with descriptions
    for event_type, description in get_all_event_types_with_descriptions().items():
        print(f"{event_type}: {description}")
        
    # Get event type from string
    bar_event = EventType.from_string("BAR")
    print(f"Event type from string 'BAR': {bar_event}")
    
    # Get market data events
    market_data_events = EventType.market_data_events()
    print(f"Market data events: {[e.name for e in market_data_events]}")



    


