"""
Event Emitters Module

This module defines event emitters for generating events in the trading system.
It provides the EventEmitter mixin class and specialized emitter implementations.
"""

import uuid
import datetime
from typing import Dict, List, Optional, Union, Any, Set

from src.events.event_types import EventType
from src.events.event_bus import Event, EventBus


class EventEmitter:
    """
    Mixin class for components that emit events.
    
    This class provides a standard interface for emitting events
    to the event bus. It can be mixed into any class that needs
    to generate events.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize event emitter.
        
        Args:
            event_bus: Event bus to emit events to
        """
        self.event_bus = event_bus
    
    def emit(self, event_type: EventType, data: Any = None) -> Event:
        """
        Create and emit an event.
        
        Args:
            event_type: Type of event to emit
            data: Optional data payload
            
        Returns:
            The emitted event
        """
        event = Event(event_type, data)
        self.emit_event(event)
        return event
    
    def emit_event(self, event: Event) -> None:
        """
        Emit an existing event.
        
        Args:
            event: Event to emit
        """
        self.event_bus.emit(event)


class MarketDataEmitter(EventEmitter):
    """
    Event emitter for market data.
    
    This class emits bar, tick, and market open/close events.
    """
    
    def emit_bar(self, bar_data: Dict[str, Any]) -> Event:
        """
        Emit a bar event.
        
        Args:
            bar_data: Dictionary containing bar data
            
        Returns:
            The emitted event
        """
        return self.emit(EventType.BAR, bar_data)
    
    def emit_tick(self, tick_data: Dict[str, Any]) -> Event:
        """
        Emit a tick event.
        
        Args:
            tick_data: Dictionary containing tick data
            
        Returns:
            The emitted event
        """
        return self.emit(EventType.TICK, tick_data)
    
    def emit_market_open(self, timestamp: Optional[datetime.datetime] = None,
                       additional_data: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a market open event.
        
        Args:
            timestamp: Optional timestamp for the event
            additional_data: Optional additional data
            
        Returns:
            The emitted event
        """
        data = additional_data or {}
        data['timestamp'] = timestamp or datetime.datetime.now()
        return self.emit(EventType.MARKET_OPEN, data)
    
    def emit_market_close(self, timestamp: Optional[datetime.datetime] = None,
                        additional_data: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a market close event.
        
        Args:
            timestamp: Optional timestamp for the event
            additional_data: Optional additional data
            
        Returns:
            The emitted event
        """
        data = additional_data or {}
        data['timestamp'] = timestamp or datetime.datetime.now()
        return self.emit(EventType.MARKET_CLOSE, data)


class SignalEmitter(EventEmitter):
    """
    Event emitter for trading signals.
    
    This class emits signal events from strategies.
    """
    
    def emit_signal(self, symbol: str, signal_type: str, price: float,
                  confidence: float = 1.0, rule_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a signal event.
        
        Args:
            symbol: Instrument symbol
            signal_type: Signal type ('BUY', 'SELL', 'NEUTRAL')
            price: Price at signal generation
            confidence: Signal confidence (0-1)
            rule_id: Optional ID of the rule that generated the signal
            metadata: Optional additional signal data
            
        Returns:
            The emitted event
        """
        data = {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
            'confidence': confidence,
            'timestamp': datetime.datetime.now()
        }
        
        if rule_id:
            data['rule_id'] = rule_id
            
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.SIGNAL, data)


class OrderEmitter(EventEmitter):
    """
    Event emitter for order-related events.
    
    This class emits order, cancel, and modify events.
    """
    
    def emit_order(self, symbol: str, order_type: str, quantity: float,
                 direction: int, price: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit an order event.
        
        Args:
            symbol: Instrument symbol
            order_type: Order type ('MARKET', 'LIMIT', etc.)
            quantity: Order quantity
            direction: Order direction (1 for buy, -1 for sell)
            price: Optional price for limit orders
            metadata: Optional additional order data
            
        Returns:
            The emitted event
        """
        data = {
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'direction': direction,
            'timestamp': datetime.datetime.now(),
            'order_id': str(uuid.uuid4())
        }
        
        if price is not None:
            data['price'] = price
            
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.ORDER, data)
    
    def emit_cancel(self, order_id: str, reason: Optional[str] = None) -> Event:
        """
        Emit a cancel order event.
        
        Args:
            order_id: ID of order to cancel
            reason: Optional reason for cancellation
            
        Returns:
            The emitted event
        """
        data = {
            'order_id': order_id,
            'timestamp': datetime.datetime.now()
        }
        
        if reason:
            data['reason'] = reason
            
        return self.emit(EventType.CANCEL, data)
    
    def emit_modify(self, order_id: str, changes: Dict[str, Any]) -> Event:
        """
        Emit a modify order event.
        
        Args:
            order_id: ID of order to modify
            changes: Dictionary of changes to apply
            
        Returns:
            The emitted event
        """
        data = {
            'order_id': order_id,
            'changes': changes,
            'timestamp': datetime.datetime.now()
        }
        
        return self.emit(EventType.MODIFY, data)


class FillEmitter(EventEmitter):
    """
    Event emitter for fill-related events.
    
    This class emits fill and partial fill events.
    """
    
    def emit_fill(self, order_id: str, symbol: str, quantity: float,
                price: float, direction: int, transaction_cost: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a fill event.
        
        Args:
            order_id: ID of filled order
            symbol: Instrument symbol
            quantity: Filled quantity
            price: Fill price
            direction: Fill direction (1 for buy, -1 for sell)
            transaction_cost: Optional transaction cost
            metadata: Optional additional fill data
            
        Returns:
            The emitted event
        """
        data = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'direction': direction,
            'transaction_cost': transaction_cost,
            'timestamp': datetime.datetime.now()
        }
        
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.FILL, data)
    
    def emit_partial_fill(self, order_id: str, symbol: str, quantity: float,
                        price: float, direction: int, partial_quantity: float,
                        remaining_quantity: float, transaction_cost: float = 0.0,
                        metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a partial fill event.
        
        Args:
            order_id: ID of filled order
            symbol: Instrument symbol
            quantity: Original order quantity
            price: Fill price
            direction: Fill direction (1 for buy, -1 for sell)
            partial_quantity: Quantity filled in this partial fill
            remaining_quantity: Quantity remaining to be filled
            transaction_cost: Optional transaction cost
            metadata: Optional additional fill data
            
        Returns:
            The emitted event
        """
        data = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'direction': direction,
            'partial_quantity': partial_quantity,
            'remaining_quantity': remaining_quantity,
            'transaction_cost': transaction_cost,
            'timestamp': datetime.datetime.now()
        }
        
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.PARTIAL_FILL, data)
    
    def emit_reject(self, order_id: str, reason: str,
                  metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit an order rejection event.
        
        Args:
            order_id: ID of rejected order
            reason: Reason for rejection
            metadata: Optional additional rejection data
            
        Returns:
            The emitted event
        """
        data = {
            'order_id': order_id,
            'reason': reason,
            'timestamp': datetime.datetime.now()
        }
        
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.REJECT, data)


class PortfolioEmitter(EventEmitter):
    """
    Event emitter for portfolio-related events.
    
    This class emits position opened, closed, and modified events.
    """
    
    def emit_position_opened(self, symbol: str, quantity: float,
                           entry_price: float, direction: int,
                           metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a position opened event.
        
        Args:
            symbol: Instrument symbol
            quantity: Position quantity
            entry_price: Entry price
            direction: Position direction (1 for long, -1 for short)
            metadata: Optional additional position data
            
        Returns:
            The emitted event
        """
        data = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'direction': direction,
            'timestamp': datetime.datetime.now(),
            'position_id': str(uuid.uuid4())
        }
        
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.POSITION_OPENED, data)
    
    def emit_position_closed(self, position_id: str, symbol: str,
                           exit_price: float, pnl: float,
                           metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a position closed event.
        
        Args:
            position_id: ID of closed position
            symbol: Instrument symbol
            exit_price: Exit price
            pnl: Profit/loss from the position
            metadata: Optional additional position data
            
        Returns:
            The emitted event
        """
        data = {
            'position_id': position_id,
            'symbol': symbol,
            'exit_price': exit_price,
            'pnl': pnl,
            'timestamp': datetime.datetime.now()
        }
        
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.POSITION_CLOSED, data)
    
    def emit_position_modified(self, position_id: str, symbol: str,
                             changes: Dict[str, Any]) -> Event:
        """
        Emit a position modified event.
        
        Args:
            position_id: ID of modified position
            symbol: Instrument symbol
            changes: Dictionary of changes made
            
        Returns:
            The emitted event
        """
        data = {
            'position_id': position_id,
            'symbol': symbol,
            'changes': changes,
            'timestamp': datetime.datetime.now()
        }
        
        return self.emit(EventType.POSITION_MODIFIED, data)


class SystemEmitter(EventEmitter):
    """
    Event emitter for system-related events.
    
    This class emits system start, stop, pause, resume, and error events.
    """
    
    def emit_start(self, component_name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a system start event.
        
        Args:
            component_name: Optional name of the starting component
            metadata: Optional additional start data
            
        Returns:
            The emitted event
        """
        data = {'timestamp': datetime.datetime.now()}
        
        if component_name:
            data['component'] = component_name
            
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.START, data)
    
    def emit_stop(self, component_name: Optional[str] = None,
                reason: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a system stop event.
        
        Args:
            component_name: Optional name of the stopping component
            reason: Optional reason for stopping
            metadata: Optional additional stop data
            
        Returns:
            The emitted event
        """
        data = {'timestamp': datetime.datetime.now()}
        
        if component_name:
            data['component'] = component_name
            
        if reason:
            data['reason'] = reason
            
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.STOP, data)
    
    def emit_error(self, error_message: str, component_name: Optional[str] = None,
                 exception: Optional[Exception] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Event:
        """
        Emit a system error event.
        
        Args:
            error_message: Error message
            component_name: Optional name of the component with error
            exception: Optional exception object
            metadata: Optional additional error data
            
        Returns:
            The emitted event
        """
        data = {
            'message': error_message,
            'timestamp': datetime.datetime.now()
        }
        
        if component_name:
            data['component'] = component_name
            
        if exception:
            data['exception'] = {
                'type': type(exception).__name__,
                'str': str(exception)
            }
            
        if metadata:
            data['metadata'] = metadata
            
        return self.emit(EventType.ERROR, data)


# Example usage
if __name__ == "__main__":
    from event_bus import EventBus, Event
    from event_handlers import LoggingHandler
    
    # Create event bus
    event_bus = EventBus()
    
    # Create logging handler
    logger = LoggingHandler(list(EventType))
    
    # Register handler for all event types
    for event_type in EventType:
        event_bus.register(event_type, logger)
    
    # Create emitters
    market_data_emitter = MarketDataEmitter(event_bus)
    signal_emitter = SignalEmitter(event_bus)
    order_emitter = OrderEmitter(event_bus)
    
    # Emit events
    market_data_emitter.emit_bar({
        'symbol': 'AAPL',
        'open': 150.0,
        'high': 151.5,
        'low': 149.5,
        'close': 151.0,
        'volume': 1000000
    })
    
    signal_emitter.emit_signal(
        symbol='AAPL',
        signal_type='BUY',
        price=151.0,
        confidence=0.8,
        rule_id='sma_crossover',
        metadata={'ma_fast': 10, 'ma_slow': 30}
    )
    
    order_emitter.emit_order(
        symbol='AAPL',
        order_type='MARKET',
        quantity=100,
        direction=1,
        metadata={'strategy': 'trend_following'}
    )
