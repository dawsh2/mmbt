"""
Event Emitters Module

This module defines event emitters for generating events in the trading system.
It provides the EventEmitter mixin class and specialized emitter implementations.
"""

import uuid
import datetime
from typing import Dict, List, Optional, Union, Any, Set

from src.events.event_types import EventType
from src.events.event_base import Event
from src.events.signal_event import SignalEvent


class EventEmitter:
    """
    Mixin class for components that emit events.
    
    This class provides a standard interface for emitting events
    to the event bus. It can be mixed into any class that needs
    to generate events.
    """
    
    def __init__(self, event_bus):
        """
        Initialize event emitter.
        
        Args:
            event_bus: Event bus to emit events to
        """
        self.event_bus = event_bus
    
    def emit(self, event_type: EventType, event_object) -> Event:
        """
        Create and emit an event.
        
        Args:
            event_type: Type of event to emit
            event_object: Event object to emit
            
        Returns:
            The emitted event
        """
        # Ensure event_object is not a dictionary
        if isinstance(event_object, dict):
            raise TypeError("Event object cannot be a dictionary. Use appropriate event class.")
            
        # Create Event if event_object is not already an Event
        if not isinstance(event_object, Event):
            event = Event(event_type, event_object)
        else:
            # If it's already an Event, use it directly
            event = event_object
            
        self.emit_event(event)
        return event
    
    def emit_event(self, event: Event) -> None:
        """
        Emit an existing event.
        
        Args:
            event: Event to emit
        """
        if not isinstance(event, Event):
            raise TypeError(f"Expected Event object, got {type(event).__name__}")
            
        self.event_bus.emit(event)


class MarketDataEmitter(EventEmitter):
    """
    Event emitter for market data.
    
    This class emits bar, tick, and market open/close events.
    
    Attributes:
        default_symbol (str, optional): Default symbol to use when bar data doesn't include one
    """
    
    def __init__(self, event_bus, default_symbol=None):
        """
        Initialize the MarketDataEmitter.
        
        Args:
            event_bus (EventBus): Event bus to emit events to
            default_symbol (str, optional): Default symbol to use if not present in bar data
        """
        super().__init__(event_bus)
        self.default_symbol = default_symbol
    
    def emit_bar(self, bar_event) -> Event:
        """
        Emit a bar event.
        
        Args:
            bar_event: BarEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure bar_event is a BarEvent
        from src.events.event_types import BarEvent
        if not isinstance(bar_event, BarEvent):
            raise TypeError(f"Expected BarEvent object, got {type(bar_event).__name__}")
            
        # Set default symbol if not present
        if not bar_event.get_symbol() and self.default_symbol:
            # Create a new BarEvent with default symbol if needed
            # This assumes BarEvent has copy or similar functionality
            # You may need to adjust based on your actual implementation
            bar_data = bar_event.data.copy() if isinstance(bar_event.data, dict) else {}
            bar_data['symbol'] = self.default_symbol
            bar_event = BarEvent(bar_data, bar_event.timestamp)
        
        # Emit the event
        return self.emit(EventType.BAR, bar_event)
    
    def emit_tick(self, tick_event) -> Event:
        """
        Emit a tick event.
        
        Args:
            tick_event: TickEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure tick_event is a proper event object, not a dictionary
        if isinstance(tick_event, dict):
            raise TypeError("Expected TickEvent object, got dict. Use TickEvent class instead.")
            
        return self.emit(EventType.TICK, tick_event)
    
    def emit_market_open(self, market_open_event) -> Event:
        """
        Emit a market open event.
        
        Args:
            market_open_event: MarketOpenEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure proper event object
        if isinstance(market_open_event, dict):
            raise TypeError("Expected MarketOpenEvent object, got dict. Use MarketOpenEvent class instead.")
            
        return self.emit(EventType.MARKET_OPEN, market_open_event)
    
    def emit_market_close(self, market_close_event) -> Event:
        """
        Emit a market close event.
        
        Args:
            market_close_event: MarketCloseEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure proper event object
        if isinstance(market_close_event, dict):
            raise TypeError("Expected MarketCloseEvent object, got dict. Use MarketCloseEvent class instead.")
            
        return self.emit(EventType.MARKET_CLOSE, market_close_event)
class SignalEmitter(EventEmitter):
    """
    Event emitter for trading signals.
    
    This class emits signal events from strategies.
    """
    
    def emit_signal(self, signal_event) -> Event:
        """
        Emit a signal event.
        
        Args:
            signal_event: SignalEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure signal_event is a SignalEvent
        if not isinstance(signal_event, SignalEvent):
            raise TypeError(f"Expected SignalEvent object, got {type(signal_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.SIGNAL, signal_event)


class OrderEmitter(EventEmitter):
    """
    Event emitter for order-related events.
    
    This class emits order, cancel, and modify events.
    """
    
    def emit_order(self, order_event) -> Event:
        """
        Emit an order event.
        
        Args:
            order_event: OrderEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure order_event is an OrderEvent
        from src.events.order_events import OrderEvent  # Assuming you have this class
        if not isinstance(order_event, OrderEvent):
            raise TypeError(f"Expected OrderEvent object, got {type(order_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.ORDER, order_event)
    
    def emit_cancel(self, cancel_event) -> Event:
        """
        Emit a cancel order event.
        
        Args:
            cancel_event: CancelOrderEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure cancel_event is a CancelOrderEvent
        from src.events.order_events import CancelOrderEvent  # Assuming you have this class
        if not isinstance(cancel_event, CancelOrderEvent):
            raise TypeError(f"Expected CancelOrderEvent object, got {type(cancel_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.CANCEL, cancel_event)
    
    def emit_modify(self, modify_event) -> Event:
        """
        Emit a modify order event.
        
        Args:
            modify_event: ModifyOrderEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure modify_event is a ModifyOrderEvent
        from src.events.order_events import ModifyOrderEvent  # Assuming you have this class
        if not isinstance(modify_event, ModifyOrderEvent):
            raise TypeError(f"Expected ModifyOrderEvent object, got {type(modify_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.MODIFY, modify_event)    


class FillEmitter(EventEmitter):
    """
    Event emitter for fill-related events.
    
    This class emits fill and partial fill events.
    """
    
    def emit_fill(self, fill_event) -> Event:
        """
        Emit a fill event.
        
        Args:
            fill_event: FillEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure fill_event is a FillEvent
        from src.events.order_events import FillEvent  # Assuming you have this class
        if not isinstance(fill_event, FillEvent):
            raise TypeError(f"Expected FillEvent object, got {type(fill_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.FILL, fill_event)
    
    def emit_partial_fill(self, partial_fill_event) -> Event:
        """
        Emit a partial fill event.
        
        Args:
            partial_fill_event: PartialFillEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure partial_fill_event is a PartialFillEvent
        from src.events.order_events import PartialFillEvent  # Assuming you have this class
        if not isinstance(partial_fill_event, PartialFillEvent):
            raise TypeError(f"Expected PartialFillEvent object, got {type(partial_fill_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.PARTIAL_FILL, partial_fill_event)
    
    def emit_reject(self, reject_event) -> Event:
        """
        Emit an order rejection event.
        
        Args:
            reject_event: RejectEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure reject_event is a RejectEvent
        from src.events.order_events import RejectEvent  # Assuming you have this class
        if not isinstance(reject_event, RejectEvent):
            raise TypeError(f"Expected RejectEvent object, got {type(reject_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.REJECT, reject_event)



class PortfolioEmitter(EventEmitter):
    """
    Event emitter for portfolio-related events.
    
    This class emits position opened, closed, and modified events.
    """
    
    def emit_position_opened(self, position_opened_event) -> Event:
        """
        Emit a position opened event.
        
        Args:
            position_opened_event: PositionOpenedEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure position_opened_event is a PositionOpenedEvent
        from src.events.portfolio_events import PositionOpenedEvent
        if not isinstance(position_opened_event, PositionOpenedEvent):
            raise TypeError(f"Expected PositionOpenedEvent object, got {type(position_opened_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.POSITION_OPENED, position_opened_event)
    
    def emit_position_closed(self, position_closed_event) -> Event:
        """
        Emit a position closed event.
        
        Args:
            position_closed_event: PositionClosedEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure position_closed_event is a PositionClosedEvent
        from src.events.portfolio_events import PositionClosedEvent
        if not isinstance(position_closed_event, PositionClosedEvent):
            raise TypeError(f"Expected PositionClosedEvent object, got {type(position_closed_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.POSITION_CLOSED, position_closed_event)
    
    def emit_position_modified(self, position_modified_event) -> Event:
        """
        Emit a position modified event.
        
        Args:
            position_modified_event: PositionModifiedEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure position_modified_event is a PositionModifiedEvent
        from src.events.portfolio_events import PositionModifiedEvent
        if not isinstance(position_modified_event, PositionModifiedEvent):
            raise TypeError(f"Expected PositionModifiedEvent object, got {type(position_modified_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.POSITION_MODIFIED, position_modified_event)



class SystemEmitter(EventEmitter):
    """
    Event emitter for system-related events.
    
    This class emits system start, stop, pause, resume, and error events.
    """
    
    def emit_start(self, start_event) -> Event:
        """
        Emit a system start event.
        
        Args:
            start_event: StartEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure start_event is a StartEvent
        from src.events.system_events import StartEvent  # Assuming you have this class
        if not isinstance(start_event, StartEvent):
            raise TypeError(f"Expected StartEvent object, got {type(start_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.START, start_event)
    
    def emit_stop(self, stop_event) -> Event:
        """
        Emit a system stop event.
        
        Args:
            stop_event: StopEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure stop_event is a StopEvent
        from src.events.system_events import StopEvent  # Assuming you have this class
        if not isinstance(stop_event, StopEvent):
            raise TypeError(f"Expected StopEvent object, got {type(stop_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.STOP, stop_event)
    
    def emit_error(self, error_event) -> Event:
        """
        Emit a system error event.
        
        Args:
            error_event: ErrorEvent object to emit
            
        Returns:
            The emitted event
        """
        # Ensure error_event is an ErrorEvent
        from src.events.system_events import ErrorEvent  # Assuming you have this class
        if not isinstance(error_event, ErrorEvent):
            raise TypeError(f"Expected ErrorEvent object, got {type(error_event).__name__}")
            
        # Emit the event
        return self.emit(EventType.ERROR, error_event)    
