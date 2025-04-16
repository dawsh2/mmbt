#!/usr/bin/env python3
# event_manager.py - Central component for managing event flow

import datetime
import logging
from typing import Dict, List, Any, Optional

from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType

# Set up logging
logger = logging.getLogger(__name__)


class EventManager:
    """
    Central manager for event flow between components in the trading system.
    Ensures proper event handling and data transformation between components.
    """
    
    def __init__(self, 
                 event_bus, 
                 strategy, 
                 position_manager, 
                 execution_engine,
                 portfolio=None):
        """
        Initialize the events manager with all required components.
        
        Args:
            event_bus: The central event bus for communication
            strategy: The trading strategy to use
            position_manager: The position manager
            execution_engine: The execution engine
            portfolio: Optional portfolio instance
        """
        self.event_bus = event_bus
        self.strategy = strategy
        self.position_manager = position_manager
        self.execution_engine = execution_engine
        self.portfolio = portfolio
        
        # Tracking variables
        self.price_history = {}  # Store price history by symbol
        self.signal_history = []
        self.orders_generated = []
        self.fills_processed = []
        
        # Event handlers
        self.handlers = {}
        
    def initialize(self):
        """Register all event handlers and initialize components."""
        # Register event handlers
        self._register_event_handlers()
        
        # Initialize components if they have initialize methods
        for component in [self.strategy, self.position_manager, 
                        self.execution_engine, self.portfolio]:
            if component and hasattr(component, 'initialize'):
                component.initialize()
                
        logger.info("Events manager initialized")
        
    def _register_event_handlers(self):
        """Register all necessary event handlers."""
        # Bar event handler
        self.handlers['bar'] = self._create_bar_handler()
        self.event_bus.register(EventType.BAR, self.handlers['bar'])
        
        # Signal event handler
        self.handlers['signal'] = self._create_signal_handler()
        self.event_bus.register(EventType.SIGNAL, self.handlers['signal'])
        
        # Order event handler
        self.handlers['order'] = self._create_order_handler()
        self.event_bus.register(EventType.ORDER, self.handlers['order'])
        
        # Fill event handler
        self.handlers['fill'] = self._create_fill_handler()
        self.event_bus.register(EventType.FILL, self.handlers['fill'])
        
        # Market open/close handlers
        self.event_bus.register(EventType.MARKET_OPEN, 
                              lambda e: logger.info(f"Market open: {e.timestamp if hasattr(e, 'timestamp') else datetime.datetime.now()}"))
        self.event_bus.register(EventType.MARKET_CLOSE, 
                              lambda e: logger.info(f"Market close: {e.timestamp if hasattr(e, 'timestamp') else datetime.datetime.now()}"))
                              
        logger.info("Event handlers registered")


    def _create_bar_handler(self):
        """Create a handler for BAR events."""
        def handle_bar_event(event):
            """Extract bar data properly for strategy consumption."""
            try:
                # Extract bar data from various event structures
                bar_data = None

                # Case 1: Event has bar data directly in data attribute
                if hasattr(event, 'data') and isinstance(event.data, dict) and 'Close' in event.data:
                    bar_data = event.data

                # Case 2: Event data is a BarEvent with a bar attribute
                elif hasattr(event, 'data') and hasattr(event.data, 'bar'):
                    bar_data = event.data.bar

                # Case 3: Event is a BarEvent itself
                elif hasattr(event, 'bar'):
                    bar_data = event.bar

                if bar_data is None:
                    logger.warning(f"Could not extract bar data from event: {event}")
                    return

                # Update price history
                symbol = bar_data.get('symbol', 'UNKNOWN')
                if symbol not in self.price_history:
                    self.price_history[symbol] = []

                self.price_history[symbol].append(bar_data)

                # Always wrap in a standard BarEvent
                if not isinstance(event.data, BarEvent):
                    bar_event = BarEvent(bar_data)
                    event_to_pass = Event(EventType.BAR, bar_event, event.timestamp)
                else:
                    event_to_pass = event

                # Process through strategy
                if hasattr(self.strategy, 'on_bar'):
                    self.strategy.on_bar(event_to_pass)
                elif hasattr(self.strategy, 'handle_event'):
                    self.strategy.handle_event(event_to_pass)

                logger.debug(f"Processed bar: {symbol} at {bar_data.get('timestamp')}")

            except Exception as e:
                logger.error(f"Error processing bar event: {str(e)}", exc_info=True)

        return handle_bar_event


    def _create_signal_handler(self):
        """Create a handler for SIGNAL events."""
        def handle_signal_event(event):
            """Process signals and create position actions."""
            try:
                signal = event.data

                # Store signal directly in history
                self.signal_history.append(signal)

                # Process through position manager
                if hasattr(self.position_manager, 'on_signal'):
                    actions = self.position_manager.on_signal(signal)
                    
                    # Execute position actions
                    if actions:
                        for action in actions:
                            self.position_manager.execute_position_action(
                                action=action,
                                current_time=signal_dict['timestamp']
                            )
                
                logger.info(f"Processed signal: {signal_dict['symbol']} "
                          f"type={signal_dict['signal_type']} price={signal_dict['price']}")
                          
            except Exception as e:
                logger.error(f"Error processing signal event: {str(e)}", exc_info=True)
        
        return handle_signal_event
    
    def _create_order_handler(self):
        """Create a handler for ORDER events."""
        def handle_order_event(event):
            """Process order events and send to execution engine."""
            try:
                order = event.data
                
                # Store order
                self.orders_generated.append(order)
                
                # Pass to execution engine
                if hasattr(self.execution_engine, 'on_order'):
                    self.execution_engine.on_order(event)
                
                logger.info(f"Processed order: {order.get('symbol', 'UNKNOWN')} "
                          f"quantity={order.get('quantity', 0)} "
                          f"price={order.get('price', 0)}")
                          
            except Exception as e:
                logger.error(f"Error processing order event: {str(e)}", exc_info=True)
        
        return handle_order_event
    
    def _create_fill_handler(self):
        """Create a handler for FILL events."""
        def handle_fill_event(event):
            """Process fill events and update portfolio."""
            try:
                fill = event.data
                
                # Store fill
                self.fills_processed.append(fill)
                
                # Update portfolio
                if self.portfolio and hasattr(self.portfolio, 'on_fill'):
                    self.portfolio.on_fill(event)
                
                logger.info(f"Processed fill: {fill.get('symbol', 'UNKNOWN')} "
                          f"quantity={fill.get('quantity', 0)} "
                          f"price={fill.get('fill_price', 0)}")
                          
            except Exception as e:
                logger.error(f"Error processing fill event: {str(e)}", exc_info=True)
        
        return handle_fill_event

    # In your event_manager.py

    def process_market_data(self, bar_data):
        """Process a single bar of market data.

        Args:
            bar_data: Dictionary or BarEvent containing market data
        """
        # Wrap in BarEvent if it's not already
        if not isinstance(bar_data, BarEvent) and isinstance(bar_data, dict):
            bar_event = BarEvent(bar_data)
        else:
            bar_event = bar_data

        # Create and emit a BAR event
        event = Event(EventType.BAR, bar_event)
        self.event_bus.emit(event)

    
 
    def get_status(self):
        """
        Get the current status of the trading system.
        
        Returns:
            dict: Status information
        """
        return {
            "price_history_length": {symbol: len(history) 
                                   for symbol, history in self.price_history.items()},
            "signals_generated": len(self.signal_history),
            "orders_generated": len(self.orders_generated),
            "fills_processed": len(self.fills_processed)
        }
    
    def reset(self):
        """Reset the events manager state."""
        # Reset internal state
        self.price_history = {}
        self.signal_history = []
        self.orders_generated = []
        self.fills_processed = []
        
        # Reset components
        for component in [self.strategy, self.position_manager, 
                        self.execution_engine, self.portfolio]:
            if component and hasattr(component, 'reset'):
                component.reset()
                
        logger.info("Events manager reset")
