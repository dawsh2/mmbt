"""
Execution Engine with Standardized Event Objects

This is a reference implementation of the ExecutionEngine class
that uses standardized event objects throughout.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union, Any

from src.position_management.position import Position
from src.position_management.portfolio import EventPortfolio
from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent, OrderEvent, FillEvent
from src.events.signal_event import SignalEvent

# Set up logging
logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Handles order execution, position tracking, and portfolio management.
    """
    
    def __init__(self, position_manager=None, market_simulator=None):
        """Initialize the execution engine."""
        self.position_manager = position_manager
        self.market_simulator = market_simulator
        self.portfolio = None 
        self.pending_orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.signal_history = []
        self.event_bus = None  # Will be set when registered with event bus
        self.last_known_prices = {}  # Cache for last known prices by symbol
        
        logger.info("Execution engine initialized")

    def on_signal(self, event):
        """
        Process a signal and convert to an order if appropriate.

        Args:
            event: Event containing a SignalEvent

        Returns:
            OrderEvent if order was created, None otherwise
        """
        # Extract the SignalEvent from the event
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event)}")
            return None
            
        if not isinstance(event.data, SignalEvent):
            logger.warning(f"Expected SignalEvent in event.data, got {type(event.data)}")
            # For backward compatibility
            if isinstance(event.data, dict) and 'signal_type' in event.data:
                # Try to create a SignalEvent
                try:
                    from src.events.signal_event import SignalEvent
                    signal = SignalEvent(
                        signal_value=event.data.get('signal_type'),
                        price=event.data.get('price', 0),
                        symbol=event.data.get('symbol', 'default'),
                        rule_id=event.data.get('rule_id'),
                        metadata=event.data.get('metadata', {})
                    )
                    logger.warning("Converted dictionary to SignalEvent (deprecated)")
                except Exception as e:
                    logger.error(f"Failed to convert dictionary to SignalEvent: {e}")
                    return None
            else:
                return None
        else:
            signal = event.data

        logger.debug(f"ExecutionEngine received signal: {signal.get_signal_name()}")

        # Skip neutral signals
        if signal.get_signal_value() == SignalEvent.NEUTRAL:
            logger.debug(f"Skipping neutral signal")
            return None

        # Store the signal for history
        self.signal_history.append(signal)

        # Extract signal data using proper getters
        symbol = signal.get_symbol()
        direction = signal.get_signal_value()  # BUY (1) or SELL (-1)
        price = signal.get_price()
        timestamp = signal.timestamp
        
        # Validate direction
        if direction not in [SignalEvent.BUY, SignalEvent.SELL]:
            logger.error(f"Invalid signal direction: {direction}")
            return None

        # Calculate position size if position manager is available
        quantity = 100  # Default quantity
        if self.position_manager and hasattr(self.position_manager, 'calculate_position_size'):
            size = self.position_manager.calculate_position_size(signal, self.portfolio, price)
            if size != 0:
                quantity = abs(size)

        # Create OrderEvent
        order = OrderEvent(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type="MARKET",
            timestamp=timestamp
        )

        # Add to pending orders
        self.pending_orders.append(order)

        logger.info(f"Created order from signal: {order}")

        # If the event bus is available, emit the order event
        if self.event_bus:
            self.event_bus.emit(Event(EventType.ORDER, order))

        return order
    
    def on_order(self, event):
        """
        Handle incoming order events.
        
        Args:
            event: Order event
        """
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event)}")
            return
            
        if not isinstance(event.data, OrderEvent):
            logger.warning(f"Expected OrderEvent in event.data, got {type(event.data)}")
            # Handle backward compatibility
            if isinstance(event.data, dict) and 'symbol' in event.data and 'quantity' in event.data:
                # Try to convert to OrderEvent
                try:
                    order = OrderEvent(
                        symbol=event.data.get('symbol', 'default'),
                        direction=event.data.get('direction', 1),
                        quantity=event.data.get('quantity', 0),
                        price=event.data.get('price'),
                        order_type=event.data.get('order_type', 'MARKET'),
                        order_id=event.data.get('order_id')
                    )
                    logger.warning("Converted dictionary to OrderEvent (deprecated)")
                except Exception as e:
                    logger.error(f"Failed to convert dictionary to OrderEvent: {e}")
                    return
            else:
                logger.error("Unable to process order event with invalid data")
                return
        else:
            order = event.data
        
        logger.info(f"Order received: {order}")
        
        # Add to pending orders
        self.pending_orders.append(order)
        
        # For market orders, execute immediately if possible
        if order.get_order_type() == 'MARKET':
            logger.info(f"Executing market order immediately: {order}")
            try:
                # Get current price (assuming last known price)
                symbol = order.get_symbol()
                price = self._get_last_known_price(symbol)
                
                if price:
                    fill = self._execute_order(order, price, datetime.datetime.now())
                    if fill:
                        # Emit FILL event
                        self._emit_fill_event(fill)
                        
                        # Remove from pending orders
                        if order in self.pending_orders:
                            self.pending_orders.remove(order)
                else:
                    logger.warning(f"No price available for {symbol}, order will be executed on next bar")
            except Exception as e:
                logger.error(f"Error executing market order immediately: {e}")
        
        logger.info(f"Added order to pending list: {order}")
    
    def execute_pending_orders(self, bar, market_simulator=None):
        """
        Execute any pending orders based on current bar data.
        
        Args:
            bar: Current market data (BarEvent or dict)
            market_simulator: Optional market simulator for calculating execution prices
            
        Returns:
            List of Fill objects for executed orders
        """
        if not self.pending_orders:
            return []

        # Use provided market simulator or instance variable
        simulator = market_simulator or self.market_simulator
        
        # Extract bar data properly
        if isinstance(bar, BarEvent):
            symbol = bar.get_symbol()
            close_price = bar.get_price()  # Gets Close price
            timestamp = bar.get_timestamp()
            bar_data = bar.get_data()
        elif isinstance(bar, dict) and 'Close' in bar:
            # Backward compatibility with dictionary-based bars
            logger.warning("Using deprecated dictionary-based bar data, convert to BarEvent")
            symbol = bar.get('symbol', 'default')
            close_price = bar.get('Close', 0)
            timestamp = bar.get('timestamp', datetime.datetime.now())
            bar_data = bar
        else:
            logger.error(f"Unsupported bar data type: {type(bar)}")
            return []
        
        # Update last known price
        if close_price > 0:
            self.last_known_prices[symbol] = close_price

        # Log the current state before execution
        logger.info(f"Processing {len(self.pending_orders)} pending orders for bar: {symbol}, close={close_price}")
        logger.info(f"Portfolio before execution - Cash: {self.portfolio.cash:.2f}, Equity: {self.portfolio.equity:.2f}")

        # Track which orders were successfully executed
        executed_orders = []
        fills = []

        # Process each pending order
        for order in list(self.pending_orders):  # Create a copy to safely modify during iteration
            logger.info(f"Attempting to execute order: {order}")

            # Skip orders for other symbols if bar data is symbol-specific
            order_symbol = order.get_symbol() if hasattr(order, 'get_symbol') else order.symbol
            if symbol != 'default' and symbol != order_symbol:
                logger.debug(f"Skipping order for {order_symbol} during {symbol} bar")
                continue

            # Get execution price from market simulator or use close price
            try:
                if simulator and hasattr(simulator, 'calculate_execution_price'):
                    execution_price = simulator.calculate_execution_price(order, bar_data)
                    logger.info(f"Market simulator calculated price: {execution_price}")
                else:
                    # Simple slippage model as fallback
                    execution_price = close_price
                    order_direction = order.get_direction() if hasattr(order, 'get_direction') else order.direction
                    slippage = execution_price * 0.001 * order_direction  # 0.1% slippage
                    execution_price += slippage
                    logger.info(f"Using fallback price with slippage: {execution_price}")
            except Exception as e:
                logger.error(f"Error calculating execution price: {e}")
                execution_price = close_price  # Fallback to close price

            # Calculate commission
            commission = 0.0
            if simulator and hasattr(simulator, 'calculate_fees'):
                try:
                    commission = simulator.calculate_fees(order, execution_price)
                except Exception as e:
                    logger.error(f"Error calculating fees: {e}")
                    order_quantity = order.get_quantity() if hasattr(order, 'get_quantity') else order.quantity
                    commission = order_quantity * execution_price * 0.001  # 10 bps default

            # Execute the order
            try:
                fill = self._execute_order(order, execution_price, timestamp, commission)
                if fill:
                    fills.append(fill)
                    self.trade_history.append(fill)
                    executed_orders.append(order)
                    
                    # Emit fill event
                    self._emit_fill_event(fill)
                    
                    logger.info(f"Order executed successfully, Fill price: {execution_price:.2f}")
                else:
                    logger.warning(f"Order execution returned no fill")
            except Exception as e:
                logger.error(f"Error executing order - DETAILS: {e}")
                import traceback
                traceback.print_exc()

        # Remove executed orders from pending list
        for order in executed_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)

        # Log the final state after execution
        logger.info(f"Executed {len(executed_orders)} of {len(self.pending_orders) + len(executed_orders)} orders")
        logger.info(f"Portfolio after execution - Cash: {self.portfolio.cash:.2f}, Equity: {self.portfolio.equity:.2f}")
        logger.info(f"Remaining pending orders: {len(self.pending_orders)}")

        return fills


    def _execute_order(self, order, price, timestamp, commission=0.0):
        """
        Execute a single order and update portfolio.

        Args:
            order: OrderEvent to execute
            price: Execution price
            timestamp: Execution timestamp
            commission: Commission cost

        Returns:
            FillEvent if successful
        """
        # Extract order details with proper getter method or attribute access
        symbol = order.get_symbol() if hasattr(order, 'get_symbol') else getattr(order, 'symbol', None)
        direction = order.get_direction() if hasattr(order, 'get_direction') else getattr(order, 'direction', 0)
        quantity = order.get_quantity() if hasattr(order, 'get_quantity') else getattr(order, 'quantity', 0)
        order_id = order.get_order_id() if hasattr(order, 'get_order_id') else getattr(order, 'order_id', None)

        # Validate order parameters
        if not symbol or direction == 0 or quantity == 0:
            logger.error(f"Invalid order parameters: symbol={symbol}, direction={direction}, quantity={quantity}")
            return None

        logger.info(f"Executing {direction > 0 and 'BUY' or 'SELL'} order: {quantity} shares at {price}")

        # Update portfolio with new position
        quantity_delta = quantity * direction
        success = self.portfolio.update_position(
            symbol,
            quantity_delta,
            price,
            timestamp
        )

        if not success:
            logger.error(f"Failed to execute order: Quantity: {quantity_delta}")
            return None

        # Create fill record
        fill = FillEvent(
            symbol=symbol,
            quantity=quantity,
            price=price,
            direction=direction,
            order_id=order_id,
            transaction_cost=commission,
            timestamp=timestamp
        )

        # If commission was charged, deduct from portfolio cash
        if commission > 0:
            self.portfolio.cash -= commission
            logger.info(f"Commission charged: {commission:.2f}, remaining cash: {self.portfolio.cash:.2f}")

        return fill
    

    
    def _emit_fill_event(self, fill):
        """Emit fill event to the event bus."""
        if self.event_bus:
            self.event_bus.emit(Event(EventType.FILL, fill))
            logger.info(f"Emitted FILL event: {fill}")
        else:
            logger.warning("No event bus available to emit FILL event")
    
    def _get_last_known_price(self, symbol):
        """Get the last known price for a symbol."""
        if symbol in self.last_known_prices:
            return self.last_known_prices[symbol]
        
        # Try to find from portfolio history
        if self.portfolio_history:
            for history_item in reversed(self.portfolio_history):
                positions = history_item.get('positions', {})
                if symbol in positions:
                    return positions[symbol].get('avg_price', None)
        
        return None
    
    def update(self, bar):
        """
        Update portfolio with latest market data.
        
        Args:
            bar: Current market data (BarEvent or dict)
        """
        # Extract data appropriately
        if isinstance(bar, BarEvent):
            timestamp = bar.get_timestamp()
        elif isinstance(bar, dict) and 'timestamp' in bar:
            timestamp = bar.get('timestamp')
        else:
            timestamp = datetime.datetime.now()
        
        # Mark-to-market all positions
        self.portfolio.mark_to_market(bar)
        
        # Record portfolio state
        portfolio_snapshot = {
            'timestamp': timestamp,
            'equity': self.portfolio.equity,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.get_position_snapshot()
        }
        
        self.portfolio_history.append(portfolio_snapshot)
    
    def get_trade_history(self):
        """Get the history of all executed trades."""
        return self.trade_history
    
    def get_portfolio_history(self):
        """Get the history of portfolio states."""
        return self.portfolio_history
    
    def get_signal_history(self):
        """Get the history of signals received."""
        return self.signal_history


    def on_position_action(self, event):
        """
        Handle position action events.

        Args:
            event: Event containing position action
        """
        # Verify this is an Event with position action data
        if not isinstance(event, Event) or not hasattr(event, 'data'):
            logger.error(f"Invalid position action event: {event}")
            return

        action = event.data

        # Only handle entry actions (entry actions become orders)
        if isinstance(action, dict) and action.get('action_type') == 'entry':
            # Extract data from action
            symbol = action.get('symbol')
            direction = action.get('direction', 0)
            size = action.get('size', 0)
            price = action.get('price', 0)

            # Create an order
            order = OrderEvent(
                symbol=symbol,
                direction=direction,
                quantity=size,
                price=price,
                order_type="MARKET"
            )

            # Log the order creation
            logger.info(f"Created order from position action: {symbol} {'BUY' if direction > 0 else 'SELL'} {size} @ {price}")

            # Process the order - either emit an event or handle directly
            order_event = Event(EventType.ORDER, order)

            # Either add to pending orders or emit event
            self.on_order(order_event)


    def reset(self):
        """Reset the execution engine state."""
        # Determine initial capital from portfolio if it exists
        initial_capital = 100000  # Default fallback value
        if hasattr(self, 'portfolio') and self.portfolio:
            if hasattr(self.portfolio, 'initial_capital'):
                initial_capital = self.portfolio.initial_capital

        # Store reference to event bus before resetting
        event_bus = None
        if hasattr(self, 'event_bus'):
            event_bus = self.event_bus

        # Create new portfolio or reset existing one
        if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'reset'):
            self.portfolio.reset()
        else:
            # Make sure we have an event bus to pass to the portfolio
            if event_bus is None:
                # Try to get event bus from position manager
                if self.position_manager and hasattr(self.position_manager, 'event_bus'):
                    event_bus = self.position_manager.event_bus
                else:
                    # Create a new event bus as a last resort
                    from src.events.event_bus import EventBus
                    event_bus = EventBus()
                    logger.warning("Creating new EventBus for portfolio in reset")

            # Create a new portfolio with the stored event bus
            from src.position_management.portfolio import EventPortfolio
            self.portfolio = EventPortfolio(initial_capital, event_bus)

        # Reset other state
        self.pending_orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.signal_history = []
        self.last_known_prices = {}

        logger.info(f"Execution engine reset with {initial_capital:.2f} capital")
