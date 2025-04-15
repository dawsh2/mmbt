"""
Execution Engine for Trading System - Fixed version with improved debugging

This module handles order execution, position tracking, and portfolio management
within the backtesting system. It serves as a core component that processes orders,
manages positions, and tracks portfolio state throughout a backtest.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
from enum import Enum, auto
import logging

# Get logger
logger = logging.getLogger(__name__)

# Import EventType from events module
from src.events.event_types import EventType
# Import Event from events module
from src.events.event_bus import Event



class Order:
    """Represents a trading order."""
    def __init__(
        self, 
        symbol: str, 
        order_type: str,
        quantity: float, 
        direction: int, 
        timestamp: datetime
    ):
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.timestamp = timestamp
    
    def __str__(self):
        return f"Order(symbol={self.symbol}, type={self.order_type}, qty={self.quantity}, dir={self.direction})"

class Fill:
    """Represents an order fill (execution)."""
    def __init__(
        self, 
        order: Order, 
        fill_price: float, 
        timestamp: datetime,
        commission: float = 0.0
    ):
        self.order = order
        self.fill_price = fill_price
        self.timestamp = timestamp
        self.commission = commission
        
        # Add these fields to make it more accessible
        self.symbol = order.symbol
        self.quantity = order.quantity
        self.direction = order.direction
    
    def __str__(self):
        return f"Fill(symbol={self.symbol}, qty={self.quantity}, dir={self.direction}, price={self.fill_price})"

class Position:
    """
    Represents a single position in a financial instrument.
    """
    
    def __init__(self, symbol: str, quantity: float = 0, avg_price: float = 0, timestamp: Optional[datetime] = None):
        """Initialize a position."""
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.cost_basis = quantity * avg_price if quantity else 0
        self.market_value = self.cost_basis
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.entry_timestamp = timestamp
        self.last_update_timestamp = timestamp
    
    def update(self, quantity_delta: float, price: float, timestamp: datetime):
        """Update the position with a new trade."""
        if quantity_delta == 0:
            return
            
        # Calculate cost of new shares
        new_cost = quantity_delta * price
        
        # If closing or reducing position, calculate realized P&L
        if (self.quantity > 0 and quantity_delta < 0) or (self.quantity < 0 and quantity_delta > 0):
            # Realized P&L from this trade (average price method)
            cost_per_share = self.avg_price if self.quantity != 0 else 0
            realized_pnl_delta = -quantity_delta * (price - cost_per_share)
            
            # Close entire position
            if abs(quantity_delta) >= abs(self.quantity):
                realized_pnl_delta = self.quantity * (price - cost_per_share)
                self.realized_pnl += realized_pnl_delta
                self.quantity = 0
                self.cost_basis = 0
                self.avg_price = 0
                
            # Partial close
            else:
                self.realized_pnl += realized_pnl_delta
                self.quantity += quantity_delta
                self.cost_basis = self.quantity * self.avg_price
                
        # If opening or adding to position, update average price
        else:
            total_cost = self.cost_basis + new_cost
            self.quantity += quantity_delta
            if self.quantity != 0:
                self.avg_price = total_cost / self.quantity
            else:
                self.avg_price = 0
            self.cost_basis = self.quantity * self.avg_price
            
        self.last_update_timestamp = timestamp
        
        # Log the position update
        logger.debug(f"Position updated: {self.symbol}, Qty: {self.quantity}, Avg Price: {self.avg_price:.2f}")
            
    def mark_to_market(self, price: float):
        """Mark position to market at current price."""
        self.market_value = self.quantity * price
        self.unrealized_pnl = self.market_value - self.cost_basis
        
    def get_info(self) -> Dict[str, Any]:
        """Get position information dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'cost_basis': self.cost_basis,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'entry_timestamp': self.entry_timestamp,
            'last_update_timestamp': self.last_update_timestamp
        }

class Portfolio:
    """
    Manages a collection of positions and overall portfolio state.
    """
    
    def __init__(self, initial_cash: float = 100000):
        """Initialize the portfolio."""
        self.cash = initial_cash
        self.initial_capital = initial_cash
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.equity = initial_cash
        self.history = []
        
        logger.info(f"Portfolio initialized with {initial_cash:.2f} cash")
    
    def update_position(self, symbol: str, quantity_delta: float, price: float, timestamp: datetime):
        """Update a position in the portfolio."""
        # Log the trade
        direction = "BUY" if quantity_delta > 0 else "SELL"
        logger.info(f"Trade: {direction} {abs(quantity_delta)} {symbol} @ {price:.2f}")
        
        # Calculate trade cost
        trade_cost = quantity_delta * price
        
        # Check if we have enough cash (for buys)
        if trade_cost > 0 and trade_cost > self.cash:
            logger.warning(f"Not enough cash for trade: {trade_cost:.2f} needed, {self.cash:.2f} available")
            # You might want to adjust the quantity here instead of rejecting
            return False
        
        # Adjust cash
        self.cash -= trade_cost
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0, timestamp)
            
        # Update position
        position = self.positions[symbol]
        position.update(quantity_delta, price, timestamp)
        
        # If position is closed, remove it (optional)
        if position.quantity == 0:
            self.positions.pop(symbol)
            logger.debug(f"Position closed and removed: {symbol}")
            
        # Update overall equity
        self._update_equity()
        
        return True
    
    def mark_to_market(self, bar_data):
        """Mark all positions to market with latest prices."""
        # Handle different bar data formats
        if isinstance(bar_data, dict) and 'Close' in bar_data:
            # Single symbol bar
            price = bar_data['Close']
            symbol = bar_data.get('symbol', 'default')
            
            # Mark positions with this price
            if symbol in self.positions:
                self.positions[symbol].mark_to_market(price)
                
        elif hasattr(bar_data, 'get') and bar_data.get('Close') is not None:
            # Another form of single symbol bar
            price = bar_data.get('Close')
            symbol = bar_data.get('symbol', 'default')
            
            # Mark positions with this price
            if symbol in self.positions:
                self.positions[symbol].mark_to_market(price)
                
        elif hasattr(bar_data, 'bar') and hasattr(bar_data.bar, 'get'):
            # Event with bar attribute
            bar = bar_data.bar
            price = bar.get('Close')
            symbol = bar.get('symbol', 'default')
            
            # Mark positions with this price
            if symbol in self.positions:
                self.positions[symbol].mark_to_market(price)
                
        else:
            # Try to mark all positions with a single price (for testing)
            for symbol, position in self.positions.items():
                if hasattr(bar_data, symbol) and hasattr(bar_data[symbol], 'Close'):
                    # Multi-symbol format
                    price = bar_data[symbol]['Close']
                    position.mark_to_market(price)
                elif hasattr(bar_data, 'Close'):
                    # Assume same price for all positions
                    price = bar_data['Close']
                    position.mark_to_market(price)
            
        # Update overall equity
        self._update_equity()
    
    def _update_equity(self):
        """Update the portfolio equity value."""
        position_value = sum(p.market_value for p in self.positions.values())
        self.equity = self.cash + position_value
        
        # Debug log
        logger.debug(f"Portfolio equity: {self.equity:.2f} (Cash: {self.cash:.2f}, Positions: {position_value:.2f})")
    
    def get_position_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions as a dictionary."""
        return {symbol: pos.get_info() for symbol, pos in self.positions.items()}


class ExecutionEngine:
    """
    Handles order execution, position tracking, and portfolio management.
    """
    
    def __init__(self, position_manager=None):
        """Initialize the execution engine."""
        self.position_manager = position_manager
        self.portfolio = Portfolio()
        self.pending_orders: List[Order] = []
        self.trade_history: List[Fill] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.signal_history = []
        self.event_bus = None  # Will be set by Backtester
        
        logger.info("Execution engine initialized")
    
    def on_order(self, event: Event):
        """Handle incoming order events."""
        order = event.data
        self.pending_orders.append(order)
        logger.debug(f"Order received: {order}")

    def on_signal(self, signal):
        """
        Process a signal and convert to an order if appropriate.

        Args:
            signal: Signal object to process

        Returns:
            Order object if order was created, None otherwise
        """
        # Skip neutral signals
        if signal.signal_type == SignalType.NEUTRAL:
            logger.debug(f"Skipping neutral signal: {signal}")
            return None

        # Store the signal for history
        self.signal_history.append(signal)

        # Extract symbol from signal
        symbol = getattr(signal, 'symbol', 'default')

        # Set direction based on signal type
        direction = 1 if signal.signal_type == SignalType.BUY else -1

        # Calculate position size if position manager is available
        quantity = 100  # Default quantity
        if self.position_manager and hasattr(self.position_manager, 'calculate_position_size'):
            size = self.position_manager.calculate_position_size(signal, self.portfolio)
            if size != 0:
                quantity = abs(size)

        # Create order
        order = Order(
            symbol=symbol,
            order_type="MARKET",
            quantity=quantity,
            direction=direction,
            timestamp=signal.timestamp if hasattr(signal, 'timestamp') else datetime.now()
        )

        # Add to pending orders
        self.pending_orders.append(order)

        logger.info(f"Created order from signal: {order}")

        return order

    def execute_pending_orders(self, bar: Dict[str, Any], market_simulator):
        """Execute any pending orders based on current bar data."""
        if not self.pending_orders:
            return

        # Log the current bar
        symbol = bar.get('symbol', 'default') if hasattr(bar, 'get') else 'default'
        close_price = bar.get('Close', 0) if hasattr(bar, 'get') else 0

        logger.info(f"Processing {len(self.pending_orders)} pending orders for bar: {symbol}, close={close_price}")

        # Track which orders were executed
        executed_orders = []
        fills = []

        for order in self.pending_orders:
            logger.info(f"Attempting to execute order: {order}")

            # Get execution price from market simulator
            if market_simulator and hasattr(market_simulator, 'calculate_execution_price'):
                try:
                    execution_price = market_simulator.calculate_execution_price(order, bar)
                    logger.info(f"Market simulator price: {execution_price}")
                except Exception as e:
                    logger.error(f"Error calculating execution price: {e}")
                    execution_price = close_price  # Fallback to close price
            else:
                # Simple slippage model as fallback
                execution_price = close_price
                # Add a small slippage (0.1%)
                slippage = execution_price * 0.001 * (1 if order.direction > 0 else -1)
                execution_price += slippage
                logger.info(f"Using fallback price with slippage: {execution_price}")

            # Get timestamp
            timestamp = bar.get('timestamp', datetime.now()) if hasattr(bar, 'get') else datetime.now()

            # Execute the order
            try:
                fill = self._execute_order(order, execution_price, timestamp)
                fills.append(fill)
                executed_orders.append(order)
                logger.info(f"Order executed successfully: {order}, Fill price: {execution_price:.2f}")
            except Exception as e:
                logger.error(f"Error executing order - DETAILS: {e}")
                logger.error(f"Order: {order}, Direction: {order.direction}, Quantity: {order.quantity}")
                logger.error(f"Portfolio cash: {self.portfolio.cash}, Current equity: {self.portfolio.equity}")

    def _execute_order(self, order: Order, price: float, timestamp: datetime) -> Fill:
        """Execute a single order and update portfolio."""
        # For buy orders, use positive quantity
        # For sell orders, use negative quantity
        quantity_delta = order.quantity * order.direction

        logger.info(f"Executing {order.direction > 0 and 'BUY' or 'SELL'} order: {abs(quantity_delta)} shares at {price}")

        # Update portfolio with new position
        success = self.portfolio.update_position(
            order.symbol,
            quantity_delta,
            price,
            timestamp
        )

        if not success:
            logger.error(f"Failed to execute order: {order}, Quantity: {quantity_delta}")
            logger.error(f"Portfolio state: Cash={self.portfolio.cash}, Equity={self.portfolio.equity}")
            logger.error(f"Current positions: {self.portfolio.get_position_snapshot()}")
            raise ValueError(f"Order execution failed for {order}")

        # Create fill record
        fill = Fill(
            order=order,
            fill_price=price,
            timestamp=timestamp
        )

        return fill


    def update(self, bar: Dict[str, Any]):
        """Update portfolio with latest market data."""
        # Mark-to-market all positions
        self.portfolio.mark_to_market(bar)
        
        # Get timestamp from bar
        timestamp = bar.get('timestamp', datetime.now()) if hasattr(bar, 'get') else datetime.now()
        
        # Record portfolio state
        portfolio_snapshot = {
            'timestamp': timestamp,
            'equity': self.portfolio.equity,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.get_position_snapshot()
        }
        
        self.portfolio_history.append(portfolio_snapshot)
    
    def get_trade_history(self) -> List[Fill]:
        """Get the history of all executed trades."""
        return self.trade_history
    
    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """Get the history of portfolio states."""
        return self.portfolio_history
    
    def get_signal_history(self) -> List[Any]:
        """Get the history of signals received."""
        return self.signal_history
    
    def reset(self):
        """Reset the execution engine state."""
        initial_capital = self.portfolio.initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.pending_orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.signal_history = []
        logger.info(f"Execution engine reset with {initial_capital:.2f} capital")
