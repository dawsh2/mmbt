"""
Execution Engine for Trading System

This module handles order execution, position tracking, and portfolio management
within the backtesting system. It serves as a core component that processes orders,
manages positions, and tracks portfolio state throughout a backtest.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import numpy as np
from enum import Enum, auto

# Event types for the event system
class EventType(Enum):
    BAR = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()

class Event:
    """Base class for all events in the system."""
    def __init__(self, event_type: EventType, data: Any = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()

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
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.equity = initial_cash
        self.history = []
    
    def update_position(self, symbol: str, quantity_delta: float, price: float, timestamp: datetime):
        """Update a position in the portfolio."""
        # Adjust cash
        self.cash -= quantity_delta * price
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0, timestamp)
            
        # Update position
        position = self.positions[symbol]
        position.update(quantity_delta, price, timestamp)
        
        # If position is closed, remove it (optional)
        if position.quantity == 0:
            self.positions.pop(symbol)
            
        # Update overall equity
        self._update_equity()
    
    def mark_to_market(self, bar: Dict[str, Any]):
        """Mark all positions to market with latest prices."""
        # Loop through each symbol in the bar data
        for symbol, position in list(self.positions.items()):
            if isinstance(bar, dict) and symbol in bar:  # Multi-symbol bar data
                price = bar[symbol]['Close']
            else:  # Single symbol bar data
                price = bar['Close']
                
            position.mark_to_market(price)
            
        # Update overall equity
        self._update_equity()
    
    def _update_equity(self):
        """Update the portfolio equity value."""
        position_value = sum(p.market_value for p in self.positions.values())
        self.equity = self.cash + position_value
    
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
    
    def on_order(self, event: Event):
        """Handle incoming order events."""
        order = event.data
        self.pending_orders.append(order)
    
    def execute_pending_orders(self, bar: Dict[str, Any], market_simulator):
        """Execute any pending orders based on current bar data."""
        for order in list(self.pending_orders):
            # Apply market simulation effects (slippage, etc.)
            execution_price = market_simulator.calculate_execution_price(
                order, bar
            )
            
            # Execute the order
            fill = self._execute_order(order, execution_price, bar['timestamp'])
            
            # Record the fill
            self.trade_history.append(fill)
            
            # Remove from pending
            self.pending_orders.remove(order)
            
            # Emit fill event if event bus is available
            if self.event_bus:
                fill_event = Event(EventType.FILL, data=fill)
                self.event_bus.emit(fill_event)
    
    def _execute_order(self, order: Order, price: float, timestamp: datetime) -> Fill:
        """Execute a single order and update portfolio."""
        # Update portfolio with new position
        self.portfolio.update_position(
            order.symbol,
            order.quantity,
            price,
            timestamp
        )
        
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
        
        # Record portfolio state
        self.portfolio_history.append({
            'timestamp': bar['timestamp'],
            'equity': self.portfolio.equity,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.get_position_snapshot()
        })
    
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
        self.portfolio = Portfolio()
        self.pending_orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.signal_history = []
