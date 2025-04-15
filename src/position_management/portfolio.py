"""
Portfolio Module

This module defines the Portfolio class for managing multiple positions and tracking
overall account performance.
"""

import uuid
import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from collections import defaultdict

from src.position_management.position import Position, PositionStatus, PositionFactory, EntryType, ExitType

# Set up logging
logger = logging.getLogger(__name__)


class Portfolio:
    """
    Represents a portfolio of positions.
    
    A portfolio manages multiple positions across different instruments and tracks
    overall account performance.
    """
    
    def __init__(self, initial_capital: float, portfolio_id: Optional[str] = None,
                name: Optional[str] = None, currency: str = 'USD',
                allow_fractional_shares: bool = True,
                margin_enabled: bool = False, leverage: float = 1.0):
        """
        Initialize a portfolio.
        
        Args:
            initial_capital: Initial account capital
            portfolio_id: Unique portfolio ID (generated if None)
            name: Optional portfolio name
            currency: Portfolio base currency
            allow_fractional_shares: Whether fractional shares are allowed
            margin_enabled: Whether margin trading is enabled
            leverage: Maximum allowed leverage
        """
        self.portfolio_id = portfolio_id or str(uuid.uuid4())
        self.name = name or f"Portfolio_{self.portfolio_id[:8]}"
        self.currency = currency
        self.allow_fractional_shares = allow_fractional_shares
        self.margin_enabled = margin_enabled
        self.leverage = leverage
        
        # Capital and margin
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity = initial_capital
        self.margin_used = 0.0
        self.margin_available = initial_capital if margin_enabled else 0.0
        
        # Positions
        self.positions = {}  # Currently open positions by ID
        self.positions_by_symbol = defaultdict(list)  # Open positions by symbol
        self.closed_positions = {}  # Historical closed positions
        
        # Performance tracking
        self.high_water_mark = initial_capital
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # History
        self.equity_history = [(datetime.datetime.now(), initial_capital)]
        self.cash_history = [(datetime.datetime.now(), initial_capital)]
        self.trades_history = []
        
        logger.info(f"Portfolio initialized: {self.name} with {self.initial_capital} {self.currency}")
    
    def open_position(self, symbol: str, direction: int, quantity: float, 
                     entry_price: float, entry_time: datetime.datetime,
                     entry_type: EntryType = EntryType.MARKET,
                     strategy_id: Optional[str] = None,
                     entry_order_id: Optional[str] = None,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     transaction_cost: float = 0.0,
                     metadata: Optional[Dict[str, Any]] = None) -> Position:
        """
        Open a new position in the portfolio.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            quantity: Position size in units
            entry_price: Entry price
            entry_time: Entry timestamp
            entry_type: Type of entry (market, limit, etc.)
            strategy_id: ID of the strategy that created the position
            entry_order_id: ID of the order that opened the position
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            transaction_cost: Transaction cost for opening the position
            metadata: Optional additional position metadata
            
        Returns:
            The newly created Position object
            
        Raises:
            ValueError: If position cannot be opened due to insufficient capital
        """
        # Check if fractional shares are allowed
        if not self.allow_fractional_shares and quantity != int(quantity):
            quantity = int(quantity)
            logger.warning(f"Adjusted quantity to {quantity} as fractional shares are not allowed")
        
        # Calculate position value
        position_value = quantity * entry_price
        
        # Check if margin is required
        required_margin = position_value
        if self.margin_enabled and direction < 0:  # Short requires margin
            required_margin = position_value / self.leverage
        
        # Check if we have enough capital
        if required_margin > self.cash + self.margin_available:
            raise ValueError(f"Insufficient capital to open position. " 
                            f"Required: {required_margin}, Available: {self.cash + self.margin_available}")
        
        # Create position
        position = PositionFactory.create_position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            entry_type=entry_type,
            strategy_id=strategy_id,
            entry_order_id=entry_order_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
        
        # Update portfolio
        self.positions[position.position_id] = position
        self.positions_by_symbol[symbol].append(position)
        
        # Update capital
        self.cash -= position_value + transaction_cost
        if self.margin_enabled and direction < 0:  # Short requires margin
            self.margin_used += required_margin
            self.margin_available = (self.initial_capital * self.leverage) - self.margin_used
        
        # Track transaction cost
        position.transaction_costs += transaction_cost
        
        # Update metrics
        self._update_metrics()
        
        logger.info(f"Position opened: {position.position_id} {symbol} "
                    f"{'LONG' if direction > 0 else 'SHORT'} {quantity} @ {entry_price}")
        
        return position
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_time: datetime.datetime,
                      exit_type: ExitType = ExitType.MARKET,
                      exit_order_id: Optional[str] = None,
                      transaction_cost: float = 0.0) -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            position_id: ID of position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            exit_order_id: ID of exit order
            transaction_cost: Transaction cost for closing the position
            
        Returns:
            Dictionary with position summary information
            
        Raises:
            ValueError: If position ID is not found
        """
        # Find position
        if position_id not in self.positions:
            raise ValueError(f"Position not found: {position_id}")
            
        position = self.positions[position_id]
        
        # Add transaction cost to position
        position.transaction_costs += transaction_cost
        
        # Close position
        summary = position.close(
            exit_price=exit_price,
            exit_time=exit_time,
            exit_type=exit_type,
            exit_order_id=exit_order_id
        )
        
        # Update portfolio
        realized_pnl = position.realized_pnl
        
        # Move from open to closed positions
        del self.positions[position_id]
        self.closed_positions[position_id] = position
        
        # Remove from symbol-based index
        symbol = position.symbol
        self.positions_by_symbol[symbol] = [p for p in self.positions_by_symbol[symbol] 
                                          if p.position_id != position_id]
        
        # Update cash and margin
        position_value = position.initial_quantity * position.entry_price
        self.cash += position_value + realized_pnl - transaction_cost
        
        if self.margin_enabled and position.direction < 0:  # Short releases margin
            required_margin = position_value / self.leverage
            self.margin_used -= required_margin
            self.margin_available = (self.initial_capital * self.leverage) - self.margin_used
        
        # Update trade statistics
        self.total_trades += 1
        if realized_pnl > 0:
            self.winning_trades += 1
        elif realized_pnl < 0:
            self.losing_trades += 1
            
        self.realized_pnl += realized_pnl
        
        # Add to trade history
        trade_record = {
            'position_id': position_id,
            'symbol': position.symbol,
            'direction': position.direction,
            'quantity': position.initial_quantity,
            'entry_price': position.entry_price,
            'entry_time': position.entry_time,
            'exit_price': position.exit_price,
            'exit_time': position.exit_time,
            'exit_type': position.exit_type,
            'realized_pnl': position.realized_pnl,
            'transaction_costs': position.transaction_costs,
            'strategy_id': position.strategy_id
        }
        self.trades_history.append(trade_record)
        
        # Update metrics
        self._update_metrics()
        
        logger.info(f"Position closed: {position_id} {position.symbol} "
                    f"{'LONG' if position.direction > 0 else 'SHORT'} "
                    f"{position.initial_quantity} @ {exit_price} P&L: {realized_pnl:.2f}")
        
        return summary
    
    def close_all_positions(self, current_prices: Dict[str, float], 
                          exit_time: datetime.datetime,
                          exit_type: ExitType = ExitType.MARKET) -> List[Dict[str, Any]]:
        """
        Close all open positions.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            exit_time: Exit timestamp
            exit_type: Type of exit
            
        Returns:
            List of position summary dictionaries
        """
        summaries = []
        
        # Make a copy of position IDs to avoid modifying during iteration
        position_ids = list(self.positions.keys())
        
        for position_id in position_ids:
            position = self.positions[position_id]
            
            # Get current price for symbol
            symbol = position.symbol
            if symbol not in current_prices:
                logger.warning(f"No price available for {symbol}, using last known price")
                exit_price = position.current_price
            else:
                exit_price = current_prices[symbol]
            
            # Close the position
            summary = self.close_position(
                position_id=position_id,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_type=exit_type
            )
            
            summaries.append(summary)
        
        return summaries
    
    def close_positions_by_symbol(self, symbol: str, exit_price: float, 
                                exit_time: datetime.datetime,
                                exit_type: ExitType = ExitType.MARKET) -> List[Dict[str, Any]]:
        """
        Close all positions for a specific symbol.
        
        Args:
            symbol: Symbol to close positions for
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            
        Returns:
            List of position summary dictionaries
        """
        summaries = []
        
        # Make a copy of positions to avoid modifying during iteration
        positions = self.positions_by_symbol.get(symbol, [])[:]
        
        for position in positions:
            summary = self.close_position(
                position_id=position.position_id,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_type=exit_type
            )
            
            summaries.append(summary)
        
        return summaries
    
    def update_prices(self, current_prices: Dict[str, float], 
                     timestamp: datetime.datetime) -> List[Dict[str, Any]]:
        """
        Update all positions with current prices and check for exits.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            timestamp: Current timestamp
            
        Returns:
            List of exit information dictionaries for triggered exits
        """
        exits = []
        
        # Update each position
        for position in list(self.positions.values()):
            symbol = position.symbol
            
            # Get current price for symbol
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Update position with current price
                exit_info = position.update_price(current_price, timestamp)
                
                # If exit triggered, close position
                if exit_info:
                    self.close_position(
                        position_id=position.position_id,
                        exit_price=exit_info['exit_price'],
                        exit_time=exit_info['exit_time'],
                        exit_type=exit_info['exit_type']
                    )
                    
                    exits.append(exit_info)
        
        # Update metrics
        self._update_metrics()
        
        # Add to history
        self.equity_history.append((timestamp, self.equity))
        self.cash_history.append((timestamp, self.cash))
        
        return exits

    def get_cash(self) -> float:
        """
        Get the current cash balance.

        Returns:
            Current cash balance
        """
        return self.cash
    
    def get_position_value(self) -> float:
        """
        Get the total value of all open positions.
        
        Returns:
            Total position value
        """
        return sum(p.current_price * p.quantity for p in self.positions.values() if p.current_price is not None)
    
    def get_position_exposure(self) -> Dict[str, float]:
        """
        Get the exposure by symbol.
        
        Returns:
            Dictionary mapping symbols to exposure values
        """
        exposure = defaultdict(float)
        
        for position in self.positions.values():
            symbol = position.symbol
            value = position.current_price * position.quantity
            exposure[symbol] += value
        
        return dict(exposure)
    
    def get_position_count(self) -> Dict[str, int]:
        """
        Get the number of positions by symbol.
        
        Returns:
            Dictionary mapping symbols to position counts
        """
        counts = {}
        for symbol, positions in self.positions_by_symbol.items():
            counts[symbol] = len(positions)
        
        return counts
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if portfolio has any positions for a symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if portfolio has positions for the symbol, False otherwise
        """
        return symbol in self.positions_by_symbol and len(self.positions_by_symbol[symbol]) > 0
    
    def get_net_position(self, symbol: str) -> float:
        """
        Get the net position size for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Net position size (positive for net long, negative for net short)
        """
        positions = self.positions_by_symbol.get(symbol, [])
        if not positions:
            return 0.0
            
        return sum(p.quantity * p.direction for p in positions)
    
    def _update_metrics(self) -> None:
        """Update portfolio metrics."""
        # Calculate unrealized P&L
        self.unrealized_pnl = sum(p._calculate_unrealized_pnl() for p in self.positions.values())
        
        # Calculate equity
        position_value = self.get_position_value()
        self.equity = self.cash + position_value
        
        # Update high water mark and drawdown
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity
            self.drawdown = 0.0
        else:
            self.drawdown = (self.high_water_mark - self.equity) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, self.drawdown)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the portfolio.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate metrics
        roi = (self.equity / self.initial_capital) - 1 if self.initial_capital > 0 else 0
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'initial_capital': self.initial_capital,
            'current_equity': self.equity,
            'cash': self.cash,
            'position_value': self.get_position_value(),
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'roi': roi,
            'roi_percent': roi * 100,
            'drawdown': self.drawdown,
            'drawdown_percent': self.drawdown * 100,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown * 100,
            'high_water_mark': self.high_water_mark,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'win_rate_percent': win_rate * 100,
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions)
        }
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get information about all open positions.
        
        Returns:
            List of position summary dictionaries
        """
        return [p.get_summary() for p in self.positions.values()]
    
    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """
        Get information about all closed positions.
        
        Returns:
            List of position summary dictionaries
        """
        return [p.get_summary() for p in self.closed_positions.values()]
    
    def get_equity_curve(self) -> List[Tuple[datetime.datetime, float]]:
        """
        Get the equity curve data.
        
        Returns:
            List of (timestamp, equity) tuples
        """
        return self.equity_history
    
    def get_trades_history(self) -> List[Dict[str, Any]]:
        """
        Get the trades history.
        
        Returns:
            List of trade record dictionaries
        """
        return self.trades_history
    
    def reset(self) -> None:
        """Reset the portfolio to initial state."""
        # Reset capital
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.margin_used = 0.0
        self.margin_available = self.initial_capital if self.margin_enabled else 0.0
        
        # Reset positions
        self.positions = {}
        self.positions_by_symbol = defaultdict(list)
        self.closed_positions = {}
        
        # Reset performance tracking
        self.high_water_mark = self.initial_capital
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Reset history
        current_time = datetime.datetime.now()
        self.equity_history = [(current_time, self.initial_capital)]
        self.cash_history = [(current_time, self.initial_capital)]
        self.trades_history = []
        
        logger.info(f"Portfolio reset: {self.name}")


# Example usage
if __name__ == "__main__":
    # Create a portfolio
    portfolio = Portfolio(initial_capital=100000, name="Test Portfolio")
    
    # Open some positions
    entry_time = datetime.datetime.now()
    
    # Long position
    portfolio.open_position(
        symbol="AAPL",
        direction=1,
        quantity=100,
        entry_price=150.0,
        entry_time=entry_time,
        stop_loss=145.0,
        take_profit=160.0,
        strategy_id="trend_following"
    )
    
    # Short position
    portfolio.open_position(
        symbol="MSFT",
        direction=-1,
        quantity=50,
        entry_price=250.0,
        entry_time=entry_time,
        stop_loss=260.0,
        take_profit=240.0,
        strategy_id="mean_reversion"
    )
    
    # Update prices
    current_prices = {
        "AAPL": 155.0,
        "MSFT": 245.0
    }
    
    exits = portfolio.update_prices(current_prices, entry_time + datetime.timedelta(hours=1))
    
    # Print portfolio metrics
    metrics = portfolio.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Print open positions
    print("\nOpen positions:")
    for position in portfolio.get_open_positions():
        print(f"{position['symbol']} {position['direction']} {position['quantity']} @ {position['entry_price']}")
        print(f"Unrealized P&L: {position['unrealized_pnl']:.2f}")
