"""
Event-Based Portfolio Module

This module provides an event-driven portfolio implementation that
communicates through events rather than direct method calls.
"""

import datetime
import logging
import uuid
from typing import Dict, List, Any, Optional

from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType 
from src.events.event_handlers import EventHandler
from src.events.portfolio_events import (
    PortfolioUpdateEvent, 
    PositionOpenedEvent, 
    PositionClosedEvent
)
from src.position_management.position import Position, PositionStatus, EntryType, ExitType

# Set up logging
logger = logging.getLogger(__name__)


class EventPortfolio(EventHandler):
    """
    Event-driven portfolio that manages positions based on events.
    """
    
    def __init__(self, initial_capital: float, event_bus: EventBus, 
                 portfolio_id: Optional[str] = None,
                 name: Optional[str] = None, currency: str = 'USD',
                 allow_fractional_shares: bool = True,
                 margin_enabled: bool = False, leverage: float = 1.0):
        """
        Initialize event-driven portfolio.
        
        Args:
            initial_capital: Initial account capital
            event_bus: Event bus for communication
            portfolio_id: Unique portfolio ID (generated if None)
            name: Optional portfolio name
            currency: Portfolio base currency
            allow_fractional_shares: Whether fractional shares are allowed
            margin_enabled: Whether margin trading is enabled
            leverage: Maximum allowed leverage
        """
        super().__init__([EventType.POSITION_ACTION, EventType.FILL])
        
        self.event_bus = event_bus
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
        self.positions_by_symbol = {}  # Open positions by symbol
        self.closed_positions = {}  # Historical closed positions
        
        # Performance tracking
        self.high_water_mark = initial_capital
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Register with event bus
        for event_type in self.event_types:
            event_bus.register(event_type, self)
        
        # Emit initial portfolio state
        self._emit_portfolio_update()
        
        logger.info(f"Portfolio initialized: {self.name} with {self.initial_capital} {self.currency}")
    
    def _process_event(self, event: Event) -> None:
        """
        Process incoming events.
        
        Args:
            event: Event to process
        """
        if event.event_type == EventType.POSITION_ACTION:
            self._handle_position_action(event)
        elif event.event_type == EventType.FILL:
            self._handle_fill(event)

    # Add this method to your EventPortfolio class
    def mark_to_market(self, bar_event):
        """
        Update portfolio positions with current market prices.

        Args:
            bar_event: Bar event containing current market data
        """
        # Extract data from bar event
        symbol = bar_event.get_symbol()
        current_price = bar_event.get_price()  # Typically close price
        timestamp = bar_event.get_timestamp()

        # Update positions for this symbol
        if symbol in self.positions_by_symbol:
            for position in self.positions_by_symbol[symbol]:
                # Update position with current price
                if hasattr(position, 'update_price'):
                    position.update_price(current_price, timestamp)
                # If position doesn't have update_price, update current_price directly
                else:
                    position.current_price = current_price

        # Update portfolio metrics
        self._update_metrics()

        # Emit portfolio update event
        self._emit_portfolio_update()


    # Add this method to your EventPortfolio class
    def get_position_snapshot(self):
        """
        Get a snapshot of all current positions.

        Returns:
            Dictionary with position information
        """
        position_data = {}

        for symbol, positions in self.positions_by_symbol.items():
            if not positions:
                continue

            position_data[symbol] = []
            for position in positions:
                # Create a snapshot for this position
                snapshot = {
                    'position_id': position.position_id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': getattr(position, 'current_price', position.entry_price),
                    'entry_time': position.entry_time,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'status': position.status if hasattr(position, 'status') else 'OPEN',
                    'unrealized_pnl': position._calculate_unrealized_pnl() if hasattr(position, '_calculate_unrealized_pnl') else 0.0
                }
                position_data[symbol].append(snapshot)

        return position_data
            
    def _handle_position_action(self, event: Event) -> None:
        """
        Handle position action events.
        
        Args:
            event: Position action event
        """
        action_data = event.data
        action_type = action_data.get('action_type')
        
        if action_type == 'entry':
            # Extract entry data
            symbol = action_data.get('symbol')
            direction = action_data.get('direction')
            size = action_data.get('size')
            price = action_data.get('price')
            stop_loss = action_data.get('stop_loss')
            take_profit = action_data.get('take_profit')
            strategy_id = action_data.get('strategy_id')
            entry_time = action_data.get('timestamp', datetime.datetime.now())
            
            try:
                # Create position
                position = self._open_position(
                    symbol=symbol,
                    direction=direction,
                    quantity=size,
                    entry_price=price,
                    entry_time=entry_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_id=strategy_id
                )
                
                # Emit position opened event
                self._emit_position_opened(position)
                
                # Update portfolio state
                self._update_metrics()
                self._emit_portfolio_update()
                
                logger.info(f"Position opened: {position.position_id} {symbol} "
                           f"{'LONG' if direction > 0 else 'SHORT'} {size} @ {price}")
            except Exception as e:
                logger.error(f"Failed to open position: {e}")
                
        elif action_type == 'exit':
            # Extract exit data
            position_id = action_data.get('position_id')
            price = action_data.get('price')
            exit_time = action_data.get('timestamp', datetime.datetime.now())
            exit_type = action_data.get('exit_type', ExitType.STRATEGY)
            
            try:
                # Close position
                result = self._close_position(
                    position_id=position_id,
                    exit_price=price,
                    exit_time=exit_time,
                    exit_type=exit_type
                )
                
                # Emit position closed event
                self._emit_position_closed(result)
                
                # Update portfolio state
                self._update_metrics()
                self._emit_portfolio_update()
                
                logger.info(f"Position closed: {position_id} "
                           f"PnL: {result.get('realized_pnl', 0):.2f}")
            except Exception as e:
                logger.error(f"Failed to close position: {e}")
    
    def _handle_fill(self, event: Event) -> None:
        """
        Handle fill events.
        
        Args:
            event: Fill event
        """
        # Implementation for handling fills
        pass
    
    def _open_position(self, symbol: str, direction: int, quantity: float, 
                    entry_price: float, entry_time: datetime.datetime,
                    stop_loss: Optional[float] = None, 
                    take_profit: Optional[float] = None,
                    strategy_id: Optional[str] = None) -> Position:
        """
        Open a new position.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            quantity: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            strategy_id: Optional strategy ID
            
        Returns:
            Newly created Position
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
        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_id=strategy_id
        )
        
        # Update portfolio
        self.positions[position.position_id] = position
        if symbol not in self.positions_by_symbol:
            self.positions_by_symbol[symbol] = []
        self.positions_by_symbol[symbol].append(position)
        
        # Update capital
        self.cash -= position_value
        if self.margin_enabled and direction < 0:  # Short requires margin
            self.margin_used += required_margin
            self.margin_available = (self.initial_capital * self.leverage) - self.margin_used
        
        return position
    
    def _close_position(self, position_id: str, exit_price: float, 
                      exit_time: datetime.datetime,
                      exit_type: ExitType = ExitType.STRATEGY) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            position_id: ID of position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            
        Returns:
            Position summary dictionary
        """
        # Find position
        if position_id not in self.positions:
            raise ValueError(f"Position not found: {position_id}")
            
        position = self.positions[position_id]
        
        # Close position
        summary = position.close(exit_price, exit_time, exit_type)
        
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
        self.cash += position_value + realized_pnl
        self.realized_pnl += realized_pnl
        
        if self.margin_enabled and position.direction < 0:  # Short releases margin
            required_margin = position_value / self.leverage
            self.margin_used -= required_margin
            self.margin_available = (self.initial_capital * self.leverage) - self.margin_used
        
        return summary
    
    def _update_metrics(self) -> None:
        """Update portfolio metrics."""
        # Calculate unrealized P&L
        self.unrealized_pnl = sum(p._calculate_unrealized_pnl() for p in self.positions.values())
        
        # Calculate equity
        position_value = sum(p.quantity * p.current_price for p in self.positions.values() if p.current_price)
        self.equity = self.cash + position_value
        
        # Update high water mark and drawdown
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity
            self.drawdown = 0.0
        else:
            self.drawdown = (self.high_water_mark - self.equity) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, self.drawdown)
    
    def _emit_portfolio_update(self) -> None:
        """Emit portfolio update event."""
        if not self.event_bus:
            return
            
        # Create portfolio state data
        portfolio_state = {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'equity': self.equity,
            'cash': self.cash,
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'drawdown': self.drawdown,
            'max_drawdown': self.max_drawdown,
            'positions_count': len(self.positions),
            'timestamp': datetime.datetime.now()
        }
        
        # Create and emit event
        update_event = PortfolioUpdateEvent(portfolio_state)
        self.event_bus.emit(update_event)
    
    def _emit_position_opened(self, position: Position) -> None:
        """
        Emit position opened event.
        
        Args:
            position: Position that was opened
        """
        if not self.event_bus:
            return
            
        # Create position data
        position_data = {
            'position_id': position.position_id,
            'symbol': position.symbol,
            'direction': position.direction,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'entry_time': position.entry_time,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'strategy_id': position.strategy_id
        }
        
        # Create and emit event
        opened_event = PositionOpenedEvent(position_data, position.entry_time)
        self.event_bus.emit(opened_event)
    
    def _emit_position_closed(self, position_summary: Dict[str, Any]) -> None:
        """
        Emit position closed event.
        
        Args:
            position_summary: Position summary dictionary
        """
        if not self.event_bus:
            return
            
        # Create and emit event
        closed_event = PositionClosedEvent(position_summary, position_summary.get('exit_time'))
        self.event_bus.emit(closed_event)




    def get_net_position(self, symbol):
        """Get net position for a symbol."""
        net_position = 0
        if symbol in self.positions_by_symbol:
            for pos in self.positions_by_symbol[symbol]:
                net_position += pos.quantity * pos.direction
        return net_position

    def get_positions_by_symbol(self, symbol):
        """Get positions for a symbol."""
        return self.positions_by_symbol.get(symbol, [])        
