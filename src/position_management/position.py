"""
Position Module

This module defines the Position class and related utilities for representing
and managing trading positions within the system.
"""

import uuid
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum, auto
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Enumeration of possible position statuses."""
    OPEN = auto()           # Position is currently open
    CLOSED = auto()         # Position has been closed
    PARTIALLY_CLOSED = auto() # Position has been partially closed
    PENDING_OPEN = auto()    # Position open order is pending
    PENDING_CLOSE = auto()   # Position close order is pending
    ERROR = auto()           # Position is in an error state


class EntryType(Enum):
    """Enumeration of possible position entry types."""
    MARKET = auto()          # Market order entry
    LIMIT = auto()           # Limit order entry
    STOP = auto()            # Stop order entry
    STOP_LIMIT = auto()      # Stop-limit order entry


class ExitType(Enum):
    """Enumeration of possible position exit types."""
    MARKET = auto()          # Market order exit
    LIMIT = auto()           # Limit order exit
    STOP = auto()            # Stop order exit (stop-loss)
    STOP_LIMIT = auto()      # Stop-limit order exit
    TAKE_PROFIT = auto()     # Take profit exit
    TRAILING_STOP = auto()   # Trailing stop exit
    TIME_STOP = auto()       # Time-based exit
    STRATEGY = auto()        # Strategy-based exit
    MANUAL = auto()          # Manual exit


class Position:
    """
    Represents a trading position.
    
    A position encapsulates information about an open or closed trading position,
    including its size, entry and exit details, profit/loss, and current status.
    """
    
    def __init__(self, symbol: str, direction: int, quantity: float, 
                entry_price: float, entry_time: datetime.datetime,
                entry_type: EntryType = EntryType.MARKET,
                position_id: Optional[str] = None,
                strategy_id: Optional[str] = None,
                entry_order_id: Optional[str] = None,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                initial_risk: Optional[float] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new position.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            quantity: Position size in units
            entry_price: Entry price
            entry_time: Entry timestamp
            entry_type: Type of entry (market, limit, etc.)
            position_id: Unique position ID (generated if None)
            strategy_id: ID of the strategy that created the position
            entry_order_id: ID of the order that opened the position
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            initial_risk: Optional initial risk amount
            metadata: Optional additional position metadata
        """
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.entry_type = entry_type
        self.position_id = position_id or str(uuid.uuid4())
        self.strategy_id = strategy_id
        self.entry_order_id = entry_order_id
        
        # Risk management
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = None
        self.trailing_stop_distance = None
        self.trailing_stop_high = None if direction > 0 else entry_price
        self.trailing_stop_low = None if direction < 0 else entry_price
        self.initial_risk = initial_risk
        
        # Exit details
        self.exit_price = None
        self.exit_time = None
        self.exit_type = None
        self.exit_order_id = None
        
        # Status and metrics
        self.status = PositionStatus.OPEN
        self.realized_pnl = 0.0
        self.current_price = entry_price
        self.max_favorable_excursion = 0.0  # Maximum profit during position
        self.max_adverse_excursion = 0.0    # Maximum loss during position
        self.transaction_costs = 0.0
        
        # Position modifications
        self.avg_entry_price = entry_price
        self.initial_quantity = quantity
        self.closed_quantity = 0.0
        self.modifications = []
        
        # Additional metadata
        self.metadata = metadata or {}
        
        # For partial fills/closes
        self.fills = []
        
        logger.info(f"Position opened: {self.position_id} {self.symbol} "
                    f"{'LONG' if direction > 0 else 'SHORT'} {self.quantity} @ {self.entry_price}")
    
    def update_price(self, current_price: float, timestamp: datetime.datetime) -> Dict[str, Any]:
        """
        Update position with current price and check for exits.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with exit information if exit triggered, None otherwise
        """
        self.current_price = current_price
        
        # Calculate unrealized P&L
        unrealized_pnl = self._calculate_unrealized_pnl()
        
        # Update max favorable/adverse excursion
        if self.direction > 0:  # Long position
            if current_price > self.entry_price:
                # Favorable excursion (profit)
                favorable_excursion = (current_price - self.entry_price) * self.quantity
                self.max_favorable_excursion = max(self.max_favorable_excursion, favorable_excursion)
            else:
                # Adverse excursion (loss)
                adverse_excursion = (self.entry_price - current_price) * self.quantity
                self.max_adverse_excursion = max(self.max_adverse_excursion, adverse_excursion)
                
            # Update trailing stop high (for long positions)
            if self.trailing_stop_high is None or current_price > self.trailing_stop_high:
                self.trailing_stop_high = current_price
                
                # Update trailing stop if activated
                if self.trailing_stop is not None and self.trailing_stop_distance is not None:
                    self.trailing_stop = current_price - self.trailing_stop_distance
        else:  # Short position
            if current_price < self.entry_price:
                # Favorable excursion (profit)
                favorable_excursion = (self.entry_price - current_price) * self.quantity
                self.max_favorable_excursion = max(self.max_favorable_excursion, favorable_excursion)
            else:
                # Adverse excursion (loss)
                adverse_excursion = (current_price - self.entry_price) * self.quantity
                self.max_adverse_excursion = max(self.max_adverse_excursion, adverse_excursion)
                
            # Update trailing stop low (for short positions)
            if self.trailing_stop_low is None or current_price < self.trailing_stop_low:
                self.trailing_stop_low = current_price
                
                # Update trailing stop if activated
                if self.trailing_stop is not None and self.trailing_stop_distance is not None:
                    self.trailing_stop = current_price + self.trailing_stop_distance
        
        # Check for exit conditions
        exit_info = self._check_exits(current_price, timestamp)
        
        # Return exit information if triggered
        return exit_info
    
    def _check_exits(self, current_price: float, timestamp: datetime.datetime) -> Optional[Dict[str, Any]]:
        """
        Check if any exit conditions are triggered.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with exit information if exit triggered, None otherwise
        """
        if self.status != PositionStatus.OPEN:
            return None
            
        exit_type = None
        exit_reason = None
        
        # Check stop loss
        if self.stop_loss is not None:
            if (self.direction > 0 and current_price <= self.stop_loss) or \
               (self.direction < 0 and current_price >= self.stop_loss):
                exit_type = ExitType.STOP
                exit_reason = "Stop loss triggered"
        
        # Check take profit
        if exit_type is None and self.take_profit is not None:
            if (self.direction > 0 and current_price >= self.take_profit) or \
               (self.direction < 0 and current_price <= self.take_profit):
                exit_type = ExitType.TAKE_PROFIT
                exit_reason = "Take profit triggered"
        
        # Check trailing stop
        if exit_type is None and self.trailing_stop is not None:
            if (self.direction > 0 and current_price <= self.trailing_stop) or \
               (self.direction < 0 and current_price >= self.trailing_stop):
                exit_type = ExitType.TRAILING_STOP
                exit_reason = "Trailing stop triggered"
        
        # If exit condition triggered, return exit information
        if exit_type is not None:
            return {
                'position_id': self.position_id,
                'symbol': self.symbol,
                'exit_type': exit_type,
                'exit_reason': exit_reason,
                'exit_price': current_price,
                'exit_time': timestamp,
                'quantity': self.quantity,
                'pnl': self._calculate_realized_pnl(current_price)
            }
        
        return None
    
    def close(self, exit_price: float, exit_time: datetime.datetime, 
             exit_type: ExitType = ExitType.MARKET, 
             exit_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close the position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            exit_order_id: ID of exit order
            
        Returns:
            Dictionary with position summary information
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to close already closed position: {self.position_id}")
            return self.get_summary()
        
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_type = exit_type
        self.exit_order_id = exit_order_id
        self.status = PositionStatus.CLOSED
        
        # Calculate realized P&L
        self.realized_pnl = self._calculate_realized_pnl(exit_price)
        
        logger.info(f"Position closed: {self.position_id} {self.symbol} "
                    f"{'LONG' if self.direction > 0 else 'SHORT'} {self.quantity} @ {self.exit_price} "
                    f"P&L: {self.realized_pnl:.2f}")
        
        return self.get_summary()
    
    def partially_close(self, quantity: float, exit_price: float, 
                       exit_time: datetime.datetime,
                       exit_type: ExitType = ExitType.MARKET,
                       exit_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Partially close the position.
        
        Args:
            quantity: Quantity to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            exit_order_id: ID of exit order
            
        Returns:
            Dictionary with partial close information
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to partially close already closed position: {self.position_id}")
            return self.get_summary()
        
        if quantity > self.quantity:
            logger.warning(f"Attempted to close more than available quantity: {quantity} > {self.quantity}")
            quantity = self.quantity
        
        # Calculate P&L for closed portion
        partial_pnl = (exit_price - self.avg_entry_price) * quantity * self.direction
        
        # Update position
        self.closed_quantity += quantity
        self.quantity -= quantity
        self.realized_pnl += partial_pnl
        
        # Add to fills
        fill = {
            'quantity': quantity,
            'price': exit_price,
            'time': exit_time,
            'type': exit_type,
            'order_id': exit_order_id,
            'pnl': partial_pnl
        }
        self.fills.append(fill)
        
        # Update status
        if self.quantity <= 0:
            # If fully closed, update final exit details
            self.status = PositionStatus.CLOSED
            self.exit_price = exit_price
            self.exit_time = exit_time
            self.exit_type = exit_type
            self.exit_order_id = exit_order_id
            
            logger.info(f"Position closed: {self.position_id} {self.symbol} "
                        f"{'LONG' if self.direction > 0 else 'SHORT'} @ {self.exit_price} "
                        f"P&L: {self.realized_pnl:.2f}")
        else:
            # If partially closed, update status
            self.status = PositionStatus.PARTIALLY_CLOSED
            
            logger.info(f"Position partially closed: {self.position_id} {self.symbol} "
                        f"{'LONG' if self.direction > 0 else 'SHORT'} {quantity} @ {exit_price} "
                        f"Partial P&L: {partial_pnl:.2f}, Remaining: {self.quantity}")
        
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'closed_quantity': quantity,
            'remaining_quantity': self.quantity,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_type': exit_type,
            'partial_pnl': partial_pnl,
            'total_realized_pnl': self.realized_pnl,
            'status': self.status
        }
    
    def add(self, quantity: float, price: float, time: datetime.datetime,
           order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add to the position (increase size).
        
        Args:
            quantity: Quantity to add
            price: Price of additional units
            time: Timestamp of addition
            order_id: ID of order that added to position
            
        Returns:
            Dictionary with addition information
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to add to closed position: {self.position_id}")
            return self.get_summary()
        
        # Update average entry price
        total_cost = (self.avg_entry_price * self.quantity) + (price * quantity)
        self.quantity += quantity
        self.avg_entry_price = total_cost / self.quantity
        
        # Record modification
        modification = {
            'type': 'add',
            'quantity': quantity,
            'price': price,
            'time': time,
            'order_id': order_id,
            'new_avg_price': self.avg_entry_price,
            'new_quantity': self.quantity
        }
        self.modifications.append(modification)
        
        logger.info(f"Position increased: {self.position_id} {self.symbol} "
                    f"{'LONG' if self.direction > 0 else 'SHORT'} +{quantity} @ {price} "
                    f"New size: {self.quantity}, Avg price: {self.avg_entry_price:.4f}")
        
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'added_quantity': quantity,
            'total_quantity': self.quantity,
            'price': price,
            'avg_entry_price': self.avg_entry_price,
            'time': time
        }
    
    def modify_stop_loss(self, new_stop_loss: float) -> Dict[str, Any]:
        """
        Modify the stop loss price.
        
        Args:
            new_stop_loss: New stop loss price
            
        Returns:
            Dictionary with modification information
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to modify stop loss for closed position: {self.position_id}")
            return self.get_summary()
            
        old_stop_loss = self.stop_loss
        self.stop_loss = new_stop_loss
        
        logger.info(f"Position stop loss modified: {self.position_id} {self.symbol} "
                    f"Old: {old_stop_loss}, New: {new_stop_loss}")
        
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'old_stop_loss': old_stop_loss,
            'new_stop_loss': new_stop_loss
        }
    
    def modify_take_profit(self, new_take_profit: float) -> Dict[str, Any]:
        """
        Modify the take profit price.
        
        Args:
            new_take_profit: New take profit price
            
        Returns:
            Dictionary with modification information
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to modify take profit for closed position: {self.position_id}")
            return self.get_summary()
            
        old_take_profit = self.take_profit
        self.take_profit = new_take_profit
        
        logger.info(f"Position take profit modified: {self.position_id} {self.symbol} "
                    f"Old: {old_take_profit}, New: {new_take_profit}")
        
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'old_take_profit': old_take_profit,
            'new_take_profit': new_take_profit
        }
    
    def set_trailing_stop(self, distance: float) -> Dict[str, Any]:
        """
        Set a trailing stop for the position.
        
        Args:
            distance: Distance from current price for trailing stop
            
        Returns:
            Dictionary with trailing stop information
        """
        if self.status == PositionStatus.CLOSED:
            logger.warning(f"Attempting to set trailing stop for closed position: {self.position_id}")
            return self.get_summary()
            
        self.trailing_stop_distance = distance
        
        # Initialize trailing stop based on current price and direction
        if self.direction > 0:  # Long
            self.trailing_stop = self.current_price - distance
        else:  # Short
            self.trailing_stop = self.current_price + distance
            
        logger.info(f"Position trailing stop set: {self.position_id} {self.symbol} "
                    f"Distance: {distance}, Initial stop: {self.trailing_stop}")
        
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'trailing_stop_distance': distance,
            'initial_trailing_stop': self.trailing_stop
        }
    
    def _calculate_unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L for the position.
        
        Returns:
            Unrealized P&L
        """
        if self.status == PositionStatus.CLOSED:
            return self.realized_pnl
            
        if self.current_price is None:
            return 0.0
            
        return (self.current_price - self.avg_entry_price) * self.quantity * self.direction
    
    def _calculate_realized_pnl(self, exit_price: float) -> float:
        """
        Calculate realized P&L for the position.
        
        Args:
            exit_price: Exit price
            
        Returns:
            Realized P&L
        """
        # Calculate P&L from any partial closes
        partial_pnl = self.realized_pnl
        
        # Calculate P&L for remaining quantity
        remaining_pnl = (exit_price - self.avg_entry_price) * self.quantity * self.direction
        
        # Subtract transaction costs
        total_pnl = partial_pnl + remaining_pnl - self.transaction_costs
        
        return total_pnl
    
    def get_duration(self) -> Optional[datetime.timedelta]:
        """
        Get the duration of the position.
        
        Returns:
            Timedelta if position is closed, None otherwise
        """
        if self.exit_time is None:
            return None
            
        return self.exit_time - self.entry_time
    
    def get_current_return(self) -> float:
        """
        Get the current return percentage for the position.
        
        Returns:
            Current return percentage
        """
        if self.avg_entry_price == 0:
            return 0.0
            
        unrealized_pnl = self._calculate_unrealized_pnl()
        initial_value = self.avg_entry_price * self.initial_quantity
        
        if initial_value == 0:
            return 0.0
            
        return unrealized_pnl / initial_value
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """
        Get the risk-reward ratio for the position.
        
        Returns:
            Risk-reward ratio if stop loss and take profit are set, None otherwise
        """
        if self.stop_loss is None or self.take_profit is None:
            return None
            
        # Calculate potential risk and reward
        if self.direction > 0:  # Long
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:  # Short
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
            
        if risk == 0:
            return None
            
        return reward / risk
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the position.
        
        Returns:
            Dictionary with position summary information
        """
        duration = self.get_duration()
        
        summary = {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'strategy_id': self.strategy_id,
            'status': self.status,
            'quantity': self.quantity,
            'initial_quantity': self.initial_quantity,
            'entry_price': self.entry_price,
            'avg_entry_price': self.avg_entry_price,
            'entry_time': self.entry_time,
            'entry_type': self.entry_type,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'exit_type': self.exit_type,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'return_pct': self.get_current_return() * 100,
            'transaction_costs': self.transaction_costs,
            'duration': duration.total_seconds() if duration else None,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'risk_reward_ratio': self.get_risk_reward_ratio(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'modifications': len(self.modifications),
            'metadata': self.metadata
        }
        
        return summary
        
    def __str__(self) -> str:
        """String representation of the position."""
        return (f"Position(id={self.position_id}, symbol={self.symbol}, "
                f"direction={'LONG' if self.direction > 0 else 'SHORT'}, "
                f"quantity={self.quantity}, "
                f"entry_price={self.entry_price}, "
                f"status={self.status.name})")



class PositionFactory:
    """
    Factory for creating Position objects with different configurations.
    
    This factory simplifies position creation with sensible defaults
    and provides utility methods for creating positions with specific
    risk parameters.
    """
    
    @classmethod
    def create_position(cls, symbol: str, direction: int, quantity: float, 
                      entry_price: float, entry_time: datetime.datetime,
                      **kwargs) -> Position:
        """
        Create a basic position with the essential parameters.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            quantity: Position size in units
            entry_price: Entry price
            entry_time: Entry timestamp
            **kwargs: Additional position parameters
            
        Returns:
            Position: Newly created position
        """
        return Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            **kwargs
        )
    
    @classmethod
    def create_position_with_stops(cls, symbol: str, direction: int, quantity: float,
                                entry_price: float, entry_time: datetime.datetime,
                                stop_loss_pct: float = None, stop_loss_price: float = None,
                                take_profit_pct: float = None, take_profit_price: float = None,
                                **kwargs) -> Position:
        """
        Create a position with stop loss and/or take profit levels.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            quantity: Position size in units
            entry_price: Entry price
            entry_time: Entry timestamp
            stop_loss_pct: Optional stop loss percentage from entry
            stop_loss_price: Optional explicit stop loss price
            take_profit_pct: Optional take profit percentage from entry
            take_profit_price: Optional explicit take profit price
            **kwargs: Additional position parameters
            
        Returns:
            Position: Position with stop loss and/or take profit
        """
        # Calculate stop loss and take profit prices if percentages provided
        if stop_loss_pct is not None and stop_loss_price is None:
            if direction > 0:  # Long
                stop_loss_price = entry_price * (1 - stop_loss_pct)
            else:  # Short
                stop_loss_price = entry_price * (1 + stop_loss_pct)
        
        if take_profit_pct is not None and take_profit_price is None:
            if direction > 0:  # Long
                take_profit_price = entry_price * (1 + take_profit_pct)
            else:  # Short
                take_profit_price = entry_price * (1 - take_profit_pct)
        
        # Create position with stops
        return Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            **kwargs
        )
    
    @classmethod
    def create_position_with_risk(cls, symbol: str, direction: int,
                               entry_price: float, entry_time: datetime.datetime,
                               account_size: float, risk_pct: float, stop_loss_price: float,
                               take_profit_price: float = None, max_position_pct: float = None,
                               **kwargs) -> Position:
        """
        Create a position sized according to risk percentage of account.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            entry_price: Entry price
            entry_time: Entry timestamp
            account_size: Current account size
            risk_pct: Percentage of account to risk
            stop_loss_price: Stop loss price
            take_profit_price: Optional take profit price
            max_position_pct: Optional maximum position size as percentage of account
            **kwargs: Additional position parameters
            
        Returns:
            Position: Risk-sized position
        """
        # Calculate risk amount
        risk_amount = account_size * risk_pct
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            logger.warning("Stop distance is zero, using default 1% of entry price")
            stop_distance = entry_price * 0.01
        
        # Calculate position size based on risk
        quantity = risk_amount / stop_distance
        
        # Apply maximum position size constraint if provided
        if max_position_pct is not None:
            max_quantity = (account_size * max_position_pct) / entry_price
            quantity = min(quantity, max_quantity)
        
        # Calculate initial risk
        initial_risk = quantity * stop_distance
        
        # Create position
        return Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            initial_risk=initial_risk,
            **kwargs
        )
    
    @classmethod
    def create_position_with_trailing_stop(cls, symbol: str, direction: int, quantity: float,
                                       entry_price: float, entry_time: datetime.datetime,
                                       trailing_stop_distance: float, activation_pct: float = None,
                                       **kwargs) -> Position:
        """
        Create a position with a trailing stop.
        
        Args:
            symbol: Instrument symbol
            direction: Position direction (1 for long, -1 for short)
            quantity: Position size in units
            entry_price: Entry price
            entry_time: Entry timestamp
            trailing_stop_distance: Distance for trailing stop
            activation_pct: Optional percentage move to activate trailing stop
            **kwargs: Additional position parameters
            
        Returns:
            Position: Position with trailing stop
        """
        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time,
            **kwargs
        )
        
        # Set trailing stop
        position.trailing_stop_distance = trailing_stop_distance
        
        # Set initial trailing stop level
        if direction > 0:  # Long
            position.trailing_stop = entry_price - trailing_stop_distance
        else:  # Short
            position.trailing_stop = entry_price + trailing_stop_distance
        
        # If activation percentage provided, store it in metadata
        if activation_pct is not None:
            if position.metadata is None:
                position.metadata = {}
            position.metadata['trailing_stop_activation_pct'] = activation_pct
            
            # Calculate activation level
            if direction > 0:  # Long
                activation_level = entry_price * (1 + activation_pct)
            else:  # Short
                activation_level = entry_price * (1 - activation_pct)
                
            position.metadata['trailing_stop_activation_level'] = activation_level
            
            # Initially disable trailing stop until activation level reached
            position.trailing_stop = None
        
        return position
    
    @classmethod
    def create_from_signal(cls, signal: Dict[str, Any], account_size: float, 
                        risk_pct: float = 0.01, position_pct: float = None,
                        use_stops: bool = True, **kwargs) -> Position:
        """
        Create a position from a trading signal.
        
        Args:
            signal: Trading signal dictionary
            account_size: Current account size
            risk_pct: Percentage of account to risk
            position_pct: Optional position size as percentage of account
            use_stops: Whether to use stop loss/take profit from signal
            **kwargs: Additional position parameters
            
        Returns:
            Position: Position based on signal
            
        Raises:
            ValueError: If signal is missing required fields
        """
        # Validate signal
        required_fields = ['symbol', 'direction', 'price', 'timestamp']
        for field in required_fields:
            if field not in signal:
                raise ValueError(f"Signal missing required field: {field}")
        
        # Extract basic signal data
        symbol = signal['symbol']
        direction = 1 if signal['direction'] in [1, 'long', 'buy'] else -1
        entry_price = float(signal['price'])
        entry_time = signal['timestamp']
        
        # Extract stop loss and take profit if available
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        
        # Create position parameters
        position_params = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            **kwargs
        }
        
        # Add stops if provided and requested
        if use_stops:
            if stop_loss is not None:
                position_params['stop_loss'] = stop_loss
            if take_profit is not None:
                position_params['take_profit'] = take_profit
        
        # Create metadata with signal information
        metadata = kwargs.get('metadata', {})
        metadata['signal_source'] = signal.get('source', 'unknown')
        metadata['signal_confidence'] = signal.get('confidence', 1.0)
        metadata['signal_id'] = signal.get('id')
        position_params['metadata'] = metadata
        
        # Determine position sizing method
        if stop_loss is not None and risk_pct is not None:
            # Use risk-based sizing
            stop_distance = abs(entry_price - stop_loss)
            risk_amount = account_size * risk_pct
            quantity = risk_amount / stop_distance if stop_distance > 0 else 0
            position_params['initial_risk'] = risk_amount
        elif position_pct is not None:
            # Use percentage of account sizing
            quantity = (account_size * position_pct) / entry_price
        else:
            # Default to 1% of account
            quantity = (account_size * 0.01) / entry_price
        
        position_params['quantity'] = quantity
        
        # Create position
        return Position(**position_params)


class PositionManager:
    """
    Manages a collection of positions.
    
    This class provides functionality for tracking, updating, and managing
    multiple trading positions across different symbols.
    """
    
    def __init__(self):
        """Initialize the position manager."""
        self.positions = {}  # Maps position_id to Position objects
        self.positions_by_symbol = {}  # Maps symbol to lists of position_ids
        
    def add_position(self, position: Position) -> str:
        """
        Add a position to the manager.
        
        Args:
            position: Position to add
            
        Returns:
            str: Position ID
        """
        position_id = position.position_id
        
        # Store by ID
        self.positions[position_id] = position
        
        # Store by symbol
        symbol = position.symbol
        if symbol not in self.positions_by_symbol:
            self.positions_by_symbol[symbol] = []
        self.positions_by_symbol[symbol].append(position_id)
        
        logger.info(f"Position added to manager: {position_id} {symbol}")
        
        return position_id
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(position_id)
    
    def get_positions_for_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            List of Position objects
        """
        position_ids = self.positions_by_symbol.get(symbol, [])
        return [self.positions[pid] for pid in position_ids if pid in self.positions]
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open Position objects
        """
        return [p for p in self.positions.values() 
                if p.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]]
    
    def get_open_positions_for_symbol(self, symbol: str) -> List[Position]:
        """
        Get open positions for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            List of open Position objects for the symbol
        """
        positions = self.get_positions_for_symbol(symbol)
        return [p for p in positions 
                if p.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]]
    
    def update_positions(self, symbol: str, current_price: float, 
                         timestamp: datetime.datetime) -> List[Dict[str, Any]]:
        """
        Update all positions for a symbol with current price.
        
        Args:
            symbol: Instrument symbol
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            List of exit information for any triggered exits
        """
        positions = self.get_open_positions_for_symbol(symbol)
        exit_infos = []
        
        for position in positions:
            exit_info = position.update_price(current_price, timestamp)
            if exit_info is not None:
                exit_infos.append(exit_info)
                
        return exit_infos
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_time: datetime.datetime,
                      exit_type: ExitType = ExitType.MARKET,
                      exit_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close a position.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            exit_order_id: ID of exit order
            
        Returns:
            Dictionary with position summary or None if position not found
        """
        position = self.get_position(position_id)
        if position is None:
            logger.warning(f"Attempt to close non-existent position: {position_id}")
            return None
            
        return position.close(exit_price, exit_time, exit_type, exit_order_id)
    
    def close_all_positions(self, exit_price_func, exit_time: datetime.datetime,
                         exit_type: ExitType = ExitType.MARKET) -> List[Dict[str, Any]]:
        """
        Close all open positions.
        
        Args:
            exit_price_func: Function that takes a position and returns exit price
            exit_time: Exit timestamp
            exit_type: Type of exit
            
        Returns:
            List of position summaries
        """
        open_positions = self.get_open_positions()
        summaries = []
        
        for position in open_positions:
            exit_price = exit_price_func(position)
            summary = position.close(exit_price, exit_time, exit_type)
            summaries.append(summary)
            
        return summaries
    
    def get_portfolio_value(self, price_func) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            price_func: Function that takes a symbol and returns current price
            
        Returns:
            Total portfolio value
        """
        total_value = 0.0
        
        for position in self.get_open_positions():
            current_price = price_func(position.symbol)
            position_value = position.quantity * current_price
            total_value += position_value
            
        return total_value
    
    def get_portfolio_metrics(self, price_func) -> Dict[str, Any]:
        """
        Calculate portfolio metrics.
        
        Args:
            price_func: Function that takes a symbol and returns current price
            
        Returns:
            Dictionary with portfolio metrics
        """
        open_positions = self.get_open_positions()
        
        # Calculate basic metrics
        num_positions = len(open_positions)
        total_value = 0.0
        total_pnl = 0.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        
        for position in open_positions:
            current_price = price_func(position.symbol)
            position.current_price = current_price
            
            position_value = position.quantity * current_price
            total_value += position_value
            
            position_unrealized_pnl = position._calculate_unrealized_pnl()
            unrealized_pnl += position_unrealized_pnl
        
        # Include realized P&L from closed positions
        for position in self.positions.values():
            if position.status == PositionStatus.CLOSED:
                realized_pnl += position.realized_pnl
        
        total_pnl = realized_pnl + unrealized_pnl
        
        return {
            'num_open_positions': num_positions,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl
        }
    
    def record_transaction_costs(self, position_id: str, costs: float) -> bool:
        """
        Record transaction costs for a position.
        
        Args:
            position_id: Position ID
            costs: Transaction costs
            
        Returns:
            True if successful, False if position not found
        """
        position = self.get_position(position_id)
        if position is None:
            return False
            
        position.transaction_costs += costs
        return True
    
    def apply_to_positions(self, func, filter_func=None) -> List[Any]:
        """
        Apply a function to positions optionally filtered by a filter function.
        
        Args:
            func: Function to apply to each position
            filter_func: Optional function to filter positions
            
        Returns:
            List of function results
        """
        positions = self.positions.values()
        
        if filter_func is not None:
            positions = filter(filter_func, positions)
            
        return [func(position) for position in positions]
    
    def get_position_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summaries for all positions.
        
        Returns:
            List of position summaries
        """
        return [position.get_summary() for position in self.positions.values()]
