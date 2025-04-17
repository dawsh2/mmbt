"""
Position Manager Module

This module provides the PositionManager class that works with standardized
SignalEvent objects and integrates with the RiskManager.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

from src.events.event_bus import Event
from src.events.event_types import EventType
from src.events.signal_event import SignalEvent
from src.position_management.position import EntryType, ExitType
from src.position_management.position_utils import create_entry_action, create_exit_action

# Set up logging
logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages trading positions, sizing, and allocation.
    
    The position manager integrates with risk management to determine appropriate
    position sizes based on signals, market conditions, and portfolio state.
    """
    def __init__(self, portfolio, position_sizer=None,
            allocation_strategy=None, risk_manager=None, max_positions=0, event_bus=None):
        """
        Initialize position manager.

        Args:
            portfolio: Portfolio to manage
            position_sizer: Strategy for determining position sizes
            allocation_strategy: Strategy for allocating capital across instruments
            risk_manager: Risk management component
            max_positions: Maximum number of positions (0 for unlimited)
            event_bus: Event bus for emitting events
        """
        self.portfolio = portfolio
        self.position_sizer = position_sizer
        self.allocation_strategy = allocation_strategy
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        self.event_bus = event_bus  # Store event bus

        # Cache for position decisions
        self.pending_entries = {}  # Symbols pending entry
        self.pending_exits = {}    # Position IDs pending exit
        self.rejected_entries = {} # Symbols with rejected entries

        # For tracking and testing
        self.signal_history = []
        self.orders_generated = []

        # Register with event bus if provided
        if event_bus:
            event_bus.register(EventType.SIGNAL, self.on_signal)
    

    def on_signal(self, event: Event) -> List[Dict[str, Any]]:
        """
        Process a signal event into position actions.

        Args:
            event: Event containing a SignalEvent

        Returns:
            List of position actions
        """
        # Validate event data is a SignalEvent
        if not isinstance(event.data, SignalEvent):
            logger.error(f"Expected SignalEvent in event.data, got {type(event.data)}")
            return []

        signal = event.data
        self.signal_history.append(signal)

        # Get direction directly from SignalEvent's get_signal_value method
        direction = signal.get_signal_value()

        # Skip neutral signals
        if direction == SignalEvent.NEUTRAL:
            logger.debug(f"Skipping neutral signal for {signal.get_symbol()}")
            return []

        symbol = signal.get_symbol()
        price = signal.get_price()
        timestamp = signal.timestamp

        # Check if we already have a position in this symbol
        net_position = 0
        if hasattr(self.portfolio, 'get_net_position'):
            net_position = self.portfolio.get_net_position(symbol)
        elif hasattr(self.portfolio, 'positions_by_symbol') and symbol in self.portfolio.positions_by_symbol:
            # Sum up positions
            for pos in self.portfolio.positions_by_symbol[symbol]:
                net_position += pos.quantity * pos.direction

        actions = []

        if net_position != 0:
            # We have an existing position
            if (direction > 0 and net_position < 0) or (direction < 0 and net_position > 0):
                # Signal direction opposite of current position - exit current position
                if hasattr(self.portfolio, 'positions_by_symbol') and symbol in self.portfolio.positions_by_symbol:
                    for position in self.portfolio.positions_by_symbol[symbol]:
                        # Create exit action
                        exit_action = create_exit_action(
                            position_id=position.position_id,
                            symbol=symbol,
                            price=price,
                            exit_type=ExitType.STRATEGY,
                            reason='Signal direction change',
                            timestamp=timestamp
                        )
                        actions.append(exit_action)

                        logger.info(f"Signal direction change: Exiting {symbol} position {position.position_id}")

        # For new position or adding to existing position
        # Calculate position size using risk manager or position sizer
        position_size = 0

        if self.risk_manager:
            position_size = self.risk_manager.calculate_position_size(signal, self.portfolio, price)
        elif self.position_sizer:
            # Use position sizer with the SignalEvent
            position_size = self.position_sizer.calculate_position_size(signal, self.portfolio, price)
        else:
            # Fallback sizing (simple % of portfolio)
            equity = getattr(self.portfolio, 'equity', 100000)
            risk_pct = 0.01  # Default to 1% risk
            size = (equity * risk_pct) / price
            position_size = size if direction > 0 else -size

        if position_size == 0:
            logger.debug(f"Calculated position size is zero for {symbol}")
            return actions  # Return any exit actions without creating new position

        # Check if we have too many positions
        if self.max_positions > 0:
            current_positions = 0
            if hasattr(self.portfolio, 'positions'):
                current_positions = len(self.portfolio.positions)
            elif hasattr(self.portfolio, 'get_position_count'):
                current_positions = self.portfolio.get_position_count()

            if current_positions >= self.max_positions and net_position == 0:
                logger.warning(f"Maximum positions ({self.max_positions}) reached, skipping entry for {symbol}")
                return actions  # Return any exit actions without creating new position

        # Extract metadata from signal using get_metadata method
        metadata = {}
        if hasattr(signal, 'get_metadata'):
            metadata = signal.get_metadata() or {}

        # Create entry action if we don't have a position or are adding to it
        if net_position == 0 or (net_position > 0 and direction > 0) or (net_position < 0 and direction < 0):
            entry_action = create_entry_action(
                symbol=symbol,
                direction=1 if position_size > 0 else -1,
                size=abs(position_size),
                price=price,
                stop_loss=metadata.get('stop_loss'),
                take_profit=metadata.get('take_profit'),
                strategy_id=signal.get_rule_id() if hasattr(signal, 'get_rule_id') else None,
                entry_type=EntryType.STRATEGY,
                timestamp=timestamp,
                metadata=metadata
            )

            actions.append(entry_action)
            logger.info(f"Creating {'LONG' if direction > 0 else 'SHORT'} position for {symbol}")

        # Emit position action events if we have an event bus
        if self.event_bus:
            for action in actions:
                self.event_bus.emit(Event(EventType.POSITION_ACTION, action))

        return actions

 
    def execute_position_action(self, action: Dict[str, Any], 
                              current_time: datetime.datetime) -> Optional[Dict[str, Any]]:
        """
        Execute a position action.
        
        Args:
            action: Position action dictionary
            current_time: Current timestamp
            
        Returns:
            Result dictionary or None if action failed
        """
        action_type = action.get('action_type', '')  # Use action_type as the key
        
        if action_type == 'entry':
            # Execute entry
            symbol = action['symbol']
            direction = action['direction']
            size = action['size']
            price = action['price']
            
            try:
                # Open position in portfolio
                if hasattr(self.portfolio, 'open_position'):
                    position = self.portfolio.open_position(
                        symbol=symbol,
                        direction=direction,
                        quantity=abs(size),
                        entry_price=price,
                        entry_time=current_time,
                        stop_loss=action.get('stop_loss'),
                        take_profit=action.get('take_profit'),
                        strategy_id=action.get('strategy_id')
                    )
                    
                    result = {
                        'action_type': 'entry',
                        'success': True,
                        'position_id': position.position_id,
                        'symbol': symbol,
                        'direction': direction,
                        'size': size,
                        'price': price,
                        'time': current_time
                    }
                    
                    logger.info(f"Executed entry: {symbol} {'LONG' if direction > 0 else 'SHORT'} "
                               f"{abs(size):.4f} @ {price:.4f}")
                    
                    return result
                else:
                    logger.error("Portfolio does not have open_position method")
                    return None
            except Exception as e:
                logger.error(f"Failed to execute entry: {str(e)}")
                return {
                    'action_type': 'entry',
                    'success': False,
                    'symbol': symbol,
                    'error': str(e)
                }
                
        elif action_type == 'exit':
            # Execute exit
            position_id = action['position_id']
            price = action['price']
            
            try:
                # Close position in portfolio
                if hasattr(self.portfolio, 'close_position'):
                    summary = self.portfolio.close_position(
                        position_id=position_id,
                        exit_price=price,
                        exit_time=current_time,
                        exit_type=action.get('exit_type', ExitType.STRATEGY)
                    )
                    
                    result = {
                        'action_type': 'exit',
                        'success': True,
                        'position_id': position_id,
                        'symbol': summary.get('symbol'),
                        'price': price,
                        'pnl': summary.get('realized_pnl', 0),
                        'time': current_time
                    }
                    
                    logger.info(f"Executed exit: {summary.get('symbol')} "
                               f"{'LONG' if summary.get('direction', 0) > 0 else 'SHORT'} "
                               f"P&L: {summary.get('realized_pnl', 0):.4f}")
                    
                    return result
                else:
                    logger.error("Portfolio does not have close_position method")
                    return None
            except Exception as e:
                logger.error(f"Failed to execute exit: {str(e)}")
                return {
                    'action_type': 'exit',
                    'success': False,
                    'position_id': position_id,
                    'error': str(e)
                }
        
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None
