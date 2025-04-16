"""
Position Manager Module (Updated)

This module provides the updated PositionManager class that works with standardized
SignalEvent objects rather than legacy Signal objects.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

from src.events.event_bus import Event
from src.events.event_types import EventType
from src.events.signal_event import SignalEvent
from src.position_management.position import EntryType, ExitType
from src.position_management.position_utils import get_signal_direction, create_entry_action, create_exit_action

# Set up logging
logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages trading positions, sizing, and allocation.
    
    The position manager integrates position sizing, allocation, and risk
    management to determine appropriate position sizes based on signals,
    market conditions, and portfolio state.
    
    This updated version works directly with SignalEvent objects rather than
    legacy Signal objects wrapped in Events.
    """
    
    def __init__(self, portfolio, position_sizer=None,
                allocation_strategy=None, risk_manager=None, max_positions=0):
        """
        Initialize position manager.
        
        Args:
            portfolio: Portfolio to manage
            position_sizer: Strategy for determining position sizes
            allocation_strategy: Strategy for allocating capital across instruments
            risk_manager: Risk management component
            max_positions: Maximum number of positions (0 for unlimited)
        """
        self.portfolio = portfolio
        self.position_sizer = position_sizer
        self.allocation_strategy = allocation_strategy
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        
        # Cache for position decisions
        self.pending_entries = {}  # Symbols pending entry
        self.pending_exits = {}    # Position IDs pending exit
        self.rejected_entries = {} # Symbols with rejected entries
        
        # For tracking and testing
        self.signal_history = []
        self.orders_generated = []

    def _process_signal(self, signal_event):
        """Process a signal into position action events."""
        # Determine action to take
        symbol = signal_event.symbol
        direction = signal_event.direction

        # Create and emit position action event
        action_event = PositionActionEvent(
            action_type='entry',
            symbol=symbol,
            direction=direction,
            size=100,  # Calculate appropriate size
            price=signal_event.price,
            # Other parameters...
        )

        self.event_bus.emit(action_event)        
    

    # In position_manager_updated.py
    def _process_signal(self, signal_event: SignalEvent) -> List[Dict[str, Any]]:
        """
        Process a signal event into position actions.

        Args:
            signal_event: Signal event

        Returns:
            List of position action dictionaries
        """
        actions = []

        # Get signal direction
        direction = get_signal_direction(signal_event)
        symbol = signal_event.symbol
        price = signal_event.price

        # Check if we already have a position in this symbol
        net_position = self.portfolio.get_net_position(symbol)

        if net_position != 0:
            # We have an existing position
            if (direction > 0 and net_position < 0) or (direction < 0 and net_position > 0):
                # Signal direction opposite of current position - exit current position
                for position in self.portfolio.positions_by_symbol.get(symbol, []):
                    # Create and emit exit action event
                    exit_event = PositionActionEvent(
                        action_type='exit',
                        position_id=position.position_id,
                        symbol=symbol,
                        price=price,
                        exit_type=ExitType.STRATEGY,
                        reason='Signal direction change',
                        timestamp=signal_event.timestamp
                    )

                    self.event_bus.emit(exit_event)
                    actions.append({'action': 'exit', 'position_id': position.position_id})
        else:
            # No existing position - evaluate new entry
            # Calculate position size
            size = 0
            if self.position_sizer:
                size = self.position_sizer.calculate_position_size(
                    signal_event, self.portfolio, price
                )
            else:
                equity = getattr(self.portfolio, 'equity', 100000)
                size = (equity * 0.01) / price * direction

            # Create and emit entry action event
            if size != 0:
                metadata = signal_event.metadata.copy() if hasattr(signal_event, 'metadata') else {}
                entry_event = PositionActionEvent(
                    action_type='entry',
                    symbol=symbol,
                    direction=direction,
                    size=abs(size),
                    price=price,
                    stop_loss=metadata.get('stop_loss'),
                    take_profit=metadata.get('take_profit'),
                    strategy_id=signal_event.rule_id if hasattr(signal_event, 'rule_id') else None,
                    timestamp=signal_event.timestamp,
                    metadata=metadata
                )

                self.event_bus.emit(entry_event)
                actions.append({'action': 'entry', 'symbol': symbol, 'direction': direction})

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
        action_type = action.get('action', '')
        
        if action_type == 'entry':
            # Execute entry
            symbol = action['symbol']
            direction = action['direction']
            size = action['size']
            price = action['price']
            
            try:
                # Open position in portfolio
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
                    'action': 'entry',
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
            except Exception as e:
                logger.error(f"Failed to execute entry: {str(e)}")
                return {
                    'action': 'entry',
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
                summary = self.portfolio.close_position(
                    position_id=position_id,
                    exit_price=price,
                    exit_time=current_time,
                    exit_type=action.get('exit_type', ExitType.STRATEGY)
                )
                
                result = {
                    'action': 'exit',
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
            except Exception as e:
                logger.error(f"Failed to execute exit: {str(e)}")
                return {
                    'action': 'exit',
                    'success': False,
                    'position_id': position_id,
                    'error': str(e)
                }
        
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None
