"""
Modified Version of Position Manager

This is a modified version of the position_manager.py file,
updated to correctly handle SignalEvent objects and perform proper type checking.
"""

import datetime
import logging
import uuid
from typing import Dict, List, Any, Optional, Union

from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType 
from src.events.signal_event import SignalEvent
from src.position_management.position import EntryType, ExitType
from src.events.portfolio_events import PositionActionEvent

# Set up logging
logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages trading positions, sizing, and allocation.
    
    The position manager integrates position sizing, allocation, and risk
    management to determine appropriate position sizes based on signals,
    market conditions, and portfolio state.
    
    This updated version works directly with SignalEvent objects with proper type checking.
    """
    
    def __init__(self, portfolio, position_sizer=None,
                allocation_strategy=None, risk_manager=None, max_positions=0,
                event_bus=None):
        """
        Initialize position manager.
        
        Args:
            portfolio: Portfolio to manage
            position_sizer: Strategy for determining position sizes
            allocation_strategy: Strategy for allocating capital across instruments
            risk_manager: Risk management component
            max_positions: Maximum number of positions (0 for unlimited)
            event_bus: Optional event bus for emitting events
        """
        self.portfolio = portfolio
        self.position_sizer = position_sizer
        self.allocation_strategy = allocation_strategy
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        self.event_bus = event_bus
        
        # Cache for position decisions
        self.pending_entries = {}  # Symbols pending entry
        self.pending_exits = {}    # Position IDs pending exit
        self.rejected_entries = {} # Symbols with rejected entries
        
        # For tracking and testing
        self.signal_history = []
        self.orders_generated = []

    def on_signal(self, event_or_signal) -> List[PositionActionEvent]:
        """
        Process a signal and determine position actions.

        Args:
            event_or_signal: Either Event containing a SignalEvent or a SignalEvent directly

        Returns:
            List of position action events
        """
        # Extract the SignalEvent with proper type checking
        signal = None
        
        # Case 1: Argument is an Event with a SignalEvent in data attribute
        if isinstance(event_or_signal, Event):
            if hasattr(event_or_signal, 'data') and isinstance(event_or_signal.data, SignalEvent):
                signal = event_or_signal.data
            else:
                logger.error(f"Expected Event with SignalEvent in data, got {type(event_or_signal.data)}")
                return []
        # Case 2: Argument is a SignalEvent directly
        elif isinstance(event_or_signal, SignalEvent):
            signal = event_or_signal
        else:
            logger.error(f"Expected Event or SignalEvent, got {type(event_or_signal)}")
            return []

        # Store signal for tracking
        self.signal_history.append(signal)

        # Process signal and get position actions
        actions = self._process_signal(signal)

        # Emit position actions if we have an event bus
        if self.event_bus:
            for action in actions:
                action_event = Event(EventType.POSITION_ACTION, action)
                self.event_bus.emit(action_event)

        return actions

    def _process_signal(self, signal: SignalEvent) -> List[Any]:
        """
        Process a signal into position actions.

        Args:
            signal: SignalEvent to process

        Returns:
            List of PositionActionEvent objects
        """
        actions = []

        # Get signal data using proper type-safe methods
        symbol = signal.get_symbol()
        direction = signal.get_signal_value()  # BUY (1) or SELL (-1)
        price = signal.get_price()
        timestamp = signal.timestamp

        # Skip neutral signals
        if direction == SignalEvent.NEUTRAL:
            logger.debug(f"Skipping neutral signal for {symbol}")
            return []

        # Check if we already have a position in this symbol
        net_position = self.portfolio.get_net_position(symbol) if hasattr(self.portfolio, 'get_net_position') else 0

        if net_position != 0:
            # We have an existing position
            if (direction > 0 and net_position < 0) or (direction < 0 and net_position > 0):
                # Signal direction opposite of current position - exit current position
                positions = self.portfolio.get_positions_by_symbol(symbol) if hasattr(self.portfolio, 'get_positions_by_symbol') else []

                for position in positions:
                    # Create exit action
                    action = PositionActionEvent(
                        action_type='exit',
                        symbol=symbol,
                        position_id=position.position_id,
                        price=price,
                        exit_type=getattr(self, 'ExitType', {}).get('STRATEGY', 'strategy'),
                        exit_reason="Signal direction change",
                        timestamp=timestamp
                    )

                    actions.append(action)
                    logger.info(f"Generated exit action for {symbol} position {position.position_id}")

        # Evaluate new entry based on signal direction
        if (direction > 0 and net_position <= 0) or (direction < 0 and net_position >= 0):
            # Calculate position size
            size = 0
            if self.position_sizer:
                size = self.position_sizer.calculate_position_size(signal, self.portfolio, price)
            else:
                # Default sizing logic if no position sizer provided
                equity = getattr(self.portfolio, 'equity', 100000)
                size = (equity * 0.01) / price  # Default to 1% of equity
                size = size if direction > 0 else -size

            # Only create entry action if size is non-zero
            if size != 0:
                # Get stop loss and take profit from signal metadata if available
                metadata = signal.get_metadata() if hasattr(signal, 'get_metadata') else {}
                stop_loss = metadata.get('stop_loss')
                take_profit = metadata.get('take_profit')

                # Create entry action
                action = PositionActionEvent(
                    action_type='entry',
                    symbol=symbol,
                    direction=direction,
                    size=abs(size),
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_id=signal.get_rule_id() if hasattr(signal, 'get_rule_id') else None,
                    timestamp=timestamp
                )

                actions.append(action)
                logger.info(f"Generated entry action for {symbol} direction {direction} size {abs(size)}")

        return actions

    def calculate_position_size(self, signal, portfolio, current_price=None):
        """
        Calculate appropriate position size for a signal.

        Args:
            signal: SignalEvent
            portfolio: Portfolio state
            current_price: Optional override for current price

        Returns:
            Position size (positive for long, negative for short)
        """
        # Proper type checking for SignalEvent
        if not isinstance(signal, SignalEvent):
            logger.error(f"Expected SignalEvent, got {type(signal)}")
            return 0
            
        # Skip neutral signals
        if signal.get_signal_value() == SignalEvent.NEUTRAL:
            return 0

        # Get direction directly from signal
        direction = signal.get_signal_value()

        # Use provided price or signal price
        price = current_price
        if price is None:
            price = signal.get_price()

        if price is None or price <= 0:
            logger.warning(f"Invalid price for position sizing: {price}")
            return 0

        # Use position sizer if available
        if self.position_sizer:
            size = self.position_sizer.calculate_position_size(signal, portfolio, price)
            return size * direction if size > 0 else 0

        # Default sizing if no position sizer available
        equity = getattr(portfolio, 'equity', 100000)
        risk_pct = 0.01  # Default to 1% risk

        # Get stop loss from metadata if available
        metadata = signal.get_metadata() if hasattr(signal, 'get_metadata') else {}
        stop_loss = metadata.get('stop_loss')

        if stop_loss:
            # Calculate risk-based position size
            risk_amount = equity * risk_pct
            stop_distance = abs(price - stop_loss)

            if stop_distance > 0:
                size = risk_amount / stop_distance
            else:
                size = (equity * risk_pct) / price
        else:
            # Default to percentage of equity
            size = (equity * risk_pct) / price

        return size * direction

    def execute_position_action(self, action: Dict[str, Any], 
                             current_time: datetime.datetime) -> Optional[Dict[str, Any]]:
        """
        Execute a position action.
        
        Args:
            action: Position action dictionary or PositionActionEvent
            current_time: Current timestamp
            
        Returns:
            Result dictionary or None if action failed
        """
        # Extract action type with proper type checking
        if isinstance(action, PositionActionEvent):
            action_type = action.get_action_type()
            action_data = action.data
        elif isinstance(action, dict):
            action_type = action.get('action_type', action.get('action', ''))
            action_data = action
        else:
            logger.error(f"Unsupported action type: {type(action)}")
            return None
        
        if action_type == 'entry':
            # Execute entry
            symbol = action_data.get('symbol')
            direction = action_data.get('direction')
            size = action_data.get('size')
            price = action_data.get('price')
            
            try:
                # Open position in portfolio
                position = self.portfolio.open_position(
                    symbol=symbol,
                    direction=direction,
                    quantity=abs(size),
                    entry_price=price,
                    entry_time=current_time,
                    stop_loss=action_data.get('stop_loss'),
                    take_profit=action_data.get('take_profit'),
                    strategy_id=action_data.get('strategy_id')
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
            position_id = action_data.get('position_id')
            price = action_data.get('price')
            
            try:
                # Close position in portfolio
                summary = self.portfolio.close_position(
                    position_id=position_id,
                    exit_price=price,
                    exit_time=current_time,
                    exit_type=action_data.get('exit_type', ExitType.STRATEGY)
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
