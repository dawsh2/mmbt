"""
Fixed Position Manager Implementation

This is a fixed version of the position manager that properly handles
signals, creates position actions, and integrates with the portfolio.
"""

import datetime
import logging
from typing import Dict, List, Optional, Any, Union

from src.events.event_bus import Event
from src.events.event_types import EventType
from src.events.signal_event import SignalEvent
from src.position_management.position import EntryType, ExitType

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
        """
        self.portfolio = portfolio
        self.position_sizer = position_sizer
        self.allocation_strategy = allocation_strategy
        self.risk_manager = risk_manager
        self.max_positions = max_positions
        self.event_bus = event_bus

        # Register with event bus if provided
        if event_bus is not None:
            event_bus.register(EventType.SIGNAL, self.on_signal)

        # Cache for position decisions
        self.pending_entries = {}
        self.pending_exits = {}
        self.rejected_entries = {}

        # For tracking and testing
        self.signal_history = []
        self.orders_generated = []



    def on_signal(self, event):
        """
        Process a signal event into position actions.

        Args:
            event: Event containing a SignalEvent in its data field

        Returns:
            List of position actions
        """
        import logging
        from src.events.signal_event import SignalEvent
        logger = logging.getLogger(__name__)

        # Extract signal from event
        if not hasattr(event, 'data'):
            error_msg = f"Event does not have 'data' attribute: {type(event)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        signal = event.data

        # Strictly validate that signal is a SignalEvent
        if not isinstance(signal, SignalEvent):
            error_msg = f"Expected SignalEvent, got {type(signal)}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Extract required data from SignalEvent
        direction = signal.get_signal_value()
        price = signal.get_price()
        symbol = signal.get_symbol()
        rule_id = signal.get_rule_id() if hasattr(signal, 'get_rule_id') else None
        timestamp = getattr(signal, 'timestamp', None)
        metadata = signal.get_metadata() if hasattr(signal, 'get_metadata') else {}

        # Skip neutral signals
        if direction == 0:
            logger.debug(f"Skipping neutral signal for {symbol}")
            return []

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
                        from .position_utils import create_exit_action
                        exit_action = create_exit_action(
                            position_id=position.position_id,
                            symbol=symbol,
                            price=price,
                            exit_type='STRATEGY',
                            reason='Signal direction change',
                            timestamp=timestamp
                        )
                        actions.append(exit_action)

                        logger.info(f"Signal direction change: Exiting {symbol} position {position.position_id}")

        # For new position or adding to existing position
        # Calculate position size using risk manager or position sizer
        position_size = 0

        if hasattr(self, 'risk_manager') and self.risk_manager:
            position_size = self.risk_manager.calculate_position_size(signal, self.portfolio, price)
        elif hasattr(self, 'position_sizer') and self.position_sizer:
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
        if hasattr(self, 'max_positions') and self.max_positions > 0:
            current_positions = 0
            if hasattr(self.portfolio, 'positions'):
                current_positions = len(self.portfolio.positions)
            elif hasattr(self.portfolio, 'get_position_count'):
                current_positions = self.portfolio.get_position_count()

            if current_positions >= self.max_positions and net_position == 0:
                logger.warning(f"Maximum positions ({self.max_positions}) reached, skipping entry for {symbol}")
                return actions  # Return any exit actions without creating new position

        # Create entry action if appropriate
        if net_position == 0 or (net_position > 0 and direction > 0) or (net_position < 0 and direction < 0):
            from .position_utils import create_entry_action
            entry_action = create_entry_action(
                symbol=symbol,
                direction=1 if position_size > 0 else -1,
                size=abs(position_size),
                price=price,
                stop_loss=metadata.get('stop_loss'),
                take_profit=metadata.get('take_profit'),
                strategy_id=rule_id,
                entry_type='STRATEGY',
                timestamp=timestamp,
                metadata=metadata
            )

            actions.append(entry_action)
            logger.info(f"Creating {'LONG' if direction > 0 else 'SHORT'} position for {symbol}")

        # Emit position action events if we have an event bus
        if hasattr(self, 'event_bus') and self.event_bus:
            from src.events.event_base import Event
            from src.events.event_types import EventType

            for action in actions:
                self.event_bus.emit(Event(EventType.POSITION_ACTION, action))

        return actions



    def _process_signal(self, signal):
        """
        Process a signal into position actions.

        Args:
            signal: SignalEvent to process

        Returns:
            List of position action dictionaries
        """
        # Strict type checking
        from src.events.signal_event import SignalEvent
        if not isinstance(signal, SignalEvent):
            error_msg = f"Expected SignalEvent, got {type(signal).__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        actions = []

        # Get signal data using proper type-safe methods
        symbol = signal.get_symbol()
        direction = signal.get_signal_value()  # BUY (1) or SELL (-1)
        price = signal.get_price()
        timestamp = getattr(signal, 'timestamp', datetime.datetime.now())

        # Skip neutral signals
        if direction == 0:
            logger.debug(f"Skipping neutral signal for {symbol}")
            return []

        # Check if we have existing positions in this symbol with opposite direction
        positions_to_exit = []
        net_position = 0

        # Get positions for this symbol
        if hasattr(self.portfolio, 'positions_by_symbol') and symbol in self.portfolio.positions_by_symbol:
            positions = self.portfolio.positions_by_symbol[symbol]

            # First calculate net position
            for pos in positions:
                pos_direction = getattr(pos, 'direction', 0)
                pos_quantity = getattr(pos, 'quantity', 0)

                # Only count positions with actual size
                if pos_quantity > 0:
                    net_position += pos_quantity * pos_direction

                    # Collect positions to exit if signal is opposite direction
                    if (direction > 0 and pos_direction < 0) or (direction < 0 and pos_direction > 0):
                        positions_to_exit.append(pos)

        logger.info(f"Net position for {symbol}: {net_position}, Found {len(positions_to_exit)} positions to exit")

        # Create exit actions for opposite positions
        for position in positions_to_exit:
            # Create exit action
            exit_action = {
                'action_type': 'exit',
                'position_id': position.position_id,
                'symbol': symbol,
                'price': price,
                'exit_type': 'strategy',
                'reason': 'Signal direction change',
                'timestamp': timestamp
            }
            actions.append(exit_action)
            logger.info(f"Signal direction change: Creating exit action for {symbol} position {position.position_id}")

        # For new position or adding to existing position in same direction
        # Only enter if we have no net position in opposite direction
        should_enter = False

        if len(positions_to_exit) > 0:
            # Exit existing opposite positions first
            should_enter = False
        elif direction > 0 and net_position <= 0:
            # Long signal with no net long position
            should_enter = True
        elif direction < 0 and net_position >= 0:
            # Short signal with no net short position
            should_enter = True

        if should_enter:
            # Calculate position size - ensure it's non-zero
            position_size = self._calculate_position_size(signal)

            if position_size != 0:
                # Create entry action
                entry_action = {
                    'action_type': 'entry',
                    'symbol': symbol,
                    'direction': 1 if direction > 0 else -1,
                    'size': abs(position_size),  # Ensure size is positive
                    'price': price,
                    'stop_loss': None,
                    'take_profit': None,
                    'strategy_id': getattr(signal, 'rule_id', None),
                    'timestamp': timestamp
                }
                actions.append(entry_action)
                logger.info(f"Creating {'LONG' if direction > 0 else 'SHORT'} position for {symbol} size {abs(position_size)}")
            else:
                logger.warning(f"Calculated position size is zero - no position created")

        return actions

 

    def _calculate_position_size(self, signal):
        """
        Calculate position size based on signal and portfolio.

        Args:
            signal: Trading signal

        Returns:
            Position size (positive for buy, negative for sell)
        """
        direction = signal.get_signal_value()
        price = signal.get_price()

        # Skip if no valid price
        if price is None or price <= 0:
            logger.warning(f"Invalid price for position sizing: {price}")
            return 0

        # Use a fixed default size if all else fails
        default_size = 10  # Use a reasonable default size

        # Use risk manager if available
        if self.risk_manager:
            try:
                size = self.risk_manager.calculate_position_size(signal, self.portfolio, price)
                if size == 0:
                    size = default_size
                return size
            except Exception as e:
                logger.error(f"Error in risk manager: {e}")
                # Fall through to position sizer

        # Use position sizer if available
        if self.position_sizer:
            try:
                size = self.position_sizer.calculate_position_size(signal, self.portfolio, price)
                if size != 0:
                    return size
                logger.warning("Position sizer returned zero size, using default")
            except Exception as e:
                logger.error(f"Error in position sizer: {e}")
                # Fall through to default sizing

        # Default sizing (simple fixed size)
        logger.info(f"Using default position size: {default_size}")
        return default_size if direction > 0 else -default_size
    
 

    def execute_position_action(self, action, current_time=None):
        """
        Execute a position action.
        
        Args:
            action: Position action dictionary
            current_time: Current timestamp (defaults to now)
            
        Returns:
            Result dictionary or None if action failed
        """
        if current_time is None:
            current_time = datetime.datetime.now()
            
        # Extract action type with flexible handling
        if isinstance(action, dict):
            action_type = action.get('action_type', action.get('action', ''))
        else:
            # Try to get action_type attribute or method
            action_type = getattr(action, 'action_type', None)
            if callable(action_type):
                action_type = action_type()
            elif action_type is None:
                action_type = getattr(action, 'get_action_type', lambda: '')()
                
        logger.info(f"Executing position action: {action_type}")
        
        if action_type == 'entry':
            # Execute entry
            try:
                # Handle both dictionary and object forms
                if isinstance(action, dict):
                    symbol = action.get('symbol')
                    direction = action.get('direction')
                    size = action.get('size')
                    price = action.get('price')
                    stop_loss = action.get('stop_loss')
                    take_profit = action.get('take_profit')
                    strategy_id = action.get('strategy_id')
                else:
                    symbol = getattr(action, 'symbol', None)
                    direction = getattr(action, 'direction', None)
                    size = getattr(action, 'size', None)
                    price = getattr(action, 'price', None)
                    stop_loss = getattr(action, 'stop_loss', None)
                    take_profit = getattr(action, 'take_profit', None)
                    strategy_id = getattr(action, 'strategy_id', None)
                
                # Open position in portfolio - handle different method signatures
                open_method = getattr(self.portfolio, 'open_position', None)
                if open_method is None:
                    open_method = getattr(self.portfolio, '_open_position', None)
                    
                if open_method is None:
                    raise AttributeError("Portfolio has no open_position or _open_position method")
                    
                position = open_method(
                    symbol=symbol,
                    direction=direction,
                    quantity=abs(size),
                    entry_price=price,
                    entry_time=current_time,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy_id=strategy_id
                )
                
                result = {
                    'action_type': 'entry',
                    'success': True,
                    'position_id': getattr(position, 'position_id', None),
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
                    'action_type': 'entry',
                    'success': False,
                    'error': str(e)
                }
                
        elif action_type == 'exit':
            # Execute exit
            try:
                # Handle both dictionary and object forms
                if isinstance(action, dict):
                    position_id = action.get('position_id')
                    price = action.get('price')
                    exit_type = action.get('exit_type', 'strategy')
                else:
                    position_id = getattr(action, 'position_id', None)
                    price = getattr(action, 'price', None)
                    exit_type = getattr(action, 'exit_type', 'strategy')
                
                # Close position in portfolio - handle different method signatures
                close_method = getattr(self.portfolio, 'close_position', None)
                if close_method is None:
                    close_method = getattr(self.portfolio, '_close_position', None)
                    
                if close_method is None:
                    raise AttributeError("Portfolio has no close_position or _close_position method")
                
                summary = close_method(
                    position_id=position_id,
                    exit_price=price,
                    exit_time=current_time,
                    exit_type=exit_type
                )
                
                # Handle different return types from close_position
                if summary is None:
                    summary = {}
                
                result = {
                    'action_type': 'exit',
                    'success': True,
                    'position_id': position_id,
                    'symbol': summary.get('symbol', ''),
                    'price': price,
                    'pnl': summary.get('realized_pnl', 0),
                    'time': current_time
                }
                
                logger.info(f"Executed exit: {summary.get('symbol', '')} "
                           f"PnL: {summary.get('realized_pnl', 0):.4f}")
                
                return result
            except Exception as e:
                logger.error(f"Failed to execute exit: {str(e)}")
                return {
                    'action_type': 'exit',
                    'success': False,
                    'position_id': position_id if 'position_id' in locals() else None,
                    'error': str(e)
                }
        
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None
