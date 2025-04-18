"""
Position Manager Implementation

This module contains the PositionManager class that properly handles signals
and creates trades.
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
        self.event_bus = event_bus

        # Cache for tracking positions
        self.pending_entries = {}
        self.pending_exits = {}
        self.rejected_entries = {}

        # For tracking and testing
        self.signal_history = []
        self.orders_generated = []
        
        # Initialize cache to prevent duplicate action processing
        self._action_cache = set()
        
        logger.info(f"Position manager initialized with portfolio {portfolio.portfolio_id if hasattr(portfolio, 'portfolio_id') else 'unknown'}")

    def on_signal(self, event):
        """
        Process a signal event into position actions.

        Args:
            event: Event containing a SignalEvent

        Returns:
            List of position actions
        """
        # Extract signal with proper validation
        signal = None
        if isinstance(event, Event) and hasattr(event, 'data'):
            signal = event.data
        else:
            signal = event  # Assume it's the signal object directly

        # Validate signal type
        if not isinstance(signal, SignalEvent):
            logger.error(f"Expected SignalEvent, got {type(signal).__name__}")
            return []

        # Add detailed logging
        direction_name = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
        logger.info(f"Position manager received {direction_name} signal for {signal.get_symbol()} at price {signal.get_price()}")

        # Store the signal for history
        self.signal_history.append(signal)

        # Process the signal
        actions = self._process_signal(signal)
        
        # Log created actions
        if actions:
            logger.info(f"Created {len(actions)} position actions: {[action.get('action_type', 'unknown') for action in actions]}")
        else:
            logger.info("No position actions created from signal")

        # Emit position actions if we have an event bus
        if self.event_bus and actions:
            for action in actions:
                # Use the event system
                self.event_bus.emit(Event(EventType.POSITION_ACTION, action))
                logger.info(f"Emitted position action: {action.get('action_type', 'unknown')} for {action.get('symbol', 'unknown')}")

        return actions

    def _process_signal(self, signal):
        """
        Process a signal into position actions.

        Args:
            signal: SignalEvent to process

        Returns:
            List of position action dictionaries
        """
        actions = []

        # Get signal data using proper type-safe methods
        symbol = signal.get_symbol()
        direction = signal.get_signal_value()  # BUY (1) or SELL (-1) or NEUTRAL (0)
        price = signal.get_price()
        timestamp = getattr(signal, 'timestamp', datetime.datetime.now())

        # Skip neutral signals
        if direction == 0:
            logger.debug(f"Skipping neutral signal for {symbol}")
            return []

        # Check for existing positions to avoid duplicates or handle exits
        existing_positions = []
        opposite_positions = []
        positions_in_direction = []

        # Get positions using the most reliable method available
        if hasattr(self.portfolio, 'get_positions_by_symbol'):
            existing_positions = self.portfolio.get_positions_by_symbol(symbol)
        elif hasattr(self.portfolio, 'positions_by_symbol') and symbol in self.portfolio.positions_by_symbol:
            existing_positions = self.portfolio.positions_by_symbol[symbol]

        # Categorize positions by direction
        for pos in existing_positions:
            pos_direction = getattr(pos, 'direction', 0)
            if pos_direction == direction:
                positions_in_direction.append(pos)
            elif pos_direction != 0:  # Opposite direction
                opposite_positions.append(pos)

        # Log current position state
        logger.info(f"Current positions for {symbol}: {len(existing_positions)} total, "
                   f"{len(positions_in_direction)} in same direction, "
                   f"{len(opposite_positions)} in opposite direction")

        # Check if we already have a position in this direction
        if positions_in_direction:
            logger.info(f"Already have position in {'LONG' if direction > 0 else 'SHORT'} direction - skipping entry")
            return []  # Skip creating duplicate position in same direction

        # Calculate position size
        position_size = self._calculate_position_size(signal)
        if position_size == 0:
            logger.info(f"Position sizer returned zero size - skipping")
            return []

        # Check required capital BEFORE creating actions
        capital_required = abs(position_size) * price
        available_capital = 0

        if hasattr(self.portfolio, 'cash'):
            available_capital = self.portfolio.cash
        elif hasattr(self.portfolio, 'get_cash'):
            available_capital = self.portfolio.get_cash()

        if capital_required > available_capital:
            logger.warning(f"Insufficient capital for new position. Required: {capital_required:.2f}, Available: {available_capital:.2f}")
            return []  # Return empty list instead of creating an action that will fail

        # Create exit actions for opposite direction positions
        for position in opposite_positions:
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
            logger.info(f"Created exit action for position {position.position_id} due to signal direction change")

        # Create entry action
        entry_action = {
            'action_type': 'entry',
            'symbol': symbol,
            'direction': direction,  # 1 for BUY, -1 for SELL
            'size': abs(position_size),  # Ensure size is positive
            'price': price,
            'stop_loss': None,  # Could be calculated based on risk parameters
            'take_profit': None,  # Could be calculated based on risk parameters
            'strategy_id': getattr(signal, 'rule_id', None),
            'timestamp': timestamp
        }
        actions.append(entry_action)
        logger.info(f"Created {'LONG' if direction > 0 else 'SHORT'} position entry action for {symbol} size {abs(position_size)}")

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

        # Use risk manager if available
        if self.risk_manager:
            try:
                logger.info("Using risk manager for position sizing")
                size = self.risk_manager.calculate_position_size(signal, self.portfolio, price)
                if size != 0:
                    return size * direction  # Apply direction
                logger.warning("Risk manager returned zero size")
            except Exception as e:
                logger.error(f"Error in risk manager: {e}")
                # Fall through to position sizer

        # Use position sizer if available
        if self.position_sizer:
            try:
                logger.info("Using position sizer for position sizing")
                size = self.position_sizer.calculate_position_size(signal, self.portfolio, price)
                if size != 0:
                    return size * direction  # Apply direction
                logger.warning("Position sizer returned zero size")
            except Exception as e:
                logger.error(f"Error in position sizer: {e}")
                # Fall through to default sizing

        # Default sizing (simple fixed size)
        default_size = 10
        logger.info(f"Using default position size: {default_size}")
        return default_size * direction  # Apply direction

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

        # Extract action type with safe handling
        if isinstance(action, dict):
            action_type = action.get('action_type', '')
        else:
            # Try to get action_type attribute or method
            action_type = getattr(action, 'action_type', None)
            if callable(action_type):
                action_type = action_type()
            elif action_type is None:
                try:
                    action_type = action.get_action_type()
                except:
                    action_type = ''

        logger.info(f"Executing position action: {action_type}")

        # Create a unique ID for this action to prevent duplicate processing
        action_id = None
        if isinstance(action, dict):
            if 'position_id' in action:
                action_id = f"{action_type}_{action.get('position_id')}"
            elif 'symbol' in action and 'direction' in action:
                action_id = f"{action_type}_{action.get('symbol')}_{action.get('direction')}_{current_time}"

        # Check if we've already processed this exact action recently
        if action_id and action_id in self._action_cache:
            logger.warning(f"Duplicate action detected, skipping: {action_id}")
            return {
                'action_type': action_type,
                'success': False,
                'error': 'Duplicate action'
            }

        # Add to action cache before processing
        if action_id:
            self._action_cache.add(action_id)
            # Limit cache size to prevent memory growth
            if len(self._action_cache) > 1000:
                self._action_cache = set(list(self._action_cache)[-500:])

        # Handle entry actions
        if action_type == 'entry':
            # Execute entry
            try:
                # Extract entry parameters safely
                if isinstance(action, dict):
                    symbol = action.get('symbol', '')
                    direction = action.get('direction', 0)
                    size = action.get('size', 0)
                    price = action.get('price', 0)
                    stop_loss = action.get('stop_loss')
                    take_profit = action.get('take_profit')
                    strategy_id = action.get('strategy_id')
                else:
                    symbol = getattr(action, 'symbol', '')
                    direction = getattr(action, 'direction', 0)
                    size = getattr(action, 'size', 0)
                    price = getattr(action, 'price', 0)
                    stop_loss = getattr(action, 'stop_loss', None)
                    take_profit = getattr(action, 'take_profit', None)
                    strategy_id = getattr(action, 'strategy_id', None)

                # Check for required attributes
                if not symbol or direction == 0 or size == 0 or price <= 0:
                    missing = []
                    if not symbol: missing.append('symbol')
                    if direction == 0: missing.append('direction')
                    if size == 0: missing.append('size')
                    if price <= 0: missing.append('price')
                    error_msg = f"Missing required attributes for entry: {', '.join(missing)}"
                    logger.error(error_msg)
                    return {'action_type': 'entry', 'success': False, 'error': error_msg}

                # Check capital before attempting to open position
                if hasattr(self.portfolio, 'cash'):
                    required_capital = abs(size) * price
                    available_capital = self.portfolio.cash

                    if required_capital > available_capital:
                        error_msg = f"Insufficient capital: Required {required_capital:.2f}, Available {available_capital:.2f}"
                        logger.warning(error_msg)
                        return {
                            'action_type': 'entry',
                            'success': False,
                            'error': error_msg
                        }

                # Open position in portfolio - handle different method signatures
                open_method = None
                if hasattr(self.portfolio, 'open_position'):
                    open_method = self.portfolio.open_position
                elif hasattr(self.portfolio, '_open_position'):
                    open_method = self.portfolio._open_position

                if open_method is None:
                    error_msg = "Portfolio has no open_position or _open_position method"
                    logger.error(error_msg)
                    return {'action_type': 'entry', 'success': False, 'error': error_msg}

                # Call the appropriate method
                logger.info(f"Opening position: {symbol} {direction > 0 and 'LONG' or 'SHORT'} {size} @ {price}")
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

                # Return success result
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

                logger.info(f"Successfully executed entry: {symbol} {'LONG' if direction > 0 else 'SHORT'} "
                          f"{abs(size):.4f} @ {price:.4f}")

                return result
            except Exception as e:
                logger.error(f"Failed to execute entry: {str(e)}", exc_info=True)
                return {
                    'action_type': 'entry',
                    'success': False,
                    'error': str(e)
                }

        # Handle exit actions
        elif action_type == 'exit':
            # Execute exit
            try:
                # Extract exit parameters safely
                if isinstance(action, dict):
                    position_id = action.get('position_id', '')
                    price = action.get('price', 0)
                    exit_type = action.get('exit_type', 'strategy')
                else:
                    position_id = getattr(action, 'position_id', '')
                    price = getattr(action, 'price', 0)
                    exit_type = getattr(action, 'exit_type', 'strategy')

                # Check for required attributes
                if not position_id or price <= 0:
                    missing = []
                    if not position_id: missing.append('position_id')
                    if price <= 0: missing.append('price')
                    error_msg = f"Missing required attributes for exit: {', '.join(missing)}"
                    logger.error(error_msg)
                    return {'action_type': 'exit', 'success': False, 'error': error_msg}

                # Check if position exists
                if hasattr(self.portfolio, 'positions'):
                    if position_id not in self.portfolio.positions:
                        logger.info(f"Position {position_id} is already closed or doesn't exist - skipping exit action")
                        return {
                            'action_type': 'exit',
                            'success': False,
                            'position_id': position_id,
                            'error': 'Position already closed or not found'
                        }

                # Find the close method
                close_method = None
                if hasattr(self.portfolio, 'close_position'):
                    close_method = self.portfolio.close_position
                elif hasattr(self.portfolio, '_close_position'):
                    close_method = self.portfolio._close_position

                if close_method is None:
                    error_msg = "Portfolio has no close_position or _close_position method"
                    logger.error(error_msg)
                    return {'action_type': 'exit', 'success': False, 'error': error_msg}

                # Call the close method
                logger.info(f"Closing position: {position_id} @ {price}")
                try:
                    summary = close_method(
                        position_id=position_id,
                        exit_price=price,
                        exit_time=current_time,
                        exit_type=exit_type
                    )
                except Exception as e:
                    # Handle position not found in a cleaner way
                    if "Position not found" in str(e):
                        logger.warning(f"Position not found when trying to close: {position_id}")
                        return {
                            'action_type': 'exit',
                            'success': False,
                            'position_id': position_id,
                            'error': f"Position not found: {position_id}"
                        }
                    else:
                        # Re-raise other exceptions
                        raise

                # Handle different return types from close_position
                if summary is None:
                    logger.warning("close_position returned None")
                    summary = {}

                # Create result with all available information
                result = {
                    'action_type': 'exit',
                    'success': True,
                    'position_id': position_id,
                    'symbol': summary.get('symbol', ''),
                    'price': price,
                    'pnl': summary.get('realized_pnl', 0),
                    'time': current_time
                }

                logger.info(f"Successfully executed exit: {summary.get('symbol', '')} "
                          f"PnL: {summary.get('realized_pnl', 0):.4f}")

                return result
            except Exception as e:
                logger.error(f"Failed to execute exit: {str(e)}", exc_info=True)
                return {
                    'action_type': 'exit',
                    'success': False,
                    'position_id': position_id if 'position_id' in locals() else None,
                    'error': str(e)
                }

        # Handle modify actions
        elif action_type == 'modify':
            # Execute position modification
            try:
                # Extract modify parameters safely
                if isinstance(action, dict):
                    position_id = action.get('position_id', '')
                    stop_loss = action.get('stop_loss')
                    take_profit = action.get('take_profit')
                else:
                    position_id = getattr(action, 'position_id', '')
                    stop_loss = getattr(action, 'stop_loss', None)
                    take_profit = getattr(action, 'take_profit', None)

                # Check for required attributes
                if not position_id:
                    error_msg = "Missing required attribute for modify: position_id"
                    logger.error(error_msg)
                    return {'action_type': 'modify', 'success': False, 'error': error_msg}

                # Check if either stop_loss or take_profit is provided
                if stop_loss is None and take_profit is None:
                    error_msg = "Modify action must include either stop_loss or take_profit"
                    logger.error(error_msg)
                    return {'action_type': 'modify', 'success': False, 'error': error_msg}

                # Check if position exists
                if hasattr(self.portfolio, 'positions'):
                    if position_id not in self.portfolio.positions:
                        logger.info(f"Position {position_id} not found - skipping modify action")
                        return {
                            'action_type': 'modify',
                            'success': False,
                            'position_id': position_id,
                            'error': 'Position not found'
                        }

                # Get position from portfolio
                position = self.portfolio.positions.get(position_id)
                if position is None:
                    error_msg = f"Position not found: {position_id}"
                    logger.error(error_msg)
                    return {'action_type': 'modify', 'success': False, 'error': error_msg}

                # Perform modifications
                result = {'action_type': 'modify', 'success': True, 'position_id': position_id}

                if stop_loss is not None:
                    if hasattr(position, 'modify_stop_loss'):
                        stop_result = position.modify_stop_loss(stop_loss)
                        result['stop_loss'] = stop_loss
                        result['old_stop_loss'] = stop_result.get('old_stop_loss')
                        logger.info(f"Modified stop loss for {position_id}: {stop_loss}")
                    else:
                        logger.warning(f"Position {position_id} does not support modify_stop_loss")

                if take_profit is not None:
                    if hasattr(position, 'modify_take_profit'):
                        tp_result = position.modify_take_profit(take_profit)
                        result['take_profit'] = take_profit
                        result['old_take_profit'] = tp_result.get('old_take_profit')
                        logger.info(f"Modified take profit for {position_id}: {take_profit}")
                    else:
                        logger.warning(f"Position {position_id} does not support modify_take_profit")

                return result
            except Exception as e:
                logger.error(f"Failed to execute modify: {str(e)}", exc_info=True)
                return {
                    'action_type': 'modify',
                    'success': False,
                    'position_id': position_id if 'position_id' in locals() else None,
                    'error': str(e)
                }
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return {'action_type': action_type, 'success': False, 'error': f"Unknown action type: {action_type}"}

    def reset(self):
        """Reset position manager state."""
        self.pending_entries = {}
        self.pending_exits = {}
        self.rejected_entries = {}
        self.signal_history = []
        self.orders_generated = []
        self._action_cache = set()
        logger.info("Position manager reset")
