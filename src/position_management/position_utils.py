"""
Position Utilities Module

This module provides utility functions for working with position management components,
particularly for handling events and creating standardized position actions.
"""

from typing import Tuple, Dict, Any, Optional, List, Union
import datetime
from src.signals import Signal, SignalType

# should be recycled from event_utils 
def unpack_signal_event(event) -> Tuple[Any, str, float, Any]:
    """
    Extract signal data from an event object.
    
    Args:
        event: Event object containing signal data
        
    Returns:
        tuple: (signal, symbol, price, signal_type)
    """
    if not hasattr(event, 'data'):
        raise TypeError(f"Expected Event object with data attribute")
        
    signal = event.data
    if not hasattr(signal, 'signal_type'):
        raise TypeError(f"Expected Signal object in event.data")
        
    symbol = signal.symbol if hasattr(signal, 'symbol') else 'default'
    price = signal.price if hasattr(signal, 'price') else None
    signal_type = signal.signal_type
    
    return signal, symbol, price, signal_type


def create_position_action(
    action_type: str, 
    symbol: str, 
    **kwargs
) -> Dict[str, Any]:
    """
    Create a standardized position action dictionary.
    
    Args:
        action_type: Type of action ('entry', 'exit', 'modify')
        symbol: Instrument symbol
        **kwargs: Additional action parameters
        
    Returns:
        dict: Standardized position action
    """
    action = {
        'action': action_type,
        'symbol': symbol,
        **kwargs
    }
    return action


def create_entry_action(
    symbol: str,
    direction: int,
    size: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized entry position action.
    
    Args:
        symbol: Instrument symbol
        direction: Position direction (1 for long, -1 for short)
        size: Position size
        price: Entry price
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        metadata: Optional additional metadata
        
    Returns:
        dict: Entry action
    """
    action = create_position_action(
        action_type='entry',
        symbol=symbol,
        direction=direction,
        size=size,
        price=price
    )
    
    if stop_loss is not None:
        action['stop_loss'] = stop_loss
        
    if take_profit is not None:
        action['take_profit'] = take_profit
        
    if metadata:
        action['metadata'] = metadata
        
    return action


def create_exit_action(
    position_id: str,
    symbol: str,
    price: float,
    reason: str = 'strategy_signal',
    exit_type: Any = None
) -> Dict[str, Any]:
    """
    Create a standardized exit position action.
    
    Args:
        position_id: ID of position to exit
        symbol: Instrument symbol
        price: Exit price
        reason: Reason for exit
        exit_type: Type of exit (e.g., ExitType enum value)
        
    Returns:
        dict: Exit action
    """
    return create_position_action(
        action_type='exit',
        symbol=symbol,
        position_id=position_id,
        price=price,
        reason=reason,
        exit_type=exit_type
    )


def create_modify_action(
    position_id: str,
    symbol: str,
    **modifications
) -> Dict[str, Any]:
    """
    Create a standardized position modification action.
    
    Args:
        position_id: ID of position to modify
        symbol: Instrument symbol
        **modifications: Modifications to apply (stop_loss, take_profit, etc.)
        
    Returns:
        dict: Modify action
    """
    return create_position_action(
        action_type='modify',
        symbol=symbol,
        position_id=position_id,
        **modifications
    )


def calculate_position_size(
    equity: float,
    price: float,
    risk_pct: float = 0.02,
    stop_distance_pct: Optional[float] = None,
    stop_price: Optional[float] = None,
    max_position_pct: float = 0.25
) -> float:
    """
    Calculate position size based on risk parameters.
    
    Args:
        equity: Account equity
        price: Current price
        risk_pct: Percentage of equity to risk (0.02 = 2%)
        stop_distance_pct: Percentage distance to stop (alternative to stop_price)
        stop_price: Explicit stop price (alternative to stop_distance_pct)
        max_position_pct: Maximum position size as percentage of equity
        
    Returns:
        float: Position size in units
    """
    # Calculate risk amount
    risk_amount = equity * risk_pct
    
    # Calculate stop distance
    if stop_price is not None:
        stop_distance = abs(price - stop_price)
    elif stop_distance_pct is not None:
        stop_distance = price * stop_distance_pct
    else:
        # Default to 2% stop distance
        stop_distance = price * 0.02
    
    # Calculate position size based on risk
    if stop_distance <= 0:
        return 0
    
    position_size = risk_amount / stop_distance
    
    # Limit by maximum position size
    max_size = (equity * max_position_pct) / price
    position_size = min(position_size, max_size)
    
    return position_size


def get_signal_direction(signal) -> int:
    """
    Extract direction from a signal.
    
    Args:
        signal: Signal object or dictionary
        
    Returns:
        Direction as an integer (1 for buy/long, -1 for sell/short, 0 for neutral)
    """
    # Handle Signal object
    if hasattr(signal, 'signal_type'):
        # Get direction from signal_type (assuming it has a value attribute)
        if hasattr(signal.signal_type, 'value'):
            return signal.signal_type.value
        elif hasattr(signal, 'direction'):
            return signal.direction
        return 0  # Default to neutral
    
    # Handle dictionary (legacy code)
    if isinstance(signal, dict):
        # Check different fields where direction might be specified
        if 'direction' in signal:
            direction = signal['direction']
            if isinstance(direction, int):
                return 1 if direction > 0 else -1 if direction < 0 else 0
            elif isinstance(direction, str):
                if direction.upper() in ['BUY', 'LONG']:
                    return 1
                elif direction.upper() in ['SELL', 'SHORT']:
                    return -1
                else:
                    return 0
        
        if 'signal_type' in signal:
            signal_type = signal['signal_type']
            if hasattr(signal_type, 'value'):
                # Enum with value attribute
                return signal_type.value
            elif isinstance(signal_type, str):
                if signal_type.upper() in ['BUY', 'LONG']:
                    return 1
                elif signal_type.upper() in ['SELL', 'SHORT']:
                    return -1
                else:
                    return 0
    
    # Default to neutral if direction cannot be determined
    return 0


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: int
) -> float:
    """
    Calculate risk-reward ratio for a position.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        direction: Position direction (1 for long, -1 for short)
        
    Returns:
        float: Risk-reward ratio (reward divided by risk)
    """
    if direction > 0:  # Long position
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # Short position
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        
    if risk <= 0:
        return 0.0
        
    return reward / risk
