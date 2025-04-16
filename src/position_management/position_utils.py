"""
Position Management Utilities Module

This module provides utility functions for working with positions and position management
in the trading system. It standardizes position actions and calculations.
"""

import datetime
from typing import Dict, List, Any, Optional, Union

from src.events.signal_event import SignalEvent
from src.events.event_utils import get_signal_direction
from src.position_management.position import ExitType, EntryType


def get_signal_direction(signal_event: SignalEvent) -> int:
    """
    Extract direction from a signal event.
    
    Args:
        signal_event: Signal event
        
    Returns:
        Direction as an integer (1 for buy/long, -1 for sell/short, 0 for neutral)
    """
    # Use the direction property if available
    if hasattr(signal_event, 'direction'):
        return signal_event.direction
    
    # Extract from signal data
    signal_data = signal_event.data
    
    # Check different fields where direction might be specified
    if 'direction' in signal_data:
        direction = signal_data['direction']
        if isinstance(direction, int):
            return 1 if direction > 0 else -1 if direction < 0 else 0
        elif isinstance(direction, str):
            if direction.upper() in ['BUY', 'LONG']:
                return 1
            elif direction.upper() in ['SELL', 'SHORT']:
                return -1
            else:
                return 0
    
    if 'signal_type' in signal_data:
        signal_type = signal_data['signal_type']
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


def create_position_action(action: str, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized position action dictionary.
    
    Args:
        action: Action type ('entry', 'exit', 'modify')
        **kwargs: Action-specific parameters
        
    Returns:
        Standardized position action dictionary
    """
    # Create base action
    action_dict = {
        'action': action,
        'timestamp': kwargs.get('timestamp', datetime.datetime.now())
    }
    
    # Add action-specific parameters
    action_dict.update(kwargs)
    
    return action_dict


def create_entry_action(symbol: str, direction: int, size: float, price: float,
                      stop_loss: Optional[float] = None, 
                      take_profit: Optional[float] = None,
                      strategy_id: Optional[str] = None,
                      entry_type: EntryType = EntryType.MARKET,
                      timestamp: Optional[datetime.datetime] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized entry action.
    
    Args:
        symbol: Instrument symbol
        direction: Position direction (1 for long, -1 for short)
        size: Position size
        price: Entry price
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        strategy_id: Optional strategy ID
        entry_type: Type of entry
        timestamp: Action timestamp
        metadata: Additional action metadata
        
    Returns:
        Entry action dictionary
    """
    action_data = {
        'symbol': symbol,
        'direction': direction,
        'size': size,
        'price': price,
        'entry_type': entry_type
    }
    
    # Add optional parameters if provided
    if stop_loss is not None:
        action_data['stop_loss'] = stop_loss
        
    if take_profit is not None:
        action_data['take_profit'] = take_profit
        
    if strategy_id is not None:
        action_data['strategy_id'] = strategy_id
        
    if metadata is not None:
        action_data['metadata'] = metadata
    
    return create_position_action('entry', timestamp=timestamp, **action_data)


def create_exit_action(position_id: str, symbol: str, price: float,
                     exit_type: ExitType = ExitType.MARKET,
                     reason: Optional[str] = None,
                     timestamp: Optional[datetime.datetime] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a standardized exit action.
    
    Args:
        position_id: ID of position to exit
        symbol: Instrument symbol
        price: Exit price
        exit_type: Type of exit
        reason: Optional reason for exit
        timestamp: Action timestamp
        metadata: Additional action metadata
        
    Returns:
        Exit action dictionary
    """
    action_data = {
        'position_id': position_id,
        'symbol': symbol,
        'price': price,
        'exit_type': exit_type
    }
    
    # Add optional parameters if provided
    if reason is not None:
        action_data['reason'] = reason
        
    if metadata is not None:
        action_data['metadata'] = metadata
    
    return create_position_action('exit', timestamp=timestamp, **action_data)


def calculate_position_size(signal_event: SignalEvent, portfolio, risk_pct: float = 0.01) -> float:
    """
    Calculate position size based on risk percentage.
    
    Args:
        signal_event: Signal event
        portfolio: Portfolio instance
        risk_pct: Percentage of portfolio to risk (0.01 = 1%)
        
    Returns:
        Position size
    """
    # Get portfolio equity
    equity = getattr(portfolio, 'equity', 0)
    if equity <= 0:
        return 0
        
    # Get price from signal
    price = signal_event.price if hasattr(signal_event, 'price') else signal_event.data.get('price', 0)
    if price <= 0:
        return 0
        
    # Get direction
    direction = get_signal_direction(signal_event)
    if direction == 0:
        return 0
        
    # Calculate risk amount
    risk_amount = equity * risk_pct
    
    # Get stop loss from signal metadata if available
    metadata = signal_event.metadata if hasattr(signal_event, 'metadata') else signal_event.data.get('metadata', {})
    stop_loss = metadata.get('stop_loss')
    
    if stop_loss is not None:
        # Calculate size based on stop distance
        stop_distance = abs(price - stop_loss)
        if stop_distance > 0:
            size = risk_amount / stop_distance
        else:
            # Default to percentage of equity if stop distance is zero
            size = (equity * risk_pct) / price
    else:
        # Default to percentage of equity if no stop loss provided
        size = (equity * risk_pct) / price
    
    return size * direction


def calculate_risk_reward_ratio(entry_price: float, stop_loss: Optional[float], 
                              take_profit: Optional[float], direction: int) -> Optional[float]:
    """
    Calculate risk-reward ratio for a position.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        direction: Position direction (1 for long, -1 for short)
        
    Returns:
        Risk-reward ratio or None if not calculable
    """
    if stop_loss is None or take_profit is None:
        return None
        
    # Calculate risk and reward
    if direction > 0:  # Long
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # Short
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        
    # Ensure risk is positive
    if risk <= 0:
        return None
        
    return reward / risk


def signal_to_position_action(signal_event: SignalEvent, portfolio) -> Optional[Dict[str, Any]]:
    """
    Convert a signal event to a position action.
    
    Args:
        signal_event: Signal event
        portfolio: Portfolio instance
        
    Returns:
        Position action dictionary or None if no action needed
    """
    # Get direction from signal
    direction = get_signal_direction(signal_event)
    if direction == 0:
        return None
        
    # Get price from signal
    price = signal_event.price if hasattr(signal_event, 'price') else signal_event.data.get('price', 0)
    if price <= 0:
        return None
        
    # Get symbol from signal
    symbol = signal_event.symbol if hasattr(signal_event, 'symbol') else signal_event.data.get('symbol', 'default')
    
    # Calculate position size
    size = calculate_position_size(signal_event, portfolio)
    
    # Get metadata from signal
    metadata = signal_event.metadata if hasattr(signal_event, 'metadata') else signal_event.data.get('metadata', {})
    
    # Get stop loss and take profit if available
    stop_loss = metadata.get('stop_loss')
    take_profit = metadata.get('take_profit')
    
    # Create entry action
    return create_entry_action(
        symbol=symbol,
        direction=direction,
        size=size,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_id=signal_event.rule_id if hasattr(signal_event, 'rule_id') else None,
        timestamp=signal_event.timestamp,
        metadata=metadata
    )
