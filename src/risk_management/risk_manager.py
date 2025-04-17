"""
Risk manager module for implementing MAE, MFE, and ETD based risk management.

This module provides a RiskManager class that can be used to apply data-driven
risk management rules to any trading strategy, using parameters derived from
historical trade analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from src.risk_management.types import RiskParameters, ExitReason, TradeMetrics
from src.risk_management.collector import RiskMetricsCollector
from src.events.signal_event import SignalEvent


class RiskManager:
    """
    Applies risk management rules based on MAE, MFE, and ETD analysis.
    
    This class applies stop-loss, take-profit, trailing stop, and time-based
    exit rules to trading positions, using parameters derived from historical
    trade analysis. Additionally, it handles position sizing through integration
    with position sizers.
    """
    
    def __init__(self, 
                risk_params: RiskParameters,
                position_sizer: Optional[Any] = None,
                metrics_collector: Optional[RiskMetricsCollector] = None,
                position_size_calculator: Optional[Callable] = None):
        """
        Initialize the risk manager.
        
        Args:
            risk_params: Risk parameters for stop-loss, take-profit, etc.
            position_sizer: Optional position sizer for calculating position sizes
            metrics_collector: Optional collector for tracking trade metrics
            position_size_calculator: Optional function for calculating position sizes (legacy)
        """
        self.risk_params = risk_params
        self.metrics_collector = metrics_collector
        self.position_sizer = position_sizer
        self.position_size_calculator = position_size_calculator
        
        # Active trade tracking
        self.active_trades = {}  # trade_id -> trade details
        
    def calculate_position_size(self, signal: SignalEvent, 
                              portfolio: Any, 
                              current_price: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters and signal.
        
        Args:
            signal: Trading signal event
            portfolio: Portfolio or dictionary with equity information
            current_price: Optional current price (uses signal price if None)
            
        Returns:
            Position size (positive for buy, negative for sell)
        """
        # Get price from signal if not provided
        if current_price is None and hasattr(signal, 'price'):
            current_price = signal.price
        
        if current_price is None or current_price <= 0:
            return 0
        
        # Use configured position sizer if available
        if self.position_sizer and hasattr(self.position_sizer, 'calculate_position_size'):
            return self.position_sizer.calculate_position_size(signal, portfolio, current_price)
            
        # Legacy position size calculator
        if self.position_size_calculator:
            try:
                return self.position_size_calculator(
                    direction=signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0,
                    entry_price=current_price,
                    stop_price=self._calculate_stop_price(current_price, signal),
                    target_price=self._calculate_target_price(current_price, signal),
                    risk_reward_ratio=self.risk_params.risk_reward_ratio,
                    expected_win_rate=self.risk_params.expected_win_rate
                )
            except Exception as e:
                print(f"Error calculating position size: {str(e)}")
                # Fall through to default calculation
        
        # Default sizing based on risk parameters
        # Get direction from signal
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        
        # Get equity
        equity = getattr(portfolio, 'equity', 0)
        if hasattr(portfolio, 'get_equity'):
            equity = portfolio.get_equity()
        elif isinstance(portfolio, dict) and 'equity' in portfolio:
            equity = portfolio['equity']
            
        if equity <= 0:
            return 0
        
        # Calculate risk amount
        risk_per_trade = getattr(self.risk_params, 'risk_per_trade_pct', 1.0)
        risk_amount = equity * (risk_per_trade / 100)
        
        # Calculate stop distance
        if self.risk_params.stop_loss_pct > 0:
            # For long positions
            if direction > 0:
                stop_price = current_price * (1 - self.risk_params.stop_loss_pct / 100)
                stop_distance = current_price - stop_price
            # For short positions
            elif direction < 0:
                stop_price = current_price * (1 + self.risk_params.stop_loss_pct / 100)
                stop_distance = stop_price - current_price
            else:
                return 0  # Neutral signal
                
            # Calculate position size based on risk amount and stop distance
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
            else:
                position_size = 0
        else:
            # Fallback to percentage of equity
            position_size = (equity * 0.01) / current_price if current_price > 0 else 0
        
        # Apply maximum position size constraint if specified
        max_position_pct = getattr(self.risk_params, 'max_position_pct', 0.25)
        max_position_size = (equity * max_position_pct) / current_price
        position_size = min(position_size, max_position_size)
        
        # Apply direction
        return position_size * direction
    
    def _calculate_stop_price(self, current_price: float, signal: SignalEvent) -> Optional[float]:
        """
        Calculate stop price based on risk parameters and signal.
        
        Args:
            current_price: Current market price
            signal: Trading signal event
            
        Returns:
            Stop price or None if not calculable
        """
        # Check if stop price is in signal metadata
        if hasattr(signal, 'metadata') and signal.metadata:
            if 'stop_loss' in signal.metadata:
                return signal.metadata['stop_loss']
                
        # Calculate based on risk parameters
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        
        if self.risk_params.stop_loss_pct > 0:
            if direction > 0:  # Long
                return current_price * (1 - self.risk_params.stop_loss_pct / 100)
            elif direction < 0:  # Short
                return current_price * (1 + self.risk_params.stop_loss_pct / 100)
        
        return None
    
    def _calculate_target_price(self, current_price: float, signal: SignalEvent) -> Optional[float]:
        """
        Calculate target price based on risk parameters and signal.
        
        Args:
            current_price: Current market price
            signal: Trading signal event
            
        Returns:
            Target price or None if not calculable
        """
        # Check if target price is in signal metadata
        if hasattr(signal, 'metadata') and signal.metadata:
            if 'take_profit' in signal.metadata:
                return signal.metadata['take_profit']
                
        # Calculate based on risk parameters
        direction = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
        
        if self.risk_params.take_profit_pct:
            if direction > 0:  # Long
                return current_price * (1 + self.risk_params.take_profit_pct / 100)
            elif direction < 0:  # Short
                return current_price * (1 - self.risk_params.take_profit_pct / 100)
        
        # If we have stop loss and risk reward ratio, calculate target
        if self.risk_params.stop_loss_pct > 0 and self.risk_params.risk_reward_ratio:
            stop_distance = self.risk_params.stop_loss_pct / 100 * current_price
            target_distance = stop_distance * self.risk_params.risk_reward_ratio
            
            if direction > 0:  # Long
                return current_price + target_distance
            elif direction < 0:  # Short
                return current_price - target_distance
        
        return None
    
    def open_trade(self, trade_id: str, direction: str, entry_price: float, 
                 entry_time: datetime, initial_stop_price: Optional[float] = None,
                 initial_target_price: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Register a new trade with the risk manager.
        
        Args:
            trade_id: Unique identifier for the trade
            direction: 'long' or 'short'
            entry_price: Price at entry
            entry_time: Time of entry
            initial_stop_price: Optional custom stop-loss price (overrides calculated stop)
            initial_target_price: Optional custom take-profit price (overrides calculated target)
            **kwargs: Additional parameters for trade tracking
            
        Returns:
            Dictionary with calculated risk parameters for the trade
        """
        if direction not in ('long', 'short'):
            raise ValueError(f"Direction must be 'long' or 'short', got {direction}")
        
        # Calculate stop loss and take profit levels based on risk parameters
        if direction == 'long':
            stop_price = initial_stop_price if initial_stop_price is not None else \
                        entry_price * (1 - self.risk_params.stop_loss_pct / 100)
            
            target_price = initial_target_price if initial_target_price is not None else \
                          (entry_price * (1 + self.risk_params.take_profit_pct / 100) 
                           if self.risk_params.take_profit_pct else None)
                           
            # Calculate values for trailing stop if applicable
            if self.risk_params.trailing_stop_activation_pct:
                trail_activation_price = entry_price * (1 + self.risk_params.trailing_stop_activation_pct / 100)
                trail_distance = entry_price * (self.risk_params.trailing_stop_distance_pct / 100)
            else:
                trail_activation_price = None
                trail_distance = None
                
        else:  # short
            stop_price = initial_stop_price if initial_stop_price is not None else \
                        entry_price * (1 + self.risk_params.stop_loss_pct / 100)
            
            target_price = initial_target_price if initial_target_price is not None else \
                          (entry_price * (1 - self.risk_params.take_profit_pct / 100) 
                           if self.risk_params.take_profit_pct else None)
                           
            # Calculate values for trailing stop if applicable
            if self.risk_params.trailing_stop_activation_pct:
                trail_activation_price = entry_price * (1 - self.risk_params.trailing_stop_activation_pct / 100)
                trail_distance = entry_price * (self.risk_params.trailing_stop_distance_pct / 100)
            else:
                trail_activation_price = None
                trail_distance = None
        
        # Calculate time-based exit
        if self.risk_params.max_duration:
            if isinstance(self.risk_params.max_duration, timedelta):
                max_exit_time = entry_time + self.risk_params.max_duration
            else:
                # Assume max_duration is in seconds if not a timedelta
                max_exit_time = entry_time + timedelta(seconds=self.risk_params.max_duration)
        else:
            max_exit_time = None
            
        # Calculate position size if position_size_calculator is available
        position_size = None
        if 'symbol' in kwargs and 'portfolio' in kwargs:
            try:
                # Create a SignalEvent to calculate position size
                from src.signals import SignalType
                
                signal_type = SignalType.BUY if direction == 'long' else SignalType.SELL
                
                signal = SignalEvent(
                    signal_type=signal_type,
                    price=entry_price,
                    symbol=kwargs.get('symbol', 'default'),
                    timestamp=entry_time,
                    metadata={
                        'stop_loss': stop_price,
                        'take_profit': target_price
                    }
                )
                
                # Calculate position size
                portfolio = kwargs.get('portfolio')
                position_size = self.calculate_position_size(signal, portfolio, entry_price)
                
            except Exception as e:
                print(f"Error calculating position size: {str(e)}")
                position_size = None
                
        # Store trade details
        trade_details = {
            'id': trade_id,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'stop_price': stop_price,
            'target_price': target_price,
            'max_exit_time': max_exit_time,
            'trail_activation_price': trail_activation_price,
            'trail_distance': trail_distance,
            'trailing_stop_price': None,  # Will be updated once activated
            'is_trailing_active': False,
            'trailing_stop_high_watermark': None,  # For tracking the best price for trailing
            'position_size': position_size,
            'mae_pct': 0.0,  # Track MAE as percentage
            'mfe_pct': 0.0,  # Track MFE as percentage
            'highest_price': entry_price,  # For tracking MFE
            'lowest_price': entry_price,   # For tracking MAE
            'last_price': entry_price,
            'last_time': entry_time,
            'trade_bars': 0
        }
        
        # Store in active trades dictionary
        self.active_trades[trade_id] = trade_details
        
        # Start tracking with metrics collector if available
        if self.metrics_collector:
            self.metrics_collector.start_trade(
                trade_id=trade_id,
                entry_time=entry_time,
                entry_price=entry_price,
                direction=direction
            )
        
        # Return a copy of the trade details
        return trade_details.copy()
    
    # Remaining methods from original RiskManager class would follow here
    # update_price, _check_exit_conditions, close_trade, etc.



# """
# Risk manager module for implementing MAE, MFE, and ETD based risk management.

# This module provides a RiskManager class that can be used to apply data-driven
# risk management rules to any trading strategy, using parameters derived from
# historical trade analysis.
# """

# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# from src.risk_management.types import RiskParameters, ExitReason, TradeMetrics
# from src.risk_management.collector import RiskMetricsCollector


# class RiskManager:
#     """
#     Applies risk management rules based on MAE, MFE, and ETD analysis.
    
#     This class applies stop-loss, take-profit, trailing stop, and time-based
#     exit rules to trading positions, using parameters derived from historical
#     trade analysis.
#     """
    
#     def __init__(self, 
#                 risk_params: RiskParameters,
#                 metrics_collector: Optional[RiskMetricsCollector] = None,
#                 position_size_calculator: Optional[Callable] = None):
#         """
#         Initialize the risk manager.
        
#         Args:
#             risk_params: Risk parameters for stop-loss, take-profit, etc.
#             metrics_collector: Optional collector for tracking trade metrics
#             position_size_calculator: Optional function for calculating position sizes
#         """
#         self.risk_params = risk_params
#         self.metrics_collector = metrics_collector
#         self.position_size_calculator = position_size_calculator
        
#         # Active trade tracking
#         self.active_trades = {}  # trade_id -> trade details
        
#     def open_trade(self, trade_id: str, direction: str, entry_price: float, 
#                   entry_time: datetime, initial_stop_price: Optional[float] = None,
#                   initial_target_price: Optional[float] = None) -> Dict[str, Any]:
#         """
#         Register a new trade with the risk manager.
        
#         Args:
#             trade_id: Unique identifier for the trade
#             direction: 'long' or 'short'
#             entry_price: Trade entry price
#             entry_time: Trade entry time
#             initial_stop_price: Optional custom stop-loss price (overrides calculated stop)
#             initial_target_price: Optional custom take-profit price (overrides calculated target)
            
#         Returns:
#             Dictionary with calculated risk parameters for the trade
#         """
#         if direction not in ('long', 'short'):
#             raise ValueError(f"Direction must be 'long' or 'short', got {direction}")
        
#         # Calculate stop loss and take profit levels based on risk parameters
#         if direction == 'long':
#             stop_price = initial_stop_price if initial_stop_price is not None else \
#                         entry_price * (1 - self.risk_params.stop_loss_pct / 100)
            
#             target_price = initial_target_price if initial_target_price is not None else \
#                           (entry_price * (1 + self.risk_params.take_profit_pct / 100) 
#                            if self.risk_params.take_profit_pct else None)
                           
#             # Calculate values for trailing stop if applicable
#             if self.risk_params.trailing_stop_activation_pct:
#                 trail_activation_price = entry_price * (1 + self.risk_params.trailing_stop_activation_pct / 100)
#                 trail_distance = entry_price * (self.risk_params.trailing_stop_distance_pct / 100)
#             else:
#                 trail_activation_price = None
#                 trail_distance = None
                
#         else:  # short
#             stop_price = initial_stop_price if initial_stop_price is not None else \
#                         entry_price * (1 + self.risk_params.stop_loss_pct / 100)
            
#             target_price = initial_target_price if initial_target_price is not None else \
#                           (entry_price * (1 - self.risk_params.take_profit_pct / 100) 
#                            if self.risk_params.take_profit_pct else None)
                           
#             # Calculate values for trailing stop if applicable
#             if self.risk_params.trailing_stop_activation_pct:
#                 trail_activation_price = entry_price * (1 - self.risk_params.trailing_stop_activation_pct / 100)
#                 trail_distance = entry_price * (self.risk_params.trailing_stop_distance_pct / 100)
#             else:
#                 trail_activation_price = None
#                 trail_distance = None
        
#         # Calculate time-based exit
#         if self.risk_params.max_duration:
#             if isinstance(self.risk_params.max_duration, timedelta):
#                 max_exit_time = entry_time + self.risk_params.max_duration
#             else:
#                 # Assume max_duration is in seconds if not a timedelta
#                 max_exit_time = entry_time + timedelta(seconds=self.risk_params.max_duration)
#         else:
#             max_exit_time = None
            
#         # Calculate position size if calculator is provided
#         position_size = None
#         if self.position_size_calculator:
#             try:
#                 position_size = self.position_size_calculator(
#                     direction=direction,
#                     entry_price=entry_price,
#                     stop_price=stop_price,
#                     target_price=target_price,
#                     risk_reward_ratio=self.risk_params.risk_reward_ratio,
#                     expected_win_rate=self.risk_params.expected_win_rate
#                 )
#             except Exception as e:
#                 position_size = None
#                 print(f"Error calculating position size: {str(e)}")
                
#         # Store trade details
#         trade_details = {
#             'id': trade_id,
#             'direction': direction,
#             'entry_price': entry_price,
#             'entry_time': entry_time,
#             'stop_price': stop_price,
#             'target_price': target_price,
#             'max_exit_time': max_exit_time,
#             'trail_activation_price': trail_activation_price,
#             'trail_distance': trail_distance,
#             'trailing_stop_price': None,  # Will be updated once activated
#             'is_trailing_active': False,
#             'trailing_stop_high_watermark': None,  # For tracking the best price for trailing
#             'position_size': position_size,
#             'mae_pct': 0.0,  # Track MAE as percentage
#             'mfe_pct': 0.0,  # Track MFE as percentage
#             'highest_price': entry_price,  # For tracking MFE
#             'lowest_price': entry_price,   # For tracking MAE
#             'last_price': entry_price,
#             'last_time': entry_time,
#             'trade_bars': 0
#         }
        
#         # Store in active trades dictionary
#         self.active_trades[trade_id] = trade_details
        
#         # Start tracking with metrics collector if available
#         if self.metrics_collector:
#             self.metrics_collector.start_trade(
#                 trade_id=trade_id,
#                 entry_time=entry_time,
#                 entry_price=entry_price,
#                 direction=direction
#             )
        
#         # Return a copy of the trade details
#         return trade_details.copy()
    
#     def update_price(self, trade_id: str, current_price: float, current_time: datetime,
#                     bar_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Update trade with current price information and check exit conditions.
        
#         Args:
#             trade_id: Unique identifier for the trade
#             current_price: Current market price
#             current_time: Current time
#             bar_data: Optional complete bar data for metrics collection
            
#         Returns:
#             Dictionary with update status and any exit signals
#         """
#         if trade_id not in self.active_trades:
#             return {'status': 'error', 'message': f"Trade {trade_id} not found"}
        
#         trade = self.active_trades[trade_id]
#         direction = trade['direction']
        
#         # Update trade bar count
#         trade['trade_bars'] += 1
        
#         # Update high/low water marks for MAE/MFE tracking
#         if current_price > trade['highest_price']:
#             trade['highest_price'] = current_price
#         if current_price < trade['lowest_price']:
#             trade['lowest_price'] = current_price
        
#         # Calculate MFE and MAE as percentages
#         if direction == 'long':
#             mfe_pct = (trade['highest_price'] - trade['entry_price']) / trade['entry_price'] * 100
#             mae_pct = (trade['entry_price'] - trade['lowest_price']) / trade['entry_price'] * 100
#         else:  # short
#             mfe_pct = (trade['entry_price'] - trade['lowest_price']) / trade['entry_price'] * 100
#             mae_pct = (trade['highest_price'] - trade['entry_price']) / trade['entry_price'] * 100
        
#         trade['mfe_pct'] = mfe_pct
#         trade['mae_pct'] = mae_pct
        
#         # Update last price information
#         trade['last_price'] = current_price
#         trade['last_time'] = current_time
        
#         # Update metrics collector if available
#         if self.metrics_collector and bar_data:
#             self.metrics_collector.update_price_path(trade_id, bar_data)
        
#         # Check exit conditions
#         exit_info = self._check_exit_conditions(trade, current_price, current_time)
        
#         # Return updated status
#         return {
#             'status': 'active' if not exit_info['exit_signal'] else 'exit',
#             'trade': trade,
#             'exit_info': exit_info,
#             'mfe_pct': mfe_pct,
#             'mae_pct': mae_pct
#         }
    
#     def _check_exit_conditions(self, trade: Dict[str, Any], current_price: float, 
#                               current_time: datetime) -> Dict[str, Any]:
#         """
#         Check if any exit conditions have been met.
        
#         Args:
#             trade: Trade details dictionary
#             current_price: Current market price
#             current_time: Current time
            
#         Returns:
#             Dictionary with exit signal information
#         """
#         direction = trade['direction']
#         result = {
#             'exit_signal': False,
#             'exit_reason': None,
#             'exit_price': None
#         }
        
#         # Check stop loss
#         if direction == 'long' and current_price <= trade['stop_price']:
#             result['exit_signal'] = True
#             result['exit_reason'] = ExitReason.STOP_LOSS
#             result['exit_price'] = trade['stop_price']
#             return result
#         elif direction == 'short' and current_price >= trade['stop_price']:
#             result['exit_signal'] = True
#             result['exit_reason'] = ExitReason.STOP_LOSS
#             result['exit_price'] = trade['stop_price']
#             return result
        
#         # Check take profit
#         if trade['target_price'] is not None:
#             if direction == 'long' and current_price >= trade['target_price']:
#                 result['exit_signal'] = True
#                 result['exit_reason'] = ExitReason.TAKE_PROFIT
#                 result['exit_price'] = trade['target_price']
#                 return result
#             elif direction == 'short' and current_price <= trade['target_price']:
#                 result['exit_signal'] = True
#                 result['exit_reason'] = ExitReason.TAKE_PROFIT
#                 result['exit_price'] = trade['target_price']
#                 return result
        
#         # Check trailing stop logic
#         if trade['trail_activation_price'] is not None:
#             # Check if trailing stop should be activated
#             if not trade['is_trailing_active']:
#                 if (direction == 'long' and current_price >= trade['trail_activation_price']) or \
#                    (direction == 'short' and current_price <= trade['trail_activation_price']):
#                     # Activate trailing stop
#                     trade['is_trailing_active'] = True
#                     trade['trailing_stop_high_watermark'] = current_price
                    
#                     if direction == 'long':
#                         trade['trailing_stop_price'] = current_price - trade['trail_distance']
#                     else:  # short
#                         trade['trailing_stop_price'] = current_price + trade['trail_distance']
            
#             # Update trailing stop if active
#             if trade['is_trailing_active']:
#                 # Update the high watermark and trailing stop level if price improved
#                 if direction == 'long' and current_price > trade['trailing_stop_high_watermark']:
#                     trade['trailing_stop_high_watermark'] = current_price
#                     trade['trailing_stop_price'] = current_price - trade['trail_distance']
#                 elif direction == 'short' and current_price < trade['trailing_stop_high_watermark']:
#                     trade['trailing_stop_high_watermark'] = current_price
#                     trade['trailing_stop_price'] = current_price + trade['trail_distance']
                
#                 # Check if the trailing stop has been triggered
#                 if direction == 'long' and current_price <= trade['trailing_stop_price']:
#                     result['exit_signal'] = True
#                     result['exit_reason'] = ExitReason.TRAILING_STOP
#                     result['exit_price'] = trade['trailing_stop_price']
#                     return result
#                 elif direction == 'short' and current_price >= trade['trailing_stop_price']:
#                     result['exit_signal'] = True
#                     result['exit_reason'] = ExitReason.TRAILING_STOP
#                     result['exit_price'] = trade['trailing_stop_price']
#                     return result
        
#         # Check time-based exit
#         if trade['max_exit_time'] is not None and current_time >= trade['max_exit_time']:
#             result['exit_signal'] = True
#             result['exit_reason'] = ExitReason.TIME_EXIT
#             result['exit_price'] = current_price  # Use current price for time exits
#             return result
        
#         # No exit conditions met
#         return result
    
#     def close_trade(self, trade_id: str, exit_price: float, exit_time: datetime, 
#                    exit_reason: ExitReason = ExitReason.STRATEGY_EXIT) -> Dict[str, Any]:
#         """
#         Close a trade and record final metrics.
        
#         Args:
#             trade_id: Unique identifier for the trade
#             exit_price: Price at exit
#             exit_time: Time of exit
#             exit_reason: Reason for the exit
            
#         Returns:
#             Dictionary with trade summary information
#         """
#         if trade_id not in self.active_trades:
#             return {'status': 'error', 'message': f"Trade {trade_id} not found"}
        
#         trade = self.active_trades[trade_id]
        
#         # Calculate final trade metrics
#         direction = trade['direction']
#         entry_price = trade['entry_price']
        
#         if direction == 'long':
#             return_pct = (exit_price - entry_price) / entry_price * 100
#             is_winner = exit_price > entry_price
#         else:  # short
#             return_pct = (entry_price - exit_price) / entry_price * 100
#             is_winner = exit_price < entry_price
        
#         # Record metrics with collector if available
#         if self.metrics_collector:
#             self.metrics_collector.end_trade(
#                 trade_id=trade_id,
#                 exit_time=exit_time,
#                 exit_price=exit_price,
#                 exit_reason=exit_reason
#             )
        
#         # Create trade summary
#         trade_summary = {
#             'id': trade_id,
#             'direction': direction,
#             'entry_price': entry_price,
#             'entry_time': trade['entry_time'],
#             'exit_price': exit_price,
#             'exit_time': exit_time,
#             'exit_reason': exit_reason,
#             'return_pct': return_pct,
#             'is_winner': is_winner,
#             'mae_pct': trade['mae_pct'],
#             'mfe_pct': trade['mfe_pct'],
#             'duration': exit_time - trade['entry_time'],
#             'trade_bars': trade['trade_bars'],
#             'initial_stop_price': trade['stop_price'],
#             'initial_target_price': trade['target_price'],
#             'trailing_stop_activated': trade['is_trailing_active'],
#             'trailing_stop_price': trade['trailing_stop_price']
#         }
        
#         # Remove from active trades
#         del self.active_trades[trade_id]
        
#         return {
#             'status': 'closed',
#             'trade_summary': trade_summary
#         }
    
#     def update_risk_parameters(self, new_params: RiskParameters) -> None:
#         """
#         Update risk parameters for future trades.
        
#         Args:
#             new_params: New risk parameters
#         """
#         self.risk_params = new_params
    
#     def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
#         """
#         Get all currently active trades.
        
#         Returns:
#             Dictionary of active trades
#         """
#         return {trade_id: trade.copy() for trade_id, trade in self.active_trades.items()}
    
#     def get_trade_details(self, trade_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get details for a specific trade.
        
#         Args:
#             trade_id: Unique identifier for the trade
            
#         Returns:
#             Trade details dictionary or None if not found
#         """
#         return self.active_trades.get(trade_id, {}).copy()
    
#     def modify_stop_loss(self, trade_id: str, new_stop_price: float) -> Dict[str, Any]:
#         """
#         Modify the stop loss for an active trade.
        
#         Args:
#             trade_id: Unique identifier for the trade
#             new_stop_price: New stop loss price
            
#         Returns:
#             Dictionary with update status
#         """
#         if trade_id not in self.active_trades:
#             return {'status': 'error', 'message': f"Trade {trade_id} not found"}
        
#         trade = self.active_trades[trade_id]
#         direction = trade['direction']
        
#         # Validate stop price direction
#         if direction == 'long' and new_stop_price >= trade['entry_price']:
#             return {'status': 'error', 'message': "Stop loss must be below entry price for long positions"}
#         elif direction == 'short' and new_stop_price <= trade['entry_price']:
#             return {'status': 'error', 'message': "Stop loss must be above entry price for short positions"}
        
#         # Update stop price
#         trade['stop_price'] = new_stop_price
        
#         return {
#             'status': 'updated',
#             'trade_id': trade_id,
#             'new_stop_price': new_stop_price
#         }
    
#     def modify_take_profit(self, trade_id: str, new_target_price: float) -> Dict[str, Any]:
#         """
#         Modify the take profit for an active trade.
        
#         Args:
#             trade_id: Unique identifier for the trade
#             new_target_price: New take profit price
            
#         Returns:
#             Dictionary with update status
#         """
#         if trade_id not in self.active_trades:
#             return {'status': 'error', 'message': f"Trade {trade_id} not found"}
        
#         trade = self.active_trades[trade_id]
#         direction = trade['direction']
        
#         # Validate target price direction
#         if direction == 'long' and new_target_price <= trade['entry_price']:
#             return {'status': 'error', 'message': "Take profit must be above entry price for long positions"}
#         elif direction == 'short' and new_target_price >= trade['entry_price']:
#             return {'status': 'error', 'message': "Take profit must be below entry price for short positions"}
        
#         # Update target price
#         trade['target_price'] = new_target_price
        
#         return {
#             'status': 'updated',
#             'trade_id': trade_id,
#             'new_target_price': new_target_price
#         }
    
#     def calculate_expectancy(self) -> float:
#         """
#         Calculate mathematical expectancy based on risk parameters.
        
#         Returns:
#             Expectancy value
#         """
#         # Simplified expectancy calculation
#         if not (self.risk_params.risk_reward_ratio and self.risk_params.expected_win_rate):
#             return 0.0
        
#         win_rate = self.risk_params.expected_win_rate
#         rr_ratio = self.risk_params.risk_reward_ratio
        
#         # E = (Win% * Average Win) - (Loss% * Average Loss)
#         # Simplified to: E = (Win% * RR) - (1 - Win%)
#         return (win_rate * rr_ratio) - (1 - win_rate)
    
#     def get_position_size_suggestion(self, account_size: float, risk_per_trade_pct: float = 1.0, 
#                                     max_position_pct: float = 5.0) -> Dict[str, Any]:
#         """
#         Get position sizing suggestions based on risk parameters.
        
#         Args:
#             account_size: Current account size
#             risk_per_trade_pct: Percentage of account to risk per trade
#             max_position_pct: Maximum percentage of account for any position
            
#         Returns:
#             Dictionary with position sizing information
#         """
#         # Calculate Kelly position size
#         win_rate = self.risk_params.expected_win_rate or 0.5
#         rr_ratio = self.risk_params.risk_reward_ratio or 1.0
        
#         # Kelly formula: K% = W - [(1 - W) / R]
#         # Where W = Win Rate, R = Risk/Reward ratio
#         kelly_pct = win_rate - ((1 - win_rate) / rr_ratio)
        
#         # Cap Kelly between 0% and max_position_pct
#         kelly_pct = max(0, min(kelly_pct, max_position_pct / 100))
        
#         # Calculate position size based on fixed risk percentage
#         risk_amount = account_size * (risk_per_trade_pct / 100)
#         position_size_fixed_risk = risk_amount / (self.risk_params.stop_loss_pct / 100)
        
#         # Calculate position size based on Kelly criterion
#         position_size_kelly = account_size * kelly_pct
        
#         # Calculate half Kelly (safer approach)
#         position_size_half_kelly = position_size_kelly / 2
        
#         # Cap position sizes based on max percentage
#         max_position_size = account_size * (max_position_pct / 100)
#         position_size_fixed_risk = min(position_size_fixed_risk, max_position_size)
#         position_size_kelly = min(position_size_kelly, max_position_size)
#         position_size_half_kelly = min(position_size_half_kelly, max_position_size)
        
#         return {
#             'account_size': account_size,
#             'fixed_risk_pct': risk_per_trade_pct,
#             'max_position_pct': max_position_pct,
#             'kelly_pct': kelly_pct * 100,  # Convert to percentage
#             'position_size_fixed_risk': position_size_fixed_risk,
#             'position_size_kelly': position_size_kelly,
#             'position_size_half_kelly': position_size_half_kelly,
#             'recommended_size': position_size_half_kelly,  # Half Kelly is typically recommended
#             'risk_reward_ratio': rr_ratio,
#             'expected_win_rate': win_rate,
#             'expected_profit_pct': self.calculate_expectancy() * risk_per_trade_pct
#         }
