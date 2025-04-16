"""
Position Manager Module

This module provides the PositionManager class, which coordinates position sizing,
allocation, and risk management for the trading system.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict

from src.position_management.position import Position, PositionStatus, EntryType, ExitType
from src.position_management.portfolio import Portfolio
from src.position_management.position_sizers import PositionSizer, PositionSizerFactory
from src.position_management.allocation import AllocationStrategy, AllocationStrategyFactory

# Set up logging
logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages trading positions, sizing, and allocation.
    
    The position manager integrates position sizing, allocation, and risk
    management to determine appropriate position sizes based on signals,
    market conditions, and portfolio state.
    """
    
    def __init__(self, portfolio: Portfolio, position_sizer: Optional[PositionSizer] = None,
                allocation_strategy: Optional[AllocationStrategy] = None,
                risk_manager: Optional[Any] = None, max_positions: int = 0):
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
        
    def set_position_sizer(self, position_sizer: Union[PositionSizer, str, Dict[str, Any]]) -> None:
        """
        Set the position sizing strategy.
        
        Args:
            position_sizer: Position sizer object, type name, or configuration
        """
        if isinstance(position_sizer, PositionSizer):
            self.position_sizer = position_sizer
        elif isinstance(position_sizer, str):
            self.position_sizer = PositionSizerFactory.create_sizer(position_sizer)
        elif isinstance(position_sizer, dict):
            self.position_sizer = PositionSizerFactory.create_from_config(position_sizer)
        else:
            raise ValueError("position_sizer must be a PositionSizer object, type name, or configuration dict")
            
    def set_allocation_strategy(self, allocation_strategy: Union[AllocationStrategy, str, Dict[str, Any]]) -> None:
        """
        Set the allocation strategy.
        
        Args:
            allocation_strategy: Allocation strategy object, type name, or configuration
        """
        if isinstance(allocation_strategy, AllocationStrategy):
            self.allocation_strategy = allocation_strategy
        elif isinstance(allocation_strategy, str):
            self.allocation_strategy = AllocationStrategyFactory.create_strategy(allocation_strategy)
        elif isinstance(allocation_strategy, dict):
            self.allocation_strategy = AllocationStrategyFactory.create_from_config(allocation_strategy)
        else:
            raise ValueError("allocation_strategy must be an AllocationStrategy object, type name, or configuration dict")
            
    def set_risk_manager(self, risk_manager: Any) -> None:
        """
        Set the risk manager.
        
        Args:
            risk_manager: Risk management component
        """
        self.risk_manager = risk_manager


    def on_signal(self, event):
        """Process a signal event and generate position actions.

        Args:
            event: Event object containing a Signal in its data attribute

        Returns:
            List[Dict[str, Any]]: List of position actions (entry/exit/modify)
        """
        # Extract the Signal from the Event
        if hasattr(event, 'data'):
            signal = event.data

            # Record the signal in our history for tracking/analysis
            if hasattr(self, 'signal_history'):
                self.signal_history.append(signal)

            # Convert to format expected by _process_signals method
            signals_dict = {}
            prices_dict = {}

            # Extract data from the Signal object
            symbol = signal.symbol if hasattr(signal, 'symbol') else 'default'
            price = signal.price if hasattr(signal, 'price') else 0

            # Add to our dictionaries for processing
            signals_dict[symbol] = signal
            prices_dict[symbol] = price

            # Generate position actions from the signal
            actions = self._process_signals(signals_dict, prices_dict)

            # Return the actions
            return actions
        else:
            # For backward compatibility, check if a Signal was passed directly
            if hasattr(event, 'signal_type'):
                signal = event
                if hasattr(self, 'signal_history'):
                    self.signal_history.append(signal)

                symbol = signal.symbol if hasattr(signal, 'symbol') else 'default'
                price = signal.price if hasattr(signal, 'price') else 0

                signals_dict = {symbol: signal}
                prices_dict = {symbol: price}

                return self._process_signals(signals_dict, prices_dict)
            else:
                raise TypeError(f"Expected Event object with data attribute or Signal object")


    def on_fill(self, fill_event):
        """
        Handle fill events and update the portfolio accordingly.

        Args:
            fill_event: Fill event data containing order execution details

        Returns:
            Dictionary with updated position information or None if fill cannot be processed
        """

        print("\n==== POSITION MANAGER FILL HANDLER CALLED ====")
        print(f"Fill event type: {fill_event.event_type}")
        if not hasattr(fill_event, 'data'):
            logger.warning("Fill event has no data attribute")
            return None

        fill_data = fill_event.data
        logger.info(f"Processing fill: {fill_data}")

        # Extract fill data
        symbol = fill_data.get('symbol')
        direction = fill_data.get('direction')
        quantity = fill_data.get('quantity')
        fill_price = fill_data.get('fill_price')
        order_id = fill_data.get('order_id')
        timestamp = fill_data.get('timestamp', datetime.datetime.now())
        commission = fill_data.get('commission', 0.0)

        # Ensure we have the required data
        if not all([symbol, direction, quantity, fill_price]):
            logger.warning(f"Fill missing required fields: {fill_data}")
            return None

        # Check if this is an entry or exit
        # For now, we'll treat it as a new position (entry)
        try:
            # Create a new position in the portfolio
            position = self.portfolio.open_position(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=fill_price,
                entry_time=timestamp,
                entry_order_id=order_id,
                transaction_cost=commission
            )

            logger.info(f"New position created from fill: {position.position_id}")
            print("==== POSITION MANAGER FILL HANDLER COMPLETED ====")
            return {
                "action": "entry",
                "position_id": position.position_id,
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity,
                "price": fill_price,
                "timestamp": timestamp,
                "commission": commission
            }
        except Exception as e:
            logger.error(f"Error processing fill: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _process_signals(self, signals: Dict[str, Dict[str, Any]], 
                       current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Process multiple signals and determine position actions.
        
        Args:
            signals: Dictionary of signals by symbol
            current_prices: Dictionary of current prices by symbol
            
        Returns:
            List of position action dictionaries
        """
        actions = []
        
        if not signals:
            return actions
            
        # Calculate portfolio allocation if we have an allocation strategy
        if self.allocation_strategy:
            allocations = self.allocation_strategy.allocate(
                self.portfolio, signals, current_prices
            )
        else:
            # Default to equal allocation
            weight = 1.0 / len(signals)
            allocations = {symbol: weight for symbol in signals.keys()}
        
        # Calculate equity to allocate (considering current positions)
        available_equity = self.portfolio.equity
        num_open_positions = len(self.portfolio.positions)
        max_new_positions = self.max_positions - num_open_positions if self.max_positions > 0 else len(signals)
        
        # Process each signal
        for symbol, signal in signals.items():
            price = current_prices.get(symbol, 0)
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                continue
                
            # Get signal direction
            direction = self._get_signal_direction(signal)
            
            # Check if we already have a position in this symbol
            net_position = self.portfolio.get_net_position(symbol)
            
            if net_position != 0:
                # We have an existing position
                if (direction > 0 and net_position < 0) or (direction < 0 and net_position > 0):
                    # Signal direction opposite of current position - exit current position
                    for position in self.portfolio.positions_by_symbol.get(symbol, []):
                        exit_action = {
                            'action': 'exit',
                            'position_id': position.position_id,
                            'symbol': symbol,
                            'price': price,
                            'reason': 'Signal direction change',
                            'exit_type': ExitType.STRATEGY
                        }
                        actions.append(exit_action)
                        
                elif direction == 0:
                    # Neutral signal - close existing position
                    for position in self.portfolio.positions_by_symbol.get(symbol, []):
                        exit_action = {
                            'action': 'exit',
                            'position_id': position.position_id,
                            'symbol': symbol,
                            'price': price,
                            'reason': 'Neutral signal',
                            'exit_type': ExitType.STRATEGY
                        }
                        actions.append(exit_action)
                else:
                    # Same direction - evaluate modifying position size
                    # (Not implemented here - could add position scaling)
                    pass
            else:
                # No existing position - evaluate new entry
                if direction != 0 and max_new_positions > 0:
                    # Calculate allocation for this symbol
                    allocation = allocations.get(symbol, 0)
                    allocated_equity = available_equity * allocation
                    
                    # Calculate position size
                    if self.position_sizer:
                        size = self.position_sizer.calculate_position_size(
                            signal, self.portfolio, price
                        )
                    else:
                        # Default size calculation based on allocation
                        size = allocated_equity / price * (1 if direction > 0 else -1)
                    
                    # Evaluate position with risk manager if available
                    if self.risk_manager and hasattr(self.risk_manager, 'evaluate_position'):
                        approved, adjusted_size, risk_metrics = self.risk_manager.evaluate_position(
                            symbol=symbol, 
                            direction=direction, 
                            size=size, 
                            price=price,
                            signal=signal,
                            portfolio=self.portfolio
                        )
                        
                        if approved and adjusted_size != 0:
                            size = adjusted_size
                        else:
                            # Position rejected by risk manager
                            self.rejected_entries[symbol] = {
                                'time': datetime.datetime.now(),
                                'reason': risk_metrics.get('reason', 'Rejected by risk manager'),
                                'signal': signal
                            }
                            continue
                    
                    # Check if size is significant
                    if abs(size) < 0.0001:
                        logger.debug(f"Position size too small for {symbol}: {size}")
                        continue
                        
                    # Create entry action
                    entry_action = {
                        'action': 'entry',
                        'symbol': symbol,
                        'direction': direction,
                        'size': size,
                        'price': price,
                        'signal': signal
                    }
                    
                    # Add risk parameters if available
                    if self.risk_manager:
                        # Calculate stop loss price
                        if hasattr(self.risk_manager, 'calculate_stop_loss'):
                            stop_loss = self.risk_manager.calculate_stop_loss(
                                symbol=symbol,
                                direction=direction,
                                entry_price=price,
                                atr=signal.get('atr', None),
                                volatility=signal.get('volatility', None)
                            )
                            
                            if stop_loss > 0:
                                entry_action['stop_loss'] = stop_loss
                        
                        # Calculate take profit price
                        if hasattr(self.risk_manager, 'calculate_take_profit'):
                            take_profit = self.risk_manager.calculate_take_profit(
                                symbol=symbol,
                                direction=direction,
                                entry_price=price,
                                stop_loss=entry_action.get('stop_loss', None)
                            )
                            
                            if take_profit > 0:
                                entry_action['take_profit'] = take_profit
                    
                    actions.append(entry_action)
                    max_new_positions -= 1
        
        return actions
    
    def _get_signal_direction(self, signal: Dict[str, Any]) -> int:
        """
        Extract direction from a signal.
        
        Args:
            signal: Signal data
            
        Returns:
            Direction as an integer (1 for buy/long, -1 for sell/short, 0 for neutral)
        """
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
        
        if 'type' in signal:
            signal_type = signal['type']
            if isinstance(signal_type, str):
                if signal_type.upper() in ['BUY', 'LONG']:
                    return 1
                elif signal_type.upper() in ['SELL', 'SHORT']:
                    return -1
        
        # Default to neutral if direction cannot be determined
        return 0
    
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
                
        elif action_type == 'modify':
            # Modify existing position
            position_id = action['position_id']
            
            try:
                # Get the position
                if position_id not in self.portfolio.positions:
                    raise ValueError(f"Position not found: {position_id}")
                    
                position = self.portfolio.positions[position_id]
                
                # Apply modifications
                modifications = {}
                
                if 'stop_loss' in action:
                    old_stop = position.stop_loss
                    new_stop = action['stop_loss']
                    position.modify_stop_loss(new_stop)
                    modifications['stop_loss'] = {'old': old_stop, 'new': new_stop}
                
                if 'take_profit' in action:
                    old_target = position.take_profit
                    new_target = action['take_profit']
                    position.modify_take_profit(new_target)
                    modifications['take_profit'] = {'old': old_target, 'new': new_target}
                
                if 'trailing_stop' in action:
                    distance = action['trailing_stop']
                    trailing_info = position.set_trailing_stop(distance)
                    modifications['trailing_stop'] = trailing_info
                
                result = {
                    'action': 'modify',
                    'success': True,
                    'position_id': position_id,
                    'symbol': position.symbol,
                    'modifications': modifications,
                    'time': current_time
                }
                
                logger.info(f"Modified position: {position_id} {position.symbol}")
                
                return result
            except Exception as e:
                logger.error(f"Failed to execute modification: {str(e)}")
                return {
                    'action': 'modify',
                    'success': False,
                    'position_id': position_id,
                    'error': str(e)
                }
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return None
    
    def update_positions(self, current_prices: Dict[str, float], 
                       current_time: datetime.datetime) -> List[Dict[str, Any]]:
        """
        Update positions with current prices and check for exits.
        
        Args:
            current_prices: Dictionary of current prices by symbol
            current_time: Current timestamp
            
        Returns:
            List of exit actions for triggered exits
        """
        # Update portfolio with current prices
        exits = self.portfolio.update_prices(current_prices, current_time)
        
        # Convert exits to actions
        exit_actions = []
        
        for exit_info in exits:
            exit_action = {
                'action': 'exit',
                'position_id': exit_info['position_id'],
                'symbol': exit_info['symbol'],
                'price': exit_info['exit_price'],
                'reason': exit_info['exit_reason'],
                'exit_type': exit_info['exit_type'],
                'time': current_time
            }
            
            exit_actions.append(exit_action)
        
        return exit_actions
    
    def get_portfolio_risk_exposure(self) -> Dict[str, Any]:
        """
        Get current portfolio risk exposure metrics.
        
        Returns:
            Dictionary of risk exposure metrics
        """
        metrics = {
            'capital_at_risk': 0.0,
            'capital_at_risk_pct': 0.0,
            'exposure_by_symbol': {},
            'exposure_by_direction': {
                'long': 0.0,
                'short': 0.0
            },
            'positions_count': len(self.portfolio.positions),
            'diversification': 0.0
        }
        
        # Exit if no positions
        if not self.portfolio.positions:
            return metrics
        
        # Get symbol exposure
        symbol_exposure = self.portfolio.get_position_exposure()
        metrics['exposure_by_symbol'] = symbol_exposure
        
        # Calculate direction exposure
        for position in self.portfolio.positions.values():
            value = position.quantity * position.current_price
            if position.direction > 0:
                metrics['exposure_by_direction']['long'] += value
            else:
                metrics['exposure_by_direction']['short'] += value
        
        # Calculate capital at risk
        total_exposure = sum(abs(v) for v in symbol_exposure.values())
        metrics['capital_at_risk'] = total_exposure
        
        if self.portfolio.equity > 0:
            metrics['capital_at_risk_pct'] = total_exposure / self.portfolio.equity
        
        # Calculate diversification (1 - Herfindahl-Hirschman Index)
        if symbol_exposure:
            total_exposure = sum(abs(v) for v in symbol_exposure.values())
            if total_exposure > 0:
                weights = [abs(v) / total_exposure for v in symbol_exposure.values()]
                hhi = sum(w**2 for w in weights)
                metrics['diversification'] = 1 - hhi
        
        return metrics
    
    def get_position_info(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific position.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position information dictionary or None if not found
        """
        if position_id in self.portfolio.positions:
            return self.portfolio.positions[position_id].get_summary()
        elif position_id in self.portfolio.closed_positions:
            return self.portfolio.closed_positions[position_id].get_summary()
        else:
            return None
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get information about all open positions.
        
        Returns:
            List of position summary dictionaries
        """
        return self.portfolio.get_open_positions()
    
    def close_all_positions(self, current_prices: Dict[str, float], 
                          current_time: datetime.datetime,
                          reason: str = "System shutdown") -> List[Dict[str, Any]]:
        """
        Close all open positions.
        
        Args:
            current_prices: Dictionary of current prices by symbol
            current_time: Current timestamp
            reason: Reason for closing all positions
            
        Returns:
            List of position close summaries
        """
        return self.portfolio.close_all_positions(current_prices, current_time, ExitType.STRATEGY)


# Example usage
if __name__ == "__main__":
    from position_sizers import PositionSizerFactory
    from allocation import AllocationStrategyFactory
    
    # Create portfolio
    portfolio = Portfolio(initial_capital=100000)
    
    # Create position sizer
    position_sizer = PositionSizerFactory.create_sizer('volatility', risk_pct=0.01)
    
    # Create allocation strategy
    allocation_strategy = AllocationStrategyFactory.create_strategy('equal_weight')
    
    # Create position manager
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        allocation_strategy=allocation_strategy,
        max_positions=10
    )
    
    # Generate sample signals
    signals = {
        'AAPL': {
            'symbol': 'AAPL',
            'signal_type': 'BUY',
            'price': 150.0,
            'confidence': 0.8,
            'volatility': 0.012,
            'atr': 1.8
        },
        'MSFT': {
            'symbol': 'MSFT',
            'signal_type': 'BUY',
            'price': 300.0,
            'confidence': 0.6,
            'volatility': 0.010,
            'atr': 3.0
        }
    }
    
    current_prices = {
        'AAPL': 150.0,
        'MSFT': 300.0
    }
    
    # Process signals and get actions
    actions = position_manager._process_signals(signals, current_prices)
    
    # Execute actions
    current_time = datetime.datetime.now()
    results = []
    
    for action in actions:
        result = position_manager.execute_position_action(action, current_time)
        results.append(result)
        
    # Print position summaries
    for position in position_manager.get_open_positions():
        print(f"{position['symbol']} {position['direction']} {position['quantity']} @ {position['entry_price']}")
        
    # Get portfolio risk metrics
    risk_metrics = position_manager.get_portfolio_risk_exposure()
    print(f"Capital at risk: {risk_metrics['capital_at_risk_pct']:.2%}")
    print(f"Diversification: {risk_metrics['diversification']:.2f}")
