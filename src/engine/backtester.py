"""
Backtester for Trading System

This module provides the main Backtester class which orchestrates the backtesting process,
connecting strategy, execution, and data components into a cohesive simulation environment.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Callable, Union

from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType
from src.signals import Signal, SignalType
from src.engine import ExecutionEngine, MarketSimulator
from src.position_management.position_manager import PositionManager, PositionSizer

# Set up logging
logger = logging.getLogger(__name__)

class Order:
    """Simple Order class for backtesting."""
    
    def __init__(self, symbol, order_type, quantity, direction, timestamp=None, price=None):
        """
        Initialize an order.
        
        Args:
            symbol: Instrument symbol
            order_type: Type of order ('MARKET', 'LIMIT', etc.)
            quantity: Order quantity
            direction: Direction (1 for buy, -1 for sell)
            timestamp: Optional timestamp
            price: Optional price (for limit orders)
        """
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.timestamp = timestamp
        self.price = price
        self.order_id = id(self)  # Simple unique ID
        
    def __str__(self):
        return f"Order({self.symbol}, {self.order_type}, {self.direction * self.quantity}, {self.timestamp})"


class Backtester:
    """
    Main orchestration class that coordinates the backtest execution.
    Acts as the facade for the backtesting subsystem.
    """
    
    def __init__(self, config, data_handler, strategy, position_manager=None):
        """
        Initialize the backtester with configuration and dependencies.
        
        Args:
            config: Configuration dictionary or ConfigManager instance
            data_handler: Data handler providing market data
            strategy: Trading strategy to test
            position_manager: Optional position manager for risk management
        """
        self.config = config
        self.data_handler = data_handler
        self.strategy = strategy
        
        # Initialize execution components
        self.position_manager = position_manager or DefaultPositionManager()
        self.execution_engine = ExecutionEngine(self.position_manager)
        
        # Initialize event system
        self.event_bus = EventBus()
        
        # Initialize market simulator
        market_sim_config = self._extract_market_sim_config(config)
        self.market_simulator = MarketSimulator(market_sim_config)
        
        # Configure initial capital
        self.initial_capital = self._extract_initial_capital(config)
        if hasattr(self.execution_engine, 'portfolio'):
            self.execution_engine.portfolio.cash = self.initial_capital
            self.execution_engine.portfolio.initial_capital = self.initial_capital
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Track signals for debugging
        self.signals = []
        self.orders = []
        self.fills = []
        
        logger.info(f"Backtester initialized with initial capital: {self.initial_capital}")
    
    def _extract_market_sim_config(self, config):
        """Extract market simulation configuration."""
        if isinstance(config, dict):
            if 'backtester' in config and 'market_simulation' in config['backtester']:
                return config['backtester']['market_simulation']
            elif 'market_simulation' in config:
                return config['market_simulation']
        elif hasattr(config, 'get'):
            return config.get('backtester.market_simulation', {})
        return {}
    
    def _extract_initial_capital(self, config):
        """Extract initial capital from configuration."""
        if isinstance(config, dict):
            if 'backtester' in config and 'initial_capital' in config['backtester']:
                return config['backtester']['initial_capital']
            elif 'initial_capital' in config:
                return config['initial_capital']
        elif hasattr(config, 'get'):
            return config.get('backtester.initial_capital', 100000)
        return 100000
    
    def _setup_event_handlers(self):
        """Set up event handlers for backtesting."""
        # Register signal handler
        self.event_bus.register(EventType.SIGNAL, self._on_signal)
        
        # Register order handler
        self.event_bus.register(EventType.ORDER, self._on_order)
        
        # Register fill handler
        self.event_bus.register(EventType.FILL, self._on_fill)
    
    def run(self, use_test_data=False):
        """
        Run the backtest.
        
        Args:
            use_test_data: Whether to use test data (True) or training data (False)
            
        Returns:
            dict: Backtest results
        """
        # Reset all components before running
        self.reset()
        logger.info("Starting backtest...")
        
        # Select data iterator based on data set
        iterator = self.data_handler.iter_test() if use_test_data else self.data_handler.iter_train()
        
        # Process each bar
        for bar_event in iterator:
            # Get the bar data
            bar = bar_event.bar if hasattr(bar_event, 'bar') else bar_event
            
            # Log periodically for long backtests
            if hasattr(bar, 'get') and bar.get('timestamp') and hasattr(bar.get('timestamp'), 'day'):
                timestamp = bar.get('timestamp')
                if timestamp.day == 1 and timestamp.hour == 0 and timestamp.minute == 0:
                    logger.info(f"Processing data for {timestamp}")
            
            # Process bar through strategy to get signals
            signal_or_signals = self.strategy.on_bar(bar_event)
            
            # Handle signals from strategy
            self._process_signals(signal_or_signals, bar)
            
            # Update portfolio with current bar data
            self.execution_engine.update(bar)
            
            # Execute any pending orders
            self.execution_engine.execute_pending_orders(bar, self.market_simulator)
        
        logger.info("Backtest completed. Collecting results...")
        
        # Collect and return results
        return self.collect_results()
    
    def _process_signals(self, signal_or_signals, bar):
        """
        Process signals from strategy.
        
        Args:
            signal_or_signals: Single signal or list of signals from strategy
            bar: Current bar data
        """
        if signal_or_signals is None:
            return
        
        # Ensure signals is a list
        signals = signal_or_signals if isinstance(signal_or_signals, list) else [signal_or_signals]
        
        # Process each signal
        for signal in signals:
            if signal is None:
                continue
                
            # Store signal for debugging
            self.signals.append(signal)
            
            # Create signal event
            signal_event = Event(EventType.SIGNAL, signal)
            
            # Emit signal event to event bus
            self.event_bus.emit(signal_event)
    
    def _on_signal(self, event):
        """
        Handle signal events by generating orders.
        
        Args:
            event: Signal event
        """
        signal = event.data
        
        # Skip neutral signals
        if hasattr(signal, 'signal_type') and signal.signal_type == SignalType.NEUTRAL:
            return
        
        # Calculate position size
        position_size = self.position_manager.calculate_position_size(
            signal, 
            self.execution_engine.portfolio if hasattr(self.execution_engine, 'portfolio') else {'equity': self.initial_capital}
        )
        
        # Skip if position size is zero
        if position_size == 0:
            return
        
        # Determine direction
        direction = 1  # Default to buy
        if hasattr(signal, 'signal_type'):
            direction = 1 if signal.signal_type == SignalType.BUY else -1
        elif hasattr(signal, 'direction'):
            direction = signal.direction
        
        # Extract timestamp
        timestamp = None
        if hasattr(signal, 'timestamp'):
            timestamp = signal.timestamp
        
        # Extract price
        price = None
        if hasattr(signal, 'price'):
            price = signal.price
        
        # Extract symbol
        symbol = 'default'
        if hasattr(signal, 'symbol') and signal.symbol:
            symbol = signal.symbol
        
        # Create order
        order = Order(
            symbol=symbol,
            order_type="MARKET",
            quantity=abs(position_size),
            direction=direction,
            timestamp=timestamp,
            price=price
        )
        
        # Store order for debugging
        self.orders.append(order)
        
        # Create order event
        order_event = Event(EventType.ORDER, order)
        
        # Emit order event
        self.event_bus.emit(order_event)
        
        logger.debug(f"Generated order from signal: {order}")
    
    def _on_order(self, event):
        """
        Handle order events.
        
        Args:
            event: Order event
        """
        order = event.data
        
        # Forward to execution engine
        if hasattr(self.execution_engine, 'on_order'):
            self.execution_engine.on_order(event)
        else:
            logger.warning("Execution engine does not have on_order method")
    
    def _on_fill(self, event):
        """
        Handle fill events.
        
        Args:
            event: Fill event
        """
        fill = event.data
        
        # Store fill for debugging
        self.fills.append(fill)
        
        # Apply additional processing if needed
    
    def calculate_sharpe(self, risk_free_rate=0.0, annualization_factor=252):
        """
        Calculate Sharpe ratio for the backtest results.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            annualization_factor: Factor to annualize returns (default: 252 trading days)
            
        Returns:
            float: Sharpe ratio
        """
        # Get portfolio history
        portfolio_history = self.execution_engine.get_portfolio_history()
        
        if not portfolio_history or len(portfolio_history) < 2:
            return 0.0
        
        # Extract equity values
        equity_values = [p.get('equity', p.get('total_equity', 0)) for p in portfolio_history]
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Calculate Sharpe ratio
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
            
        sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns)
        
        # Annualize
        sharpe = sharpe * np.sqrt(annualization_factor)
        
        return sharpe
    
    def calculate_max_drawdown(self):
        """
        Calculate maximum drawdown for the backtest.
        
        Returns:
            float: Maximum drawdown percentage
        """
        # Get portfolio history
        portfolio_history = self.execution_engine.get_portfolio_history()
        
        if not portfolio_history:
            return 0.0
        
        # Extract equity values
        equity_values = [p.get('equity', p.get('total_equity', 0)) for p in portfolio_history]
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)
        
        # Calculate drawdown
        drawdowns = (running_max - equity_values) / running_max
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return max_drawdown
    
    def collect_results(self):
        """
        Collect backtest results for analysis.
        
        Returns:
            dict: Results dictionary with trades, portfolio history, etc.
        """
        # Get trade history
        trade_history = self.execution_engine.get_trade_history()
        
        # Process trades into standard format
        processed_trades = self._process_trades(trade_history)
        
        # Calculate performance metrics
        total_return, total_log_return, avg_return = self._calculate_performance_metrics(processed_trades)
        
        # Get portfolio history
        portfolio_history = self.execution_engine.get_portfolio_history()
        
        # Calculate Sharpe ratio
        sharpe_ratio = self.calculate_sharpe()
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        # Return structured results
        results = {
            'trades': processed_trades,
            'num_trades': len(processed_trades),
            'total_log_return': total_log_return,
            'total_percent_return': total_return,
            'average_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_history': portfolio_history,
            'signals': self.signals,
            'config': self.config
        }
        
        logger.info(f"Backtest results: {len(processed_trades)} trades, {total_return:.2f}% return, Sharpe: {sharpe_ratio:.2f}")
        
        return results
    
    def _process_trades(self, trade_history):
        """
        Process trade history into standard format.
        
        Args:
            trade_history: Raw trade history
            
        Returns:
            list: Processed trades in standard format
        """
        # Simple implementation - convert each fill to a trade record
        processed_trades = []
        
        # Pair trades (buy and sell)
        open_positions = {}
        
        for fill in trade_history:
            # Extract trade information
            symbol = getattr(fill, 'symbol', 'default')
            direction = getattr(fill, 'direction', 0)
            if direction == 0 and hasattr(fill, 'order') and hasattr(fill.order, 'direction'):
                direction = fill.order.direction
                
            timestamp = getattr(fill, 'timestamp', None)
            price = getattr(fill, 'fill_price', getattr(fill, 'price', 0))
            quantity = getattr(fill, 'quantity', 0)
            
            # Skip invalid trades
            if direction == 0 or price == 0 or quantity == 0:
                continue
            
            # Check if this closes an existing position
            position_key = f"{symbol}_{1 if direction < 0 else -1}"  # Opposite direction for closing
            
            if position_key in open_positions:
                # This is a closing trade
                entry = open_positions[position_key]
                
                # Calculate return
                if entry['direction'] > 0:  # Long position
                    log_return = np.log(price / entry['price'])
                else:  # Short position
                    log_return = np.log(entry['price'] / price)
                
                # Create trade record
                trade = (
                    entry['timestamp'],
                    'long' if entry['direction'] > 0 else 'short',
                    entry['price'],
                    timestamp,
                    price,
                    log_return
                )
                
                processed_trades.append(trade)
                
                # Remove closed position
                del open_positions[position_key]
            else:
                # This is an opening trade
                position_key = f"{symbol}_{direction}"
                open_positions[position_key] = {
                    'timestamp': timestamp,
                    'price': price,
                    'direction': direction,
                    'quantity': quantity
                }
        
        # Add any remaining open positions with no return
        for key, entry in open_positions.items():
            trade = (
                entry['timestamp'],
                'long' if entry['direction'] > 0 else 'short',
                entry['price'],
                None,  # No exit time
                None,  # No exit price
                0.0    # No return
            )
            processed_trades.append(trade)
        
        return processed_trades
    
    def _calculate_performance_metrics(self, processed_trades):
        """
        Calculate performance metrics from processed trades.
        
        Args:
            processed_trades: List of processed trade records
            
        Returns:
            tuple: (total_return, total_log_return, avg_return)
        """
        # Calculate total log return
        total_log_return = sum(trade[5] for trade in processed_trades)
        
        # Calculate total percentage return
        total_return = (np.exp(total_log_return) - 1) * 100 if processed_trades else 0
        
        # Calculate average return
        avg_return = total_log_return / len(processed_trades) if processed_trades else 0
        
        return total_return, total_log_return, avg_return
    
    def reset(self):
        """Reset the backtester state."""
        # Reset execution engine
        if hasattr(self.execution_engine, 'reset'):
            self.execution_engine.reset()
        
        # Reset strategy
        if hasattr(self.strategy, 'reset'):
            self.strategy.reset()
        
        # Reset position manager
        if hasattr(self.position_manager, 'reset'):
            self.position_manager.reset()
        
        # Reset event bus
        if hasattr(self.event_bus, 'reset'):
            self.event_bus.reset()
            
            # Re-setup event handlers
            self._setup_event_handlers()
        
        # Clear debug tracking
        self.signals = []
        self.orders = []
        self.fills = []


class DefaultPositionManager:
    """
    Default position manager that implements basic functionality.
    Used as a fallback if no position manager is provided.
    """
    
    def __init__(self):
        """Initialize the position manager."""
        self.default_size = 100
    
    def calculate_position_size(self, signal, portfolio):
        """
        Calculate position size based on signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            
        Returns:
            float: Position size (positive for buy, negative for sell)
        """
        # Default position size
        size = self.default_size
        
        # Adjust direction based on signal
        direction = 1  # Default to buy
        
        if hasattr(signal, 'signal_type'):
            if signal.signal_type == SignalType.SELL:
                direction = -1
            elif signal.signal_type == SignalType.NEUTRAL:
                return 0
        elif hasattr(signal, 'direction'):
            direction = signal.direction
        
        # Apply direction to size
        return size * direction
    
    def reset(self):
        """Reset the position manager state."""
        pass

# """
# Backtester for Trading System

# This module provides the main Backtester class which orchestrates the backtesting process,
# connecting strategy, execution, and data components into a cohesive simulation environment.
# """

# from typing import Dict, Any, List, Optional, Callable
# from src.engine import ExecutionEngine, MarketSimulator
# from src.engine.position_manager import PositionManager, DefaultPositionManager

# class EventBus:
#     """
#     Simple event bus for handling and routing events in the system.
#     """
    
#     def __init__(self):
#         """Initialize the event bus."""
#         self.handlers = {}
        
#     def register(self, event_type, handler):
#         """
#         Register a handler for an event type.
        
#         Args:
#             event_type: Type of event to handle
#             handler: Function to call when event is emitted
#         """
#         if event_type not in self.handlers:
#             self.handlers[event_type] = []
#         self.handlers[event_type].append(handler)
        
#     def emit(self, event):
#         """
#         Emit an event to registered handlers.
        
#         Args:
#             event: Event object to emit
#         """
#         if event.event_type in self.handlers:
#             for handler in self.handlers[event.event_type]:
#                 handler(event)
                
#     def reset(self):
#         """Clear all handlers."""
#         self.handlers = {}


# class Backtester:
#     """
#     Main orchestration class that coordinates the backtest execution.
#     Acts as the facade for the backtesting subsystem.
#     """
    
#     def __init__(self, config, data_handler, strategy, position_manager=None):
#         """
#         Initialize the backtester with configuration and dependencies.
        
#         Args:
#             config: Configuration dictionary or ConfigManager instance
#             data_handler: Data handler providing market data
#             strategy: Trading strategy to test
#             position_manager: Optional position manager for risk management
#         """
#         self.config = config
#         self.data_handler = data_handler
#         self.strategy = strategy
#         self.position_manager = position_manager or DefaultPositionManager()
#         self.execution_engine = ExecutionEngine(self.position_manager)
        
#         # Extract market simulation config if present
#         market_sim_config = {}
#         if isinstance(config, dict) and 'market_simulation' in config:
#             market_sim_config = config['market_simulation']
#         elif hasattr(config, 'get'):
#             market_sim_config = config.get('backtester.market_simulation', {})
            
#         self.market_simulator = MarketSimulator(market_sim_config)
#         self.event_bus = EventBus()
        
#         # Set event bus in execution engine
#         self.execution_engine.event_bus = self.event_bus
        
#         # Register event handlers
#         self.event_bus.register(EventType.SIGNAL, self.on_signal)
#         self.event_bus.register(EventType.ORDER, self.execution_engine.on_order)
#         self.event_bus.register(EventType.FILL, self.on_fill)
    
#     def run(self, use_test_data=False):
#         """
#         Run the backtest.
        
#         Args:
#             use_test_data: Whether to use test data (True) or training data (False)
            
#         Returns:
#             dict: Backtest results
#         """
#         # Reset all components
#         self.reset()
        
#         # Select data source and iterator method
#         if use_test_data:
#             self.data_handler.reset_test()
#             data_iter = self._iterate_test_data
#         else:
#             self.data_handler.reset_train()
#             data_iter = self._iterate_train_data
        
#         # Process each bar
#         for bar in data_iter():
#             # Create bar event
#             bar_event = Event(EventType.BAR, data=bar)
#             bar_event.bar = bar  # Add this line to set the bar attribute
            
#             # Process bar through strategy
#             signals = self.strategy.on_bar(bar_event)
            
#             # Emit signal events
#             if signals:
#                 # Handle both individual signals and lists of signals
#                 if not isinstance(signals, list):
#                     signals = [signals]
                    
#                 for signal in signals:
#                     signal_event = Event(EventType.SIGNAL, data=signal)
#                     self.event_bus.emit(signal_event)
            
#             # Update portfolio state with latest bar
#             self.execution_engine.update(bar)
            
#             # Execute any pending orders
#             self.execution_engine.execute_pending_orders(
#                 bar, self.market_simulator
#             )
        
#         # Collect results
#         return self.collect_results()
    
#     def _iterate_train_data(self):
#         """
#         Iterator for training data.
        
#         Yields:
#             dict: Bar data
#         """
#         while True:
#             bar = self.data_handler.get_next_train_bar()
#             if bar is None:
#                 break
#             yield bar
    
#     def _iterate_test_data(self):
#         """
#         Iterator for test data.
        
#         Yields:
#             dict: Bar data
#         """
#         while True:
#             bar = self.data_handler.get_next_test_bar()
#             if bar is None:
#                 break
#             yield bar
    
#     def on_signal(self, event):
#         """
#         Handle signal events by generating orders.
        
#         Args:
#             event: Signal event
#         """
#         signal = event.data
        
#         # Record signal in history
#         self.execution_engine.signal_history.append(signal)
        
#         # Get position sizing from position manager
#         position_size = self.position_manager.calculate_position_size(
#             signal, self.execution_engine.portfolio
#         )
        
#         # Create order if position size is non-zero
#         if position_size != 0:
#             order = Order(
#                 symbol=signal.symbol if hasattr(signal, 'symbol') else 'default',
#                 order_type="MARKET",
#                 quantity=abs(position_size),
#                 direction=1 if position_size > 0 else -1,
#                 timestamp=signal.timestamp if hasattr(signal, 'timestamp') else None
#             )
            
#             # Emit order event
#             order_event = Event(EventType.ORDER, data=order)
#             self.event_bus.emit(order_event)
    
#     def on_fill(self, event):
#         """
#         Handle fill events.
        
#         Args:
#             event: Fill event
#         """
#         # Process fill information - implement specific logic here if needed
#         pass
    
#     def collect_results(self):
#         """
#         Collect backtest results for analysis.
        
#         Returns:
#             dict: Results dictionary with trades, portfolio history, etc.
#         """
#         # Get trade history from execution engine
#         trades = [
#             (fill.timestamp, 
#              "long" if fill.order.direction > 0 else "short",
#              fill.fill_price,
#              fill.timestamp,  # Using same timestamp for entry and exit for simplicity
#              fill.fill_price,
#              0.0)  # Log return placeholder - would be calculated from paired trades in a real implementation
#             for fill in self.execution_engine.trade_history
#         ]
        
#         # Calculate log returns for trades (in a real implementation)
#         # This is a simplified version that assumes sequential pairs of trades
#         entry_price = None
#         entry_type = None
#         entry_time = None
#         processed_trades = []
        
#         for i, trade in enumerate(trades):
#             timestamp, direction, price, _, _, _ = trade
            
#             if entry_price is None:
#                 # This is an entry
#                 entry_price = price
#                 entry_type = direction
#                 entry_time = timestamp
#             else:
#                 # This is an exit, calculate return
#                 if entry_type == "long":
#                     log_return = np.log(price / entry_price)
#                 else:  # short
#                     log_return = np.log(entry_price / price)
                    
#                 # Create complete trade record
#                 processed_trade = (
#                     entry_time,
#                     entry_type,
#                     entry_price,
#                     timestamp,
#                     price,
#                     log_return
#                 )
#                 processed_trades.append(processed_trade)
                
#                 # Reset for next pair
#                 entry_price = None
        
#         # If odd number of trades, add the last one with no return
#         if entry_price is not None:
#             processed_trade = (
#                 entry_time,
#                 entry_type,
#                 entry_price,
#                 None,  # No exit time
#                 None,  # No exit price
#                 0.0    # No return
#             )
#             processed_trades.append(processed_trade)
        
#         # Calculate basic metrics
#         total_log_return = sum(trade[5] for trade in processed_trades)
#         total_return = (np.exp(total_log_return) - 1) * 100 if processed_trades else 0
#         avg_log_return = total_log_return / len(processed_trades) if processed_trades else 0
        
#         # Return structured results
#         return {
#             'trades': processed_trades,
#             'num_trades': len(processed_trades),
#             'total_log_return': total_log_return,
#             'total_percent_return': total_return,
#             'average_log_return': avg_log_return,
#             'portfolio_history': self.execution_engine.get_portfolio_history(),
#             'signals': self.execution_engine.get_signal_history(),
#             'config': self.config
#         }
    
#     def reset(self):
#         """Reset the backtester state."""
#         self.execution_engine.reset()
#         self.strategy.reset()
#         self.event_bus.reset()
        
#         # Reset position manager if it has a reset method
#         if hasattr(self.position_manager, 'reset'):
#             self.position_manager.reset()


# class DefaultPositionManager:
#     """
#     Default position manager that implements basic functionality.
#     Used as a fallback if no position manager is provided.
#     """
    
#     def calculate_position_size(self, signal, portfolio):
#         """
#         Calculate position size based on signal.
        
#         This simple implementation just returns a fixed position size
#         in the direction of the signal.
        
#         Args:
#             signal: Trading signal
#             portfolio: Current portfolio
            
#         Returns:
#             float: Position size (positive for buy, negative for sell)
#         """
#         # Default position size (100 units)
#         size = 100.0
        
#         # Adjust direction based on signal
#         if hasattr(signal, 'signal_type'):
#             if signal.signal_type.value < 0:  # Sell signal
#                 size = -size
#             elif signal.signal_type.value == 0:  # Neutral signal
#                 size = 0
#         elif hasattr(signal, 'signal') and signal.signal < 0:
#             size = -size
#         elif hasattr(signal, 'direction') and signal.direction < 0:
#             size = -size
            
#         return size
    
#     def reset(self):
#         """Reset the position manager state."""
#         pass


# # Import these from other modules in real implementation
# from enum import Enum, auto
# import numpy as np

# class EventType(Enum):
#     BAR = auto()
#     SIGNAL = auto()
#     ORDER = auto()
#     FILL = auto()

# class Event:
#     def __init__(self, event_type, data=None):
#         self.event_type = event_type
#         self.data = data

# class Order:
#     def __init__(self, symbol, order_type, quantity, direction, timestamp):
#         self.symbol = symbol
#         self.order_type = order_type
#         self.quantity = quantity
#         self.direction = direction
#         self.timestamp = timestamp
