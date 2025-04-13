"""
Backtester for Trading System

This module provides the main Backtester class which orchestrates the backtesting process,
connecting strategy, execution, and data components into a cohesive simulation environment.
"""

from typing import Dict, Any, List, Optional, Callable
from . import ExecutionEngine, MarketSimulator
from ..position_management import PositionManager, DefaultPositionManager


class EventBus:
    """
    Simple event bus for handling and routing events in the system.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.handlers = {}
        
    def register(self, event_type, handler):
        """
        Register a handler for an event type.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event is emitted
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
    def emit(self, event):
        """
        Emit an event to registered handlers.
        
        Args:
            event: Event object to emit
        """
        if event.event_type in self.handlers:
            for handler in self.handlers[event.event_type]:
                handler(event)
                
    def reset(self):
        """Clear all handlers."""
        self.handlers = {}


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
        self.position_manager = position_manager or DefaultPositionManager()
        self.execution_engine = ExecutionEngine(self.position_manager)
        
        # Extract market simulation config if present
        market_sim_config = {}
        if isinstance(config, dict) and 'market_simulation' in config:
            market_sim_config = config['market_simulation']
        elif hasattr(config, 'get'):
            market_sim_config = config.get('backtester.market_simulation', {})
            
        self.market_simulator = MarketSimulator(market_sim_config)
        self.event_bus = EventBus()
        
        # Set event bus in execution engine
        self.execution_engine.event_bus = self.event_bus
        
        # Register event handlers
        self.event_bus.register(EventType.SIGNAL, self.on_signal)
        self.event_bus.register(EventType.ORDER, self.execution_engine.on_order)
        self.event_bus.register(EventType.FILL, self.on_fill)
    
    def run(self, use_test_data=False):
        """
        Run the backtest.
        
        Args:
            use_test_data: Whether to use test data (True) or training data (False)
            
        Returns:
            dict: Backtest results
        """
        # Reset all components
        self.reset()
        
        # Select data source and iterator method
        if use_test_data:
            self.data_handler.reset_test()
            data_iter = self._iterate_test_data
        else:
            self.data_handler.reset_train()
            data_iter = self._iterate_train_data
        
        # Process each bar
        for bar in data_iter():
            # Create bar event
            bar_event = Event(EventType.BAR, data=bar)
            
            # Process bar through strategy
            signals = self.strategy.on_bar(bar_event)
            
            # Emit signal events
            if signals:
                # Handle both individual signals and lists of signals
                if not isinstance(signals, list):
                    signals = [signals]
                    
                for signal in signals:
                    signal_event = Event(EventType.SIGNAL, data=signal)
                    self.event_bus.emit(signal_event)
            
            # Update portfolio state with latest bar
            self.execution_engine.update(bar)
            
            # Execute any pending orders
            self.execution_engine.execute_pending_orders(
                bar, self.market_simulator
            )
        
        # Collect results
        return self.collect_results()
    
    def _iterate_train_data(self):
        """
        Iterator for training data.
        
        Yields:
            dict: Bar data
        """
        while True:
            bar = self.data_handler.get_next_train_bar()
            if bar is None:
                break
            yield bar
    
    def _iterate_test_data(self):
        """
        Iterator for test data.
        
        Yields:
            dict: Bar data
        """
        while True:
            bar = self.data_handler.get_next_test_bar()
            if bar is None:
                break
            yield bar
    
    def on_signal(self, event):
        """
        Handle signal events by generating orders.
        
        Args:
            event: Signal event
        """
        signal = event.data
        
        # Record signal in history
        self.execution_engine.signal_history.append(signal)
        
        # Get position sizing from position manager
        position_size = self.position_manager.calculate_position_size(
            signal, self.execution_engine.portfolio
        )
        
        # Create order if position size is non-zero
        if position_size != 0:
            order = Order(
                symbol=signal.symbol if hasattr(signal, 'symbol') else 'default',
                order_type="MARKET",
                quantity=abs(position_size),
                direction=1 if position_size > 0 else -1,
                timestamp=signal.timestamp if hasattr(signal, 'timestamp') else None
            )
            
            # Emit order event
            order_event = Event(EventType.ORDER, data=order)
            self.event_bus.emit(order_event)
    
    def on_fill(self, event):
        """
        Handle fill events.
        
        Args:
            event: Fill event
        """
        # Process fill information - implement specific logic here if needed
        pass
    
    def collect_results(self):
        """
        Collect backtest results for analysis.
        
        Returns:
            dict: Results dictionary with trades, portfolio history, etc.
        """
        # Get trade history from execution engine
        trades = [
            (fill.timestamp, 
             "long" if fill.order.direction > 0 else "short",
             fill.fill_price,
             fill.timestamp,  # Using same timestamp for entry and exit for simplicity
             fill.fill_price,
             0.0)  # Log return placeholder - would be calculated from paired trades in a real implementation
            for fill in self.execution_engine.trade_history
        ]
        
        # Calculate log returns for trades (in a real implementation)
        # This is a simplified version that assumes sequential pairs of trades
        entry_price = None
        entry_type = None
        entry_time = None
        processed_trades = []
        
        for i, trade in enumerate(trades):
            timestamp, direction, price, _, _, _ = trade
            
            if entry_price is None:
                # This is an entry
                entry_price = price
                entry_type = direction
                entry_time = timestamp
            else:
                # This is an exit, calculate return
                if entry_type == "long":
                    log_return = np.log(price / entry_price)
                else:  # short
                    log_return = np.log(entry_price / price)
                    
                # Create complete trade record
                processed_trade = (
                    entry_time,
                    entry_type,
                    entry_price,
                    timestamp,
                    price,
                    log_return
                )
                processed_trades.append(processed_trade)
                
                # Reset for next pair
                entry_price = None
        
        # If odd number of trades, add the last one with no return
        if entry_price is not None:
            processed_trade = (
                entry_time,
                entry_type,
                entry_price,
                None,  # No exit time
                None,  # No exit price
                0.0    # No return
            )
            processed_trades.append(processed_trade)
        
        # Calculate basic metrics
        total_log_return = sum(trade[5] for trade in processed_trades)
        total_return = (np.exp(total_log_return) - 1) * 100 if processed_trades else 0
        avg_log_return = total_log_return / len(processed_trades) if processed_trades else 0
        
        # Return structured results
        return {
            'trades': processed_trades,
            'num_trades': len(processed_trades),
            'total_log_return': total_log_return,
            'total_percent_return': total_return,
            'average_log_return': avg_log_return,
            'portfolio_history': self.execution_engine.get_portfolio_history(),
            'signals': self.execution_engine.get_signal_history(),
            'config': self.config
        }
    
    def reset(self):
        """Reset the backtester state."""
        self.execution_engine.reset()
        self.strategy.reset()
        self.event_bus.reset()
        
        # Reset position manager if it has a reset method
        if hasattr(self.position_manager, 'reset'):
            self.position_manager.reset()


class DefaultPositionManager:
    """
    Default position manager that implements basic functionality.
    Used as a fallback if no position manager is provided.
    """
    
    def calculate_position_size(self, signal, portfolio):
        """
        Calculate position size based on signal.
        
        This simple implementation just returns a fixed position size
        in the direction of the signal.
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio
            
        Returns:
            float: Position size (positive for buy, negative for sell)
        """
        # Default position size (100 units)
        size = 100.0
        
        # Adjust direction based on signal
        if hasattr(signal, 'signal_type'):
            if signal.signal_type.value < 0:  # Sell signal
                size = -size
            elif signal.signal_type.value == 0:  # Neutral signal
                size = 0
        elif hasattr(signal, 'signal') and signal.signal < 0:
            size = -size
        elif hasattr(signal, 'direction') and signal.direction < 0:
            size = -size
            
        return size
    
    def reset(self):
        """Reset the position manager state."""
        pass


# Import these from other modules in real implementation
from enum import Enum, auto
import numpy as np

class EventType(Enum):
    BAR = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()

class Event:
    def __init__(self, event_type, data=None):
        self.event_type = event_type
        self.data = data

class Order:
    def __init__(self, symbol, order_type, quantity, direction, timestamp):
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.timestamp = timestamp
