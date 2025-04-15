"""
Backtester for Trading System

This module provides the main Backtester class which orchestrates the backtesting process,
connecting strategy, execution, and data components into a cohesive simulation environment.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime


from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType
from src.signals import Signal, SignalType
from src.engine.execution_engine import ExecutionEngine
from src.engine.market_simulator import MarketSimulator


# Try to import comprehensive position management
try:
    from src.position_management.position_manager import PositionManager as ComprehensivePositionManager
    from src.position_management.portfolio import Portfolio
    POSITION_MANAGEMENT_AVAILABLE = True
except ImportError:
    POSITION_MANAGEMENT_AVAILABLE = False
    logging.warning("Comprehensive position management module not available, using simplified version")

# Set up logging
logger = logging.getLogger(__name__)


class BarEvent:
    def __init__(self, bar):
        self.bar = bar


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


class DefaultPositionManager:
    """
    Simple position manager that implements basic functionality.
    Used as a fallback if the comprehensive position management module is not available.
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
            if hasattr(signal.signal_type, 'value'):
                direction = signal.signal_type.value
            elif isinstance(signal.signal_type, str):
                if signal.signal_type in ['SELL', 'SHORT']:
                    direction = -1
                elif signal.signal_type == 'NEUTRAL':
                    return 0
        elif hasattr(signal, 'direction'):
            direction = signal.direction
        
        # Apply direction to size
        return size * direction
    
    def reset(self):
        """Reset the position manager state."""
        pass


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
        
        # Initialize position management
        if position_manager is not None:
            self.position_manager = position_manager
        elif POSITION_MANAGEMENT_AVAILABLE:
            # Use the comprehensive position management if available
            initial_capital = self._extract_initial_capital(config)
            portfolio = Portfolio(initial_capital=initial_capital)
            self.position_manager = ComprehensivePositionManager(
                portfolio=portfolio, 
                max_positions=config.get('max_positions', 0) if hasattr(config, 'get') else 0
            )
        else:
            # Fall back to the simple position manager
            self.position_manager = DefaultPositionManager()
        
        # Initialize execution components
        self.execution_engine = ExecutionEngine(self.position_manager)
        
        # Initialize event system
        self.event_bus = EventBus()
        self.execution_engine.event_bus = self.event_bus
        
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
        try:
            # Reset all components before running
            self.reset()
            logger.info("Starting backtest...")

            # Select data iterator based on data set
            iterator = self.data_handler.iter_test() if use_test_data else self.data_handler.iter_train()

            # Process each bar
            for bar_event in iterator:
                # Get the bar data - handle both raw dict and BarEvent
                if hasattr(bar_event, 'bar'):
                    bar = bar_event.bar  # It's already a BarEvent
                else:
                    bar = bar_event  # It's a raw dict
                    bar_event = BarEvent(bar)  # Wrap it

                # Log some bars for debugging
                # if isinstance(bar, dict) and 'timestamp' in bar:
                    # logger.info(f"Processing bar for {bar['timestamp']} - Close: {bar.get('Close')}")

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
        except Exception as e:
            logger.error(f"Error during backtest execution: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Return empty results instead of None
            return {
                'error': str(e),
                'trades': [],
                'num_trades': 0,
                'signals': self.signals,
                'orders': self.orders,
                'total_percent_return': 0,
                'total_log_return': 0,
                'average_return': 0,
                'portfolio_history': []
            }


    def _on_signal(self, event):
        """
        Handle signal events by generating orders.
        
        Args:
            event: Signal event
        """
        signal = event.data
        
        # Skip neutral signals
        if hasattr(signal, 'signal_type') and (
            signal.signal_type == SignalType.NEUTRAL or 
            getattr(signal.signal_type, 'value', 0) == 0
        ):
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
            if hasattr(signal.signal_type, 'value'):
                direction = 1 if signal.signal_type.value > 0 else -1
            elif isinstance(signal.signal_type, str):
                if signal.signal_type in ['SELL', 'SHORT']:
                    direction = -1
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
        logger.info(f"Backtester handling order: {order}")

        # Forward to execution engine
        if hasattr(self.execution_engine, 'on_order'):
            self.execution_engine.on_order(event)
            logger.info(f"Forwarded order to execution engine")
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
        try:
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
                'orders': self.orders,
                'config': self.config
            }

            logger.info(f"Backtest results: {len(processed_trades)} trades, {total_return:.2f}% return, Sharpe: {sharpe_ratio:.2f}")

            return results
        except Exception as e:
            logger.error(f"Error collecting backtest results: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Return basic results
            return {
                'error': str(e),
                'trades': [],
                'num_trades': 0,
                'signals': self.signals,
                'orders': self.orders,
                'total_percent_return': 0,
                'total_log_return': 0,
                'average_return': 0,
                'portfolio_history': []
            }


    def _process_trades(self, trade_history):
        """
        Process trade history into a standardized format.

        Args:
            trade_history: List of Fill objects representing trades

        Returns:
            List of processed trade records
        """
        processed_trades = []

        # Create pairs of entry and exit for each position
        current_position = None

        for fill in trade_history:
            if current_position is None:
                # Start a new position
                current_position = (
                    fill.timestamp,  # Entry time
                    fill.direction,  # Direction
                    fill.fill_price,  # Entry price
                    None,  # Exit time (to be filled later)
                    None,  # Exit price (to be filled later)
                    0.0    # Log return (to be calculated)
                )
            else:
                # Close the position
                entry_time, direction, entry_price, _, _, _ = current_position

                # Calculate log return
                if direction > 0:  # Long position
                    log_return = np.log(fill.fill_price / entry_price)
                else:  # Short position
                    log_return = np.log(entry_price / fill.fill_price)

                # Complete the trade record
                trade_record = (
                    entry_time,
                    direction,
                    entry_price,
                    fill.timestamp,  # Exit time
                    fill.fill_price,  # Exit price
                    log_return
                )

                processed_trades.append(trade_record)
                current_position = None

        # If we have an open position at the end
        if current_position is not None:
            processed_trades.append(current_position)

        return processed_trades

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

            # Debug log the raw signal details
            logger.debug(f"Raw signal: {signal}")
            if hasattr(signal, 'signal_type'):
                logger.debug(f"Signal type: {signal.signal_type}")
            if hasattr(signal, 'confidence'):
                logger.debug(f"Signal confidence: {signal.confidence}")

            # Skip neutral signals
            if hasattr(signal, 'signal_type') and (
                signal.signal_type == SignalType.NEUTRAL or 
                getattr(signal.signal_type, 'value', 0) == 0
            ):
                logger.debug("Skipping neutral signal")
                continue

            # Calculate position size - FIXED PART
            # Instead of calling calculate_position_size on the position_manager,
            # we need to use the position_sizer if available
            if hasattr(self.position_manager, 'position_sizer') and self.position_manager.position_sizer is not None:
                # Use the position_sizer from the position_manager
                position_size = self.position_manager.position_sizer.calculate_position_size(
                    signal, 
                    self.position_manager.portfolio, 
                    bar.get('Close', 0)
                )
            else:
                # Fallback to a default size
                position_size = 100  # Default to 100 shares/contracts

            # Skip if position size is zero
            if position_size == 0:
                logger.debug("Position size is zero, skipping order generation")
                continue

            # Determine direction
            direction = 1  # Default to buy
            if hasattr(signal, 'signal_type'):
                if hasattr(signal.signal_type, 'value'):
                    direction = 1 if signal.signal_type.value > 0 else -1
                elif isinstance(signal.signal_type, str):
                    if signal.signal_type in ['SELL', 'SHORT']:
                        direction = -1
            elif hasattr(signal, 'direction'):
                direction = signal.direction

            # Extract timestamp
            timestamp = getattr(signal, 'timestamp', datetime.now())

            # Extract price
            price = getattr(signal, 'price', bar.get('Close', 0))

            # Extract symbol
            symbol = getattr(signal, 'symbol', 'default')

            logger.info(f"Creating order from signal: direction={direction}, size={abs(position_size)}")

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

 
    def _calculate_performance_metrics(self, processed_trades):
        """
        Calculate performance metrics from processed trades.
        
        Args:
            processed_trades: List of processed trade records
            
        Returns:
            tuple: (total_return, total_log_return, avg_return)
        """
        # Calculate total log return
        total_log_return = sum(trade[5] for trade in processed_trades if len(trade) > 5 and trade[5] is not None)
        
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
