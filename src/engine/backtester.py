"""
Backtester for Trading System - Standardized Version

This module provides the main Backtester class which orchestrates the backtesting process,
using standardized event objects throughout the workflow.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

from src.events.event_bus import Event, EventBus
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.events.event_utils import create_error_event
from src.engine.execution_engine import ExecutionEngine
from src.engine.market_simulator import MarketSimulator

# Set up logging
logger = logging.getLogger(__name__)

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
        self.position_manager = position_manager
        
        # Initialize event system
        self.event_bus = EventBus()
        
        # Initialize execution components
        self.execution_engine = ExecutionEngine(self.position_manager)
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
        self._event_handlers = {}
        
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

    # In Backtester._setup_event_handlers
    def _setup_event_handlers(self):
        """Set up event handlers with proper registration."""
        # Create and store handlers with strong references
        self._event_handlers['bar'] = self._create_bar_handler()
        self._event_handlers['signal'] = self._create_signal_handler()
        self._event_handlers['order'] = self._create_order_handler()
        self._event_handlers['fill'] = self._create_fill_handler()

        # Register handlers with event bus
        self.event_bus.register(EventType.BAR, self._event_handlers['bar'])
        self.event_bus.register(EventType.SIGNAL, self._event_handlers['signal'])
        self.event_bus.register(EventType.ORDER, self._event_handlers['order'])
        self.event_bus.register(EventType.FILL, self._event_handlers['fill'])

        logger.info("Event handlers registered")
    


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
            iterator = self.data_handler.iter_test(use_bar_events=True) if use_test_data else self.data_handler.iter_train(use_bar_events=True)
            
            # Process each bar
            for bar_event in iterator:
                # Ensure we have a proper BarEvent
                if not isinstance(bar_event, BarEvent):
                    logger.warning(f"Expected BarEvent but got {type(bar_event)}, attempting to convert")
                    try:
                        if isinstance(bar_event, dict) and 'Close' in bar_event:
                            # Convert dict to BarEvent
                            bar_event = BarEvent(bar_event)
                        else:
                            logger.error(f"Could not convert {type(bar_event)} to BarEvent")
                            continue
                    except Exception as e:
                        logger.error(f"Error converting to BarEvent: {e}")
                        continue
                
                # Process bar through strategy to get signals
                try:
                    # Create Event containing BarEvent
                    bar_event_container = Event(EventType.BAR, bar_event)
                    
                    # Process through strategy
                    signal_event = self.strategy.on_bar(bar_event_container)
                    
                    # Handle signals from strategy
                    if signal_event:
                        # Process single signal
                        if isinstance(signal_event, SignalEvent):
                            self._process_signal(signal_event, bar_event)
                        # Process multiple signals if returned as a list
                        elif isinstance(signal_event, list):
                            for signal in signal_event:
                                if isinstance(signal, SignalEvent) and signal is not None:
                                    self._process_signal(signal, bar_event)
                except Exception as e:
                    logger.error(f"Error processing bar through strategy: {e}")
                    # Emit error event
                    self.event_bus.emit(create_error_event("Backtester", str(e), "StrategyProcessingError"))
                
                # Update portfolio with current bar data
                self.execution_engine.update(bar_event)
                
                # Execute any pending orders
                self.execution_engine.execute_pending_orders(bar_event, self.market_simulator)
            
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
        # Extract signal from event
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event)}")
            return
            
        if not isinstance(event.data, SignalEvent):
            logger.error(f"Expected SignalEvent in event.data, got {type(event.data)}")
            return
            
        signal = event.data
        
        # Skip neutral signals
        if signal.get_signal_value() == SignalEvent.NEUTRAL:
            return
        
        # Calculate position size
        position_size = self.position_manager.calculate_position_size(
            signal, 
            self.execution_engine.portfolio if hasattr(self.execution_engine, 'portfolio') else {'equity': self.initial_capital}
        )
        
        # Skip if position size is zero
        if position_size == 0:
            return
        
        # Create order event
        order_event = OrderEvent(
            symbol=signal.get_symbol(),
            direction=signal.get_signal_value(),
            quantity=abs(position_size),
            price=signal.get_price(),
            order_type="MARKET",
            timestamp=signal.timestamp
        )
        
        # Store order for debugging
        self.orders.append(order_event)
        
        # Emit order event
        self.event_bus.emit(Event(EventType.ORDER, order_event))
        
        logger.debug(f"Generated order from signal: {order_event}")

    def _on_order(self, event):
        """
        Handle order events.

        Args:
            event: Order event
        """
        # Extract order from event
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event)}")
            return
            
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
        # Extract fill from event
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event)}")
            return
            
        fill = event.data
        
        # Store fill for debugging
        self.fills.append(fill)
        
        # Apply additional processing if needed
    
    def _process_signal(self, signal_event: SignalEvent, bar_event: BarEvent):
        """
        Process a single signal.
        
        Args:
            signal_event: Signal to process
            bar_event: Current bar data
        """
        if signal_event is None:
            return
            
        # Store signal for debugging
        self.signals.append(signal_event)
        
        # Create and emit signal event
        self.event_bus.emit(Event(EventType.SIGNAL, signal_event))
        
        logger.debug(f"Emitted signal event: {signal_event.get_signal_name()} for {signal_event.get_symbol()}")

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
        """Process trade history into a standardized format."""
        # Implementation remains the same
        pass
        
    def _calculate_performance_metrics(self, processed_trades):
        """Calculate performance metrics from processed trades."""
        # Implementation remains the same
        pass
        
    def calculate_sharpe(self, risk_free_rate=0.0, annualization_factor=252):
        """Calculate Sharpe ratio for the backtest results."""
        # Implementation remains the same
        pass
        
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown for the backtest."""
        # Implementation remains the same
        pass

