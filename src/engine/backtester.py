"""
Backtester for Trading System - Fixed Version

This module provides the main Backtester class which orchestrates the backtesting process,
ensuring proper initialization of components and event flow.
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
        
        # Configure initial capital
        self.initial_capital = self._extract_initial_capital(config)
        
        # Initialize market simulator
        market_sim_config = self._extract_market_sim_config(config)
        self.market_simulator = MarketSimulator(market_sim_config)
        
        # Initialize execution components - CRITICAL: Do this after setting up position_manager
        # and properly access the portfolio reference from position_manager
        self.execution_engine = ExecutionEngine(self.position_manager)
        self.execution_engine.event_bus = self.event_bus
        
        # CRITICAL FIX: Set portfolio reference properly if available
        if self.position_manager and hasattr(self.position_manager, 'portfolio'):
            self.execution_engine.portfolio = self.position_manager.portfolio
            logger.info("Set execution engine portfolio reference from position manager")
        else:
            logger.warning("No portfolio reference available in position manager")
        
        # Initialize event handlers dictionary
        self._event_handlers = {}
        
        # Set up event handlers
        self._setup_event_handlers()
        
        
        # Track signals for debugging
        self.signals = []
        self.orders = []
        self.fills = []
        
        logger.info(f"Backtester initialized with initial capital: {self.initial_capital}")

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
        
        # CRITICAL: Register position action handler and execution engine
        # This connects position actions to order creation
        self.event_bus.register(EventType.POSITION_ACTION, self.execution_engine.on_position_action)
        logger.info("Registered execution engine for POSITION_ACTION events")
        
        # CRITICAL: Register position manager to receive signal events
        self.event_bus.register(EventType.SIGNAL, self.position_manager.on_signal)
        logger.info("Registered position manager for SIGNAL events")
        
        # Add debugging handler for all events
        def debug_event_flow(event):
            event_type_name = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
            logger.debug(f"Event flow: {event_type_name} event received")
            
        # Register for all event types
        for event_type in EventType:
            self.event_bus.register(event_type, debug_event_flow)        
    
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
        """Set up event handlers with proper registration."""
        # Create and store handlers with strong references
        self._event_handlers['bar'] = self._create_bar_handler()
        self._event_handlers['signal'] = self._create_signal_handler()
        self._event_handlers['order'] = self._create_order_handler()
        self._event_handlers['fill'] = self._create_fill_handler()
        
        # CRITICAL FIX: Create position action handler
        self._event_handlers['position_action'] = self._create_position_action_handler()

        # Register handlers with event bus
        self.event_bus.register(EventType.BAR, self._event_handlers['bar'])
        self.event_bus.register(EventType.SIGNAL, self._event_handlers['signal'])
        self.event_bus.register(EventType.ORDER, self._event_handlers['order'])
        self.event_bus.register(EventType.FILL, self._event_handlers['fill'])
        
        # CRITICAL FIX: Register position action handler
        self.event_bus.register(EventType.POSITION_ACTION, self._event_handlers['position_action'])
        
        # CRITICAL FIX: Directly register position manager to receive signal events
        if self.position_manager and hasattr(self.position_manager, 'on_signal'):
            self.event_bus.register(EventType.SIGNAL, self.position_manager.on_signal)
            logger.info("Registered position manager for SIGNAL events")
        
        # CRITICAL FIX: Register execution engine to receive order events
        if self.execution_engine:
            self.event_bus.register(EventType.ORDER, self.execution_engine.on_order)
            logger.info("Registered execution engine for ORDER events")
        
        # Add debugging handler for all events
        def debug_event_flow(event):
            event_type_name = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
            logger.debug(f"Event flow: {event_type_name} event received")
            
            if event.event_type == EventType.SIGNAL:
                signal = event.data
                if hasattr(signal, 'get_signal_value') and hasattr(signal, 'get_symbol') and hasattr(signal, 'get_price'):
                    direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
                    logger.debug(f"Signal: {direction} for {signal.get_symbol()} at {signal.get_price()}")
            
            elif event.event_type == EventType.POSITION_ACTION:
                action = event.data
                if isinstance(action, dict):
                    logger.debug(f"Position action: {action.get('action_type', 'unknown')}")
            
            elif event.event_type == EventType.ORDER:
                order = event.data
                if hasattr(order, 'get_symbol'):
                    logger.debug(f"Order: {order.get_symbol()}")
                    
            elif event.event_type == EventType.FILL:
                fill = event.data
                if hasattr(fill, 'get_symbol'):
                    logger.debug(f"Fill: {fill.get_symbol()}")
                    
        # Register for all event types
        for event_type in EventType:
            self.event_bus.register(event_type, debug_event_flow)

        logger.info("Event handlers registered")
    
    def _create_bar_handler(self):
        """Create a handler for BAR events."""
        def handle_bar_event(event):
            """Extract bar data properly for strategy consumption."""
            try:
                # Ensure we're working with an Event object
                if not isinstance(event, Event):
                    logger.warning(f"Expected Event object, got {type(event).__name__}")
                    return

                # Extract BarEvent with proper type checking
                bar_event = None
                
                # Case 1: Event has a BarEvent in data attribute
                if hasattr(event, 'data') and isinstance(event.data, BarEvent):
                    bar_event = event.data
                # Case 2: Event data is a dictionary (deprecated)
                elif hasattr(event, 'data') and isinstance(event.data, dict) and 'Close' in event.data:
                    logger.warning(
                        "Using dictionary for bar data is deprecated. Use BarEvent objects instead."
                    )
                    bar_event = BarEvent(event.data)
                # Case 3: Event is a BarEvent itself
                elif isinstance(event, BarEvent):
                    bar_event = event

                if bar_event is None:
                    logger.warning(f"Could not extract BarEvent from event: {event}")
                    return

                # Create standard Event with BarEvent if needed
                if not isinstance(event, Event) or not isinstance(event.data, BarEvent):
                    event_to_pass = Event(EventType.BAR, bar_event, bar_event.get_timestamp())
                else:
                    event_to_pass = event

                # Process through strategy
                if hasattr(self.strategy, 'on_bar'):
                    self.strategy.on_bar(event_to_pass)
                elif hasattr(self.strategy, 'handle_event'):
                    self.strategy.handle_event(event_to_pass)

                logger.debug(f"Processed bar: {bar_event.get_symbol()} at {bar_event.get_timestamp()}")

            except Exception as e:
                logger.error(f"Error processing bar event: {str(e)}", exc_info=True)

        return handle_bar_event

    def _create_signal_handler(self):
        """Create a handler for SIGNAL events."""
        def handle_signal_event(event):
            """Process signals and emit to the event bus."""
            try:
                # Ensure we're working with a SignalEvent
                if not hasattr(event, 'data') or not isinstance(event.data, SignalEvent):
                    logger.warning(f"Expected SignalEvent in event.data, got {type(event.data).__name__ if hasattr(event, 'data') else 'None'}")
                    return
                
                signal = event.data

                # Store signal for debugging
                self.signals.append(signal)
                
                # Log the signal
                if hasattr(signal, 'get_signal_value') and hasattr(signal, 'get_symbol'):
                    direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
                    logger.debug(f"Emitted signal event: {direction} for {signal.get_symbol()}")
                
            except Exception as e:
                logger.error(f"Error processing signal event: {str(e)}", exc_info=True)
        
        return handle_signal_event
    
    def _create_order_handler(self):
        """Create a handler for ORDER events."""
        def handle_order_event(event):
            """Process order events and store for history."""
            try:
                order = event.data
                
                # Store order for debugging
                self.orders.append(order)
                
                logger.debug(f"Processed order event: {order}")
                
            except Exception as e:
                logger.error(f"Error processing order event: {str(e)}", exc_info=True)
        
        return handle_order_event
    
    def _create_fill_handler(self):
        """Create a handler for FILL events."""
        def handle_fill_event(event):
            """Process fill events and store for history."""
            try:
                fill = event.data
                
                # Store fill for debugging
                self.fills.append(fill)
                
                logger.debug(f"Processed fill event: {fill}")
                
            except Exception as e:
                logger.error(f"Error processing fill event: {str(e)}", exc_info=True)
        
        return handle_fill_event
        
    def _create_position_action_handler(self):
        """Create a handler for POSITION_ACTION events."""
        def handle_position_action_event(event):
            """Process position action events."""
            try:
                # Extract position action from event
                if not hasattr(event, 'data'):
                    logger.warning("Position action event has no data")
                    return
                    
                position_action = event.data
                
                # Log the position action
                if isinstance(position_action, dict):
                    action_type = position_action.get('action_type', 'unknown')
                    symbol = position_action.get('symbol', 'unknown')
                    logger.debug(f"Position action received: {action_type} for {symbol}")
                else:
                    logger.debug(f"Position action received: {position_action}")
                
                # Execute the position action using the position manager
                if self.position_manager and hasattr(self.position_manager, 'execute_position_action'):
                    self.position_manager.execute_position_action(position_action)
                    logger.debug(f"Position action executed")
                else:
                    logger.warning("Position manager missing or doesn't have execute_position_action method")
                
            except Exception as e:
                logger.error(f"Error processing position action: {str(e)}", exc_info=True)
        
        return handle_position_action_event


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

            # Track event counts for debugging
            event_counts = {
                'bar': 0,
                'signal': 0,
                'order': 0,
                'fill': 0,
                'position_action': 0
            }

            # Process each bar
            for bar_data in iterator:
                # Ensure we have a proper BarEvent
                if not isinstance(bar_data, BarEvent):
                    # Convert to BarEvent if it's a dictionary
                    if isinstance(bar_data, dict) and 'Close' in bar_data:
                        logger.debug(f"Converting dictionary to BarEvent")
                        bar_data = BarEvent(bar_data)
                    else:
                        logger.error(f"Could not process data: {type(bar_data)}")
                        continue

                # Create and emit BAR event
                bar_event_container = Event(EventType.BAR, bar_data)
                logger.debug(f"EMITTING BAR: {bar_data.get_symbol()} @ {bar_data.get_timestamp()}")
                self.event_bus.emit(bar_event_container)
                event_counts['bar'] += 1

                # If strategy has direct on_bar method, call it with the event
                if hasattr(self.strategy, 'on_bar'):
                    signal_event = self.strategy.on_bar(bar_event_container)

                    # If the strategy returned a signal directly, emit it
                    if signal_event is not None:
                        # Process single signal
                        if isinstance(signal_event, SignalEvent):
                            logger.debug(f"Emitting signal from strategy: {signal_event}")
                            self.event_bus.emit(Event(EventType.SIGNAL, signal_event))
                            event_counts['signal'] += 1
                        # Process multiple signals if returned as a list
                        elif isinstance(signal_event, list):
                            for signal in signal_event:
                                if isinstance(signal, SignalEvent) and signal is not None:
                                    logger.debug(f"Emitting signal from strategy list: {signal}")
                                    self.event_bus.emit(Event(EventType.SIGNAL, signal))
                                    event_counts['signal'] += 1

                # Update portfolio with current bar data
                self.execution_engine.update(bar_data)

                # Execute any pending orders
                fills = self.execution_engine.execute_pending_orders(bar_data, self.market_simulator)
                if fills:
                    event_counts['fill'] += len(fills)
                    logger.debug(f"Executed {len(fills)} fills")

                # Log progress every 100 bars
                if event_counts['bar'] % 100 == 0:
                    logger.info(f"Processed {event_counts['bar']} bars, {event_counts['signal']} signals, "
                               f"{event_counts['order']} orders, {event_counts['fill']} fills")

            logger.info("Backtest completed. Collecting results...")

            # Log final event counts
            logger.info(f"Final counts: {event_counts['bar']} bars, {event_counts['signal']} signals, "
                       f"{event_counts['order']} orders, {event_counts['fill']} fills, "
                       f"{event_counts['position_action']} position actions")

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
        # If trade_history is None or empty, return empty list
        if not trade_history:
            return []
        
        # Process each trade
        processed_trades = []
        
        for trade in trade_history:
            # Create processed trade with standard fields
            processed_trade = {
                'trade_id': trade.get('trade_id', trade.get('position_id', '')),
                'symbol': trade.get('symbol', ''),
                'direction': trade.get('direction', 0),
                'quantity': trade.get('quantity', 0),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'entry_time': trade.get('entry_time', None),
                'exit_time': trade.get('exit_time', None),
                'realized_pnl': trade.get('realized_pnl', trade.get('pnl', 0)),
                'return_pct': 0,  # Will calculate below
                'holding_period': 0  # Will calculate below
            }
            
            # Calculate return percentage
            if processed_trade['entry_price'] > 0:
                price_change = processed_trade['exit_price'] - processed_trade['entry_price']
                if processed_trade['direction'] < 0:  # Short trade
                    price_change = -price_change
                processed_trade['return_pct'] = (price_change / processed_trade['entry_price']) * 100
            
            # Calculate holding period in days if timestamps available
            if processed_trade['entry_time'] and processed_trade['exit_time']:
                duration = processed_trade['exit_time'] - processed_trade['entry_time']
                processed_trade['holding_period'] = duration.total_seconds() / 86400  # Convert to days
                
            processed_trades.append(processed_trade)
            
        return processed_trades
        
    def _calculate_performance_metrics(self, processed_trades):
        """Calculate performance metrics from processed trades."""
        # If no trades, return zeros
        if not processed_trades:
            return 0, 0, 0
            
        # Calculate total return
        total_return = sum(trade['return_pct'] for trade in processed_trades)
        
        # Calculate log return
        total_log_return = sum(np.log(1 + trade['return_pct']/100) for trade in processed_trades)
        
        # Calculate average return
        avg_return = total_return / len(processed_trades) if processed_trades else 0
        
        return total_return, total_log_return, avg_return
        
    def calculate_sharpe(self, risk_free_rate=0.0, annualization_factor=252):
        """Calculate Sharpe ratio for the backtest results."""
        # If execution engine doesn't have portfolio_history or it's empty, return 0
        if not hasattr(self.execution_engine, 'portfolio_history') or not self.execution_engine.portfolio_history:
            return 0
            
        # Extract equity values from portfolio history
        equity_values = []
        for snapshot in self.execution_engine.portfolio_history:
            if 'equity' in snapshot:
                equity_values.append(snapshot['equity'])
                
        # If we have fewer than 2 values, return 0
        if len(equity_values) < 2:
            return 0
            
        # Calculate daily returns
        returns = []
        for i in range(1, len(equity_values)):
            returns.append((equity_values[i] / equity_values[i-1]) - 1)
            
        # Calculate Sharpe ratio
        returns = np.array(returns)
        excess_returns = returns - (risk_free_rate / annualization_factor)
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)
        
        return sharpe
        
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown for the backtest."""
        # If execution engine doesn't have portfolio_history or it's empty, return 0
        if not hasattr(self.execution_engine, 'portfolio_history') or not self.execution_engine.portfolio_history:
            return 0
            
        # Extract equity values from portfolio history
        equity_values = []
        for snapshot in self.execution_engine.portfolio_history:
            if 'equity' in snapshot:
                equity_values.append(snapshot['equity'])
                
        # If we have fewer than 2 values, return 0
        if len(equity_values) < 2:
            return 0
            
        # Calculate drawdown
        running_max = equity_values[0]
        drawdowns = []
        
        for equity in equity_values:
            if equity > running_max:
                running_max = equity
            drawdown = (running_max - equity) / running_max if running_max > 0 else 0
            drawdowns.append(drawdown)
            
        # Return maximum drawdown
        return max(drawdowns) if drawdowns else 0
