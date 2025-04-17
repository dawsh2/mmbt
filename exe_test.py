#!/usr/bin/env python3
"""
Complete Integration Script - Fixed Version

This script connects all components with debugging focused on the signal -> order -> fill flow.
"""

import datetime
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the codebase
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.events.portfolio_events import PositionActionEvent

from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.position_management.position_sizers import AdjustedFixedSizer
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine

def display_results(results):
    """Display backtest results."""
    logger.info("=========== BACKTEST RESULTS ===========")
    logger.info(f"Initial equity: ${results['initial_equity']:,.2f}")
    logger.info(f"Final equity: ${results['final_equity']:,.2f}")
    logger.info(f"Return: {results['return_pct']:.2f}%")
    logger.info(f"Bars processed: {results['bars_processed']}")
    logger.info(f"Signals generated: {results['signals_generated']}")
    logger.info(f"Orders generated: {results['orders_generated']}")
    logger.info(f"Fills processed: {results['fills_processed']}")
    logger.info(f"Positions opened: {results['positions_opened']}")
    logger.info(f"Positions closed: {results['positions_closed']}")
    logger.info("=======================================")

def run_backtest(config, start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """
    Run a complete backtest with enhanced debugging for the signal -> order -> fill flow.
    """
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # Create event bus
    event_bus = EventBus()
    logger.info("Created EventBus")
    
    # Create result trackers
    trackers = {
        'signals_generated': 0,
        'orders_generated': 0,
        'fills_processed': 0,
        'positions_opened': 0,
        'positions_closed': 0,
        'position_actions': 0
    }
    
    # Create portfolio with initial capital
    initial_capital = config.get('initial_capital', 100000)
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    logger.info(f"Created portfolio with {initial_capital} capital")
    
    # Create position sizer and position manager
    position_sizer = AdjustedFixedSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
    )    
    # CRITICAL: Set event bus on position manager
    position_manager.event_bus = event_bus
    logger.info("Created position manager with fixed sizer")
    
    # Create rule
    rule = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy",
        event_bus=event_bus
    )
    logger.info(f"Created {rule}")
    
    # Create execution engine
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.event_bus = event_bus  # Set event bus on execution engine
    logger.info("Created execution engine")
    
    # --------- DEBUG HANDLERS ---------
    
    # Signal monitoring handler
    def signal_monitor(event):
        signal = event.data if isinstance(event.data, SignalEvent) else None
        
        if signal:
            trackers['signals_generated'] += 1
            signal_value = signal.get_signal_value()
            symbol = signal.get_symbol()
            price = signal.get_price()
            
            signal_type = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
            logger.info(f"SIGNAL MONITOR: {signal_type} signal for {symbol} at price {price}")
            
            # Debug call to position manager
            logger.info("SIGNAL MONITOR: Directly calling position_manager.on_signal with event")
            actions = position_manager.on_signal(event)
            
            if actions:
                logger.info(f"SIGNAL MONITOR: Position manager returned {len(actions)} actions")
                for action in actions:
                    logger.info(f"SIGNAL MONITOR: Action: {action}")
            else:
                logger.info("SIGNAL MONITOR: Position manager returned no actions")
    
    # Position action monitoring handler
    def position_action_monitor(event):
        trackers['position_actions'] += 1
        
        # Log action details
        if hasattr(event.data, 'get_action_type'):
            action_type = event.data.get_action_type()
            logger.info(f"POSITION ACTION: {action_type}")
        elif isinstance(event.data, dict):
            action_type = event.data.get('action_type', 'unknown')
            symbol = event.data.get('symbol', 'unknown')
            direction = event.data.get('direction', 0)
            price = event.data.get('price', 0)
            logger.info(f"POSITION ACTION: {action_type} for {symbol} direction={direction} price={price}")
        
        # Forward to portfolio for processing
        if hasattr(portfolio, '_handle_position_action'):
            logger.info("POSITION ACTION: Forwarding to portfolio._handle_position_action")
            portfolio._handle_position_action(event)
        else:
            logger.error("POSITION ACTION: Portfolio doesn't have _handle_position_action method")
    
    # Order monitoring handler
    def order_monitor(event):
        trackers['orders_generated'] += 1
        logger.info(f"ORDER MONITOR: Order generated for {event.data.get('symbol', 'unknown')}")
        
        # Forward to execution engine
        if hasattr(execution_engine, 'on_order'):
            logger.info("ORDER MONITOR: Forwarding to execution_engine.on_order")
            execution_engine.on_order(event)
    
    # Fill monitoring handler
    def fill_monitor(event):
        trackers['fills_processed'] += 1
        logger.info(f"FILL MONITOR: Fill processed for {event.data.get('symbol', 'unknown')}")
        
        # Forward to portfolio
        if hasattr(portfolio, '_handle_fill'):
            logger.info("FILL MONITOR: Forwarding to portfolio._handle_fill")
            portfolio._handle_fill(event)
    
    # Position opened/closed monitoring
    def position_opened_monitor(event):
        trackers['positions_opened'] += 1
        logger.info(f"POSITION OPENED: {event.data.get('symbol', 'unknown')}")
    
    def position_closed_monitor(event):
        trackers['positions_closed'] += 1
        logger.info(f"POSITION CLOSED: {event.data.get('symbol', 'unknown')}")
    
    # Register debugging handlers
    event_bus.register(EventType.SIGNAL, signal_monitor)
    event_bus.register(EventType.POSITION_ACTION, position_action_monitor)
    event_bus.register(EventType.ORDER, order_monitor)
    event_bus.register(EventType.FILL, fill_monitor)
    event_bus.register(EventType.POSITION_OPENED, position_opened_monitor)
    event_bus.register(EventType.POSITION_CLOSED, position_closed_monitor)
    
    # Register rule with BAR events
    event_bus.register(EventType.BAR, rule.on_bar)
    logger.info("Registered handlers with event bus")
    
    # Create data handler and load data
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source=data_source)
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Create direct method check for position_manager
    logger.info("----- CHECKING POSITION MANAGER METHODS -----")
    # Check on_signal method
    if hasattr(position_manager, 'on_signal'):
        logger.info("position_manager.on_signal method exists")
        
        # Check method signature and create a test call
        dummy_signal = SignalEvent(
            signal_value=1,  # BUY
            price=100.0,
            symbol="TEST",
        )
        dummy_event = Event(EventType.SIGNAL, dummy_signal)
        
        try:
            logger.info("Testing position_manager.on_signal with dummy signal")
            result = position_manager.on_signal(dummy_event)
            
            if result:
                logger.info(f"on_signal returned {len(result)} action(s)")
            else:
                logger.info("on_signal returned no actions")
                
            # Check _process_signal method
            if hasattr(position_manager, '_process_signal'):
                logger.info("position_manager._process_signal method exists")
                
                try:
                    direct_result = position_manager._process_signal(dummy_signal)
                    if direct_result:
                        logger.info(f"_process_signal returned {len(direct_result)} action(s)")
                    else:
                        logger.info("_process_signal returned no actions")
                except Exception as e:
                    logger.error(f"Error testing _process_signal: {e}")
            else:
                logger.error("position_manager._process_signal method does not exist")
                
        except Exception as e:
            logger.error(f"Error testing on_signal: {e}")
    else:
        logger.error("position_manager.on_signal method does not exist")
    
    logger.info("----- CHECKING PORTFOLIO METHODS -----")
    # Check portfolio methods
    if hasattr(portfolio, '_handle_position_action'):
        logger.info("portfolio._handle_position_action method exists")
    else:
        logger.error("portfolio._handle_position_action method does not exist")
        
    if hasattr(portfolio, '_open_position'):
        logger.info("portfolio._open_position method exists")
    else:
        logger.error("portfolio._open_position method does not exist")
    
    # PATCH position manager directly if needed
    if hasattr(position_manager, "on_signal") and not position_manager.on_signal(dummy_event):
        logger.warning("Position manager on_signal method returned no actions - applying patch")
        
        # Create direct patch for position_manager.on_signal
        def patched_on_signal(self, event):
            """Patched method to correctly handle signal events."""
            # Extract signal from event
            if not isinstance(event.data, SignalEvent):
                logger.error(f"Expected SignalEvent in event.data, got {type(event.data)}")
                return []

            signal = event.data
            
            # Store signal for tracking
            self.signal_history.append(signal)
            
            # Process signal and get position actions
            actions = self._process_signal(signal)
            
            # Emit position actions
            if self.event_bus:
                for action in actions:
                    position_action_event = Event(EventType.POSITION_ACTION, action)
                    self.event_bus.emit(position_action_event)
                    logger.info(f"Patched on_signal emitted position action: {action.get('action_type', 'unknown')}")
            
            return actions
        
        # Apply patch
        import types
        position_manager.on_signal = types.MethodType(patched_on_signal, position_manager)
        logger.info("Applied patch to position_manager.on_signal")
        
        # Test the patch
        test_result = position_manager.on_signal(dummy_event)
        if test_result:
            logger.info(f"Patched on_signal returned {len(test_result)} action(s)")
        else:
            logger.info("Patched on_signal returned no actions")
    
    # Run backtest
    logger.info("Starting backtest loop")
    bars_processed = 0
    
    for bar in data_handler.iter_train():
        # Track progress
        bars_processed += 1
        
        # Create and emit bar event
        if not isinstance(bar, BarEvent):
            bar_event = BarEvent(bar)
        else:
            bar_event = bar
            
        # Emit bar event to trigger the chain of events
        event_bus.emit(Event(EventType.BAR, bar_event))
        
        # Execute pending orders
        fills = execution_engine.execute_pending_orders(bar_event)
        if fills:
            logger.info(f"Execute pending orders returned {len(fills)} fills")
            for fill in fills:
                fill_event = Event(EventType.FILL, fill)
                event_bus.emit(fill_event)
        
        # Log progress every 50 bars
        if bars_processed % 50 == 0:
            logger.info(f"Processed {bars_processed} bars")
            logger.info(f"Portfolio equity: {portfolio.equity:.2f}")
            logger.info(f"Signals: {trackers['signals_generated']}, Position Actions: {trackers['position_actions']}")
            logger.info(f"Orders: {trackers['orders_generated']}, Fills: {trackers['fills_processed']}")
            
            # Log open positions
            if hasattr(portfolio, 'positions'):
                logger.info(f"Open positions: {len(portfolio.positions)}")
                for pos_id, pos in portfolio.positions.items():
                    if hasattr(pos, 'symbol') and hasattr(pos, 'quantity'):
                        logger.info(f"  Position {pos_id}: {pos.symbol} quantity={pos.quantity}")
    
    # Collect results
    results = {
        'initial_equity': initial_capital,
        'final_equity': portfolio.equity,
        'return_pct': (portfolio.equity / initial_capital - 1) * 100,
        'bars_processed': bars_processed,
        'signals_generated': trackers['signals_generated'],
        'orders_generated': trackers['orders_generated'],
        'fills_processed': trackers['fills_processed'],
        'positions_opened': trackers['positions_opened'], 
        'positions_closed': trackers['positions_closed'],
        'position_actions': trackers['position_actions']
    }
    
    # Display final results
    display_results(results)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run algorithmic trading backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to trade')
    parser.add_argument('--timeframe', type=str, default='1m', help='Data timeframe')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = datetime.datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.datetime.strptime(args.end, '%Y-%m-%d') if args.end else None
    
    # Create config
    config = {
        'initial_capital': args.capital
    }
    
    # Run backtest
    run_backtest(
        config=config,
        start_date=start_date,
        end_date=end_date,
        symbols=[args.symbol],
        timeframe=args.timeframe
    )
