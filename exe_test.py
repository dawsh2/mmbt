#!/usr/bin/env python3
"""
Clean Minimal Integration Test

This script runs a clean backtest after the code fixes are applied.
"""

import datetime
import logging
import argparse
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from your codebase
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.position_management.position_sizers import FixedSizeSizer
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine

def run_backtest(config, start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """
    Run a clean backtest with the fixed components
    """
    # Set default values if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # Create components
    event_bus = EventBus()
    logger.info("Created event bus")
    
    initial_capital = config.get('initial_capital', 100000)
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    logger.info(f"Created portfolio with ${initial_capital}")
    
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer
    )
    position_manager.event_bus = event_bus
    logger.info("Created position manager")
    
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.event_bus = event_bus
    logger.info("Created execution engine")
    
    rule = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy"
    )
    rule.set_event_bus(event_bus)
    logger.info("Created SMA crossover rule")
    
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source=data_source)
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )


    # Only register handlers once!
    event_bus.register(EventType.BAR, rule.on_bar)
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)  # Position manager creates actions

    # Signal logging only (no position creation)
    def log_signal(event):
        signal = event.data
        if isinstance(signal, SignalEvent):
            direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
            logger.info(f"Signal: {direction} for {signal.get_symbol()} at {signal.get_price()}")
        else:
            logger.warning(f"Non-SignalEvent in signal data: {type(signal)}")

    event_bus.register(EventType.SIGNAL, log_signal)  # Just for logging

    # REMOVE THIS DUPLICATE REGISTRATION
    # event_bus.register(EventType.SIGNAL, log_signal)  # THIS IS DUPLICATED IN YOUR CODE!
 
 
    # Process position actions when emitted
    def process_position_action(event):
        if event.event_type == EventType.POSITION_ACTION:
            action = event.data
            logger.info(f"Position action: {action.get('action_type', 'unknown')} for {action.get('symbol', 'unknown')}")
            
            # Execute the action
            result = position_manager.execute_position_action(
                action=action,
                current_time=action.get('timestamp', datetime.datetime.now())
            )
            
            if result and result.get('success', False):
                logger.info(f"Position action successful: {result}")



    # Register position action processor
    event_bus.register(EventType.POSITION_ACTION, process_position_action)
    
    # Register monitoring handlers for logging
    def log_signal(event):
        signal = event.data
        if isinstance(signal, SignalEvent):
            direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
            logger.info(f"Signal: {direction} for {signal.get_symbol()} at {signal.get_price()}")
        else:
            logger.warning(f"Non-SignalEvent in signal data: {type(signal)}")
    
    event_bus.register(EventType.SIGNAL, log_signal)
    
    # Run the backtest loop
    logger.info("Starting backtest loop")
    bars_processed = 0
    
    for bar in data_handler.iter_train():
        # Process each bar
        bars_processed += 1
        
        # Create bar event if needed
        if not isinstance(bar, BarEvent):
            bar_event = BarEvent(bar)
        else:
            bar_event = bar
            
        # Emit bar event to trigger the event chain
        event_bus.emit(Event(EventType.BAR, bar_event))
        
        # Execute any pending orders
        if hasattr(execution_engine, 'execute_pending_orders'):
            fills = execution_engine.execute_pending_orders(bar_event)
            if fills:
                for fill in fills:
                    event_bus.emit(Event(EventType.FILL, fill))
        
        # Update portfolio with latest prices
        portfolio.mark_to_market(bar_event)
        
        # Log progress periodically
        if bars_processed % 50 == 0:
            logger.info(f"Processed {bars_processed} bars")
            logger.info(f"Portfolio equity: ${portfolio.equity:.2f}")
            
            # Log open positions
            if hasattr(portfolio, 'positions'):
                logger.info(f"Open positions: {len(portfolio.positions)}")
                for pos_id, pos in list(portfolio.positions.items())[:5]:  # Show first 5
                    direction = "LONG" if getattr(pos, 'direction', 0) > 0 else "SHORT"
                    logger.info(f"  {direction} {pos.symbol} {pos.quantity} @ {pos.entry_price}")
    
    # Calculate results
    logger.info(f"Backtest completed - processed {bars_processed} bars")
    logger.info(f"Initial capital: ${initial_capital:.2f}")
    logger.info(f"Final equity: ${portfolio.equity:.2f}")
    logger.info(f"Return: {(portfolio.equity / initial_capital - 1) * 100:.2f}%")
    
    # Show open positions at the end
    if hasattr(portfolio, 'positions'):
        logger.info(f"Open positions at end: {len(portfolio.positions)}")
        for pos_id, pos in portfolio.positions.items():
            direction = "LONG" if getattr(pos, 'direction', 0) > 0 else "SHORT"
            logger.info(f"  {direction} {pos.symbol} {pos.quantity} @ {pos.entry_price}")
    
    # Show closed positions
    if hasattr(portfolio, 'closed_positions'):
        logger.info(f"Closed positions: {len(portfolio.closed_positions)}")
        
        # Print trade history details
        if portfolio.closed_positions:
            logger.info("Trade history:")
            for pos_id, pos in list(portfolio.closed_positions.items())[:10]:  # Show first 10
                entry_price = getattr(pos, 'entry_price', 0)
                exit_price = getattr(pos, 'exit_price', 0) 
                realized_pnl = getattr(pos, 'realized_pnl', 0)
                direction = "LONG" if getattr(pos, 'direction', 0) > 0 else "SHORT"
                logger.info(f"  {direction} {pos.symbol} Entry: {entry_price} Exit: {exit_price} PnL: {realized_pnl:.2f}")
    
    return {
        'initial_equity': initial_capital,
        'final_equity': portfolio.equity,
        'return_pct': (portfolio.equity / initial_capital - 1) * 100,
        'bars_processed': bars_processed
    }

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
