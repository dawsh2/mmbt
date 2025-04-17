#!/usr/bin/env python3
"""
Minimal Integration Script

This script connects all components and focuses on proper event flow,
using your existing code structure without patches or workarounds.
"""

import datetime
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from your codebase
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.events.event_handlers import SignalHandler, MarketDataHandler, FillHandler
from src.events.portfolio_events import PositionActionEvent

from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.position_management.position_sizers import FixedSizeSizer
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine

def run_backtest(config, start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """
    Run a backtest using the proper event-driven architecture
    """
    # Set default values if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # 1. Create event bus - central communication hub
    event_bus = EventBus()
    logger.info("Created event bus")
    
    # 2. Create portfolio
    initial_capital = config.get('initial_capital', 100000)
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    logger.info(f"Created portfolio with ${initial_capital}")
    
    # 3. Create position manager with position sizer
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        event_bus=event_bus
    )
    logger.info("Created position manager")
    
    # 4. Create execution engine
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.event_bus = event_bus
    logger.info("Created execution engine")
    
    # 5. Create trading rule
    rule = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy"
    )
    logger.info("Created SMA crossover rule")
    
    # 6. Create data handler and load data
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source=data_source)
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # 7. Set up event handlers - THIS IS KEY
    # These handlers follow your existing event handler structure
    
    # Handler for market data -> rules
    market_data_handler = MarketDataHandler(rule)
    event_bus.register(EventType.BAR, market_data_handler)
    
    # Handler for signals -> position manager
    signal_handler = SignalHandler(position_manager)
    event_bus.register(EventType.SIGNAL, signal_handler)
    
    # Handler for fills -> portfolio
    fill_handler = FillHandler(position_manager)
    event_bus.register(EventType.FILL, fill_handler)
    
    # Basic monitoring handlers
    def log_event(event):
        """Simple logging handler for all events"""
        event_type = event.event_type.name
        if event_type == "BAR":
            # Too verbose to log every bar
            return
            
        if event_type == "SIGNAL":
            signal = event.data
            direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
            logger.info(f"SIGNAL: {direction} for {signal.get_symbol()} at {signal.get_price()}")
        
        elif event_type == "POSITION_ACTION":
            action = event.data
            action_type = action.get('action_type', 'unknown')
            symbol = action.get('symbol', 'unknown')
            logger.info(f"POSITION ACTION: {action_type} for {symbol}")
        
        elif event_type == "POSITION_OPENED":
            logger.info(f"POSITION OPENED: {event.data.get('symbol', 'unknown')}")
        
        elif event_type == "POSITION_CLOSED":
            logger.info(f"POSITION CLOSED: {event.data.get('symbol', 'unknown')}")
            
        elif event_type == "FILL":
            logger.info(f"FILL: {event.data.get('symbol', 'unknown')}")
    
    # Register the logger for important events
    for event_type in [EventType.SIGNAL, EventType.POSITION_ACTION, 
                       EventType.POSITION_OPENED, EventType.POSITION_CLOSED, 
                       EventType.FILL]:
        event_bus.register(event_type, log_event)
    
    # 8. Run the backtest loop
    logger.info("Starting backtest loop")
    bars_processed = 0
    position_actions = 0
    positions_opened = 0
    positions_closed = 0
    
    for bar in data_handler.iter_train():
        # Process each bar
        bars_processed += 1
        
        # Create bar event
        if not isinstance(bar, BarEvent):
            bar_event = BarEvent(bar)
        else:
            bar_event = bar
            
        # Emit bar event to trigger the event chain
        event_bus.emit(Event(EventType.BAR, bar_event))
        
        # Execute pending orders with the latest price data
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
    
    # 9. Calculate results
    logger.info("Backtest completed")
    logger.info(f"Processed {bars_processed} bars")
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
