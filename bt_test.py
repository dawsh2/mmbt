#!/usr/bin/env python3
"""
Fixed Backtester Integration Test

This script runs a backtest ensuring proper component initialization order
and connection.
"""

import datetime
import logging
import argparse
import sys
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG for detailed event flow logs
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from your codebase
from src.events.event_bus import EventBus
from src.events.event_types import EventType
from src.events.event_base import Event
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.position_management.position_sizers import FixedSizeSizer
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine
from src.engine.backtester import Backtester
from src.engine.market_simulator import MarketSimulator

# First, patch the EventBus class to add event_counts
original_emit = EventBus.emit

def patched_emit(self, event):
    """Patched emit method to track event counts"""
    # Initialize event_counts if it doesn't exist
    if not hasattr(self, 'event_counts'):
        self.event_counts = {}
    
    # Count the event
    if event.event_type in self.event_counts:
        self.event_counts[event.event_type] += 1
    else:
        self.event_counts[event.event_type] = 1
    
    # Call the original emit method
    return original_emit(self, event)

# Apply the patch
EventBus.emit = patched_emit

def run_backtest(config, start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """
    Run a backtest with proper component initialization order
    """
    # Set default values if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # 1. Create event system
    event_bus = EventBus()
    
    # CRITICAL: Initialize event_counts attribute
    event_bus.event_counts = {}
    logger.info("Created event bus with event counting")
    
    # 2. Set up data handler
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source=data_source)
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # 3. Create portfolio first (needed by both position manager and execution engine)
    initial_capital = config.get('initial_capital', 100000)
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    logger.info(f"Created portfolio with initial capital: ${initial_capital}")
    
    # 4. Create position manager and ensure it has the event bus reference
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        event_bus=event_bus  # CRITICAL: Pass event bus explicitly
    )
    logger.info("Created position manager")
    
    # 5. Create market simulator
    market_sim_config = {
        'slippage_model': 'fixed',
        'slippage_bps': 1,  # 0.01% slippage
        'fee_model': 'fixed',
        'fee_bps': 1        # 0.01% commission
    }
    market_simulator = MarketSimulator(market_sim_config)
    logger.info("Created market simulator")
    
    # 6. Create strategy with event bus reference
    strategy = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy",
        event_bus=event_bus  # CRITICAL: Pass event bus explicitly
    )
    logger.info("Created strategy")

    # 7. Create backtester configuration
    backtester_config = {
        'backtester': {
            'initial_capital': initial_capital,
            'market_simulation': market_sim_config
        }
    }


    # 8. Create and configure execution engine
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.portfolio = portfolio  # CRITICAL: Set portfolio reference
    execution_engine.event_bus = event_bus
    execution_engine.market_simulator = market_simulator
    logger.info("Created and configured execution engine")

    # 9. Create backtester with all components
    backtester = Backtester(
        config=backtester_config,
        data_handler=data_handler,
        strategy=strategy,
        position_manager=position_manager
    )

    # 10. Provide the configured execution engine to the backtester
    backtester.execution_engine = execution_engine  # Override the backtester's execution engine


    # 11. CRITICAL: Register event handlers directly for guaranteed setup
    # Register position manager to receive SIGNAL events
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)
    logger.info("Registered position manager for SIGNAL events")
    
    # Create position action handler
    def handle_position_action(event):
        if event.event_type == EventType.POSITION_ACTION and hasattr(event, 'data'):
            action = event.data
            logger.info(f"Handling position action: {action.get('action_type', 'unknown') if isinstance(action, dict) else 'unknown'}")
            # Forward to execution engine
            if hasattr(execution_engine, 'on_position_action'):
                execution_engine.on_position_action(event)
            else:
                # Fallback to position manager
                position_manager.execute_position_action(action)
    
    # Register position action handler
    event_bus.register(EventType.POSITION_ACTION, handle_position_action)
    logger.info("Registered handler for POSITION_ACTION events")
    
    # Register execution engine for ORDER events
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    logger.info("Registered execution engine for ORDER events")
    
    # 12. Add debugging event tracing
    def trace_event(event):
        event_type = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
        logger.debug(f"Event trace: {event_type} received")
        
        # For important events, log more details
        if event.event_type == EventType.SIGNAL:
            signal = event.data
            if hasattr(signal, 'get_signal_value') and hasattr(signal, 'get_symbol'):
                direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
                logger.debug(f"Signal details: {direction} for {signal.get_symbol()} @ {signal.get_price()}")
        
        elif event.event_type == EventType.POSITION_ACTION:
            action = event.data
            if isinstance(action, dict):
                logger.debug(f"Position action details: {action.get('action_type', 'unknown')} for {action.get('symbol', 'unknown')}")
        
        elif event.event_type == EventType.ORDER:
            order = event.data
            if hasattr(order, 'get_symbol') and hasattr(order, 'get_direction'):
                direction = "BUY" if order.get_direction() > 0 else "SELL"
                logger.debug(f"Order details: {direction} {order.get_quantity()} {order.get_symbol()} @ {order.get_price()}")
    
    # Register tracer for all events
    for event_type in EventType:
        event_bus.register(event_type, trace_event)
    
    # 13. Run the backtest
    logger.info("Starting backtest")
    results = backtester.run(use_test_data=False)
    
    # 14. Count events by type and display
    event_counts = {}
    try:
        # Use event_counts from event_bus if available
        if hasattr(event_bus, 'event_counts'):
            event_counts = event_bus.event_counts
        # Try different ways to access event counts as fallback
        elif hasattr(event_bus, 'metrics') and 'events_processed' in event_bus.metrics:
            event_counts = event_bus.metrics['events_processed']
        elif hasattr(event_bus, 'history'):
            # Count manually from history
            for event in event_bus.history:
                event_type = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
                if event_type not in event_counts:
                    event_counts[event_type] = 0
                event_counts[event_type] += 1
    except Exception as e:
        logger.error(f"Error counting events: {e}")
    
    # 15. Display results
    logger.info("Backtest completed")
    logger.info(f"Initial capital: ${initial_capital:.2f}")
    
    # Access results from the backtester
    trades = results.get('trades', [])
    
    # Display final portfolio state
    logger.info(f"Final equity: ${portfolio.equity:.2f}")
    logger.info(f"Final cash: ${portfolio.cash:.2f}")
    logger.info(f"Return: {(portfolio.equity / initial_capital - 1) * 100:.2f}%")
    
    # Display trade statistics
    logger.info(f"Total trades: {len(trades)}")
    
    if trades:
        # Calculate trade statistics
        winning_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_profit = sum(t.get('realized_pnl', 0) for t in trades) / len(trades) if trades else 0
        
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(f"Average profit: ${avg_profit:.2f}")
        
        # Show trade history
        logger.info("Trade history:")
        for i, trade in enumerate(trades[:10]):  # Show first 10 trades
            direction = "LONG" if trade.get('direction', 0) > 0 else "SHORT"
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('realized_pnl', 0)
            symbol = trade.get('symbol', 'Unknown')
            logger.info(f"  {i+1}. {direction} {symbol} Entry: {entry_price} Exit: {exit_price} PnL: {pnl:.2f}")
    
    # Show portfolio positions at end
    portfolio_snapshot = portfolio.get_position_snapshot()
    num_open_positions = sum(len(positions) for positions in portfolio_snapshot.values())
    logger.info(f"Open positions at end: {num_open_positions}")
    
    for symbol, positions in portfolio_snapshot.items():
        for pos in positions:
            direction = "LONG" if pos.get('direction', 0) > 0 else "SHORT"
            quantity = pos.get('quantity', 0)
            entry_price = pos.get('entry_price', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            logger.info(f"  {direction} {symbol} {quantity} @ {entry_price} (PnL: ${unrealized_pnl:.2f})")
    
    # Show event statistics
    logger.info("Event statistics:")
    for event_type, count in event_counts.items():
        event_type_name = event_type.name if hasattr(event_type, 'name') else str(event_type)
        logger.info(f"  {event_type_name}: {count} events")
    
    return {
        'initial_equity': initial_capital,
        'final_equity': portfolio.equity,
        'return_pct': (portfolio.equity / initial_capital - 1) * 100,
        'trade_count': len(trades),
        'portfolio': portfolio,
        'event_counts': event_counts
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
    results = run_backtest(
        config=config,
        start_date=start_date,
        end_date=end_date,
        symbols=[args.symbol],
        timeframe=args.timeframe
    )
    
    # Print summary
    logger.info("Backtest Summary:")
    logger.info(f"Initial Equity: ${results['initial_equity']:.2f}")
    logger.info(f"Final Equity: ${results['final_equity']:.2f}")
    logger.info(f"Return: {results['return_pct']:.2f}%")
    logger.info(f"Trades Executed: {results['trade_count']}")
    
    # Print event statistics 
    logger.info("Event Statistics:")
    for event_type, count in results.get('event_counts', {}).items():
        event_type_name = event_type.name if hasattr(event_type, 'name') else str(event_type)
        logger.info(f"  {event_type_name}: {count}")
