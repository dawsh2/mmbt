#!/usr/bin/env python3
"""
Enhanced Backtester Implementation

This script fixes the event flow issues in the algorithmic trading system
to ensure signals are properly converted to trades.
"""

import datetime
import logging
import argparse
import sys
import os

# Configure logging with more detail for debugging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for more verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_debug.log"),
        logging.StreamHandler()
    ]
)
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

# Add debugging event listener to trace all events
class EventTracer:
    """Utility class to trace all events flowing through the event bus."""
    
    def __init__(self, log_level=logging.DEBUG):
        self.logger = logging.getLogger(__name__ + ".EventTracer")
        self.logger.setLevel(log_level)
        self.event_counts = {}
    
    def trace_event(self, event):
        """Log event details for tracing the event flow."""
        event_type = event.event_type.name if hasattr(event, 'event_type') else 'Unknown'
        
        # Track counts
        if event_type not in self.event_counts:
            self.event_counts[event_type] = 0
        self.event_counts[event_type] += 1
        
        # Log event details
        self.logger.debug(f"EVENT: {event_type} #{self.event_counts[event_type]} - "
                         f"Time: {event.timestamp if hasattr(event, 'timestamp') else 'N/A'}")
        
        # Log more details for important events
        if event_type == 'SIGNAL':
            signal = event.data
            if hasattr(signal, 'get_signal_value') and hasattr(signal, 'get_symbol'):
                direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
                self.logger.debug(f"SIGNAL DETAILS: {direction} for {signal.get_symbol()} "
                                f"at price {signal.get_price()}")
        
        elif event_type == 'POSITION_ACTION':
            action = event.data
            if isinstance(action, dict):
                action_type = action.get('action_type', 'unknown')
                symbol = action.get('symbol', 'unknown')
                self.logger.debug(f"POSITION ACTION DETAILS: {action_type} for {symbol}")
                
        elif event_type == 'ORDER':
            order = event.data
            if hasattr(order, 'get_symbol') and hasattr(order, 'get_direction'):
                self.logger.debug(f"ORDER DETAILS: {order.get_symbol()} "
                                f"{'BUY' if order.get_direction() > 0 else 'SELL'} "
                                f"{order.get_quantity()} @ {order.get_price()}")
        
        elif event_type == 'FILL':
            fill = event.data
            if hasattr(fill, 'get_symbol'):
                self.logger.debug(f"FILL DETAILS: {fill.get_symbol()} "
                                f"{'BUY' if fill.get_direction() > 0 else 'SELL'} "
                                f"{fill.get_quantity()} @ {fill.get_price()}")

def run_enhanced_backtest(config, start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """
    Run an enhanced backtest with fixed event flow
    """
    # Set default values if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # 1. Create event system with tracing
    event_bus = EventBus()
    event_tracer = EventTracer()
    logger.info("Created event bus with tracing")
    
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
    
    # 4. Create position manager with explicit event bus reference
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        event_bus=event_bus  # Explicit event bus reference
    )
    logger.info("Created position manager")
    
    # 5. Create execution engine and explicitly set the portfolio
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.portfolio = portfolio  # CRITICAL: Make sure to set the portfolio
    execution_engine.event_bus = event_bus  # Explicit event bus reference
    logger.info("Created execution engine")
    
    # 6. Create strategy with explicit event bus
    strategy = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy",
        event_bus=event_bus  # Explicit event bus reference
    )
    logger.info("Created strategy")
    
    # 7. Create market simulator
    market_sim_config = {
        'slippage_model': 'fixed',
        'slippage_bps': 1,  # 0.01% slippage
        'fee_model': 'fixed',
        'fee_bps': 1        # 0.01% commission
    }
    market_simulator = MarketSimulator(market_sim_config)
    logger.info("Created market simulator")
    
    # 8. Create backtester configuration
    backtester_config = {
        'backtester': {
            'initial_capital': initial_capital,
            'market_simulation': market_sim_config
        }
    }
    
    # 9. Create and configure the backtester
    logger.info("Creating backtester with all components")
    backtester = Backtester(
        config=backtester_config,
        data_handler=data_handler,
        strategy=strategy,
        position_manager=position_manager
    )
    
    # Set execution engine in backtester
    backtester.execution_engine = execution_engine
    backtester.event_bus = event_bus
    
    # Make sure execution engine has the portfolio reference
    if hasattr(backtester.execution_engine, 'portfolio') and backtester.execution_engine.portfolio is None:
        backtester.execution_engine.portfolio = portfolio
    
    # 10. CRITICAL FIX: Explicitly register event handlers for the complete event flow
    # Register event tracer for all event types
    for event_type in EventType:
        event_bus.register(event_type, event_tracer.trace_event)
    
    # CRITICAL FIX: Register position manager to receive SIGNAL events
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)
    logger.info("Registered position manager for SIGNAL events")
    
    # CRITICAL FIX: Create a handler for position actions
    def handle_position_action(event):
        if event.event_type == EventType.POSITION_ACTION:
            action = event.data
            logger.info(f"Handling position action: {action}")
            position_manager.execute_position_action(action)
    
    # Register the position action handler
    event_bus.register(EventType.POSITION_ACTION, handle_position_action)
    logger.info("Registered handler for POSITION_ACTION events")
    
    # CRITICAL FIX: Register execution engine for ORDER events
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    logger.info("Registered execution engine for ORDER events")
    
    # CRITICAL FIX: Register portfolio for FILL events
    event_bus.register(EventType.FILL, portfolio.on_fill if hasattr(portfolio, 'on_fill') else lambda x: None)
    logger.info("Registered portfolio for FILL events")
    
    # 11. Add signal logging
    def log_signal(event):
        signal = event.data
        if hasattr(signal, 'get_signal_value') and hasattr(signal, 'get_symbol'):
            direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
            logger.info(f"Signal: {direction} for {signal.get_symbol()} at {signal.get_price()}")
        else:
            logger.warning(f"Non-standard signal in signal data: {type(signal)}")
    
    event_bus.register(EventType.SIGNAL, log_signal)
    
    # 12. Run the backtest
    logger.info("Starting backtest")
    results = backtester.run(use_test_data=False)
    
    # 13. Display results
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
    
    # 14. Log event statistics from the tracer
    logger.info("Event statistics:")
    for event_type, count in event_tracer.event_counts.items():
        logger.info(f"  {event_type}: {count} events")
    
    return {
        'initial_equity': initial_capital,
        'final_equity': portfolio.equity,
        'return_pct': (portfolio.equity / initial_capital - 1) * 100,
        'trade_count': len(trades),
        'portfolio': portfolio,
        'event_counts': event_tracer.event_counts
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run enhanced algorithmic trading backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to trade')
    parser.add_argument('--timeframe', type=str, default='1m', help='Data timeframe')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse dates if provided
    start_date = datetime.datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.datetime.strptime(args.end, '%Y-%m-%d') if args.end else None
    
    # Create config
    config = {
        'initial_capital': args.capital
    }
    
    # Run enhanced backtest
    results = run_enhanced_backtest(
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
    for event_type, count in results['event_counts'].items():
        logger.info(f"  {event_type}: {count}")
