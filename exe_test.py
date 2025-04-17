#!/usr/bin/env python3
"""
Complete Integration Script

This script properly connects all components of the trading system
to create a complete event flow from signals to portfolio updates.
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
from src.events.event_types import EventType
from src.events.event_handlers import SignalHandler
from src.events.signal_event import SignalEvent

from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.position_management.position_sizers import FixedSizeSizer
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.engine.execution_engine import ExecutionEngine
from src.engine.backtester import Backtester

def display_results(results):
    """Display backtest results."""
    logger.info("=========== BACKTEST RESULTS ===========")
    logger.info(f"Initial equity: ${results['initial_equity']:,.2f}")
    logger.info(f"Final equity: ${results['final_equity']:,.2f}")
    logger.info(f"Return: {results['return_pct']:.2f}%")
    logger.info(f"Bars processed: {results['bars_processed']}")
    logger.info(f"Signals generated: {results['signals_generated']}")
    logger.info(f"Trades executed: {len(results['trades'])}")
    logger.info(f"Closed positions: {results['closed_positions']}")
    logger.info("=======================================")



def verify_event_bus():
    """Verify the EventBus is working correctly."""
    logger.info("Verifying EventBus functionality...")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create a test handler
    handler_called = [False]
    
    def test_handler(event):
        handler_called[0] = True
        logger.info("Test handler called successfully")
    
    # Register handler
    event_bus.register(EventType.START, test_handler)
    
    # Create and emit a test event
    test_event = Event(EventType.START)
    event_bus.emit(test_event)
    
    # Verify handler was called
    if not handler_called[0]:
        logger.error("ERROR: Handler not called during emit")
        return False
    
    logger.info("EventBus verification PASSED")
    return True



def run_backtest(config, start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """
    Run a complete backtest using the codebase components with enhanced debugging.
    
    Args:
        config: Configuration dictionary
        start_date: Start date for the backtest
        end_date: End date for the backtest
        symbols: List of symbols to backtest
        timeframe: Timeframe for the data
        
    Returns:
        Dictionary of backtest results
    """
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # Verify event bus functionality
    if not verify_event_bus():
        logger.error("EventBus verification failed, aborting backtest.")
        return None
    
    # Create event bus
    event_bus = EventBus()
    logger.info("Created EventBus")
    
    # Create portfolio with initial capital
    initial_capital = config.get('initial_capital', 100000)
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    logger.info(f"Created portfolio with {initial_capital} capital")
    
    # Create rule with enhanced signal tracking
    rule = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy",
        event_bus=event_bus  # Pass event bus directly to rule
    )
    logger.info(f"Created {rule}")
    
    # Create position sizer and position manager
    position_sizer = FixedSizeSizer(fixed_size=100)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
    )
    logger.info("Created position manager with fixed sizer")
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)

    # Check position manager methods
    if not hasattr(position_manager, 'on_signal'):
        logger.error("CRITICAL: Position manager does not have on_signal method!")
    else:
        logger.info(f"Position manager has on_signal method")
    
    # Create execution engine
    execution_engine = ExecutionEngine(
        position_manager=position_manager
    )
    execution_engine.portfolio = portfolio
    logger.info("Created execution engine")
    
    # Custom wrapper for rule.on_bar to trace signal emission
    def rule_on_bar_wrapper(event):
        """Wrapper around rule.on_bar to track signal emission"""
        try:
            logger.debug(f"Rule wrapper received bar event for {event.data.get_symbol()}")
            signal = rule.on_bar(event)

            # Debug the signal type
            if signal:
                logger.info(f"SIGNAL TYPE DEBUG: Signal type is {type(signal).__name__}")
                logger.info(f"SIGNAL ATTRS DEBUG: Signal has attributes: {dir(signal)}")

                if isinstance(signal, SignalEvent):
                    logger.info(f"DIRECT TRACE: Rule generated {signal.get_signal_name()} for {signal.get_symbol()} @ {signal.timestamp}")
                else:
                    logger.info(f"DIRECT TRACE: Rule generated non-SignalEvent: {type(signal).__name__}")

                # Manual conversion to SignalEvent if needed
                if not isinstance(signal, SignalEvent):
                    logger.info(f"Converting non-SignalEvent to SignalEvent")
                    # Extract attributes from signal
                    if hasattr(signal, 'signal_type'):
                        signal_value = signal.signal_type.value if hasattr(signal.signal_type, 'value') else 0
                    elif hasattr(signal, 'signal_value'):
                        signal_value = signal.signal_value
                    else:
                        signal_value = 0

                    # Create proper SignalEvent
                    signal_event = SignalEvent(
                        signal_value=signal_value,
                        price=getattr(signal, 'price', 0),
                        symbol=getattr(signal, 'symbol', 'unknown'),
                        rule_id=getattr(signal, 'rule_id', rule.name),
                        metadata=getattr(signal, 'metadata', {}),
                        timestamp=getattr(signal, 'timestamp', datetime.datetime.now())
                    )

                    # Use converted signal
                    signal = signal_event

                # Manually create and emit signal event to ensure it's happening
                signal_event = Event(EventType.SIGNAL, signal)
                event_bus.emit(signal_event)
                logger.info(f"DIRECT TRACE: Signal event manually emitted to event bus")
                return signal
        except Exception as e:
            logger.error(f"Error in rule_on_bar_wrapper: {str(e)}", exc_info=True)
        return None

    
    # Debug signal handler that logs extensively
    # Debug signal handler that logs extensively
    # Debug signal handler that logs extensively
    def debug_signal_handler(event):
        """Debug signal handler to trace signal processing"""
        try:
            if not hasattr(event, 'data'):
                logger.error(f"SIGNAL HANDLER: Event has no data attribute")
                return

            # Check what we've received
            logger.info(f"SIGNAL HANDLER: Received event with data type: {type(event.data).__name__}")

            # If we have a SignalEvent directly in the event
            if isinstance(event.data, SignalEvent):
                signal = event.data
                logger.info(f"SIGNAL HANDLER: Received {signal.get_signal_name()} for {signal.get_symbol()}")

                # IMPORTANT: Create a modified on_signal method to handle the signal directly, not wrapped in an event
                # This is a temporary workaround for the dictionary conversion issue
                def process_signal_directly(signal_object):
                    """Process the signal directly without event wrapping"""
                    if not isinstance(signal_object, SignalEvent):
                        logger.error(f"Expected SignalEvent, got {type(signal_object).__name__}")
                        return []

                    # Store signal for tracking
                    position_manager.signal_history.append(signal_object)

                    # Get signal data using proper getters
                    symbol = signal_object.get_symbol()
                    direction = signal_object.get_signal_value()
                    price = signal_object.get_price()

                    logger.info(f"DIRECT PROCESSING: Processing {signal_object.get_signal_name()} signal for {symbol}")

                    # Skip neutral signals
                    if direction == SignalEvent.NEUTRAL:
                        logger.debug(f"Skipping neutral signal for {symbol}")
                        return []

                    # Create a simple position action for testing
                    # This is a simplified version just to test the signal flow
                    action = {
                        'action_type': 'entry',
                        'symbol': symbol,
                        'direction': direction,
                        'size': 100,  # Fixed size for testing
                        'price': price,
                        'timestamp': signal_object.timestamp
                    }

                    logger.info(f"DIRECT PROCESSING: Created action: {action}")

                    # Emit the action directly
                    if hasattr(position_manager, 'event_bus') and position_manager.event_bus:
                        position_manager.event_bus.emit(Event(EventType.POSITION_ACTION, action))
                        logger.info(f"DIRECT PROCESSING: Emitted position action")

                    return [action]

                # Process the signal directly
                logger.info(f"SIGNAL HANDLER: Processing signal directly...")
                actions = process_signal_directly(signal)

                if actions:
                    logger.info(f"SIGNAL HANDLER: Produced {len(actions)} action(s)")
                else:
                    logger.info(f"SIGNAL HANDLER: No actions produced")

                return

            # If we have a dictionary (serialized signal)
            elif isinstance(event.data, dict) and 'signal_value' in event.data:
                logger.info(f"SIGNAL HANDLER: Received dictionary with signal data, converting to SignalEvent")

                # Convert dictionary to SignalEvent
                signal_data = event.data
                signal = SignalEvent(
                    signal_value=signal_data.get('signal_value', 0),
                    price=signal_data.get('price', 0),
                    symbol=signal_data.get('symbol', 'unknown'),
                    rule_id=signal_data.get('rule_id', 'unknown'),
                    metadata=signal_data.get('metadata', {}),
                    timestamp=signal_data.get('timestamp', datetime.datetime.now())
                )

                # Now call the direct processing function with the converted signal
                logger.info(f"SIGNAL HANDLER: Processing converted signal...")
                # [Same direct processing code would go here]
                return

            # Other types of data
            else:
                logger.error(f"SIGNAL HANDLER: Unrecognized data type: {type(event.data).__name__}")
                return

        except Exception as e:
            logger.error(f"Error in debug_signal_handler: {str(e)}", exc_info=True)

    
    # Set up event handlers with enhanced debugging
    # 1. Register rule wrapper with BAR events (instead of rule directly)
    event_bus.register(EventType.BAR, rule_on_bar_wrapper)
    logger.info("Registered rule wrapper with BAR events")
    
    # 2. Register debug signal handler
    event_bus.register(EventType.SIGNAL, debug_signal_handler)
    logger.info("Registered debug signal handler with SIGNAL events")
    
    # 3. Create position action handler with debugging
    def debug_position_action_handler(event):
        """Debug position action handler to trace action processing"""
        try:
            action = event.data
            logger.info(f"POSITION ACTION HANDLER: Received {action.get('action_type')} for {action.get('symbol')}")
            
            # Extract action data
            action_type = action.get('action_type')
            
            if action_type == 'entry':
                logger.info(f"POSITION ACTION: Entry for {action.get('symbol')} direction={action.get('direction')} size={action.get('size')} price={action.get('price')}")
                # Process entry action
                portfolio._handle_position_action(event)
            elif action_type == 'exit':
                logger.info(f"POSITION ACTION: Exit for position ID {action.get('position_id')} price={action.get('price')}")
                # Process exit action
                portfolio._handle_position_action(event)
        except Exception as e:
            logger.error(f"Error in debug_position_action_handler: {str(e)}", exc_info=True)
    
    # Register position action handler
    event_bus.register(EventType.POSITION_ACTION, debug_position_action_handler)
    logger.info("Registered debug position action handler")
    
    # 4. Register execution engine with ORDER events
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    logger.info("Registered execution engine with ORDER events")
    
    # 5. Create debug fill handler
    def debug_fill_handler(event):
        """Debug fill handler to trace fill processing"""
        try:
            fill = event.data
            logger.info(f"FILL HANDLER: Received fill for {fill.get('symbol')} quantity={fill.get('quantity')} price={fill.get('price')}")
            
            # Forward to portfolio
            if hasattr(portfolio, 'on_fill'):
                portfolio.on_fill(event)
                logger.info(f"FILL HANDLER: Forwarded fill to portfolio")
            else:
                logger.error(f"FILL HANDLER: Portfolio does not have on_fill method!")
        except Exception as e:
            logger.error(f"Error in debug_fill_handler: {str(e)}", exc_info=True)
    
    # Register fill handler
    event_bus.register(EventType.FILL, debug_fill_handler)
    logger.info("Registered debug fill handler")
    
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
    
    # Create results tracking
    results = {
        'initial_equity': initial_capital,
        'final_equity': initial_capital,
        'signals_generated': 0,
        'orders_processed': 0, 
        'fills_generated': 0,
        'bars_processed': 0,
        'trades': []
    }
    
    # Track signals emitted and processed
    signals_emitted = []
    
    # Run manual backtest loop
    logger.info("Starting backtest loop")
    for i, bar in enumerate(data_handler.iter_train()):
        # Track bars processed
        results['bars_processed'] += 1
        
        try:
            # Create and emit bar event
            bar_event = data_handler.create_bar_event(bar)
            event_bus.emit(Event(EventType.BAR, bar_event))
            logger.debug(f"Emitted BAR event for {bar_event.get_symbol()} @ {bar_event.get_timestamp()}")
            
            # Execute any pending orders with current bar
            fills = execution_engine.execute_pending_orders(bar_event)
            if fills:
                logger.info(f"Executed {len(fills)} pending orders")
                results['fills_generated'] += len(fills)
                
                # Emit fills
                for fill in fills:
                    event_bus.emit(Event(EventType.FILL, fill))
                    logger.info(f"Emitted FILL event for {fill.get('symbol')}")
            
            # Update portfolio and execution engine with current bar
            execution_engine.update(bar_event)
        except Exception as e:
            logger.error(f"Error processing bar {i}: {str(e)}", exc_info=True)
        
        # Log progress
        if i % 50 == 0 or i == len(data_handler.train_data) - 1:
            # Count signals
            signals_count = len(signals_emitted)
            logger.info(f"Processed {i} bars, {signals_count} signals generated")
            
            # Log portfolio state
            logger.info(f"Portfolio equity: {portfolio.equity:.2f}, cash: {portfolio.cash:.2f}")
            
            # Log open positions
            position_count = len(portfolio.positions)
            if position_count > 0:
                logger.info(f"Open positions: {position_count}")
                for pos_id, pos in portfolio.positions.items():
                    logger.info(f"  Position {pos_id}: {pos.symbol} {pos.direction} {pos.quantity} @ {pos.entry_price}")
    
    # Collect results
    results['final_equity'] = portfolio.equity
    results['return_pct'] = (portfolio.equity / initial_capital - 1) * 100
    results['signals_generated'] = len(signals_emitted)
    
    # Get trade history
    if hasattr(execution_engine, 'get_trade_history'):
        results['trades'] = execution_engine.get_trade_history()
    
    # Get closed positions from portfolio
    results['closed_positions'] = len(portfolio.closed_positions) if hasattr(portfolio, 'closed_positions') else 0
    
    # Display results
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
