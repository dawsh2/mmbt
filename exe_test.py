#!/usr/bin/env python3
# integrated_backtest.py - Using codebase components with fixed event system

import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the actual codebase
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule
from src.engine.backtester import Backtester
from src.position_management.position_sizers import FixedSizeSizer
from src.engine.execution_engine import ExecutionEngine
from src.position_management.portfolio import EventPortfolio

# Create a simple position manager class
class SimplePositionManager:
    def __init__(self, position_sizer=None):
        self.position_sizer = position_sizer
        logger.info("SimplePositionManager initialized")
    
    def calculate_position_size(self, signal, portfolio, current_price=None):
        """Calculate position size based on signal."""
        if self.position_sizer:
            return self.position_sizer.calculate_position_size(signal, portfolio, current_price)
        
        # Default implementation if no position sizer
        direction = signal.get_signal_value()
        return direction * 100  # Return positive for buy, negative for sell
    
    def reset(self):
        """Reset the position manager."""
        pass

# Create a simple strategy class
class SimpleStrategy:
    def __init__(self, rule, name="SimpleStrategy"):
        self.rule = rule
        self.event_bus = None
        self.signals_generated = 0
        logger.info(f"SimpleStrategy initialized with rule: {rule}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus for the strategy."""
        self.event_bus = event_bus
        # If rule has a set_event_bus method, call it
        if hasattr(self.rule, 'set_event_bus'):
            self.rule.set_event_bus(event_bus)
    
    def on_bar(self, event):
        """Process a bar event and generate signals."""
        # Delegate to rule and count signals
        signal = self.rule.on_bar(event)
        if signal:
            self.signals_generated += 1
            logger.info(f"Strategy generated signal #{self.signals_generated}: {signal}")
        return signal
    
    def reset(self):
        """Reset the strategy."""
        self.rule.reset()
        self.signals_generated = 0

# Function to create custom backtester
def create_custom_backtester(config, data_handler, strategy, position_manager):
    """Create a custom backtester with properly initialized components."""
    # Extract initial capital from config
    initial_capital = config.get('backtester', {}).get('initial_capital', 100000)
    
    # Create event bus
    event_bus = EventBus()
    
    # Set event bus on strategy
    strategy.set_event_bus(event_bus)
    
    # Create portfolio
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    
    # Create execution engine - check arguments it accepts
    try:
        # First try with common parameters
        execution_engine = ExecutionEngine(
            position_manager=position_manager
        )
        # If portfolio should be set separately
        if hasattr(execution_engine, 'set_portfolio'):
            execution_engine.set_portfolio(portfolio)
        elif hasattr(execution_engine, 'portfolio'):
            execution_engine.portfolio = portfolio
    except TypeError as e:
        logger.warning(f"ExecutionEngine init error: {e}")
        # Try with different parameter combinations
        try:
            execution_engine = ExecutionEngine(
                initial_capital=initial_capital
            )
            # Set position manager if needed
            if hasattr(execution_engine, 'set_position_manager'):
                execution_engine.set_position_manager(position_manager)
            elif hasattr(execution_engine, 'position_manager'):
                execution_engine.position_manager = position_manager
            
            # Set portfolio if needed
            if hasattr(execution_engine, 'set_portfolio'):
                execution_engine.set_portfolio(portfolio)
            elif hasattr(execution_engine, 'portfolio'):
                execution_engine.portfolio = portfolio
        except TypeError as e:
            logger.error(f"Failed to create ExecutionEngine: {e}")
            # Create a simple execution engine
            class SimpleExecutionEngine:
                def __init__(self):
                    self.portfolio = portfolio
                    self.position_manager = position_manager
                    
                def on_order(self, event):
                    logger.info(f"Processing order: {event}")
                    # Simple implementation
                    
                def reset(self):
                    pass
                    
            execution_engine = SimpleExecutionEngine()
            logger.warning("Created SimpleExecutionEngine")
    
    # Set event bus on execution engine if needed
    if hasattr(execution_engine, 'set_event_bus'):
        execution_engine.set_event_bus(event_bus)
    
    # Create backtester with all components
    try:
        # Try with all components
        backtester = Backtester(
            config=config,
            data_handler=data_handler,
            strategy=strategy,
            position_manager=position_manager,
            execution_engine=execution_engine,
            portfolio=portfolio,
            event_bus=event_bus
        )
    except TypeError as e:
        logger.warning(f"Backtester init error with all parameters: {e}")
        # Try with required components only
        backtester = Backtester(
            config=config,
            data_handler=data_handler,
            strategy=strategy,
            position_manager=position_manager
        )
        
        # Set other components if needed
        if hasattr(backtester, 'set_execution_engine'):
            backtester.set_execution_engine(execution_engine)
        elif hasattr(backtester, 'execution_engine'):
            backtester.execution_engine = execution_engine
            
        if hasattr(backtester, 'set_portfolio'):
            backtester.set_portfolio(portfolio)
        elif hasattr(backtester, 'portfolio'):
            backtester.portfolio = portfolio
            
        if hasattr(backtester, 'set_event_bus'):
            backtester.set_event_bus(event_bus)
        elif hasattr(backtester, 'event_bus'):
            backtester.event_bus = event_bus
    
    return backtester

# Function to run a backtest
def run_backtest(start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """Run a complete backtest using the codebase components."""
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # Create rule
    rule = SMACrossoverRule(
        name="sma_crossover",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover strategy"
    )
    
    # Create strategy
    strategy = SimpleStrategy(rule)
    
    # Create position sizer
    position_sizer = FixedSizeSizer(fixed_size=100)
    
    # Create position manager with position sizer
    position_manager = SimplePositionManager(position_sizer=position_sizer)
    
    # Create configuration
    config = {
        'backtester': {
            'initial_capital': 100000,
            'market_simulation': {
                'slippage_model': 'fixed',
                'slippage_bps': 5,
                'fee_model': 'fixed',
                'fee_bps': 10
            }
        }
    }
    
    # Create data handler
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source)
    
    # Load data
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Create custom backtester with all components properly initialized
    backtester = create_custom_backtester(
        config=config,
        data_handler=data_handler,
        strategy=strategy,
        position_manager=position_manager
    )
    
    # Run backtest
    logger.info("Running backtest")
    results = backtester.run()
    
    # Display results
    display_results(results)
    
    return results

def display_results(results):
    """Display backtest results."""
    logger.info("=== BACKTEST RESULTS ===")
    
    # Adapt based on what your results dictionary contains
    if isinstance(results, dict):
        if 'signals_generated' in results:
            logger.info(f"Signals generated: {results['signals_generated']}")
        
        if 'orders_created' in results:
            logger.info(f"Orders created: {results['orders_created']}")
        
        if 'num_trades' in results:
            logger.info(f"Trades executed: {results['num_trades']}")
        
        if 'win_rate' in results and results.get('num_trades', 0) > 0:
            logger.info(f"Win rate: {results['win_rate']:.2%}")
            
        if 'trade_history' in results and len(results['trade_history']) > 0:
            logger.info("Sample trades:")
            sample_size = min(3, len(results['trade_history']))
            for i in range(sample_size):
                trade = results['trade_history'][i]
                # Format based on your trade dictionary structure
                logger.info(f"Trade {i+1}: {trade}")
        
        if 'initial_equity' in results and 'final_equity' in results:
            logger.info(f"Initial equity: ${results['initial_equity']:.2f}")
            logger.info(f"Final equity: ${results['final_equity']:.2f}")
            
            if 'return_pct' in results:
                logger.info(f"Return: {results['return_pct']:.2%}")
            else:
                # Calculate if not provided
                return_pct = ((results['final_equity'] / results['initial_equity']) - 1) * 100
                logger.info(f"Return: {return_pct:.2%}")
    else:
        logger.info(f"Results: {results}")
            
    logger.info("=======================")

# Main entry point
if __name__ == "__main__":
    run_backtest()
