import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import essential components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.data.data_handler import DataHandler, CSVDataSource

# Import rule and backtester
from src.rules.crossover_rules import SMACrossoverRule
from src.engine.backtester import Backtester

# Import position management components if needed
from src.position_management.position_sizers import FixedSizeSizer
from src.engine.execution_engine import ExecutionEngine

# Signal handler for monitoring (optional)
signal_count = 0
def handle_signal(event):
    global signal_count
    signal = event.data
    signal_count += 1
    logger.info(f"SIGNAL {signal_count}: {signal.get_signal_name()} for {signal.get_symbol()} @ {signal.get_price():.2f}")

# Create a strategy with our rule
class SimpleStrategy:
    def __init__(self, rule, name="simple_strategy"):
        self.rule = rule
        self.name = name
        self.event_bus = None
        logger.info(f"Strategy created with rule: {rule}")
    
    def set_event_bus(self, event_bus):
        self.event_bus = event_bus
        self.rule.set_event_bus(event_bus)
        logger.info("Event bus set on strategy and rule")
    
    def on_bar(self, event):
        # Just delegate to rule
        return self.rule.on_bar(event)
    
    def reset(self):
        self.rule.reset()

# Create our components
try:
    logger.info("Creating components...")
    
    # Create event bus
    event_bus = EventBus()
    
    # Register signal monitor (optional)
    event_bus.register(EventType.SIGNAL, handle_signal)
    
    # Create rule
    rule = SMACrossoverRule(
        name="sma_crossover_test",
        params={
            "fast_window": 5, 
            "slow_window": 15
        },
        description="SMA Crossover test"
    )
    
    # Create strategy with rule
    strategy = SimpleStrategy(rule)
    
    # Create position sizer (for backtester)
    position_sizer = FixedSizeSizer(fixed_size=100)
    
    # Create data source and handler
    data_dir = "./data"
    data_source = CSVDataSource(data_dir)
    data_handler = DataHandler(data_source)
    
    # Create execution engine
    execution_engine = ExecutionEngine()
    
    # Create configuration for backtester
    config = {
        'backtester': {
            'initial_capital': 100000,
            'market_simulation': {
                'slippage_model': 'fixed',
                'slippage_bps': 5,  # 5 basis points
                'fee_model': 'fixed',
                'fee_bps': 10  # 10 basis points
            }
        }
    }
    
    # Create backtester
    backtester = Backtester(
        config=config,
        data_handler=data_handler,
        strategy=strategy, 
        execution_engine=execution_engine
    )
    
    # Set up data for backtesting
    symbols = ["SPY"]
    start_date = datetime.datetime(2024, 3, 26)
    end_date = datetime.datetime(2024, 3, 27)
    timeframe = "1m"
    
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols, 
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Run the backtest
    logger.info("Running backtest...")
    results = backtester.run()
    
    # Print results
    logger.info("Backtest completed!")
    logger.info(f"Signal handler received {signal_count} signals")
    
    # Display trade results
    logger.info("Trade Results:")
    trade_history = results.get('trade_history', [])
    logger.info(f"Total Trades: {len(trade_history)}")
    
    if trade_history:
        winning_trades = [t for t in trade_history if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len(trade_history) if trade_history else 0
        logger.info(f"Win Rate: {win_rate:.2%}")
        
        total_profit = sum(t.get('profit', 0) for t in trade_history)
        logger.info(f"Total Profit: ${total_profit:.2f}")
        
        # Print first and last few trades
        if len(trade_history) > 0:
            logger.info("Sample Trades:")
            sample_size = min(3, len(trade_history))
            for i in range(sample_size):
                trade = trade_history[i]
                logger.info(f"Trade {i+1}: {trade.get('direction', '')} {trade.get('symbol', '')} "
                          f"@ {trade.get('entry_price', 0):.2f} -> {trade.get('exit_price', 0):.2f}, "
                          f"Profit: ${trade.get('profit', 0):.2f}")
    
    # Display equity curve summary
    equity_curve = results.get('equity_curve', [])
    if equity_curve:
        initial = equity_curve[0] if equity_curve else config['backtester']['initial_capital']
        final = equity_curve[-1] if equity_curve else initial
        logger.info(f"Initial Equity: ${initial:.2f}")
        logger.info(f"Final Equity: ${final:.2f}")
        logger.info(f"Return: {(final/initial - 1):.2%}")
    
except Exception as e:
    logger.error(f"Error in backtest: {e}", exc_info=True)
