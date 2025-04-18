"""
Algorithmic Trading System Integration Test

This script tests the complete event flow from signal to fill,
with special focus on short selling and error handling.
"""

import logging
import datetime
import sys
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from your codebase
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent
from src.events.event_base import Event
from src.events.signal_event import SignalEvent
from src.position_management.portfolio import EventPortfolio
from src.position_management.position_manager import PositionManager
from src.position_management.position_sizers import FixedSizeSizer
from src.engine.execution_engine import ExecutionEngine
from src.engine.market_simulator import MarketSimulator

def test_short_selling():
    """Test short selling with no existing positions."""
    
    # Create event bus
    event_bus = EventBus()
    event_bus.event_counts = {}
    
    # Create portfolio
    initial_capital = 100000
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus,
        margin_enabled=True  # Enable margin trading for short selling
    )
    
    # Create position manager
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        event_bus=event_bus
    )
    
    # Create execution engine
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.portfolio = portfolio
    execution_engine.event_bus = event_bus
    execution_engine.market_simulator = MarketSimulator({})
    
    # Register handlers
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)
    event_bus.register(EventType.POSITION_ACTION, execution_engine.on_position_action)
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    
    if hasattr(portfolio, 'handle_fill') and callable(portfolio.handle_fill):
        event_bus.register(EventType.FILL, portfolio.handle_fill)
    else:
        logger.warning("Portfolio does not have a callable handle_fill method")

    if hasattr(portfolio, 'handle_position_action') and callable(portfolio.handle_position_action):
        event_bus.register(EventType.POSITION_ACTION, portfolio.handle_position_action)
    else:
        logger.warning("Portfolio does not have a callable handle_position_action method")
    
    # Create a SELL signal
    timestamp = datetime.datetime.now()
    symbol = "AAPL"
    price = 100.0
    
    signal = SignalEvent(
        signal_value=SignalEvent.SELL,  # -1 for SELL
        price=price,
        symbol=symbol,
        rule_id="test",
        timestamp=timestamp
    )
    
    # Emit signal
    logger.info("Emitting SELL signal")
    event_bus.emit(Event(EventType.SIGNAL, signal))
    
    # Create a mock bar to use for executing pending orders
    bar_data = {
        'symbol': symbol,
        'Open': price,
        'High': price + 1,
        'Low': price - 1,
        'Close': price,
        'Volume': 1000,
        'timestamp': timestamp
    }
    bar_event = BarEvent(bar_data, timestamp)
    
    # Execute pending orders
    logger.info("Executing pending orders")
    fills = execution_engine.execute_pending_orders(bar_event)
    
    # Check results
    logger.info(f"Fills generated: {len(fills)}")
    for fill in fills:
        logger.info(f"Fill: {fill}")
    
    # Check portfolio state
    logger.info(f"Final portfolio state:")
    logger.info(f"Cash: {portfolio.cash}")
    logger.info(f"Equity: {portfolio.equity}")
    
    # Check positions
    portfolio_snapshot = portfolio.get_position_snapshot()
    for symbol, positions in portfolio_snapshot.items():
        for pos in positions:
            direction = "LONG" if pos.get('direction', 0) > 0 else "SHORT"
            quantity = pos.get('quantity', 0)
            entry_price = pos.get('entry_price', 0)
            logger.info(f"Position: {direction} {symbol} {quantity} @ {entry_price}")
    
    # Check event counts
    logger.info("Event counts:")
    for event_type, count in event_bus.event_counts.items():
        if hasattr(event_type, 'name'):
            logger.info(f"  {event_type.name}: {count}")
        else:
            logger.info(f"  {event_type}: {count}")

def test_long_buying():
    """Test buying long positions."""
    
    # Create event bus
    event_bus = EventBus()
    event_bus.event_counts = {}
    
    # Create portfolio
    initial_capital = 100000
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus
    )
    
    # Create position manager
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        event_bus=event_bus
    )
    
    # Create execution engine
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.portfolio = portfolio
    execution_engine.event_bus = event_bus
    execution_engine.market_simulator = MarketSimulator({})
    
    # Register handlers
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)
    event_bus.register(EventType.POSITION_ACTION, execution_engine.on_position_action)
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    
    if hasattr(portfolio, 'handle_fill') and callable(portfolio.handle_fill):
        event_bus.register(EventType.FILL, portfolio.handle_fill)
    else:
        logger.warning("Portfolio does not have a callable handle_fill method")

    if hasattr(portfolio, 'handle_position_action') and callable(portfolio.handle_position_action):
        event_bus.register(EventType.POSITION_ACTION, portfolio.handle_position_action)
    else:
        logger.warning("Portfolio does not have a callable handle_position_action method")
    
    # Create a BUY signal
    timestamp = datetime.datetime.now()
    symbol = "AAPL"
    price = 100.0
    
    signal = SignalEvent(
        signal_value=SignalEvent.BUY,  # 1 for BUY
        price=price,
        symbol=symbol,
        rule_id="test",
        timestamp=timestamp
    )
    
    # Emit signal
    logger.info("Emitting BUY signal")
    event_bus.emit(Event(EventType.SIGNAL, signal))
    
    # Create a mock bar to use for executing pending orders
    bar_data = {
        'symbol': symbol,
        'Open': price,
        'High': price + 1,
        'Low': price - 1,
        'Close': price,
        'Volume': 1000,
        'timestamp': timestamp
    }
    bar_event = BarEvent(bar_data, timestamp)
    
    # Execute pending orders
    logger.info("Executing pending orders")
    fills = execution_engine.execute_pending_orders(bar_event)
    
    # Check results
    logger.info(f"Fills generated: {len(fills)}")
    for fill in fills:
        logger.info(f"Fill: {fill}")
    
    # Check portfolio state
    logger.info(f"Final portfolio state:")
    logger.info(f"Cash: {portfolio.cash}")
    logger.info(f"Equity: {portfolio.equity}")
    
    # Check positions
    portfolio_snapshot = portfolio.get_position_snapshot()
    for symbol, positions in portfolio_snapshot.items():
        for pos in positions:
            direction = "LONG" if pos.get('direction', 0) > 0 else "SHORT"
            quantity = pos.get('quantity', 0)
            entry_price = pos.get('entry_price', 0)
            logger.info(f"Position: {direction} {symbol} {quantity} @ {entry_price}")
    
    # Check event counts
    logger.info("Event counts:")
    for event_type, count in event_bus.event_counts.items():
        if hasattr(event_type, 'name'):
            logger.info(f"  {event_type.name}: {count}")
        else:
            logger.info(f"  {event_type}: {count}")

def test_complete_flow():
    """Test complete event flow from bar to fill."""
    
    # Create event bus
    event_bus = EventBus()
    event_bus.event_counts = {}
    
    # Create portfolio
    initial_capital = 100000
    portfolio = EventPortfolio(
        initial_capital=initial_capital,
        event_bus=event_bus,
        margin_enabled=True  # Enable margin trading for short selling
    )
    
    # Create position manager
    position_sizer = FixedSizeSizer(fixed_size=10)
    position_manager = PositionManager(
        portfolio=portfolio,
        position_sizer=position_sizer,
        event_bus=event_bus
    )
    
    # Create execution engine
    execution_engine = ExecutionEngine(position_manager=position_manager)
    execution_engine.portfolio = portfolio
    execution_engine.event_bus = event_bus
    execution_engine.market_simulator = MarketSimulator({})
    
    # Create a simple strategy that creates signals
    class SimpleStrategy:
        def __init__(self, event_bus):
            self.event_bus = event_bus
            
        def on_bar(self, event):
            # Extract bar data
            if not isinstance(event, Event) or not hasattr(event, 'data'):
                return
                
            bar_data = event.data
            if not isinstance(bar_data, BarEvent):
                return
                
            # Create a signal based on bar data
            symbol = bar_data.get_symbol()
            price = bar_data.get_price()
            
            # Simple strategy: BUY if price > 100, SELL if price < 100
            if price > 100:
                signal_value = SignalEvent.BUY
            elif price < 100:
                signal_value = SignalEvent.SELL
            else:
                signal_value = SignalEvent.NEUTRAL
                
            # Skip neutral signals
            if signal_value == SignalEvent.NEUTRAL:
                return
                
            # Create and emit signal
            signal = SignalEvent(
                signal_value=signal_value,
                price=price,
                symbol=symbol,
                rule_id="simple_strategy",
                timestamp=bar_data.get_timestamp()
            )
            
            self.event_bus.emit(Event(EventType.SIGNAL, signal))
    
    # Create strategy
    strategy = SimpleStrategy(event_bus)
    
    # Register handlers
    event_bus.register(EventType.BAR, strategy.on_bar)
    event_bus.register(EventType.SIGNAL, position_manager.on_signal)
    event_bus.register(EventType.POSITION_ACTION, execution_engine.on_position_action)
    event_bus.register(EventType.ORDER, execution_engine.on_order)
    
    if hasattr(portfolio, 'handle_fill') and callable(portfolio.handle_fill):
        event_bus.register(EventType.FILL, portfolio.handle_fill)
    else:
        logger.warning("Portfolio does not have a callable handle_fill method")

    if hasattr(portfolio, 'handle_position_action') and callable(portfolio.handle_position_action):
        event_bus.register(EventType.POSITION_ACTION, portfolio.handle_position_action)
    else:
        logger.warning("Portfolio does not have a callable handle_position_action method")
    
    # Test with a series of bars with different prices
    symbols = ["AAPL", "GOOGL", "MSFT"]
    prices = [95, 105, 98, 110, 92]
    
    for i, price in enumerate(prices):
        for symbol in symbols:
            timestamp = datetime.datetime.now() + datetime.timedelta(minutes=i)
            
            # Create bar data
            bar_data = {
                'symbol': symbol,
                'Open': price - 1,
                'High': price + 2,
                'Low': price - 2,
                'Close': price,
                'Volume': 1000,
                'timestamp': timestamp
            }
            
            # Create bar event
            bar_event = BarEvent(bar_data, timestamp)
            
            # Emit bar event
            logger.info(f"Emitting BAR event: {symbol} @ {price}")
            event_bus.emit(Event(EventType.BAR, bar_event))
            
            # Execute pending orders
            fills = execution_engine.execute_pending_orders(bar_event)
            
            if fills:
                logger.info(f"Generated {len(fills)} fills")
    
    # Check final portfolio state
    logger.info(f"Final portfolio state:")
    logger.info(f"Cash: {portfolio.cash}")
    logger.info(f"Equity: {portfolio.equity}")
    
    # Check positions
    portfolio_snapshot = portfolio.get_position_snapshot()
    for symbol, positions in portfolio_snapshot.items():
        for pos in positions:
            direction = "LONG" if pos.get('direction', 0) > 0 else "SHORT"
            quantity = pos.get('quantity', 0)
            entry_price = pos.get('entry_price', 0)
            logger.info(f"Position: {direction} {symbol} {quantity} @ {entry_price}")
    
    # Check event counts
    logger.info("Event counts:")
    for event_type, count in event_bus.event_counts.items():
        if hasattr(event_type, 'name'):
            logger.info(f"  {event_type.name}: {count}")
        else:
            logger.info(f"  {event_type}: {count}")

if __name__ == "__main__":
    logger.info("Testing short selling")
    test_short_selling()
    
    logger.info("\n\nTesting long buying")
    test_long_buying()
    
    logger.info("\n\nTesting complete flow")
    test_complete_flow()
