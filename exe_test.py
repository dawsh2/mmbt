#!/usr/bin/env python3
# fixed_trading_system.py - Complete fixed implementation

import datetime
import logging
import numpy as np
import uuid  # Required for OrderEvent

# Configure detailed logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import essential components
from src.events.event_base import Event
from src.events.event_bus import EventBus
from src.events.event_types import EventType, BarEvent, OrderEvent
from src.events.signal_event import SignalEvent
from src.data.data_handler import DataHandler, CSVDataSource
from src.rules.crossover_rules import SMACrossoverRule

# First ensure uuid is available in event_types
import src.events.event_types
src.events.event_types.uuid = uuid

# Simple position manager
class SimplePositionManager:
    def __init__(self, fixed_size=100):
        self.fixed_size = fixed_size
        logger.info(f"SimplePositionManager initialized with fixed size {fixed_size}")
    
    def calculate_position_size(self, signal, portfolio, current_price=None):
        """Calculate position size based on signal."""
        direction = signal.get_signal_value()
        return direction * self.fixed_size  # Return positive for buy, negative for sell
    
    def reset(self):
        """Reset the position manager."""
        pass

# Create a simple portfolio class for testing
class SimplePortfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.equity = initial_capital
        self.positions = {}
        self.trade_history = []  # Add this to track trades
        logger.info(f"SimplePortfolio initialized with ${initial_capital}")
    
    def update_position(self, symbol, quantity_delta, price, timestamp):
        """Update a position with a quantity change."""
        # If new position
        if symbol not in self.positions:
            if quantity_delta == 0:
                return True
                
            self.positions[symbol] = {
                'quantity': quantity_delta,
                'avg_price': price,
                'timestamp': timestamp,
                'entry_price': price,
                'entry_time': timestamp
            }
            
            # Record the trade opening
            trade = {
                'symbol': symbol,
                'entry_price': price,
                'entry_time': timestamp,
                'quantity': quantity_delta,
                'direction': 1 if quantity_delta > 0 else -1
            }
            
            if quantity_delta > 0:
                logger.info(f"Opened LONG position: {symbol} {quantity_delta} @ {price}")
            else:
                logger.info(f"Opened SHORT position: {symbol} {abs(quantity_delta)} @ {price}")
            
        else:
            # Update existing position
            current_qty = self.positions[symbol]['quantity']
            new_qty = current_qty + quantity_delta
            
            # If closing position
            if new_qty == 0:
                position = self.positions[symbol]
                position_value = current_qty * price
                self.cash += position_value
                
                # Calculate P&L
                entry_price = position['entry_price']
                direction = 1 if current_qty > 0 else -1
                
                if direction > 0:  # Long
                    profit = (price - entry_price) * current_qty
                else:  # Short
                    profit = (entry_price - price) * abs(current_qty)
                
                # Record the completed trade
                trade = {
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'entry_time': position['entry_time'],
                    'exit_price': price,
                    'exit_time': timestamp,
                    'quantity': abs(current_qty),
                    'direction': direction,
                    'profit': profit,
                    'percent_return': ((price / entry_price) - 1) * 100 * direction
                }
                
                self.trade_history.append(trade)
                
                logger.info(f"Closed position: {symbol} P&L: ${profit:.2f}")
                
                del self.positions[symbol]
                return True
            
            # Update position
            self.positions[symbol]['quantity'] = new_qty
            self.positions[symbol]['timestamp'] = timestamp
        
        # Update cash (deduct cost of purchase)
        position_cost = quantity_delta * price
        self.cash -= position_cost
        
        # Update equity
        self.mark_to_market({symbol: price})
        
        return True
    
    def mark_to_market(self, price_data):
        """Update portfolio value based on current prices."""
        self.equity = self.cash
        
        for symbol, position in self.positions.items():
            # Get price for this symbol
            if isinstance(price_data, BarEvent):
                if price_data.get_symbol() == symbol:
                    price = price_data.get_price()
                else:
                    price = position['avg_price']  # Use last known price
            elif isinstance(price_data, dict):
                if 'symbol' in price_data and price_data['symbol'] == symbol:
                    price = price_data['Close']
                elif symbol in price_data:
                    price = price_data[symbol]
                else:
                    price = position['avg_price']
            else:
                price = position['avg_price']
            
            # Update equity
            self.equity += position['quantity'] * price
    
    def get_position_snapshot(self):
        """Get a snapshot of current positions."""
        return self.positions.copy()
    
    def get_trade_history(self):
        """Get the trade history."""
        return self.trade_history

# Custom execution engine
class CustomExecutionEngine:
    def __init__(self, position_manager=None, market_simulator=None, initial_capital=100000):
        """Custom execution engine for testing."""
        self.position_manager = position_manager
        self.market_simulator = market_simulator
        self.portfolio = SimplePortfolio(initial_capital)
        self.initial_capital = initial_capital
        self.pending_orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.signal_history = []
        self.event_bus = None
        self.last_known_prices = {}
        logger.info("CustomExecutionEngine initialized")
    
    def on_order(self, event):
        """Process order events."""
        logger.info(f"ExecutionEngine received order event: {event}")
        
        if not hasattr(event, 'data'):
            logger.error("Event has no data attribute")
            return
            
        order = event.data
        
        # Extract order details
        try:
            if hasattr(order, 'get_symbol'):
                symbol = order.get_symbol()
                quantity = order.get_quantity()
                direction = order.get_direction()
                price = order.get_price()
            else:
                # Dictionary-based order (deprecated)
                symbol = order.get('symbol', 'UNKNOWN')
                quantity = order.get('quantity', 0)
                direction = order.get('direction', 0)
                price = order.get('price', 0)
            
            # Adjust quantity based on direction
            actual_quantity = quantity * direction
            
            # Execute the order (update portfolio)
            current_time = getattr(order, 'timestamp', datetime.datetime.now())
            
            # Update the portfolio position
            result = self.portfolio.update_position(
                symbol=symbol,
                quantity_delta=actual_quantity,
                price=price,
                timestamp=current_time
            )
            
            # Store trade if it executed
            if result:
                if isinstance(self.portfolio, SimplePortfolio):
                    # Make sure trade history is updated for reporting
                    self.trade_history = self.portfolio.get_trade_history()
                    
                # Record portfolio state
                self.portfolio_history.append({
                    'timestamp': current_time,
                    'equity': self.portfolio.equity,
                    'cash': self.portfolio.cash,
                    'positions': self.portfolio.get_position_snapshot()
                })
                
                logger.info(f"Order executed: {symbol} {direction * quantity} @ {price}")
                
                # Create and emit fill event
                fill_data = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'direction': direction,
                    'timestamp': current_time,
                }
                
                fill_event = Event(EventType.FILL, fill_data)
                
                if self.event_bus:
                    self.event_bus.emit(fill_event)
                    logger.info(f"Fill event emitted: {fill_event}")
            else:
                logger.warning(f"Order execution failed: {symbol} {direction * quantity} @ {price}")
                
        except Exception as e:
            logger.error(f"Error processing order: {e}", exc_info=True)
    
    def get_trade_history(self):
        """Get the trade history."""
        if isinstance(self.portfolio, SimplePortfolio):
            return self.portfolio.get_trade_history()
        return self.trade_history
    
    def get_portfolio_history(self):
        """Get the portfolio history."""
        return self.portfolio_history
    
    def reset(self):
        """Reset the execution engine."""
        self.portfolio = SimplePortfolio(self.initial_capital)
        self.pending_orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.signal_history = []
        self.last_known_prices = {}

# Class-based signal handler that maintains references
class SignalHandler:
    def __init__(self, execution_engine, position_manager, event_bus):
        self.execution_engine = execution_engine
        self.position_manager = position_manager
        self.event_bus = event_bus
        self.signals_processed = 0
        self.orders_created = 0
        logger.info("SignalHandler initialized")
        
    def handle_signal(self, event):
        """Process a signal event and create an order."""
        self.signals_processed += 1
        logger.info(f"SignalHandler processing signal #{self.signals_processed}: {event}")
        
        # Extract signal
        if not hasattr(event, 'data') or not isinstance(event.data, SignalEvent):
            logger.error(f"Expected SignalEvent in event.data, got {type(event.data) if hasattr(event, 'data') else 'None'}")
            return
            
        signal = event.data
        logger.info(f"Processing signal: {signal.get_signal_name()} for {signal.get_symbol()} @ {signal.get_price():.2f}")
        
        # Skip neutral signals
        if signal.get_signal_value() == SignalEvent.NEUTRAL:
            logger.info("Skipping neutral signal")
            return
        
        # Calculate position size if position manager available
        position_size = 100  # Default
        if self.position_manager:
            try:
                portfolio = self.execution_engine.portfolio
                position_size = self.position_manager.calculate_position_size(signal, portfolio)
                logger.info(f"Calculated position size: {position_size}")
            except Exception as e:
                logger.error(f"Error calculating position size: {e}")
        
        # Skip if position size is zero
        if position_size == 0:
            logger.info("Skipping signal due to zero position size")
            return
        
        try:
            # Create order event
            order = OrderEvent(
                symbol=signal.get_symbol(),
                direction=signal.get_signal_value(),
                quantity=abs(position_size),
                price=signal.get_price(),
                order_type="MARKET",
                timestamp=signal.timestamp
            )
            
            self.orders_created += 1
            logger.info(f"Created order #{self.orders_created}: {order}")
            
            # Emit order event
            order_event = Event(EventType.ORDER, order)
            self.event_bus.emit(order_event)
            logger.info(f"Emitted order event: {order_event}")
            
            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}", exc_info=True)
            return None

# Simple strategy class
class SimpleStrategy:
    def __init__(self, rule, event_bus=None):
        self.rule = rule
        self.event_bus = event_bus
        if event_bus:
            self.rule.set_event_bus(event_bus)
        self.signals_generated = 0
        logger.info(f"SimpleStrategy initialized with rule: {rule}")
    
    def set_event_bus(self, event_bus):
        """Set the event bus on the strategy and rule."""
        self.event_bus = event_bus
        self.rule.set_event_bus(event_bus)
        logger.info(f"Event bus set on strategy and rule: {event_bus}")
    
    def on_bar(self, event):
        """Process a bar event and generate signals."""
        logger.debug(f"Strategy received bar: {event}")
        # Just delegate to rule and count signals
        signal = self.rule.on_bar(event)
        if signal:
            self.signals_generated += 1
            logger.info(f"Strategy generated signal #{self.signals_generated}: {signal}")
        return signal
    
    def reset(self):
        """Reset the strategy."""
        self.rule.reset()
        self.signals_generated = 0

# Fixed backtester class
class FixedBacktester:
    def __init__(self, config, data_handler, strategy, position_manager=None):
        """Initialize the backtester properly."""
        self.config = config
        self.data_handler = data_handler
        self.strategy = strategy
        self.position_manager = position_manager
        
        # Initialize event bus
        self.event_bus = EventBus()
        logger.info(f"Created event bus: {self.event_bus}")
        
        # Get initial capital from config
        self.initial_capital = self.config.get('backtester', {}).get('initial_capital', 100000)
        
        # Create execution engine
        self.execution_engine = CustomExecutionEngine(
            position_manager=self.position_manager,
            initial_capital=self.initial_capital
        )
        self.execution_engine.event_bus = self.event_bus
        
        # Create signal handler (crucial fix)
        self.signal_handler = SignalHandler(
            execution_engine=self.execution_engine,
            position_manager=self.position_manager,
            event_bus=self.event_bus
        )
        
        # Set event bus on strategy
        self.strategy.set_event_bus(self.event_bus)
        
        # Set up event handlers with proper registration
        self._setup_event_handlers()
        
        # Tracking
        self.signals = []
        self.orders = []
        logger.info(f"FixedBacktester initialized with initial capital: ${self.initial_capital}")
    
    def _setup_event_handlers(self):
        """Register event handlers properly with strong references."""
        # Strategy processes bar events
        self.event_bus.register(EventType.BAR, self.strategy.on_bar)
        
        # Signal handler processes signal events
        self.event_bus.register(EventType.SIGNAL, self.signal_handler.handle_signal)
        
        # Execution engine processes order events
        self.event_bus.register(EventType.ORDER, self.execution_engine.on_order)
        
        logger.info(f"Registered event handlers: {self.event_bus.handlers}")
    
    def run(self, use_test_data=False):
        """Run the backtest."""
        logger.info(f"Starting backtest with {'test' if use_test_data else 'training'} data")
        
        # Choose data set
        data_iterator = self.data_handler.iter_test() if use_test_data else self.data_handler.iter_train()
        
        # Process each bar
        for i, bar in enumerate(data_iterator):
            # Convert bar to BarEvent if needed
            if not isinstance(bar, BarEvent):
                bar_event = BarEvent(bar)
            else:
                bar_event = bar
            
            # Create and emit bar event
            event = Event(EventType.BAR, bar_event)
            
            if i % 100 == 0:
                logger.info(f"Processing bar {i}: {bar_event.get_symbol()} @ {bar_event.get_timestamp()}")
                
            # Emit the event (this triggers the entire event chain)
            self.event_bus.emit(event)
            
            # Update the portfolio with current prices
            self.execution_engine.portfolio.mark_to_market(bar_event)
            
            # Record portfolio state periodically
            if i % 50 == 0:
                self.execution_engine.portfolio_history.append({
                    'timestamp': bar_event.get_timestamp(),
                    'equity': self.execution_engine.portfolio.equity,
                    'cash': self.execution_engine.portfolio.cash,
                    'positions': self.execution_engine.portfolio.get_position_snapshot()
                })
        
        # Collect and return results
        return self.collect_results()
    
    def collect_results(self):
        """Collect results from the backtest."""
        logger.info("Collecting backtest results")
        
        # Get trade history
        trade_history = self.execution_engine.get_trade_history()
        
        # Get portfolio history
        portfolio_history = self.execution_engine.get_portfolio_history()
        
        # Calculate basic metrics
        num_trades = len(trade_history)
        total_return = sum(t.get('percent_return', 0) for t in trade_history) if trade_history else 0
        
        # Calculate win rate
        if trade_history:
            winning_trades = [t for t in trade_history if t.get('profit', 0) > 0]
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        else:
            win_rate = 0
        
        # Calculate final equity
        initial_equity = self.initial_capital
        final_equity = portfolio_history[-1]['equity'] if portfolio_history else initial_equity
        
        # Return results
        results = {
            'trade_history': trade_history,
            'portfolio_history': portfolio_history,
            'num_trades': num_trades,
            'total_return': total_return,
            'win_rate': win_rate,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'return_pct': ((final_equity / initial_equity) - 1) * 100 if initial_equity > 0 else 0,
            'signals_generated': self.strategy.signals_generated,
            'orders_created': self.signal_handler.orders_created
        }
        
        logger.info(f"Backtest results: {num_trades} trades, {total_return:.2f}% return")
        return results
    
    def reset(self):
        """Reset all components."""
        self.strategy.reset()
        self.execution_engine.reset()
        self.signals = []
        self.orders = []

# Create helpers for starting the system
def create_trading_system(fixed_size=100, initial_capital=100000):
    """Create a complete trading system with all components."""
    # Ensure uuid is imported in event_types
    import src.events.event_types
    src.events.event_types.uuid = uuid
    
    # Create event bus
    event_bus = EventBus()
    
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
    
    # Create position manager
    position_manager = SimplePositionManager(fixed_size=fixed_size)
    
    # Create data handler
    data_source = CSVDataSource("./data")
    data_handler = DataHandler(data_source)
    
    # Create configuration
    config = {
        'backtester': {
            'initial_capital': initial_capital,
            'market_simulation': {
                'slippage_model': 'fixed',
                'slippage_bps': 5,  # 5 basis points
                'fee_model': 'fixed',
                'fee_bps': 10  # 10 basis points
            }
        }
    }
    
    # Create fixed backtester
    backtester = FixedBacktester(
        config=config,
        data_handler=data_handler,
        strategy=strategy,
        position_manager=position_manager
    )
    
    return backtester, data_handler

def run_backtest(start_date=None, end_date=None, symbols=None, timeframe="1m"):
    """Run a complete backtest."""
    # Set default dates if not provided
    if start_date is None:
        start_date = datetime.datetime(2024, 3, 26)
    if end_date is None:
        end_date = datetime.datetime(2024, 3, 27)
    if symbols is None:
        symbols = ["SPY"]
    
    # Create system
    backtester, data_handler = create_trading_system()
    
    # Load data
    logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    data_handler.load_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Run backtest
    logger.info("Running backtest")
    results = backtester.run()
    
    # Show results
    display_results(results)
    
    return results

def display_results(results):
    """Display backtest results."""
    logger.info("=== BACKTEST RESULTS ===")
    logger.info(f"Signals generated: {results['signals_generated']}")
    logger.info(f"Orders created: {results['orders_created']}")
    logger.info(f"Trades executed: {results['num_trades']}")
    
    if results['num_trades'] > 0:
        logger.info(f"Win rate: {results['win_rate']:.2%}")
        
        # Display a few sample trades
        if len(results['trade_history']) > 0:
            logger.info("Sample trades:")
            sample_size = min(3, len(results['trade_history']))
            for i in range(sample_size):
                trade = results['trade_history'][i]
                logger.info(f"Trade {i+1}: {trade.get('direction', 0) > 0 and 'BUY' or 'SELL'} {trade.get('symbol', '')} "
                          f"@ {trade.get('entry_price', 0):.2f} -> {trade.get('exit_price', 0):.2f}, "
                          f"Profit: ${trade.get('profit', 0):.2f}")
    
    logger.info(f"Initial equity: ${results['initial_equity']:.2f}")
    logger.info(f"Final equity: ${results['final_equity']:.2f}")
    logger.info(f"Return: {results['return_pct']:.2%}")
    logger.info("=======================")

# Main entry point
if __name__ == "__main__":
    run_backtest()
