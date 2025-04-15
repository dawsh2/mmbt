#!/usr/bin/env python3
# main.py - Main entry point for the algorithmic trading system

import datetime
import os
import logging
import pandas as pd
import numpy as np

# Import system components
from src.events import EventBus, Event, EventType
from src.events.event_handlers import LoggingHandler

# Import configuration
from src.config import ConfigManager

# Import data handling
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource

# Import our custom AlwaysBuyRule
from always_buy_rule import AlwaysBuyRule

# Import strategies
from src.strategies import WeightedStrategy
from strategy_wrapper import StrategyWrapper

# Import position management
from src.position_management.portfolio import Portfolio
from src.position_management.position_manager import PositionManager
from src.position_management.position_sizers import PercentOfEquitySizer

# Import engine components
from src.engine.execution_engine import ExecutionEngine
from src.engine.market_simulator import MarketSimulator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None):
    """Create synthetic price data for testing."""
    # Create a synthetic dataset
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    # Create a price series with some randomness and a trend
    base_price = 100
    prices = [base_price]
    
    # Add a trend with noise
    for i in range(1, len(dates)):
        # Random daily change between -1% and +1% with a slight upward bias
        daily_change = np.random.normal(0.0005, 0.01) 
        
        # Add some regime changes to test rules
        if i % 60 == 0:  # Every ~2 months, change trend
            daily_change = -0.02 if prices[-1] > base_price * 1.1 else 0.02
            
        new_price = prices[-1] * (1 + daily_change)
        prices.append(new_price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.normal(100000, 20000)) for _ in prices],
        'symbol': symbol
    })
    
    # Save to CSV if filename provided
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Created synthetic data with {len(df)} bars, saved to {filename}")
    
    return df

class EventProcessor:
    """
    Helper class to process and track events in the system.
    """
    def __init__(self, event_bus, strategy, position_manager, execution_engine, portfolio):
        self.event_bus = event_bus
        self.strategy = strategy
        self.position_manager = position_manager
        self.execution_engine = execution_engine
        self.portfolio = portfolio
        self.signal_count = 0
        self.order_count = 0
        self.fill_count = 0
        
    def process_bar(self, bar):
        """Process a bar of market data through the system."""
        # Create and emit a bar event
        bar_event = Event(EventType.BAR, bar)
        self.event_bus.emit(bar_event)
        
        # Update execution engine with latest prices
        self.execution_engine.update(bar)
        
    def handle_signal(self, event):
        """Handle a signal event."""
        self.signal_count += 1
        logger.info(f"Signal #{self.signal_count} received: {event.data.signal_type} at price {event.data.price}")
        
        # Generate an order based on the signal
        try:
            # Get signal data
            signal_data = event.data
            symbol = signal_data.metadata.get('symbol', 'SYNTHETIC')
            direction = signal_data.signal_type.value  # 1 for BUY, -1 for SELL
            price = signal_data.price
            
            # Create order data - hard-coded for testing
            order_data = {
                'symbol': symbol,
                'order_type': 'MARKET',
                'direction': direction,
                'quantity': 100,  # Fixed quantity for testing
                'price': price,
                'timestamp': signal_data.timestamp
            }
            
            # Create and emit order event
            logger.info(f"Creating order from signal: {order_data}")
            order_event = Event(EventType.ORDER, order_data)
            self.event_bus.emit(order_event)
            
            # Increment order count
            self.order_count += 1
            
        except Exception as e:
            logger.error(f"Error creating order from signal: {e}", exc_info=True)
        
        # Forward to position manager (in case forwarding in event bus fails)
        try:
            self.position_manager.on_signal(event)
        except Exception as e:
            logger.error(f"Error in position_manager.on_signal: {e}")
    
    def handle_order(self, event):
        """Handle an order event."""
        self.order_count += 1
        logger.info(f"Order #{self.order_count} received: {event.data}")
        
        # Forward to execution engine (in case forwarding in event bus fails)
        try:
            self.execution_engine.on_order(event)
        except Exception as e:
            logger.error(f"Error in execution_engine.on_order: {e}")
            
        # Directly create a fill for testing
        try:
            # Get order data
            order_data = event.data
            symbol = order_data.get('symbol', 'SYNTHETIC')
            direction = order_data.get('direction', 0)
            quantity = order_data.get('quantity', 100)
            price = order_data.get('price', 0)
            
            # Create fill data - hard-coded for testing
            fill_data = {
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'timestamp': order_data.get('timestamp', datetime.datetime.now()),
                'commission': 1.0,
                'slippage': 0.01
            }
            
            # Create and emit fill event
            logger.info(f"Creating fill from order: {fill_data}")
            fill_event = Event(EventType.FILL, fill_data)
            self.event_bus.emit(fill_event)
            
        except Exception as e:
            logger.error(f"Error creating fill from order: {e}", exc_info=True)
    
    def handle_fill(self, event):
        """Handle a fill event."""
        self.fill_count += 1
        logger.info(f"Fill #{self.fill_count} received: {event.data}")
        
        # Update portfolio based on fill
        try:
            # Assuming on_fill is available on portfolio
            if hasattr(self.portfolio, 'on_fill'):
                self.portfolio.on_fill(event)
                logger.info("Called portfolio.on_fill")
            else:
                logger.warning("Portfolio does not have on_fill method")
                
                # Try to update portfolio manually
                fill_data = event.data
                if isinstance(fill_data, dict):
                    symbol = fill_data.get('symbol')
                    direction = fill_data.get('direction', 0)
                    quantity = fill_data.get('quantity', 0) 
                    price = fill_data.get('price', 0)
                    timestamp = fill_data.get('timestamp', datetime.datetime.now())
                    
                    if direction > 0:
                        logger.info(f"Opening long position: {quantity} shares of {symbol} at {price}")
                        position = self.portfolio.open_position(symbol, 1, quantity, price, timestamp)
                        logger.info(f"Created position: {position}")
                    elif direction < 0:
                        logger.info(f"Opening short position: {quantity} shares of {symbol} at {price}")
                        position = self.portfolio.open_position(symbol, -1, quantity, price, timestamp)
                        logger.info(f"Created position: {position}")
                    else:
                        logger.warning("Fill direction is 0, cannot create position")
                else:
                    logger.warning(f"Fill data is not a dictionary: {fill_data}")
                
        except Exception as e:
            logger.error(f"Error handling fill: {e}", exc_info=True)
    
    def register_handlers(self):
        """Register event handlers with the event bus."""
        # Register this instance's handler methods
        self.event_bus.register(EventType.SIGNAL, self.handle_signal)
        self.event_bus.register(EventType.ORDER, self.handle_order)
        self.event_bus.register(EventType.FILL, self.handle_fill)
        
        logger.info("Event processor handlers registered")
    
    def get_status(self):
        """Get the current status of the event processor."""
        return {
            'signals_generated': self.signal_count,
            'orders_generated': self.order_count,
            'fills_processed': self.fill_count
        }

def main():
    """Main entry point for the trading system."""
    logger.info("Starting the trading system")
    
    # Initialize result with default values
    result = {
        "status": "error",
        "message": "Unknown error",
        "bars_processed": 0,
        "signals_generated": 0,
        "orders_generated": 0,
        "fills_processed": 0,
        "final_equity": 0,
        "total_return": 0
    }
    
    try:
        # 1. Initialize the event system
        logger.info("Initializing event system")
        event_bus = EventBus()
        
        # Add a logging handler for debugging
        log_handler = LoggingHandler([EventType.MARKET_OPEN, EventType.MARKET_CLOSE])
        event_bus.register(EventType.MARKET_OPEN, log_handler)
        event_bus.register(EventType.MARKET_CLOSE, log_handler)
        
        # 2. Create and configure the config manager
        logger.info("Loading configuration")
        config = ConfigManager()
        
        # Set default configuration values
        config.set('backtester.initial_capital', 100000)
        config.set('backtester.market_simulation.slippage_model', 'fixed')
        config.set('backtester.market_simulation.slippage_bps', 5)
        config.set('position_management.position_sizing.method', 'percent_equity')
        config.set('position_management.position_sizing.percent', 0.02)
        
        # 3. Set up market data
        symbol = "SYNTHETIC"
        timeframe = "1d"
        filename = f"{symbol}_{timeframe}.csv"
        
        # Check if we need to create synthetic data
        if not os.path.exists(filename):
            create_synthetic_data(symbol, timeframe, filename)
        
        # 4. Set up data sources and handler
        logger.info("Setting up data handler")
        data_source = CSVDataSource(".")  # Look for CSV files in current directory
        data_handler = DataHandler(data_source)
        
        # 5. Create trading strategy with rules
        logger.info("Setting up trading strategy")
        
        # Create our AlwaysBuyRule (already has on_bar method)
        always_buy_rule = AlwaysBuyRule(
            name="always_buy_debug",
            params={
                'frequency': 2,  # Generate a signal every 2 bars
                'confidence': 1.0  # Full confidence for testing
            },
            description="Debug rule that always generates buy signals"
        )
        
        # Test the AlwaysBuyRule directly before using it
        logger.info("Directly testing AlwaysBuyRule...")
        test_bar = {
            "timestamp": datetime.datetime.now(),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 10000,
            "symbol": "TEST"
        }
        test_event = Event(EventType.BAR, test_bar)
        
        # Call the rule directly
        test_signal = always_buy_rule.on_bar(test_event)
        logger.info(f"Direct test of AlwaysBuyRule result: {test_signal}")
        logger.info(f"Rule state after direct test: {always_buy_rule.get_state()}")
        
        # Create strategy with our rule
        strategy = WeightedStrategy(
            components=[always_buy_rule],
            weights=[1.0],  # Only one rule, so full weight
            buy_threshold=0.5,
            sell_threshold=-0.5,
            name="DebugStrategy"
        )
        
        # Wrap the strategy to handle events properly
        wrapped_strategy = StrategyWrapper(strategy)
        
        # 6. Set up portfolio and position manager
        logger.info("Setting up portfolio and position management")
        
        # Create portfolio
        portfolio = Portfolio(initial_capital=config.get('backtester.initial_capital'))
        
        # Create position sizer based on config
        position_sizer = PercentOfEquitySizer(
            percent=config.get('position_management.position_sizing.percent')
        )
        
        # Create position manager
        position_manager = PositionManager(
            portfolio=portfolio,
            position_sizer=position_sizer
        )
        
        # 7. Set up execution engine and market simulator
        logger.info("Setting up execution engine")
        
        # Create market simulator for backtesting
        market_simulator = MarketSimulator({
            'slippage_model': config.get('backtester.market_simulation.slippage_model'),
            'slippage_bps': config.get('backtester.market_simulation.slippage_bps')
        })
        
        # Create execution engine
        execution_engine = ExecutionEngine(position_manager)
        execution_engine.market_simulator = market_simulator
        
        # 8. Create event processor
        logger.info("Creating event processor")
        event_processor = EventProcessor(
            event_bus=event_bus,
            strategy=wrapped_strategy,
            position_manager=position_manager,
            execution_engine=execution_engine,
            portfolio=portfolio
        )
        
        # Register event handlers
        logger.info("Registering event handlers")
        # Register the strategy to handle bar events
        event_bus.register(EventType.BAR, wrapped_strategy.on_bar)
        # Register other handlers via the event processor
        event_processor.register_handlers()
        
        # 9. Load market data
        logger.info("Loading market data")
        start_date = datetime.datetime(2022, 1, 1)
        end_date = datetime.datetime(2022, 12, 31)
        
        # Load data using the DataHandler
        data_handler.load_data(
            symbols=[symbol],  # DataHandler expects a list
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        )
        
        logger.info(f"Successfully loaded {len(data_handler.train_data)} bars of data")
        
        # 10. Process market data through the event system
        logger.info("Processing market data")
        
        # Emit market open event
        market_open_event = Event(EventType.MARKET_OPEN, {'timestamp': start_date})
        event_bus.emit(market_open_event)
        
        # Process all bars
        count = 0
        for bar in data_handler.iter_train():
            # Process the bar through the event processor
            # Create bar event
            bar_event = Event(EventType.BAR, bar)
            
            # DIRECTLY call the rule's on_bar method
            logger.info(f"Directly calling AlwaysBuyRule.on_bar for bar {count}")
            signal = always_buy_rule.on_bar(bar_event)
            
            # If we got a signal, emit it
            if signal is not None:
                logger.info(f"Signal generated directly: {signal.signal_type}")
                
                # Create signal event
                signal_event = Event(EventType.SIGNAL, signal)
                
                # Emit the signal event
                event_bus.emit(signal_event)
                
            event_processor.process_bar(bar)
            count += 1
            
            # Print status periodically
            if count % 10 == 0:
                logger.info(f"Processed {count} bars of data")
                
                # Log rule state periodically
                rule_state = always_buy_rule.get_state()
                logger.info(f"Rule state: {rule_state}")
                
                # Log portfolio state
                portfolio_metrics = portfolio.get_performance_metrics()
                logger.info(f"Current equity: ${portfolio_metrics['current_equity']:.2f}")
                logger.info(f"Open positions: {len(portfolio.positions)}")
                
                # Log event stats
                event_stats = event_processor.get_status()
                logger.info(f"Event stats: Signals={event_stats['signals_generated']}, " + 
                           f"Orders={event_stats['orders_generated']}, " +
                           f"Fills={event_stats['fills_processed']}")
                
                # Log open positions details
                if portfolio.positions:
                    logger.info("Open positions:")
                    for pos_id, position in portfolio.positions.items():
                        logger.info(f"  - {position.symbol}: {position.quantity} shares at {position.entry_price:.2f}")
        
        # Emit market close event
        market_close_event = Event(EventType.MARKET_CLOSE, {'timestamp': end_date})
        event_bus.emit(market_close_event)
        
        # 11. Display results
        logger.info("Backtest complete")
        
        # Get event processor status
        status = event_processor.get_status()
        logger.info(f"Event processor status: {status}")
        
        # Calculate portfolio performance
        final_equity = portfolio.get_performance_metrics()['current_equity']
        initial_equity = config.get('backtester.initial_capital')
        total_return = (final_equity / initial_equity - 1) * 100
        
        logger.info(f"Initial equity: ${initial_equity:.2f}")
        logger.info(f"Final equity: ${final_equity:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Update result with success values
        result.update({
            "status": "backtest_complete", 
            "bars_processed": count,
            "signals_generated": status['signals_generated'],
            "orders_generated": status['orders_generated'],
            "fills_processed": status['fills_processed'],
            "final_equity": final_equity,
            "total_return": total_return
        })
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        result["message"] = str(e)
    
    return result

if __name__ == "__main__":
    try:
        result = main()
        print("\nScript executed successfully!")
        print(f"Status: {result['status']}")
        print(f"Bars processed: {result['bars_processed']}")
        print(f"Signals generated: {result['signals_generated']}")
        print(f"Orders generated: {result['orders_generated']}")
        print(f"Fills processed: {result['fills_processed']}")
        print(f"Final equity: ${result['final_equity']:.2f}")
        print(f"Total return: {result['total_return']:.2f}%")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
