#!/usr/bin/env python3
# main.py - Main entry point for the algorithmic trading system

import datetime
import os
import logging
import pandas as pd
import numpy as np

# Import system components
from src.events import EventBus, Event, EventType, LoggingHandler, MarketDataEmitter, EventManager

# Import configuration
from src.config import ConfigManager

# Import data handling
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource

# Import rules
from src.rules import create_rule
# Import our custom AlwaysBuyRule
from always_buy_rule import AlwaysBuyRule

# Import strategies
from src.strategies import WeightedStrategy

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
        log_handler = LoggingHandler([EventType.BAR, EventType.SIGNAL, 
                                    EventType.ORDER, EventType.FILL,
                                    EventType.MARKET_OPEN, EventType.MARKET_CLOSE])
        for event_type in [EventType.BAR, EventType.SIGNAL, EventType.ORDER, EventType.FILL,
                          EventType.MARKET_OPEN, EventType.MARKET_CLOSE]:
            event_bus.register(event_type, log_handler)
        
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
        
        # Create AlwaysBuyRule directly instead of using create_rule
        always_buy_rule = AlwaysBuyRule(
            name="always_buy_debug",
            params={
                'frequency': 10,  # Generate a signal every 10 bars
                'confidence': 0.9  # High confidence for testing
            },
            description="Debug rule that always generates buy signals"
        )
        
        # Create SMA crossover rule as a backup
        try:
            sma_rule = create_rule('SMAcrossoverRule', {
                'fast_window': 5,  # Smaller window to be more sensitive
                'slow_window': 15,
                'smooth_signals': True  # Generate signals when MAs are aligned
            })
            rule_components = [always_buy_rule, sma_rule]
            rule_weights = [0.7, 0.3]  # Weight heavily towards the always buy rule
        except Exception as e:
            logger.warning(f"Could not create SMAcrossoverRule: {e}. Using only AlwaysBuyRule.")
            rule_components = [always_buy_rule]
            rule_weights = [1.0]
        
        # Create strategy with the AlwaysBuyRule
        strategy = WeightedStrategy(
            components=rule_components,
            weights=rule_weights,
            buy_threshold=0.5,  # Lower threshold to ensure signals trigger actions
            sell_threshold=-0.5,
            name="DebugStrategy"
        )
        
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
        
        # 8. Create event manager
        logger.info("Creating event manager")
        event_manager = EventManager(
            event_bus=event_bus,
            strategy=strategy,
            position_manager=position_manager,
            execution_engine=execution_engine,
            portfolio=portfolio
        )
        
        # Initialize the system
        event_manager.initialize()
        
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
        
        # 10. Process market data through the event manager
        logger.info("Processing market data")
        
        # Emit market open event - FIXED: Create Event object
        market_open_event = Event(EventType.MARKET_OPEN, {'timestamp': start_date})
        event_bus.emit(market_open_event)
        
        # Process all bars
        count = 0
        for bar in data_handler.iter_train():
            # Process the bar through the event manager
            event_manager.process_market_data(bar)
            count += 1
            
            # Print status periodically
            if count % 20 == 0:
                logger.info(f"Processed {count} bars of data")
                
                # Log rule state periodically if the rule has a get_state method
                if hasattr(always_buy_rule, 'get_state'):
                    try:
                        rule_state = always_buy_rule.get_state()
                        logger.info(f"Rule state: {rule_state}")
                    except Exception as e:
                        logger.warning(f"Could not get rule state: {str(e)}")
                
        # Emit market close event - FIXED: Create Event object
        market_close_event = Event(EventType.MARKET_CLOSE, {'timestamp': end_date})
        event_bus.emit(market_close_event)
        
        # 11. Display results
        logger.info("Backtest complete")
        
        # Get system status
        status = event_manager.get_status()
        logger.info(f"System status: {status}")
        
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
            "signals_generated": status.get('signals_generated', 0),
            "orders_generated": status.get('orders_generated', 0),
            "fills_processed": status.get('fills_processed', 0),
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
