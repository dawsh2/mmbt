#!/usr/bin/env python
"""
Debugging script for running a basic backtest.
This script initializes all necessary components and runs a simple backtest
with detailed logging to help diagnose issues.
"""

import logging
import datetime
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from inspect import signature
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backtest_debug")
logger.info("Starting backtest debugging script")

# Import required components
try:
    from src.config import ConfigManager
    from src.data.data_sources import CSVDataSource
    from src.data.data_handler import DataHandler
    from src.rules import create_rule
    from src.strategies import WeightedStrategy
    from src.signals import Signal, SignalType
    from src.engine import Backtester
    
    logger.info("Successfully imported all components")
except ImportError as e:
    logger.error(f"Failed to import required components: {e}")
    logger.error("Please check your PYTHONPATH and ensure all modules are installed")
    raise

# Create configuration
try:
    config = ConfigManager()
    config.set('backtester.initial_capital', 100000)
    config.set('backtester.market_simulation.slippage_model', 'fixed')
    config.set('backtester.market_simulation.slippage_bps', 5)
    config.set('backtester.market_simulation.fee_bps', 10)
    logger.info("Successfully created configuration")
except Exception as e:
    logger.error(f"Failed to create configuration: {e}")
    raise

# Create data source and handler
try:
    # For debugging, let's create a simple synthetic dataset
    logger.info("Creating synthetic data for testing")
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    # Create a sawtooth pattern to ensure multiple crossovers
    prices = []
    for i in range(len(dates)):
        # Create a price that oscillates to ensure crossovers
        if i % 20 < 10:
            prices.append(100 + i % 20)  # Rising
        else:
            prices.append(110 - (i % 20 - 10))  # Falling
    
    data = {
        'timestamp': dates,
        'Open': prices,
        'High': [p + 1 for p in prices],
        'Low': [p - 1 for p in prices],
        'Close': [p + 0.5 for p in prices],
        'Volume': [10000 for _ in prices]
    }
    
    df = pd.DataFrame(data)
    
    # Create a single CSV file with the CORRECT filename pattern
    symbol = "SYNTHETIC"
    timeframe = "1d"
    filename = f"{symbol}_{timeframe}.csv"
    df.to_csv(filename, index=False)
    logger.info(f"Saved synthetic data to {os.path.abspath(filename)}")
    
    # Create data source
    data_source = CSVDataSource(".")  # Current directory
    
    # Verify the file exists
    expected_file = f"{symbol}_{timeframe}.csv"
    logger.info(f"Looking for file: {expected_file}")
    logger.info(f"File exists: {os.path.exists(expected_file)}")
    
    # Create data handler
    data_handler = DataHandler(data_source)
    
    # Load data using the correct method signature
    # IMPORTANT: symbols parameter takes a LIST of strings
    data_handler.load_data(
        symbols=[symbol],  # This is a LIST with one symbol
        start_date=dates[0],
        end_date=dates[-1],
        timeframe=timeframe
    )
    logger.info(f"Successfully created data handler and loaded {len(data_handler.full_data)} rows of data")
except Exception as e:
    logger.error(f"Failed to create data source/handler: {e}")
    raise


# Create rule and strategy
try:
    # Create a more sensitive crossover rule
    sma_rule = create_rule('SMAcrossoverRule', {
        'fast_window': 5,  # Faster
        'slow_window': 15,  # Faster
        'smooth_signals': False
    })
    
    # Make strategy more sensitive
    strategy = WeightedStrategy(
        components=[sma_rule],
        weights=[1.0],
        buy_threshold=0.0001,  # Extremely low threshold
        sell_threshold=-0.0001,  # Extremely low threshold
        name="Debug_SMA_Strategy"
    )
    logger.info("Successfully created rule and strategy")
except Exception as e:
    logger.error(f"Failed to create rule/strategy: {e}")
    raise

# After loading data but before running the backtest
# Test if the rule can generate signals on some data
logger.info("Testing signal generation on sample data")
sample_idx = 50
if len(data_handler.full_data) > sample_idx:
    sample_bar = data_handler.full_data.iloc[sample_idx].to_dict()
    logger.info(f"Sample bar: {sample_bar}")
    
    # Test rule directly
    rule_signal = sma_rule.on_bar(sample_bar)
    logger.info(f"Rule signal: {rule_signal}")
    
    # Create a BarEvent wrapper for testing the strategy
    class BarEvent:
        def __init__(self, bar):
            self.bar = bar
    
    # Test strategy
    strategy_signal = strategy.on_bar(BarEvent(sample_bar))
    logger.info(f"Strategy signal: {strategy_signal}")

try:
    # Create backtester
    backtester = Backtester(config, data_handler, strategy)
    logger.info("Successfully created backtester")
    
    # Run backtest without the verbose parameter (it's not supported)
    logger.info("Running backtest...")
    results = backtester.run(use_test_data=False)
    
    # Check if results is None
    if results is None:
        logger.error("Backtest returned None instead of results dictionary")
        results = {
            'total_percent_return': 0,
            'num_trades': 0,
            'trades': [],
            'portfolio_history': [],
            'signals': []
        }
    
    # Log results
    logger.info("Backtest completed")
    logger.info(f"Total Return: {results.get('total_percent_return', 0):.2f}%")
    logger.info(f"Number of Trades: {results.get('num_trades', 0)}")
    
    # Check for error
    if 'error' in results:
        logger.error(f"Backtest encountered an error: {results['error']}")
    
    # Log all trades for analysis
    if 'trades' in results and results['trades']:
        logger.info("Trade history:")
        for i, trade in enumerate(results['trades'], 1):
            if len(trade) >= 6:
                entry_time, direction, entry_price, exit_time, exit_price, pnl = trade[:6]
                logger.info(f"Trade {i}: {'BUY' if direction > 0 else 'SELL'} at {entry_price:.2f} on {entry_time}, "
                         f"exit at {exit_price:.2f} on {exit_time}, P&L: {pnl:.2f}%")
            else:
                logger.info(f"Trade {i}: {trade}")
    else:
        logger.warning("No trades were executed during the backtest")
    
    # Save results to CSV for analysis
    if 'portfolio_history' in results and results['portfolio_history']:
        df_portfolio = pd.DataFrame(results['portfolio_history'])
        df_portfolio.to_csv('portfolio_history.csv', index=False)
        logger.info("Saved portfolio history to portfolio_history.csv")
    
    # Calculate additional metrics if available
    try:
        sharpe = backtester.calculate_sharpe()
        logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate Sharpe ratio: {e}")
    
except Exception as e:
    logger.error(f"Failed to run backtest: {e}")
    # Print detailed exception info
    import traceback
    logger.error(traceback.format_exc())
    raise


# Add this check
if results.get('signals'):
    logger.info(f"Signals were generated! Count: {len(results['signals'])}")
    for i, signal in enumerate(results['signals'][:5]):
        logger.info(f"Signal {i+1}: {signal}")
        if hasattr(signal, 'signal_type'):
            logger.info(f"Signal type: {signal.signal_type}")
        if hasattr(signal, 'confidence'):
            logger.info(f"Confidence: {signal.confidence}")
else:
    logger.info("No signals were generated by the strategy")

# Check the orders
if hasattr(backtester, 'orders') and backtester.orders:
    logger.info(f"Orders were created! Count: {len(backtester.orders)}")
    for i, order in enumerate(backtester.orders[:5]):
        logger.info(f"Order {i+1}: {order}")
else:
    logger.info("No orders were created!")

# Check position manager
logger.info(f"Position manager type: {type(backtester.position_manager)}")
if hasattr(backtester.position_manager, 'default_size'):
    logger.info(f"Default position size: {backtester.position_manager.default_size}")


logger.info("=== DETAILED DEBUGGING INFORMATION ===")

# 1. Check the data handler's data
logger.info(f"Data handler has {len(data_handler.full_data)} bars of data")
logger.info(f"First few bars: {data_handler.full_data.head().to_dict()}")

# 2. Manually check for SMA crossovers
logger.info("Calculating SMA values to check for crossovers")
# Calculate the SMAs
import pandas as pd
import numpy as np

df = data_handler.full_data.copy()
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_15'] = df['Close'].rolling(window=15).mean()
df['Crossover'] = np.where(df['SMA_5'] > df['SMA_15'], 1, -1)
df['Signal'] = df['Crossover'].diff()

# Check for crossover points
crossovers = df[df['Signal'] != 0].dropna()
logger.info(f"Found {len(crossovers)} SMA crossover points in the data")
if not crossovers.empty:
    logger.info(f"Crossover points: {crossovers[['timestamp', 'Close', 'SMA_5', 'SMA_15', 'Signal']].head().to_dict()}")
else:
    logger.info("No SMA crossovers found in the data - this might explain the lack of trades")

# 3. Check strategy and rule in detail
logger.info("Testing SMA rule on multiple bars")

# Test the rule on several bars
test_indices = [40, 50, 60, 70, 80]  # Test multiple points

for idx in test_indices:
    if idx < len(data_handler.full_data):
        sample_bar = data_handler.full_data.iloc[idx].to_dict()
        logger.info(f"\nTesting with bar at index {idx}, date {sample_bar.get('timestamp')}")
        
        # Test rule directly
        try:
            rule_signal = sma_rule.on_bar(sample_bar)
            logger.info(f"Rule generated signal: {rule_signal}")
            
            # Check rule internals if possible
            if hasattr(sma_rule, 'state'):
                logger.info(f"Rule state: {sma_rule.state}")
            
            # If Signal object has attributes we can inspect
            if hasattr(rule_signal, 'signal_type'):
                logger.info(f"Signal type: {rule_signal.signal_type}")
            if hasattr(rule_signal, 'confidence'):
                logger.info(f"Signal confidence: {rule_signal.confidence}")
                
        except Exception as e:
            logger.error(f"Error testing rule: {e}")
            
        # Test strategy with the same bar
        try:
            class BarEvent:
                def __init__(self, bar):
                    self.bar = bar
            
            strategy_signal = strategy.on_bar(BarEvent(sample_bar))
            logger.info(f"Strategy generated signal: {strategy_signal}")
            
            # If Signal object has attributes we can inspect
            if hasattr(strategy_signal, 'signal_type'):
                logger.info(f"Strategy signal type: {strategy_signal.signal_type}")
            if hasattr(strategy_signal, 'confidence'):
                logger.info(f"Strategy signal confidence: {strategy_signal.confidence}")
                
        except Exception as e:
            logger.error(f"Error testing strategy: {e}")

# 4. Check the backtester event handling
logger.info("\nChecking backtester signal processing")
try:
    # Get the event bus if it exists
    if hasattr(backtester, 'event_bus'):
        logger.info(f"Event bus: {backtester.event_bus}")
        
    # Check execution engine
    if hasattr(backtester, 'execution_engine'):
        logger.info(f"Execution engine: {backtester.execution_engine}")
        
        # Check for pending orders
        if hasattr(backtester.execution_engine, 'get_pending_orders'):
            pending_orders = backtester.execution_engine.get_pending_orders()
            logger.info(f"Pending orders: {pending_orders}")
            
except Exception as e:
    logger.error(f"Error checking backtester: {e}")

# 5. Check thresholds and parameters
logger.info("\nChecking strategy parameters")
logger.info(f"Strategy buy threshold: {strategy.buy_threshold}")
logger.info(f"Strategy sell threshold: {strategy.sell_threshold}")
logger.info(f"Strategy weights: {strategy.weights}")

logger.info("\nChecking rule parameters")
logger.info(f"SMA Rule parameters: {sma_rule.params}")

# 6. Suggest adjustments
logger.info("\nPossible solutions to try:")
logger.info("1. Lower the strategy thresholds (e.g., buy_threshold=0.001, sell_threshold=-0.001)")
logger.info("2. Try different SMA periods (e.g., fast_window=3, slow_window=10)")
logger.info("3. Use a different rule type that might generate more signals")
logger.info("4. Check if the execution engine is properly converting signals to orders")
logger.info("5. Verify that the backtester is correctly passing events between components")

# Debug signal flow if no trades were executed
if results.get('num_trades', 0) == 0:
    logger.warning("No trades executed - debugging signal flow")
    
    # Check if signals were generated
    if 'signals' in results and results['signals']:
        logger.info(f"Found {len(results['signals'])} signals but no trades")
        logger.info("First 5 signals:")
        for i, signal in enumerate(results['signals'][:5], 1):
            logger.info(f"Signal {i}: {signal}")
        
        # The issue might be in converting signals to orders
        logger.info("Check ExecutionEngine.on_signal and execute_pending_orders methods")
    else:
        logger.warning("No signals were generated")
        logger.info("Check Strategy.on_bar and Rule.generate_signal methods")

logger.info("Backtest debugging completed")
