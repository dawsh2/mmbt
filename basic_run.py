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
from src.strategies import WeightedStrategy
logging.info(f"WeightedStrategy signature: {signature(WeightedStrategy.__init__)}")


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
    prices = list(range(100, 100 + len(dates)))
    
    # Introduce a pattern for testing
    for i in range(20, 40):
        prices[i] += 20
    for i in range(60, 80):
        prices[i] -= 15
        
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
    # Create a simple SMA crossover rule
    sma_rule = create_rule('SMAcrossoverRule', {
        'fast_window': 10,
        'slow_window': 30
    })
    
    # Using components instead of rules based on the actual signature
    strategy = WeightedStrategy(
        components=[sma_rule],  # Changed from 'rules' to 'components'
        weights=[1.0],
        buy_threshold=0.1,
        sell_threshold=-0.1,
        name="Debug_SMA_Strategy"
    )
    logger.info("Successfully created rule and strategy")
except Exception as e:
    logger.error(f"Failed to create rule/strategy: {e}")
    raise


try:
    # Create backtester
    backtester = Backtester(config, data_handler, strategy)
    logger.info("Successfully created backtester")
    
    # Run backtest without the verbose parameter (it's not supported)
    logger.info("Running backtest...")
    results = backtester.run(use_test_data=False)  # Changed from verbose=True to use_test_data=False
    
    # Log results
    logger.info("Backtest completed")
    logger.info(f"Total Return: {results.get('total_percent_return', 0):.2f}%")
    logger.info(f"Number of Trades: {results.get('num_trades', 0)}")
    
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
