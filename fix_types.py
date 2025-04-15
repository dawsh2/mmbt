#!/usr/bin/env python
"""
This script fixes the EventType import issue in the backtester.py file.
"""

import os
import re
import sys

def fix_backtester_file():
    """Fix the backtester.py file to use the correct EventType import."""
    backtester_path = "src/engine/backtester.py"
    
    if not os.path.exists(backtester_path):
        print(f"ERROR: Could not find {backtester_path}")
        return False
    
    with open(backtester_path, 'r') as f:
        content = f.read()
    
    # Check if there's a local EventType definition
    local_enum_def = re.search(r'class EventType\(Enum\):', content)
    if local_enum_def:
        # Option 1: Remove the local EventType definition and use the imported one
        
        # First, make sure we have the correct import
        if "from src.events.event_types import EventType" not in content:
            # Find the imports section
            import_section = re.search(r'import.*?\n\n', content, re.DOTALL)
            if import_section:
                import_end = import_section.end()
                # Add the import right after the existing imports
                content = content[:import_end] + "from src.events.event_types import EventType\n" + content[import_end:]
            else:
                # Add import at the top
                content = "from src.events.event_types import EventType\n\n" + content
        
        # Now remove the local EventType class definition
        content = re.sub(r'class EventType\(Enum\):.*?(?=\n\n)', '', content, flags=re.DOTALL)
    
    # Write the updated content back
    with open(backtester_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {backtester_path}")
    return True

def fix_execution_engine_file():
    """Fix the execution_engine.py file to use the correct EventType import."""
    execution_engine_path = "src/engine/execution_engine.py"
    
    if not os.path.exists(execution_engine_path):
        print(f"ERROR: Could not find {execution_engine_path}")
        return False
    
    with open(execution_engine_path, 'r') as f:
        content = f.read()
    
    # Check if there's a local EventType definition
    local_enum_def = re.search(r'class EventType\(Enum\):', content)
    if local_enum_def:
        # Option 1: Remove the local EventType definition and use the imported one
        
        # First, make sure we have the correct import
        if "from src.events.event_types import EventType" not in content:
            # Find the imports section
            import_section = re.search(r'import.*?\n\n', content, re.DOTALL)
            if import_section:
                import_end = import_section.end()
                # Add the import right after the existing imports
                content = content[:import_end] + "from src.events.event_types import EventType\n" + content[import_end:]
            else:
                # Add import at the top
                content = "from src.events.event_types import EventType\n\n" + content
        
        # Now remove the local EventType class definition
        content = re.sub(r'class EventType\(Enum\):.*?(?=\n\n)', '', content, flags=re.DOTALL)
    
    # Write the updated content back
    with open(execution_engine_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {execution_engine_path}")
    return True

def fix_init_file():
    """Fix the __init__.py file in events module to ensure consistent imports."""
    init_path = "src/events/__init__.py"
    
    if not os.path.exists(init_path):
        print(f"ERROR: Could not find {init_path}")
        return False
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    # Check imports - we want to directly import EventType from event_types
    if "from src.events.event_types import EventType" not in content and "from .event_types import EventType" not in content:
        # Find where to add the import
        if "from src.events.event_bus import Event, EventBus" in content:
            content = content.replace(
                "from src.events.event_bus import Event, EventBus", 
                "from src.events.event_bus import Event, EventBus\nfrom src.events.event_types import EventType"
            )
        elif "from enum import Enum, auto" in content:
            # Remove the alternative EventType definition
            pattern = r'# Define a minimal EventType.*?__all__ ='
            match = re.search(pattern, content, re.DOTALL)
            if match:
                modified_content = content[:match.start()] + "\n# Import the real EventType\nfrom src.events.event_types import EventType\n\n__all__ ="
                content = modified_content + content[match.end():]
    
    # Make sure EventType is in __all__
    if "__all__ = [" in content and "EventType" not in content:
        content = content.replace("__all__ = [", "__all__ = [\n    'EventType',")
    
    # Write the updated content back
    with open(init_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {init_path}")
    return True

def create_basic_run_script():
    """Create a simplified basic_run.py script that should work."""
    script_path = "simple_run.py"
    
    content = """#!/usr/bin/env python
"""\"
Simplified backtesting script for debugging.
This script provides a minimal setup to run a backtest.
\"""

import logging
import pandas as pd
from datetime import datetime
import os
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backtest")

# Import required components
from src.config import ConfigManager
from src.data.data_sources import CSVDataSource
from src.data.data_handler import DataHandler
from src.rules import create_rule
from src.strategies import WeightedStrategy
from src.signals import Signal, SignalType
from src.engine import Backtester

# Create configuration
config = ConfigManager()
config.set('backtester.initial_capital', 100000)
config.set('backtester.market_simulation.slippage_model', 'fixed')
config.set('backtester.market_simulation.slippage_bps', 5)
config.set('backtester.market_simulation.fee_bps', 10)
logger.info("Created configuration")

# Create synthetic data
logger.info("Creating synthetic data")
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

# Create sawtooth price pattern to ensure multiple crossovers
prices = []
for i in range(len(dates)):
    # Price oscillates between 100 and 120
    if i % 30 < 15:
        prices.append(100 + i % 15)  # Rising
    else:
        prices.append(115 - (i % 30 - 15))  # Falling

data = {
    'timestamp': dates,
    'Open': prices,
    'High': [p + 1 for p in prices],
    'Low': [p - 1 for p in prices],
    'Close': [p + 0.5 for p in prices],
    'Volume': [10000 for _ in prices]
}

df = pd.DataFrame(data)

# Save to CSV
symbol = "SYNTHETIC"
timeframe = "1d"
filename = f"{symbol}_{timeframe}.csv"
df.to_csv(filename, index=False)
logger.info(f"Saved synthetic data to {os.path.abspath(filename)}")

# Create data source and handler
data_source = CSVDataSource(".")
data_handler = DataHandler(data_source)

# Load data
data_handler.load_data(
    symbols=[symbol],
    start_date=dates[0],
    end_date=dates[-1],
    timeframe=timeframe
)
logger.info(f"Loaded {len(data_handler.full_data)} bars of data")

# Create rule and strategy
sma_rule = create_rule('SMAcrossoverRule', {
    'fast_window': 5,
    'slow_window': 15,
    'smooth_signals': False
})

strategy = WeightedStrategy(
    components=[sma_rule],
    weights=[1.0],
    buy_threshold=0.1,
    sell_threshold=-0.1,
    name="SMA_Strategy"
)
logger.info("Created strategy")

# Create and run backtester
try:
    backtester = Backtester(config, data_handler, strategy)
    logger.info("Running backtest...")
    results = backtester.run(use_test_data=False)
    
    # Log results
    logger.info(f"Total Return: {results.get('total_percent_return', 0):.2f}%")
    logger.info(f"Number of Trades: {results.get('num_trades', 0)}")
    
    # Log all trades
    if 'trades' in results and results['trades']:
        logger.info(f"Found {len(results['trades'])} trades")
        for i, trade in enumerate(results['trades'][:5], 1):  # Show first 5 trades
            if len(trade) >= 6:
                entry_time, direction, entry_price, exit_time, exit_price, pnl = trade[:6]
                logger.info(f"Trade {i}: {'BUY' if direction > 0 else 'SELL'} at {entry_price:.2f}, "
                         f"exit at {exit_price:.2f}, P&L: {pnl:.2f}")
    else:
        logger.warning("No trades were executed")
    
    # Check signals
    if hasattr(backtester, 'signals') and backtester.signals:
        logger.info(f"Found {len(backtester.signals)} signals")
        for i, signal in enumerate(backtester.signals[:5], 1):  # Show first 5 signals
            if hasattr(signal, 'signal_type'):
                logger.info(f"Signal {i}: {signal.signal_type}")
            else:
                logger.info(f"Signal {i}: {signal}")
    else:
        logger.warning("No signals were generated")
    
    # Get Sharpe ratio
    if hasattr(backtester, 'calculate_sharpe'):
        sharpe = backtester.calculate_sharpe()
        logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    
except Exception as e:
    logger.error(f"Error running backtest: {e}", exc_info=True)
"""
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created {script_path}")
    return True

def main():
    """Main function to fix the EventType issue."""
    print("Fixing EventType import issues...")
    
    success = True
    success = fix_backtester_file() and success
    success = fix_execution_engine_file() and success
    success = fix_init_file() and success
    
    if success:
        print("\nAll files fixed successfully!")
        create_basic_run_script()
        print("\nTo run the simplified backtest script:")
        print("python simple_run.py")
    else:
        print("\nSome fixes failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
