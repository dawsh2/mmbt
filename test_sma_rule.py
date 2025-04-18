"""
Diagnostic Script to Check SMACrossoverRule Signal Generation

This script creates a minimal test to verify that the SMACrossoverRule
creates proper SignalEvent objects.
"""

import sys
import os
import logging
import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path if needed
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required components
from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.rules.crossover_rules import SMACrossoverRule

def test_sma_rule_signal_generation():
    """Test that SMACrossoverRule generates proper SignalEvent objects"""
    
    # Create the rule
    rule = SMACrossoverRule(
        name="test_sma",
        params={
            "fast_window": 2,  # Small values for quick testing
            "slow_window": 4
        },
        description="Test SMA Crossover"
    )
    
    # Create test data
    prices = [100, 105, 110, 105, 100, 95, 100, 105]
    dates = [
        datetime.datetime(2024, 3, 1, 10, 0) + datetime.timedelta(minutes=i)
        for i in range(len(prices))
    ]
    
    # Create bar data dictionaries
    bars = []
    for i, price in enumerate(prices):
        bars.append({
            'timestamp': dates[i],
            'symbol': 'TEST',
            'Open': price - 1,
            'High': price + 1,
            'Low': price - 2,
            'Close': price,
            'Volume': 1000
        })
    
    signals = []
    
    # Process each bar
    for bar_dict in bars:
        # Create bar event
        bar_event = BarEvent(bar_dict)
        
        # Process bar with rule
        signal = rule.on_bar(Event(EventType.BAR, bar_event))
        
        # Check if signal was generated
        if signal:
            signals.append(signal)
            logger.info(f"Signal generated at {bar_dict['timestamp']}: {signal}")
            
            # Verify signal is a SignalEvent
            if not isinstance(signal, SignalEvent):
                logger.error(f"PROBLEM: Signal is not a SignalEvent! Type: {type(signal)}")
            else:
                direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
                logger.info(f"Signal details: {direction} for {signal.get_symbol()} at {signal.get_price()}")
    
    # Print summary
    logger.info(f"Processed {len(bars)} bars, generated {len(signals)} signals")
    for i, signal in enumerate(signals):
        if isinstance(signal, SignalEvent):
            direction = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL" 
            logger.info(f"Signal {i+1}: {direction} at {signal.get_price()}")
        else:
            logger.error(f"Signal {i+1} is not a SignalEvent! Type: {type(signal)}")
    
    return len(signals) > 0

if __name__ == "__main__":
    logger.info("Starting SMACrossoverRule signal generation test")
    result = test_sma_rule_signal_generation()
    if result:
        logger.info("Test completed successfully")
    else:
        logger.error("Test failed - no signals generated")
