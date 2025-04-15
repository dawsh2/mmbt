#!/usr/bin/env python3
# test_rule.py - Simple test script for the AlwaysBuyRule

import logging
import datetime
from always_buy_rule import AlwaysBuyRule
from src.events import Event, EventType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rule():
    """Test the AlwaysBuyRule directly."""
    # Create the rule
    rule = AlwaysBuyRule(
        name="test_rule",
        params={
            'frequency': 2,  # Generate a signal every 2 bars
            'confidence': 1.0  # Full confidence for testing
        }
    )
    
    logger.info(f"Created rule: {rule.name}")
    logger.info(f"Rule params: {rule.params}")
    
    # Initial state
    logger.info(f"Initial state: {rule.get_state()}")
    
    # Create fake bar data
    bars = []
    for i in range(10):
        bar = {
            "timestamp": datetime.datetime.now(),
            "Open": 100 + i,
            "High": 101 + i,
            "Low": 99 + i,
            "Close": 100.5 + i,
            "Volume": 1000 * (i + 1),
            "symbol": "TEST"
        }
        bars.append(bar)
    
    # Process bars through rule
    signals = []
    for i, bar in enumerate(bars):
        # Create an event
        event = Event(EventType.BAR, bar)
        
        # Process with rule
        signal = rule.on_bar(event)
        
        # Log result
        if signal:
            logger.info(f"Bar {i} generated signal: {signal.signal_type}")
            signals.append(signal)
        else:
            logger.info(f"Bar {i} generated no signal")
        
        # Log rule state
        logger.info(f"Rule state after bar {i}: {rule.get_state()}")
    
    # Summary
    logger.info(f"Total bars processed: {len(bars)}")
    logger.info(f"Total signals generated: {len(signals)}")
    logger.info(f"Final rule state: {rule.get_state()}")
    
    return signals

if __name__ == "__main__":
    logger.info("Starting rule test")
    signals = test_rule()
    logger.info("Test complete")
