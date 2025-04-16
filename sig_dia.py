#!/usr/bin/env python3
"""
Signal Generation Diagnostic Script
"""

import datetime
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import system components
from src.events import Event, EventType
from src.data.data_handler import DataHandler
from src.data.data_sources import CSVDataSource
from src.rules import create_rule, Rule
from src.signals import Signal, SignalType

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_extreme_test_data(symbol="SYNTHETIC", timeframe="1d", filename=None, days=100):
    """Create extreme synthetic data guaranteed to trigger signals."""
    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    
    # Create an extreme price series with clear patterns
    prices = []
    for i in range(days):
        # Create a sawtooth pattern
        if i % 20 < 10:
            # Strong uptrend 
            price = 100 + (i % 20) * 5
        else:
            # Strong downtrend
            price = 150 - (i % 20 - 10) * 5
        prices.append(price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],  # 2% higher
        'Low': [p * 0.98 for p in prices],   # 2% lower
        'Close': prices,
        'Volume': [100000 for _ in prices],
        'symbol': symbol
    })
    
    # Save to CSV if filename provided
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Created extreme test data with {len(df)} bars, saved to {filename}")
    
    return df

def test_rule_directly(rule, data_df):
    """Test a single rule directly with bar data."""
    signals = []
    
    logger.info(f"Testing rule {rule.name} with parameters: {rule.params}")
    
    # Reset the rule to ensure clean state
    rule.reset()
    
    # Process each bar directly
    for i, row in data_df.iterrows():
        bar_data = row.to_dict()
        
        # Call generate_signal directly
        try:
            signal = rule.generate_signal(bar_data)
            
            # Log if a non-neutral signal was generated
            if signal and signal.signal_type != SignalType.NEUTRAL:
                signals.append(signal)
                logger.info(f"Bar {i}: Generated {signal.signal_type} signal at price {signal.price}")
        except Exception as e:
            logger.error(f"Error generating signal at bar {i}: {e}")
    
    logger.info(f"Total signals generated: {len(signals)}")
    logger.info(f"Buy signals: {sum(1 for s in signals if s.signal_type == SignalType.BUY)}")
    logger.info(f"Sell signals: {sum(1 for s in signals if s.signal_type == SignalType.SELL)}")
    
    return signals

def debug_rule_with_manual_data():
    """Debug rule behavior with manually constructed data."""
    logger.info("Testing rule with manually constructed extreme data")
    
    # Create a simple rule for testing
    sma_rule = create_rule('SMAcrossoverRule', {
        'fast_window': 3,   # Very short for quick signals
        'slow_window': 6,
        'smooth_signals': True  # Generate signal at every bar
    })
    
    # Create some extreme test data
    # Prices that will definitively cross (alternating up/down trend)
    prices = [100, 102, 105, 108, 106, 103, 101, 104, 107, 110, 108, 105]
    
    # Create dataframe with test data
    dates = pd.date_range(start='2022-01-01', periods=len(prices), freq='D')
    test_df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': [10000 for _ in prices],
        'symbol': 'TEST'
    })
    
    # Test the rule with manual data
    signals = test_rule_directly(sma_rule, test_df)
    
    # If no signals, there's a fundamental issue with the rule
    if not signals:
        logger.error("No signals generated even with extreme test data!")
        logger.error("There may be a bug in the rule implementation.")
        
        # Try to debug by tracing through rule execution
        logger.info("Tracing through rule execution step by step...")
        
        sma_rule.reset()
        fast_sma_values = []
        slow_sma_values = []
        
        for i, row in test_df.iterrows():
            bar_data = row.to_dict()
            logger.info(f"Processing bar {i}: Close = {bar_data['Close']}")
            
            # Manually implement SMA calculation
            if hasattr(sma_rule, 'prices'):
                # Add current price
                sma_rule.prices.append(bar_data['Close'])
                logger.info(f"  Prices history: {list(sma_rule.prices)}")
                
                # Calculate SMAs if enough data
                fast_window = sma_rule.params['fast_window']
                slow_window = sma_rule.params['slow_window']
                
                if len(sma_rule.prices) >= slow_window:
                    fast_sma = sum(list(sma_rule.prices)[-fast_window:]) / fast_window
                    slow_sma = sum(list(sma_rule.prices)[-slow_window:]) / slow_window
                    
                    fast_sma_values.append(fast_sma)
                    slow_sma_values.append(slow_sma)
                    
                    logger.info(f"  Fast SMA: {fast_sma}, Slow SMA: {slow_sma}")
                    logger.info(f"  Crossover condition: {fast_sma > slow_sma}")
                    
                    # Check if there's a crossover
                    if len(fast_sma_values) >= 2 and len(slow_sma_values) >= 2:
                        prev_fast = fast_sma_values[-2]
                        prev_slow = slow_sma_values[-2]
                        
                        crossover_up = prev_fast <= prev_slow and fast_sma > slow_sma
                        crossover_down = prev_fast >= prev_slow and fast_sma < slow_sma
                        
                        logger.info(f"  Previous Fast SMA: {prev_fast}, Previous Slow SMA: {prev_slow}")
                        logger.info(f"  Crossover Up: {crossover_up}, Crossover Down: {crossover_down}")
                else:
                    logger.info(f"  Not enough data for SMAs. Have {len(sma_rule.prices)} prices, need {slow_window}")
    
    return signals is not None and len(signals) > 0

def run_sma_rule_test():
    """Run comprehensive tests on the SMA crossover rule."""
    logger.info("Running SMA rule diagnostic tests")
    
    # 1. Test with extreme data
    debug_success = debug_rule_with_manual_data()
    
    if not debug_success:
        logger.error("Rule debugging failed - fundamental problem with rule implementation")
        return False
    
    # 2. Test with synthetic data file
    symbol = "SYNTHETIC"
    timeframe = "1d"
    test_filename = f"{symbol}_test_{timeframe}.csv"
    
    # Create extreme test data file
    create_extreme_test_data(symbol, timeframe, test_filename)
    
    # Load the data
    try:
        test_df = pd.read_csv(test_filename)
        logger.info(f"Loaded test data with {len(test_df)} bars")
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return False
    
    # 3. Test different SMA parameter sets
    param_sets = [
        {'fast_window': 3, 'slow_window': 6, 'smooth_signals': True},
        {'fast_window': 5, 'slow_window': 10, 'smooth_signals': True},
        {'fast_window': 10, 'slow_window': 20, 'smooth_signals': False}
    ]
    
    success = False
    for params in param_sets:
        logger.info(f"Testing SMA rule with parameters: {params}")
        
        # Create rule
        sma_rule = create_rule('SMAcrossoverRule', params)
        
        # Test rule
        signals = test_rule_directly(sma_rule, test_df)
        
        if signals and len(signals) > 0:
            logger.info(f"Success! Generated {len(signals)} signals with parameters: {params}")
            success = True
            break
    
    if not success:
        logger.error("Failed to generate signals with any parameter set")
        
        # Check if SMAcrossoverRule is properly implemented
        logger.info("Checking SMAcrossoverRule implementation...")
        import inspect
        from src.rules.crossover_rules import SMAcrossoverRule
        
        # Check generate_signal method
        logger.info("SMAcrossoverRule generate_signal method:")
        if hasattr(SMAcrossoverRule, 'generate_signal'):
            # Check if the method is calling the right logic
            sig_code = inspect.getsource(SMAcrossoverRule.generate_signal)
            logger.info(sig_code)
        
    return success

def test_rsi_rule():
    """Test the RSI rule implementation."""
    logger.info("Testing RSI rule implementation")
    
    # Create extreme test data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    test_filename = f"{symbol}_rsi_test_{timeframe}.csv"
    
    # Create specialized RSI test data (strong trends followed by reversals)
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    
    # Create price series optimized for RSI signals
    prices = []
    # Start with uptrend to create overbought
    for i in range(25):
        prices.append(100 + i * 2)  # Strong uptrend
    
    # Sharp reversal
    for i in range(25):
        prices.append(150 - i * 2)  # Strong downtrend
    
    # Another uptrend to create overbought
    for i in range(25):
        prices.append(100 + i * 2)  # Strong uptrend
    
    # Final reversal
    for i in range(25):
        prices.append(150 - i * 2)  # Strong downtrend
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': [10000 for _ in prices],
        'symbol': symbol
    })
    
    # Save test data
    test_df.to_csv(test_filename, index=False)
    
    # Create RSI rule with different parameters
    param_sets = [
        {'rsi_period': 7, 'overbought': 70, 'oversold': 30, 'signal_type': 'levels'},
        {'rsi_period': 14, 'overbought': 70, 'oversold': 30, 'signal_type': 'levels'},
        {'rsi_period': 7, 'overbought': 80, 'oversold': 20, 'signal_type': 'levels'}
    ]
    
    success = False
    for params in param_sets:
        logger.info(f"Testing RSI rule with parameters: {params}")
        
        # Create rule
        rsi_rule = create_rule('RSIRule', params)
        
        # Test rule
        signals = test_rule_directly(rsi_rule, test_df)
        
        if signals and len(signals) > 0:
            logger.info(f"Success! Generated {len(signals)} signals with parameters: {params}")
            success = True
            break
    
    return success

def run_rule_diagnostics():
    """Run diagnostics on multiple rule types."""
    logger.info("Starting rule diagnostics")
    
    # Test SMA rule
    sma_success = run_sma_rule_test()
    logger.info(f"SMA rule test {'PASSED' if sma_success else 'FAILED'}")
    
    # Test RSI rule
    rsi_success = test_rsi_rule()
    logger.info(f"RSI rule test {'PASSED' if rsi_success else 'FAILED'}")
    
    return {
        "sma_rule_success": sma_success,
        "rsi_rule_success": rsi_success
    }

if __name__ == "__main__":
    try:
        results = run_rule_diagnostics()
        print("\nRule Diagnostics Completed")
        
        if results["sma_rule_success"] and results["rsi_rule_success"]:
            print("SUCCESS: All rule tests passed. The rule implementations work correctly.")
            print("The issue is likely in how the rules are connected to the rest of the system.")
            
            print("\nTROUBLESHOOTING STEPS:")
            print("1. Check that strategy.on_bar() is correctly calling rule.on_bar()")
            print("2. Verify the event bus registration for BAR events to strategy")
            print("3. Ensure strategy properly emits SIGNAL events to the event bus")
            
        elif not results["sma_rule_success"] and not results["rsi_rule_success"]:
            print("CRITICAL FAILURE: All rules failed to generate signals")
            print("There appears to be a fundamental issue with the rule implementation.")
            
        else:
            print("PARTIAL SUCCESS: Some rules work, others don't.")
            print(f"- SMA Rule: {'PASSED' if results['sma_rule_success'] else 'FAILED'}")
            print(f"- RSI Rule: {'PASSED' if results['rsi_rule_success'] else 'FAILED'}")
            
    except Exception as e:
        print(f"\nError during diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()
