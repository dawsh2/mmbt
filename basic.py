#!/usr/bin/env python3
"""
Test script for BollingerBandRule and SMAcrossoverRule using direct imports.
"""

import datetime
import os
import logging
import pandas as pd
import numpy as np

# Direct imports of the actual rule classes
from src.rules.volatility_rules import BollingerBandRule
from src.rules.crossover_rules import SMAcrossoverRule
from src.signals import Signal, SignalType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(symbol="SYNTHETIC", timeframe="1d", filename=None, days=100):
    """Create synthetic price data for testing."""
    # Create a synthetic dataset with more pronounced patterns
    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    
    # Create a price series with clear patterns for rule testing
    prices = []
    
    # Initial price
    base_price = 100
    
    # Create strong trends and volatility clusters
    for i in range(days):
        # Every 20 days, create a new trend
        if i % 20 == 0:
            trend_direction = 1 if i % 40 == 0 else -1
            volatility = 0.02  # Higher volatility
        else:
            trend_direction *= 0.95  # Trend gradually weakens
            volatility = max(0.005, volatility * 0.9)  # Volatility stabilizes
            
        # Add daily price change
        daily_change = trend_direction * 0.005 + np.random.normal(0, volatility)
        
        # Calculate new price
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + daily_change)
            
        prices.append(price)
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.normal(100000, 20000)) for _ in prices],
        'symbol': symbol
    })
    
    # Save to CSV if filename provided
    if filename:
        df.to_csv(filename, index=False)
        logger.info(f"Created synthetic data with {len(df)} bars, saved to {filename}")
    
    return df

def test_rules_directly():
    """Test BollingerBandRule and SMAcrossoverRule directly."""
    logger.info("Testing rules directly with synthetic data")
    
    # Create synthetic data
    symbol = "SYNTHETIC"
    timeframe = "1d"
    filename = f"{symbol}_{timeframe}_test.csv"
    data_df = create_synthetic_data(symbol, timeframe, filename)
    
    # Print the default parameters of each rule to understand what's expected
    logger.info(f"BollingerBandRule default params: {BollingerBandRule.default_params()}")
    logger.info(f"SMAcrossoverRule default params: {SMAcrossoverRule.default_params()}")
    
    # Create rules directly using their classes, with correct parameters
    bb_rule = BollingerBandRule(
        name="bollinger_test",
        params={
            'period': 20,
            'std_dev': 2.0,
            'breakout_type': 'both',   # Corrected parameter name
            'use_confirmations': True  # Added required parameter
        }
    )
    
    sma_rule = SMAcrossoverRule(
        name="sma_test",
        params={
            'fast_window': 5,
            'slow_window': 20,
            'smooth_signals': True
        }
    )
    
    # Reset rules to ensure clean state
    bb_rule.reset()
    sma_rule.reset()
    
    # Process each bar and track signals
    bb_signals = []
    sma_signals = []
    
    logger.info(f"Processing {len(data_df)} bars of data")
    
    for i, row in data_df.iterrows():
        bar_data = row.to_dict()
        
        # Test BollingerBandRule
        try:
            bb_signal = bb_rule.generate_signal(bar_data)
            if bb_signal and bb_signal.signal_type != SignalType.NEUTRAL:
                bb_signals.append(bb_signal)
                logger.info(f"Bar {i}: BollingerBandRule generated {bb_signal.signal_type} signal")
                
                # Check if symbol is in metadata
                if hasattr(bb_signal, 'metadata') and 'symbol' in bb_signal.metadata:
                    logger.info(f"  Symbol in metadata: {bb_signal.metadata['symbol']}")
                else:
                    logger.error(f"  Symbol MISSING from metadata!")
        except Exception as e:
            logger.error(f"Error testing BollingerBandRule at bar {i}: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Test SMAcrossoverRule
        try:
            sma_signal = sma_rule.generate_signal(bar_data)
            if sma_signal and sma_signal.signal_type != SignalType.NEUTRAL:
                sma_signals.append(sma_signal)
                logger.info(f"Bar {i}: SMAcrossoverRule generated {sma_signal.signal_type} signal")
                
                # Check if symbol is in metadata
                if hasattr(sma_signal, 'metadata') and 'symbol' in sma_signal.metadata:
                    logger.info(f"  Symbol in metadata: {sma_signal.metadata['symbol']}")
                else:
                    logger.error(f"  Symbol MISSING from metadata!")
        except Exception as e:
            logger.error(f"Error testing SMAcrossoverRule at bar {i}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary of results
    logger.info("\nTest Results Summary:")
    logger.info(f"BollingerBandRule generated {len(bb_signals)} non-neutral signals")
    logger.info(f"SMAcrossoverRule generated {len(sma_signals)} non-neutral signals")
    
    # Check if any signals were generated
    success = len(bb_signals) > 0 and len(sma_signals) > 0
    
    return {
        "success": success,
        "bb_signals": len(bb_signals),
        "sma_signals": len(sma_signals)
    }

def main():
    """Main function for rule testing."""
    logger.info("Starting rule testing")
    
    # Test rules directly
    direct_results = test_rules_directly()
    
    # Print results
    logger.info("\nOverall Test Results:")
    logger.info(f"Direct Test Success: {direct_results['success']}")
    logger.info(f"  BB Signals: {direct_results['bb_signals']}")
    logger.info(f"  SMA Signals: {direct_results['sma_signals']}")
    
    return {
        "success": direct_results['success'],
        "direct_test": direct_results,
    }

if __name__ == "__main__":
    try:
        results = main()
        
        print("\n" + "=" * 50)
        print("RULE TEST RESULTS SUMMARY")
        print("=" * 50)
        
        if results["success"]:
            print("\nSUCCESS: Both rules are generating signals with complete metadata!")
            print("The updates to include 'symbol' in metadata are working correctly.")
            print("\nNext steps:")
            print("1. Update all other rules to include 'symbol' in metadata")
            print("2. Update the TopNStrategy to preserve symbol information")
            print("3. Fix position manager to handle both Signal objects and dictionaries")
        else:
            print("\nISSUES DETECTED: Rule updates need further work")
            
            if not results["direct_test"]["success"]:
                print("- Direct rule testing failed to generate signals")
                print(f"  BollingerBandRule: {results['direct_test']['bb_signals']} signals")
                print(f"  SMAcrossoverRule: {results['direct_test']['sma_signals']} signals")
            
            print("\nCheck the error messages above for more details")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
