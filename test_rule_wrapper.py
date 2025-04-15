#!/usr/bin/env python3
# test_rule_wrapper.py - Simple test script for the fixed rule wrapper

import sys
import datetime
from src.rules import create_rule
from src.events import Event, EventType
from rules_wrapper import wrap_rule, wrap_rules

def test_bar_event_handling():
    """Test that the wrapper properly handles BarEvent objects."""
    print("Testing rule wrapper with BarEvent objects...")
    
    # Create a standard rule
    sma_rule = create_rule('SMAcrossoverRule', {
        'fast_window': 10,
        'slow_window': 30,
        'smooth_signals': True
    })
    
    # Wrap the rule
    wrapped_rule = wrap_rule(sma_rule)
    
    # Create a bar event
    bar_data = {
        'timestamp': datetime.datetime.now(),
        'Open': 100.0,
        'High': 102.0,
        'Low': 99.0,
        'Close': 101.0,
        'Volume': 1000,
        'symbol': 'TEST'
    }
    bar_event = Event(EventType.BAR, bar_data)
    
    # Test direct processing of event
    print("Testing direct event processing...")
    try:
        signal = wrapped_rule.on_bar(bar_event)
        print(f"Result: {'SUCCESS' if signal is not None else 'FAILURE - returned None'}")
    except Exception as e:
        print(f"FAILURE - Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test with dictionary data
    print("\nTesting with dictionary data...")
    try:
        signal = wrapped_rule.on_bar(bar_data)
        print(f"Result: {'SUCCESS' if signal is not None else 'FAILURE - returned None'}")
    except Exception as e:
        print(f"FAILURE - Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create more bar data for proper signal generation
    print("\nFeeding more data for proper signal generation...")
    for i in range(30):  # Feed enough data for SMA calculation
        new_price = 100.0 + (i * 0.1)  # Upward trend
        new_bar_data = {
            'timestamp': datetime.datetime.now() + datetime.timedelta(minutes=i),
            'Open': new_price - 0.5,
            'High': new_price + 0.5,
            'Low': new_price - 0.7,
            'Close': new_price,
            'Volume': 1000 + i * 10,
            'symbol': 'TEST'
        }
        new_bar_event = Event(EventType.BAR, new_bar_data)
        try:
            signal = wrapped_rule.on_bar(new_bar_event)
            if signal and signal.signal_type.value != 0:  # Non-neutral signal
                print(f"Generated {signal.signal_type.name} signal after {i+1} bars")
                break
        except Exception as e:
            print(f"FAILURE on bar {i+1} - Exception: {str(e)}")
            break
    
    # Check rule state
    print("\nChecking rule state...")
    state = wrapped_rule.get_state()
    if state:
        print(f"Rule has state: {state}")
    else:
        print("Rule state is empty")
    
    # Check signal history
    print("\nChecking signal history...")
    if wrapped_rule.signals:
        print(f"Rule has {len(wrapped_rule.signals)} signals in history")
    else:
        print("Rule signal history is empty")

def test_multiple_rule_types():
    """Test wrapper with multiple rule types."""
    print("\nTesting wrapper with multiple rule types...")
    
    # Create different types of rules
    rules = [
        create_rule('SMAcrossoverRule', {'fast_window': 10, 'slow_window': 30}),
        create_rule('RSIRule', {'rsi_period': 14, 'overbought': 70, 'oversold': 30, 'signal_type': 'levels'}),
        create_rule('MACDCrossoverRule', {'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
    ]
    
    # Wrap all rules
    wrapped_rules = wrap_rules(rules)
    
    # Test each rule with a bar event
    bar_data = {
        'timestamp': datetime.datetime.now(),
        'Open': 100.0,
        'High': 102.0,
        'Low': 99.0,
        'Close': 101.0,
        'Volume': 1000,
        'symbol': 'TEST'
    }
    bar_event = Event(EventType.BAR, bar_data)
    
    for i, rule in enumerate(wrapped_rules):
        print(f"\nTesting rule {i+1}: {rule.name}...")
        try:
            signal = rule.on_bar(bar_event)
            print(f"Result: {'SUCCESS' if signal is not None else 'FAILURE - returned None'}")
        except Exception as e:
            print(f"FAILURE - Exception: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_bar_event_handling()
    test_multiple_rule_types()
    print("\nAll tests completed.")
