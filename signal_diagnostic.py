"""
Signal Flow Diagnostic Tool

This script helps diagnose signal handling issues by:
1. Directly consuming the test data 
2. Printing out each signal and resulting trade decision
3. Comparing with expected behavior
"""

import pandas as pd
import numpy as np
from signals import Signal, SignalType
from test_data_generator import create_test_rule_class

def debug_signal_flow():
    """Diagnose signal flow issues by tracking each step of signal processing."""
    print("\n=== SIGNAL FLOW DIAGNOSTIC ===")
    
    # Create test rule and load test data
    TestThresholdRule = create_test_rule_class()
    test_rule = TestThresholdRule()
    
    try:
        # Load test data
        df = pd.read_csv('test_data/test_ohlc_data.csv')
        print(f"Loaded {len(df)} bars of test data")
        
        # Load expected trades for comparison
        expected_trades_df = pd.read_csv('test_data/expected_trades.csv')
        print(f"Expected {len(expected_trades_df)} trades")
        
        # Dictionary to track position state
        position_state = {
            'current_position': 0,  # 0 = flat, 1 = long, -1 = short
            'entry_price': None,
            'entry_time': None,
            'trades': []
        }
        
        # Step through each bar and track signal flow
        print("\nProcessing bars and tracking signals:")
        print(f"{'Bar':<4} {'Date':<12} {'Close':<8} {'Signal':<8} {'Position':<10} {'Action':<8}")
        print("-" * 60)
        
        for i, row in df.iterrows():
            # Skip displaying all bars to keep output manageable
            if i > 0 and i % 10 != 0 and i < len(df) - 10:
                continue
                
            # Get bar data
            bar = row.to_dict()
            
            # Generate signal from rule
            signal = test_rule.on_bar(bar)
            
            # Extract signal type (using different approaches to debug)
            signal_type = None
            signal_value = None
            
            if hasattr(signal, 'signal_type'):
                signal_type = signal.signal_type
                signal_value = signal.signal_type.value
            elif isinstance(signal, dict) and 'signal' in signal:
                signal_value = signal['signal']
                signal_type = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
            elif isinstance(signal, (int, float)):
                signal_value = signal
                signal_type = "BUY" if signal_value == 1 else "SELL" if signal_value == -1 else "NEUTRAL"
            else:
                signal_type = "UNKNOWN"
                signal_value = "?"
                
            # Process the signal with standard backtester logic
            action = "HOLD"
            prev_position = position_state['current_position']
            
            if position_state['current_position'] == 0:  # Not in a position
                if signal_value == 1:  # Buy signal
                    position_state['current_position'] = 1
                    position_state['entry_price'] = bar['Close']
                    position_state['entry_time'] = bar['timestamp']
                    action = "BUY"
                elif signal_value == -1:  # Sell signal
                    position_state['current_position'] = -1
                    position_state['entry_price'] = bar['Close']
                    position_state['entry_time'] = bar['timestamp']
                    action = "SELL"
            elif position_state['current_position'] == 1:  # In long position
                if signal_value == -1 or signal_value == 0:  # Exit signal
                    log_return = np.log(bar['Close'] / position_state['entry_price'])
                    position_state['trades'].append({
                        'entry_time': position_state['entry_time'],
                        'entry_price': position_state['entry_price'],
                        'exit_time': bar['timestamp'],
                        'exit_price': bar['Close'],
                        'log_return': log_return,
                        'position': 'LONG'
                    })
                    position_state['current_position'] = 0
                    action = "EXIT LONG"
            elif position_state['current_position'] == -1:  # In short position
                if signal_value == 1 or signal_value == 0:  # Exit signal
                    log_return = np.log(position_state['entry_price'] / bar['Close'])
                    position_state['trades'].append({
                        'entry_time': position_state['entry_time'],
                        'entry_price': position_state['entry_price'],
                        'exit_time': bar['timestamp'],
                        'exit_price': bar['Close'],
                        'log_return': log_return,
                        'position': 'SHORT'
                    })
                    position_state['current_position'] = 0
                    action = "EXIT SHORT"
                    
            # Print the state
            position_str = {0: "FLAT", 1: "LONG", -1: "SHORT"}[position_state['current_position']]
            print(f"{i:<4} {bar['timestamp']:<12} {bar['Close']:<8.2f} {signal_type!s:<8} {position_str:<10} {action:<8}")
            
            # Print position change for detailed tracking
            if prev_position != position_state['current_position']:
                print(f"    Position changed: {prev_position} → {position_state['current_position']}")
                
        # Calculate performance metrics from our trades
        total_log_return = sum(trade['log_return'] for trade in position_state['trades'])
        total_return = (np.exp(total_log_return) - 1) * 100
        
        print("\n=== DIAGNOSTIC SUMMARY ===")
        print(f"Diagnostic found {len(position_state['trades'])} trades")
        print(f"Expected {len(expected_trades_df)} trades")
        print(f"Diagnostic total return: {total_return:.2f}%")
        
        # Compare trade patterns
        if position_state['trades'] and len(expected_trades_df) > 0:
            diag_trade_times = [t['entry_time'] for t in position_state['trades']]
            expected_trade_times = expected_trades_df['entry_time'].tolist()
            
            # Find trades that are in expected but not in diagnostic
            missing_trades = set(expected_trade_times) - set(diag_trade_times)
            if missing_trades:
                print("\nMissing trades (first 5):")
                for i, trade_time in enumerate(sorted(list(missing_trades))[:5]):
                    expected_trade = expected_trades_df[expected_trades_df['entry_time'] == trade_time].iloc[0]
                    print(f"  {trade_time}: Entry ${expected_trade['entry_price']:.2f}, "
                          f"Exit ${expected_trade['exit_price']:.2f}, "
                          f"Return: {expected_trade['log_return']:.4f}")
                          
            # Find diagnostic vs expected comparison of trades that match
            matching_trades = []
            for diag_trade in position_state['trades']:
                for _, exp_trade in expected_trades_df.iterrows():
                    if diag_trade['entry_time'] == exp_trade['entry_time']:
                        matching_trades.append((diag_trade, exp_trade))
                        break
                        
            if matching_trades:
                print("\nMatching trades comparison (first 3):")
                for i, (diag, exp) in enumerate(matching_trades[:3]):
                    print(f"  Trade {i+1} at {diag['entry_time']}:")
                    print(f"    Diagnostic: Entry ${diag['entry_price']:.2f}, Exit ${diag['exit_price']:.2f}, "
                          f"Return: {diag['log_return']:.4f}")
                    print(f"    Expected:   Entry ${exp['entry_price']:.2f}, Exit ${exp['exit_price']:.2f}, "
                          f"Return: {exp['log_return']:.4f}")
        
        print("\n=== SIGNAL INTERPRETATION ===")
        print("If signals aren't being interpreted correctly, check:")
        print("1. How your backtester extracts the signal value (numeric vs. object)")
        print("2. How your backtester decides to enter/exit positions")
        print("3. If your signal format matches what the backtester expects")
        print("\nTry using this diagnostic code in your backtester to trace the actual signals:")
        print("""
if hasattr(signal, 'signal_type'):
    signal_value = signal.signal_type.value  # Using Signal objects
elif isinstance(signal, dict) and 'signal' in signal:
    signal_value = signal['signal']  # Using dictionary format
elif isinstance(signal, (int, float)):
    signal_value = signal  # Using numeric signals
else:
    print(f"Unrecognized signal format: {type(signal)}")
    signal_value = 0
        """)
        
    except Exception as e:
        import traceback
        print(f"Error in diagnostic: {str(e)}")
        traceback.print_exc()
        
if __name__ == "__main__":
    debug_signal_flow()
