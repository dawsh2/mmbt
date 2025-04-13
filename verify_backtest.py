import pandas as pd
import numpy as np
import math

def verify_backtester():
    # Load test data
    df = pd.read_csv('test_data/test_ohlc_data.csv')
    
    # Simple variables to track state
    threshold = 100.0
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = None
    entry_time = None
    trades = []
    
    # Process each bar
    for i, row in df.iterrows():
        timestamp = row['timestamp']
        close = row['Close']
        
        # Generate signal based on threshold
        signal = 1 if close > threshold else -1
        
        # Process signal based on current position
        if position == 0:  # Not in a position
            if signal == 1:  # Buy signal
                position = 1
                entry_price = close
                entry_time = timestamp
                print(f"Enter LONG at {timestamp}: ${close:.2f}")
            elif signal == -1:  # Sell signal
                position = -1
                entry_price = close
                entry_time = timestamp
                print(f"Enter SHORT at {timestamp}: ${close:.2f}")
        
        elif position == 1:  # In long position
            if signal == -1:  # Exit signal
                log_return = math.log(close / entry_price)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'position': 'long',
                    'log_return': log_return
                })
                print(f"Exit LONG at {timestamp}: ${close:.2f}, Return: {log_return:.4f}")
                
                # Enter new position
                position = -1
                entry_price = close
                entry_time = timestamp
                print(f"Enter SHORT at {timestamp}: ${close:.2f}")
                
        elif position == -1:  # In short position
            if signal == 1:  # Exit signal
                log_return = math.log(entry_price / close)
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'position': 'short',
                    'log_return': log_return
                })
                print(f"Exit SHORT at {timestamp}: ${close:.2f}, Return: {log_return:.4f}")
                
                # Enter new position
                position = 1
                entry_price = close
                entry_time = timestamp
                print(f"Enter LONG at {timestamp}: ${close:.2f}")
    
    # Calculate overall return
    total_log_return = sum(trade['log_return'] for trade in trades)
    total_return = (math.exp(total_log_return) - 1) * 100
    
    print(f"\nVerification Results:")
    print(f"Total trades: {len(trades)}")
    print(f"Total log return: {total_log_return:.4f}")
    print(f"Total return: {total_return:.2f}%")
    
    return trades, total_return

# Run verification
verify_backtester()
