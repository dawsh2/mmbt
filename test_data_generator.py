"""
Test Data Generator for Backtesting Validation

This script generates:
1. A synthetic OHLC price series with known patterns
2. A simple test rule that buys when price is above threshold and sells when below
3. Expected signals and returns based on the rule logic
"""

import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from signals import Signal, SignalType

# Parameters for synthetic data
NUM_BARS = 200
THRESHOLD = 100.0  # Price threshold for buy/sell decisions
PRICE_VARIANCE = 5.0  # How much prices vary randomly
OSCILLATION_PERIOD = 20  # Bars between price oscillations

def generate_test_data():
    """Generate synthetic OHLC data with predictable patterns for testing."""
    start_date = datetime.datetime(2023, 1, 1)
    timestamps = [start_date + datetime.timedelta(days=i) for i in range(NUM_BARS)]
    
    # Create base price that oscillates above and below threshold
    base_prices = []
    for i in range(NUM_BARS):
        # Oscillate between prices above and below threshold
        cycle_position = i % OSCILLATION_PERIOD
        if cycle_position < OSCILLATION_PERIOD // 2:
            # Above threshold
            base_price = THRESHOLD + 10.0
        else:
            # Below threshold
            base_price = THRESHOLD - 10.0
        base_prices.append(base_price)
    
    # Add some randomness to make it realistic
    np.random.seed(42)  # For reproducibility
    random_component = np.random.normal(0, PRICE_VARIANCE, NUM_BARS)
    closes = np.array(base_prices) + random_component
    
    # Create OHLC data with some intraday movement
    data = []
    for i in range(NUM_BARS):
        close = closes[i]
        # Create reasonable Open, High, Low values
        open_price = close * (1 + np.random.normal(0, 0.01))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        
        data.append({
            'timestamp': timestamps[i].strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': int(np.random.uniform(1000, 5000))
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def calculate_expected_signals_and_returns(df):
    """Calculate expected signals and returns based on the threshold rule."""
    expected_signals = []
    current_position = 0
    entry_price = None
    entry_time = None
    trades = []
    
    # Identify signals and calculate expected trades
    for i, row in df.iterrows():
        timestamp = row['timestamp']
        close = row['Close']
        
        # Determine signal based on price threshold
        if close > THRESHOLD:
            signal_type = SignalType.BUY  
        else:
            signal_type = SignalType.SELL
            
        # Record the signal
        expected_signals.append({
            'timestamp': timestamp,
            'signal_type': signal_type,
            'price': close
        })
        
        # Process the signal to generate trades
        if current_position == 0:  # Not in a position
            if signal_type == SignalType.BUY:
                # Enter long position
                current_position = 1
                entry_price = close
                entry_time = timestamp
            elif signal_type == SignalType.SELL:
                # Enter short position
                current_position = -1
                entry_price = close
                entry_time = timestamp
        elif current_position == 1:  # In long position
            if signal_type == SignalType.SELL:
                # Exit long position
                log_return = np.log(close / entry_price)
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': timestamp,
                    'exit_price': close,
                    'log_return': log_return,
                    'position': 'LONG'
                })
                current_position = 0
        elif current_position == -1:  # In short position
            if signal_type == SignalType.BUY:
                # Exit short position
                log_return = np.log(entry_price / close)
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': timestamp,
                    'exit_price': close,
                    'log_return': log_return,
                    'position': 'SHORT'
                })
                current_position = 0
    
    # Calculate expected performance metrics
    if trades:
        total_log_return = sum(trade['log_return'] for trade in trades)
        total_return = (np.exp(total_log_return) - 1) * 100  # Convert to percentage
        avg_log_return = total_log_return / len(trades)
        win_count = sum(1 for trade in trades if trade['log_return'] > 0)
        win_rate = win_count / len(trades) if trades else 0
    else:
        total_log_return = 0
        total_return = 0
        avg_log_return = 0
        win_rate = 0
    
    expected_results = {
        'total_log_return': total_log_return,
        'total_percent_return': total_return,
        'average_log_return': avg_log_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'trades': trades
    }
    
    return expected_signals, expected_results

def plot_test_data(df, expected_signals):
    """Plot the test data with signals for visualization."""
    plt.figure(figsize=(15, 8))
    
    # Plot price
    plt.plot(df['timestamp'], df['Close'], label='Close Price')
    
    # Add threshold line
    plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')
    
    # Add buy/sell markers
    buy_timestamps = [signal['timestamp'] for signal in expected_signals if signal['signal_type'] == SignalType.BUY]
    buy_prices = [signal['price'] for signal in expected_signals if signal['signal_type'] == SignalType.BUY]
    
    sell_timestamps = [signal['timestamp'] for signal in expected_signals if signal['signal_type'] == SignalType.SELL]
    sell_prices = [signal['price'] for signal in expected_signals if signal['signal_type'] == SignalType.SELL]
    
    plt.scatter(buy_timestamps, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sell_timestamps, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
    
    plt.title('Test Data with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('test_data_visualization.png')
    plt.close()

def create_test_rule_class():
    """Create a test rule class that implements the threshold strategy."""
    class TestThresholdRule:
        """
        Test rule that generates BUY signals when price is above threshold
        and SELL signals when price is below threshold.
        """
        def __init__(self, params=None):
            self.threshold = params.get('threshold', THRESHOLD) if params else THRESHOLD
            self.current_signal_type = SignalType.NEUTRAL
            self.rule_id = "TestThresholdRule"
        
        def on_bar(self, bar):
            """Process a bar and generate a signal based on threshold comparison."""
            close = bar['Close']
            
            if close > self.threshold:
                self.current_signal_type = SignalType.BUY
            else:
                self.current_signal_type = SignalType.SELL
                
            # Create and return a Signal object
            return Signal(
                timestamp=bar["timestamp"],
                signal_type=self.current_signal_type,
                price=bar["Close"],
                rule_id=self.rule_id,
                confidence=1.0,
                metadata={"threshold": self.threshold}
            )
        
        def reset(self):
            """Reset the rule's state."""
            self.current_signal_type = SignalType.NEUTRAL
    
    return TestThresholdRule

if __name__ == "__main__":
    # Generate test data
    print("Generating test data...")
    df = generate_test_data()
    
    # Calculate expected signals and returns
    print("Calculating expected signals and returns...")
    expected_signals, expected_results = calculate_expected_signals_and_returns(df)
    
    # Create output directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Save to CSV
    csv_path = 'test_data/test_ohlc_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved test data to {csv_path}")
    
    # Save expected results to CSV for reference
    expected_trades_df = pd.DataFrame(expected_results['trades'])
    expected_trades_df.to_csv('test_data/expected_trades.csv', index=False)
    
    # Save general stats
    with open('test_data/expected_results.txt', 'w') as f:
        f.write(f"Total Log Return: {expected_results['total_log_return']:.4f}\n")
        f.write(f"Total Return (%): {expected_results['total_percent_return']:.2f}%\n")
        f.write(f"Average Log Return: {expected_results['average_log_return']:.4f}\n")
        f.write(f"Number of Trades: {expected_results['num_trades']}\n")
        f.write(f"Win Rate: {expected_results['win_rate']:.2f}\n")
    
    # Plot the data with signals
    plot_test_data(df, expected_signals)
    print("Generated visualization: test_data_visualization.png")
    
    # Explain how to use the test rule
    print("\nTest Rule Implementation:")
    TestThresholdRule = create_test_rule_class()
    print("Created TestThresholdRule class that buys above", THRESHOLD, "and sells below")
    print("\nExample usage:")
    print("from test_data_generator import create_test_rule_class")
    print("TestThresholdRule = create_test_rule_class()")
    print("test_rule = TestThresholdRule()")
    
    print("\nRunning test backtest with 10 bars for quick validation:")
    # Quick demonstration with first 10 bars
    from backtester import Backtester, BarEvent
    
    class SimpleDataHandler:
        def __init__(self, data):
            self.data = data
            self.index = 0
            
        def get_next_train_bar(self):
            if self.index < len(self.data):
                bar = self.data.iloc[self.index].to_dict()
                self.index += 1
                return bar
            return None
            
        def reset_train(self):
            self.index = 0
            
        def get_next_test_bar(self):
            return self.get_next_train_bar()
            
        def reset_test(self):
            self.index = 0
    
    # Create a simple strategy using the test rule
    class SimpleStrategy:
        def __init__(self, rule):
            self.rule = rule
            
        def on_bar(self, event):
            bar = event.bar
            return self.rule.on_bar(bar)
            
        def reset(self):
            self.rule.reset()
    
    # Test with the first 10 bars
    test_data = df.head(10)
    data_handler = SimpleDataHandler(test_data)
    test_rule = TestThresholdRule()
    strategy = SimpleStrategy(test_rule)
    backtester = Backtester(data_handler, strategy)
    
    results = backtester.run(use_test_data=False)
    
    print(f"  Total Log Return: {results['total_log_return']:.4f}")
    print(f"  Number of Trades: {results['num_trades']}")
    
    print("\nNow you can run a full backtest with your existing framework using TestThresholdRule")
