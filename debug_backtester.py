"""
Add this code to a new file debug_backtester.py to diagnose the issue 
with signal handling in your backtester.
"""

from data_handler import CSVDataHandler
from backtester import Backtester, BarEvent
import math

class DebugThresholdRule:
    """Simple rule that buys when price > threshold, sells when below."""
    def __init__(self, threshold=100.0):
        self.threshold = threshold
        
    def on_bar(self, bar):
        # Extract price from bar
        if isinstance(bar, dict):
            close = bar['Close'] 
        elif hasattr(bar, 'bar'):
            close = bar.bar['Close']
        else:
            close = 0
            
        # Generate signal (1 for buy, -1 for sell)
        signal = 1 if close > self.threshold else -1
        print(f"Bar close: {close:.2f}, Signal: {signal}")
        return signal
        
    def reset(self):
        pass

class DebugBacktester(Backtester):
    """Extends Backtester with detailed logging."""
    
    def __init__(self, data_handler, strategy):
        super().__init__(data_handler, strategy)
        self.signal_history = []
        self.position_history = []
        
    def _process_signal_for_trades(self, signal, timestamp, price):
        """
        Process a signal with detailed logging to diagnose issues.
        """
        # Extract signal value
        if hasattr(signal, 'signal_type'):
            signal_value = signal.signal_type.value
        elif isinstance(signal, dict) and 'signal' in signal:
            signal_value = signal['signal']
        elif isinstance(signal, (int, float)):
            signal_value = signal
        else:
            signal_value = 0
            
        # Record signal in history
        self.signal_history.append((timestamp, signal_value, price, self.current_position))
        
        # Log the current state
        print(f"\nProcessing: Time={timestamp}, Signal={signal_value}, Price={price:.2f}, Position={self.current_position}")
        print(f"  Entry price: {self.entry_price}, Entry time: {self.entry_time}")
        
        # Process signal with detailed logging
        old_position = self.current_position
        action_taken = "NO ACTION"
        
        if signal_value == 1:  # Buy signal
            if self.current_position <= 0:  # Not in long position
                if self.current_position < 0:  # In short position
                    # Close short position
                    log_return = math.log(self.entry_price / price) if price > 0 else 0
                    self.trades.append((
                        self.entry_time,
                        'short',
                        self.entry_price,
                        timestamp,
                        price,
                        log_return
                    ))
                    action_taken = f"CLOSE SHORT (Return: {log_return:.4f})"
                
                # Enter long position
                self.current_position = 1
                self.entry_price = price
                self.entry_time = timestamp
                action_taken += ", ENTER LONG"
            else:
                action_taken = "ALREADY LONG (HOLD)"
                
        elif signal_value == -1:  # Sell signal
            if self.current_position >= 0:  # Not in short position
                if self.current_position > 0:  # In long position
                    # Close long position
                    log_return = math.log(price / self.entry_price) if self.entry_price > 0 else 0
                    self.trades.append((
                        self.entry_time,
                        'long',
                        self.entry_price,
                        timestamp,
                        price,
                        log_return
                    ))
                    action_taken = f"CLOSE LONG (Return: {log_return:.4f})"
                
                # Enter short position
                self.current_position = -1
                self.entry_price = price
                self.entry_time = timestamp
                action_taken += ", ENTER SHORT"
            else:
                action_taken = "ALREADY SHORT (HOLD)"
                
        elif signal_value == 0:  # Neutral signal
            if self.current_position > 0:  # In long position
                # Close long position
                log_return = math.log(price / self.entry_price) if self.entry_price > 0 else 0
                self.trades.append((
                    self.entry_time,
                    'long',
                    self.entry_price,
                    timestamp,
                    price,
                    log_return
                ))
                self.current_position = 0
                action_taken = f"CLOSE LONG (Return: {log_return:.4f})"
            elif self.current_position < 0:  # In short position
                # Close short position
                log_return = math.log(self.entry_price / price) if price > 0 else 0
                self.trades.append((
                    self.entry_time,
                    'short',
                    self.entry_price,
                    timestamp,
                    price,
                    log_return
                ))
                self.current_position = 0
                action_taken = f"CLOSE SHORT (Return: {log_return:.4f})"
        
        # Record position change in history
        self.position_history.append((timestamp, old_position, self.current_position, action_taken))
        
        print(f"  Action: {action_taken}")
        print(f"  New position: {self.current_position}, Trade count: {len(self.trades)}")

    def print_summary(self):
        """Print a summary of what happened during the backtest."""
        print("\n=== BACKTEST SUMMARY ===")
        print(f"Total trades: {len(self.trades)}")
        
        # Count position changes
        position_changes = sum(1 for i in range(1, len(self.position_history)) 
                            if self.position_history[i][1] != self.position_history[i][2])
        print(f"Position changes: {position_changes}")
        
        # Count signals by type
        buy_signals = sum(1 for s in self.signal_history if s[1] == 1)
        sell_signals = sum(1 for s in self.signal_history if s[1] == -1)
        neutral_signals = sum(1 for s in self.signal_history if s[1] == 0)
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"Neutral signals: {neutral_signals}")
        
        # Print signal stats
        signal_transitions = []
        prev_signal = None
        for _, signal, _, _ in self.signal_history:
            if prev_signal is not None and prev_signal != signal:
                signal_transitions.append((prev_signal, signal))
            prev_signal = signal
            
        print(f"\nSignal transitions: {len(signal_transitions)}")
        if signal_transitions:
            print("First 5 transitions:")
            for i, (from_signal, to_signal) in enumerate(signal_transitions[:5]):
                from_str = "BUY" if from_signal == 1 else "SELL" if from_signal == -1 else "NEUTRAL"
                to_str = "BUY" if to_signal == 1 else "SELL" if to_signal == -1 else "NEUTRAL"
                print(f"  {i+1}. {from_str} → {to_str}")
        
        # Print the recorded trades
        print("\nTrade summary:")
        if self.trades:
            print(f"{'Entry time':<12} {'Type':<6} {'Entry':<8} {'Exit time':<12} {'Exit':<8} {'Return':<8}")
            print("-" * 60)
            for i, t in enumerate(self.trades[:5]):  # Show first 5 trades
                entry_time = t[0]
                direction = t[1]
                entry_price = t[2]
                exit_time = t[3]
                exit_price = t[4]
                log_return = t[5]
                print(f"{entry_time:<12} {direction:<6} {entry_price:<8.2f} {exit_time:<12} {exit_price:<8.2f} {log_return:<8.4f}")
            
            if len(self.trades) > 5:
                print(f"... and {len(self.trades) - 5} more trades")

# Function to run the debug backtester
def run_debug_backtest():
    # Load test data
    data_path = 'test_data/test_ohlc_data.csv'
    data_handler = CSVDataHandler(data_path, train_fraction=0.7)
    
    # Create rule and strategy
    rule = DebugThresholdRule(threshold=100.0)
    
    class SimpleStrategy:
        def __init__(self, rule):
            self.rule = rule
        def on_bar(self, event):
            return self.rule.on_bar(event.bar)
        def reset(self):
            self.rule.reset()
    
    strategy = SimpleStrategy(rule)
    
    # Create debug backtester
    backtester = DebugBacktester(data_handler, strategy)
    
    # Run backtest
    print("Running debug backtest...")
    results = backtester.run(use_test_data=True)
    
    # Print summary
    backtester.print_summary()
    
    # Print performance metrics
    print("\nPerformance metrics:")
    print(f"Total Return: {results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Average Return per Trade: {results['average_log_return']:.4f}")
    
    return results

if __name__ == "__main__":
    run_debug_backtest()
