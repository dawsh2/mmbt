"""
Debug and Analysis Tools for Backtest Validation

This module provides tools to help diagnose issues with the backtesting engine
and compare actual vs expected trade signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from signals import SignalType
from backtester import Backtester, BarEvent
from threshold_rule import ThresholdRule
from validation_script import SimpleStrategy, SimpleDataHandler

class SignalTracer:
    """
    Traces signals through the backtesting process to identify issues.
    """
    def __init__(self, data_df, rule, upper_threshold, lower_threshold):
        self.data = data_df
        self.rule = rule
        self.signals = []
        self.positions = []
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
    
    def trace_signals(self):
        """
        Process each bar through the rule and record all signals.
        """
        current_position = 0  # 0 = flat, 1 = long, -1 = short
        
        for i, row in self.data.iterrows():
            bar = row.to_dict()
            signal = self.rule.on_bar(bar)
            
            # Log the signal
            self.signals.append({
                'index': i,
                'timestamp': row['timestamp'],
                'close': row['Close'],
                'signal_type': signal.signal_type,
                'signal_value': signal.signal_type.value,
                'above_upper': row['Close'] > self.upper_threshold,
                'below_lower': row['Close'] < self.lower_threshold
            })
            
            # Process position logic
            if current_position == 0:  # Not in a position
                if signal.signal_type == SignalType.BUY:
                    current_position = 1  # Enter long
                elif signal.signal_type == SignalType.SELL:
                    current_position = -1  # Enter short
            
            elif current_position == 1:  # Long position
                if signal.signal_type == SignalType.SELL or signal.signal_type == SignalType.NEUTRAL:
                    current_position = 0  # Exit long
            
            elif current_position == -1:  # Short position
                if signal.signal_type == SignalType.BUY or signal.signal_type == SignalType.NEUTRAL:
                    current_position = 0  # Exit short
            
            # Log the position
            self.positions.append({
                'index': i,
                'timestamp': row['timestamp'],
                'close': row['Close'],
                'position': current_position
            })
        
        return pd.DataFrame(self.signals), pd.DataFrame(self.positions)
    
    def compare_with_backtest(self, backtest_trades):
        """
        Compare traced signals with actual backtest trades.
        
        Args:
            backtest_trades: List of trades produced by the backtester
            
        Returns:
            DataFrame with comparison results
        """
        # Create a dictionary of trade entry/exit times for easier lookup
        trade_entries = {}
        trade_exits = {}
        
        for trade in backtest_trades:
            entry_time = trade[0]
            exit_time = trade[3]
            trade_type = trade[1]  # 'long' or 'short'
            
            trade_entries[entry_time] = trade_type
            trade_exits[exit_time] = trade_type
        
        # Add trade entry/exit flags to our signals dataframe
        signals_df = pd.DataFrame(self.signals)
        signals_df['trade_entry'] = signals_df['timestamp'].apply(
            lambda x: trade_entries.get(x, None))
        signals_df['trade_exit'] = signals_df['timestamp'].apply(
            lambda x: trade_exits.get(x, None))
        
        # Check for discrepancies between signal type and trade entry/exit
        signals_df['signal_matches_trade'] = None
        
        for i, row in signals_df.iterrows():
            # Check entries
            if row['trade_entry'] == 'long':
                signals_df.at[i, 'signal_matches_trade'] = row['signal_type'] == SignalType.BUY
            elif row['trade_entry'] == 'short':
                signals_df.at[i, 'signal_matches_trade'] = row['signal_type'] == SignalType.SELL
            
            # Check exits (logic depends on your exit rules)
            elif row['trade_exit'] == 'long' and i > 0:
                # For long exits, current signal should be SELL or NEUTRAL
                prev_signal = signals_df.at[i-1, 'signal_type']
                curr_signal = row['signal_type']
                signals_df.at[i, 'signal_matches_trade'] = (
                    curr_signal == SignalType.SELL or 
                    curr_signal == SignalType.NEUTRAL or
                    prev_signal != curr_signal
                )
            elif row['trade_exit'] == 'short' and i > 0:
                # For short exits, current signal should be BUY or NEUTRAL
                prev_signal = signals_df.at[i-1, 'signal_type']
                curr_signal = row['signal_type']
                signals_df.at[i, 'signal_matches_trade'] = (
                    curr_signal == SignalType.BUY or 
                    curr_signal == SignalType.NEUTRAL or
                    prev_signal != curr_signal
                )
        
        return signals_df
    
    def visualize_trace(self, signals_df=None, positions_df=None, backtest_trades=None):
        """
        Visualize the signal trace with positions and trades.
        """
        if signals_df is None or positions_df is None:
            signals_df, positions_df = self.trace_signals()
        
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Price and thresholds
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(signals_df['timestamp'], signals_df['close'], label='Close Price')
        ax1.axhline(y=self.upper_threshold, color='green', linestyle='--', 
                   label=f'Upper Threshold ({self.upper_threshold})')
        ax1.axhline(y=self.lower_threshold, color='red', linestyle='--', 
                   label=f'Lower Threshold ({self.lower_threshold})')
        ax1.set_title('Price and Thresholds')
        ax1.legend()
        ax1.set_xticklabels([])
        
        # Subplot 2: Signals
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(signals_df['timestamp'], signals_df['signal_value'], label='Signal Value')
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['SELL', 'NEUTRAL', 'BUY'])
        ax2.set_title('Signal Values')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xticklabels([])
        
        # Subplot 3: Positions
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(positions_df['timestamp'], positions_df['position'], label='Position')
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['SHORT', 'FLAT', 'LONG'])
        ax3.set_title('Position')
        ax3.legend()
        ax3.grid(True)
        
        # Add trade markers if provided
        if backtest_trades:
            # Add trade entries to the price chart
            for trade in backtest_trades:
                entry_time = trade[0]
                entry_price = trade[2]
                trade_type = trade[1]
                
                if trade_type == 'long':
                    ax1.scatter(entry_time, entry_price, color='green', marker='^', s=100)
                else:  # short
                    ax1.scatter(entry_time, entry_price, color='red', marker='v', s=100)
                
                # Add exit points
                exit_time = trade[3]
                exit_price = trade[4]
                
                if trade_type == 'long':
                    ax1.scatter(exit_time, exit_price, color='blue', marker='v', s=100)
                else:  # short
                    ax1.scatter(exit_time, exit_price, color='orange', marker='^', s=100)
        
        plt.tight_layout()
        plt.savefig('signal_trace_visualization.png')
        plt.close()
        print("Saved signal trace visualization to signal_trace_visualization.png")
        
        return signals_df, positions_df

def run_detailed_analysis(data_df, upper_threshold, lower_threshold, backtest_results=None):
    """
    Run a detailed analysis of the backtesting process.
    
    Args:
        data_df: DataFrame containing price data
        upper_threshold: Upper price threshold
        lower_threshold: Lower price threshold
        backtest_results: Optional backtest results to compare against
    """
    print("="*80)
    print("DETAILED BACKTEST ANALYSIS")
    print("="*80)
    
    # Create the rule and tracer
    rule = ThresholdRule({
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold
    })
    
    tracer = SignalTracer(data_df, rule, upper_threshold, lower_threshold)
    
    # Trace signals and positions
    print("\nTracing signals and positions...")
    signals_df, positions_df = tracer.trace_signals()
    
    # If backtest results are provided, compare them
    if backtest_results and 'trades' in backtest_results:
        print("\nComparing with backtest results...")
        comparison_df = tracer.compare_with_backtest(backtest_results['trades'])
        
        # Look for discrepancies
        discrepancies = comparison_df[
            (comparison_df['trade_entry'].notna() | comparison_df['trade_exit'].notna()) & 
            (comparison_df['signal_matches_trade'] == False)
        ]
        
        if len(discrepancies) > 0:
            print(f"\n⚠️ Found {len(discrepancies)} discrepancies between signals and trades:")
            for i, row in discrepancies.iterrows():
                print(f"  Bar {row['index']}, Time: {row['timestamp']}, "
                      f"Close: {row['close']:.2f}, Signal: {row['signal_type'].name}")
                if row['trade_entry'] is not None:
                    print(f"    Trade Entry: {row['trade_entry']}")
                if row['trade_exit'] is not None:
                    print(f"    Trade Exit: {row['trade_exit']}")
        else:
            print("✅ No discrepancies found between signals and trades")
            
        # Analyze trade characteristics
        if backtest_results['num_trades'] > 0:
            print("\nTrade characteristics:")
            trades = backtest_results['trades']
            
            # Direction distribution
            long_trades = [t for t in trades if t[1] == 'long']
            short_trades = [t for t in trades if t[1] == 'short']
            
            print(f"  Long trades: {len(long_trades)} ({len(long_trades)/len(trades)*100:.1f}%)")
            print(f"  Short trades: {len(short_trades)} ({len(short_trades)/len(trades)*100:.1f}%)")
            
            # Returns analysis
            returns = [t[5] for t in trades]
            percent_returns = [(np.exp(r) - 1) * 100 for r in returns]
            
            print(f"  Average return: {np.mean(percent_returns):.2f}%")
            print(f"  Median return: {np.median(percent_returns):.2f}%")
            print(f"  Max return: {max(percent_returns):.2f}%")
            print(f"  Min return: {min(percent_returns):.2f}%")
            print(f"  Win rate: {sum(1 for r in returns if r > 0)/len(returns):.2%}")
            
            # Create return distribution chart
            plt.figure(figsize=(10, 6))
            plt.hist(percent_returns, bins=20, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Trade Return Distribution')
            plt.xlabel('Return (%)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig('trade_return_distribution.png')
            plt.close()
            print("Saved trade return distribution chart to trade_return_distribution.png")
    
    # Visualize the trace
    print("\nVisualizing signal trace...")
    tracer.visualize_trace(signals_df, positions_df, 
                          backtest_results['trades'] if backtest_results else None)
    
    # Print signal statistics
    print("\nSignal statistics:")
    signal_counts = signals_df['signal_type'].value_counts()
    total_signals = len(signals_df)
    
    for signal_type in SignalType:
        count = signal_counts.get(signal_type, 0)
        percentage = count / total_signals * 100
        print(f"  {signal_type.name}: {count} ({percentage:.1f}%)")
    
    # Analyze position transitions
    print("\nPosition transitions:")
    transitions = []
    
    for i in range(1, len(positions_df)):
        prev_pos = positions_df.iloc[i-1]['position']
        curr_pos = positions_df.iloc[i]['position']
        
        if prev_pos != curr_pos:
            transitions.append((prev_pos, curr_pos))
    
    transition_counts = {}
    for t in transitions:
        transition_counts[t] = transition_counts.get(t, 0) + 1
    
    for (prev, curr), count in sorted(transition_counts.items()):
        prev_label = "LONG" if prev == 1 else "SHORT" if prev == -1 else "FLAT"
        curr_label = "LONG" if curr == 1 else "SHORT" if curr == -1 else "FLAT"
        print(f"  {prev_label} → {curr_label}: {count}")
    
    print("\n" + "="*80)
    print("✅ DETAILED ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Example usage
    from test_data_generator import generate_test_data
    from validation_script import run_backtest_validation
    
    # Generate test data
    df = generate_test_data()
    
    # Run backtester
    from backtester import Backtester, BarEvent
    
    threshold_rule = ThresholdRule({
        'upper_threshold': 110.0,
        'lower_threshold': 90.0
    })
    strategy = SimpleStrategy(threshold_rule)
    data_handler = SimpleDataHandler(df)
    
    backtester = Backtester(data_handler, strategy)
    results = backtester.run(use_test_data=True)
    
    # Run detailed analysis
    run_detailed_analysis(df, 110.0, 90.0, results)
