"""
Backtester Validation Script

This script performs comprehensive validation of the backtesting system
to ensure accuracy of trade execution, return calculations, and optimization metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import random
import time

# Import your trading system components
from data_handler import CSVDataHandler
from backtester import Backtester, BarEvent
from signals import Signal, SignalType
from strategy import TopNStrategy
from rule_system import EventDrivenRuleSystem

# Create a simple mock data handler for testing
class MockDataHandler:
    def __init__(self, data):
        """Initialize with predetermined data for testing."""
        self.train_data = data[:int(len(data)*0.8)]
        self.test_data = data[int(len(data)*0.8):]
        self.reset_train()
        self.reset_test()
        
    def get_next_train_bar(self):
        """Get next bar from training data."""
        if self.train_index < len(self.train_data):
            bar = self.train_data[self.train_index]
            self.train_index += 1
            return bar
        return None
        
    def get_next_test_bar(self):
        """Get next bar from test data."""
        if self.test_index < len(self.test_data):
            bar = self.test_data[self.test_index]
            self.test_index += 1
            return bar
        return None
        
    def reset_train(self):
        """Reset training data index."""
        self.train_index = 0
        
    def reset_test(self):
        """Reset test data index."""
        self.test_index = 0

class DeterministicRule:
    """
    A rule that generates predetermined signals for testing.
    """
    def __init__(self, signals_sequence=None):
        self.signals_sequence = signals_sequence or [1, 0, -1, 0, 1]  # Default sequence
        self.index = 0
        self.rule_id = "DeterministicRule"
        
    def on_bar(self, bar):
        # Generate signal based on predetermined sequence
        if self.index < len(self.signals_sequence):
            signal_value = self.signals_sequence[self.index]
            self.index += 1
            
            if signal_value == 1:
                signal_type = SignalType.BUY
            elif signal_value == -1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL
                
            return Signal(
                timestamp=bar["timestamp"],
                signal_type=signal_type,
                price=bar["Close"],
                rule_id=self.rule_id
            )
        else:
            # Cycle back to beginning if we've gone through the sequence
            self.index = 0
            return Signal(
                timestamp=bar["timestamp"],
                signal_type=SignalType.NEUTRAL,
                price=bar["Close"],
                rule_id=self.rule_id
            )
    
    def reset(self):
        """Reset rule state."""
        self.index = 0

class AlwaysBuyRule:
    """
    A rule that always generates buy signals for testing.
    """
    def __init__(self):
        self.rule_id = "AlwaysBuyRule"
        
    def on_bar(self, bar):
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=SignalType.BUY,
            price=bar["Close"],
            rule_id=self.rule_id
        )
    
    def reset(self):
        """Reset rule state."""
        pass

class BuyAndHoldStrategy:
    """
    A simple buy and hold strategy for testing.
    """
    def __init__(self):
        self.position = 0
        
    def on_bar(self, event):
        bar = event.bar
        
        # Buy on first bar, then hold
        if self.position == 0:
            self.position = 1
            return Signal(
                timestamp=bar["timestamp"],
                signal_type=SignalType.BUY,
                price=bar["Close"],
                rule_id="buy_and_hold"
            )
        else:
            return Signal(
                timestamp=bar["timestamp"],
                signal_type=SignalType.BUY,  # Keep the long signal on
                price=bar["Close"],
                rule_id="buy_and_hold"
            )
    
    def reset(self):
        """Reset strategy state."""
        self.position = 0

class PerfectForesightRule:
    """
    A rule that uses future price information to make perfect trading decisions.
    This is for testing only - would be impossible in real trading.
    """
    def __init__(self, future_bars=1):
        self.rule_id = "PerfectForesightRule"
        self.future_bars = future_bars
        self.all_data = []
        self.current_index = 0
        
    def set_data(self, data):
        """Set full data history for look-ahead."""
        self.all_data = data
        self.current_index = 0
        
    def on_bar(self, bar):
        # Find current index
        for i, b in enumerate(self.all_data):
            if b["timestamp"] == bar["timestamp"]:
                self.current_index = i
                break
                
        # Look ahead to future price
        future_index = min(self.current_index + self.future_bars, len(self.all_data) - 1)
        if future_index > self.current_index:
            future_price = self.all_data[future_index]["Close"]
            current_price = bar["Close"]
            
            # Generate signal based on future price
            if future_price > current_price * 1.01:  # 1% threshold
                signal_type = SignalType.BUY
            elif future_price < current_price * 0.99:  # -1% threshold
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL
        else:
            signal_type = SignalType.NEUTRAL
            
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.rule_id
        )
    
    def reset(self):
        """Reset rule state."""
        self.current_index = 0

def generate_mock_data(num_bars=100, trend=0.0001, volatility=0.001, start_price=100):
    """
    Generate mock price data for testing.
    
    Args:
        num_bars: Number of bars to generate
        trend: Daily trend component (e.g., 0.0001 for slight uptrend)
        volatility: Daily volatility
        start_price: Starting price
        
    Returns:
        list: List of bar dictionaries
    """
    # Start with current datetime and go backward
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_bars)
    
    # Generate dates
    dates = [start_date + timedelta(days=i) for i in range(num_bars)]
    
    # Generate prices with trend and random component
    prices = [start_price]
    for i in range(1, num_bars):
        random_component = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + trend + random_component)
        prices.append(new_price)
    
    # Create bar dictionaries
    bars = []
    for i in range(num_bars):
        # Calculate high/low based on close with some randomness
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = prices[i-1] if i > 0 else close * (1 - np.random.normal(0, volatility/2))
        
        bar = {
            "timestamp": dates[i].strftime("%Y-%m-%d"),
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.randint(1000, 100000)
        }
        bars.append(bar)
    
    return bars

def test_backtest_accuracy():
    """
    Test the accuracy of backtest calculations using known input/output scenarios.
    """
    print("\n=== Testing Backtest Calculation Accuracy ===")
    
    # Create a deterministic dataset where we know the expected outcome
    data = generate_mock_data(num_bars=20, trend=0.001, volatility=0.0001, start_price=100)
    
    # Print the first few bars
    print("Sample data:")
    for i, bar in enumerate(data[:5]):
        print(f"Bar {i}: {bar['timestamp']} - Close: {bar['Close']:.2f}")
    
    # Create a test strategy with predetermined signals
    signals = [1, 1, 0, -1, -1, 0, 1, 1, 0, 0]  # Buy, buy, neutral, sell, sell, neutral, buy, buy, neutral, neutral
    test_rule = DeterministicRule(signals)
    test_strategy = TopNStrategy([test_rule])
    
    # Create data handler and backtester
    data_handler = MockDataHandler(data)
    backtester = Backtester(data_handler, test_strategy)
    
    # Run backtest
    results = backtester.run(use_test_data=False)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_percent_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Print trades
    print("\nTrades:")
    for i, trade in enumerate(results['trades']):
        print(f"Trade {i+1}: {trade[0]} - {trade[3]}, {trade[1]}, Entry: {trade[2]:.2f}, Exit: {trade[4]:.2f}, Return: {trade[5]:.4f}")
    
    # Manually calculate expected returns for validation
    print("\nManually verifying trade returns:")
    for i, trade in enumerate(results['trades']):
        if trade[1] == 'long':
            expected_return = math.log(trade[4] / trade[2])
        else:  # short
            expected_return = math.log(trade[2] / trade[4])
        
        actual_return = trade[5]
        is_correct = abs(expected_return - actual_return) < 0.0001
        print(f"Trade {i+1}: Expected return: {expected_return:.4f}, Actual: {actual_return:.4f}, Correct: {is_correct}")
    
    # Calculate expected total return
    expected_total_log_return = sum(trade[5] for trade in results['trades'])
    expected_total_return = (math.exp(expected_total_log_return) - 1) * 100
    
    print(f"\nExpected total log return: {expected_total_log_return:.4f}")
    print(f"Expected total return: {expected_total_return:.2f}%")
    print(f"Actual total return: {results['total_percent_return']:.2f}%")
    print(f"Return calculation correct: {abs(expected_total_return - results['total_percent_return']) < 0.01}")
    
    return results

def test_sharpe_calculation():
    """
    Test the accuracy of Sharpe ratio calculations.
    """
    print("\n=== Testing Sharpe Ratio Calculation ===")
    
    # Create known trades with predictable outcomes
    # Format: (entry_time, direction, entry_price, exit_time, exit_price, log_return)
    
    # Scenario 1: Consistent positive returns (should have high Sharpe)
    consistent_trades = []
    start_date = datetime(2023, 1, 1)
    for i in range(10):
        entry_time = start_date + timedelta(days=i*7)
        exit_time = entry_time + timedelta(days=1)
        entry_price = 100
        exit_price = 101  # 1% return each time
        log_return = math.log(exit_price / entry_price)
        
        consistent_trades.append((
            entry_time.strftime("%Y-%m-%d"),
            'long',
            entry_price,
            exit_time.strftime("%Y-%m-%d"),
            exit_price,
            log_return
        ))
    
    # Scenario 2: Volatile returns (should have lower Sharpe)
    volatile_trades = []
    for i in range(10):
        entry_time = start_date + timedelta(days=i*7)
        exit_time = entry_time + timedelta(days=1)
        entry_price = 100
        
        # Alternate between gains and losses
        if i % 2 == 0:
            exit_price = 105  # 5% gain
        else:
            exit_price = 96   # 4% loss
        
        log_return = math.log(exit_price / entry_price)
        
        volatile_trades.append((
            entry_time.strftime("%Y-%m-%d"),
            'long',
            entry_price,
            exit_time.strftime("%Y-%m-%d"),
            exit_price,
            log_return
        ))
    
    # Scenario 3: Negative returns (should have negative Sharpe)
    negative_trades = []
    for i in range(10):
        entry_time = start_date + timedelta(days=i*7)
        exit_time = entry_time + timedelta(days=1)
        entry_price = 100
        exit_price = 99  # 1% loss each time
        log_return = math.log(exit_price / entry_price)
        
        negative_trades.append((
            entry_time.strftime("%Y-%m-%d"),
            'long',
            entry_price,
            exit_time.strftime("%Y-%m-%d"),
            exit_price,
            log_return
        ))
    
    # Create backtester instances for each scenario
    backtester1 = Backtester(None, None)
    backtester1.trades = consistent_trades
    
    backtester2 = Backtester(None, None)
    backtester2.trades = volatile_trades
    
    backtester3 = Backtester(None, None)
    backtester3.trades = negative_trades
    
    # Calculate Sharpe ratios
    sharpe1 = backtester1.calculate_sharpe()
    sharpe2 = backtester2.calculate_sharpe()
    sharpe3 = backtester3.calculate_sharpe()
    
    print(f"Consistent positive returns Sharpe: {sharpe1:.4f}")
    print(f"Volatile returns Sharpe: {sharpe2:.4f}")
    print(f"Consistent negative returns Sharpe: {sharpe3:.4f}")
    
    # Verify expected relationships
    print(f"\nExpected: Sharpe1 > Sharpe2 > Sharpe3")
    print(f"Actual: {sharpe1:.4f} > {sharpe2:.4f} > {sharpe3:.4f}")
    print(f"Relationship correct: {sharpe1 > sharpe2 > sharpe3}")
    
    # Try calculating Sharpe with edge cases
    edge_case_trades = consistent_trades[:1]  # Only one trade
    backtester_edge = Backtester(None, None)
    backtester_edge.trades = edge_case_trades
    
    try:
        sharpe_edge = backtester_edge.calculate_sharpe()
        print(f"\nSharpe with only one trade: {sharpe_edge:.4f}")
    except Exception as e:
        print(f"\nException with one trade: {str(e)}")
    
    return sharpe1, sharpe2, sharpe3

def test_market_condition_impact():
    """
    Test how the backtester performs in different market conditions.
    """
    print("\n=== Testing Impact of Market Conditions ===")
    
    # Create 3 different datasets with distinct characteristics
    # 1. Uptrend
    uptrend_data = generate_mock_data(num_bars=100, trend=0.002, volatility=0.005, start_price=100)
    
    # 2. Downtrend
    downtrend_data = generate_mock_data(num_bars=100, trend=-0.002, volatility=0.005, start_price=100)
    
    # 3. Sideways/Choppy
    sideways_data = generate_mock_data(num_bars=100, trend=0.0, volatility=0.008, start_price=100)
    
    # Print characteristic of each dataset
    print("Market conditions:")
    print(f"Uptrend: Start={uptrend_data[0]['Close']:.2f}, End={uptrend_data[-1]['Close']:.2f}, Change={(uptrend_data[-1]['Close']/uptrend_data[0]['Close']-1)*100:.2f}%")
    print(f"Downtrend: Start={downtrend_data[0]['Close']:.2f}, End={downtrend_data[-1]['Close']:.2f}, Change={(downtrend_data[-1]['Close']/downtrend_data[0]['Close']-1)*100:.2f}%")
    print(f"Sideways: Start={sideways_data[0]['Close']:.2f}, End={sideways_data[-1]['Close']:.2f}, Change={(sideways_data[-1]['Close']/sideways_data[0]['Close']-1)*100:.2f}%")
    
    # Test with different strategies
    strategies = {
        "Buy and Hold": BuyAndHoldStrategy(),
        "Always Buy": TopNStrategy([AlwaysBuyRule()]),
        "Deterministic": TopNStrategy([DeterministicRule()])
    }
    
    # Run tests on each market condition with each strategy
    results = {}
    
    for strategy_name, strategy in strategies.items():
        strategy_results = {}
        
        # Test on uptrend
        data_handler = MockDataHandler(uptrend_data)
        backtester = Backtester(data_handler, strategy)
        uptrend_result = backtester.run(use_test_data=False)
        strategy_results['uptrend'] = uptrend_result
        
        # Test on downtrend
        strategy.reset()
        data_handler = MockDataHandler(downtrend_data)
        backtester = Backtester(data_handler, strategy)
        downtrend_result = backtester.run(use_test_data=False)
        strategy_results['downtrend'] = downtrend_result
        
        # Test on sideways
        strategy.reset()
        data_handler = MockDataHandler(sideways_data)
        backtester = Backtester(data_handler, strategy)
        sideways_result = backtester.run(use_test_data=False)
        strategy_results['sideways'] = sideways_result
        
        results[strategy_name] = strategy_results
    
    # Print results
    print("\nStrategy performance across market conditions:")
    print(f"{'Strategy':<20} {'Market':<10} {'Return':<10} {'Trades':<10} {'Sharpe':<10}")
    print("-" * 60)
    
    for strategy_name, strategy_results in results.items():
        for market, result in strategy_results.items():
            # Calculate Sharpe ratio
            backtester = Backtester(None, None)
            backtester.trades = result['trades']
            sharpe = backtester.calculate_sharpe() if result['trades'] else 0
            
            print(f"{strategy_name:<20} {market:<10} {result['total_percent_return']:>8.2f}% {result['num_trades']:>10} {sharpe:>9.4f}")
    
    # Return the results dictionary for further analysis
    return results

def test_corner_cases():
    """
    Test the backtester with corner cases and edge conditions.
    """
    print("\n=== Testing Corner Cases ===")
    
    # 1. No trades case
    print("\n1. Testing with no trades:")
    no_signal_rule = DeterministicRule([0, 0, 0, 0, 0])  # All neutral signals
    no_trade_strategy = TopNStrategy([no_signal_rule])
    data = generate_mock_data(num_bars=20)
    data_handler = MockDataHandler(data)
    backtester = Backtester(data_handler, no_trade_strategy)
    no_trade_result = backtester.run(use_test_data=False)
    
    print(f"No trade result - Trades: {no_trade_result['num_trades']}, Return: {no_trade_result['total_percent_return']:.2f}%")
    
    # 2. Single trade case
    print("\n2. Testing with a single trade:")
    single_trade_rule = DeterministicRule([1, 0, 0, 0, 0])  # One buy signal, then all neutral
    single_trade_strategy = TopNStrategy([single_trade_rule])
    backtester = Backtester(data_handler, single_trade_strategy)
    single_trade_result = backtester.run(use_test_data=False)
    
    print(f"Single trade result - Trades: {single_trade_result['num_trades']}, Return: {single_trade_result['total_percent_return']:.2f}%")
    if single_trade_result['trades']:
        print(f"Trade details: {single_trade_result['trades'][0]}")
    
    # 3. Extreme price moves
    print("\n3. Testing with extreme price moves:")
    # Create data with an extreme price spike
    extreme_data = generate_mock_data(num_bars=20, trend=0, volatility=0.001, start_price=100)
    # Insert a price spike
    extreme_data[10]['Close'] = extreme_data[9]['Close'] * 2  # 100% price spike
    extreme_data[10]['High'] = extreme_data[10]['Close'] * 1.05
    extreme_data[11]['Close'] = extreme_data[10]['Close'] * 0.5  # 50% price drop
    extreme_data[11]['Low'] = extreme_data[11]['Close'] * 0.95
    
    print(f"Price spike: {extreme_data[9]['Close']:.2f} -> {extreme_data[10]['Close']:.2f} -> {extreme_data[11]['Close']:.2f}")
    
    # Test with a strategy that would trade during the spike
    spike_rule = DeterministicRule([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    spike_strategy = TopNStrategy([spike_rule])
    data_handler = MockDataHandler(extreme_data)
    backtester = Backtester(data_handler, spike_strategy)
    spike_result = backtester.run(use_test_data=False)
    
    print(f"Extreme price result - Trades: {spike_result['num_trades']}, Return: {spike_result['total_percent_return']:.2f}%")
    if spike_result['trades']:
        print(f"Trade details: {spike_result['trades'][0]}")
    
    # 4. Test with incomplete trade at the end
    print("\n4. Testing with incomplete trade at the end:")
    incomplete_rule = DeterministicRule([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    incomplete_strategy = TopNStrategy([incomplete_rule])
    data_handler = MockDataHandler(data)
    backtester = Backtester(data_handler, incomplete_strategy)
    incomplete_result = backtester.run(use_test_data=False)
    
    print(f"Incomplete trade result - Trades: {incomplete_result['num_trades']}, Return: {incomplete_result['total_percent_return']:.2f}%")
    
    return {
        'no_trade': no_trade_result,
        'single_trade': single_trade_result,
        'spike': spike_result,
        'incomplete': incomplete_result
    }

def test_perfect_foresight():
    """
    Test a perfect foresight strategy to establish theoretical maximum performance.
    """
    print("\n=== Testing Perfect Foresight Strategy ===")
    
    # Generate data for testing
    data = generate_mock_data(num_bars=100, trend=0.0005, volatility=0.01, start_price=100)
    
    # Create perfect foresight rule
    perfect_rule = PerfectForesightRule(future_bars=1)
    perfect_rule.set_data(data)
    perfect_strategy = TopNStrategy([perfect_rule])
    
    # Create data handler and run backtest
    data_handler = MockDataHandler(data)
    backtester = Backtester(data_handler, perfect_strategy)
    perfect_result = backtester.run(use_test_data=False)
    
    # Calculate Sharpe ratio
    sharpe = backtester.calculate_sharpe()
    
    print(f"Perfect foresight strategy results:")
    print(f"Total Return: {perfect_result['total_percent_return']:.2f}%")
    print(f"Number of Trades: {perfect_result['num_trades']}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Calculate win rate
    if perfect_result['num_trades'] > 0:
        winning_trades = sum(1 for trade in perfect_result['trades'] if trade[5] > 0)
        win_rate = winning_trades / perfect_result['num_trades']
        print(f"Win Rate: {win_rate:.2%}")
    
    # Compare to buy and hold
    buy_hold_strategy = BuyAndHoldStrategy()
    data_handler = MockDataHandler(data)
    backtester = Backtester(data_handler, buy_hold_strategy)
    buy_hold_result = backtester.run(use_test_data=False)
    
    print(f"\nBuy and hold strategy results:")
    print(f"Total Return: {buy_hold_result['total_percent_return']:.2f}%")
    
    # Theoretical maximum return (if we could perfectly time every price move)
    theoretical_max = 1.0
    for i in range(1, len(data)):
        price_change = data[i]['Close'] / data[i-1]['Close']
        theoretical_max *= max(price_change, 1/price_change)  # Take the best of going long or short
    
    theoretical_max_return = (theoretical_max - 1) * 100
    print(f"\nTheoretical maximum return (perfect timing of every move): {theoretical_max_return:.2f}%")
    
    return perfect_result, buy_hold_result, theoretical_max_return

def test_strategy_stability():
    """
    Test the stability of strategies across multiple identical runs.
    """
    print("\n=== Testing Strategy Stability ===")
    
    # Generate fixed data
    data = generate_mock_data(num_bars=100, trend=0.001, volatility=0.01, start_price=100)
    
    # Create a simple deterministic strategy
    rule = DeterministicRule()
    strategy = TopNStrategy([rule])
    
    # Run multiple times with same data and strategy
    results = []
    for i in range(5):
        data_handler = MockDataHandler(data)
        backtester = Backtester(data_handler, strategy)
        rule.reset()  # Ensure rule is reset before each run
        result = backtester.run(use_test_data=False)
        results.append(result)
        
        print(f"Run {i+1}: Return: {result['total_percent_return']:.2f}%, Trades: {result['num_trades']}")
    
    # Check if all results are identical
    returns = [result['total_percent_return'] for result in results]
    trade_counts = [result['num_trades'] for result in results]
    
    is_stable = (len(set(returns)) == 1 and len(set(trade_counts)) == 1)
    print(f"\nStrategy stability: {'Stable' if is_stable else 'Unstable'}")
    print(f"Return values: {returns}")
    print(f"Trade counts: {trade_counts}")
    
    # If not stable, try to identify why
    if not is_stable:
        print("\nAnalyzing instability...")
        
        # Check for random number usage in strategy or backtester
        # This is more conceptual as we can't easily detect this programmatically
        print("Potential causes:")
        print("1. Random number generation in strategy or rule")
        print("2. Time-dependent calculations")
        print("3. State not being properly reset between runs")
    
    return is_stable, results

def test_train_test_consistency():
    """
    Test consistency between training and testing splits.
    """
    print("\n=== Testing Train/Test Consistency ===")
    
    # Generate data
    all_data = generate_mock_data(num_bars=100, trend=0.001, volatility=0.01, start_price=100)
    
    # Create data handler
    data_handler = MockDataHandler(all_data)
    
    # Count bars in training and testing sets
    train_count = 0
    train_bars = []
    data_handler.reset_train()
    while True:
        bar = data_handler.get_next_train_bar()
        if bar is None:
            break
        train_count += 1
        train_bars.append(bar)
    
    test_count = 0
    test_bars = []
    data_handler.reset_test()
    while True:
        bar = data_handler.get_next_test_bar()
        if bar is None:
            break
        test_count += 1
        test_bars.append(bar)
    
    # Check split ratio
    total_bars = train_count + test_count
    train_ratio = train_count / total_bars if total_bars > 0 else 0
    
    print(f"Total bars: {total_bars}")
    print(f"Training bars: {train_count} ({train_ratio:.2%})")
    print(f"Testing bars: {test_count} ({1-train_ratio:.2%})")
    
    # Check for data leakage
    has_overlap = any(t1['timestamp'] == t2['timestamp'] for t1 in train_bars for t2 in test_bars)
    print(f"Data leakage test: {'Failed - overlap detected' if has_overlap else 'Passed - no overlap'}")
    
    # Check for chronological ordering
    train_dates = [datetime.strptime(bar['timestamp'], "%Y-%m-%d") for bar in train_bars]
    test_dates = [datetime.strptime(bar['timestamp'], "%Y-%m-%d") for bar in test_bars]
    
    train_ordered = all(train_dates[i] <= train_dates[i+1] for i in range(len(train_dates)-1))
    test_ordered = all(test_dates[i] <= test_dates[i+1] for i in range(len(test_dates)-1))
    
    print(f"Chronological ordering: {'Correct' if train_ordered and test_ordered else 'Incorrect'}")
    
    # Verify test data comes after train data
    if train_dates and test_dates:
        train_test_boundary_correct = max(train_dates) < min(test_dates)
        print(f"Train/test boundary: {'Correct' if train_test_boundary_correct else 'Incorrect'}")
    else:
        print(f"Train/test boundary: Cannot verify - missing data")

def create_summary_report(test_results):
    """
    Create a summary report of all test results.
    """
    print("\n=== Summary Report ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List tests and results
    tests = [
        "Backtest Calculation Accuracy",
        "Sharpe Ratio Calculation",
        "Market Condition Impact",
        "Corner Cases",
        "Perfect Foresight",
        "Strategy Stability",
        "Train/Test Consistency"
    ]
    
    print(f"\n{'Test':<30} {'Status':<15}")
    print("-" * 45)
    
    for test in tests:
        # This is just a placeholder as we don't have actual pass/fail statuses
        # In a real implementation, you'd track test successes/failures
        status = "Run"
        print(f"{test:<30} {status:<15}")
    
    # Add recommendations based on test results
    print("\nRecommendations:")
    print("1. Compare your strategy returns with simple benchmark strategies")
    print("2. Verify Sharpe ratio calculations match expected values")
    print("3. Check for consistent behavior across multiple runs")
    print("4. Validate that train/test split is working correctly")
    print("5. Test with extreme market conditions to check robustness")

def test_with_real_data():
    """
    Run tests with real market data if available.
    """
    print("\n=== Testing with Real Market Data ===")
    
    try:
        # Try to load real data
        filepath = os.path.expanduser("~/mmbt/data/data.csv")
        if not os.path.exists(filepath):
            print(f"Data file not found: {filepath}")
            return None
        
        # Load data using your CSVDataHandler
        data_handler = CSVDataHandler(filepath, train_fraction=0.8)
        
        # Test with a simple strategy
        strategy = BuyAndHoldStrategy()
        backtester = Backtester(data_handler, strategy)
        
        # Run backtest
        results = backtester.run(use_test_data=True)
        
        # Calculate Sharpe
        sharpe = backtester.calculate_sharpe()
        
        # Print results
        print(f"Buy and hold results on real data:")
        print(f"Total Return: {results['total_percent_return']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error testing with real data: {str(e)}")
        return None

def compare_backtester_implementations():
    """
    Compare optimized backtester to baseline implementation if available.
    This can help identify potential issues with optimization.
    """
    print("\n=== Comparing Backtester Implementations ===")
    
    try:
        # Try to import alternative backtester implementation
        # from baseline_backtester import BaselineBacktester
        
        # Since we likely don't have this, we'll simulate the comparison
        # by creating a simplified version of the backtester
        
        class SimpleBacktester:
            """
            A simplified backtester implementation for comparison.
            """
            def __init__(self, data_handler, strategy):
                self.data_handler = data_handler
                self.strategy = strategy
                self.trades = []
            
            def run(self, use_test_data=False):
                # Reset state
                self.trades = []
                current_position = 0
                entry_price = None
                entry_time = None
                
                # Select data source
                if use_test_data:
                    self.data_handler.reset_test()
                    get_next_bar = self.data_handler.get_next_test_bar
                else:
                    self.data_handler.reset_train()
                    get_next_bar = self.data_handler.get_next_train_bar
                
                # Process each bar
                while True:
                    bar_data = get_next_bar()
                    if bar_data is None:
                        break
                    
                    event = BarEvent(bar_data)
                    signal = self.strategy.on_bar(event)
                    
                    if signal is None:
                        continue
                        
                    # Extract signal value
                    if hasattr(signal, 'signal_type'):
                        signal_value = signal.signal_type.value  # Signal object
                    elif isinstance(signal, dict) and 'signal' in signal:
                        signal_value = signal['signal']  # Dictionary format
                    elif isinstance(signal, (int, float)):
                        signal_value = signal  # Numeric signal
                    else:
                        signal_value = 0  # Default to neutral
                    
                    # Process signals
                    timestamp = bar_data['timestamp']
                    price = bar_data['Close']
                    
                    if current_position == 0:  # Not in a position
                        if signal_value == 1:  # Buy signal
                            current_position = 1
                            entry_price = price
                            entry_time = timestamp
                        elif signal_value == -1:  # Sell signal
                            current_position = -1
                            entry_price = price
                            entry_time = timestamp
                    
                    elif current_position == 1:  # In long position
                        if signal_value == -1 or signal_value == 0:  # Sell or neutral
                            # Exit long position
                            log_return = math.log(price / entry_price) if entry_price > 0 else 0
                            self.trades.append((
                                entry_time,
                                'long',
                                entry_price,
                                timestamp,
                                price,
                                log_return
                            ))
                            
                            if signal_value == -1:  # Enter short if sell signal
                                current_position = -1
                                entry_price = price
                                entry_time = timestamp
                            else:
                                current_position = 0
                                entry_price = None
                                entry_time = None
                    
                    elif current_position == -1:  # In short position
                        if signal_value == 1 or signal_value == 0:  # Buy or neutral
                            # Exit short position
                            log_return = math.log(entry_price / price) if price > 0 else 0
                            self.trades.append((
                                entry_time,
                                'short',
                                entry_price,
                                timestamp,
                                price,
                                log_return
                            ))
                            
                            if signal_value == 1:  # Enter long if buy signal
                                current_position = 1
                                entry_price = price
                                entry_time = timestamp
                            else:
                                current_position = 0
                                entry_price = None
                                entry_time = None
                
                # Calculate returns
                total_log_return = sum(trade[5] for trade in self.trades) if self.trades else 0
                total_return = (math.exp(total_log_return) - 1) * 100  # Convert to percentage
                avg_log_return = total_log_return / len(self.trades) if self.trades else 0
                
                return {
                    'total_log_return': total_log_return,
                    'total_percent_return': total_return,
                    'average_log_return': avg_log_return,
                    'num_trades': len(self.trades),
                    'trades': self.trades
                }
            
            def calculate_sharpe(self, risk_free_rate=0):
                """
                A simplified Sharpe calculation for comparison.
                """
                if len(self.trades) < 2:
                    return 0.0
                
                returns = [trade[5] for trade in self.trades]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return == 0:
                    return 0.0
                
                sharpe = (avg_return - risk_free_rate) / std_return
                annualized_sharpe = sharpe * np.sqrt(252)  # Assuming daily data
                
                return annualized_sharpe
        
        # Generate test data
        data = generate_mock_data(num_bars=100, trend=0.001, volatility=0.01, start_price=100)
        data_handler = MockDataHandler(data)
        
        # Create test strategy
        rule = DeterministicRule()
        strategy = TopNStrategy([rule])
        
        # Run backtests with both implementations
        original_backtester = Backtester(data_handler, strategy)
        simplified_backtester = SimpleBacktester(data_handler, strategy)
        
        # Make sure rules and data are in the same state for both tests
        rule.reset()
        data_handler.reset_train()
        
        # Run original implementation
        original_results = original_backtester.run(use_test_data=False)
        original_sharpe = original_backtester.calculate_sharpe()
        
        # Reset for simplified implementation
        rule.reset()
        data_handler.reset_train()
        
        # Run simplified implementation
        simplified_results = simplified_backtester.run(use_test_data=False)
        simplified_sharpe = simplified_backtester.calculate_sharpe()
        
        # Compare results
        print(f"Original Backtester: Return={original_results['total_percent_return']:.2f}%, "
              f"Trades={original_results['num_trades']}, Sharpe={original_sharpe:.4f}")
        
        print(f"Simplified Backtester: Return={simplified_results['total_percent_return']:.2f}%, "
              f"Trades={simplified_results['num_trades']}, Sharpe={simplified_sharpe:.4f}")
        
        # Calculate differences
        return_diff = original_results['total_percent_return'] - simplified_results['total_percent_return']
        trades_diff = original_results['num_trades'] - simplified_results['num_trades']
        sharpe_diff = original_sharpe - simplified_sharpe
        
        print(f"\nDifferences: Return={return_diff:.2f}%, Trades={trades_diff}, Sharpe={sharpe_diff:.4f}")
        
        # Check for significant differences
        return_threshold = 0.01  # 0.01% difference threshold
        if abs(return_diff) > return_threshold or trades_diff != 0:
            print("\nWarning: Significant differences detected between implementations!")
            print("This could indicate calculation errors or inconsistencies.")
            
            # Analyze trade differences
            if original_results['num_trades'] > 0 and simplified_results['num_trades'] > 0:
                print("\nAnalyzing first few trades from each implementation:")
                
                # Compare first few trades
                num_trades_to_compare = min(3, original_results['num_trades'], simplified_results['num_trades'])
                for i in range(num_trades_to_compare):
                    original_trade = original_results['trades'][i]
                    simplified_trade = simplified_results['trades'][i]
                    
                    print(f"\nTrade {i+1}")
                    print(f"Original: {original_trade}")
                    print(f"Simplified: {simplified_trade}")
                    
                    # Check for differences in each component
                    entry_time_match = original_trade[0] == simplified_trade[0]
                    direction_match = original_trade[1] == simplified_trade[1]
                    entry_price_match = abs(original_trade[2] - simplified_trade[2]) < 0.0001
                    exit_time_match = original_trade[3] == simplified_trade[3]
                    exit_price_match = abs(original_trade[4] - simplified_trade[4]) < 0.0001
                    return_match = abs(original_trade[5] - simplified_trade[5]) < 0.0001
                    
                    print(f"Matches: Entry Time={entry_time_match}, Direction={direction_match}, "
                          f"Entry Price={entry_price_match}, Exit Time={exit_time_match}, "
                          f"Exit Price={exit_price_match}, Return={return_match}")
        else:
            print("\nImplementations appear to be consistent.")
        
        return original_results, simplified_results
            
    except Exception as e:
        print(f"Error comparing implementations: {str(e)}")
        return None, None

def main():
    """
    Run all validation tests on the backtester.
    """
    print("=== Backtester Validation Script ===")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Record results for summary
    test_results = {}
    
    # Run tests
    test_results['backtest_accuracy'] = test_backtest_accuracy()
    test_results['sharpe_calculation'] = test_sharpe_calculation()
    test_results['market_condition_impact'] = test_market_condition_impact()
    test_results['corner_cases'] = test_corner_cases()
    test_results['perfect_foresight'] = test_perfect_foresight()
    test_results['strategy_stability'] = test_strategy_stability()
    test_train_test_consistency()
    test_results['real_data'] = test_with_real_data()
    test_results['implementation_comparison'] = compare_backtester_implementations()
    
    # Create summary report
    create_summary_report(test_results)
    
    print("\nBacktester validation complete!")
    return test_results

if __name__ == "__main__":
    main()
