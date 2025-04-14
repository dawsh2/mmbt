#!/usr/bin/env python
"""
Trading System Backtest Runner with Direct Signal-to-Trade Conversion

This script fixes the issue where signals aren't being converted to trades
by implementing a simpler backtesting approach that directly converts signals to trades.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# First try importing from src module, fall back to local implementations if needed
try:
    from src.signals.signal_processing import Signal, SignalType
    have_src_imports = True
    print("Using system imports from src module")
except ImportError:
    have_src_imports = False
    print("Using local implementations")
    
    # Local implementation of SignalType and Signal
    class SignalType:
        BUY = 1
        SELL = -1
        NEUTRAL = 0
    
    class Signal:
        def __init__(self, timestamp, signal_type, price, rule_id=None, confidence=1.0, metadata=None):
            self.timestamp = timestamp
            self.signal_type = signal_type
            self.price = price
            self.rule_id = rule_id
            self.confidence = confidence
            self.metadata = metadata or {}


class CSVDataHandler:
    """Data handler for loading and processing CSV files."""
    
    def __init__(self, filepath, date_column='Date', train_fraction=0.8):
        """
        Initialize with data file path.
        
        Args:
            filepath: Path to CSV file
            date_column: Column name for dates
            train_fraction: Fraction of data to use for training
        """
        self.filepath = filepath
        self.date_column = date_column
        self.train_fraction = train_fraction
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_index = 0
        self.test_index = 0
        
        self._load_data()
        
    def _load_data(self):
        """Load data from CSV file."""
        try:
            # Load the data
            self.data = pd.read_csv(self.filepath)
            
            # Print column names for debugging
            print(f"Data columns: {list(self.data.columns)}")
            
            # Convert date column to datetime
            if self.date_column in self.data.columns:
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
                self.data = self.data.sort_values(self.date_column)
            
            # Check if required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                print("Available columns:", list(self.data.columns))
                
                # Try to map columns if possible
                for req_col in missing_columns:
                    for col in self.data.columns:
                        if req_col.lower() == col.lower():
                            print(f"Mapping {col} to {req_col}")
                            self.data[req_col] = self.data[col]
            
            # Print data sample
            print("\nData sample:")
            print(self.data.head())
            
            # Split into train/test sets
            split_idx = int(len(self.data) * self.train_fraction)
            self.train_data = self.data.iloc[:split_idx].reset_index(drop=True)
            self.test_data = self.data.iloc[split_idx:].reset_index(drop=True)
            
            print(f"Loaded data: {len(self.data)} rows total, "
                  f"{len(self.train_data)} for training, {len(self.test_data)} for testing")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.data = pd.DataFrame()
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
    
    def reset_train(self):
        """Reset training data iterator."""
        self.train_index = 0
    
    def reset_test(self):
        """Reset testing data iterator."""
        self.test_index = 0
    
    def get_next_train_bar(self):
        """Get next bar from training data."""
        if self.train_index >= len(self.train_data):
            return None
        
        bar = self._convert_row_to_bar(self.train_data.iloc[self.train_index])
        self.train_index += 1
        return bar
    
    def get_next_test_bar(self):
        """Get next bar from testing data."""
        if self.test_index >= len(self.test_data):
            return None
        
        bar = self._convert_row_to_bar(self.test_data.iloc[self.test_index])
        self.test_index += 1
        return bar
    
    def _convert_row_to_bar(self, row):
        """Convert DataFrame row to bar dictionary."""
        bar = {}
        
        # Extract OHLCV data if available
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in row:
                bar[col] = row[col]
        
        # Extract timestamp
        if self.date_column in row:
            bar['timestamp'] = row[self.date_column]
        else:
            bar['timestamp'] = datetime.now()
        
        return bar


# Simple moving average calculator
def calculate_sma(prices, period):
    """Calculate simple moving average."""
    return pd.Series(prices).rolling(window=period).mean().values


# Moving average crossover feature
class MACrossoverFeature:
    """Feature that calculates moving average crossover signals."""
    
    def __init__(self, name, fast_period=10, slow_period=30):
        """
        Initialize the MA crossover feature.
        
        Args:
            name: Feature name
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        self.name = name
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.price_history = []
        self.signal_history = []
        
    def calculate(self, bar):
        """
        Calculate feature value from bar data.
        
        Args:
            bar: Bar data dictionary
            
        Returns:
            float: Signal value (-1 to 1)
        """
        # Add current price to history
        self.price_history.append(bar['Close'])
        
        # Need enough history for calculation
        if len(self.price_history) < self.slow_period + 1:
            self.signal_history.append(0)
            return 0
        
        # Calculate moving averages
        prices = self.price_history[-self.slow_period-1:]
        fast_ma = calculate_sma(prices, self.fast_period)[-1]
        slow_ma = calculate_sma(prices, self.slow_period)[-1]
        
        # Previous moving averages
        prev_prices = self.price_history[-self.slow_period-2:-1]
        prev_fast_ma = calculate_sma(prev_prices, self.fast_period)[-1]
        prev_slow_ma = calculate_sma(prev_prices, self.slow_period)[-1]
        
        # Generate crossover signal
        signal = 0
        
        # Bullish crossover (fast crosses above slow)
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            signal = 1
            print(f"BULLISH CROSSOVER DETECTED! Fast MA: {fast_ma:.2f} crossed above Slow MA: {slow_ma:.2f}")
        # Bearish crossover (fast crosses below slow)
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            signal = -1
            print(f"BEARISH CROSSOVER DETECTED! Fast MA: {fast_ma:.2f} crossed below Slow MA: {slow_ma:.2f}")
        
        # Store signal
        self.signal_history.append(signal)
        
        return signal
    
    def reset(self):
        """Reset feature state."""
        self.price_history = []
        self.signal_history = []


class VolatilityFeature:
    """Feature that calculates market volatility."""
    
    def __init__(self, name, period=20, threshold=0.015):
        """
        Initialize the volatility feature.
        
        Args:
            name: Feature name
            period: Lookback period for volatility calculation
            threshold: Volatility threshold for signals
        """
        self.name = name
        self.period = period
        self.threshold = threshold
        self.price_history = []
        
    def calculate(self, bar):
        """
        Calculate feature value from bar data.
        
        Args:
            bar: Bar data dictionary
            
        Returns:
            float: Signal value (-1 to 1)
        """
        # Add current price to history
        self.price_history.append(bar['Close'])
        
        # Need enough history for calculation
        if len(self.price_history) < self.period + 1:
            return 0
        
        # Calculate returns
        prices = np.array(self.price_history[-self.period-1:])
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility
        volatility = np.std(returns)
        
        # Generate signal based on volatility
        signal = 0
        
        # High volatility (potential reversal or breakdown)
        if volatility > self.threshold:
            # Determine direction based on recent price movement
            recent_change = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
            
            if recent_change > 0:
                signal = -0.5  # High volatility after up move suggests caution
            else:
                signal = 0.5   # High volatility after down move could be bottoming
        # Low volatility (potential for breakout)
        elif volatility < self.threshold / 2:
            signal = 0.3  # Lower volatility can precede breakouts, slight bullish bias
        
        return signal
    
    def reset(self):
        """Reset feature state."""
        self.price_history = []


class FeatureRule:
    """Rule that uses a feature to generate signals."""
    
    def __init__(self, feature, buy_threshold=0.5, sell_threshold=-0.5, name=None):
        """
        Initialize the feature-based rule.
        
        Args:
            feature: Feature object that calculates values
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
            name: Rule name
        """
        self.feature = feature
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.name = name or f"{feature.name}_rule"
    
    def on_bar(self, bar):
        """
        Process a bar and generate a signal.
        
        Args:
            bar: Bar data dictionary
            
        Returns:
            Signal object
        """
        # Calculate feature value
        feature_value = self.feature.calculate(bar)
        
        # Determine signal type
        if feature_value > self.buy_threshold:
            signal_type = SignalType.BUY
        elif feature_value < self.sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create signal
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.name,
            confidence=abs(feature_value)
        )
    
    def reset(self):
        """Reset the rule state."""
        # Reset feature if it's stateful
        if hasattr(self.feature, 'reset'):
            self.feature.reset()


class WeightedStrategy:
    """Strategy that combines signals from multiple rules using weights."""
    
    def __init__(self, rules, weights=None, buy_threshold=0.5, sell_threshold=-0.5, name=None):
        """
        Initialize the weighted strategy.
        
        Args:
            rules: List of rule objects
            weights: List of weights for each rule (defaults to equal weighting)
            buy_threshold: Threshold above which to generate a buy signal
            sell_threshold: Threshold below which to generate a sell signal
            name: Strategy name
        """
        self.rules = rules
        self.name = name or "WeightedStrategy"
        
        # Initialize weights (equal by default)
        if weights is None:
            self.weights = np.ones(len(rules)) / len(rules)
        else:
            # Normalize weights to sum to 1
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                self.weights = np.array(weights) / weights_sum
            else:
                # Fallback to equal weights if sum is 0 or negative
                self.weights = np.ones(len(rules)) / len(rules)
        
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_signal = None
    
    def on_bar(self, event):
        """
        Process a bar and generate a weighted signal.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            Signal: Combined signal based on weighted rules
        """
        bar = event.bar if hasattr(event, 'bar') else event
        
        # Get signals from all rules
        combined_signals = []
        rule_signals = {}  # For metadata
        
        for i, rule in enumerate(self.rules):
            signal_object = rule.on_bar(bar)
            
            if signal_object and hasattr(signal_object, 'signal_type'):
                signal_value = signal_object.signal_type
                # Handle both enum and int representations
                if hasattr(signal_value, 'value'):
                    signal_value = signal_value.value
                combined_signals.append(signal_value * self.weights[i])
                rule_signals[getattr(rule, 'name', f'rule_{i}')] = signal_value
            else:
                combined_signals.append(0)
                rule_signals[getattr(rule, 'name', f'rule_{i}')] = 0
        
        # Calculate weighted sum
        weighted_sum = np.sum(combined_signals)
        
        # Determine final signal
        if weighted_sum > self.buy_threshold:
            final_signal_type = SignalType.BUY
        elif weighted_sum < self.sell_threshold:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL
        
        # Create signal object
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id=self.name,
            confidence=min(1.0, abs(weighted_sum)),  # Scale confidence
            metadata={
                "weighted_sum": weighted_sum,
                "rule_signals": rule_signals
            }
        )
        
        return self.last_signal
    
    def reset(self):
        """Reset all rules in the strategy."""
        for rule in self.rules:
            if hasattr(rule, 'reset'):
                rule.reset()
        self.last_signal = None


class Event:
    """Simple event class."""
    def __init__(self, event_type, data=None):
        self.event_type = event_type
        self.data = data
        self.bar = data  # For compatibility with some implementations


class DirectBacktester:
    """
    Backtester that directly converts signals to trades.
    
    This simplified backtester bypasses the complex event/order/execution flow
    and directly creates trades from signals, which is useful for debugging.
    """
    
    def __init__(self, data_handler, strategy, initial_capital=100000, 
                 fixed_position_size=100, slippage_pct=0.0005, commission_pct=0.001):
        """
        Initialize the backtester.
        
        Args:
            data_handler: Data handler providing market data
            strategy: Trading strategy to test
            initial_capital: Initial capital for the backtest
            fixed_position_size: Fixed position size for trades
            slippage_pct: Slippage percentage (both entry and exit)
            commission_pct: Commission percentage (both entry and exit)
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fixed_position_size = fixed_position_size
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        
        # Portfolio tracking
        self.equity = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.position_value = 0
        
        # Trade tracking
        self.trades = []
        self.signals = []
        self.equity_curve = [initial_capital]
        self.positions = []
    
    def run(self, use_test_data=False):
        """
        Run the backtest.
        
        Args:
            use_test_data: Whether to use test data (True) or training data (False)
            
        Returns:
            dict: Backtest results
        """
        # Reset components
        self.strategy.reset()
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.signals = []
        self.equity_curve = [self.initial_capital]
        self.positions = []
        
        # Select data source
        if use_test_data:
            self.data_handler.reset_test()
            next_bar = self.data_handler.get_next_test_bar
        else:
            self.data_handler.reset_train()
            next_bar = self.data_handler.get_next_train_bar
        
        # Current position tracking
        entry_price = None
        entry_time = None
        entry_position = 0
        
        # Process each bar
        bar_count = 0
        signal_count = 0
        
        while True:
            bar = next_bar()
            if bar is None:
                break
                
            bar_count += 1
            
            # Create bar event
            bar_event = Event("BAR", bar)
            
            # Process bar through strategy
            signal = self.strategy.on_bar(bar_event)
            current_price = bar['Close']
            
            # Record signal
            if signal:
                self.signals.append(signal)
                signal_count += 1
                
                # Get signal type (handle both enum and value)
                if hasattr(signal.signal_type, 'value'):
                    signal_type = signal.signal_type.value
                else:
                    signal_type = signal.signal_type
                
                # Close existing position if signal is opposite direction or neutral
                if entry_position != 0:
                    if (entry_position > 0 and signal_type <= 0) or (entry_position < 0 and signal_type >= 0):
                        # Apply slippage
                        exit_price = current_price
                        if entry_position > 0:
                            exit_price *= (1 - self.slippage_pct)  # Selling, so lower price
                        else:
                            exit_price *= (1 + self.slippage_pct)  # Buying back short, so higher price
                            
                        # Calculate P&L
                        if entry_position > 0:
                            pnl = (exit_price - entry_price) * abs(entry_position)
                        else:
                            pnl = (entry_price - exit_price) * abs(entry_position)
                            
                        # Apply commissions
                        commission = abs(exit_price * entry_position) * self.commission_pct
                        pnl -= commission
                        
                        # Update portfolio
                        self.cash += (exit_price * abs(entry_position)) + pnl
                        self.position = 0
                        self.position_value = 0
                        self.equity = self.cash
                        
                        # Record trade
                        # trade = (entry_time, direction, entry_price, exit_time, exit_price, pnl)
                        direction = "long" if entry_position > 0 else "short"
                        log_return = np.log(self.equity / self.equity_curve[-1])
                        trade = (
                            entry_time,
                            direction,
                            entry_price,
                            bar['timestamp'],
                            exit_price,
                            log_return
                        )
                        self.trades.append(trade)
                        
                        # Reset position tracking
                        entry_price = None
                        entry_time = None
                        entry_position = 0
                        
                        print(f"Closed {direction} position at {exit_price:.2f}, P&L: {pnl:.2f}")
                
                # Open new position if signal is directional and no existing position
                if entry_position == 0 and signal_type != 0:
                    # Determine position size (fixed for simplicity)
                    position_size = self.fixed_position_size
                    if signal_type < 0:
                        position_size = -position_size
                        
                    # Apply slippage
                    entry_price = current_price
                    if position_size > 0:
                        entry_price *= (1 + self.slippage_pct)  # Buying, so higher price
                    else:
                        entry_price *= (1 - self.slippage_pct)  # Shorting, so lower price
                        
                    # Calculate position cost with commission
                    position_cost = abs(entry_price * position_size)
                    commission = position_cost * self.commission_pct
                    total_cost = position_cost + commission
                    
                    # Ensure enough cash
                    if total_cost <= self.cash:
                        # Update portfolio
                        self.cash -= total_cost
                        self.position = position_size
                        self.position_value = position_cost
                        
                        # Record entry
                        entry_time = bar['timestamp']
                        entry_position = position_size
                        
                        direction = "long" if position_size > 0 else "short"
                        print(f"Opened {direction} position of {abs(position_size)} shares at {entry_price:.2f}")
            
            # Update portfolio value
            if self.position != 0:
                self.position_value = abs(self.position * current_price)
            self.equity = self.cash + self.position_value
            self.equity_curve.append(self.equity)
            self.positions.append(self.position)
            
        print(f"Processed {bar_count} bars, generated {signal_count} signals, executed {len(self.trades)} trades")
        
        # Calculate performance metrics
        total_return = ((self.equity - self.initial_capital) / self.initial_capital) * 100
        log_return = np.log(self.equity / self.initial_capital)
        
        # Return results
        return {
            'trades': self.trades,
            'num_trades': len(self.trades),
            'signals': self.signals,
            'equity_curve': self.equity_curve,
            'positions': self.positions,
            'total_percent_return': total_return,
            'total_log_return': log_return,
            'final_equity': self.equity
        }


def calculate_metrics(results):
    """
    Calculate performance metrics from backtest results.
    
    Args:
        results: Results dictionary from backtest
        
    Returns:
        dict: Performance metrics
    """
    trades = results['trades']
    equity_curve = results['equity_curve']
    
    if not trades:
        return {
            'total_return': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    # Calculate wins/losses
    returns = [trade[5] for trade in trades]
    wins = sum(1 for r in returns if r > 0)
    losses = len(returns) - wins
    
    # Calculate win rate
    win_rate = wins / len(trades) if trades else 0
    
    # Calculate profit factor (sum of gains / sum of losses)
    gains = sum(r for r in returns if r > 0) if wins > 0 else 0
    losses_sum = abs(sum(r for r in returns if r < 0)) if losses > 0 else 1
    profit_factor = gains / losses_sum if losses_sum > 0 else float('inf')
    
    # Calculate Sharpe ratio
    mean_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if len(returns) > 1 else 1
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'total_return': results['total_percent_return'],
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def plot_results(results, title='Backtest Results', filename=None):
    """
    Plot equity curve and returns distribution.
    
    Args:
        results: Results dictionary from backtest
        title: Plot title
        filename: Optional filename to save the plot
    """
    if not results['trades']:
        print("No trades to plot.")
        return
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot equity curve
    axes[0, 0].plot(results['equity_curve'])
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_xlabel('Bar')
    axes[0, 0].set_ylabel('Equity ($)')
    axes[0, 0].grid(True)
    
    # Plot position sizes over time
    axes[0, 1].plot(results['positions'])
    axes[0, 1].set_title('Position Size')
    axes[0, 1].set_xlabel('Bar')
    axes[0, 1].set_ylabel('Shares')
    axes[0, 1].grid(True)
    
    # Plot returns distribution
    returns = [trade[5] for trade in results['trades']]
    axes[1, 0].hist(returns, bins=20, alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Log Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Plot drawdown
    equity = np.array(results['equity_curve'])
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    
    axes[1, 1].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    axes[1, 1].set_title('Drawdown')
    axes[1, 1].set_xlabel('Bar')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    if filename:
        plt.savefig(filename)
    
    plt.show()


def main():
    """Main function to run the trading system backtest."""
    # Define paths
    data_file = os.path.join("data", "data.csv")
    output_dir = os.path.join("output")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        print("Please put your market data in data/data.csv")
        return
    
    # Create data handler
    print(f"Loading data from {data_file}...")
    data_handler = CSVDataHandler(data_file)
    
    # Create features for our rules
    ma_crossover_feature = MACrossoverFeature(
        name="ma_crossover", 
        fast_period=10,
        slow_period=30
    )
    
    volatility_feature = VolatilityFeature(
        name="volatility", 
        period=20,
        threshold=0.01  # Lower threshold to generate more signals
    )
    
    # Create rules from features
    ma_rule = FeatureRule(
        feature=ma_crossover_feature,
        buy_threshold=0.5,
        sell_threshold=-0.5,
        name="MA_Crossover_Rule"
    )
    
    volatility_rule = FeatureRule(
        feature=volatility_feature,
        buy_threshold=0.1,  # Very low threshold
        sell_threshold=-0.1,
        name="Volatility_Rule"
    )
    
    # Create strategy with rules
    strategy = WeightedStrategy(
        rules=[ma_rule, volatility_rule],
        weights=[0.7, 0.3],
        buy_threshold=0.05,  # Extremely low threshold to generate more signals
        sell_threshold=-0.05,
        name="Market_Strategy"
    )
    
    # Create simple backtester with direct signal-to-trade conversion
    direct_backtester = DirectBacktester(
        data_handler=data_handler,
        strategy=strategy,
        initial_capital=100000,
        fixed_position_size=100,
        slippage_pct=0.0005,  # 5 basis points
        commission_pct=0.001   # 10 basis points
    )
    
    # Run the backtest on training data
    print("\nRunning backtest on training data...")
    train_results = direct_backtester.run(use_test_data=False)
    
    print("\nTraining Backtest Results:")
    print(f"Number of trades: {train_results['num_trades']}")
    print(f"Total return: {train_results['total_percent_return']:.2f}%")
    
    if train_results['num_trades'] > 0:
        # Calculate metrics
        metrics = calculate_metrics(train_results)
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Profit factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Plot results
        plot_results(train_results, 'Training Data Results', 
                    os.path.join(output_dir, 'training_results.png'))
    
    # Run the backtest on test data
    print("\nRunning backtest on test data...")
    test_results = direct_backtester.run(use_test_data=True)
    
    print("\nTest Backtest Results:")
    print(f"Number of trades: {test_results['num_trades']}")
    print(f"Total return: {test_results['total_percent_return']:.2f}%")
    
    if test_results['num_trades'] > 0:
        # Calculate metrics
        metrics = calculate_metrics(test_results)
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Profit factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Plot results
        plot_results(test_results, 'Test Data Results', 
                    os.path.join(output_dir, 'test_results.png'))
    
    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    main()
