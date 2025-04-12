import numpy as np
import math
from signals import SignalType
from memory_profiler import profile


class BarEvent:
    def __init__(self, data):
        self.bar = data

class Backtester:
    def __init__(self, data_handler, strategy):
        self.data_handler = data_handler
        self.strategy = strategy
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None

    def run(self, use_test_data=False):
        """
        Run backtest and return only the necessary results.
        """
        # Reset state
        self.strategy.reset()
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        
        total_log_return = 0.0
        win_count = 0

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
            
            if signal is not None:
                self.signals.append({
                    'timestamp': bar_data['timestamp'],
                    'signal': signal.signal_type.value if hasattr(signal, 'signal_type') else signal,
                    'price': bar_data['Close']
                })
                
                # Process the signal to generate trades
                self._process_signal_for_trades(signal, bar_data['timestamp'], bar_data['Close'])

        # Calculate final metrics
        if self.trades:
            total_log_return = sum(trade['log_return'] for trade in self.trades)
            total_return = (math.exp(total_log_return) - 1) * 100  # Convert to percentage
            avg_log_return = total_log_return / len(self.trades)
            win_count = sum(1 for trade in self.trades if trade['log_return'] > 0)
            win_rate = win_count / len(self.trades) if self.trades else 0
        else:
            total_log_return = 0
            total_return = 0
            avg_log_return = 0
            win_rate = 0

        # Return only necessary results
        return {
            'total_log_return': total_log_return,
            'total_percent_return': total_return,
            'average_log_return': avg_log_return,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'trades': self.trades  # Now a list of dictionaries, not a DataFrame
        }

    def _process_signal_for_trades(self, signal, timestamp, price):
        """
        Process a signal to generate trades, using simple lists instead of DataFrames.
        """
        if signal is None:
            return

        # Extract signal type
        if hasattr(signal, 'signal_type'):
            signal_type = signal.signal_type
        elif isinstance(signal, dict) and 'signal' in signal:
            signal_value = signal['signal']
            signal_type = SignalType.BUY if signal_value == 1 else \
                         SignalType.SELL if signal_value == -1 else SignalType.NEUTRAL
        elif isinstance(signal, int):
            signal_type = SignalType.BUY if signal == 1 else \
                         SignalType.SELL if signal == -1 else SignalType.NEUTRAL
        else:
            signal_type = SignalType.NEUTRAL

        # Handle position entry/exit based on signal
        if self.current_position == 0:
            if signal_type == SignalType.BUY:
                self.current_position = 1
                self.entry_price = price
                self.entry_time = timestamp
            elif signal_type == SignalType.SELL:
                self.current_position = -1
                self.entry_price = price
                self.entry_time = timestamp
        elif self.current_position == 1:
            if signal_type == SignalType.SELL or signal_type == SignalType.NEUTRAL:
                log_return = math.log(price / self.entry_price) if self.entry_price > 0 else 0
                
                self.trades.append({
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': timestamp,
                    'exit_price': price,
                    'log_return': log_return,
                    'position': 'LONG'
                })
                
                self.current_position = 0
        elif self.current_position == -1:
            if signal_type == SignalType.BUY or signal_type == SignalType.NEUTRAL:
                log_return = math.log(self.entry_price / price) if price > 0 else 0
                
                self.trades.append({
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_time': timestamp,
                    'exit_price': price,
                    'log_return': log_return,
                    'position': 'SHORT'
                })
                
                self.current_position = 0

    def calculate_sharpe(self, risk_free_rate=0):
        """
        Calculate the annualized Sharpe ratio for minute data.

        Args:
            risk_free_rate: Annual risk-free rate (not adjusted for minutes)

        Returns:
            float: Annualized Sharpe ratio
        """
        if len(self.trades) < 2:
            return 0.0

        # Extract returns from trades
        returns = [trade['log_return'] for trade in self.trades]
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Calculate minutes per year (assuming 6.5 trading hours per day)
        minutes_per_day = 6.5 * 60  # 390 minutes in a typical trading day
        trading_days_per_year = 252  # Standard trading days per year
        minutes_per_year = minutes_per_day * trading_days_per_year

        # Convert annual risk-free rate to per-minute rate
        minute_risk_free_rate = risk_free_rate / minutes_per_year

        # Calculate Sharpe and annualize by multiplying by sqrt(minutes_per_year)
        sharpe = (avg_return - minute_risk_free_rate) / std_return
        annualized_sharpe = sharpe * np.sqrt(minutes_per_year)

        return annualized_sharpe

    def reset(self):
        """Reset the backtester state."""
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        self.strategy.reset()
