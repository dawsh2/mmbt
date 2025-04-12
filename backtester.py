 # backtester.py
import numpy as np
import pandas as pd
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
        self.signals_df = pd.DataFrame(columns=['timestamp', 'signal', 'price'])
        self.trades_df = pd.DataFrame(columns=['entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 'position'])
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None

    #@profile
    def run(self, use_test_data=False):
        # === Debug: Track number of times run() has been called on this instance
        if hasattr(self, "run_count"):
            self.run_count += 1
        else:
            self.run_count = 1

        self.strategy.reset()
        all_signals = []
        data = []
        all_signals_data = []
        all_trades_data = []

        if not use_test_data:
            self.data_handler.reset_train()
            get_next_bar = self.data_handler.get_next_train_bar
        else:
            self.data_handler.reset_test()
            get_next_bar = self.data_handler.get_next_test_bar

        while True:
            event_data = get_next_bar()
            if event_data is None:
                break
            data.append(event_data)
            event = BarEvent(event_data)
            signal = self.strategy.on_bar(event)
            if signal is not None:
                all_signals.append(signal)
                all_signals_data.append({'timestamp': signal.timestamp, 'signal': signal.signal_type.value, 'price': signal.price})

            self._process_signal_for_trades(signal, event_data.get('timestamp'), event_data.get('Close'), all_trades_data)

        self.signals_df = pd.DataFrame(all_signals_data)
        self.trades_df = pd.DataFrame(all_trades_data)
        results_df = pd.DataFrame(data)
        results_df = pd.merge(results_df, self.signals_df, on='timestamp', how='left')
        results_df['trades'] = [self.trades_df] * len(results_df) # Store the trades DataFrame in each row for simplicity

        return results_df

    #@profile
    def _process_signal_for_trades(self, signal, timestamp, price, trades_list):
        if signal is None:
            return

        signal_type = signal.signal_type

        if self.current_position == 0:
            if signal_type == SignalType.BUY:
                self.current_position = 1
                self.entry_price = price
                self.entry_time = timestamp
                #print(f"{timestamp} - BUY at {price:.2f}")
            elif signal_type == SignalType.SELL:
                self.current_position = -1
                self.entry_price = price
                self.entry_time = timestamp
                #print(f"{timestamp} - SELL at {price:.2f}")
        elif self.current_position == 1:
            if signal_type == SignalType.SELL or signal_type == SignalType.NEUTRAL:
                exit_price = price
                exit_time = timestamp
                profit = (exit_price - self.entry_price)
                trades_list.append({'entry_time': self.entry_time, 'entry_price': self.entry_price,
                                    'exit_time': exit_time, 'exit_price': exit_price,
                                    'profit': profit, 'position': 'LONG'})
                self.current_position = 0
                #print(f"{exit_time} - Exit LONG at {exit_price:.2f}, Profit: {profit:.2f}")
        elif self.current_position == -1:
            if signal_type == SignalType.BUY or signal_type == SignalType.NEUTRAL:
                exit_price = price
                exit_time = timestamp
                profit = (self.entry_price - exit_price)
                trades_list.append({'entry_time': self.entry_time, 'entry_price': self.entry_price,
                                    'exit_time': exit_time, 'exit_price': exit_price,
                                    'profit': profit, 'position': 'SHORT'})
                self.current_position = 0
                #print(f"{exit_time} - Exit SHORT at {exit_price:.2f}, Profit: {profit:.2f}")


    #@profile
    def calculate_returns(self, signals):
        # This method is now largely handled within _process_signal_for_trades
        return pd.DataFrame(self.trades_df)

    #@profile
    def calculate_sharpe(self, risk_free_rate=0):
        """
        Calculates the annualized Sharpe ratio for minute data, considering only market hours
        (assuming 6.5 hours per day).

        Args:
            self: The instance of the Backtester class.
            risk_free_rate (float): The annualized risk-free rate (default is 0).

        Returns:
            float: The annualized Sharpe ratio. Returns 0.0 if there are fewer than 2 trades.
        """
        if len(self.trades_df) < 2:
            return 0.0

        returns = self.trades_df['profit']
        excess_returns = returns - risk_free_rate

        # Calculate the number of trading minutes in a year
        minutes_per_day = 6.5 * 60
        trading_days_per_year = 252  # Standard for US markets
        minutes_per_year = minutes_per_day * trading_days_per_year

        # Annualize the Sharpe ratio by the square root of the number of periods in a year
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(minutes_per_year)

        return sharpe_ratio

    #@profile
    def reset(self):
        self.signals_df = pd.DataFrame(columns=['timestamp', 'signal', 'price'])
        self.trades_df = pd.DataFrame(columns=['entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 'position'])
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        self.strategy.reset()
