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
        # Reset state
        self.strategy.reset()
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None

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
                self._process_signal_for_trades(signal, bar_data['timestamp'], bar_data['Close'])

        # Calculate metrics using tuple format - assuming log_return is at index 5
        if self.trades:
            # Handle tuples instead of dictionaries
            total_log_return = sum(trade[5] for trade in self.trades)  # Index 5 for log_return
            total_return = (math.exp(total_log_return) - 1) * 100  # Convert to percentage
            avg_log_return = total_log_return / len(self.trades)
            win_count = sum(1 for trade in self.trades if trade[5] > 0)  # Index 5 for log_return
            win_rate = win_count / len(self.trades) if self.trades else 0
        else:
            total_log_return = 0
            total_return = 0
            avg_log_return = 0
            win_rate = 0

        # Return results
        # print(f"Debug - Actual trades generated: {len(self.trades)}, Trades list: {self.trades}")
        return {
            'total_log_return': total_log_return,
            'total_percent_return': total_return,
            'average_log_return': avg_log_return,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'trades': self.trades
        }




    def _process_signal_for_trades(self, signal, timestamp, price):
        # Extract signal value
        if hasattr(signal, 'signal_type'):
            signal_value = signal.signal_type.value  # Signal object
        elif isinstance(signal, dict) and 'signal' in signal:
            signal_value = signal['signal']  # Dictionary format
        elif isinstance(signal, (int, float)):
            signal_value = signal  # Numeric signal
        else:
            # Default to neutral if signal format is unrecognized
            signal_value = 0

        # print(f"Processing signal: {signal_value}, Current position: {self.current_position}, Entry price: {self.entry_price}")

        # Process signal based on current position
        if self.current_position == 0:  # Not in a position
            if signal_value == 1:  # Buy signal
                # Enter long position
                self.current_position = 1
                self.entry_price = price
                self.entry_time = timestamp
                # print(f"Entering long position at price: {price}")
            elif signal_value == -1:  # Sell signal
                # Enter short position
                self.current_position = -1
                self.entry_price = price
                self.entry_time = timestamp
                # print(f"Entering short position at price: {price}")

        elif self.current_position == 1:  # In long position
            if signal_value == -1 or signal_value == 0:  # Sell or neutral signal
                # Exit long position
                log_return = math.log(price / self.entry_price) if self.entry_price > 0 else 0
                self.trades.append((
                    self.entry_time,
                    'long',
                    self.entry_price,
                    timestamp,
                    price,
                    log_return
                ))
                # print(f"Exiting long position, adding trade: {self.entry_time} to {timestamp}, Return: {log_return:.4f}")

                # If sell signal, enter short position
                if signal_value == -1:
                    self.current_position = -1
                    self.entry_price = price
                    self.entry_time = timestamp
                   # print(f"Entering short position at price: {price}")
                else:
                    self.current_position = 0
                    self.entry_price = None
                    self.entry_time = None

        elif self.current_position == -1:  # In short position
            if signal_value == 1 or signal_value == 0:  # Buy or neutral signal
                # Exit short position
                log_return = math.log(self.entry_price / price) if price > 0 else 0
                self.trades.append((
                    self.entry_time,
                    'short',
                    self.entry_price,
                    timestamp,
                    price,
                    log_return
                ))
                # print(f"Exiting short position, adding trade: {self.entry_time} to {timestamp}, Return: {log_return:.4f}")

                # If buy signal, enter long position
                if signal_value == 1:
                    self.current_position = 1
                    self.entry_price = price
                    self.entry_time = timestamp
                    # print(f"Entering long position at price: {price}")
                else:
                    self.current_position = 0
                    self.entry_price = None
                    self.entry_time = None


    def calculate_sharpe(self, risk_free_rate=0):
        """
        Calculate Sharpe ratio using minute-by-minute returns.

        This method creates a series of returns for every minute in the backtest
        period, properly accounting for periods with no trades.

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            float: Annualized Sharpe ratio
        """
        if not self.trades:
            return 0.0

        # First, determine the start and end times of your backtest
        start_time = self.trades[0][0]  # First trade entry time
        end_time = self.trades[-1][3]   # Last trade exit time

        # Create a dictionary to store minute-by-minute equity values
        # You'll need to adjust this based on how your timestamps are formatted
        from datetime import datetime, timedelta

        # Convert timestamps to datetime objects if they're strings
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        # Generate a list of all minute timestamps in the backtest period
        current_time = start_time
        all_minutes = []
        while current_time <= end_time:
            # Only include timestamps during trading hours (e.g., 9:30 AM to 4:00 PM)
            hour = current_time.hour
            minute = current_time.minute
            if (9 <= hour < 16) or (hour == 16 and minute == 0):
                # Skip weekends
                if current_time.weekday() < 5:  # Monday=0, Sunday=6
                    all_minutes.append(current_time)
            current_time += timedelta(minutes=1)

        # Initialize equity curve with starting value (e.g., $10,000)
        initial_equity = 10000
        equity_curve = {minute: initial_equity for minute in all_minutes}

        # Update equity curve at trade entry/exit points
        current_equity = initial_equity
        for trade in self.trades:
            entry_time = trade[0]
            exit_time = trade[3]
            log_return = trade[5]

            # Convert timestamps if needed
            if isinstance(entry_time, str):
                entry_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
            if isinstance(exit_time, str):
                exit_time = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")

            # Calculate new equity after this trade
            current_equity *= np.exp(log_return)

            # Update equity value at exit time and all subsequent minutes
            for minute in all_minutes:
                if minute >= exit_time:
                    equity_curve[minute] = current_equity

        # Calculate minute-by-minute returns
        minute_returns = []
        prev_equity = None

        for minute, equity in sorted(equity_curve.items()):
            if prev_equity is not None:
                minute_return = np.log(equity / prev_equity)
                minute_returns.append(minute_return)
            prev_equity = equity

        # Calculate Sharpe ratio using minute returns
        if len(minute_returns) < 2:
            return 0.0

        avg_minute_return = np.mean(minute_returns)
        std_minute_return = np.std(minute_returns)

        if std_minute_return == 0:
            return 0.0

        # Annualization factor for minutes (assuming 6.5 trading hours per day)
        minutes_per_day = 6.5 * 60  # 390 minutes in a typical trading day
        trading_days_per_year = 252  # Standard trading days per year
        minutes_per_year = minutes_per_day * trading_days_per_year

        # Convert annual risk-free rate to per-minute rate
        minute_risk_free_rate = risk_free_rate / minutes_per_year

        # Calculate Sharpe and annualize
        sharpe = (avg_minute_return - minute_risk_free_rate) / std_minute_return
        annualized_sharpe = sharpe * np.sqrt(minutes_per_year)

        return annualized_sharpe
                    


    # def calculate_sharpe(self, risk_free_rate=0):
    #     """
    #     Calculate the annualized Sharpe ratio for minute data.

    #     Args:
    #         risk_free_rate: Annual risk-free rate (not adjusted for minutes)

    #     Returns:
    #         float: Annualized Sharpe ratio
    #     """
    #     if len(self.trades) < 2:
    #         return 0.0

    #     # Extract returns from trades (using index 5 for log_return)
    #     returns = [trade[5] for trade in self.trades]
    #     avg_return = np.mean(returns)
    #     std_return = np.std(returns)

    #     if std_return == 0:
    #         return 0.0

    #     # Calculate minutes per year (assuming 6.5 trading hours per day)
    #     minutes_per_day = 6.5 * 60  # 390 minutes in a typical trading day
    #     trading_days_per_year = 252  # Standard trading days per year
    #     minutes_per_year = minutes_per_day * trading_days_per_year

    #     # Convert annual risk-free rate to per-minute rate
    #     minute_risk_free_rate = risk_free_rate / minutes_per_year

    #     # Calculate Sharpe and annualize by multiplying by sqrt(minutes_per_year)
    #     sharpe = (avg_return - minute_risk_free_rate) / std_return
    #     annualized_sharpe = sharpe * np.sqrt(minutes_per_year)

    #     return annualized_sharpe


    def reset(self):
        """Reset the backtester state."""
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
