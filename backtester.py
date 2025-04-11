import numpy as np
import pandas as pd
import math

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
        # === Debug: Track number of times run() has been called on this instance
        if hasattr(self, "run_count"):
            self.run_count += 1
        else:
            self.run_count = 1
        print(f"\n📊 Backtest Run #{self.run_count} — {'Test' if use_test_data else 'Train'} Data")

        self.strategy.reset()
        all_signals = []
        if not use_test_data:
            self.data_handler.reset_train()
        else:
            self.data_handler.reset_test()
        while True:
            event_data = self.data_handler.get_next_train_bar() if not use_test_data else self.data_handler.get_next_test_bar()
            if event_data is None:
                break
            event = BarEvent(event_data)
            signal = self.strategy.on_bar(event)
            if signal is not None:
                all_signals.append(signal.copy())
        results = self.calculate_returns(all_signals)
        results['signals'] = all_signals
        return results


    def calculate_returns(self, signals):
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_signal = None  # NEW

        for i in range(len(signals)):
            current_signal = signals[i]['signal']
            timestamp = signals[i]['timestamp']
            price = signals[i]['price']  # Close price of current bar

            # Entry logic
            if current_signal == 1 and self.current_position == 0:
                self.current_position = 1
                self.entry_price = price
                self.entry_time = timestamp
                self.entry_signal = current_signal  # Store the entry signal

            elif current_signal == -1 and self.current_position == 0:
                self.current_position = -1
                self.entry_price = price
                self.entry_time = timestamp
                self.entry_signal = current_signal  # Store the entry signal

            # Exit logic
            elif self.current_position == 1 and (current_signal == 0 or current_signal == -1):
                if self.entry_price is not None:
                    log_return = math.log(price / self.entry_price)
                    self.trades.append((
                        self.entry_time, "BUY", self.entry_price,
                        timestamp, price, log_return,
                        self.entry_signal, current_signal  # entry and exit signals
                    ))
                    self.current_position = 0
                    self.entry_price = None
                    self.entry_time = None
                    self.entry_signal = None

            elif self.current_position == -1 and (current_signal == 0 or current_signal == 1):
                if self.entry_price is not None:
                    log_return = math.log(self.entry_price / price)
                    self.trades.append((
                        self.entry_time, "SELL", self.entry_price,
                        timestamp, price, log_return,
                        self.entry_signal, current_signal  # entry and exit signals
                    ))
                    self.current_position = 0
                    self.entry_price = None
                    self.entry_time = None
                    self.entry_signal = None

        # Print first few trades for debugging
        # print("\n=== Sample Trades ===")
        # for trade in self.trades[:10]:
        #     print(
        #         f"{trade[0]} → {trade[3]} | {trade[1]} at {trade[2]:.2f} → {trade[4]:.2f} "
        #         f"| Signal: {trade[6]} → {trade[7]} | Log Return: {trade[5]:.6f}"
        #     )

        return {
            "trades": self.trades,
            "total_log_return": sum([t[5] for t in self.trades]),
            "average_log_return": np.mean([t[5] for t in self.trades]) if self.trades else 0,
            "num_trades": len(self.trades),
            "total_percent_return": (math.exp(sum([t[5] for t in self.trades])) - 1) * 100 if self.trades else 0
        }
 

    # WARNING: The following is only accurate for minute data
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
        returns = [trade[5] for trade in self.trades]  # Extract log returns

        if len(returns) < 2:
            return 0.0

        returns_series = pd.Series(returns)
        excess_returns = returns_series - risk_free_rate

        # Calculate the number of trading minutes in a year
        minutes_per_day = 6.5 * 60
        trading_days_per_year = 252  # Standard for US markets
        minutes_per_year = minutes_per_day * trading_days_per_year

        # Annualize the Sharpe ratio by the square root of the number of periods in a year
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(minutes_per_year)

        return sharpe_ratio
    

    def reset(self):
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        self.strategy.reset()
