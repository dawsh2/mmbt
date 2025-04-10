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

        for i in range(len(signals)):
            current_signal = signals[i]['signal']
            timestamp = signals[i]['timestamp']
            price = signals[i]['price']  # Close price of current bar

            # Make trading decisions based on the current bar's signal
            if current_signal == 1 and self.current_position == 0:
                print(f"{timestamp} - Execute BUY at {price:.2f}")
                self.current_position = 1
                self.entry_price = price
                self.entry_time = timestamp
            elif current_signal == -1 and self.current_position == 0:
                print(f"{timestamp} - Execute SELL at {price:.2f}")
                self.current_position = -1
                self.entry_price = price
                self.entry_time = timestamp
            elif self.current_position == 1 and \
                 (current_signal == 0 or current_signal == -1):
                if self.entry_price is not None:
                    log_return = math.log(price / self.entry_price)
                    self.trades.append((self.entry_time, "BUY", self.entry_price, timestamp, price, log_return))
                    print(f"{timestamp} - Exit BUY at {price:.2f}, Log Return: {log_return:.4f}")
                    self.current_position = 0
                    self.entry_price = None
                    self.entry_time = None
            elif self.current_position == -1 and \
                 (current_signal == 0 or current_signal == 1):
                if self.entry_price is not None:
                    log_return = math.log(self.entry_price / price)
                    self.trades.append((self.entry_time, "SELL", self.entry_price, timestamp, price, log_return))
                    print(f"{timestamp} - Exit SELL at {price:.2f}, Log Return: {log_return:.4f}")
                    self.current_position = 0
                    self.entry_price = None
                    self.entry_time = None

        return {
            "trades": self.trades,
            "total_log_return": sum([t[5] for t in self.trades]),
            "average_log_return": np.mean([t[5] for t in self.trades]) if self.trades else 0,
            "num_trades": len(self.trades),
            "total_percent_return": (math.exp(sum([t[5] for t in self.trades])) - 1) * 100 if self.trades else 0
        }

 

    # def calculate_returns(self, signals):
    #     self.trades = []
    #     self.current_position = 0
    #     self.entry_price = None
    #     self.entry_time = None
    #     total_log_return = 0
    #     trade_returns = []

    #     for i in range(1, len(signals)):
    #         prev_signal = signals[i-1]['signal']
    #         current_signal = signals[i]['signal']
    #         price = signals[i]['price']
    #         timestamp = signals[i]['timestamp']

    #         if current_signal == 1 and self.current_position == 0:
    #             self.current_position = 1
    #             self.entry_price = price
    #             self.entry_time = timestamp
    #         elif current_signal == -1 and self.current_position == 0:
    #             self.current_position = -1
    #             self.entry_price = price
    #             self.entry_time = timestamp
    #         elif (current_signal == 0 and self.current_position == 1) or \
    #              (current_signal == -1 and self.current_position == 1):
    #             if self.entry_price is not None:
    #                 log_return = math.log(price / self.entry_price)
    #                 self.trades.append((self.entry_time, "BUY", self.entry_price, timestamp, price, log_return))
    #                 total_log_return += log_return
    #                 trade_returns.append(log_return)
    #                 self.current_position = 0
    #                 self.entry_price = None
    #                 self.entry_time = None
    #         elif (current_signal == 0 and self.current_position == -1) or \
    #              (current_signal == 1 and self.current_position == -1):
    #             if self.entry_price is not None:
    #                 log_return = math.log(self.entry_price / price)
    #                 self.trades.append((self.entry_time, "SELL", self.entry_price, timestamp, price, log_return))
    #                 total_log_return += log_return
    #                 trade_returns.append(log_return)
    #                 self.current_position = 0
    #                 self.entry_price = None
    #                 self.entry_time = None

    #     num_trades = len(self.trades)
    #     average_log_return = np.mean(trade_returns) if trade_returns else 0
    #     total_percent_return = (math.exp(total_log_return) - 1) * 100 if total_log_return else 0

    #     return {
    #         "trades": self.trades,
    #         "total_log_return": total_log_return,
    #         "average_log_return": average_log_return,
    #         "num_trades": num_trades,
    #         "total_percent_return": total_percent_return
    #     }
    
 
    def calculate_sharpe(self, risk_free_rate=0):
        returns = []
        for trade in self.trades:
            returns.append(trade[5]) # Log returns

        if len(returns) < 2:
            return 0.0

        returns_series = pd.Series(returns)
        excess_returns = returns_series - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) # Assuming daily data, annualized
        return sharpe_ratio

    def reset(self):
        self.signals = []
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.entry_time = None
        self.strategy.reset()
