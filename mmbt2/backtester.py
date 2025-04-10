import math
from event import MarketEvent

class Backtester:
    def __init__(self, data_handler, strategy):
        self.data_handler = data_handler
        self.strategy = strategy
        self.events = []

    def run(self):
        bars = self.data_handler.load_data()
        for bar in bars:
            event = MarketEvent(bar)
            self.events.append(event)
            self.strategy.on_bar(event)
        return self.strategy.signals

    def calculate_returns(self):
        signals = self.strategy.signals
        position = None  # "long", "short", or None
        entry_price = None
        trades = []
        log_returns = []

        for signal in signals:
            signal_type = signal["signal"]
            timestamp = signal["timestamp"]
            price = signal["price"]

            if signal_type == 1 and position is None:
                position = "long"
                entry_price = price

            elif signal_type == -1 and position is None:
                position = "short"
                entry_price = price

            elif signal_type == 0 and position == "long":
                log_ret = math.log(price / entry_price)
                trades.append((timestamp, "long", entry_price, price, log_ret))
                log_returns.append(log_ret)
                position = None
                entry_price = None

            elif signal_type == 0 and position == "short":
                log_ret = math.log(entry_price / price)
                trades.append((timestamp, "short", entry_price, price, log_ret))
                log_returns.append(log_ret)
                position = None
                entry_price = None

        total_log_return = sum(log_returns)
        total_percent_return = (math.exp(total_log_return) - 1) * 100
        avg_log_return = total_log_return / len(log_returns) if log_returns else 0

        return {
            "trades": trades,
            "total_log_return": total_log_return,
            "total_percent_return": total_percent_return,
            "average_log_return": avg_log_return,
            "num_trades": len(trades)
        }

