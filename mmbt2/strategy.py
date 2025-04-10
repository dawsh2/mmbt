class Strategy:
    def __init__(self, short_window=5, long_window=20):
        self.signals = []
        self.prices = []
        self.short_window = short_window
        self.long_window = long_window
        self.prev_short_sma = None
        self.prev_long_sma = None
        self.current_position = 0  # 0 = flat, 1 = long, -1 = short

    def compute_sma(self, data, window):
        if len(data) < window:
            return None
        return sum(data[-window:]) / window


    def on_bar(self, event):
        bar = event.bar
        close_price = bar["Close"]
        self.prices.append(close_price)

        short_sma = self.compute_sma(self.prices, self.short_window)
        long_sma = self.compute_sma(self.prices, self.long_window)

        if short_sma is None or long_sma is None:
            return  # Not enough data yet

        print(f"{bar['timestamp']} | Close: {close_price:.2f} | Short SMA: {short_sma:.2f} | Long SMA: {long_sma:.2f}")

        signal = None

        # ✅ Only evaluate crossover logic after we have previous SMAs
        if self.prev_short_sma is not None and self.prev_long_sma is not None:
            if self.current_position == 0 and self.prev_short_sma <= self.prev_long_sma and short_sma > long_sma:
                signal = 1
                self.current_position = 1

            elif self.current_position == 0 and self.prev_short_sma >= self.prev_long_sma and short_sma < long_sma:
                signal = -1
                self.current_position = -1

            elif self.current_position == 1 and short_sma < long_sma:
                signal = 0
                self.current_position = 0

            elif self.current_position == -1 and short_sma > long_sma:
                signal = 0
                self.current_position = 0

        if signal is not None:
            self.signals.append({
                "timestamp": bar["timestamp"],
                "signal": signal,
                "price": close_price,
                "short_sma": short_sma,
                "long_sma": long_sma
            })

        self.prev_short_sma = short_sma
        self.prev_long_sma = long_sma


  
