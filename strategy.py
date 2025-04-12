from collections import deque
import pandas as pd
import numpy as np
from signals import Signal, SignalType



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

from signals import Signal, SignalType

class Rule0:
    """
    Rule0 using Signal objects directly.
    """
    def __init__(self, params):
        self.fast_window = int(params['fast_window'])
        self.slow_window = int(params['slow_window'])
        self.prices = []
        self.fast_sma_history = []
        self.slow_sma_history = []
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule0"

    def on_bar(self, bar):
        close = bar["Close"]
        self.prices.append(close)

        if len(self.prices) >= max(self.fast_window, self.slow_window):
            fast_sma = sum(self.prices[-self.fast_window:]) / self.fast_window
            slow_sma = sum(self.prices[-self.slow_window:]) / self.slow_window
            self.fast_sma_history.append(fast_sma)
            self.slow_sma_history.append(slow_sma)

            # Key change: Check the current relationship between SMAs
            # rather than only detecting crossovers
            if len(self.fast_sma_history) >= 1 and len(self.slow_sma_history) >= 1:
                current_fast_sma = self.fast_sma_history[-1]
                current_slow_sma = self.slow_sma_history[-1]

                # Signal based on current relationship, not just crossover
                if current_fast_sma > current_slow_sma:
                    self.current_signal_type = SignalType.BUY  # Bullish state
                elif current_fast_sma < current_slow_sma:
                    self.current_signal_type = SignalType.SELL  # Bearish state
                else:
                    self.current_signal_type = SignalType.NEUTRAL  # Equal (rare)
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL
        
        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.prices = []
        self.fast_sma_history = []
        self.slow_sma_history = []
        self.current_signal_type = SignalType.NEUTRAL



from signals import Signal, SignalType

class Rule1:
    """
    Rule1 using Signal objects directly.
    """
    def __init__(self, params):
        self.ma1_period = params.get('ma1', 15)
        self.ma2_period = params.get('ma2', 45)
        self.history = deque(maxlen=max(self.ma1_period, self.ma2_period))
        self.sum1 = 0  # Initialize sum1
        self.sum2 = 0  # Initialize sum2
        self.ma1_history = deque()  # Corrected attribute name
        self.ma2_history = deque()  # Corrected attribute name
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule1"

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        # Efficient SMA 1 calculation
        if len(self.history) > self.ma1_period:
            self.sum1 -= self.history[-(self.ma1_period + 1)]
        self.sum1 += close
        if len(self.history) >= self.ma1_period:
            sma1 = self.sum1 / self.ma1_period
            self.ma1_history.append(sma1)  # Corrected attribute name

        # Efficient SMA 2 calculation
        if len(self.history) > self.ma2_period:
            self.sum2 -= self.history[-(self.ma2_period + 1)]
        self.sum2 += close
        if len(self.history) >= self.ma2_period:
            sma2 = self.sum2 / self.ma2_period
            self.ma2_history.append(sma2)  # Corrected attribute name

        if len(self.ma1_history) >= 1 and len(self.ma2_history) >= 1:
            # Get the most recent values
            curr_s1 = self.ma1_history[-1]  # Using correct attribute
            curr_s2 = self.ma2_history[-1]  # Using correct attribute

            # Signal based on current MA relationship, not just crossover
            if curr_s1 > curr_s2:
                self.current_signal_type = SignalType.BUY  # Bullish state
            elif curr_s1 < curr_s2:
                self.current_signal_type = SignalType.SELL  # Bearish state
            else:
                self.current_signal_type = SignalType.NEUTRAL  # Equal (rare)
        else:
            self.current_signal_type = SignalType.NEUTRAL

        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={}
        )

    def reset(self):
        self.history.clear()
        self.sum1 = 0
        self.sum2 = 0
        self.ma1_history.clear()  # Corrected attribute name
        self.ma2_history.clear()  # Corrected attribute name
        self.current_signal_type = SignalType.NEUTRAL



from signals import Signal, SignalType

class Rule2:
    """
    Rule2 using Signal objects directly.
    """
    def __init__(self, params):
        self.ema1_period = params.get('ema1_period', 12)  # Default value if not provided
        self.ma2_period = params.get('ma2_period', 26)  # Default value if not provided
        self.history = deque(maxlen=max(self.ema1_period if self.ema1_period > 0 else 1, self.ma2_period if self.ma2_period > 0 else 1))
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0
        self.ema1_value = np.nan
        self.ema1_history = deque()
        self.ma2_sum = 0
        self.ma2_history = deque()
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule2" # Ensure rule_id is set

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        # Incremental EMA calculation
        if self.alpha_ema1 > 0:
            if np.isnan(self.ema1_value):
                self.ema1_value = close  # Initialize with the first data point
            else:
                self.ema1_value = (close * self.alpha_ema1) + (self.ema1_value * (1 - self.alpha_ema1))
            self.ema1_history.append(self.ema1_value)
        else:
            self.ema1_history.append(np.nan)

        # Efficient SMA calculation
        if len(self.history) > self.ma2_period:
            self.ma2_sum -= self.history[-(self.ma2_period + 1)]
        self.ma2_sum += close
        if len(self.history) >= self.ma2_period:
            ma2_val = self.ma2_sum / self.ma2_period
            self.ma2_history.append(ma2_val)
        else:
            self.ma2_history.append(np.nan)

        if len(self.ema1_history) >= 1 and len(self.ma2_history) >= 1:
            curr_ema1 = self.ema1_history[-1]
            curr_ma2 = self.ma2_history[-1]
            current_price = close

            if not np.isnan(curr_ma2) and not np.isnan(curr_ema1):
                # Persistent signal based on current state, not just crossover
                # Also includes price confirmation
                if curr_ema1 > curr_ma2 and current_price > curr_ma2:
                    self.current_signal_type = SignalType.BUY
                elif curr_ema1 < curr_ma2 and current_price < curr_ma2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL
        
        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.history = []
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ma2_sum = 0
        self.ma2_history = []
        self.current_signal_type = SignalType.NEUTRAL
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0



from signals import Signal, SignalType


class Rule3:
    def __init__(self, params):
        self.ema1_period = params.get('ema1_period', 20)
        self.ema2_period = params.get('ema2_period', 40)
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0
        self.alpha_ema2 = 2 / (self.ema2_period + 1) if self.ema2_period > 0 else 0
        self.ema1_value = np.nan
        self.ema1_history = deque(maxlen=200)
        self.ema2_value = np.nan
        self.ema2_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule3"

    def on_bar(self, bar):
        close = bar['Close']
        self.ema1_history.append(close)
        self.ema2_history.append(close)

        if len(self.ema1_history) >= self.ema1_period:
            if np.isnan(self.ema1_value):
                self.ema1_value = np.mean(list(self.ema1_history)[-self.ema1_period:])
            else:
                self.ema1_value = (close * self.alpha_ema1) + (self.ema1_value * (1 - self.alpha_ema1))

        if len(self.ema2_history) >= self.ema2_period:
            if np.isnan(self.ema2_value):
                self.ema2_value = np.mean(list(self.ema2_history)[-self.ema2_period:])
            else:
                self.ema2_value = (close * self.alpha_ema2) + (self.ema2_value * (1 - self.alpha_ema2))

        if not np.isnan(self.ema1_value) and not np.isnan(self.ema2_value):
            if self.ema1_value > self.ema2_value:
                self.current_signal_type = SignalType.BUY
            elif self.ema1_value < self.ema2_value:
                self.current_signal_type = SignalType.SELL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        metadata = {
            "ema1": self.ema1_value,
            "ema2": self.ema2_value
        }

        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.ema1_value = np.nan
        self.ema1_history = deque(maxlen=200)
        self.ema2_value = np.nan
        self.ema2_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL

class Rule4:
    def __init__(self, params):
        self.ma2_period = params.get('ma2_period', 50)
        self.dema1_period = params.get('dema1_period', 20)
        self.alpha1 = None # Basic initialization
        self.history = deque(maxlen=200) # Assuming a maxlen
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule4"
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)
        # Incremental EMA 1
        if (self.alpha1 is not None and self.alpha1) > 0:
            if np.isnan(self.ema1_value):
                self.ema1_value = close
            else:
                self.ema1_value = (close * self.alpha1) + (self.ema1_value * (1 - self.alpha1))

            # Incremental EMA 2 (of EMA 1)
            if np.isnan(self.ema2_value):
                self.ema2_value = self.ema1_value
            else:
                self.ema2_value = (self.ema1_value * self.alpha1) + (self.ema2_value * (1 - self.alpha1))

            # Calculate DEMA 1
            self.dema1_value = (2 * self.ema1_value) - self.ema2_value
        else:
            self.dema1_value = np.nan

        # Optimized SMA calculation
        self.ma2_sum += close
        self.ma2_count += 1
        if self.ma2_count > self.ma2_period:
            self.ma2_sum -= self.history[-(self.ma2_period + 1)]
        ma2_val = self.ma2_sum / self.ma2_period if self.ma2_count >= self.ma2_period else np.nan

        # Fixed section of Rule4.on_bar()
        if len(self.history) >= max((self.dema1_period * 2 - 1 if self.dema1_period > 0 else 0), self.ma2_period):
            curr_dema1 = self.dema1_value
            curr_ma2 = ma2_val

            if not np.isnan(curr_dema1) and not np.isnan(curr_ma2):
                # Persistent signal based on current state
                if curr_dema1 > curr_ma2:
                    self.current_signal_type = SignalType.BUY
                elif curr_dema1 < curr_ma2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL


        # Create metadata with any additional information
            metadata = {}

            # Return a Signal object
            return Signal(
                timestamp=bar["timestamp"],
                type=self.current_signal_type,
                price=bar["Close"],
                rule_id=self.rule_id,
                confidence=1.0,
                metadata=metadata
            )

    def reset(self):
        self.history = deque(maxlen=200)
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.current_signal_type = SignalType.NEUTRAL
        self.alpha1 = 2 / (self.dema1_period + 1) if self.dema1_period > 0 else 0
        self.alpha1 = None # Basic initialization

class Rule5:
    def __init__(self, params):
        self.dema2_period = params.get('dema2_period', 30)
        self.dema1_period = params.get('dema1_period', 15)
        self.alpha1 = None # Basic initialization
        self.history = deque(maxlen=200) # Assuming a maxlen
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule5"
        self.ema1_value = np.nan
        self.ema1_history = deque()
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.dema1_history = deque()
        self.ema3_value = np.nan
        self.ema3_history = deque()
        self.ema4_value = np.nan
        self.dema2_value = np.nan
        self.dema2_history = deque()
        self.alpha2 = 2 / (self.dema2_period + 1) if self.dema2_period > 0 else 0


    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        # Incremental DEMA 1 calculation
        if (self.alpha1 is not None and self.alpha1) > 0:
            if np.isnan(self.ema1_value):
                self.ema1_value = close
            else:
                self.ema1_value = (close * self.alpha1) + (self.ema1_value * (1 - self.alpha1))
            self.ema1_history.append(self.ema1_value)

            if len(self.ema1_history) >= 1:
                prev_ema2 = self.ema2_value
                if np.isnan(self.ema2_value):
                    self.ema2_value = self.ema1_history[-1]
                else:
                    self.ema2_value = (self.ema1_history[-1] * self.alpha1) + (self.ema2_value * (1 - self.alpha1))

                self.dema1_value = (2 * self.ema1_value) - self.ema2_value
                self.dema1_history.append(self.dema1_value)
            else:
                self.dema1_history.append(np.nan)
        else:
            self.dema1_history.append(np.nan)

        # Incremental DEMA 2 calculation
        if (self.alpha2 is not None and self.alpha2) > 0:
            if np.isnan(self.ema3_value):
                self.ema3_value = close
            else:
                self.ema3_value = (close * self.alpha2) + (self.ema3_value * (1 - self.alpha2))
            self.ema3_history.append(self.ema3_value)

            if len(self.ema3_history) >= 1:
                prev_ema4 = self.ema4_value
                if np.isnan(self.ema4_value):
                    self.ema4_value = self.ema3_history[-1]
                else:
                    self.ema4_value = (self.ema3_history[-1] * self.alpha2) + (self.ema4_value * (1 - self.alpha2))

                self.dema2_value = (2 * self.ema3_value) - self.ema4_value
                self.dema2_history.append(self.dema2_value)
            else:
                self.dema2_history.append(np.nan)
        else:
            self.dema2_history.append(np.nan)

        if len(self.dema1_history) >= 1 and len(self.dema2_history) >= 1:
            curr_dema1 = self.dema1_history[-1]
            curr_dema2 = self.dema2_history[-1]

            if not np.isnan(curr_dema1) and not np.isnan(curr_dema2):
                # Persistent signal based on current DEMA relationship
                if curr_dema1 > curr_dema2:
                    self.current_signal_type = SignalType.BUY
                elif curr_dema1 < curr_dema2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

from collections import deque
import numpy as np
from signals import Signal, SignalType

class Rule6:
    def __init__(self, params):
        self.ma2_period = params.get('ma2_period', 30)
        self.tema1_period = params.get('tema1_period', 10)
        self.history = deque(maxlen=200) # Assuming a maxlen
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule6"
        self.alpha1 = None # Basic initialization
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.ema3_value = np.nan
        self.tema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.close_history = deque(maxlen=200) # Initialize close_history here

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)
        self.close_history.append(close) # Update close_history

        # Incremental EMA 1
        if (self.alpha1 is not None and self.alpha1) > 0:
            if np.isnan(self.ema1_value):
                self.ema1_value = close
            else:
                self.ema1_value = (close * self.alpha1) + (self.ema1_value * (1 - self.alpha1))

            # Incremental EMA 2 (of EMA 1)
            if np.isnan(self.ema2_value):
                self.ema2_value = self.ema1_value
            else:
                self.ema2_value = (self.ema1_value * self.alpha1) + (self.ema2_value * (1 - self.alpha1))

            # Incremental EMA 3 (of EMA 2)
            if np.isnan(self.ema3_value):
                self.ema3_value = self.ema2_value
            else:
                self.ema3_value = (self.ema2_value * self.alpha1) + (self.ema3_value * (1 - self.alpha1))

            # Calculate TEMA 1
            self.tema1_value = (3 * self.ema1_value) - (3 * self.ema2_value) + self.ema3_value
        else:
            self.tema1_value = np.nan

        # Optimized SMA calculation
        self.ma2_sum += close
        self.ma2_count += 1
        if self.ma2_count > self.ma2_period:
            self.ma2_sum -= self.history[-(self.ma2_period + 1)]
        ma2_val = self.ma2_sum / self.ma2_period if self.ma2_count >= self.ma2_period else np.nan

        if len(self.history) >= max((self.tema1_period * 3 - 2 if self.tema1_period > 0 else 0), self.ma2_period):
            curr_tema1 = self.tema1_value
            curr_ma2 = ma2_val

            if not np.isnan(curr_tema1) and not np.isnan(curr_ma2):
                # Persistent signal based on current TEMA vs MA relationship
                if curr_tema1 > curr_ma2:
                    self.current_signal_type = SignalType.BUY
                elif curr_tema1 < curr_ma2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL


        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.history = deque(maxlen=200)
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.ema3_value = np.nan
        self.tema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.current_signal_type = SignalType.NEUTRAL
        self.alpha1 = 2 / (self.tema1_period + 1) if self.tema1_period > 0 else 0
        self.close_history = deque(maxlen=200)

class Rule7:
    def __init__(self, params):
        self.stochma2_period = params.get('stochma2_period', 3)
        self.stoch1_period = params.get('stoch1_period', 14)
        self.high_history = deque(maxlen=200) # Assuming a maxlen
        self.low_history = deque(maxlen=200) # Assuming a maxlen
        self.stoch_history = deque(maxlen=200) # Assuming a maxlen
        self.stoch_ma_history = deque(maxlen=200) # Assuming a maxlen
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule7"
        self.s1_history = deque()
        self.s2_sum = 0
        self.s2_count = 0
        self.s2_value = np.nan
        self.close_history = deque(maxlen=200) # Initialize close_history here

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        if len(self.close_history) == self.stoch1_period:
            highest_high = max(self.high_history)
            lowest_low = min(self.low_history)
            if highest_high != lowest_low:
                s1 = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                s1 = 50
            self.s1_history.append(s1)
        elif len(self.close_history) > self.stoch1_period:
            highest_high = max(self.high_history)
            lowest_low = min(self.low_history)
            if highest_high != lowest_low:
                s1 = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                s1 = 50
            self.s1_history.append(s1)
            self.high_history.popleft()
            self.low_history.popleft()
            self.close_history.popleft()
        else:
            self.s1_history.append(np.nan)

        # Optimized SMA of Stochastic %K
        if len(self.s1_history) >= 1 and not np.isnan(self.s1_history[-1]):
            self.s2_sum += self.s1_history[-1]
            self.s2_count += 1
            if self.s2_count > self.stochma2_period:
                self.s2_sum -= self.s1_history[-(self.stochma2_period + 1)]
            if self.s2_count >= self.stochma2_period:
                self.s2_value = self.s2_sum / self.stochma2_period
            else:
                self.s2_value = np.nan
        else:
            self.s2_value = np.nan

        if len(self.s1_history) >= 1 and not np.isnan(self.s2_value) and len(self.s1_history) >= self.stochma2_period:
            curr_s1 = self.s1_history[-1]   # %K
            curr_s2 = self.s2_value       # %D

            if not np.isnan(curr_s1) and not np.isnan(curr_s2):
                # Persistent signal based on current %K vs %D relationship
                if curr_s1 > curr_s2:
                    self.current_signal_type = SignalType.BUY
                elif curr_s1 < curr_s2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.high_history = deque(maxlen=self.stoch1_period)
        self.low_history = deque(maxlen=self.stoch1_period)
        self.close_history = deque(maxlen=self.stoch1_period)
        self.s1_history = deque()
        self.s2_sum = 0
        self.s2_count = 0
        self.s2_value = np.nan
        self.current_signal_type = SignalType.NEUTRAL

class Rule8:
    def __init__(self, params):
        self.vortex1_period = params.get('vortex1_period', 14)
        self.vortex2_period = params.get('vortex2_period', 14)
        self.high_history = deque(maxlen=200)
        self.low_history = deque(maxlen=200)
        self.close_history = deque(maxlen=200)
        self.tr_sum = 0
        self.pm_sum = 0
        self.nm_sum = 0
        self.current_tr = 0
        self.current_pm = 0
        self.current_nm = 0
        self.s1_history = deque() # For +VI history
        self.s2_history = deque() # For -VI history
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule8"


    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        if len((self.close_history is not None and self.close_history)) >= 2:
            prev_close = self.close_history[-2]
            self.current_tr = np.max([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
            self.current_pm = np.abs(high - prev_close) if (high - prev_close) > (prev_close - low) else 0
            self.current_nm = np.abs(low - prev_close) if (prev_close - low) > (high - prev_close) else 0
        else:
            self.current_tr = 0
            self.current_pm = 0
            self.current_nm = 0

        self.tr_sum += self.current_tr
        self.pm_sum += self.current_pm
        self.nm_sum += self.current_nm

        if len(self.close_history) > self.vortex1_period:
            if len(self.close_history) >= self.vortex1_period + 1:
                prev_high_v1 = self.high_history[-(self.vortex1_period + 1)]
                prev_low_v1 = self.low_history[-(self.vortex1_period + 1)]
                prev_close_v1_minus_1 = self.close_history[-(self.vortex1_period + 2)] if len(self.close_history) >= self.vortex1_period + 2 else self.close_history[-(self.vortex1_period + 1)]

                self.tr_sum -= np.max([prev_high_v1 - prev_low_v1,
                                        np.abs(prev_high_v1 - prev_close_v1_minus_1),
                                        np.abs(prev_low_v1 - prev_close_v1_minus_1)])
                self.pm_sum -= np.abs(prev_high_v1 - prev_close_v1_minus_1) if (prev_high_v1 - prev_close_v1_minus_1) > (prev_close_v1_minus_1 - prev_low_v1) else 0
                self.nm_sum -= np.abs(prev_low_v1 - prev_close_v1_minus_1) if (prev_close_v1_minus_1 - prev_low_v1) > (prev_high_v1 - prev_close_v1_minus_1) else 0

        if len(self.close_history) >= self.vortex1_period:
            vi_pos = (self.pm_sum is not None and self.pm_sum) / (self.tr_sum is not None and self.tr_sum) if (self.tr_sum is not None and self.tr_sum) != 0 else 0
            self.s1_history.append(vi_pos)
        else:
            self.s1_history.append(np.nan)

        # Recalculate current TR, PM, NM for the second period
        if len((self.close_history is not None and self.close_history)) >= 2:
            prev_close = self.close_history[-2]
            self.current_tr = np.max([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
            self.current_pm = np.abs(high - prev_close) if (high - prev_close) > (prev_close - low) else 0
            self.current_nm = np.abs(low - prev_close) if (prev_close - low) > (high - prev_close) else 0
        else:
            self.current_tr = 0
            self.current_pm = 0
            self.current_nm = 0

        self.tr_sum += self.current_tr
        self.pm_sum += self.current_pm
        self.nm_sum += self.current_nm

        if len(self.close_history) > self.vortex2_period:
            if len(self.close_history) >= self.vortex2_period + 1:
                prev_high_v2 = self.high_history[-(self.vortex2_period + 1)]
                prev_low_v2 = self.low_history[-(self.vortex2_period + 1)]
                prev_close_v2_minus_1 = self.close_history[-(self.vortex2_period + 2)] if len(self.close_history) >= self.vortex2_period + 2 else self.close_history[-(self.vortex2_period + 1)]

                self.tr_sum -= np.max([prev_high_v2 - prev_low_v2,
                                        np.abs(prev_high_v2 - prev_close_v2_minus_1),
                                        np.abs(prev_low_v2 - prev_close_v2_minus_1)])
                self.pm_sum -= np.abs(prev_high_v2 - prev_close_v2_minus_1) if (prev_high_v2 - prev_close_v2_minus_1) > (prev_close_v2_minus_1 - prev_low_v2) else 0
                self.nm_sum -= np.abs(prev_low_v2 - prev_close_v2_minus_1) if (prev_close_v2_minus_1 - prev_low_v2) > (prev_high_v2 - prev_close_v2_minus_1) else 0

        if len(self.close_history) >= self.vortex2_period:
            vi_neg = (self.nm_sum is not None and self.nm_sum) / (self.tr_sum is not None and self.tr_sum) if (self.tr_sum is not None and self.tr_sum) != 0 else 0
            self.s2_history.append(vi_neg)
        else:
            self.s2_history.append(np.nan)

        if len((self.s1_history is not None and self.s1_history)) >= 1 and len((self.s2_history is not None and self.s2_history)) >= 1:
            curr_s1 = self.s1_history[-1]  # +VI
            curr_s2 = self.s2_history[-1]  # -VI

            if not np.isnan(curr_s1) and not np.isnan(curr_s2):
                # Persistent signal based on current VI+ vs VI- relationship
                if curr_s1 > curr_s2:
                    self.current_signal_type = SignalType.BUY
                elif curr_s1 < curr_s2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=close,
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

        def reset(self):
            self.high_history.clear()
            self.low_history.clear()
            self.close_history.clear()
            self.tr_sum = 0
            self.pm_sum = 0
            self.nm_sum = 0
            self.current_tr = 0
            self.current_pm = 0
            self.current_nm = 0
            self.s1_history.clear()
            self.s2_history.clear()
            self.current_signal_type = SignalType.NEUTRAL



            
class Rule9:
    def __init__(self, params):
        self.ichimoku_tenkan_period = params.get('ichimoku_tenkan_period', 9)
        self.ichimoku_kijun_period = params.get('ichimoku_kijun_period', 26)
        self.ichimoku_senkou_b_period = params.get('ichimoku_senkou_b_period', 52)
        self.high_history = deque(maxlen=200)
        self.low_history = deque(maxlen=200)
        self.close_history = deque(maxlen=200)
        self.tenkan_sen = np.nan
        self.kijun_sen = np.nan
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule9"

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        if len(self.high_history) >= self.ichimoku_tenkan_period and len(self.low_history) >= self.ichimoku_tenkan_period:
            highest_high_tenkan = max(list(self.high_history)[-self.ichimoku_tenkan_period:])
            lowest_low_tenkan = min(list(self.low_history)[-self.ichimoku_tenkan_period:])
            self.tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2

        if len(self.high_history) >= self.ichimoku_kijun_period and len(self.low_history) >= self.ichimoku_kijun_period:
            highest_high_kijun = max(list(self.high_history)[-self.ichimoku_kijun_period:])
            lowest_low_kijun = min(list(self.low_history)[-self.ichimoku_kijun_period:])
            self.kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2

        if not np.isnan(self.tenkan_sen) and not np.isnan(self.kijun_sen):
            if self.tenkan_sen > self.kijun_sen and self.close > self.kijun_sen:
                self.current_signal_type = SignalType.BUY
            elif self.tenkan_sen < self.kijun_sen and self.close < self.kijun_sen:
                self.current_signal_type = SignalType.SELL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        metadata = {
            "tenkan_sen": self.tenkan_sen,
            "kijun_sen": self.kijun_sen
        }

        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.high_history = deque(maxlen=200)
        self.low_history = deque(maxlen=200)
        self.close_history = deque(maxlen=200)
        self.tenkan_sen = np.nan
        self.kijun_sen = np.nan
        self.current_signal_type = SignalType.NEUTRAL

class Rule10:
    def __init__(self, params):
        self.lookback_period = params.get('lookback_period', 14)
        self.gain_threshold = params.get('gain_threshold', 0.01)
        self.loss_threshold = params.get('loss_threshold', -0.01)
        self.close_history = deque(maxlen=200)
        self.gain_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule10"

    def on_bar(self, bar):
        close = bar['Close']
        self.close_history.append(close)

        if len(self.close_history) >= 2:
            previous_close = list(self.close_history)[-2]
            gain = (close - previous_close) / previous_close
            self.gain_history.append(gain)

            if len(self.gain_history) >= self.lookback_period:
                recent_gains = list(self.gain_history)[-self.lookback_period:]
                average_gain = np.mean([g for g in recent_gains if g > 0]) if any(g > 0 for g in recent_gains) else 0
                average_loss = np.mean([abs(g) for g in recent_gains if g < 0]) if any(g < 0 for g in recent_gains) else 0

                if average_gain > self.gain_threshold and average_loss < abs(self.loss_threshold):
                    self.current_signal_type = SignalType.BUY
                elif average_loss > abs(self.loss_threshold) and average_gain < self.gain_threshold:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        metadata = {
            "gain": self.gain_history[-1] if self.gain_history else np.nan,
            "average_gain": np.mean([g for g in list(self.gain_history)[-self.lookback_period:] if g > 0]) if len(self.gain_history) >= self.lookback_period and any(g > 0 for g in list(self.gain_history)[-self.lookback_period:]) else 0,
            "average_loss": np.mean([abs(g) for g in list(self.gain_history)[-self.lookback_period:] if g < 0]) if len(self.gain_history) >= self.lookback_period and any(g < 0 for g in list(self.gain_history)[-self.lookback_period:]) else 0
        }

        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.close_history = deque(maxlen=200)
        self.gain_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL
        



class Rule11:
    def __init__(self, params):
        self.cci1_period = params.get('cci1_period', 14)
        self.cci_threshold = params.get('c2_threshold', 100)
        self.history = {'high': deque(maxlen=self.cci1_period * 3),  # Increased maxlen
                        'low': deque(maxlen=self.cci1_period * 3),
                        'close': deque(maxlen=self.cci1_period)}
        self.sma_pp = np.nan
        self.mean_deviation = np.nan
        self.cci_value = np.nan
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule11"
        self.constant = 0.015
        self.last_zone = 'neutral'

    def _calculate_pivot_price(self, highs, lows, closes):
        return (highs + lows + closes) / 3

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.history['high'].append(high)
        self.history['low'].append(low)
        self.history['close'].append(close)

        if len(self.history['close']) >= self.cci1_period:
            highs = np.array(list(self.history['high'])[-self.cci1_period:])
            lows = np.array(list(self.history['low'])[-self.cci1_period:])
            closes = np.array(list(self.history['close'])[-self.cci1_period:])
            pivot_prices = self._calculate_pivot_price(highs, lows, closes)
            current_pp = pivot_prices[-1]

            self.sma_pp = np.mean(pivot_prices)
            mean_dev_sum = np.sum(np.abs(pivot_prices - self.sma_pp))
            self.mean_deviation = mean_dev_sum / self.cci1_period if self.cci1_period > 0 else 0

            if self.mean_deviation != 0:
                self.cci_value = (current_pp - self.sma_pp) / (self.constant * self.mean_deviation)
            else:
                self.cci_value = 0

            # Signal with hysteresis for persistence
            if self.cci_value > self.cci_threshold:
                self.current_signal_type = SignalType.SELL  # Overbought condition (sell)
                self.last_zone = 'overbought'
            elif self.cci_value < -self.cci_threshold:
                self.current_signal_type = SignalType.BUY    # Oversold condition (buy)
                self.last_zone = 'oversold'
            elif self.last_zone == 'overbought' and self.cci_value > (self.cci_threshold * 0.8):
                # Stay in overbought zone until CCI drops significantly
                self.current_signal_type = SignalType.SELL
            elif self.last_zone == 'oversold' and self.cci_value < (-self.cci_threshold * 0.8):
                # Stay in oversold zone until CCI rises significantly
                self.current_signal_type = SignalType.BUY
            else:
                self.current_signal_type = SignalType.NEUTRAL
                self.last_zone = 'neutral'
        else:
            self.cci_value = np.nan
            self.current_signal_type = SignalType.NEUTRAL

        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.cci_value = np.nan
        self.history = {'high': deque(maxlen=self.cci1_period * 3),
                        'low': deque(maxlen=self.cci1_period * 3),
                        'close': deque(maxlen=self.cci1_period)}
        self.sma_pp = np.nan
        self.mean_deviation = np.nan
        self.current_signal_type = SignalType.NEUTRAL
        self.constant = 0.015
        self.last_zone = 'neutral'

class Rule12:
    def __init__(self, params):
        self.lookback_period = params.get('lookback_period', 20)
        self.upper_threshold = params.get('upper_threshold', 0.02)
        self.lower_threshold = params.get('lower_threshold', -0.02)
        self.close_history = deque(maxlen=200)
        self.gain_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule12"

    def on_bar(self, bar):
        close = bar['Close']
        self.close_history.append(close)

        if len(self.close_history) >= 2:
            previous_close = list(self.close_history)[-2]
            gain = (close - previous_close) / previous_close
            self.gain_history.append(gain)

            if len(self.gain_history) >= self.lookback_period:
                recent_gains = list(self.gain_history)[-self.lookback_period:]
                positive_gain_count = sum(1 for g in recent_gains if g > self.upper_threshold)
                negative_gain_count = sum(1 for g in recent_gains if g < self.lower_threshold)

                if positive_gain_count >= 2:
                    self.current_signal_type = SignalType.BUY
                elif negative_gain_count >= 2:
                    self.current_signal_type = SignalType.SELL
                else:
                    self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        metadata = {
            "gain": self.gain_history[-1] if self.gain_history else np.nan,
            "positive_gain_count": sum(1 for g in list(self.gain_history)[-self.lookback_period:] if g > self.upper_threshold) if len(self.gain_history) >= self.lookback_period else 0,
            "negative_gain_count": sum(1 for g in list(self.gain_history)[-self.lookback_period:] if g < self.lower_threshold) if len(self.gain_history) >= self.lookback_period else 0
        }

        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.close_history = deque(maxlen=200)
        self.gain_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL

class Rule13:
    def __init__(self, params):
        self.lookback_period = params.get('lookback_period', 5)
        self.price_change_threshold = params.get('price_change_threshold', 0.005)
        self.close_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule13"

    def on_bar(self, bar):
        close = bar['Close']
        self.close_history.append(close)

        if len(self.close_history) >= self.lookback_period:
            first_price = list(self.close_history)[-self.lookback_period]
            last_price = close
            price_change = (last_price - first_price) / first_price

            if price_change > self.price_change_threshold:
                self.current_signal_type = SignalType.BUY
            elif price_change < -self.price_change_threshold:
                self.current_signal_type = SignalType.SELL
            else:
                self.current_signal_type = SignalType.NEUTRAL
        else:
            self.current_signal_type = SignalType.NEUTRAL

        metadata = {
            "price_change": (close - list(self.close_history)[-self.lookback_period] if len(self.close_history) >= self.lookback_period else np.nan) / (list(self.close_history)[-self.lookback_period] if len(self.close_history) >= self.lookback_period and list(self.close_history)[-self.lookback_period] != 0 else np.nan),
            "lookback_period": self.lookback_period
        }

        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.close_history = deque(maxlen=200)
        self.current_signal_type = SignalType.NEUTRAL        



class Rule14:
    def __init__(self, params):
        self.atr_period = params.get('atr_period', 14)
        self.atr_multiplier = params.get('atr_multiplier', 3)
        self._calculate_tr = self._calculate_true_range  # Assuming it's a method
        self.high_history = deque(maxlen=200) # Assuming a maxlen
        self.low_history = deque(maxlen=200) # Assuming a maxlen
        self.close_history = deque(maxlen=200) # Assuming a maxlen
        self.atr_history = deque(maxlen=200) # Assuming a maxlen
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule14"
        self.position = 0
        self.trailing_stop = np.nan
        self.trend_direction = 0

    def _calculate_true_range(self, current_high, current_low, previous_close):
        return max(current_high - current_low,
                   abs(current_high - previous_close),
                   abs(current_low - previous_close))

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        if len(self.close_history) > 1:
            prev_close = list(self.close_history)[-2]
            current_tr = self._calculate_tr(high, low, prev_close)

            if len(self.close_history) <= self.atr_period:
                tr_values = [self._calculate_tr(list(self.high_history)[i], list(self.low_history)[i], list(self.close_history)[i-1])
                             for i in range(1, len(self.close_history))]
                self.atr_value = np.mean(tr_values) if tr_values else np.nan
            else:
                prev_atr = self.atr_value if not np.isnan(self.atr_value) else current_tr
                self.atr_value = (prev_atr * (self.atr_period - 1) + current_tr) / self.atr_period

            if not np.isnan(self.atr_value):
                atr_band = self.atr_value * self.atr_multiplier

                # Determine trend direction using a simple moving average
                if len(self.close_history) >= 5:
                    ma5 = np.mean(list(self.close_history)[-5:])
                    ma10 = np.mean(list(self.close_history)[-min(10, len(self.close_history)):])
                    self.trend_direction = 1 if ma5 > ma10 else (-1 if ma5 < ma10 else 0)

                # Position management with ATR trailing stop
                if self.position == 0:  # Not in a position
                    # Enter position in the direction of the trend
                    if self.trend_direction > 0:
                        self.position = 1
                        self.trailing_stop = close - atr_band
                        self.current_signal_type = SignalType.BUY
                    elif self.trend_direction < 0:
                        self.position = -1
                        self.trailing_stop = close + atr_band
                        self.current_signal_type = SignalType.SELL
                    else:
                        self.current_signal_type = SignalType.NEUTRAL

                elif self.position == 1:  # Long position
                    # Update trailing stop if price moves higher
                    new_stop = close - atr_band
                    if new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop

                    # Exit if price falls below trailing stop
                    if low < self.trailing_stop:
                        self.current_signal_type = SignalType.NEUTRAL
                        self.position = 0
                    else:
                        self.current_signal_type = SignalType.BUY  # Maintain long signal

                elif self.position == -1:  # Short position
                    # Update trailing stop if price moves lower
                    new_stop = close + atr_band
                    if new_stop < self.trailing_stop or np.isnan(self.trailing_stop):
                        self.trailing_stop = new_stop

                    # Exit if price rises above trailing stop
                    if high > self.trailing_stop:
                        self.current_signal_type = SignalType.NEUTRAL
                        self.position = 0
                    else:
                        self.current_signal_type = SignalType.SELL  # Maintain short signal
        else:
            self.atr_value = np.nan
            self.current_signal_type = SignalType.NEUTRAL

        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.high_history = deque(maxlen=200)
        self.low_history = deque(maxlen=200)
        self.close_history = deque(maxlen=200)
        self.atr_history = deque(maxlen=200)
        self.atr_value = np.nan
        self.trailing_stop = np.nan
        self.position = 0
        self.current_signal_type = SignalType.NEUTRAL
        self.trend_direction = 0        


from signals import Signal, SignalType

class Rule15:
    def __init__(self, params):
        self.bb_period = params.get('bb_period', 20)
        self.bb_high_history = deque(maxlen=200)
        self.bb_mid_history = deque(maxlen=200)
        self.bb_low_history = deque(maxlen=200)
        self.close_history = deque(maxlen=200) # Assuming a maxlen
        self.bb_high = []
        self.bb_mid = []
        self.bb_low = []
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "Rule15"

    def on_bar(self, bar):
        close = bar['Close']
        self.close_history.append(close)

        if len(self.close_history) == self.bb_period:
            closes_array = np.array(self.close_history)
            self.sma = np.mean(closes_array)
            self.std_dev = np.std(closes_array)
            self.upper_band = self.sma + (self.std_dev * self.bb_std_dev)
            self.lower_band = self.sma - (self.std_dev * self.bb_std_dev)
        elif len(self.close_history) > self.bb_period:
            closes_array = np.array(self.close_history)
            self.sma = np.mean(closes_array)
            self.std_dev = np.std(closes_array)
            self.upper_band = self.sma + (self.std_dev * self.bb_std_dev)
            self.lower_band = self.sma - (self.std_dev * self.bb_std_dev)
        else:
            self.sma = np.nan
            self.std_dev = np.nan
            self.upper_band = np.nan
            self.lower_band = np.nan

        if not np.isnan(self.lower_band) and not np.isnan(self.upper_band) and len(self.close_history) >= 2:
            prev_close = list(self.close_history)[-2]
            current_close = list(self.close_history)[-1]
            
            # Signal generation logic with state persistence
            if not self.in_position:
                # Entry logic
                if prev_close <= self.lower_band and current_close > self.lower_band:
                    self.current_signal_type = SignalType.BUY  # Buy signal on lower band breakout
                    self.in_position = True
                    self.position_type = 'lower_break'
                elif prev_close >= self.upper_band and current_close < self.upper_band:
                    self.current_signal_type = SignalType.SELL  # Sell signal on upper band breakout
                    self.in_position = True
                    self.position_type = 'upper_break'
            else:
                # Exit logic - maintain position until mean reversion or opposite band test
                if self.position_type == 'lower_break':
                    # Exit long if price reaches the middle band or upper band
                    if current_close >= self.sma or current_close >= self.upper_band:
                        self.current_signal_type = SignalType.NEUTRAL
                        self.in_position = False
                        self.position_type = None
                    else:
                        self.current_signal_type = SignalType.BUY  # Maintain long signal
                elif self.position_type == 'upper_break':
                    # Exit short if price reaches the middle band or lower band
                    if current_close <= self.sma or current_close <= self.lower_band:
                        self.current_signal_type = SignalType.NEUTRAL
                        self.in_position = False
                        self.position_type = None
                    else:
                        self.current_signal_type = SignalType.SELL  # Maintain short signal
        else:
            self.current_signal_type = SignalType.NEUTRAL

        
        # Create metadata with any additional information
        metadata = {}

        # Return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata=metadata
        )

    def reset(self):
        self.close_history = deque(maxlen=self.bb_period)
        self.sma = np.nan
        self.std_dev = np.nan
        self.upper_band = np.nan
        self.lower_band = np.nan
        self.current_signal_type = SignalType.NEUTRAL
        self.in_position = False
        self.position_type = None        




class TopNStrategy:
    """\
    A strategy that combines signals from multiple rules using a simple voting mechanism.

    This implementation focuses on its core responsibility: combining signals from
    top-performing rules to generate trading decisions.
    """
    def __init__(self, rule_objects):
        """
        Initialize the TopN strategy with rule objects.

        Args:
            rule_objects: List of rule instances to use
        """
        from signals import SignalRouter, SignalType

        self.rules = rule_objects
        self.router = SignalRouter(rule_objects)
        self.last_signal = None

    def on_bar(self, event):
        """
        Process a bar and generate a signal by combining rule signals.

        Args:
            event: Bar event containing market data

        Returns:
            dict: Signal information including timestamp, signal value, and price
        """
        # Get standardized signals from all rules via the router
        router_output = self.router.on_bar(event)
        signal_collection = router_output["signals"]

        # Use SignalCollection's weighted consensus to determine the overall signal
        consensus_signal_type = signal_collection.get_weighted_consensus()
        consensus_signal_value = consensus_signal_type.value

        # Create the final output signal
        self.last_signal = {
            "timestamp": router_output["timestamp"],
            "signal": consensus_signal_value,
            "price": router_output["price"]
        }

        return self.last_signal

    def reset(self):
        """Reset the strategy and all rules."""
        self.router.reset()
        self.last_signal = None
