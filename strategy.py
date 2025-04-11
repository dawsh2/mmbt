from collections import deque
import pandas as pd
import numpy as np
from ta import ema
from ta import DEMA
from ta import TEMA
from ta import rsi
from ta import stoch
from ta import vortex_indicator_pos
from ta import vortex_indicator_neg
from ta import ichimoku_a
from ta import ichimoku_b
from ta import cci
from ta import keltner_channel_hband
from ta import keltner_channel_lband
from ta import donchian_channel_hband
from ta import donchian_channel_lband
from ta import bollinger_hband
from ta import bollinger_lband
# 


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

class Rule0:
    """
    Event-driven version of Rule0: Simple Moving Average Crossover.
    Buy signal when fast SMA crosses above slow SMA.
    Sell signal when fast SMA crosses below slow SMA.
    """
    def __init__(self, params):
        self.fast_window = int(params['fast_window'])
        self.slow_window = int(params['slow_window'])
        self.prices = []
        self.fast_sma_history = []
        self.slow_sma_history = []
        self.current_signal = 0

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
                    self.current_signal = 1  # Bullish state
                elif current_fast_sma < current_slow_sma:
                    self.current_signal = -1  # Bearish state
                else:
                    self.current_signal = 0  # Equal (rare)
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.prices = []
        self.fast_sma_history = []
        self.slow_sma_history = []
        self.current_signal = 0


class Rule1:
    """
    Event-driven version of Rule1: Simple Moving Average Crossover - Optimized SMA with persistent signals.
    """
    def __init__(self, param):
        self.ma1_period = param['ma1']
        self.ma2_period = param['ma2']
        self.history = []
        self.sum1 = 0
        self.sum2 = 0
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        # Efficient SMA 1 calculation
        if len(self.history) > self.ma1_period:
            self.sum1 -= self.history[-(self.ma1_period + 1)]
        self.sum1 += close
        if len(self.history) >= self.ma1_period:
            sma1 = self.sum1 / self.ma1_period
            self.s1_history.append(sma1)

        # Efficient SMA 2 calculation
        if len(self.history) > self.ma2_period:
            self.sum2 -= self.history[-(self.ma2_period + 1)]
        self.sum2 += close
        if len(self.history) >= self.ma2_period:
            sma2 = self.sum2 / self.ma2_period
            self.s2_history.append(sma2)

        if len(self.s1_history) >= 1 and len(self.s2_history) >= 1:
            # Get the most recent values
            curr_s1 = self.s1_history[-1]
            curr_s2 = self.s2_history[-1]

            # Signal based on current MA relationship, not just crossover
            if curr_s1 > curr_s2:
                self.current_signal = 1  # Bullish state
            elif curr_s1 < curr_s2:
                self.current_signal = -1  # Bearish state
            else:
                self.current_signal = 0  # Equal (rare)
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = []
        self.sum1 = 0
        self.sum2 = 0
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0
        


class Rule2:
    """
    Event-driven version of Rule2: EMA Crossover with MA Confirmation - Fully Optimized with persistent signals.
    """
    def __init__(self, param):
        self.ema1_period = param['ema1_period']
        self.ma2_period = param['ma2_period']
        self.history = []
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ma2_sum = 0
        self.ma2_history = []
        self.current_signal = 0
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0

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
                    self.current_signal = 1
                elif curr_ema1 < curr_ma2 and current_price < curr_ma2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = []
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ma2_sum = 0
        self.ma2_history = []
        self.current_signal = 0
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0


class Rule3:
    """
    Event-driven version of Rule3: EMA and EMA - Optimized with persistent signals.
    """
    def __init__(self, param):
        self.ema1_period = param['ema1_period']
        self.ema2_period = param['ema2_period']
        self.history = {'close': []}
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ema2_value = np.nan
        self.ema2_history = []
        self.current_signal = 0
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0
        self.alpha_ema2 = 2 / (self.ema2_period + 1) if self.ema2_period > 0 else 0

    def on_bar(self, bar):
        close = bar['Close']
        self.history['close'].append(close)

        # Incremental EMA 1 calculation
        if self.alpha_ema1 > 0:
            if np.isnan(self.ema1_value):
                self.ema1_value = close  # Initialize with the first data point
            else:
                self.ema1_value = (close * self.alpha_ema1) + (self.ema1_value * (1 - self.alpha_ema1))
            self.ema1_history.append(self.ema1_value)
        else:
            self.ema1_history.append(np.nan)

        # Incremental EMA 2 calculation
        if self.alpha_ema2 > 0:
            if np.isnan(self.ema2_value):
                self.ema2_value = close  # Initialize with the first data point
            else:
                self.ema2_value = (close * self.alpha_ema2) + (self.ema2_value * (1 - self.alpha_ema2))
            self.ema2_history.append(self.ema2_value)
        else:
            self.ema2_history.append(np.nan)

        if len(self.ema1_history) >= 1 and len(self.ema2_history) >= 1:
            curr_ema1 = self.ema1_history[-1]
            curr_ema2 = self.ema2_history[-1]

            if not np.isnan(curr_ema1) and not np.isnan(curr_ema2):
                # Persistent signal based on current relationship between EMAs
                if curr_ema1 > curr_ema2:
                    self.current_signal = 1
                elif curr_ema1 < curr_ema2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ema2_value = np.nan
        self.ema2_history = []
        self.current_signal = 0
        self.alpha_ema1 = 2 / (self.ema1_period + 1) if self.ema1_period > 0 else 0
        self.alpha_ema2 = 2 / (self.ema2_period + 1) if self.ema2_period > 0 else 0

class Rule4:
    """
    Event-driven version of Rule4: DEMA and MA (formerly Rule4) - Further Optimized.
    """
    def __init__(self, param):
        self.dema1_period = param['dema1_period']
        self.ma2_period = param['ma2_period']
        self.history = []  # Store only close prices
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.current_signal = 0
        self.alpha1 = 2 / (self.dema1_period + 1) if self.dema1_period > 0 else 0

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        # Incremental EMA 1
        if self.alpha1 > 0:
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
        if len(self.history) >= max(self.dema1_period * 2 - 1 if self.dema1_period > 0 else 0, self.ma2_period):
            curr_dema1 = self.dema1_value
            curr_ma2 = ma2_val

            if not np.isnan(curr_dema1) and not np.isnan(curr_ma2):
                # Persistent signal based on current state
                if curr_dema1 > curr_ma2:
                    self.current_signal = 1
                elif curr_dema1 < curr_ma2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.history = []
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.current_signal = 0
        self.alpha1 = 2 / (self.dema1_period + 1) if self.dema1_period > 0 else 0


class Rule5:
    """
    Event-driven version of Rule5: DEMA and DEMA - Optimized with persistent signals.
    """
    def __init__(self, param):
        self.dema1_period = param['dema1_period']
        self.dema2_period = param['dema2_period']
        self.history = {'close': []}
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.dema1_history = []
        self.ema3_value = np.nan
        self.ema3_history = []
        self.ema4_value = np.nan
        self.dema2_value = np.nan
        self.dema2_history = []
        self.current_signal = 0
        self.alpha1 = 2 / (self.dema1_period + 1) if self.dema1_period > 0 else 0
        self.alpha2 = 2 / (self.dema2_period + 1) if self.dema2_period > 0 else 0

    def on_bar(self, bar):
        close = bar['Close']
        self.history['close'].append(close)

        # Incremental DEMA 1 calculation
        if self.alpha1 > 0:
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
        if self.alpha2 > 0:
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
                    self.current_signal = 1
                elif curr_dema1 < curr_dema2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.ema1_value = np.nan
        self.ema1_history = []
        self.ema2_value = np.nan
        self.dema1_value = np.nan
        self.dema1_history = []
        self.ema3_value = np.nan
        self.ema3_history = []
        self.ema4_value = np.nan
        self.dema2_value = np.nan
        self.dema2_history = []
        self.current_signal = 0
        self.alpha1 = 2 / (self.dema1_period + 1) if self.dema1_period > 0 else 0
        self.alpha2 = 2 / (self.dema2_period + 1) if self.dema2_period > 0 else 0



class Rule6:
    """
    Event-driven version of Rule6: TEMA and MA crossovers - Optimized TEMA and MA with persistent signals.
    """
    def __init__(self, param):
        self.tema1_period = param['tema1_period']
        self.ma2_period = param['ma2_period']
        self.history = []  # Store only close prices
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.ema3_value = np.nan
        self.tema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.current_signal = 0
        self.alpha1 = 2 / (self.tema1_period + 1) if self.tema1_period > 0 else 0

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        # Incremental EMA 1
        if self.alpha1 > 0:
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

        if len(self.history) >= max(self.tema1_period * 3 - 2 if self.tema1_period > 0 else 0, self.ma2_period):
            curr_tema1 = self.tema1_value
            curr_ma2 = ma2_val

            if not np.isnan(curr_tema1) and not np.isnan(curr_ma2):
                # Persistent signal based on current TEMA vs MA relationship
                if curr_tema1 > curr_ma2:
                    self.current_signal = 1
                elif curr_tema1 < curr_ma2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.history = []
        self.ema1_value = np.nan
        self.ema2_value = np.nan
        self.ema3_value = np.nan
        self.tema1_value = np.nan
        self.ma2_sum = 0
        self.ma2_count = 0
        self.current_signal = 0
        self.alpha1 = 2 / (self.tema1_period + 1) if self.tema1_period > 0 else 0

class Rule7:
    """
    Event-driven version of Rule7: Stochastic crossover - Further Optimized with persistent signals.
    """
    def __init__(self, param):
        self.stoch1_period = param['stoch1_period']
        self.stochma2_period = param['stochma2_period']
        self.high_history = deque(maxlen=self.stoch1_period)
        self.low_history = deque(maxlen=self.stoch1_period)
        self.close_history = deque(maxlen=self.stoch1_period)
        self.s1_history = []  # Stochastic %K values
        self.s2_sum = 0
        self.s2_count = 0
        self.s2_value = np.nan  # SMA of %K
        self.current_signal = 0

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

        if len(self.s1_history) >= 1 and self.s2_value is not np.nan and len(self.s1_history) >= self.stochma2_period:
            curr_s1 = self.s1_history[-1]  # %K
            curr_s2 = self.s2_value        # %D

            if not np.isnan(curr_s1) and not np.isnan(curr_s2):
                # Persistent signal based on current %K vs %D relationship
                if curr_s1 > curr_s2:
                    self.current_signal = 1
                elif curr_s1 < curr_s2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.high_history = deque(maxlen=self.stoch1_period)
        self.low_history = deque(maxlen=self.stoch1_period)
        self.close_history = deque(maxlen=self.stoch1_period)
        self.s1_history = []
        self.s2_sum = 0
        self.s2_count = 0
        self.s2_value = np.nan
        self.current_signal = 0



class Rule8:
    """
    Event-driven version of Rule8: Vortex indicator crossover - Optimized with persistent signals.
    """
    def __init__(self, param):
        self.vortex1_period = param['vortex1_period']
        self.vortex2_period = param['vortex2_period']
        self.history = {'high': [], 'low': [], 'close': []}
        self.tr_sum = 0
        self.pm_sum = 0
        self.nm_sum = 0
        self.current_tr = 0
        self.current_pm = 0
        self.current_nm = 0
        self.s1_history = []  # +VI
        self.s2_history = []  # -VI
        self.current_signal = 0

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']
        self.history['high'].append(high)
        self.history['low'].append(low)
        self.history['close'].append(close)

        if len(self.history['close']) >= 2:
            prev_close = self.history['close'][-2]
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

        if len(self.history['close']) > self.vortex1_period:
            # Check if enough history exists before subtracting
            if len(self.history['close']) >= self.vortex1_period + 1:
                prev_high_v1 = self.history['high'][-(self.vortex1_period + 1)]
                prev_low_v1 = self.history['low'][-(self.vortex1_period + 1)]
                prev_close_v1_minus_1 = self.history['close'][-(self.vortex1_period + 2)] if len(self.history['close']) >= self.vortex1_period + 2 else self.history['close'][-(self.vortex1_period + 1)] # Handle edge case

                self.tr_sum -= np.max([prev_high_v1 - prev_low_v1,
                                       np.abs(prev_high_v1 - prev_close_v1_minus_1),
                                       np.abs(prev_low_v1 - prev_close_v1_minus_1)])
                self.pm_sum -= np.abs(prev_high_v1 - prev_close_v1_minus_1) if (prev_high_v1 - prev_close_v1_minus_1) > (prev_close_v1_minus_1 - prev_low_v1) else 0
                self.nm_sum -= np.abs(prev_low_v1 - prev_close_v1_minus_1) if (prev_close_v1_minus_1 - prev_low_v1) > (prev_high_v1 - prev_close_v1_minus_1) else 0

        if len(self.history['close']) >= self.vortex1_period:
            vi_pos = self.pm_sum / self.tr_sum if self.tr_sum != 0 else 0
            self.s1_history.append(vi_pos)
        else:
            self.s1_history.append(np.nan)

        # Recalculate current TR, PM, NM for the second period
        if len(self.history['close']) >= 2:
            prev_close = self.history['close'][-2]
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

        if len(self.history['close']) > self.vortex2_period:
            # Check if enough history exists before subtracting
            if len(self.history['close']) >= self.vortex2_period + 1:
                prev_high_v2 = self.history['high'][-(self.vortex2_period + 1)]
                prev_low_v2 = self.history['low'][-(self.vortex2_period + 1)]
                prev_close_v2_minus_1 = self.history['close'][-(self.vortex2_period + 2)] if len(self.history['close']) >= self.vortex2_period + 2 else self.history['close'][-(self.vortex2_period + 1)] # Handle edge case

                self.tr_sum -= np.max([prev_high_v2 - prev_low_v2,
                                       np.abs(prev_high_v2 - prev_close_v2_minus_1),
                                       np.abs(prev_low_v2 - prev_close_v2_minus_1)])
                self.pm_sum -= np.abs(prev_high_v2 - prev_close_v2_minus_1) if (prev_high_v2 - prev_close_v2_minus_1) > (prev_close_v2_minus_1 - prev_low_v2) else 0
                self.nm_sum -= np.abs(prev_low_v2 - prev_close_v2_minus_1) if (prev_close_v2_minus_1 - prev_low_v2) > (prev_high_v2 - prev_close_v2_minus_1) else 0

        if len(self.history['close']) >= self.vortex2_period:
            vi_neg = self.nm_sum / self.tr_sum if self.tr_sum != 0 else 0
            self.s2_history.append(vi_neg)
        else:
            self.s2_history.append(np.nan)

        if len(self.s1_history) >= 1 and len(self.s2_history) >= 1:
            curr_s1 = self.s1_history[-1]  # +VI
            curr_s2 = self.s2_history[-1]  # -VI

            if not np.isnan(curr_s1) and not np.isnan(curr_s2):
                # Persistent signal based on current VI+ vs VI- relationship
                if curr_s1 > curr_s2:
                    self.current_signal = 1
                elif curr_s1 < curr_s2:
                    self.current_signal = -1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'high': [], 'low': [], 'close': []}
        self.tr_sum = 0
        self.pm_sum = 0
        self.nm_sum = 0
        self.current_tr = 0
        self.current_pm = 0
        self.current_nm = 0
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0
        

class Rule9:
    """
    Event-driven version of Rule9: Ichimoku cloud - Incremental Calculation with persistent signals.
    """
    def __init__(self, param):
        self.p1 = param['p1']
        self.p2 = param['p2']
        self.n2 = round((self.p1 + self.p2) / 2)
        self.n3 = 52  # For Senkou Span B
        self.high_history = deque(maxlen=max(self.p1, self.n2, self.n3))
        self.low_history = deque(maxlen=max(self.p1, self.n2, self.n3))
        self.close_history = deque(maxlen=self.n2)  # Store enough for shifting
        self.conversion_line_history = deque(maxlen=self.n2)
        self.base_line_history = deque(maxlen=self.n2)
        self.senkou_span_a_history = deque(maxlen=self.n2)
        self.senkou_span_b_history = deque(maxlen=self.n2)
        self.current_signal = 0

    def _calculate_conversion_line(self):
        if len(self.high_history) >= self.p1 and len(self.low_history) >= self.p1:
            highest_high = max(list(self.high_history)[-self.p1:])
            lowest_low = min(list(self.low_history)[-self.p1:])
            return 0.5 * (highest_high + lowest_low)
        return np.nan

    def _calculate_base_line(self):
        if len(self.high_history) >= self.n2 and len(self.low_history) >= self.n2:
            highest_high = max(list(self.high_history)[-self.n2:])
            lowest_low = min(list(self.low_history)[-self.n2:])
            return 0.5 * (highest_high + lowest_low)
        return np.nan

    def _calculate_senkou_span_a(self):
        conversion_line = self._calculate_conversion_line()
        base_line = self._calculate_base_line()
        if not np.isnan(conversion_line) and not np.isnan(base_line):
            return 0.5 * (conversion_line + base_line)
        return np.nan

    def _calculate_senkou_span_b(self):
        if len(self.high_history) >= self.n3 and len(self.low_history) >= self.n3:
            highest_high = max(list(self.high_history)[-self.n3:])
            lowest_low = min(list(self.low_history)[-self.n3:])
            return 0.5 * (highest_high + lowest_low)
        return np.nan

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        conversion_line = self._calculate_conversion_line()
        base_line = self._calculate_base_line()
        senkou_span_a = self._calculate_senkou_span_a()
        senkou_span_b = self._calculate_senkou_span_b()

        # Store calculated values
        if not np.isnan(senkou_span_a):
            self.senkou_span_a_history.append(senkou_span_a)
        if not np.isnan(senkou_span_b):
            self.senkou_span_b_history.append(senkou_span_b)
        if not np.isnan(conversion_line):
            self.conversion_line_history.append(conversion_line)
        if not np.isnan(base_line):
            self.base_line_history.append(base_line)

        # Generate signal based on current price's relationship to the cloud
        if len(self.senkou_span_a_history) >= 1 and len(self.senkou_span_b_history) >= 1:
            current_span_a = self.senkou_span_a_history[-1]
            current_span_b = self.senkou_span_b_history[-1]

            if not np.isnan(current_span_a) and not np.isnan(current_span_b):
                # Persistent signal based on price vs cloud relationship
                if close > max(current_span_a, current_span_b):
                    self.current_signal = 1  # Price above cloud (bullish)
                elif close < min(current_span_a, current_span_b):
                    self.current_signal = -1  # Price below cloud (bearish)
                else:
                    self.current_signal = 0  # Price inside cloud (neutral)
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.high_history = deque(maxlen=max(self.p1, self.n2, self.n3))
        self.low_history = deque(maxlen=max(self.p1, self.n2, self.n3))
        self.close_history = deque(maxlen=self.n2)
        self.conversion_line_history = deque(maxlen=self.n2)
        self.base_line_history = deque(maxlen=self.n2)
        self.senkou_span_a_history = deque(maxlen=self.n2)
        self.senkou_span_b_history = deque(maxlen=self.n2)
        self.current_signal = 0

class Rule10:
    """
    Event-driven version of Rule10: RSI Overbought/Oversold - Optimized with persistent signals and zones.
    """
    def __init__(self, param):
        self.rsi1_period = param['rsi1_period']
        self.overbought_level = param.get('c2_threshold', 70)  # Use c2_threshold or default to 70
        self.oversold_level = 100 - self.overbought_level  # Complement of overbought level
        self.rsi_value = np.nan
        self.avg_gain = 0
        self.avg_loss = 0
        self.history = []  # Store close prices
        self.gain_history = []
        self.loss_history = []
        self.alpha = 1 / self.rsi1_period if self.rsi1_period > 0 else 0
        self.current_signal = 0
        self.last_zone = 'neutral'  # Track which zone we're in to maintain signal

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        if len(self.history) >= 2:
            change = close - self.history[-2]
            gain = max(change, 0)
            loss = abs(min(change, 0))
            self.gain_history.append(gain)
            self.loss_history.append(loss)

            if len(self.gain_history) <= self.rsi1_period:
                self.avg_gain = np.mean(self.gain_history) if self.gain_history else 0
                self.avg_loss = np.mean(self.loss_history) if self.loss_history else 0
            else:
                self.avg_gain = (self.avg_gain * (self.rsi1_period - 1) + gain) / self.rsi1_period
                self.avg_loss = (self.avg_loss * (self.rsi1_period - 1) + loss) / self.rsi1_period

            if self.avg_loss != 0:
                rs = self.avg_gain / self.avg_loss
                self.rsi_value = 100 - (100 / (1 + rs))
            else:
                self.rsi_value = 100 if self.avg_gain > 0 else 50  # Handle cases with no loss

            # Signal with hysteresis for persistence
            if self.rsi_value > self.overbought_level:
                self.current_signal = -1  # Overbought condition (sell)
                self.last_zone = 'overbought'
            elif self.rsi_value < self.oversold_level:
                self.current_signal = 1    # Oversold condition (buy)
                self.last_zone = 'oversold'
            elif self.last_zone == 'overbought' and self.rsi_value > (self.overbought_level - 10):
                # Stay in overbought zone until RSI drops significantly
                self.current_signal = -1
            elif self.last_zone == 'oversold' and self.rsi_value < (self.oversold_level + 10):
                # Stay in oversold zone until RSI rises significantly
                self.current_signal = 1
            else:
                self.current_signal = 0
                self.last_zone = 'neutral'
        else:
            self.rsi_value = np.nan
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.rsi_value = np.nan
        self.avg_gain = 0
        self.avg_loss = 0
        self.history = []
        self.gain_history = []
        self.loss_history = []
        self.alpha = 1 / self.rsi1_period if self.rsi1_period > 0 else 0
        self.current_signal = 0
        self.last_zone = 'neutral'


class Rule11:
    """
    Event-driven version of Rule11: CCI Overbought/Oversold - Optimized with persistent signals and zones.
    """
    def __init__(self, param):
        self.cci1_period = param['cci1_period']
        self.cci_threshold = param.get('c2_threshold', 100)  # Use c2_threshold or default to 100
        self.cci_value = np.nan
        self.history = {'high': [], 'low': [], 'close': []}
        self.sma_pp = np.nan
        self.mean_deviation = np.nan
        self.current_signal = 0
        self.constant = 0.015
        self.last_zone = 'neutral'  # Track which zone we're in to maintain signal

    def _calculate_pivot_price(self, high, low, close):
        return (high + low + close) / 3.0

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.history['high'].append(high)
        self.history['low'].append(low)
        self.history['close'].append(close)

        if len(self.history['close']) >= self.cci1_period:
            highs = np.array(self.history['high'][-self.cci1_period:])
            lows = np.array(self.history['low'][-self.cci1_period:])
            closes = np.array(self.history['close'][-self.cci1_period:])
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
                self.current_signal = -1  # Overbought condition (sell)
                self.last_zone = 'overbought'
            elif self.cci_value < -self.cci_threshold:
                self.current_signal = 1    # Oversold condition (buy)
                self.last_zone = 'oversold'
            elif self.last_zone == 'overbought' and self.cci_value > (self.cci_threshold * 0.8):
                # Stay in overbought zone until CCI drops significantly
                self.current_signal = -1
            elif self.last_zone == 'oversold' and self.cci_value < (-self.cci_threshold * 0.8):
                # Stay in oversold zone until CCI rises significantly
                self.current_signal = 1
            else:
                self.current_signal = 0
                self.last_zone = 'neutral'
        else:
            self.cci_value = np.nan
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.cci_value = np.nan
        self.history = {'high': [], 'low': [], 'close': []}
        self.sma_pp = np.nan
        self.mean_deviation = np.nan
        self.current_signal = 0
        self.constant = 0.015
        self.last_zone = 'neutral'



class Rule12:
    """
    Event-driven version of Rule12: RSI-based strategy - Optimized with persistent signals and hysteresis.
    """
    def __init__(self, param):
        self.rsi_period = param['rsi_period']
        self.overbought_level = param.get('overbought', 70)
        self.oversold_level = param.get('oversold', 30)
        self.exit_buffer = 5  # Buffer for exiting a zone
        self.rsi_value = np.nan
        self.avg_gain = 0
        self.avg_loss = 0
        self.history = []
        self.gain_history = []
        self.loss_history = []
        self.current_signal = 0
        self.in_trade = False  # Track if we're in a trade

    def on_bar(self, bar):
        close = bar['Close']
        self.history.append(close)

        if len(self.history) >= 2:
            change = close - self.history[-2]
            gain = max(change, 0)
            loss = abs(min(change, 0))
            self.gain_history.append(gain)
            self.loss_history.append(loss)

            if len(self.gain_history) <= self.rsi_period:
                self.avg_gain = np.mean(self.gain_history) if self.gain_history else 0
                self.avg_loss = np.mean(self.loss_history) if self.loss_history else 0
            else:
                self.avg_gain = (self.avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
                self.avg_loss = (self.avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period

            if self.avg_loss != 0:
                rs = self.avg_gain / self.avg_loss
                self.rsi_value = 100 - (100 / (1 + rs))
            else:
                self.rsi_value = 100 if self.avg_gain > 0 else 50

            # Implement signal persistence with hysteresis for better trade management
            if not self.in_trade:
                # Entry logic
                if self.rsi_value < self.oversold_level:
                    self.current_signal = 1   # Buy signal
                    self.in_trade = True
                elif self.rsi_value > self.overbought_level:
                    self.current_signal = -1  # Sell signal
                    self.in_trade = True
                else:
                    self.current_signal = 0
            else:
                # Exit logic with hysteresis
                if self.current_signal == 1:  # In a long position
                    # Exit long position if RSI moves significantly above oversold
                    if self.rsi_value > (self.oversold_level + self.exit_buffer):
                        self.current_signal = 0
                        self.in_trade = False
                elif self.current_signal == -1:  # In a short position
                    # Exit short position if RSI moves significantly below overbought
                    if self.rsi_value < (self.overbought_level - self.exit_buffer):
                        self.current_signal = 0
                        self.in_trade = False
        else:
            self.rsi_value = np.nan
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.rsi_value = np.nan
        self.avg_gain = 0
        self.avg_loss = 0
        self.history = []
        self.gain_history = []
        self.loss_history = []
        self.current_signal = 0
        self.in_trade = False
        

class Rule13:
    """
    Event-driven version of Rule13: Stochastic Oscillator strategy - Optimized with persistent signals.
    """
    def __init__(self, param):
        self.stoch_period = param['stoch_period']
        self.stoch_d_period = param.get('stoch_d_period', 3)
        self.overbought_level = param.get('overbought', 80)
        self.oversold_level = param.get('oversold', 20)
        self.high_history = deque(maxlen=self.stoch_period)
        self.low_history = deque(maxlen=self.stoch_period)
        self.close_history = deque(maxlen=self.stoch_period)
        self.stoch_k = np.nan
        self.stoch_d = np.nan
        self.stoch_k_history = deque(maxlen=self.stoch_d_period)
        self.current_signal = 0
        self.in_trade = False
        self.trade_type = None  # 'oversold' or 'overbought'

    def on_bar(self, bar):
        high = bar['High']
        low = bar['Low']
        close = bar['Close']

        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)

        if len(self.close_history) == self.stoch_period:
            highest_high = max(self.high_history)
            lowest_low = min(self.low_history)
            if highest_high != lowest_low:
                self.stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                self.stoch_k = 50
            self.stoch_k_history.append(self.stoch_k)
        elif len(self.close_history) > self.stoch_period:
            highest_high = max(self.high_history)
            lowest_low = min(self.low_history)
            if highest_high != lowest_low:
                self.stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                self.stoch_k = 50
            self.stoch_k_history.append(self.stoch_k)
        else:
            self.stoch_k = np.nan

        if self.stoch_k_history:
            self.stoch_d = np.mean(self.stoch_k_history)
        else:
            self.stoch_d = np.nan

        if not np.isnan(self.stoch_k) and not np.isnan(self.stoch_d):
            # Persistent signal logic
            if not self.in_trade:
                # Entry signals with crossover confirmation
                if self.stoch_k > self.stoch_d and self.stoch_k < self.oversold_level:
                    self.current_signal = 1  # Buy signal - oversold with %K rising
                    self.in_trade = True
                    self.trade_type = 'oversold'
                elif self.stoch_k < self.stoch_d and self.stoch_k > self.overbought_level:
                    self.current_signal = -1  # Sell signal - overbought with %K falling
                    self.in_trade = True
                    self.trade_type = 'overbought'
            else:
                # Exit logic
                if self.trade_type == 'oversold':
                    # Exit long when stochastic moves to overbought region
                    if self.stoch_k > self.overbought_level or self.stoch_k < self.stoch_d:
                        self.current_signal = 0
                        self.in_trade = False
                        self.trade_type = None
                elif self.trade_type == 'overbought':
                    # Exit short when stochastic moves to oversold region
                    if self.stoch_k < self.oversold_level or self.stoch_k > self.stoch_d:
                        self.current_signal = 0
                        self.in_trade = False
                        self.trade_type = None
        else:
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.high_history = deque(maxlen=self.stoch_period)
        self.low_history = deque(maxlen=self.stoch_period)
        self.close_history = deque(maxlen=self.stoch_period)
        self.stoch_k = np.nan
        self.stoch_d = np.nan
        self.stoch_k_history = deque(maxlen=self.stoch_d_period)
        self.current_signal = 0
        self.in_trade = False
        self.trade_type = None


class Rule14:
    """
    Event-driven version of Rule14: ATR Trailing Stop - Optimized with better trade management.
    """
    def __init__(self, param):
        self.atr_period = param['atr_period']
        self.atr_multiplier = param.get('atr_multiplier', 3)
        self.high_history = deque(maxlen=self.atr_period + 1)
        self.low_history = deque(maxlen=self.atr_period + 1)
        self.close_history = deque(maxlen=self.atr_period + 1)
        self.atr_value = np.nan
        self.trailing_stop = np.nan
        self.position = 0  # 1 for long, -1 for short
        self.current_signal = 0
        self.trend_direction = 0  # Track the market trend

    def _calculate_tr(self, high, low, prev_close):
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

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
                tr_values = [self._calculate_tr(self.high_history[i], self.low_history[i], self.close_history[i-1])
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
                        self.current_signal = 1
                    elif self.trend_direction < 0:
                        self.position = -1
                        self.trailing_stop = close + atr_band
                        self.current_signal = -1
                    else:
                        self.current_signal = 0
                
                elif self.position == 1:  # Long position
                    # Update trailing stop if price moves higher
                    new_stop = close - atr_band
                    if new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop
                    
                    # Exit if price falls below trailing stop
                    if low < self.trailing_stop:
                        self.current_signal = 0
                        self.position = 0
                    else:
                        self.current_signal = 1  # Maintain long signal
                
                elif self.position == -1:  # Short position
                    # Update trailing stop if price moves lower
                    new_stop = close + atr_band
                    if new_stop < self.trailing_stop or np.isnan(self.trailing_stop):
                        self.trailing_stop = new_stop
                    
                    # Exit if price rises above trailing stop
                    if high > self.trailing_stop:
                        self.current_signal = 0
                        self.position = 0
                    else:
                        self.current_signal = -1  # Maintain short signal
        else:
            self.atr_value = np.nan
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.high_history = deque(maxlen=self.atr_period + 1)
        self.low_history = deque(maxlen=self.atr_period + 1)
        self.close_history = deque(maxlen=self.atr_period + 1)
        self.atr_value = np.nan
        self.trailing_stop = np.nan
        self.position = 0
        self.current_signal = 0
        self.trend_direction = 0


class Rule15:
    """
    Event-driven version of Rule15: Bollinger Bands strategy - Optimized with persistent signals.
    """
    def __init__(self, param):
        self.bb_period = param['bb_period']
        self.bb_std_dev = param.get('bb_std_dev', 2)
        self.close_history = deque(maxlen=self.bb_period)
        self.sma = np.nan
        self.std_dev = np.nan
        self.upper_band = np.nan
        self.lower_band = np.nan
        self.current_signal = 0
        self.in_position = False
        self.position_type = None  # 'upper_break' or 'lower_break'

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
                    self.current_signal = 1  # Buy signal on lower band breakout
                    self.in_position = True
                    self.position_type = 'lower_break'
                elif prev_close >= self.upper_band and current_close < self.upper_band:
                    self.current_signal = -1  # Sell signal on upper band breakout
                    self.in_position = True
                    self.position_type = 'upper_break'
            else:
                # Exit logic - maintain position until mean reversion or opposite band test
                if self.position_type == 'lower_break':
                    # Exit long if price reaches the middle band or upper band
                    if current_close >= self.sma or current_close >= self.upper_band:
                        self.current_signal = 0
                        self.in_position = False
                        self.position_type = None
                    else:
                        self.current_signal = 1  # Maintain long signal
                elif self.position_type == 'upper_break':
                    # Exit short if price reaches the middle band or lower band
                    if current_close <= self.sma or current_close <= self.lower_band:
                        self.current_signal = 0
                        self.in_position = False
                        self.position_type = None
                    else:
                        self.current_signal = -1  # Maintain short signal
        else:
            self.current_signal = 0

        return self.current_signal

    def reset(self):
        self.close_history = deque(maxlen=self.bb_period)
        self.sma = np.nan
        self.std_dev = np.nan
        self.upper_band = np.nan
        self.lower_band = np.nan
        self.current_signal = 0
        self.in_position = False
        self.position_type = None        

        

class TopNStrategy:
    """
    Combines multiple event-driven rule objects into a single strategy.
    Emits 1, -1, or 0 based on signal consensus (simple average vote).
    """
    def __init__(self, rule_objects):
        self.rules = rule_objects
        self.last_signal = None # Store the last emitted signal

    def on_bar(self, event):
        bar = event.bar
        rule_signals = [rule.on_bar(bar) for rule in self.rules]
        combined = self.combine_signals(rule_signals)

        self.last_signal = {
            "timestamp": bar["timestamp"],
            "signal": combined,
            "price": bar["Close"]
        }
        return self.last_signal # Return the signal dictionary

    def combine_signals(self, signals):
        # Basic average-vote logic
        avg = sum(signals) / len(signals)
        if avg > 0.7:
            return 1
        elif avg < -0.7:
            return -1
        return 0

    def reset(self):
        for rule in self.rules:
            rule.reset()
        self.last_signal = None
