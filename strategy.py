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
    Event-driven version of Rule0:
    Buy signal when fast SMA crosses above slow SMA.
    Sell signal when fast SMA crosses below slow SMA.
    """
    def __init__(self, params):
        self.fast_window = int(params[0])
        self.slow_window = int(params[1])
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

            if len(self.fast_sma_history) >= 2 and len(self.slow_sma_history) >= 2:
                prev_fast_sma = self.fast_sma_history[-2]
                prev_slow_sma = self.slow_sma_history[-2]
                current_fast_sma = self.fast_sma_history[-1]
                current_slow_sma = self.slow_sma_history[-1]

                if prev_fast_sma < prev_slow_sma and current_fast_sma > current_slow_sma:
                    self.current_signal = 1  # Buy signal
                elif prev_fast_sma > prev_slow_sma and current_fast_sma < current_slow_sma:
                    self.current_signal = -1 # Sell signal
                else:
                    self.current_signal = 0
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
    Event-driven version of Rule1: Simple Moving Average Crossover (formerly Rule1).
    """
    def __init__(self, param):
        self.ma1, self.ma2 = param
        self.history = []
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history.append(bar['close'])
        closes = np.array(self.history)

        if len(closes) >= max(self.ma1, self.ma2):
            s1 = np.mean(closes[-self.ma1:])
            s2 = np.mean(closes[-self.ma2:])
            self.s1_history.append(s1)
            self.s2_history.append(s2)

            if len(self.s1_history) >= 2 and len(self.s2_history) >= 2:
                prev_s1 = self.s1_history[-2]
                prev_s2 = self.s2_history[-2]

                if prev_s1 >= prev_s2 and s1 < s2:
                    self.current_signal = -1
                elif prev_s1 <= prev_s2 and s1 > s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = []
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0



class Rule2:
    """
    Event-driven version of Rule2: EMA and close (formerly Rule2).
    """
    def __init__(self, param):
        self.ema1_period, self.ma2_period = param
        self.history = {'close': []}
        self.ema1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.ema1_period:
            ema1_val = ema(closes, window=self.ema1_period)[-1]
            self.ema1_history.append(ema1_val)
        else:
            self.ema1_history.append(np.nan)

        if len(closes) >= self.ma2_period:
            s2 = np.mean(closes[-self.ma2_period:])
            self.s2_history.append(s2)
        else:
            self.s2_history.append(np.nan)

        if len(self.ema1_history) >= 2 and len(self.s2_history) >= 2:
            prev_ema1 = self.ema1_history[-2]
            curr_ema1 = self.ema1_history[-1]
            prev_s2 = self.s2_history[-2]
            curr_s2 = self.s2_history[-1]

            if not np.isnan(prev_ema1) and not np.isnan(curr_ema1) and not np.isnan(prev_s2) and not np.isnan(curr_s2):
                if prev_ema1 >= prev_s2 and curr_ema1 < curr_s2:
                    self.current_signal = -1
                elif prev_ema1 <= prev_s2 and curr_ema1 > curr_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.ema1_history = []
        self.s2_history = []
        self.current_signal = 0



class Rule3:
    """
    Event-driven version of Rule3: EMA and EMA (formerly Rule3).
    """
    def __init__(self, param):
        self.ema1_period, self.ema2_period = param
        self.history = {'close': []}
        self.ema1_history = []
        self.ema2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.ema1_period:
            ema1_val = ema(closes, window=self.ema1_period)[-1]
            self.ema1_history.append(ema1_val)
        else:
            self.ema1_history.append(np.nan)

        if len(closes) >= self.ema2_period:
            ema2_val = ema(closes, window=self.ema2_period)[-1]
            self.ema2_history.append(ema2_val)
        else:
            self.ema2_history.append(np.nan)

        if len(self.ema1_history) >= 2 and len(self.ema2_history) >= 2:
            prev_ema1 = self.ema1_history[-2]
            curr_ema1 = self.ema1_history[-1]
            prev_ema2 = self.ema2_history[-2]
            curr_ema2 = self.ema2_history[-1]

            if not np.isnan(prev_ema1) and not np.isnan(curr_ema1) and not np.isnan(prev_ema2) and not np.isnan(curr_ema2):
                if prev_ema1 >= prev_ema2 and curr_ema1 < curr_ema2:
                    self.current_signal = -1
                elif prev_ema1 <= prev_ema2 and curr_ema1 > curr_ema2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.ema1_history = []
        self.ema2_history = []
        self.current_signal = 0



class Rule4:
    """
    Event-driven version of Rule4: DEMA and MA (formerly Rule4).
    """
    def __init__(self, param):
        self.dema1_period, self.ma2_period = param
        self.history = {'close': []}
        self.dema1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.dema1_period * 2 - 1: # DEMA needs 2*n - 1 initial values
            dema1_indicator = DEMA(closes, window=self.dema1_period)
            dema1_val = dema1_indicator.dema()[-1]
            self.dema1_history.append(dema1_val)
        else:
            self.dema1_history.append(np.nan)

        if len(closes) >= self.ma2_period:
            s2 = np.mean(closes[-self.ma2_period:])
            self.s2_history.append(s2)
        else:
            self.s2_history.append(np.nan)

        if len(self.dema1_history) >= 2 and len(self.s2_history) >= 2:
            prev_dema1 = self.dema1_history[-2]
            curr_dema1 = self.dema1_history[-1]
            prev_s2 = self.s2_history[-2]
            curr_s2 = self.s2_history[-1]

            if not np.isnan(prev_dema1) and not np.isnan(curr_dema1) and not np.isnan(prev_s2) and not np.isnan(curr_s2):
                if prev_dema1 >= prev_s2 and curr_dema1 < curr_s2:
                    self.current_signal = -1
                elif prev_dema1 <= prev_s2 and curr_dema1 > curr_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.dema1_history = []
        self.s2_history = []
        self.current_signal = 0



class Rule5:
    """
    Event-driven version of Rule5: DEMA and DEMA (formerly Rule5).
    """
    def __init__(self, param):
        self.dema1_period, self.dema2_period = param
        self.history = {'close': []}
        self.dema1_history = []
        self.dema2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.dema1_period * 2 - 1:
            dema1_indicator = DEMA(closes, window=self.dema1_period)
            dema1_val = dema1_indicator.dema()[-1]
            self.dema1_history.append(dema1_val)
        else:
            self.dema1_history.append(np.nan)

        if len(closes) >= self.dema2_period * 2 - 1:
            dema2_indicator = DEMA(closes, window=self.dema2_period)
            dema2_val = dema2_indicator.dema()[-1]
            self.dema2_history.append(dema2_val)
        else:
            self.dema2_history.append(np.nan)

        if len(self.dema1_history) >= 2 and len(self.dema2_history) >= 2:
            prev_dema1 = self.dema1_history[-2]
            curr_dema1 = self.dema1_history[-1]
            prev_dema2 = self.dema2_history[-2]
            curr_dema2 = self.dema2_history[-1]

            if not np.isnan(prev_dema1) and not np.isnan(curr_dema1) and not np.isnan(prev_dema2) and not np.isnan(curr_dema2):
                if prev_dema1 >= prev_dema2 and curr_dema1 < curr_dema2:
                    self.current_signal = -1
                elif prev_dema1 <= prev_dema2 and curr_dema1 > curr_dema2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.dema1_history = []
        self.dema2_history = []
        self.current_signal = 0



class Rule6:
    """
    Event-driven version of Rule6: TEMA and ma crossovers (formerly Rule6).
    """
    def __init__(self, param):
        self.tema1_period, self.ma2_period = param
        self.history = {'close': []}
        self.tema1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.tema1_period * 3 - 2: # TEMA needs 3*n - 2 initial values
            tema1_indicator = TEMA(closes, window=self.tema1_period)
            tema1_val = tema1_indicator.tema()[-1]
            self.tema1_history.append(tema1_val)
        else:
            self.tema1_history.append(np.nan)

        if len(closes) >= self.ma2_period:
            s2 = np.mean(closes[-self.ma2_period:])
            self.s2_history.append(s2)
        else:
            self.s2_history.append(np.nan)

        if len(self.tema1_history) >= 2 and len(self.s2_history) >= 2:
            prev_tema1 = self.tema1_history[-2]
            curr_tema1 = self.tema1_history[-1]
            prev_s2 = self.s2_history[-2]
            curr_s2 = self.s2_history[-1]

            if not np.isnan(prev_tema1) and not np.isnan(curr_tema1) and not np.isnan(prev_s2) and not np.isnan(curr_s2):
                if prev_tema1 >= prev_s2 and curr_tema1 < curr_s2:
                    self.current_signal = -1
                elif prev_tema1 <= prev_s2 and curr_tema1 > curr_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.tema1_history = []
        self.s2_history = []
        self.current_signal = 0



class Rule7:
    """
    Event-driven version of Rule7: Stochastic crossover (formerly Rule7).
    """
    def __init__(self, param):
        self.stoch1_period, self.stochma2_period = param
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['high'].append(bar['high'])
        self.history['low'].append(bar['low'])
        self.history['close'].append(bar['close'])
        highs = np.array(self.history['high'])
        lows = np.array(self.history['low'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.stoch1_period:
            stoch_indicator = stoch(highs, lows, closes, window=self.stoch1_period)
            s1 = stoch_indicator.stoch()[-1]
            self.s1_history.append(s1)
        else:
            self.s1_history.append(np.nan)

        if len(self.s1_history) >= self.stochma2_period:
            s2 = np.nanmean(self.s1_history[-self.stochma2_period:])
            self.s2_history.append(s2)
        else:
            self.s2_history.append(np.nan)

        if len(self.s1_history) >= 2 and len(self.s2_history) >= 2:
            prev_s1 = self.s1_history[-2]
            curr_s1 = self.s1_history[-1]
            prev_s2 = self.s2_history[-2]
            curr_s2 = self.s2_history[-1]

            if not np.isnan(prev_s1) and not np.isnan(curr_s1) and not np.isnan(prev_s2) and not np.isnan(curr_s2):
                if prev_s1 >= prev_s2 and curr_s1 < curr_s2:
                    self.current_signal = -1
                elif prev_s1 <= prev_s2 and curr_s1 > curr_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0


class Rule8:
    """
    Event-driven version of Rule8: Vortex indicator crossover (formerly Rule8).
    """
    def __init__(self, param):
        self.vortex1_period, self.vortex2_period = param
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['high'].append(bar['high'])
        self.history['low'].append(bar['low'])
        self.history['close'].append(bar['close'])
        highs = np.array(self.history['high'])
        lows = np.array(self.history['low'])
        closes = np.array(self.history['close'])

        if len(closes) >= max(self.vortex1_period, self.vortex2_period):
            s1 = vortex_indicator_pos(highs, lows, closes, window=self.vortex1_period)[-1]
            s2 = vortex_indicator_neg(highs, lows, closes, window=self.vortex2_period)[-1]
            self.s1_history.append(s1)
            self.s2_history.append(s2)

            if len(self.s1_history) >= 2 and len(self.s2_history) >= 2:
                prev_s1 = self.s1_history[-2]
                curr_s1 = self.s1_history[-1]
                prev_s2 = self.s2_history[-2]
                curr_s2 = self.s2_history[-1]

                if prev_s1 <= prev_s2 and curr_s1 > curr_s2:
                    self.current_signal = 1
                elif prev_s1 >= prev_s2 and curr_s1 < curr_s2:
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
        self.s1_history = []
        self.s2_history = []
        self.current_signal = 0



class Rule9:
    """
    Event-driven version of Rule9: Ichimoku cloud (formerly Rule9).
    """
    def __init__(self, param):
        self.p1, self.p2 = param
        self.n2 = round((self.p1 + self.p2) / 2)
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.s3_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['high'].append(bar['high'])
        self.history['low'].append(bar['low'])
        self.history['close'].append(bar['close'])
        highs = np.array(self.history['high'])
        lows = np.array(self.history['low'])
        closes = np.array(self.history['close'])

        if len(closes) >= max(self.p1, self.n2, self.p2):
            ichi = IchimokuIndicator(highs, lows, window1=self.p1, window2=self.n2, window3=self.p2)
            s1 = ichi.ichimoku_a()[-1]
            s2 = ichi.ichimoku_b()[-1]
            s3 = closes[-1]
            self.s1_history.append(s1)
            self.s2_history.append(s2)
            self.s3_history.append(s3)

            if len(self.s1_history) >= 1:
                current_s1 = self.s1_history[-1]
                current_s2 = self.s2_history[-1]
                current_s3 = self.s3_history[-1]

                if current_s3 > current_s1 and current_s3 > current_s2:
                    self.current_signal = -1
                elif current_s3 < current_s1 and current_s3 < current_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.s3_history = []
        self.current_signal = 0



class Rule10:
    """
    Event-driven version of Rule10: RSI threshold (formerly Rule10).
    """
    def __init__(self, param):
        self.rsi1_period, self.c2_threshold = param
        self.history = {'close': []}
        self.s1_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.rsi1_period:
            rsi_indicator = rsi(closes, window=self.rsi1_period)
            s1 = rsi_indicator.rsi()[-1]
            self.s1_history.append(s1)
        else:
            self.s1_history.append(np.nan)

        if len(self.s1_history) >= 1 and not np.isnan(self.s1_history[-1]):
            s1 = self.s1_history[-1]
            if s1 < self.c2_threshold:
                self.current_signal = 1
            elif s1 > (100 - self.c2_threshold):
                self.current_signal = -1
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.s1_history = []
        self.current_signal = 0



class Rule11:
    """
    Event-driven version of Rule11: CCI threshold (formerly Rule11).
    """
    def __init__(self, param):
        self.cci1_period, self.c2_threshold = param
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['high'].append(bar['high'])
        self.history['low'].append(bar['low'])
        self.history['close'].append(bar['close'])
        highs = np.array(self.history['high'])
        lows = np.array(self.history['low'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.cci1_period:
            cci_indicator = cci(highs, lows, closes, window=self.cci1_period)
            s1 = cci_indicator.cci()[-1]
            self.s1_history.append(s1)
        else:
            self.s1_history.append(np.nan)

        if len(self.s1_history) >= 1 and not np.isnan(self.s1_history[-1]):
            s1 = self.s1_history[-1]
            if s1 < -self.c2_threshold:
                self.current_signal = 1
            elif s1 > self.c2_threshold:
                self.current_signal = -1
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.current_signal = 0



class Rule12:
    """
    Event-driven version of Rule12: RSI range (formerly Rule12).
    """
    def __init__(self, param):
        self.rsi1_period, self.hl_threshold, self.ll_threshold = param
        self.history = {'close': []}
        self.s1_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.rsi1_period:
            rsi_indicator = rsi(closes, window=self.rsi1_period)
            s1 = rsi_indicator.rsi()[-1]
            self.s1_history.append(s1)
        else:
            self.s1_history.append(np.nan)

        if len(self.s1_history) >= 1 and not np.isnan(self.s1_history[-1]):
            s1 = self.s1_history[-1]
            if s1 > self.hl_threshold:
                self.current_signal = -1
            elif s1 < self.ll_threshold:
                self.current_signal = 1
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'close': []}
        self.s1_history = []
        self.current_signal = 0



class Rule13:
    """
    Event-driven version of Rule13: CCI range (formerly Rule13).
    """
    def __init__(self, param):
        self.cci1_period, self.hl_threshold, self.ll_threshold = param
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['high'].append(bar['high'])
        self.history['low'].append(bar['low'])
        self.history['close'].append(bar['close'])
        highs = np.array(self.history['high'])
        lows = np.array(self.history['low'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.cci1_period:
            cci_indicator = cci(highs, lows, closes, window=self.cci1_period)
            s1 = cci_indicator.cci()[-1]
            self.s1_history.append(s1)
        else:
            self.s1_history.append(np.nan)

        if len(self.s1_history) >= 1 and not np.isnan(self.s1_history[-1]):
            s1 = self.s1_history[-1]
            if s1 > self.hl_threshold:
                self.current_signal = -1
            elif s1 < self.ll_threshold:
                self.current_signal = 1
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.current_signal = 0



class Rule14:
    """
    Event-driven version of Rule14: Keltner channel (formerly Rule14).
    """
    def __init__(self, param):
        self.period = param
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.s3_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['high'].append(bar['high'])
        self.history['low'].append(bar['low'])
        self.history['close'].append(bar['close'])
        highs = np.array(self.history['high'])
        lows = np.array(self.history['low'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.period:
            kc = keltner(highs, lows, closes, window=self.period)
            s1 = kc.keltner_channel_hband()[-1]
            s2 = kc.keltner_channel_lband()[-1]
            s3 = closes[-1]
            self.s1_history.append(s1)
            self.s2_history.append(s2)
            self.s3_history.append(s3)

            if len(self.s1_history) >= 1:
                current_s1 = self.s1_history[-1]
                current_s2 = self.s2_history[-1]
                current_s3 = self.s3_history[-1]

                if current_s3 > current_s1:
                    self.current_signal = -1
                elif current_s3 < current_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
        return self.current_signal

    def reset(self):
        self.history = {'high': [], 'low': [], 'close': []}
        self.s1_history = []
        self.s2_history = []
        self.s3_history = []
        self.current_signal = 0



class Rule15:
    """
    Event-driven version of Rule15: Donchian channel (formerly Rule15).
    """
    def __init__(self, param):
        self.period = param
        self.history = {'close': []}
        self.s1_history = []
        self.s2_history = []
        self.s3_history = []
        self.current_signal = 0

    def on_bar(self, bar):
        self.history['close'].append(bar['close'])
        closes = np.array(self.history['close'])

        if len(closes) >= self.period:
            dc = donchian(closes, window=self.period)
            s1 = dc.donchian_channel_hband()[-1]
            s2 = dc.donchian_channel_lband()[-1]
            s3 = closes[-1]
            self.s1_history.append(s1)
            self.s2_history.append(s2)
            self.s3_history.append(s3)

            if len(self.s1_history) >= 1:
                current_s1 = self.s1_history[-1]
                current_s2 = self.s2_history[-1]
                current_s3 = self.s3_history[-1]

                if current_s3 > current_s1:
                    self.current_signal = -1
                elif current_s3 < current_s2:
                    self.current_signal = 1
                else:
                    self.current_signal = 0
            else:
                self.current_signal = 0
        else:
            self.current_signal = 0
            

class TopNStrategy:
    """
    Combines multiple event-driven rule objects into a single strategy.
    Emits 1, -1, or 0 based on signal consensus (simple average vote).
    """
    def __init__(self, rule_objects):
        self.rules = rule_objects
        self.signals = []


    
    def on_bar(self, event):
        bar = event.bar
        rule_signals = [rule.on_bar(bar) for rule in self.rules]
        combined = self.combine_signals(rule_signals)

        self.signals.append({
            "timestamp": bar["timestamp"],
            "signal": combined,
            "price": bar["Close"]
        })


    def combine_signals(self, signals):
        # Basic average-vote logic
        avg = sum(signals) / len(signals)
        if avg > 0.5:
            return 1
        elif avg < -0.5:
            return -1
        return 0

    def reset(self):
        for rule in self.rules:
            rule.reset()
        self.signals = []
