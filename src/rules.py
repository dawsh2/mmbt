import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional

def ensure_dataframe(func):
    """
    Decorator to ensure OHLC is a DataFrame before processing.
    """
    def wrapper(param, OHLC):
        # Convert OHLC to DataFrame if it's not already
        if not isinstance(OHLC, pd.DataFrame):
            if isinstance(OHLC, tuple) or isinstance(OHLC, list):
                if isinstance(OHLC[0], pd.Series):
                    # If it's a tuple/list of Series
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0],
                        'High': OHLC[1] if len(OHLC) > 1 else None,
                        'Low': OHLC[2] if len(OHLC) > 2 else None,
                        'Close': OHLC[3] if len(OHLC) > 3 else None
                    })
                else:
                    # If it's a tuple/list of lists/arrays
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0] if len(OHLC) > 0 else [],
                        'High': OHLC[1] if len(OHLC) > 1 else [],
                        'Low': OHLC[2] if len(OHLC) > 2 else [],
                        'Close': OHLC[3] if len(OHLC) > 3 else []
                    })
            else:
                raise TypeError(f"Expected DataFrame or list/tuple for OHLC, got {type(OHLC)}")
        return func(param, OHLC)
    return wrapper

@ensure_dataframe
def Rule0(param, OHLC):
    """
    Simple moving average crossover strategy
    """
    open_prices = OHLC['Open']
    close_prices = OHLC['Close']
    
    # Calculate fast and slow moving averages
    fast_window = int(param[0])
    slow_window = int(param[1])
    
    fast_ma = close_prices.rolling(window=fast_window).mean()
    slow_ma = close_prices.rolling(window=slow_window).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[fast_ma > slow_ma] = 1    # Buy signal
    signals[fast_ma < slow_ma] = -1   # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule1(param, OHLC):
    """
    RSI strategy
    """
    close_prices = OHLC['Close']
    
    # Calculate RSI
    window = int(param[0])
    oversold = param[1]
    overbought = param[2]
    
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[rsi < oversold] = 1      # Buy when oversold
    signals[rsi > overbought] = -1   # Sell when overbought
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule2(param, OHLC):
    """
    Bollinger Bands strategy
    """
    close_prices = OHLC['Close']
    
    # Calculate Bollinger Bands
    window = int(param[0])
    num_std = param[1]
    
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[close_prices < lower_band] = 1     # Buy signal
    signals[close_prices > upper_band] = -1    # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule3(param, OHLC):
    """
    MACD strategy
    """
    close_prices = OHLC['Close']
    
    # Calculate MACD
    fast_window = int(param[0])
    slow_window = int(param[1])
    signal_window = int(param[2])
    
    fast_ema = close_prices.ewm(span=fast_window, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_window, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[macd > signal_line] = 1     # Buy signal
    signals[macd < signal_line] = -1    # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule4(param, OHLC):
    """
    Price momentum strategy
    """
    close_prices = OHLC['Close']
    
    # Calculate momentum
    window = int(param[0])
    threshold = param[1]
    
    momentum = close_prices.pct_change(window)
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[momentum > threshold] = 1        # Buy on strong positive momentum
    signals[momentum < -threshold] = -1      # Sell on strong negative momentum
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule5(param, OHLC):
    """
    Volume-weighted price strategy
    """
    close_prices = OHLC['Close']
    volumes = OHLC['Volume']
    
    # Calculate volume-weighted moving average
    window = int(param[0])
    
    vwap = (close_prices * volumes).rolling(window=window).sum() / volumes.rolling(window=window).sum()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[close_prices > vwap] = 1     # Buy signal
    signals[close_prices < vwap] = -1    # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule6(param, OHLC):
    """
    Price breakout strategy
    """
    high_prices = OHLC['High']
    low_prices = OHLC['Low']
    
    # Calculate highest high and lowest low
    window = int(param[0])
    
    highest_high = high_prices.rolling(window=window).max()
    lowest_low = low_prices.rolling(window=window).min()
    
    # Generate signals
    signals = pd.Series(0, index=high_prices.index)
    signals[high_prices > highest_high.shift(1)] = 1    # Buy on breakout above
    signals[low_prices < lowest_low.shift(1)] = -1      # Sell on breakout below
    
    # Calculate returns for evaluation
    close_prices = OHLC['Close']
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule7(param, OHLC):
    """
    Mean reversion strategy
    """
    close_prices = OHLC['Close']
    
    # Calculate Z-score
    window = int(param[0])
    threshold = param[1]
    
    rolling_mean = close_prices.rolling(window=window).mean()
    rolling_std = close_prices.rolling(window=window).std()
    
    z_score = (close_prices - rolling_mean) / rolling_std
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[z_score < -threshold] = 1    # Buy when price is below mean
    signals[z_score > threshold] = -1    # Sell when price is above mean
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule8(param, OHLC):
    """
    Stochastic oscillator strategy
    """
    high_prices = OHLC['High']
    low_prices = OHLC['Low']
    close_prices = OHLC['Close']
    
    # Calculate stochastic oscillator
    k_window = int(param[0])
    d_window = int(param[1])
    oversold = param[2]
    overbought = param[3]
    
    lowest_low = low_prices.rolling(window=k_window).min()
    highest_high = high_prices.rolling(window=k_window).max()
    
    k_percent = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[(k_percent < oversold) & (k_percent > d_percent)] = 1     # Buy signal
    signals[(k_percent > overbought) & (k_percent < d_percent)] = -1  # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule9(param, OHLC):
    """
    Triple EMA crossover strategy
    """
    close_prices = OHLC['Close']
    
    # Calculate EMAs
    fast_window = int(param[0])
    mid_window = int(param[1])
    slow_window = int(param[2])
    
    fast_ema = close_prices.ewm(span=fast_window, adjust=False).mean()
    mid_ema = close_prices.ewm(span=mid_window, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_window, adjust=False).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[(fast_ema > mid_ema) & (mid_ema > slow_ema)] = 1      # Buy signal
    signals[(fast_ema < mid_ema) & (mid_ema < slow_ema)] = -1     # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule10(param, OHLC):
    """
    Relative volume strategy
    """
    close_prices = OHLC['Close']
    volumes = OHLC['Volume']
    
    # Calculate relative volume
    window = int(param[0])
    threshold = param[1]
    
    avg_volume = volumes.rolling(window=window).mean()
    rel_volume = volumes / avg_volume
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[(rel_volume > threshold) & (close_prices.pct_change() > 0)] = 1   # Buy on high volume up days
    signals[(rel_volume > threshold) & (close_prices.pct_change() < 0)] = -1  # Sell on high volume down days
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule11(param, OHLC):
    """
    Ichimoku Cloud strategy
    """
    high_prices = OHLC['High']
    low_prices = OHLC['Low']
    close_prices = OHLC['Close']
    
    # Calculate Ichimoku components
    tenkan_window = int(param[0])
    kijun_window = int(param[1])
    senkou_span_b_window = int(param[2])
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past tenkan_window periods
    tenkan_sen = (high_prices.rolling(window=tenkan_window).max() + 
                  low_prices.rolling(window=tenkan_window).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past kijun_window periods
    kijun_sen = (high_prices.rolling(window=kijun_window).max() + 
                 low_prices.rolling(window=kijun_window).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward by kijun_window periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past senkou_span_b_window periods,
    # shifted forward by kijun_window periods
    senkou_span_b = ((high_prices.rolling(window=senkou_span_b_window).max() + 
                      low_prices.rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_window)
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[(close_prices > senkou_span_a) & (close_prices > senkou_span_b) & 
            (tenkan_sen > kijun_sen)] = 1  # Buy signal
    signals[(close_prices < senkou_span_a) & (close_prices < senkou_span_b) & 
            (tenkan_sen < kijun_sen)] = -1  # Sell signal
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule12(param, OHLC):
    """
    Volatility breakout strategy
    """
    high_prices = OHLC['High']
    low_prices = OHLC['Low']
    close_prices = OHLC['Close']
    
    # Calculate ATR
    window = int(param[0])
    multiplier = param[1]
    
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift(1))
    tr3 = abs(low_prices - close_prices.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[close_prices > close_prices.shift(1) + (multiplier * atr.shift(1))] = 1  # Buy on volatility breakout up
    signals[close_prices < close_prices.shift(1) - (multiplier * atr.shift(1))] = -1  # Sell on volatility breakout down
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule13(param, OHLC):
    """
    ADX trend strength strategy
    """
    high_prices = OHLC['High']
    low_prices = OHLC['Low']
    close_prices = OHLC['Close']
    
    # Calculate ADX
    window = int(param[0])
    threshold = param[1]
    
    # Calculate True Range
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift(1))
    tr3 = abs(low_prices - close_prices.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    plus_dm = high_prices - high_prices.shift(1)
    minus_dm = low_prices.shift(1) - low_prices
    
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    # Calculate smoothed TR, +DM, and -DM
    smoothed_tr = tr.rolling(window=window).sum()
    smoothed_plus_dm = plus_dm.rolling(window=window).sum()
    smoothed_minus_dm = minus_dm.rolling(window=window).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate Directional Index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(window=window).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[(adx > threshold) & (plus_di > minus_di)] = 1   # Buy in strong uptrend
    signals[(adx > threshold) & (plus_di < minus_di)] = -1  # Sell in strong downtrend
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule14(param, OHLC):
    """
    Range breakout strategy
    """
    open_prices = OHLC['Open']
    high_prices = OHLC['High']
    low_prices = OHLC['Low']
    close_prices = OHLC['Close']
    
    # Calculate daily range
    range_window = int(param[0])
    range_factor = param[1]
    
    daily_range = (high_prices.rolling(window=range_window).max() - 
                  low_prices.rolling(window=range_window).min())
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[close_prices > close_prices.shift(1) + (range_factor * daily_range.shift(1))] = 1   # Buy on range breakout up
    signals[close_prices < close_prices.shift(1) - (range_factor * daily_range.shift(1))] = -1  # Sell on range breakout down
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

@ensure_dataframe
def Rule15(param, OHLC):
    """
    Volume confirmation strategy
    """
    close_prices = OHLC['Close']
    volumes = OHLC['Volume']
    
    # Calculate price and volume moving averages
    price_window = int(param[0])
    volume_window = int(param[1])
    
    price_ma = close_prices.rolling(window=price_window).mean()
    volume_ma = volumes.rolling(window=volume_window).mean()
    
    # Generate signals
    signals = pd.Series(0, index=close_prices.index)
    signals[(close_prices > price_ma) & (volumes > volume_ma)] = 1    # Buy when price and volume are above average
    signals[(close_prices < price_ma) & (volumes > volume_ma)] = -1   # Sell when price is below and volume above average
    
    # Calculate returns for evaluation
    log_returns = np.log(close_prices / close_prices.shift(1)).fillna(0)
    strategy_returns = signals.shift(1).fillna(0) * log_returns
    
    # Calculate Sharpe ratio
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    return sharpe, signals

class RuleSystem:
    """
    System for managing and evaluating trading rules
    """
    def __init__(self, top_n=5, use_weights=True):
        self.top_n = top_n
        self.use_weights = use_weights
        self.rules = [
            (Rule0, [(5, 50), (10, 100), (15, 150)]),  # Fast/slow windows
            (Rule1, [(14, 30, 70), (7, 25, 75), (21, 35, 65)]),  # RSI params
            (Rule2, [(20, 2.0), (10, 1.5), (30, 2.5)]),  # Bollinger Bands
            (Rule3, [(12, 26, 9), (8, 17, 9), (5, 35, 5)]),  # MACD
            (Rule4, [(10, 0.01), (20, 0.02), (5, 0.005)]),  # Momentum
            (Rule5, [(10,), (20,), (30,)]),  # VWAP
            (Rule6, [(10,), (20,), (5,)]),  # Breakout
            (Rule7, [(20, 1.0), (10, 1.5), (30, 0.8)]),  # Mean reversion
            (Rule8, [(14, 3, 20, 80), (10, 3, 30, 70), (21, 5, 25, 75)]),  # Stochastic
            (Rule9, [(5, 10, 20), (3, 10, 30), (7, 14, 28)]),  # Triple EMA
            (Rule10, [(10, 1.5), (20, 2.0), (5, 1.3)]),  # Relative volume
            (Rule11, [(9, 26, 52), (7, 22, 44), (11, 30, 60)]),  # Ichimoku
            (Rule12, [(14, 1.0), (7, 1.5), (21, 0.8)]),  # Volatility breakout
            (Rule13, [(14, 25), (10, 20), (21, 30)]),  # ADX
            (Rule14, [(10, 0.5), (20, 0.3), (5, 0.7)]),  # Range breakout
            (Rule15, [(10, 10), (20, 20), (5, 15)])  # Volume confirmation
        ]
        self.best_params = []
        self.best_scores = []
        self.best_indices = []

    def train_rules(self, OHLC):
        """
        Train all rules and find the best parameters
        """
        # Ensure OHLC is in the right format before passing to rules
        if not isinstance(OHLC, pd.DataFrame):
            if isinstance(OHLC, tuple) or isinstance(OHLC, list):
                # Convert to DataFrame if needed
                if isinstance(OHLC[0], pd.Series):
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0],
                        'High': OHLC[1] if len(OHLC) > 1 else None,
                        'Low': OHLC[2] if len(OHLC) > 2 else None,
                        'Close': OHLC[3] if len(OHLC) > 3 else None,
                        'Volume': OHLC[4] if len(OHLC) > 4 else None
                    })
                else:
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0] if len(OHLC) > 0 else [],
                        'High': OHLC[1] if len(OHLC) > 1 else [],
                        'Low': OHLC[2] if len(OHLC) > 2 else [],
                        'Close': OHLC[3] if len(OHLC) > 3 else [],
                        'Volume': OHLC[4] if len(OHLC) > 4 else []
                    })
        
        params = []
        scores = []
        indices = []
        
        # Evaluate each rule with each parameter set
        for i, (rule, param_sets) in enumerate(self.rules):
            best_score = -np.inf
            best_param = None
            
            for param in param_sets:
                try:
                    score, _ = rule(param, OHLC)
                    print(f"Training Rule{i} score is: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_param = param
                except Exception as e:
                    print(f"Error in Rule{i} with param {param}: {e}")
            
            if best_param is not None:
                params.append(best_param)
                scores.append(best_score)
                indices.append(i)
        
        # Sort by score and select top N
        if scores:
            sorted_indices = np.argsort(scores)[::-1][:self.top_n]
            
            self.best_params = [params[i] for i in sorted_indices]
            self.best_scores = [scores[i] for i in sorted_indices]
            self.best_indices = [indices[i] for i in sorted_indices]
        
        return self.best_params, self.best_scores, self.best_indices


    def load_params(self, filename):
        """
        Load previously saved parameters from a JSON file.

        Args:
            filename: Path to the parameters file
        """
        import json
        import os

        if not os.path.exists(filename):
            print(f"Warning: Parameters file {filename} not found.")
            return False

        try:
            with open(filename, 'r') as f:
                params_dict = json.load(f)

            # Extract parameters, indices, and scores
            if "params" in params_dict and "indices" in params_dict:
                self.best_params = params_dict["params"]
                self.best_indices = params_dict["indices"]
                self.best_scores = params_dict.get("scores", [1.0] * len(self.best_indices))
                print(f"Loaded parameters from {filename}")
                return True
            else:
                print(f"Error: Invalid parameter file format in {filename}")
                return False
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False


    # In rules.py in the RuleSystem class, update the generate_signals method:

    def generate_signals(self, OHLC, rule_params=None, filter_regime=False, weights=None):
        """
        Generate trading signals using the best rules and parameters.

        Args:
            OHLC: DataFrame or list/tuple of OHLC data
            rule_params: Optional parameters to override best_params (ignored if None)
            filter_regime: Whether to apply regime filtering (passed to strategy)
            weights: Optional weights to use for rule combination

        Returns:
            DataFrame containing signals and log returns
        """
        # If weights are provided, use them
        if weights is not None:
            self.weights = weights

        # Use provided parameters if available
        params_to_use = rule_params if rule_params else self.best_params
        indices_to_use = self.best_indices

        # Ensure OHLC is a DataFrame
        if not isinstance(OHLC, pd.DataFrame):
            if isinstance(OHLC, tuple) or isinstance(OHLC, list):
                if isinstance(OHLC[0], pd.Series):
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0],
                        'High': OHLC[1] if len(OHLC) > 1 else None,
                        'Low': OHLC[2] if len(OHLC) > 2 else None,
                        'Close': OHLC[3] if len(OHLC) > 3 else None,
                        'Volume': OHLC[4] if len(OHLC) > 4 else None
                    })
                else:
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0] if len(OHLC) > 0 else [],
                        'High': OHLC[1] if len(OHLC) > 1 else [],
                        'Low': OHLC[2] if len(OHLC) > 2 else [],
                        'Close': OHLC[3] if len(OHLC) > 3 else [],
                        'Volume': OHLC[4] if len(OHLC) > 4 else []
                    })

        print(f"DEBUG - Generating signals for {len(self.best_indices)} rules")

        all_signals = []

        # Generate signals for each rule
        for i, (param, idx) in enumerate(zip(params_to_use, indices_to_use)):
            rule_func = self.rules[idx][0]
            _, signals = rule_func(param, OHLC)

            # Count signal types for debugging
            buy_count = (signals == 1).sum()
            sell_count = (signals == -1).sum()
            neutral_count = (signals == 0).sum()
            print(f"DEBUG - Rule{idx} signals: Buy={buy_count}, Sell={sell_count}, Neutral={neutral_count}")

            # Print sample signals for debugging
            t = 100  # Arbitrary index for debugging
            if len(signals) > t:
                print(f"DEBUG - Rule{idx} at t={t}: {signals.iloc[t]}")
                if 'Close' in OHLC:
                    print(f"DEBUG - Price at t={t}: {OHLC['Close'].iloc[t]}")
                    if t+1 < len(signals) and t+1 < len(OHLC):
                        log_return = np.log(OHLC['Close'].iloc[t+1] / OHLC['Close'].iloc[t])
                        print(f"DEBUG - Signal applied to return at t={t+1}: {signals.iloc[t] * log_return}")

            all_signals.append(signals)

        # Combine signals
        if not all_signals:
            # No signals generated
            final_signals = pd.Series(0, index=OHLC.index)
        elif self.use_weights and all_signals:
            # Use weighted approach
            # Normalize weights based on scores if no explicit weights provided
            if self.weights is None:
                total_score = sum(self.best_scores)
                if total_score > 0:
                    self.weights = [score / total_score for score in self.best_scores]
                else:
                    self.weights = [1.0 / len(self.best_scores)] * len(self.best_scores)

            # Apply weights to signals
            signals_df = pd.DataFrame(all_signals).T

            # Check for and handle NaN values
            signals_df = signals_df.fillna(0)

            weighted_signals = signals_df.multiply(self.weights, axis=1)
            combined_signals = weighted_signals.sum(axis=1)

            # Convert to -1/0/1 based on threshold
            threshold = 0.2
            final_signals = pd.Series(0, index=combined_signals.index)
            final_signals[combined_signals > threshold] = 1
            final_signals[combined_signals < -threshold] = -1
        else:
            # Simple majority vote
            signals_df = pd.DataFrame(all_signals).T

            # Handle NaN values
            signals_df = signals_df.fillna(0)
            print(f"DEBUG - After dropping NaNs, signals_df has {len(signals_df)} rows")

            buy_votes = (signals_df == 1).sum(axis=1)
            sell_votes = (signals_df == -1).sum(axis=1)

            final_signals = pd.Series(0, index=signals_df.index)
            final_signals[buy_votes > sell_votes] = 1
            final_signals[sell_votes > buy_votes] = -1

        # Debug signal counts
        buy_count = (final_signals == 1).sum()
        sell_count = (final_signals == -1).sum()
        neutral_count = (final_signals == 0).sum()
        print(f"DEBUG - Combined signals: Buy={buy_count}, Sell={sell_count}, Neutral={neutral_count}")

        # Calculate log returns if not in OHLC
        if 'LogReturn' in OHLC.columns:
            log_returns = OHLC['LogReturn']
        else:
            log_returns = np.log(OHLC['Close'] / OHLC['Close'].shift(1)).fillna(0)

        # Create DataFrame with signals and returns
        result_df = pd.DataFrame({
            'Signal': final_signals,
            'LogReturn': log_returns
        })

        return result_df
    

    def save_params(self, filename=None):
        """
        Save the best parameters to a JSON file.

        Args:
            filename: Path to save the parameters
        """
        import json

        if not self.best_params or not self.best_indices:
            print("Warning: No parameters to save. Train rules first.")
            return False

        if filename is None:
            filename = "best_params.json"

        # Create a dictionary to store params and indices
        params_dict = {
            "params": [],
            "indices": self.best_indices,
            "scores": self.best_scores
        }

        # Convert parameters to JSON-serializable format
        for param in self.best_params:
            if isinstance(param, (list, tuple)):
                params_dict["params"].append(list(param))
            else:
                params_dict["params"].append(param)

        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(params_dict, f, indent=4)
            print(f"Parameters saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False


    @property
    def rule_params(self):
        """
        Property to maintain backward compatibility with old codebase.
        Returns a dictionary mapping rule indices to their parameters.
        """
        if not self.best_params or not self.best_indices:
            return {}

        # Create a dictionary mapping rule indices to parameters
        params_dict = {}
        for i, (param, idx) in enumerate(zip(self.best_params, self.best_indices)):
            params_dict[idx] = param

        return params_dict
