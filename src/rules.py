"""
Trading rules implementation for the backtesting engine.
Based on the rules provided in the paste-3.txt file.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

# Import TA functions from provided code
from ta_functions import (
    ema, ma, DEMA, TEMA, rsi, stoch, stoch_signal, average_true_range,
    vortex_indicator_pos, vortex_indicator_neg, cci, bollinger_mavg,
    bollinger_hband, bollinger_lband, keltner_channel_hband, keltner_channel_lband,
    donchian_channel_hband, donchian_channel_lband, ichimoku_a, ichimoku_b
)


def extract_ohlc(df):
    return df["Open"], df["High"], df["Low"], df["Close"]

def count_trade_no(signal):
    """Count the number of trades from signal changes."""
    no_trades = 0
    prev = 10
    for i in signal:
        if i != prev and prev != 0:
            prev = i
            no_trades += 1
    return no_trades



class TradingRules:
    """Implementation of trading rules from the original code."""
    
    def __init__(self):
        """Initialize TradingRules class."""
        self.rule_functions = [
            self.Rule0, self.Rule1, self.Rule2, self.Rule3, self.Rule4,
            self.Rule5, self.Rule6, self.Rule7, self.Rule8, self.Rule9,
            self.Rule10, self.Rule11, self.Rule12, self.Rule13, self.Rule14,
            self.Rule15
        ]
        self.rule_params = None
        self.periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
    
    def Rule0(self, param, OHLC):
        """Rule 0: Simple Moving Average Crossover (formerly Rule1)"""
        ma1, ma2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = close.rolling(ma1).mean()
        s2 = close.rolling(ma2).mean()
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule1(self, param, OHLC):
        """Rule 1: EMA and close (formerly Rule2)"""
        ema1, ma2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = ema(close, ema1)
        s2 = close.rolling(ma2).mean()
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule2(self, param, OHLC):
        """Rule 2: EMA and EMA (formerly Rule3)"""
        ema1, ema2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = ema(close, ema1)
        s2 = ema(close, ema2)
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule3(self, param, OHLC):
        """Rule 3: DEMA and MA (formerly Rule4)"""
        dema1, ma2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = DEMA(close, dema1)
        s2 = close.rolling(ma2).mean()
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule4(self, param, OHLC):
        """Rule 4: DEMA and DEMA (formerly Rule5)"""
        dema1, dema2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = DEMA(close, dema1)
        s2 = DEMA(close, dema2)
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule5(self, param, OHLC):
        """Rule 5: TEMA and ma crossovers (formerly Rule6)"""
        tema1, ma2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = TEMA(close, tema1)
        s2 = close.rolling(ma2).mean()
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule6(self, param, OHLC):
        """Rule 6: Stochastic crossover (formerly Rule7)"""
        stoch1, stochma2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = stoch(high, low, close, stoch1)
        s2 = s1.rolling(stochma2, min_periods=0).mean()
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule7(self, param, OHLC):
        """Rule 7: Vortex indicator crossover (formerly Rule8)"""
        vortex1, vortex2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = vortex_indicator_pos(high, low, close, vortex1)
        s2 = vortex_indicator_neg(high, low, close, vortex2)
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule8(self, param, OHLC):
        """Rule 8: Ichimoku cloud (formerly Rule9)"""
        p1, p2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = ichimoku_a(high, low, n1=p1, n2=round((p1+p2)/2))
        s2 = ichimoku_b(high, low, n2=round((p1+p2)/2), n3=p2)
        s3 = close
        # Remove shift here - will be applied centrally
        signal = (-1*((s3>s1) & (s3>s2))+1*((s3<s2) & (s3<s1)))
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule9(self, param, OHLC):
        """Rule 9: RSI threshold (formerly Rule10)"""
        rsi1, c2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = rsi(close, rsi1)
        s2 = c2
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule10(self, param, OHLC):
        """Rule 10: CCI threshold (formerly Rule11)"""
        cci1, c2 = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = cci(high, low, close, cci1)
        s2 = c2
        # Remove shift here - will be applied centrally
        signal = 2*(s1<s2)-1
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule11(self, param, OHLC):
        """Rule 11: RSI range (formerly Rule12)"""
        rsi1, hl, ll = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = rsi(close, rsi1)
        # Remove shift here - will be applied centrally
        signal = (-1*(s1>hl)+1*(s1<ll))
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule12(self, param, OHLC):
        """Rule 12: CCI range (formerly Rule13)"""
        cci1, hl, ll = param
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = cci(high, low, close, cci1)
        # Remove shift here - will be applied centrally
        signal = (-1*(s1>hl)+1*(s1<ll))
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule13(self, period, OHLC):
        """Rule 13: Keltner channel (formerly Rule14)"""
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = keltner_channel_hband(high, low, close, n=period)
        s2 = keltner_channel_lband(high, low, close, n=period)
        s3 = close
        # Remove shift here - will be applied centrally
        signal = (-1*(s3>s1)+1*(s3<s2))
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule14(self, period, OHLC):
        """Rule 14: Donchian channel (formerly Rule15)"""
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = donchian_channel_hband(close, n=period)
        s2 = donchian_channel_lband(close, n=period)
        s3 = close
        # Remove shift here - will be applied centrally
        signal = (-1*(s3>s1)+1*(s3<s2))
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def Rule15(self, period, OHLC):
        """Rule 15: Bollinger bands (formerly Rule16)"""
        open_prices = OHLC['Open']
        high = OHLC['High']
        low = OHLC['Low']
        close = OHLC['Close']
        logr = np.log(close/close.shift(1))
        s1 = bollinger_hband(close, n=period)
        s2 = bollinger_lband(close, n=period)
        s3 = close
        # Remove shift here - will be applied centrally
        signal = (-1*(s3>s1)+1*(s3<s2))
        port_logr = signal*logr
        return (abs(port_logr.sum()), signal)

    def train_rules(self, OHLC):
        """Train trading rule parameters using the provided OHLC data."""
        periods = self.periods
        
        # Train Type 1 rules (rules 0-8)
        type1 = self.rule_functions[:9]
        type1_param = []
        type1_score = []
        
        for rule in type1:
            best = -1
            best_param = None
            for i in range(len(periods)):
                for j in range(i, len(periods)):
                    param = (periods[i], periods[j])
                    score = rule(param, OHLC)[0]
                    if score > best:
                        best = score
                        best_param = param
            type1_param.append(best_param)
            type1_score.append(best)
        
        # Train Type 2 rules (rules 9-10)
        rsi_limits = list(range(0, 101, 5))
        cci_limits = list(range(-120, 121, 20))
        limits = [rsi_limits, cci_limits]
        
        type2 = self.rule_functions[9:11]
        type2_param = []
        type2_score = []
        
        for i in range(len(type2)):
            rule = type2[i]
            params = limits[i]
            best = -1
            best_param = None
            for period in periods:
                for p in params:
                    param = (period, p)
                    score = rule(param, OHLC)[0]
                    if score > best:
                        best = score
                        best_param = param
            type2_param.append(best_param)
            type2_score.append(best)
        
        # Train Type 3 rules (rules 11-12)
        type3 = self.rule_functions[11:13]
        type3_param = []
        type3_score = []
        
        for i in range(len(type3)):
            rule = type3[i]
            params = limits[i]
            n = len(params)
            best = -1
            best_param = None
            for period in periods:
                for lb in range(n-1):
                    for ub in range(lb+1, n):
                        param = (period, params[ub], params[lb])
                        score = rule(param, OHLC)[0]
                        if score > best:
                            best = score
                            best_param = param
            type3_param.append(best_param)
            type3_score.append(best)
        
        # Train Type 4 rules (rules 13-15)
        type4 = self.rule_functions[13:16]
        type4_param = []
        type4_score = []
        
        for rule in type4:
            best = -1
            best_param = None
            for period in periods:
                score = rule(period, OHLC)[0]
                if score > best:
                    best = score
                    best_param = period
            type4_param.append(best_param)
            type4_score.append(best)
        
        # Combine all parameters and scores
        all_params = type1_param + type2_param + type3_param + type4_param
        all_scores = type1_score + type2_score + type3_score + type4_score
        
        # Print training results - use 0-based indexing in output
        for i in range(len(all_params)):
            print(f'Training Rule{i} score is: {all_scores[i]:.3f}')
        
        # Sort rules by score
        rule_indices = np.argsort(all_scores)[::-1]
        
        self.rule_params = all_params
        self.rule_scores = all_scores
        self.rule_indices = rule_indices
        
        return all_params, all_scores, rule_indices


    def generate_signals(self, OHLC, params=None, top_n=None, weights=None):
        """Generate trading signals for all rules and combine them."""
        if params is None:
            params = self.rule_params

        if params is None:
            raise ValueError("Rule parameters not set. Train rules first.")

        # Extract log returns for performance calculation
        open_prices = OHLC["Open"]
        high = OHLC["High"]
        low = OHLC["Low"]
        close = OHLC["Close"]

        logr = np.log(close/close.shift(1))

        # Create DataFrame to store signals
        signals_df = pd.DataFrame(index=close.index)
        signals_df['LogReturn'] = logr

        # Debug info
        print(f"DEBUG - Generating signals for {len(self.rule_functions)} rules")

        # Generate signals for each rule
        for i, rule in enumerate(self.rule_functions):
            try:
                _, signal = rule(params[i], OHLC)
                signals_df[f'Rule{i}'] = signal  # Use 0-based indexing for column names

                # Debug info - count signal types
                if isinstance(signal, pd.Series):
                    buy_signals = (signal == 1).sum()
                    sell_signals = (signal == -1).sum()
                    neutral_signals = (signal == 0).sum()
                    print(f"DEBUG - Rule{i} signals: Buy={buy_signals}, Sell={sell_signals}, Neutral={neutral_signals}")
            except Exception as e:
                print(f"Error generating signals for Rule{i}: {str(e)}")
                signals_df[f'Rule{i}'] = np.nan

        # Apply a single shift to all rule signals - centralized signal timing
        rule_cols = [f'Rule{i}' for i in range(len(self.rule_functions))]
        for col in rule_cols:
            signals_df[col] = signals_df[col].shift(1)
        
        # Debug signal timing
        if len(signals_df) > 100:
            print(f"DEBUG - Rule0 at t=100: {signals_df.iloc[100]['Rule0'] if 'Rule0' in signals_df else 'N/A'}")
            print(f"DEBUG - Price at t=100: {close.iloc[100]}")
            print(f"DEBUG - Signal applied to return at t=101: {logr.iloc[101] if len(logr) > 101 else 'N/A'}")

        # Remove NaN values
        signals_df.dropna(inplace=True)
        print(f"DEBUG - After dropping NaNs, signals_df has {len(signals_df)} rows")

        # Generate combined signal
        if weights is not None:
            # Weighted combination of signals
            weighted_signal = pd.Series(0, index=signals_df.index)
            
            for i, col in enumerate(rule_cols):
                if i < len(weights):  # Ensure we don't exceed weights length
                    weighted_signal += weights[i] * signals_df[col]

            # Convert to -1, 0, 1 signals
            signals_df['Signal'] = np.sign(weighted_signal)

        elif top_n is not None:
            # Use only top N rules
            if not hasattr(self, 'rule_indices'):
                raise ValueError("Rule scores not calculated. Train rules first.")

            # Get the indices of the top N performing rules
            top_indices = self.rule_indices[:top_n]
            
            # Create column names for top rules (using 0-based indexing)
            top_rule_cols = [f'Rule{i}' for i in top_indices]
            
            print(f"DEBUG - Using top {top_n} rules: {top_indices}")

            # Equal-weighted combination of top N rules
            signals_df['Signal'] = signals_df[top_rule_cols].sum(axis=1)
            signals_df['Signal'] = np.sign(signals_df['Signal'])

        else:
            # Equal-weighted combination of all rules
            signals_df['Signal'] = signals_df[rule_cols].sum(axis=1)
            signals_df['Signal'] = np.sign(signals_df['Signal'])

        # Debug the final signal distribution
        if 'Signal' in signals_df:
            buy_signals = (signals_df['Signal'] == 1).sum()
            sell_signals = (signals_df['Signal'] == -1).sum()
            neutral_signals = (signals_df['Signal'] == 0).sum()
            print(f"DEBUG - Combined signals: Buy={buy_signals}, Sell={sell_signals}, Neutral={neutral_signals}")

        return signals_df

    def save_params(self, file_path):
        """Save trained rule parameters to a file."""
        if self.rule_params is None:
            print("No parameters to save. Train rules first.")
            return False
        
        params_dict = {
            'rule_params': self.rule_params,
            'rule_scores': self.rule_scores if hasattr(self, 'rule_scores') else None,
            'rule_indices': self.rule_indices.tolist() if hasattr(self, 'rule_indices') else None
        }
        
        try:
            import json
            with open(file_path, 'w') as f:
                json.dump(params_dict, f)
            print(f"Saved rule parameters to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving parameters: {str(e)}")
            return False
    
    def load_params(self, file_path):
        """Load trained rule parameters from a file."""
        try:
            import json
            with open(file_path, 'r') as f:
                params_dict = json.load(f)
            
            self.rule_params = params_dict['rule_params']
            
            if 'rule_scores' in params_dict and params_dict['rule_scores'] is not None:
                self.rule_scores = np.array(params_dict['rule_scores'])
            
            if 'rule_indices' in params_dict and params_dict['rule_indices'] is not None:
                self.rule_indices = np.array(params_dict['rule_indices'])
            
            print(f"Loaded rule parameters from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
            return False

