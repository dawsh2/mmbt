import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional


class TradingRules:
    """
    Class containing various trading rule functions.
    """
    periods = [1, 3, 5]  # Example periods

    @staticmethod
    def ensure_dataframe(func):
        """
        Decorator to ensure OHLC is a DataFrame before processing (for instance methods).
        """
        def wrapper(*args, **kwargs):
            self_arg = args[0] if args else None # Capture 'self'
            param = args[1] if len(args) > 1 else kwargs.get('params') or kwargs.get('period')
            OHLC = args[2] if len(args) > 2 else kwargs.get('OHLC')

            if OHLC is None:
                raise ValueError("OHLC data not found in arguments")

            if not isinstance(OHLC, pd.DataFrame):
                if isinstance(OHLC, (tuple, list)):
                    if isinstance(OHLC[0], pd.Series):
                        OHLC = pd.DataFrame({
                            'Open': OHLC[0],
                            'High': OHLC[1] if len(OHLC) > 1 else None,
                            'Low': OHLC[2] if len(OHLC) > 2 else None,
                            'Close': OHLC[3] if len(OHLC) > 3 else None
                        })
                    else:
                        OHLC = pd.DataFrame({
                            'Open': OHLC[0] if len(OHLC) > 0 else [],
                            'High': OHLC[1] if len(OHLC) > 1 else [],
                            'Low': OHLC[2] if len(OHLC) > 2 else [],
                            'Close': OHLC[3] if len(OHLC) > 3 else []
                        })
                else:
                    raise TypeError(f"Expected DataFrame or list/tuple for OHLC, got {type(OHLC)}")

            return func(self_arg, param, OHLC) # Pass 'self' back
        return wrapper


    @ensure_dataframe
    def Rule0(self, params, OHLC):
        fast_window, slow_window = params
        ma_fast = OHLC['Close'].rolling(window=fast_window).mean()
        ma_slow = OHLC['Close'].rolling(window=slow_window).mean()
        signal = np.where(ma_fast > ma_slow, 1, -1)
        return signal, params

    @ensure_dataframe
    def Rule1(self, params, OHLC):
        period, lower_bound, upper_bound = params
        rsi_val = ta_functions.rsi(OHLC['Close'], period)
        signal = np.where(rsi_val < lower_bound, 1, np.where(rsi_val > upper_bound, -1, 0))
        return signal, params

    @ensure_dataframe
    def Rule2(self, params, OHLC):
        period = params
        sma = ta_functions.sma(OHLC['Close'], period)
        signal = np.where(OHLC['Close'] > sma, 1, -1)
        return signal, period # Or return signal, params

    @ensure_dataframe
    def Rule3(self, params, OHLC):
        period = params
        ema = ta_functions.ema(OHLC['Close'], period)
        signal = np.where(OHLC['Close'] > ema, 1, -1)
        return signal, params

    @ensure_dataframe
    def Rule4(self, params, OHLC):
        period = params
        momentum = ta_functions.momentum(OHLC['Close'], period)
        signal = np.where(momentum > 0, 1, -1)
        return signal, params

    @ensure_dataframe
    def Rule5(self, params, OHLC):
        period = params
        atr = ta_functions.atr(OHLC, period)
        # Example rule based on ATR - you'll need to define the logic
        signal = np.where(OHLC['High'] - OHLC['Low'] > atr, 1, 0)
        return signal, params

    @ensure_dataframe
    def Rule6(self, params, OHLC):
        period = params
        adx = ta_functions.adx(OHLC, period)
        signal = np.where(adx > 25, 1, 0) # Example threshold
        return signal, params

    @ensure_dataframe
    def Rule7(self, params, OHLC):
        period = params
        cci = ta_functions.cci(OHLC, period)
        signal = np.where(cci > 100, 1, np.where(cci < -100, -1, 0))
        return signal, params

    @ensure_dataframe
    def Rule8(self, params, OHLC):
        period = params
        willr = ta_functions.willr(OHLC, period)
        signal = np.where(willr < -80, 1, np.where(willr > -20, -1, 0))
        return signal, params

    @ensure_dataframe
    def Rule9(self, params, OHLC):
        period = params
        roc = ta_functions.roc(OHLC['Close'], period)
        signal = np.where(roc > 0, 1, -1)
        return signal, params

    @ensure_dataframe
    def Rule10(self, params, OHLC):
        period = params
        obv = ta_functions.obv(OHLC['Close'], OHLC['Volume'])
        signal = np.where(obv > obv.shift(1), 1, -1)
        return signal, params

    @ensure_dataframe
    def Rule11(self, params, OHLC):
        short_period, mid_period, long_period = params
        # Example triple moving average crossover
        sma_short = ta_functions.sma(OHLC['Close'], short_period)
        sma_mid = ta_functions.sma(OHLC['Close'], mid_period)
        sma_long = ta_functions.sma(OHLC['Close'], long_period)
        signal = np.where((sma_short > sma_mid) & (sma_mid > sma_long), 1,
                          np.where((sma_short < sma_mid) & (sma_mid < sma_long), -1, 0))
        return signal, params

    @ensure_dataframe
    def Rule12(self, params, OHLC):
        short_period, mid_period, long_period = params
        ema_short = ta_functions.ema(OHLC['Close'], short_period)
        ema_mid = ta_functions.ema(OHLC['Close'], mid_period)
        ema_long = ta_functions.ema(OHLC['Close'], long_period)
        signal = np.where((ema_short > ema_mid) & (ema_mid > ema_long), 1,
                          np.where((ema_short < ema_mid) & (ema_mid < ema_long), -1, 0))
        return signal, params

    @ensure_dataframe
    def Rule13(self, params, OHLC):
        period = params
        # Example Bollinger Bands - you'll need a ta_functions implementation
        bb_upper, bb_mid, bb_lower = ta_functions.bollinger_bands(OHLC['Close'], period)
        signal = np.where(OHLC['Close'] > bb_upper, -1, np.where(OHLC['Close'] < bb_lower, 1, 0))
        return signal, params

    @ensure_dataframe
    def Rule14(self, params, OHLC):
        period = params
        # Example Stochastic Oscillator - you'll need a ta_functions implementation
        slowk, slowd = ta_functions.stochastic_oscillator(OHLC, period)
        signal = np.where(slowk > 80, -1, np.where(slowk < 20, 1, 0))
        return signal, params

    @ensure_dataframe
    def Rule15(self, params, OHLC):
        # Example Ichimoku Cloud - you'll need a ta_functions implementation
        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ta_functions.ichimoku_cloud(OHLC)
        signal = np.where(tenkan_sen > kijun_sen, 1, np.where(tenkan_sen < kijun_sen, -1, 0))
        return signal, params


class RuleSystem:
    """
    System for managing and evaluating trading rules
    """
    def __init__(self, top_n=5, use_weights=True):
        self.top_n = top_n
        self.use_weights = use_weights
        self.tr = TradingRules()  # Instantiate TradingRules
        self.rules = [
            ("Rule0", [(5, 50), (10, 100), (15, 150)]),
            ("Rule1", [(14, 30, 70), (7, 25, 75), (21, 35, 65)]),
            ("Rule2", self.tr.periods[:2]),
            ("Rule3", self.tr.periods[:2]),
            ("Rule4", self.tr.periods[:2]),
            ("Rule5", self.tr.periods[:2]),
            ("Rule6", self.tr.periods[:2]),
            ("Rule7", self.tr.periods[:2]),
            ("Rule8", self.tr.periods[:2]),
            ("Rule9", self.tr.periods[:2]),
            ("Rule10", self.tr.periods[:2]),
            ("Rule11", self.tr.periods[:3]),
            ("Rule12", self.tr.periods[:3]),
            ("Rule13", self.tr.periods[:1]),
            ("Rule14", self.tr.periods[:1]),
            ("Rule15", self.tr.periods[:1])
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
            if isinstance(OHLC, (tuple, list)):
                if isinstance(OHLC[0], pd.Series):
                    OHLC = pd.DataFrame({'Open': OHLC[0], 'High': OHLC[1] if len(OHLC) > 1 else None, 'Low': OHLC[2] if len(OHLC) > 2 else None, 'Close': OHLC[3] if len(OHLC) > 3 else None})
                else:
                    OHLC = pd.DataFrame({'Open': OHLC[0] if len(OHLC) > 0 else [], 'High': OHLC[1] if len(OHLC) > 1 else [], 'Low': OHLC[2] if len(OHLC) > 2 else [], 'Close': OHLC[3] if len(OHLC) > 3 else []})
            else:
                raise TypeError(f"Expected DataFrame or list/tuple for OHLC, got {type(OHLC)}")

        params = []
        scores = []
        indices = []

        # Evaluate each rule with each parameter set
        for i, (rule_name, param_sets) in enumerate(self.rules):
            best_score = -np.inf
            best_param = None
            original_rule_func = getattr(self.tr, rule_name)
            decorated_rule_func = self.tr.ensure_dataframe(original_rule_func) # Apply decorator here

            for param in param_sets:
                try:
                    score = decorated_rule_func(self.tr, param, OHLC)[0] # Pass self explicitly
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
        sorted_indices = np.argsort(scores)[::-1][:self.top_n]

        self.best_params = [params[i] for i in sorted_indices]
        self.best_scores = [scores[i] for i in sorted_indices]
        self.best_indices = [indices[i] for i in sorted_indices]

        return self.best_params, self.best_scores, self.best_indices

    def generate_signals(self, OHLC):
        """
        Generate trading signals using the best rules and parameters
        """
        if not self.best_params:
            raise ValueError("Rules have not been trained yet. Call train_rules first.")

        # Ensure OHLC is in the right format
        if not isinstance(OHLC, pd.DataFrame):
            if isinstance(OHLC, tuple) or isinstance(OHLC, list):
                if isinstance(OHLC[0], pd.Series):
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0],
                        'High': OHLC[1] if len(OHLC) > 1 else None,
                        'Low': OHLC[2] if len(OHLC) > 2 else None,
                        'Close': OHLC[3] if len(OHLC) > 3 else None
                    })
                else:
                    OHLC = pd.DataFrame({
                        'Open': OHLC[0] if len(OHLC) > 0 else [],
                        'High': OHLC[1] if len(OHLC) > 1 else [],
                        'Low': OHLC[2] if len(OHLC) > 2 else [],
                        'Close': OHLC[3] if len(OHLC) > 3 else []
                    })

        print(f"DEBUG - Generating signals for {len(self.best_indices)} rules")

        all_signals = []
        for i, (param, idx) in enumerate(zip(self.best_params, self.best_indices)):
            rule_name = self.rules[idx][0]
            rule_func = getattr(self.tr, rule_name)
            _, signals = rule_func(self.tr, param, OHLC) # Pass self explicitly

            # Count signal types for debugging
            buy_count = (signals == 1).sum()
            sell_count = (signals == -1).sum()
            neutral_count = (signals == 0).sum()
            print(f"DEBUG - Rule{idx} signals: Buy={buy_count}, Sell={sell_count}, Neutral={neutral_count}")

            all_signals.append(signals)

        # Combine signals using weights if enabled
        if self.use_weights and all_signals:
            # Normalize weights based on scores
            total_score = sum(self.best_scores)
            weights = [score / total_score for score in self.best_scores]

            # Apply weights to signals
            weighted_signals = pd.DataFrame(all_signals).T.multiply(weights, axis=1)
            combined_signals = weighted_signals.sum(axis=1)

            # Convert to -1/0/1 based on threshold
            threshold = 0.2
            final_signals = pd.Series(0, index=combined_signals.index)
            final_signals[combined_signals > threshold] = 1
            final_signals[combined_signals < -threshold] = -1
        else:
            # Simple majority vote
            signals_df = pd.DataFrame(all_signals).T
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

        return final_signals

    def save_params(self, file_path):
        """Save trained rule parameters to a file."""
        if not hasattr(self, 'best_params') or self.best_params is None:
            print("No parameters to save. Train rules first.")
            return False

        params_dict = {
            'best_params': self.best_params,
            'best_scores': self.best_scores if hasattr(self, 'best_scores') else None,
            'best_indices': self
