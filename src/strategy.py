"""
Strategy module for the backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from rules import RuleSystem  # Only import RuleSystem, removed TradingRules

class Strategy:
    """Base strategy class for the backtesting engine."""

    def __init__(self, config=None):
        """Initialize the strategy with configuration."""
        self.config = config
        self.rules = RuleSystem()
        self.signals = None
        self.weights = None
    
    def __str__(self):
        """Return a string representation of the strategy."""
        return f"Base Strategy"


class TopNStrategy(Strategy):
    """Strategy that uses the top N performing rules."""
    
    def __init__(self, config=None, top_n=5):
        """Initialize the TopNStrategy with configuration."""
        super().__init__(config)
        self.top_n = top_n if config is None else config.top_n
        # Update RuleSystem initialization with top_n
        self.rules = RuleSystem(top_n=self.top_n, use_weights=False)
    
    def train(self, OHLC):
        """Train the strategy by finding the best parameters for each rule."""
        params, scores, indices = self.rules.train_rules(OHLC)
        self.top_indices = indices[:self.top_n]
        return params, scores, indices

    # For TopNStrategy
    def generate_signals(self, OHLC, rule_params=None, filter_regime=False):
        """Generate trading signals using the top N rules."""
        # Call to rules.generate_signals updated to handle legacy API
        final_signal = self.rules.generate_signals(OHLC, rule_params, filter_regime)

        if isinstance(final_signal, pd.Series):
            # Apply regime filter if needed
            if filter_regime:
                final_signal = self._apply_regime_filter(final_signal, OHLC)

            # Create a proper DataFrame for backtester compatibility
            if 'LogReturn' in OHLC.columns:
                log_returns = OHLC['LogReturn']
            else:
                log_returns = np.log(OHLC['Close'] / OHLC['Close'].shift(1)).fillna(0)

            result_df = pd.DataFrame({
                'Signal': final_signal,
                'LogReturn': log_returns
            })
            self.signals = final_signal
            return result_df
        else:
            # It's already a DataFrame with the right format
            self.signals = final_signal['Signal']
            return final_signal


    def _apply_regime_filter(self, signal_series, OHLC):
        volatility = OHLC["Close"].pct_change().rolling(20).std() * np.sqrt(252)
        avg_vol = volatility.mean()
        signal_series[volatility < avg_vol * 0.5] = 0
        return signal_series
    
    def __str__(self):
        """Return a string representation of the strategy."""
        return f"Top-{self.top_n} Rules Strategy"

class WeightedStrategy(Strategy):
    """Strategy that uses weighted combination of rules optimized by GA."""
    
    def __init__(self, config=None):
        """Initialize the WeightedStrategy with configuration."""
        super().__init__(config)
        self.weights = None
        # Set use_weights to True in RuleSystem
        self.rules = RuleSystem(top_n=config.top_n if config else 5, use_weights=True)
    
    def train(self, OHLC, ga_module=None):
        """Train the strategy by finding the best parameters and weights."""
        # First train the rules to get the parameters
        params, scores, _ = self.rules.train_rules(OHLC)
        
        # Then optimize weights using GA
        if ga_module is not None:
            self.weights = self._optimize_weights(OHLC, params, ga_module)
            # Pass weights to the RuleSystem
            self.rules.weights = self.weights
        
        return params, scores, self.weights
    
    def _optimize_weights(self, OHLC, params, ga_module):
        """Optimize rule weights using genetic algorithm."""
        # Ensure OHLC is a DataFrame
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
        
        # Generate signals for each rule
        all_signals = []
        for i, (param, idx) in enumerate(zip(params, self.rules.best_indices)):
            rule_func = self.rules.rules[idx][0]
            _, signals = rule_func(param, OHLC)
            all_signals.append(signals)
        
        # Convert to DataFrame for GA
        signals_df = pd.DataFrame(all_signals).T
        signals_df = signals_df.fillna(0)
        
        # Prepare data for GA optimization
        X = signals_df.values
        
        # Calculate log returns for the target
        log_returns = np.log(OHLC['Close'] / OHLC['Close'].shift(1)).fillna(0)
        y = log_returns.values
        
        # Create input data for GA
        equation_inputs = np.column_stack((y, X))
        
        # Run GA optimization
        pop_size = self.config.ga_pop_size if self.config else 8
        parents = self.config.ga_parents if self.config else 4
        generations = self.config.ga_generations if self.config else 100
        
        best_weights = ga_module.GA_train(equation_inputs, 
                                          sol_per_pop=pop_size,
                                          num_parents_mating=parents,
                                          num_generations=generations)
        
        # Normalize weights
        total_weight = np.sum(np.abs(best_weights))
        if total_weight > 0:
            normalized_weights = best_weights / total_weight
        else:
            normalized_weights = np.ones(len(best_weights)) / len(best_weights)
        
        print(f"Optimized weights: {normalized_weights}")
        return normalized_weights

    def generate_signals(self, OHLC, rule_params=None, filter_regime=False):
        """Generate trading signals using weighted rules."""
        # Call to rules.generate_signals updated to handle legacy API
        final_signal = self.rules.generate_signals(OHLC, rule_params, filter_regime, self.weights)

        if isinstance(final_signal, pd.Series):
            # Apply regime filter if needed
            if filter_regime:
                final_signal = self._apply_regime_filter(final_signal, OHLC)

            # Create a proper DataFrame for backtester compatibility
            if 'LogReturn' in OHLC.columns:
                log_returns = OHLC['LogReturn']
            else:
                log_returns = np.log(OHLC['Close'] / OHLC['Close'].shift(1)).fillna(0)

            result_df = pd.DataFrame({
                'Signal': final_signal,
                'LogReturn': log_returns
            })
            self.signals = final_signal
            return result_df
        else:
            # It's already a DataFrame with the right format
            self.signals = final_signal['Signal']
            return final_signal



    def _apply_regime_filter(self, signal_series, OHLC):
        """Apply regime filtering to signals."""
        # More sophisticated regime filtering based on market conditions
        # Example: Use RSI to determine market regime
        try:
            from ta_functions import rsi
            
            market_rsi = rsi(OHLC['Close'], 14)
            
            # Define regimes
            bull_regime = market_rsi > 50
            bear_regime = market_rsi <= 50
            
            # Adjust signals based on regime
            signal_series.loc[bull_regime & (signal_series < 0)] = 0
            signal_series.loc[bear_regime & (signal_series > 0)] = 0
        except Exception as e:
            print(f"Warning: Could not apply regime filter. Error: {e}")
            # Fallback if rsi function fails
            volatility = OHLC["Close"].pct_change().rolling(20).std() * np.sqrt(252)
            avg_vol = volatility.mean()
            signal_series[volatility < avg_vol * 0.5] = 0
            
        return signal_series
    
    def __str__(self):
        """Return a string representation of the strategy."""
        return "Weighted Rules Strategy (GA Optimized)"


class StrategyFactory:
    """Factory class for creating strategy instances."""
    
    @staticmethod
    def create_strategy(config):
        """Create a strategy instance based on configuration."""
        if hasattr(config, 'use_weights') and not config.use_weights:
            return TopNStrategy(config, top_n=config.top_n)
        elif hasattr(config, 'no_weights') and config.no_weights:
            return TopNStrategy(config, top_n=config.top_n)
        else:
            return WeightedStrategy(config)
