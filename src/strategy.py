"""
Strategy module for the backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from rules import TradingRules

class Strategy:
    """Base strategy class for the backtesting engine."""
    
    def __init__(self, config=None):
        """Initialize the strategy with configuration."""
        self.config = config
        self.rules = TradingRules()
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
    
    def train(self, OHLC):
        """Train the strategy by finding the best parameters for each rule."""
        params, scores, indices = self.rules.train_rules(OHLC)
        self.top_indices = indices[:self.top_n]
        return params, scores, indices
    
    def generate_signals(self, OHLC, params=None, filter_regime=False):
        """Generate trading signals using the top N rules."""
        signals_df = self.rules.generate_signals(OHLC, params, top_n=self.top_n)
        
        if filter_regime:
            signals_df = self._apply_regime_filter(signals_df)
        
        self.signals = signals_df
        return signals_df
    
    def _apply_regime_filter(self, signals_df):
        """Apply regime filtering to signals."""
        # Simple example: filter out signals during low volatility
        volatility = signals_df['LogReturn'].rolling(20).std() * np.sqrt(252)
        avg_vol = volatility.mean()
        
        # Only take signals during higher volatility
        signals_df.loc[volatility < avg_vol * 0.5, 'Signal'] = 0
        
        return signals_df
    
    def __str__(self):
        """Return a string representation of the strategy."""
        return f"Top-{self.top_n} Rules Strategy"


class WeightedStrategy(Strategy):
    """Strategy that uses weighted combination of rules optimized by GA."""
    
    def __init__(self, config=None):
        """Initialize the WeightedStrategy with configuration."""
        super().__init__(config)
        self.weights = None
    
    def train(self, OHLC, ga_module=None):
        """Train the strategy by finding the best parameters and weights."""
        # First train the rules to get the parameters
        params, scores, _ = self.rules.train_rules(OHLC)
        
        # Then optimize weights using GA
        if ga_module is not None:
            self.weights = self._optimize_weights(OHLC, params, ga_module)
        
        return params, scores, self.weights
    
    def _optimize_weights(self, OHLC, params, ga_module):
        """Optimize rule weights using genetic algorithm."""
        # Generate signals using the trained parameters
        signals_df = self.rules.generate_signals(OHLC, params)
        
        # Prepare data for GA optimization
        rule_cols = [col for col in signals_df.columns if col.startswith('Rule')]
        X = signals_df[rule_cols].values
        y = signals_df['LogReturn'].values
        
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
        normalized_weights = best_weights / np.sum(np.abs(best_weights))
        
        print(f"Optimized weights: {normalized_weights}")
        return normalized_weights
    
    def generate_signals(self, OHLC, params=None, filter_regime=False):
        """Generate trading signals using weighted rules."""
        signals_df = self.rules.generate_signals(OHLC, params, weights=self.weights)
        
        if filter_regime:
            signals_df = self._apply_regime_filter(signals_df)
        
        self.signals = signals_df
        return signals_df
    
    def _apply_regime_filter(self, signals_df):
        """Apply regime filtering to signals."""
        # More sophisticated regime filtering based on market conditions
        # Example: Use RSI to determine market regime
        from ta_functions import rsi
        
        close = signals_df['Close'] if 'Close' in signals_df else None
        if close is not None:
            market_rsi = rsi(close, 14)
            
            # Define regimes
            bull_regime = market_rsi > 50
            bear_regime = market_rsi <= 50
            
            # Adjust signals based on regime
            signals_df.loc[bull_regime & (signals_df['Signal'] < 0), 'Signal'] = 0
            signals_df.loc[bear_regime & (signals_df['Signal'] > 0), 'Signal'] = 0
        
        return signals_df
    
    def __str__(self):
        """Return a string representation of the strategy."""
        return "Weighted Rules Strategy (GA Optimized)"


class StrategyFactory:
    """Factory class for creating strategy instances."""
    
    @staticmethod
    def create_strategy(config):
        """Create a strategy instance based on configuration."""
        if config.use_weights:
            return WeightedStrategy(config)
        else:
            return TopNStrategy(config, top_n=config.top_n)
