"""
Backtester module for the trading strategy engine.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional

from data import DataHandler
from strategy import Strategy, StrategyFactory
from metrics import calculate_metrics, print_metrics, compare_strategies

class Backtester:
    """Backtester class for evaluating trading strategies."""
    
    def __init__(self, config, data_handler=None, strategy=None):
        """Initialize the backtester with configuration."""
        self.config = config
        self.data_handler = data_handler if data_handler else DataHandler(config)
        self.strategy = strategy
        self.results = {}
    
    def run(self, ga_module=None):
        """Run the backtest process."""
        print(f"\nRunning backtest with {self.config}")
        
        # Load and preprocess data
        self.data_handler.load_data()
        self.data_handler.preprocess()
        self.data_handler.split_data()
        
        # Create strategy if not provided
        if self.strategy is None:
            self.strategy = StrategyFactory.create_strategy(self.config)
        
        # Train strategy if needed
        if self.config.train:
            print(f"\nTraining {self.strategy}...")
            train_ohlc = self.data_handler.get_ohlc(train=True)
            
            if isinstance(self.strategy, StrategyFactory.create_strategy(self.config).__class__):
                if self.config.use_weights and ga_module:
                    params, scores, weights = self.strategy.train(train_ohlc, ga_module)
                else:
                    params, scores, _ = self.strategy.train(train_ohlc)
                
                # Save trained parameters
                self.strategy.rules.save_params(self.config.params_file)
            else:
                print("Strategy type not compatible with training method")
                return False
        else:
            # Load previously trained parameters
            self.strategy.rules.load_params(self.config.params_file)
        
        # Test strategy if needed
        if self.config.test:
            print(f"\nTesting {self.strategy}...")
            test_ohlc = self.data_handler.get_ohlc(train=False)
            
            # Generate signals on test data
            test_signals = self.strategy.generate_signals(
                test_ohlc, 
                self.strategy.rules.rule_params,
                filter_regime=self.config.filter_regime
            )
            
            # Calculate performance metrics
            performance = self._calculate_performance(test_signals)
            
            # Store results
            self.results = {
                'strategy': str(self.strategy),
                'signals': test_signals,
                'performance': performance
            }
            
            # Print performance metrics
            print_metrics(performance, f"Test Performance - {self.strategy}")
            
            # Save results
            self._save_results()
        
        return True

    def _calculate_performance(self, signals_df):
        """Calculate performance metrics based on signals."""
        # Make sure we have the LogReturn column
        if 'LogReturn' not in signals_df.columns:
            print("Error: LogReturn column not found in signals dataframe")
            return {
                'strategy': {},
                'buy_and_hold': {}
            }

        # Extract returns and signals and convert to numeric
        returns = pd.to_numeric(signals_df['LogReturn'], errors='coerce')
        signals = pd.to_numeric(signals_df['Signal'], errors='coerce')

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns  # Apply signals with 1-day delay
        strategy_returns = strategy_returns.dropna()

        # Count trades
        trades = (signals.diff() != 0).sum() / 2  # Each trade consists of entry and exit

        # Calculate buy-and-hold returns for comparison
        bh_returns = returns.loc[strategy_returns.index]

        try:
            # Calculate metrics
            strategy_metrics = calculate_metrics(strategy_returns)
            bh_metrics = calculate_metrics(bh_returns)

            # Add number of trades
            strategy_metrics['number_of_trades'] = trades

            return {
                'strategy': strategy_metrics,
                'buy_and_hold': bh_metrics
            }
        except Exception as e:
            print(f"Error in performance calculation: {e}")
            # Return empty metrics as fallback
            empty_metrics = {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'annualized_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'number_of_trades': trades
            }
            return {
                'strategy': empty_metrics,
                'buy_and_hold': empty_metrics
            }
    
    def _save_results(self):
        """Save backtest results to a file."""
        # Create a serializable version of results
        serializable_results = {
            'strategy': self.results['strategy'],
            'performance': self.results['performance']
        }
        
        # Save to file
        try:
            with open(self.config.output_file, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Saved results to {self.config.output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    def compare_strategies(self, strategies_list, ga_module=None):
        """Compare multiple strategies on the same data."""
        if not self.data_handler.data:
            self.data_handler.load_data()
            self.data_handler.preprocess()
            self.data_handler.split_data()
        
        # Train and test each strategy
        results_list = []
        names_list = []
        
        for strategy_config in strategies_list:
            # Create a backtester with this strategy
            strategy = StrategyFactory.create_strategy(strategy_config)
            backtester = Backtester(strategy_config, self.data_handler, strategy)
            
            # Run the backtest
            backtester.run(ga_module)
            
            # Collect results
            results_list.append(backtester.results['performance']['strategy'])
            names_list.append(str(strategy))
        
        # Compare strategies
        from metrics import compare_strategies
        compare_strategies(results_list, names_list)
