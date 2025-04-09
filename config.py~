"""
Configuration parameters for the backtesting engine.
"""

import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Strategy Backtester')
    
    # Mode options
    parser.add_argument('--train', action='store_true', help='Train rule parameters')
    parser.add_argument('--test', action='store_true', help='Test using trained parameters')
    parser.add_argument('--backtest', action='store_true', help='Run both training and testing')
    
    # Data options
    parser.add_argument('--data', type=str, default='data.csv', 
                        help='Path to the data file (CSV format)')
    parser.add_argument('--train-size', type=float, default=0.6,
                        help='Proportion of data to use for training (default: 0.6)')
    
    # Strategy options
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top-performing rules to use for unweighted strategy')
    parser.add_argument('--no-weights', action='store_true',
                        help='Use unweighted strategy (top-n rules)')
    
    # GA options
    parser.add_argument('--ga-pop-size', type=int, default=8,
                        help='Population size for genetic algorithm')
    parser.add_argument('--ga-generations', type=int, default=100,
                        help='Number of generations for genetic algorithm')
    parser.add_argument('--ga-parents', type=int, default=4,
                        help='Number of parents for genetic algorithm')
    
    # Regime filtering
    parser.add_argument('--filter-regime', action='store_true',
                        help='Apply regime filtering')
    
    # Output options
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file for results')
    parser.add_argument('--save-params', type=str, default='params.json',
                        help='File to save/load trained parameters')
    
    args = parser.parse_args()
    
    # If backtest is specified, enable both train and test
    if args.backtest:
        args.train = True
        args.test = True
    
    # If neither train nor test is specified, default to backtest
    if not (args.train or args.test):
        args.train = True
        args.test = True
    
    return args

class Config:
    """Configuration class to hold all parameters."""
    
    def __init__(self, args=None):
        """Initialize configuration from args or defaults."""
        if args is None:
            # Use default configuration
            self.data_file = 'data.csv'
            self.train_size = 0.6
            self.top_n = 5
            self.use_weights = True
            self.ga_pop_size = 8
            self.ga_generations = 100
            self.ga_parents = 4
            self.filter_regime = False
            self.output_file = 'results.json'
            self.params_file = 'params.json'
            self.train = True
            self.test = True
        else:
            # Use configuration from parsed arguments
            self.data_file = args.data
            self.train_size = args.train_size
            self.top_n = args.top_n
            self.use_weights = not args.no_weights
            self.ga_pop_size = args.ga_pop_size
            self.ga_generations = args.ga_generations
            self.ga_parents = args.ga_parents
            self.filter_regime = args.filter_regime
            self.output_file = args.output
            self.params_file = args.save_params
            self.train = args.train
            self.test = args.test
    
    def __str__(self):
        """String representation of configuration."""
        return f"""Configuration:
  Data file: {self.data_file}
  Train size: {self.train_size}
  Strategy: {'Unweighted (top-' + str(self.top_n) + ')' if not self.use_weights else 'Weighted (GA)'}
  GA parameters: pop={self.ga_pop_size}, gen={self.ga_generations}, parents={self.ga_parents}
  Regime filtering: {'Enabled' if self.filter_regime else 'Disabled'}
  Mode: {'Train & Test' if self.train and self.test else 'Train only' if self.train else 'Test only'}
"""

# Default configuration instance
config = Config()
