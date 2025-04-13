#!/usr/bin/env python
"""
Trading System Main Runner

This script provides a simple entry point to test the trading system's functionality.
It demonstrates how to use the key components of the system including data loading,
strategy setup, backtesting, and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import system components
# Core components
from src.config import ConfigManager
from src.engine.backtester import Backtester
from src.engine.market_simulator import MarketSimulator

# Data handling
from src.analytics.visualization import TradeVisualizer
from src.analytics.metrics import calculate_metrics_from_trades

# Rule and strategy components
from src.signals.signal_processing import Signal, SignalType
from src.feature_base import Feature, FeatureSet
from src.price_features import ReturnFeature, NormalizedPriceFeature
from src.technical_features import VolatilityFeature, MACrossoverFeature
from src.strategies.weighted_strategy import WeightedStrategy

# For feature-based rule creation
from src.feature_registry import get_registry, register_feature


# Simple data handler for CSV data
class SimpleCSVDataHandler:
    """Simple data handler for loading and processing CSV files."""
    
    def __init__(self, filepath, date_column='Date', train_fraction=0.8):
        """
        Initialize with data file path.
        
        Args:
            filepath: Path to CSV file
            date_column: Column name for dates
            train_fraction: Fraction of data to use for training
        """
        self.filepath = filepath
        self.date_column = date_column
        self.train_fraction = train_fraction
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_index = 0
        self.test_index = 0
        
        self._load_data()
        
    def _load_data(self):
        """Load data from CSV file."""
        try:
            # Load the data
            self.data = pd.read_csv(self.filepath)
            
            # Convert date column to datetime
            if self.date_column in self.data.columns:
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
                self.data = self.data.sort_values(self.date_column)
            
            # Split into train/test sets
            split_idx = int(len(self.data) * self.train_fraction)
            self.train_data = self.data.iloc[:split_idx].reset_index(drop=True)
            self.test_data = self.data.iloc[split_idx:].reset_index(drop=True)
            
            print(f"Loaded data: {len(self.data)} rows total, "
                  f"{len(self.train_data)} for training, {len(self.test_data)} for testing")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.data = pd.DataFrame()
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()
    
    def reset_train(self):
        """Reset training data iterator."""
        self.train_index = 0
    
    def reset_test(self):
        """Reset testing data iterator."""
        self.test_index = 0
    
    def get_next_train_bar(self):
        """Get next bar from training data."""
        if self.train_index >= len(self.train_data):
            return None
        
        bar = self._convert_row_to_bar(self.train_data.iloc[self.train_index])
        self.train_index += 1
        return bar
    
    def get_next_test_bar(self):
        """Get next bar from testing data."""
        if self.test_index >= len(self.test_data):
            return None
        
        bar = self._convert_row_to_bar(self.test_data.iloc[self.test_index])
        self.test_index += 1
        return bar
    
    def _convert_row_to_bar(self, row):
        """Convert DataFrame row to bar dictionary."""
        bar = {}
        
        # Extract OHLCV data if available
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in row:
                bar[col] = row[col]
        
        # Extract timestamp
        if self.date_column in row:
            bar['timestamp'] = row[self.date_column]
        else:
            bar['timestamp'] = datetime.now()
        
        return bar


# Simple Rule class based on a feature
class FeatureRule:
    """Rule that uses a feature to generate signals."""
    
    def __init__(self, feature, buy_threshold=0.5, sell_threshold=-0.5, name=None):
        """
        Initialize the feature-based rule.
        
        Args:
            feature: Feature object that calculates values
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
            name: Rule name
        """
        self.feature = feature
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.name = name or f"{feature.name}_rule"
    
    def on_bar(self, bar):
        """
        Process a bar and generate a signal.
        
        Args:
            bar: Bar data dictionary
            
        Returns:
            Signal object
        """
        # Calculate feature value
        feature_value = self._extract_feature_value(self.feature.calculate(bar))
        
        # Determine signal type
        if feature_value > self.buy_threshold:
            signal_type = SignalType.BUY
        elif feature_value < self.sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create signal
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=signal_type,
            price=bar["Close"],
            rule_id=self.name,
            confidence=abs(feature_value)
        )
    
    def _extract_feature_value(self, feature_result):
        """Extract a numeric value from feature result."""
        if isinstance(feature_result, (int, float)):
            return feature_result
        elif isinstance(feature_result, dict):
            # Try to extract from dictionary
            if 'signal' in feature_result:
                return feature_result['signal']
            elif 'value' in feature_result:
                return feature_result['value']
            elif 'state' in feature_result:
                return feature_result['state']
            # Try first numeric value
            for v in feature_result.values():
                if isinstance(v, (int, float)):
                    return v
        
        # Default if no proper value found
        return 0
    
    def reset(self):
        """Reset the rule state."""
        # Reset feature if it's stateful
        if hasattr(self.feature, 'reset'):
            self.feature.reset()


def create_sample_data(filepath, num_rows=1000, volatility=0.01):
    """
    Create sample price data for testing.
    
    Args:
        filepath: Path to save the CSV file
        num_rows: Number of data points to generate
        volatility: Price volatility factor
    """
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]
    
    # Generate price data with random walk
    close = [100]  # Start at $100
    for i in range(1, num_rows):
        close.append(close[-1] * (1 + np.random.normal(0, volatility)))
    
    # Generate OHLCV data
    data = []
    for i in range(num_rows):
        daily_volatility = volatility * close[i] * 0.5
        high = close[i] + abs(np.random.normal(0, daily_volatility))
        low = close[i] - abs(np.random.normal(0, daily_volatility))
        open_price = low + np.random.random() * (high - low)
        volume = int(np.random.normal(1000000, 500000))
        
        data.append({
            'Date': dates[i],
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close[i],
            'Volume': max(0, volume)
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Created sample data file: {filepath}")


def main():
    """Main function to run the trading system test."""
    # Setup output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create or load sample data
    data_file = os.path.join(output_dir, "sample_data.csv")
    if not os.path.exists(data_file):
        create_sample_data(data_file)
    
    # Create data handler
    data_handler = SimpleCSVDataHandler(data_file)
    
    # Create features for our rules
    ma_crossover_feature = MACrossoverFeature(
        name="ma_crossover", 
        params={
            'fast_ma': 'SMA_10',
            'slow_ma': 'SMA_30'
        }
    )
    
    volatility_feature = VolatilityFeature(
        name="volatility", 
        params={
            'method': 'std_dev',
            'period': 20
        }
    )
    
    # Create rules from features
    ma_rule = FeatureRule(
        feature=ma_crossover_feature,
        buy_threshold=0.5,
        sell_threshold=-0.5,
        name="MA_Crossover_Rule"
    )
    
    volatility_rule = FeatureRule(
        feature=volatility_feature,
        buy_threshold=0.6,
        sell_threshold=-0.6,
        name="Volatility_Rule"
    )
    
    # Create strategy with rules
    strategy = WeightedStrategy(
        rules=[ma_rule, volatility_rule],
        weights=[0.7, 0.3],
        buy_threshold=0.4,
        sell_threshold=-0.4,
        name="Sample_Strategy"
    )
    
    # Create backtester
    backtester = Backtester(
        config={},
        data_handler=data_handler,
        strategy=strategy
    )
    
    # Run the backtest
    print("Running backtest...")
    results = backtester.run(use_test_data=True)
    
    # Print summary results
    print("\nBacktest Results:")
    print(f"Number of trades: {results['num_trades']}")
    print(f"Total return: {results['total_percent_return']:.2f}%")
    
    # Calculate additional metrics
    if results['num_trades'] > 0:
        metrics = calculate_metrics_from_trades(results['trades'])
        print(f"Win rate: {metrics['win_rate']:.2%}")
        print(f"Profit factor: {metrics['profit_factor']:.2f}")
        print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Create visualizations
    visualizer = TradeVisualizer()
    
    if results['num_trades'] > 0:
        print("\nCreating visualizations...")
        
        # Equity curve
        equity_fig = visualizer.plot_equity_curve(
            results['trades'], 
            title="Equity Curve - Test Data"
        )
        equity_fig.savefig(os.path.join(output_dir, "equity_curve.png"))
        
        # Drawdowns
        drawdown_fig = visualizer.plot_drawdowns(
            results['trades'], 
            title="Drawdown Analysis"
        )
        drawdown_fig.savefig(os.path.join(output_dir, "drawdowns.png"))
        
        # Returns distribution
        returns_fig = visualizer.plot_returns_distribution(
            results['trades'], 
            title="Trade Returns Distribution"
        )
        returns_fig.savefig(os.path.join(output_dir, "returns_distribution.png"))
        
        print(f"Visualization images saved to {output_dir}/ directory")
    
    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    main()
