#!/usr/bin/env python
"""
Advanced Strategy Optimization Script

This script implements a sophisticated strategy optimization process including:
- Multi-stage validation (train/validate/test)
- Regime-based strategy optimization
- Multi-strategy ensemble creation
- Extensive parameter grid search
- Advanced performance metrics
- Walk-forward validation

Place this script in your project root directory (~/mmbt/).
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

class AdvancedStrategyOptimizer:
    """
    Advanced strategy optimization framework.
    """
    
    def __init__(self, data_file, price_col='Close'):
        """
        Initialize the optimizer.
        
        Args:
            data_file: Path to CSV data file
            price_col: Name of the price column to use
        """
        self.data_file = data_file
        self.price_col = price_col
        self.data = None
        self.regimes = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.best_params = {}
        self.best_strategies = {}
        self.ensemble = None
        
    def load_data(self):
        """Load and prepare data."""
        print(f"Loading data from {self.data_file}")
        
        # Read the CSV data
        self.data = pd.read_csv(self.data_file)
        print(f"Loaded data with {len(self.data)} rows")
        
        # Format timestamp
        if 'timestamp' in self.data.columns:
            if isinstance(self.data['timestamp'].iloc[0], str):
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                
            # Sort by timestamp
            self.data = self.data.sort_values('timestamp')
            print(f"Data timeframe: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            
        # Ensure price column exists
        if self.price_col not in self.data.columns:
            # Try to find alternative
            for alt_col in ['Close', 'close', 'Price', 'price']:
                if alt_col in self.data.columns:
                    self.price_col = alt_col
                    print(f"Using {self.price_col} as price column")
                    break
            else:
                raise ValueError(f"Could not find price column in data: {self.data.columns.tolist()}")
                
        # Calculate returns
        self.data['returns'] = self.data[self.price_col].pct_change()
        
        # Report basic statistics
        print("\nData Summary:")
        print(f"  Rows: {len(self.data)}")
        print(f"  Columns: {', '.join(self.data.columns)}")
        print(f"  Price range: {self.data[self.price_col].min():.2f} to {self.data[self.price_col].max():.2f}")
        print(f"  Mean daily return: {self.data['returns'].mean()*100:.4f}%")
        print(f"  Volatility (annualized): {self.data['returns'].std() * np.sqrt(252) * 100:.2f}%")
        
        return self.data
    
    def split_data(self, train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Split data into training, validation, and test sets.
        
        Args:
            train_size: Fraction of data for training
            val_size: Fraction of data for validation
            test_size: Fraction of data for testing
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if train_size + val_size + test_size != 1.0:
            raise ValueError("Split proportions must sum to 1.0")
            
        data_length = len(self.data)
        train_end = int(data_length * train_size)
        val_end = train_end + int(data_length * val_size)
        
        self.train_data = self.data.iloc[:train_end].copy()
        self.val_data = self.data.iloc[train_end:val_end].copy()
        self.test_data = self.data.iloc[val_end:].copy()
        
        print("\nSplit Data into:")
        print(f"  Training set:   {len(self.train_data)} rows ({self.train_data['timestamp'].min()} to {self.train_data['timestamp'].max()})")
        print(f"  Validation set: {len(self.val_data)} rows ({self.val_data['timestamp'].min()} to {self.val_data['timestamp'].max()})")
        print(f"  Testing set:    {len(self.test_data)} rows ({self.test_data['timestamp'].min()} to {self.test_data['timestamp'].max()})")
        
        return self.train_data, self.val_data, self.test_data
    
    def detect_regimes(self, window_size=20):
        """
        Detect market regimes based on volatility and trend.
        
        Args:
            window_size: Window size for regime detection
        
        Returns:
            DataFrame with regime indicators
        """
        print("\nDetecting market regimes...")
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Create a copy of data for regimes
        self.regimes = self.data.copy()
        
        # Calculate volatility
        self.regimes['volatility'] = self.regimes['returns'].rolling(window=window_size).std() * np.sqrt(252)
        
        # Calculate trend strength using linear regression slope
        def rolling_slope(x):
            if len(x) < 2:
                return 0
            try:
                x_array = np.arange(len(x))
                slope = np.polyfit(x_array, x, 1)[0]
                return slope * len(x)  # Scale by window size
            except:
                return 0
                
        self.regimes['trend'] = self.regimes[self.price_col].rolling(window=window_size).apply(rolling_slope, raw=True)
        
        # Define regimes
        # High volatility threshold: 80th percentile
        high_vol_threshold = self.regimes['volatility'].quantile(0.8)
        # Strong trend threshold: 80th percentile of absolute trend
        strong_trend_threshold = self.regimes['trend'].abs().quantile(0.8)
        
        # Classify regimes
        self.regimes['regime'] = 'normal'
        
        # Trending up: strong positive trend
        self.regimes.loc[self.regimes['trend'] > strong_trend_threshold, 'regime'] = 'trending_up'
        
        # Trending down: strong negative trend
        self.regimes.loc[self.regimes['trend'] < -strong_trend_threshold, 'regime'] = 'trending_down'
        
        # Volatile: high volatility
        self.regimes.loc[self.regimes['volatility'] > high_vol_threshold, 'regime'] = 'volatile'
        
        # Ranging: low trend and low volatility
        self.regimes.loc[(self.regimes['trend'].abs() < strong_trend_threshold/2) & 
                         (self.regimes['volatility'] < high_vol_threshold/2), 'regime'] = 'ranging'
        
        # Report regime distribution
        regime_counts = self.regimes['regime'].value_counts()
        print("\nRegime Distribution:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} bars ({count/len(self.regimes)*100:.1f}%)")
        
        # Apply to all datasets
        if self.train_data is not None:
            for i, dataset in enumerate([self.train_data, self.val_data, self.test_data]):
                dataset_name = ["Training", "Validation", "Testing"][i]
                dataset['regime'] = self.regimes.loc[dataset.index, 'regime']
                regime_dist = dataset['regime'].value_counts(normalize=True) * 100
                print(f"\n{dataset_name} Set Regime Distribution:")
                for regime, pct in regime_dist.items():
                    print(f"  {regime}: {pct:.1f}%")
        
        return self.regimes
    
    def optimize_moving_average_strategy(self, regimes=False):
        """
        Optimize parameters for moving average strategies, optionally by regime.
        
        Args:
            regimes: Whether to optimize separately for each regime
            
        Returns:
            Dictionary of best parameters
        """
        print("\n================================================")
        print("MOVING AVERAGE STRATEGY OPTIMIZATION")
        print("================================================")
        
        if self.train_data is None:
            raise ValueError("Data not split. Call split_data() first.")
            
        # Define parameter grid
        param_grid = {
            'fast_window': [3, 5, 8, 10, 12, 15, 20],
            'slow_window': [20, 30, 40, 50, 60, 80, 100],
            'signal_threshold': [0.0, 0.001, 0.002]  # Minimum separation to generate signal
        }
        
        # Count total combinations
        total_combos = sum(1 for fast in param_grid['fast_window'] 
                         for slow in param_grid['slow_window'] 
                         for thresh in param_grid['signal_threshold']
                         if fast < slow)
        
        print(f"Testing {total_combos} parameter combinations")
        
        if not regimes:
            # Optimize on whole training dataset
            best_params, best_metrics = self._optimize_ma_parameters(
                self.train_data, param_grid, "all")
            
            self.best_params['moving_average'] = {
                'all': best_params
            }
            
            # Validate on validation set
            val_metrics = self._evaluate_ma_strategy(
                self.val_data, best_params['fast_window'], 
                best_params['slow_window'], best_params['signal_threshold'])
            
            print("\nValidation Results:")
            print(f"  Return: {val_metrics['total_return']*100:.2f}%")
            print(f"  Sharpe: {val_metrics['sharpe_ratio']:.4f}")
            print(f"  Win Rate: {val_metrics['win_rate']*100:.2f}%")
            print(f"  Profit Factor: {val_metrics['profit_factor']:.2f}")
            
        else:
            # Optimize separately for each regime
            self.best_params['moving_average'] = {}
            
            for regime in ['trending_up', 'trending_down', 'volatile', 'ranging', 'normal']:
                # Filter training data by regime
                regime_data = self.train_data[self.train_data['regime'] == regime]
                
                # Skip if there's not enough data for this regime
                if len(regime_data) < 100:
                    print(f"\nSkipping regime '{regime}' - insufficient data ({len(regime_data)} bars)")
                    continue
                    
                print(f"\nOptimizing MA strategy for '{regime}' regime ({len(regime_data)} bars)")
                
                # Optimize for this regime
                regime_best_params, regime_metrics = self._optimize_ma_parameters(
                    regime_data, param_grid, regime)
                
                self.best_params['moving_average'][regime] = regime_best_params
                
                # Validate on validation set (same regime)
                val_regime_data = self.val_data[self.val_data['regime'] == regime]
                
                if len(val_regime_data) > 20:
                    val_metrics = self._evaluate_ma_strategy(
                        val_regime_data, regime_best_params['fast_window'], 
                        regime_best_params['slow_window'], regime_best_params['signal_threshold'])
                    
                    print(f"Validation Results ({len(val_regime_data)} bars):")
                    print(f"  Return: {val_metrics['total_return']*100:.2f}%")
                    print(f"  Sharpe: {val_metrics['sharpe_ratio']:.4f}")
                    print(f"  Win Rate: {val_metrics['win_rate']*100:.2f}%")
                else:
                    print(f"Insufficient validation data for regime '{regime}' ({len(val_regime_data)} bars)")
        
        return self.best_params['moving_average']
    
    def _optimize_ma_parameters(self, data, param_grid, regime_name):
        """
        Optimize moving average parameters on given data.
        
        Args:
            data: DataFrame with price data
            param_grid: Dictionary of parameter ranges
            regime_name: Name of the regime (for display)
            
        Returns:
            Tuple of (best parameters, best metrics)
        """
        best_sharpe = -np.inf
        best_params = None
        best_metrics = None
        
        # Track all results
        results = []
        
        # Use tqdm for progress display
        total_combos = sum(1 for fast in param_grid['fast_window'] 
                         for slow in param_grid['slow_window'] 
                         for thresh in param_grid['signal_threshold']
                         if fast < slow)
        
        combo_count = 0
        print(f"Testing {total_combos} combinations for regime '{regime_name}'...")
        
        # Grid search
        for fast in param_grid['fast_window']:
            for slow in param_grid['slow_window']:
                # Skip invalid combinations
                if fast >= slow:
                    continue
                    
                for threshold in param_grid['signal_threshold']:
                    combo_count += 1
                    if combo_count % 50 == 0:
                        print(f"  Progress: {combo_count}/{total_combos} combinations tested")
                    
                    # Evaluate this parameter set
                    metrics = self._evaluate_ma_strategy(data, fast, slow, threshold)
                    
                    # Store results
                    results.append({
                        'fast_window': fast,
                        'slow_window': slow,
                        'signal_threshold': threshold,
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'total_return': metrics['total_return'],
                        'win_rate': metrics['win_rate'],
                        'profit_factor': metrics['profit_factor']
                    })
                    
                    # Update best parameters if this is better
                    if metrics['sharpe_ratio'] > best_sharpe:
                        best_sharpe = metrics['sharpe_ratio']
                        best_params = {
                            'fast_window': fast,
                            'slow_window': slow,
                            'signal_threshold': threshold
                        }
                        best_metrics = metrics
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Show top 5 parameter sets
        if len(results_df) > 0:
            print(f"\nTop 5 MA Parameter Sets for '{regime_name}' regime:")
            top_results = results_df.sort_values('sharpe_ratio', ascending=False).head(5)
            for i, result in top_results.iterrows():
                print(f"  MA({result['fast_window']}, {result['slow_window']}, thresh={result['signal_threshold']:.4f}):")
                print(f"    Sharpe: {result['sharpe_ratio']:.4f}, Return: {result['total_return']*100:.2f}%, Win Rate: {result['win_rate']*100:.2f}%")
            
            # Display best parameters
            print(f"\nBest Parameters for '{regime_name}' regime:")
            print(f"  Fast Window: {best_params['fast_window']}")
            print(f"  Slow Window: {best_params['slow_window']}")
            print(f"  Signal Threshold: {best_params['signal_threshold']:.4f}")
            print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
            print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
            print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
            print(f"  Profit Factor: {best_metrics['profit_factor']:.2f}")
        else:
            print(f"No valid results for regime '{regime_name}'")
        
        return best_params, best_metrics
    
    def _evaluate_ma_strategy(self, data, fast_window, slow_window, threshold):
        """
        Evaluate a moving average strategy with given parameters.
        
        Args:
            data: DataFrame with price data
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            threshold: Signal threshold
            
        Returns:
            Dictionary of performance metrics
        """
        # Create a copy of the data
        df = data.copy()
        
        # Calculate moving averages
        df[f'MA_{fast_window}'] = df[self.price_col].rolling(window=fast_window).mean()
        df[f'MA_{slow_window}'] = df[self.price_col].rolling(window=slow_window).mean()
        
        # Calculate percentage spread between MAs
        df['ma_spread'] = (df[f'MA_{fast_window}'] - df[f'MA_{slow_window}']) / df[f'MA_{slow_window}']
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['ma_spread'] > threshold, 'signal'] = 1  # Buy signal
        df.loc[df['ma_spread'] < -threshold, 'signal'] = -1  # Sell signal
        
        # Calculate strategy returns
        df['strategy_returns'] = df['signal'].shift(1) * df['returns']
        
        # Calculate performance metrics
        valid_returns = df['strategy_returns'].dropna()
        
        if len(valid_returns) < 20:
            return {
                'sharpe_ratio': -np.inf,
                'total_return': -1,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        total_return = (valid_returns + 1).prod() - 1
        sharpe = np.sqrt(252) * valid_returns.mean() / valid_returns.std() if valid_returns.std() != 0 else 0
        win_rate = (valid_returns > 0).sum() / len(valid_returns)
        
        # Calculate profit factor
        winning_trades = valid_returns[valid_returns > 0].sum()
        losing_trades = abs(valid_returns[valid_returns < 0].sum())
        profit_factor = winning_trades / losing_trades if losing_trades != 0 else np.inf
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'returns': valid_returns
        }
    
    def optimize_rsi_strategy(self, regimes=False):
        """
        Optimize parameters for RSI strategies, optionally by regime.
        
        Args:
            regimes: Whether to optimize separately for each regime
            
        Returns:
            Dictionary of best parameters
        """
        print("\n================================================")
        print("RSI STRATEGY OPTIMIZATION")
        print("================================================")
        
        if self.train_data is None:
            raise ValueError("Data not split. Call split_data() first.")
            
        # Define parameter grid
        param_grid = {
            'period': [7, 10, 14, 20, 25],
            'overbought': [65, 70, 75, 80],
            'oversold': [20, 25, 30, 35],
            'exit_threshold': [40, 50, 60]  # Middle threshold for exit
        }
        
        # Count total combinations
        total_combos = sum(1 for period in param_grid['period'] 
                         for overbought in param_grid['overbought']
                         for oversold in param_grid['oversold']
                         for exit in param_grid['exit_threshold']
                         if oversold < exit < overbought)
        
        print(f"Testing {total_combos} parameter combinations")
        
        if not regimes:
            # Optimize on whole training dataset
            best_params, best_metrics = self._optimize_rsi_parameters(
                self.train_data, param_grid, "all")
            
            self.best_params['rsi'] = {
                'all': best_params
            }
            
            # Validate on validation set
            val_metrics = self._evaluate_rsi_strategy(
                self.val_data, best_params['period'], 
                best_params['overbought'], best_params['oversold'],
                best_params['exit_threshold'])
            
            print("\nValidation Results:")
            print(f"  Return: {val_metrics['total_return']*100:.2f}%")
            print(f"  Sharpe: {val_metrics['sharpe_ratio']:.4f}")
            print(f"  Win Rate: {val_metrics['win_rate']*100:.2f}%")
            
        else:
            # Optimize separately for each regime
            self.best_params['rsi'] = {}
            
            for regime in ['trending_up', 'trending_down', 'volatile', 'ranging', 'normal']:
                # Filter training data by regime
                regime_data = self.train_data[self.train_data['regime'] == regime]
                
                # Skip if there's not enough data for this regime
                if len(regime_data) < 100:
                    print(f"\nSkipping regime '{regime}' - insufficient data ({len(regime_data)} bars)")
                    continue
                    
                print(f"\nOptimizing RSI strategy for '{regime}' regime ({len(regime_data)} bars)")
                
                # Optimize for this regime
                regime_best_params, regime_metrics = self._optimize_rsi_parameters(
                    regime_data, param_grid, regime)
                
                self.best_params['rsi'][regime] = regime_best_params
                
                # Validate on validation set (same regime)
                val_regime_data = self.val_data[self.val_data['regime'] == regime]
                
                if len(val_regime_data) > 20:
                    val_metrics = self._evaluate_rsi_strategy(
                        val_regime_data, regime_best_params['period'], 
                        regime_best_params['overbought'], regime_best_params['oversold'],
                        regime_best_params['exit_threshold'])
                    
                    print(f"Validation Results ({len(val_regime_data)} bars):")
                    print(f"  Return: {val_metrics['total_return']*100:.2f}%")
                    print(f"  Sharpe: {val_metrics['sharpe_ratio']:.4f}")
                    print(f"  Win Rate: {val_metrics['win_rate']*100:.2f}%")
                else:
                    print(f"Insufficient validation data for regime '{regime}' ({len(val_regime_data)} bars)")
        
        return self.best_params['rsi']
    
    def _optimize_rsi_parameters(self, data, param_grid, regime_name):
        """
        Optimize RSI parameters on given data.
        
        Args:
            data: DataFrame with price data
            param_grid: Dictionary of parameter ranges
            regime_name: Name of the regime (for display)
            
        Returns:
            Tuple of (best parameters, best metrics)
        """
        best_sharpe = -np.inf
        best_params = None
        best_metrics = None
        
        # Track all results
        results = []
        
        # Count valid combinations for progress reporting
        total_combos = sum(1 for period in param_grid['period'] 
                         for overbought in param_grid['overbought']
                         for oversold in param_grid['oversold']
                         for exit in param_grid['exit_threshold']
                         if oversold < exit < overbought)
        
        combo_count = 0
        print(f"Testing {total_combos} combinations for regime '{regime_name}'...")
        
        # Grid search
        for period in param_grid['period']:
            for overbought in param_grid['overbought']:
                for oversold in param_grid['oversold']:
                    # Skip invalid combinations
                    if oversold >= overbought:
                        continue
                    
                    for exit_threshold in param_grid['exit_threshold']:
                        # Skip invalid combinations
                        if not (oversold < exit_threshold < overbought):
                            continue
                        
                        combo_count += 1
                        if combo_count % 50 == 0:
                            print(f"  Progress: {combo_count}/{total_combos} combinations tested")
                        
                        # Evaluate this parameter set
                        metrics = self._evaluate_rsi_strategy(data, period, overbought, oversold, exit_threshold)
                        
                        # Store results
                        results.append({
                            'period': period,
                            'overbought': overbought,
                            'oversold': oversold,
                            'exit_threshold': exit_threshold,
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'total_return': metrics['total_return'],
                            'win_rate': metrics['win_rate'],
                            'profit_factor': metrics['profit_factor']
                        })
                        
                        # Update best parameters if this is better
                        if metrics['sharpe_ratio'] > best_sharpe:
                            best_sharpe = metrics['sharpe_ratio']
                            best_params = {
                                'period': period,
                                'overbought': overbought,
                                'oversold': oversold,
                                'exit_threshold': exit_threshold
                            }
                            best_metrics = metrics
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Show top 5 parameter sets
        if len(results_df) > 0:
            print(f"\nTop 5 RSI Parameter Sets for '{regime_name}' regime:")
            top_results = results_df.sort_values('sharpe_ratio', ascending=False).head(5)
            for i, result in top_results.iterrows():
                print(f"  RSI({result['period']}, OB={result['overbought']}, OS={result['oversold']}, Exit={result['exit_threshold']}):")
                print(f"    Sharpe: {result['sharpe_ratio']:.4f}, Return: {result['total_return']*100:.2f}%, Win Rate: {result['win_rate']*100:.2f}%")
            
            # Display best parameters
            print(f"\nBest Parameters for '{regime_name}' regime:")
            print(f"  Period: {best_params['period']}")
            print(f"  Overbought: {best_params['overbought']}")
            print(f"  Oversold: {best_params['oversold']}")
            print(f"  Exit Threshold: {best_params['exit_threshold']}")
            print(f"  Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
            print(f"  Total Return: {best_metrics['total_return']*100:.2f}%")
            print(f"  Win Rate: {best_metrics['win_rate']*100:.2f}%")
            print(f"  Profit Factor: {best_metrics['profit_factor']:.2f}")
        else:
            print(f"No valid results for regime '{regime_name}'")
        
        return best_params, best_metrics
        
    def _evaluate_rsi_strategy(self, data, period, overbought, oversold, exit_threshold):
        """
        Evaluate an RSI strategy with given parameters.
        
        Args:
            data: DataFrame with price data
            period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
            exit_threshold: Middle threshold for exits
            
        Returns:
            Dictionary of performance metrics
        """
        # Create a copy of the data
        df = data.copy()
        
        # Calculate RSI
        delta = df[self.price_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['rsi_signal'] = 0
        
        # Entry signals
        df.loc[df['RSI'] < oversold, 'rsi_signal'] = 1  # Buy signal
        df.loc[df['RSI'] > overbought, 'rsi_signal'] = -1  # Sell signal
        
        # Exit signals (more sophisticated approach)
        # For long positions, exit when RSI crosses above exit_threshold
        # For short positions, exit when RSI crosses below exit_threshold
        position = 0
        for i in range(1, len(df)):
            if df.iloc[i-1]['rsi_signal'] == 1:  # Previous bar had buy signal
                position = 1
            elif df.iloc[i-1]['rsi_signal'] == -1:  # Previous bar had sell signal
                position = -1
            elif position == 1 and df.iloc[i-1]['RSI'] < exit_threshold and df.iloc[i]['RSI'] > exit_threshold:
                # Exit long position
                df.iloc[i, df.columns.get_loc('rsi_signal')] = 0
                position = 0
            elif position == -1 and df.iloc[i-1]['RSI'] > exit_threshold and df.iloc[i]['RSI'] < exit_threshold:
                # Exit short position
                df.iloc[i, df.columns.get_loc('rsi_signal')] = 0
                position = 0
        
        # Calculate strategy returns
        df['strategy_returns'] = df['rsi_signal'].shift(1) * df['returns']
        
        # Calculate performance metrics
        valid_returns = df['strategy_returns'].dropna()
        
        if len(valid_returns) < 20:
            return {
                'sharpe_ratio': -np.inf,
                'total_return': -1,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        total_return = (valid_returns + 1).prod() - 1
        sharpe = np.sqrt(252) * valid_returns.mean() / valid_returns.std() if valid_returns.std() != 0 else 0
        win_rate = (valid_returns > 0).sum() / len(valid_returns)
        
        # Calculate profit factor
        winning_trades = valid_returns[valid_returns > 0].sum()
        losing_trades = abs(valid_returns[valid_returns < 0].sum())
        profit_factor = winning_trades / losing_trades if losing_trades != 0 else np.inf
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'returns': valid_returns
        }
    
    def build_ensemble_strategy(self):
        """
        Build an ensemble strategy combining the optimized strategies.
        
        Returns:
            Dictionary with ensemble strategy results
        """
        print("\n================================================")
        print("ENSEMBLE STRATEGY CREATION")
        print("================================================")
        
        if not self.best_params:
            raise ValueError("No optimized strategies found. Run optimization first.")
            
        # Create a copy of validation data for ensemble building
        ensemble_data = self.val_data.copy()
        
        # Track all strategy signals and returns for ensemble
        strategies = {}
        
        # Add MA strategy signals
        if 'moving_average' in self.best_params:
            # If regime-specific parameters exist, use them
            if len(self.best_params['moving_average']) > 1:
                # Initialize columns for MA strategy
                ensemble_data['ma_signal'] = 0
                ensemble_data['ma_returns'] = np.nan
                
                # Apply regime-specific parameters
                for regime, params in self.best_params['moving_average'].items():
                    if regime == 'all':
                        continue
                        
                    # Filter data for this regime
                    regime_mask = ensemble_data['regime'] == regime
                    
                    if not regime_mask.any():
                        continue
                    
                    # Calculate indicators for this regime
                    fast = params['fast_window']
                    slow = params['slow_window']
                    threshold = params['signal_threshold']
                    
                    # Calculate MAs for the entire dataset (to avoid boundary issues)
                    ensemble_data[f'MA_{fast}'] = ensemble_data[self.price_col].rolling(window=fast).mean()
                    ensemble_data[f'MA_{slow}'] = ensemble_data[self.price_col].rolling(window=slow).mean()
                    
                    # Calculate spread
                    ensemble_data['ma_spread'] = (ensemble_data[f'MA_{fast}'] - ensemble_data[f'MA_{slow}']) / ensemble_data[f'MA_{slow}']
                    
                    # Generate signals for this regime
                    ensemble_data.loc[regime_mask & (ensemble_data['ma_spread'] > threshold), 'ma_signal'] = 1
                    ensemble_data.loc[regime_mask & (ensemble_data['ma_spread'] < -threshold), 'ma_signal'] = -1
                
                # Calculate returns
                ensemble_data['ma_returns'] = ensemble_data['ma_signal'].shift(1) * ensemble_data['returns']
                
            else:
                # Use single parameter set for all regimes
                params = self.best_params['moving_average']['all']
                fast = params['fast_window']
                slow = params['slow_window']
                threshold = params['signal_threshold']
                
                # Calculate MAs
                ensemble_data[f'MA_{fast}'] = ensemble_data[self.price_col].rolling(window=fast).mean()
                ensemble_data[f'MA_{slow}'] = ensemble_data[self.price_col].rolling(window=slow).mean()
                
                # Calculate spread
                ensemble_data['ma_spread'] = (ensemble_data[f'MA_{fast}'] - ensemble_data[f'MA_{slow}']) / ensemble_data[f'MA_{slow}']
                
                # Generate signals
                ensemble_data['ma_signal'] = 0
                ensemble_data.loc[ensemble_data['ma_spread'] > threshold, 'ma_signal'] = 1
                ensemble_data.loc[ensemble_data['ma_spread'] < -threshold, 'ma_signal'] = -1
                
                # Calculate returns
                ensemble_data['ma_returns'] = ensemble_data['ma_signal'].shift(1) * ensemble_data['returns']
            
            # Track MA strategy
            strategies['moving_average'] = {
                'signal_col': 'ma_signal',
                'returns_col': 'ma_returns'
            }
        
        # Add RSI strategy signals
        if 'rsi' in self.best_params:
            # If regime-specific parameters exist, use them
            if len(self.best_params['rsi']) > 1:
                # Initialize columns for RSI strategy
                ensemble_data['rsi_signal'] = 0
                ensemble_data['rsi_returns'] = np.nan
                
                # Apply regime-specific parameters
                for regime, params in self.best_params['rsi'].items():
                    if regime == 'all':
                        continue
                        
                    # Filter data for this regime
                    regime_mask = ensemble_data['regime'] == regime
                    
                    if not regime_mask.any():
                        continue
                    
                    # Extract parameters
                    period = params['period']
                    overbought = params['overbought']
                    oversold = params['oversold']
                    exit_threshold = params['exit_threshold']
                    
                    # Calculate RSI for entire dataset
                    delta = ensemble_data[self.price_col].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    
                    avg_gain = gain.rolling(window=period).mean()
                    avg_loss = loss.rolling(window=period).mean()
                    
                    # Avoid division by zero
                    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
                    ensemble_data['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Generate signals for this regime
                    ensemble_data.loc[regime_mask & (ensemble_data['RSI'] < oversold), 'rsi_signal'] = 1
                    ensemble_data.loc[regime_mask & (ensemble_data['RSI'] > overbought), 'rsi_signal'] = -1
                    
                    # TODO: Add exit logic for RSI strategy
                
                # Calculate returns
                ensemble_data['rsi_returns'] = ensemble_data['rsi_signal'].shift(1) * ensemble_data['returns']
                
            else:
                # Use single parameter set for all regimes
                params = self.best_params['rsi']['all']
                period = params['period']
                overbought = params['overbought']
                oversold = params['oversold']
                exit_threshold = params['exit_threshold']
                
                # Calculate RSI
                delta = ensemble_data[self.price_col].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # Avoid division by zero
                rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
                ensemble_data['RSI'] = 100 - (100 / (1 + rs))
                
                # Generate signals
                ensemble_data['rsi_signal'] = 0
                ensemble_data.loc[ensemble_data['RSI'] < oversold, 'rsi_signal'] = 1
                ensemble_data.loc[ensemble_data['RSI'] > overbought, 'rsi_signal'] = -1
                
                # TODO: Add exit logic for RSI strategy
                
                # Calculate returns
                ensemble_data['rsi_returns'] = ensemble_data['rsi_signal'].shift(1) * ensemble_data['returns']
            
            # Track RSI strategy
            strategies['rsi'] = {
                'signal_col': 'rsi_signal',
                'returns_col': 'rsi_returns'
            }
        
        # Build ensemble strategy using different methods
        ensemble_methods = [
            'equal_weight',       # Equal weighting of signals
            'performance_weight', # Weight by Sharpe ratio
            'voting'              # Majority vote
        ]
        
        ensemble_results = {}
        
        for method in ensemble_methods:
            print(f"\nCreating ensemble strategy using '{method}' method")
            
            if method == 'equal_weight':
                # Equal weight: average of signals
                signal_cols = [s['signal_col'] for s in strategies.values()]
                ensemble_data['ensemble_signal'] = ensemble_data[signal_cols].mean(axis=1)
                
                # Apply thresholds to generate discrete signals
                ensemble_data['ensemble_discrete'] = 0
                ensemble_data.loc[ensemble_data['ensemble_signal'] >= 0.5, 'ensemble_discrete'] = 1
                ensemble_data.loc[ensemble_data['ensemble_signal'] <= -0.5, 'ensemble_discrete'] = -1
                
                # Calculate returns
                ensemble_data['ensemble_returns'] = ensemble_data['ensemble_discrete'].shift(1) * ensemble_data['returns']
                
            elif method == 'performance_weight':
                # Weight by Sharpe ratio on validation set
                weights = {}
                total_weight = 0
                
                for name, strategy in strategies.items():
                    returns_col = strategy['returns_col']
