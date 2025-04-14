"""
Data Transformers Module

This module provides transformers for preprocessing market data before feeding
it to strategies and indicators. Transformers can handle operations like:
- Resampling to different timeframes
- Adjusting for splits and dividends
- Filling missing values
- Normalizing data
- Feature engineering
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import datetime


class DataTransformer(ABC):
    """Base class for all data transformers."""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass


class ResampleTransformer(DataTransformer):
    """Transformer for resampling time series data to different frequencies."""
    
    def __init__(self, timeframe: str = '1h', aggregation: Optional[Dict[str, str]] = None):
        """
        Initialize resampler.
        
        Args:
            timeframe: Target timeframe/frequency (e.g., '1min', '5min', '1H', '1D')
            aggregation: Custom aggregation rules. Default uses OHLCV rules.
        """
        self.timeframe = timeframe
        
        # Default aggregation for OHLCV data
        self.aggregation = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Update with custom aggregation if provided
        if aggregation:
            self.aggregation.update(aggregation)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to target timeframe.
        
        Args:
            data: Input DataFrame with timestamp index or column
            
        Returns:
            Resampled DataFrame
        """
        df = data.copy()
        
        # Ensure data has a datetime index
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index or 'timestamp' column")
        
        # Resample data
        resampled = df.resample(self.timeframe)
        
        # Apply aggregation rules to columns that exist in the data
        agg_dict = {col: rule for col, rule in self.aggregation.items() if col in df.columns}
        result = resampled.agg(agg_dict)
        
        # Reset index to keep timestamp as a column
        result = result.reset_index()
        
        return result


class MissingValueHandler(DataTransformer):
    """Transformer for handling missing values in data."""
    
    def __init__(self, method: str = 'ffill', columns: Optional[List[str]] = None):
        """
        Initialize missing value handler.
        
        Args:
            method: Method for handling missing values ('ffill', 'bfill', 'zero', 'mean', 'median')
            columns: Specific columns to apply the handling to (None for all columns)
        """
        self.method = method
        self.columns = columns
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = data.copy()
        
        # Determine columns to process
        cols = self.columns if self.columns else df.columns
        
        # Apply the specified method
        if self.method == 'ffill':
            df[cols] = df[cols].ffill()
        elif self.method == 'bfill':
            df[cols] = df[cols].bfill()
        elif self.method == 'zero':
            df[cols] = df[cols].fillna(0)
        elif self.method == 'mean':
            for col in cols:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].mean())
        elif self.method == 'median':
            for col in cols:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
        else:
            raise ValueError(f"Unsupported method: {self.method}")
            
        return df


class AdjustedCloseTransformer(DataTransformer):
    """Transformer for adjusting OHLC data using Adjusted Close."""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust OHLC data using Adjusted Close.
        
        Args:
            data: Input DataFrame with OHLC and Adj_Close columns
            
        Returns:
            DataFrame with adjusted OHLC values
        """
        df = data.copy()
        
        if 'Adj_Close' not in df.columns or 'Close' not in df.columns:
            raise ValueError("Data must have both 'Adj_Close' and 'Close' columns")
            
        # Calculate the adjustment ratio
        df['Ratio'] = df['Adj_Close'] / df['Close']
        
        # Adjust OHLC values
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = df[col] * df['Ratio']
                
        # Remove the adjustment ratio column
        df = df.drop('Ratio', axis=1)
        
        return df


class ReturnCalculator(DataTransformer):
    """Transformer for calculating returns from price data."""
    
    def __init__(self, periods: List[int] = [1], price_col: str = 'Close', 
                log_returns: bool = False):
        """
        Initialize return calculator.
        
        Args:
            periods: List of periods to calculate returns for
            price_col: Column to use for price data
            log_returns: Whether to calculate log returns (True) or simple returns (False)
        """
        self.periods = periods
        self.price_col = price_col
        self.log_returns = log_returns
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and add as new columns.
        
        Args:
            data: Input DataFrame with price data
            
        Returns:
            DataFrame with additional return columns
        """
        df = data.copy()
        
        if self.price_col not in df.columns:
            raise ValueError(f"Price column '{self.price_col}' not found in data")
            
        # Calculate returns for each period
        for period in self.periods:
            if self.log_returns:
                df[f'return_{period}'] = np.log(df[self.price_col] / df[self.price_col].shift(period))
            else:
                df[f'return_{period}'] = df[self.price_col].pct_change(period)
                
        return df


class NormalizationTransformer(DataTransformer):
    """Transformer for normalizing price data."""
    
    def __init__(self, method: str = 'z-score', window: int = 20, columns: Optional[List[str]] = None):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('z-score', 'min-max', 'decimal-scaling')
            window: Window size for rolling normalization (0 for full series)
            columns: Columns to normalize (None for all numeric columns)
        """
        self.method = method
        self.window = window
        self.columns = columns
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        df = data.copy()
        
        # Determine columns to process
        cols = self.columns if self.columns else df.select_dtypes(include=[np.number]).columns
        
        for col in cols:
            if col not in df.columns:
                continue
                
            if self.method == 'z-score':
                if self.window > 0:
                    # Rolling z-score normalization
                    mean = df[col].rolling(window=self.window).mean()
                    std = df[col].rolling(window=self.window).std()
                    df[f'{col}_norm'] = (df[col] - mean) / std
                else:
                    # Full series z-score normalization
                    df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
                    
            elif self.method == 'min-max':
                if self.window > 0:
                    # Rolling min-max normalization
                    min_val = df[col].rolling(window=self.window).min()
                    max_val = df[col].rolling(window=self.window).max()
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    # Full series min-max normalization
                    df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    
            elif self.method == 'decimal-scaling':
                # Decimal scaling normalization
                max_abs = df[col].abs().max()
                digits = len(str(int(max_abs)))
                df[f'{col}_norm'] = df[col] / (10 ** digits)
                
            else:
                raise ValueError(f"Unsupported normalization method: {self.method}")
                
        return df


class FeatureEngineeringTransformer(DataTransformer):
    """Transformer for engineering common technical features."""
    
    def __init__(self, features: List[str], params: Optional[Dict[str, Any]] = None):
        """
        Initialize feature engineer.
        
        Args:
            features: List of features to engineer ('ma', 'ema', 'rsi', 'bbands', etc.)
            params: Dictionary of parameters for features
        """
        self.features = features
        self.params = params or {}
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional feature columns
        """
        df = data.copy()
        
        for feature in self.features:
            if feature == 'ma':
                periods = self.params.get('ma_periods', [10, 20, 50, 200])
                for period in periods:
                    df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
                    
            elif feature == 'ema':
                periods = self.params.get('ema_periods', [12, 26, 50])
                for period in periods:
                    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                    
            elif feature == 'rsi':
                period = self.params.get('rsi_period', 14)
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
            elif feature == 'bbands':
                period = self.params.get('bb_period', 20)
                std_dev = self.params.get('bb_std_dev', 2)
                
                df['BB_Middle'] = df['Close'].rolling(window=period).mean()
                df['BB_Std'] = df['Close'].rolling(window=period).std()
                df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * std_dev)
                df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * std_dev)
                
            elif feature == 'macd':
                fast_period = self.params.get('macd_fast_period', 12)
                slow_period = self.params.get('macd_slow_period', 26)
                signal_period = self.params.get('macd_signal_period', 9)
                
                df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
                df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
                df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
                df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                
            elif feature == 'atr':
                period = self.params.get('atr_period', 14)
                
                df['High_Low'] = df['High'] - df['Low']
                df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
                df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
                
                df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
                df['ATR'] = df['TR'].rolling(window=period).mean()
                
                # Clean up temporary columns
                df = df.drop(['High_Low', 'High_Close', 'Low_Close', 'TR'], axis=1)
                
            elif feature == 'vwap':
                df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
                
            elif feature == 'returns':
                periods = self.params.get('return_periods', [1, 5, 10, 20])
                for period in periods:
                    df[f'Return_{period}'] = df['Close'].pct_change(period)
                
            else:
                print(f"Warning: Unknown feature type '{feature}'")
                
        return df


class TransformerPipeline(DataTransformer):
    """Pipeline for applying multiple transformers in sequence."""
    
    def __init__(self, transformers: List[DataTransformer]):
        """
        Initialize transformer pipeline.
        
        Args:
            transformers: List of transformers to apply in sequence
        """
        self.transformers = transformers
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers in sequence.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        
        for transformer in self.transformers:
            result = transformer.transform(result)
            
        return result


# Example usage
if __name__ == "__main__":
    from data_sources import CSVDataSource
    
    # Create CSV data source
    csv_source = CSVDataSource("data/csv")
    
    # Get data for a symbol
    symbols = csv_source.get_symbols()
    if symbols:
        symbol = symbols[0]
        data = csv_source.get_data(symbol)
        
        # Create a transformer pipeline
        pipeline = TransformerPipeline([
            MissingValueHandler(method='ffill'),
            ResampleTransformer(timeframe='1h'),
            FeatureEngineeringTransformer(
                features=['ma', 'rsi', 'bbands'],
                params={'ma_periods': [10, 20, 50]}
            ),
            ReturnCalculator(periods=[1, 5, 10])
        ])
        
        # Transform data
        transformed_data = pipeline.transform(data)
        
        # Print results
        print(f"Original data shape: {data.shape}")
        print(f"Transformed data shape: {transformed_data.shape}")
        print(f"New columns: {[col for col in transformed_data.columns if col not in data.columns]}")
