"""
Data handling module for the backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from config import config

class DataHandler:
    """Data handler class for loading and preprocessing data."""
    
    def __init__(self, config):
        """Initialize data handler with configuration."""
        self.config = config
        self.data = None
        self.train_data = None
        self.test_data = None


    def load_data(self, file_path=None):
        """Load data from CSV file."""
        if file_path is None:
            file_path = self.config.data_file

        try:
            # Load data from CSV
            self.data = pd.read_csv(file_path)

            # Ensure required OHLC columns exist
            required_columns = ['Open', 'High', 'Low', 'Close']
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Required column '{col}' not found in data file")

            # Handle date column - try to find it with common names
            date_column_candidates = ['Date', 'date', 'datetime', 'time', 'timestamp']
            date_column = None

            for candidate in date_column_candidates:
                if candidate in self.data.columns:
                    date_column = candidate
                    break

            # If no date column found, create one from the index
            if date_column is None:
                print("No date column found. Using row index as date.")
                self.data['Date'] = pd.date_range(start='2000-01-01', periods=len(self.data))
                date_column = 'Date'

            # Convert Date to datetime if it's not already
            if pd.api.types.is_string_dtype(self.data[date_column]):
                self.data[date_column] = pd.to_datetime(self.data[date_column])

            # Set Date as index
            self.data.set_index(date_column, inplace=True)

            # Sort by date
            self.data.sort_index(inplace=True)

            # Calculate log returns
            self.data['LogReturn'] = np.log(self.data['Close'] / self.data['Close'].shift(1))

            print(f"Loaded {len(self.data)} rows of data from {file_path}")
            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False        


        # In data.py, update the preprocess method
    def preprocess(self):
        """Preprocess the data for backtesting."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return False

        # Print statistics before removing NaNs
        print(f"Data before preprocessing: {len(self.data)} rows")

        # Check each column for NaN values
        for col in self.data.columns:
            nan_count = self.data[col].isna().sum()
            print(f"NaN values in {col}: {nan_count} ({nan_count/len(self.data)*100:.2f}%)")

        # Look for duplicate indices
        duplicate_count = self.data.index.duplicated().sum()
        print(f"Duplicate indices: {duplicate_count}")

        # Remove rows with NaN values
        before_count = len(self.data)
        self.data.dropna(inplace=True)
        after_count = len(self.data)

        print(f"Rows removed due to NaNs: {before_count - after_count} ({(before_count - after_count)/before_count*100:.2f}%)")
        print(f"Data after preprocessing: {len(self.data)} rows")

        return True



    def split_data(self, train_size=None):
        """Split data into training and testing sets."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return False

        if train_size is None:
            train_size = self.config.train_size

        # Make sure we're using clean data
        clean_data = self.data.dropna()
        print(f"Clean data after removing NaNs: {len(clean_data)} rows")

        # Calculate split point
        split_idx = int(len(clean_data) * train_size)

        # Split data
        self.train_data = clean_data.iloc[:split_idx].copy()
        self.test_data = clean_data.iloc[split_idx:].copy()

        print(f"Data split: {len(self.train_data)} rows for training, {len(self.test_data)} rows for testing")
        return True

    
    def get_ohlc(self, train=True):
        """Get OHLC data for either training or testing set."""
        data = self.train_data if train else self.test_data
        if data is None:
            print("Data not split. Call split_data() first.")
            return None
        
        return [data['Open'], data['High'], data['Low'], data['Close']]
    
    def get_shifts(self, data, lookback_periods):
        """Apply lookback period shifts to avoid lookahead bias.
        
        Args:
            data: DataFrame to shift
            lookback_periods: Dict of {column_name: periods} defining shifts
        
        Returns:
            DataFrame with shifted values
        """
        shifted_data = data.copy()
        
        for col, periods in lookback_periods.items():
            if col in shifted_data.columns:
                shifted_data[col] = shifted_data[col].shift(periods)
        
        # Remove rows with NaN values after shifting
        shifted_data.dropna(inplace=True)
        
        return shifted_data
    
    def save_processed_data(self, file_path):
        """Save processed data to a CSV file."""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return False
        
        try:
            self.data.to_csv(file_path)
            print(f"Saved processed data to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
