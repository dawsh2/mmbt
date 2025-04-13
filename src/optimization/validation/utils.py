"""
Utility functions and classes for validation components.
"""

import pandas as pd
import numpy as np

class WindowDataHandler:
    """Data handler for a specific window or fold."""
    
    def __init__(self, train_data, test_data):
        """
        Initialize the window data handler.
        
        Args:
            train_data: Training data for this window
            test_data: Testing data for this window
        """
        self.train_df = pd.DataFrame(train_data) if isinstance(train_data, list) else train_data
        self.test_df = pd.DataFrame(test_data) if isinstance(test_data, list) else test_data
        self.current_train_index = 0
        self.current_test_index = 0
    
    def get_next_train_bar(self):
        """Get the next training bar."""
        if self.current_train_index < len(self.train_df):
            bar = self.train_df.iloc[self.current_train_index].to_dict()
            self.current_train_index += 1
            return bar
        return None
    
    def get_next_test_bar(self):
        """Get the next testing bar."""
        if self.current_test_index < len(self.test_df):
            bar = self.test_df.iloc[self.current_test_index].to_dict()
            self.current_test_index += 1
            return bar
        return None
    
    def reset_train(self):
        """Reset the training data pointer."""
        self.current_train_index = 0
    
    def reset_test(self):
        """Reset the testing data pointer."""
        self.current_test_index = 0

def create_train_test_windows(data, window_size, step_size, train_pct=0.7):
    """
    Create training and testing windows for walk-forward validation.
    
    Args:
        data: Full dataset
        window_size: Size of each window
        step_size: Number of steps to roll forward between windows
        train_pct: Percentage of window to use for training
        
    Returns:
        list: List of (train_window, test_window) tuples
    """
    windows = []
    
    # Calculate the number of windows
    total_length = len(data)
    num_windows = (total_length - window_size) // step_size + 1
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Skip if window exceeds data length
        if end_idx > total_length:
            break
            
        window_data = data[start_idx:end_idx]
        
        # Split into training and testing
        train_size = int(window_size * train_pct)
        train_window = window_data[:train_size]
        test_window = window_data[train_size:end_idx]
        
        windows.append((train_window, test_window))
        
    return windows
