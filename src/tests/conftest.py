# tests/conftest.py
"""
Configuration for pytest.

This file contains fixtures and other setup for pytest.
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    # Create a DataFrame with some price data
    dates = pd.date_range('2023-01-01', periods=100)
    prices = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    
    df = pd.DataFrame({
        'Open': prices - np.random.uniform(0, 2, 100),
        'High': prices + np.random.uniform(0, 2, 100),
        'Low': prices - np.random.uniform(0, 2, 100),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return df

@pytest.fixture
def sample_bar_data():
    """Create sample bar data for testing."""
    # Create a list of bar dictionaries
    bars = []
    dates = pd.date_range('2023-01-01', periods=100)
    prices = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    
    for i, date in enumerate(dates):
        bar = {
            'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
            'Open': prices[i] - np.random.uniform(0, 2),
            'High': prices[i] + np.random.uniform(0, 2),
            'Low': prices[i] - np.random.uniform(0, 2),
            'Close': prices[i],
            'Volume': np.random.randint(1000, 10000)
        }
        bars.append(bar)
    
    return bars
