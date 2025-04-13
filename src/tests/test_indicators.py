# tests/test_indicators.py
import unittest
import numpy as np
import pandas as pd
from src.indicators.moving_averages import (
    simple_moving_average,
    exponential_moving_average,
    double_exponential_moving_average,
    triple_exponential_moving_average
)

class TestMovingAverages(unittest.TestCase):
    """Test suite for moving average indicators."""
    
    def setUp(self):
        """Set up test data."""
        # Create test price data
        self.prices = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        
    def test_simple_moving_average(self):
        """Test simple moving average calculation."""
        # Calculate SMA with window 3
        result = simple_moving_average(self.prices, 3)
        
        # Expected result: [11, 12, 13, 14, 15, 16, 17, 18, 19]
        expected = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])
        
        # Assert equal with small tolerance for floating point differences
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
    def test_exponential_moving_average(self):
        """Test exponential moving average calculation."""
        # Calculate EMA with span 3
        result = exponential_moving_average(self.prices, 3)
        
        # Expected result based on pandas EMA calculation
        df = pd.DataFrame({'price': self.prices})
        expected = df['price'].ewm(span=3, adjust=False).mean().values
        
        # Assert equal with small tolerance for floating point differences
        np.testing.assert_allclose(result, expected, rtol=1e-5)
