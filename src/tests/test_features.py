# tests/test_features.py
import unittest
import numpy as np
from src.features.feature_base import Feature
from src.features.feature_registry import FeatureRegistry
from src.features.price_features import PriceToSMA  # Assuming this is implemented

class MockFeature(Feature):
    """Mock feature for testing."""
    
    def calculate(self, bar_data, history=None):
        """Return a fixed value for testing."""
        self.value = 0.5
        return self.value

class TestFeatures(unittest.TestCase):
    """Test suite for features."""
    
    def setUp(self):
        """Set up test data."""
        # Create test bar data
        self.bar = {
            'timestamp': '2023-01-01 00:00:00',
            'Open': 10.0,
            'High': 11.0,
            'Low': 9.0,
            'Close': 10.5,
            'Volume': 1000
        }
        
        # Create test history
        self.history = [
            {'timestamp': '2022-12-31 00:00:00', 'Close': 10.0},
            {'timestamp': '2022-12-30 00:00:00', 'Close': 9.5},
            {'timestamp': '2022-12-29 00:00:00', 'Close': 9.0},
            {'timestamp': '2022-12-28 00:00:00', 'Close': 8.5},
            {'timestamp': '2022-12-27 00:00:00', 'Close': 8.0},
        ]
    
    def test_feature_base(self):
        """Test basic feature functionality."""
        feature = MockFeature(name="TestFeature")
        
        # Test name setting
        self.assertEqual(feature.name, "TestFeature")
        
        # Test calculation
        value = feature.calculate(self.bar)
        self.assertEqual(value, 0.5)
        
        # Test reset
        feature.reset()
        self.assertIsNone(feature.value)
    
    def test_feature_registry(self):
        """Test feature registry functionality."""
        # Register the mock feature
        FeatureRegistry.register(MockFeature)
        
        # Test feature lookup
        feature_names = FeatureRegistry.list_features()
        self.assertIn("MockFeature", feature_names)
        
        # Test feature instantiation
        feature = FeatureRegistry.get_feature("MockFeature", name="TestFeature")
        self.assertIsInstance(feature, MockFeature)
        self.assertEqual(feature.name, "TestFeature")
