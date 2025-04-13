# tests/test_rules.py
import unittest
from unittest.mock import MagicMock
from src.features.feature_base import Feature
from src.rules.rule_base import Rule
from src.rules.rule_registry import RuleRegistry
from signals import SignalType

# Define a mock feature for testing
class MockFeature(Feature):
    def calculate(self, bar_data, history=None):
        self.value = 0.75  # Positive value for testing
        return self.value

# Define a mock rule for testing
class MockRule(Rule):
    def _setup_features(self):
        self.features = [MockFeature(name="TestFeature")]
    
    def generate_signal(self, feature_values):
        value = feature_values["TestFeature"]
        if value > 0.5:
            return SignalType.BUY
        elif value < -0.5:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

class TestRules(unittest.TestCase):
    """Test suite for rules."""
    
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
    
    def test_rule_base(self):
        """Test basic rule functionality."""
        # Register and create a mock rule
        RuleRegistry.register(MockRule)
        rule = RuleRegistry.get_rule("MockRule")
        
        # Test on_bar method
        signal = rule.on_bar(self.bar)
        
        # Verify signal properties
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.price, 10.5)
        self.assertEqual(signal.rule_id, "MockRule")
        
        # Test rule reset
        rule.reset()
        self.assertEqual(len(rule.history), 0)
    
    def test_rule_registry(self):
        """Test rule registry functionality."""
        # Register the mock rule
        RuleRegistry.register(MockRule)
        
        # Test rule lookup
        rule_names = RuleRegistry.list_rules()
        self.assertIn("MockRule", rule_names)
        
        # Test rule instantiation
        rule = RuleRegistry.get_rule("MockRule")
        self.assertIsInstance(rule, MockRule)
