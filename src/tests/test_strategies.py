# tests/test_strategies.py
import unittest
from unittest.mock import MagicMock, patch
from src.strategies.strategy_base import Strategy
from src.rules.rule_base import Rule
from signals import Signal, SignalType

# Mock event for testing
class MockEvent:
    def __init__(self, bar):
        self.bar = bar

# Mock concrete strategy for testing
class MockStrategy(Strategy):
    def __init__(self, rules=None, name=None):
        super().__init__(name)
        self.rules = rules or []
    
    def on_bar(self, event):
        # Simple implementation that returns first rule's signal
        if self.rules:
            return self.rules[0].on_bar(event.bar)
        
        # Default signal
        return Signal(
            timestamp=event.bar["timestamp"],
            signal_type=SignalType.NEUTRAL,
            price=event.bar["Close"],
            rule_id=self.name
        )
    
    def reset(self):
        for rule in self.rules:
            rule.reset()

class TestStrategies(unittest.TestCase):
    """Test suite for strategies."""
    
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
        
        # Create mock event
        self.event = MockEvent(self.bar)
        
        # Create mock rule
        self.mock_rule = MagicMock(spec=Rule)
        self.mock_rule.on_bar.return_value = Signal(
            timestamp=self.bar["timestamp"],
            signal_type=SignalType.BUY,
            price=self.bar["Close"],
            rule_id="MockRule"
        )
    
    def test_strategy_base(self):
        """Test basic strategy functionality."""
        # Create strategy with mock rule
        strategy = MockStrategy(rules=[self.mock_rule], name="TestStrategy")
        
        # Test name setting
        self.assertEqual(strategy.name, "TestStrategy")
        
        # Test on_bar method
        signal = strategy.on_bar(self.event)
        
        # Verify signal properties
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.price, 10.5)
        
        # Verify rule was called
        self.mock_rule.on_bar.assert_called_once_with(self.bar)
        
        # Test reset
        strategy.reset()
        self.mock_rule.reset.assert_called_once()
