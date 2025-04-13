"""
Test Threshold Rule for backtesting validation.

This rule generates simple, predictable signals:
- BUY when price is above threshold (default 100.0)
- SELL when price is below threshold
"""

from signals import Signal, SignalType

class TestThresholdRule:
    """
    Test rule that generates BUY signals when price is above threshold
    and SELL signals when price is below threshold.
    """
    def __init__(self, params=None):
        # Default threshold is 100.0 if not specified
        self.threshold = params.get('threshold', 100.0) if params else 100.0
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "TestThresholdRule"
    
    def on_bar(self, bar):
        """Process a bar and generate a signal based on threshold comparison."""
        close = bar['Close']
        
        if close > self.threshold:
            self.current_signal_type = SignalType.BUY
        else:
            self.current_signal_type = SignalType.SELL
            
        # Create and return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={"threshold": self.threshold}
        )
    
    def reset(self):
        """Reset the rule's state."""
        self.current_signal_type = SignalType.NEUTRAL

if __name__ == "__main__":
    # Example usage
    rule = TestThresholdRule({"threshold": 100.0})
    
    # Test with a sample bar
    test_bar = {
        "timestamp": "2023-01-01",
        "Open": 99.5,
        "High": 101.2,
        "Low": 98.7,
        "Close": 101.0,
        "Volume": 1000
    }
    
    signal = rule.on_bar(test_bar)
    print(f"Test bar price: {test_bar['Close']}")
    print(f"Threshold: {rule.threshold}")
    print(f"Signal: {signal.signal_type.name}")
