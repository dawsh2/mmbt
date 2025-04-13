from signals import Signal, SignalType
from collections import deque

class ThresholdRule:
    """
    Simple Threshold Rule:
    - BUY when price > upper_threshold
    - SELL when price < lower_threshold
    - NEUTRAL when price is between thresholds
    """
    def __init__(self, params=None):
        params = params or {}
        self.upper_threshold = params.get('upper_threshold', 110.0)
        self.lower_threshold = params.get('lower_threshold', 90.0)
        self.current_signal_type = SignalType.NEUTRAL
        self.rule_id = "ThresholdRule"
        
    def on_bar(self, bar):
        """Process a bar and generate a signal based on threshold comparison."""
        close = bar['Close']
        
        if close > self.upper_threshold:
            self.current_signal_type = SignalType.BUY
        elif close < self.lower_threshold:
            self.current_signal_type = SignalType.SELL
        else:
            self.current_signal_type = SignalType.NEUTRAL
            
        # Create and return a Signal object
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=self.current_signal_type,
            price=bar["Close"],
            rule_id=self.rule_id,
            confidence=1.0,
            metadata={
                "upper_threshold": self.upper_threshold,
                "lower_threshold": self.lower_threshold
            }
        )
    
    def reset(self):
        """Reset the rule's state."""
        self.current_signal_type = SignalType.NEUTRAL
