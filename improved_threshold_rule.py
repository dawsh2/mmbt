class ImprovedThresholdRule:
    """
    A rule that generates buy signals when price is above threshold
    and sell signals when price is below threshold - with no neutral signals.
    
    This matches the behavior expected by the test_data_generator.
    """
    def __init__(self, params=None):
        self.threshold = params.get('threshold', 100.0) if params else 100.0
        self.rule_id = "ImprovedThresholdRule"
        
    def on_bar(self, bar):
        """Generate signals based on price relative to threshold."""
        # Extract price
        if isinstance(bar, dict):
            close = bar['Close']
        elif hasattr(bar, 'Close'):
            close = bar.Close
        elif hasattr(bar, 'bar') and isinstance(bar.bar, dict):
            close = bar.bar['Close']
        else:
            print(f"Warning: Unrecognized bar format: {type(bar)}")
            return 0
            
        # Generate signal: always 1 (buy) when above threshold, -1 (sell) when below
        # This matches the test_data_generator's expectation
        signal_value = 1 if close > self.threshold else -1
        
        print(f"[Rule] Bar close: {close:.2f}, Threshold: {self.threshold}, Signal: {signal_value}")
        return signal_value
        
    def reset(self):
        """Reset rule state."""
        pass
