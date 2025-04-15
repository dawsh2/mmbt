from src.rules.rule_base import Rule

class CanaryRule(Rule):
    """
    A special test rule that produces predictable results based on its parameters.
    Used to test optimizer parameter handling.
    """
    
    @classmethod
    def default_params(cls):
        return {
            'multiplier': 1.0,
            'threshold': 0.5,
            'buy_period': 5
        }
    
    def _validate_params(self):
        if self.params['multiplier'] <= 0:
            raise ValueError("Multiplier must be positive")
    
    def __init__(self, name="canary", params=None, description=""):
        super().__init__(name or "canary_rule", params or self.default_params(), 
                        description or "Special rule for testing optimizer parameter handling")
        self.bar_count = 0
        self.params_fingerprint = f"{self.params['multiplier']}-{self.params['threshold']}-{self.params['buy_period']}"
        self.signals_generated = 0
        print(f"Created CanaryRule instance with params fingerprint: {self.params_fingerprint}")
    
    def reset(self):
        print(f"Reset called on CanaryRule with params fingerprint: {self.params_fingerprint}")
        super().reset()
        self.bar_count = 0
        self.signals_generated = 0
    
    def generate_signal(self, data):
        """Generate predictable signals based on parameters."""
        self.bar_count += 1
        
        # Use parameters to create predictable but different behavior
        # Parameter values will directly affect returns in a predictable way
        if self.bar_count % self.params['buy_period'] == 0:
            self.signals_generated += 1
            
            # Decide signal type based on parameters
            if data['Close'] * self.params['multiplier'] > data['Open'] + self.params['threshold']:
                return Signal(
                    timestamp=data.get('timestamp', ''),
                    signal_type=SignalType.BUY,
                    price=data.get('Close', 0),
                    rule_id=self.name,
                    confidence=self.params['threshold'],
                    metadata={
                        'params_fingerprint': self.params_fingerprint,
                        'bar_count': self.bar_count
                    }
                )
            else:
                return Signal(
                    timestamp=data.get('timestamp', ''),
                    signal_type=SignalType.SELL,
                    price=data.get('Close', 0),
                    rule_id=self.name,
                    confidence=self.params['threshold'],
                    metadata={
                        'params_fingerprint': self.params_fingerprint,
                        'bar_count': self.bar_count
                    }
                )
                
        return None
