"""
Strategy implementations for the optimization framework.
"""

import numpy as np
from signals import Signal, SignalType

class WeightedComponentStrategy:
    """
    Strategy that combines signals from multiple components using weights.
    
    This is a generalized version of the original WeightedRuleStrategy.
    """
    
    def __init__(self, components, weights, buy_threshold=0.5, sell_threshold=-0.5):
        """
        Initialize the weighted strategy.
        
        Args:
            components: List of component objects that generate signals
            weights: List of weights for each component
            buy_threshold: Threshold above which to generate a buy signal
            sell_threshold: Threshold below which to generate a sell signal
        """
        self.components = components
        self.weights = np.array(weights)
        self.component_signals = [None] * len(components)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def on_bar(self, event):
        """
        Process a bar event and generate a weighted signal.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            Signal: Combined signal based on weighted components
        """
        bar = event.bar
        combined_signals = []
        
        for i, component in enumerate(self.components):
            signal_object = component.on_bar(bar)
            if signal_object and hasattr(signal_object, 'signal_type'):
                combined_signals.append(signal_object.signal_type.value * self.weights[i])
            else:
                combined_signals.append(0)
        
        weighted_sum = np.sum(combined_signals)
        
        if weighted_sum > self.buy_threshold:
            final_signal_type = SignalType.BUY
        elif weighted_sum < self.sell_threshold:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL
        
        return Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id="weighted_strategy"
        )
    
    def reset(self):
        """Reset all components."""
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()
        self.component_signals = [None] * len(self.components)
