"""
Weighted Strategy Module

This module provides the WeightedStrategy class that combines signals from multiple
rules using configurable weights.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from .strategy_base import Strategy
from .strategy_registry import StrategyRegistry
from signals import Signal, SignalType

@StrategyRegistry.register(category="core")
class WeightedStrategy(Strategy):
    """Strategy that combines signals from multiple rules using weights.
    
    This strategy takes a list of rule objects and combines their signals
    using configurable weights to generate a final trading signal.
    """
    
    def __init__(self, 
                 rules: List[Any], 
                 weights: Optional[np.ndarray] = None, 
                 buy_threshold: float = 0.5, 
                 sell_threshold: float = -0.5, 
                 name: Optional[str] = None):
        """Initialize the weighted strategy.
        
        Args:
            rules: List of rule objects
            weights: List of weights for each rule (default: equal weights)
            buy_threshold: Threshold above which to generate a buy signal
            sell_threshold: Threshold below which to generate a sell signal
            name: Strategy name
        """
        super().__init__(name or "WeightedStrategy")
        self.rules = rules
        
        # Initialize weights (equal by default)
        if weights is None:
            self.weights = np.ones(len(rules)) / len(rules)
        else:
            # Normalize weights to sum to 1
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                self.weights = np.array(weights) / weights_sum
            else:
                # Fallback to equal weights if sum is 0 or negative
                self.weights = np.ones(len(rules)) / len(rules)
        
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_signal = None
    
    def on_bar(self, event):
        """Process a bar and generate a weighted signal.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            Signal: Combined signal based on weighted rules
        """
        bar = event.bar
        
        # Get signals from all rules
        combined_signals = []
        rule_signals = {}  # For metadata
        
        for i, rule in enumerate(self.rules):
            signal_object = rule.on_bar(bar)
            
            if signal_object and hasattr(signal_object, 'signal_type'):
                signal_value = signal_object.signal_type.value
                combined_signals.append(signal_value * self.weights[i])
                rule_signals[getattr(rule, 'name', f'rule_{i}')] = signal_value
            else:
                combined_signals.append(0)
                rule_signals[getattr(rule, 'name', f'rule_{i}')] = 0
        
        # Calculate weighted sum
        weighted_sum = np.sum(combined_signals)
        
        # Determine final signal
        if weighted_sum > self.buy_threshold:
            final_signal_type = SignalType.BUY
        elif weighted_sum < self.sell_threshold:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL
        
        # Create signal object
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id=self.name,
            confidence=min(1.0, abs(weighted_sum)),  # Scale confidence
            metadata={
                "weighted_sum": weighted_sum,
                "rule_signals": rule_signals
            }
        )
        
        return self.last_signal
    
    def reset(self):
        """Reset all rules in the strategy."""
        for rule in self.rules:
            if hasattr(rule, 'reset'):
                rule.reset()
        self.last_signal = None
