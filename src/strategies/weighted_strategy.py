"""
Weighted Strategy Module

This module provides the WeightedStrategy class that combines signals from multiple
components using configurable weights.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry
from signals import Signal, SignalType

@StrategyRegistry.register(category="core")
class WeightedStrategy(Strategy):
    """Strategy that combines signals from multiple components using weights.
    
    This strategy takes a list of components (rules or other signal generators)
    and combines their signals using configurable weights to generate a final 
    trading signal.
    """
    
    def __init__(self, 
                 components: List[Any],  # Renamed from 'rules' to 'components'
                 weights: Optional[np.ndarray] = None, 
                 buy_threshold: float = 0.5, 
                 sell_threshold: float = -0.5, 
                 name: Optional[str] = None):
        """Initialize the weighted strategy.
        
        Args:
            components: List of components that generate signals
            weights: List of weights for each component (default: equal weights)
            buy_threshold: Threshold above which to generate a buy signal
            sell_threshold: Threshold below which to generate a sell signal
            name: Strategy name
        """
        super().__init__(name or "WeightedStrategy")
        self.components = components  # Renamed from 'rules' to 'components'
        
        # Initialize weights (equal by default)
        if weights is None:
            self.weights = np.ones(len(components)) / len(components)
        else:
            # Normalize weights to sum to 1
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                self.weights = np.array(weights) / weights_sum
            else:
                # Fallback to equal weights if sum is 0 or negative
                self.weights = np.ones(len(components)) / len(components)
        
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.last_signal = None

    def on_bar(self, event):
        bar = event.bar

        # Get signals from all components
        combined_signals = []
        component_signals = {}  # For metadata

        for i, component in enumerate(self.components):
            signal_object = component.on_bar(bar)

            if signal_object and hasattr(signal_object, 'signal_type'):
                # Don't modify by weight for SELL and BUY signals
                if signal_object.signal_type == SignalType.SELL:
                    combined_signals.append(-1.0 * self.weights[i])  # Fixed value for SELL
                elif signal_object.signal_type == SignalType.BUY:
                    combined_signals.append(1.0 * self.weights[i])   # Fixed value for BUY
                else:
                    combined_signals.append(0)  # NEUTRAL

                component_signals[getattr(component, 'name', f'component_{i}')] = signal_object.signal_type.value
            else:
                combined_signals.append(0)
                component_signals[getattr(component, 'name', f'component_{i}')] = 0

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
                "component_signals": component_signals  # Renamed from rule_signals
            }
        )
        
        return self.last_signal
    
    def reset(self):
        """Reset all components in the strategy."""
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()
        self.last_signal = None
