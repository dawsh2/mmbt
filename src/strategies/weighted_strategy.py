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
from src.log_system import logger


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
                # Extract signal value (-1, 0, 1) and weight it
                signal_value = signal_object.signal_type.value
                weighted_signal = signal_value * self.weights[i]

                # Store weighted signal
                combined_signals.append(weighted_signal)

                # Store raw signal value for debugging
                component_signals[getattr(component, 'name', f'component_{i}')] = signal_value
            else:
                combined_signals.append(0)
                component_signals[getattr(component, 'name', f'component_{i}')] = 0

        # Calculate weighted sum
        weighted_sum = sum(combined_signals)

        # Print for debugging
        logger.info(f"Weighted sum: {weighted_sum}, Thresholds: buy={self.buy_threshold}, sell={self.sell_threshold}")

        # Determine final signal
        if weighted_sum >= self.buy_threshold:  # Changed > to >=
            final_signal_type = SignalType.BUY
        elif weighted_sum <= self.sell_threshold:  # Changed < to <=
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL

        # Set confidence based on weighted sum magnitude
        confidence = min(abs(weighted_sum), 1.0)

        # Create signal object
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id=self.name,
            confidence=confidence,  # Use the calculated confidence
            metadata={
                "weighted_sum": weighted_sum,
                "component_signals": component_signals
            }
        )

        return self.last_signal
        

    def reset(self):
        """Reset all components in the strategy."""
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()
        self.last_signal = None
