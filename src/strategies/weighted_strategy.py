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
from src.log_system import TradeLogger


# logger = TradeLogger.get_logger('trading.strategy')


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
        
    def generate_signals(self, bar, bar_event=None):
        """
        Generate weighted trading signals.

        Args:
            bar: Dictionary containing bar data
            bar_event: Original bar event (optional)

        Returns:
            Signal object representing the weighted decision
        """
        # Get signals from all components
        component_signals = []
        
        # We pass the original bar_event to maintain the interface
        for component in self.components:
            signal = component.on_bar(bar_event)
            if signal is not None:
                component_signals.append(signal)

        if not component_signals:
            # No signals generated, return neutral
            return Signal(
                timestamp=bar.get('timestamp', datetime.now()),
                signal_type=SignalType.NEUTRAL,
                price=bar.get('Close', None),
                rule_id=self.name,
                confidence=0.0,
                metadata={'component_count': 0}
            )

        # Calculate weighted signal
        weighted_sum = 0.0
        total_weight = sum(self.weights)
        signal_weights = {}

        for i, signal in enumerate(component_signals):
            # Get weight for this component
            weight = self.weights[i] if i < len(self.weights) else 1.0/len(component_signals)

            # Get direction from signal_type
            direction = signal.signal_type.value

            # Apply weight
            weighted_sum += direction * weight * signal.confidence

            # Store for metadata
            signal_weights[signal.rule_id] = {
                'weight': weight,
                'direction': direction,
                'confidence': signal.confidence,
                'contribution': direction * weight * signal.confidence
            }

        # Normalize weighted sum
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0

        # Determine final signal type based on weighted sum
        if normalized_sum > self.buy_threshold:
            signal_type = SignalType.BUY
        elif normalized_sum < self.sell_threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL

        # Create the combined signal
        return Signal(
            timestamp=bar.get('timestamp', datetime.now()),
            signal_type=signal_type,
            price=bar.get('Close', None),
            rule_id=self.name,
            confidence=abs(normalized_sum),
            metadata={
                'weighted_sum': normalized_sum,
                'component_signals': signal_weights,
                'component_count': len(component_signals)
            }
        )

    def reset(self):
        """Reset all components in the strategy."""
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()
        self.last_signal = None
