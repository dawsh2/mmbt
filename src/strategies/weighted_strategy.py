"""
Weighted Strategy Module

This module provides the WeightedStrategy class that combines signals from multiple
components using configurable weights.
"""

import numpy as np
import datetime
from typing import List, Optional, Dict, Any, Union
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry
from src.events.event_types import BarEvent
from src.events.signal_event import SignalEvent

# Configure logging
import logging
logger = logging.getLogger(__name__)


@StrategyRegistry.register(category="core")
class WeightedStrategy(Strategy):
    """Strategy that combines signals from multiple components using weights.
    
    This strategy takes a list of components (rules or other signal generators)
    and combines their signals using configurable weights to generate a final 
    trading signal.
    """
    
    def __init__(self, 
                 components: List[Any],
                 weights: Optional[List[float]] = None, 
                 buy_threshold: float = 0.5, 
                 sell_threshold: float = -0.5, 
                 name: Optional[str] = None,
                 event_bus = None):
        """Initialize the weighted strategy.
        
        Args:
            components: List of components that generate signals
            weights: List of weights for each component (default: equal weights)
            buy_threshold: Threshold above which to generate a buy signal
            sell_threshold: Threshold below which to generate a sell signal
            name: Strategy name
            event_bus: Optional event bus for emitting events
        """
        super().__init__(name or "WeightedStrategy", event_bus)
        self.components = components
        
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
        
        # Initialize state
        self.state = {
            'last_score': 0.0,
            'last_signal': None,
            'component_signals': {}
        }
    
    def generate_signals(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Generate weighted trading signals.

        Args:
            bar_event: BarEvent containing market data

        Returns:
            SignalEvent representing the weighted decision
        """
        try:
            # Get signals from all components
            component_signals = []
            
            # Extract bar data for components that might need it in dictionary form
            bar = bar_event.get_data()
            
            # We pass bar_event to components that can handle it, or bar dict for backward compatibility
            for i, component in enumerate(self.components):
                signal = None
                
                # If component has on_bar method expecting an Event
                if hasattr(component, 'on_bar'):
                    from src.events.event_base import Event
                    from src.events.event_types import EventType
                    
                    # Create an event wrapping the bar_event
                    event = Event(EventType.BAR, bar_event, bar_event.get_timestamp())
                    signal = component.on_bar(event)
                # If component has generate_signal method expecting a BarEvent
                elif hasattr(component, 'generate_signals'):
                    signal = component.generate_signals(bar_event)
                elif hasattr(component, 'generate_signal'):
                    signal = component.generate_signal(bar_event)
                # For backward compatibility with components expecting dictionary
                else:
                    # Try passing the raw bar dictionary
                    try:
                        signal = component.on_bar(bar)
                    except Exception as e:
                        logger.warning(f"Component {i} error: {str(e)}")
                        continue

                if signal is not None:
                    component_signals.append(signal)

            if not component_signals:
                # No signals generated, return neutral
                return self._create_neutral_signal(bar_event)

            # Calculate weighted signal
            weighted_sum = 0.0
            total_weight = sum(self.weights)
            signal_weights = {}

            for i, signal in enumerate(component_signals):
                # Get weight for this component
                weight = self.weights[i] if i < len(self.weights) else 1.0/len(component_signals)

                # Get direction from signal
                if isinstance(signal, SignalEvent):
                    # Extract from SignalEvent
                    direction = signal.get_signal_type()
                    # Store for metadata
                    signal_weights[signal.get_rule_id() or f"component_{i}"] = {
                        'weight': float(weight),
                        'direction': int(direction)
                    }
                elif isinstance(signal, dict) and 'signal_type' in signal:
                    # Legacy dictionary signal
                    direction = signal['signal_type']
                    # Store for metadata
                    signal_weights[signal.get('rule_id', f"component_{i}")] = {
                        'weight': float(weight),
                        'direction': int(direction)
                    }
                else:
                    # Unknown signal format, skip
                    logger.warning(f"Unknown signal format from component {i}: {type(signal)}")
                    continue

                # Apply weight
                weighted_sum += direction * weight

            # Normalize weighted sum
            if total_weight > 0:
                normalized_sum = weighted_sum / total_weight
            else:
                normalized_sum = 0.0

            # Update state
            self.state['last_score'] = normalized_sum
            self.state['component_signals'] = signal_weights

            # Determine final signal type based on weighted sum
            if normalized_sum > self.buy_threshold:
                signal_type = SignalEvent.BUY
                self.state['last_signal'] = 'BUY'
            elif normalized_sum < self.sell_threshold:
                signal_type = SignalEvent.SELL
                self.state['last_signal'] = 'SELL'
            else:
                signal_type = SignalEvent.NEUTRAL
                self.state['last_signal'] = 'NEUTRAL'

            # Create the combined signal
            metadata = {
                'weighted_sum': float(normalized_sum),
                'component_signals': signal_weights,
                'component_count': len(component_signals)
            }
            
            return SignalEvent(
                signal_type=signal_type,
                price=bar_event.get_price(),
                symbol=bar_event.get_symbol(),
                rule_id=self.name,
                metadata=metadata,
                timestamp=bar_event.get_timestamp()
            )
        
        except Exception as e:
            logger.error(f"Error in {self.name}.generate_signals: {str(e)}", exc_info=True)
            return None

    def _create_neutral_signal(self, bar_event: BarEvent) -> SignalEvent:
        """
        Create a neutral signal.
        
        Args:
            bar_event: BarEvent containing market data
            
        Returns:
            Neutral SignalEvent
        """
        return SignalEvent(
            signal_type=SignalEvent.NEUTRAL,
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            metadata={'component_count': 0},
            timestamp=bar_event.get_timestamp()
        )

    def reset(self):
        """Reset all components in the strategy."""
        super().reset()
        
        # Reset components if they have reset method
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()
        
        # Reset state
        self.state = {
            'last_score': 0.0,
            'last_signal': None,
            'component_signals': {}
        }
