"""
Rule Base Module

This module defines the base Rule class and related abstractions for the rules layer
of the trading system. Rules transform features into trading signals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from collections import deque

from src.events.event_types import EventType, BarEvent
from src.events.event_base import Event
from src.events.signal_event import SignalEvent

# At the top of the file with other imports
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


class Rule(ABC):
    """
    Base class for all trading rules in the system.
    
    Rules transform features into trading signals by applying decision logic.
    Each rule encapsulates a specific trading strategy or signal generation logic.
    """
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None, description: str = ""):
        self.name = name
        self.params = params or self.default_params()
        self.description = description
        self.state = {}
        self.signals = []
        self._validate_params()
        # Add parameter application validation
        # self._validate_param_application()
        
    def _validate_params(self) -> None:
        """
        Validate the parameters provided to the rule.
        
        This method should be overridden by subclasses to provide
        specific parameter validation logic.
        
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        """
        Get the default parameters for this rule.
        
        Returns:
            Dictionary of default parameter values
        """
        return {}
    
    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal from the provided data.
        
        Args:
            data: Dictionary containing price data, indicators, features
                 required by this rule
                 
        Returns:
            Signal object representing the trading decision
        """
        pass

    def on_bar(self, event_or_data):
        """
        Process a bar event and generate a trading signal.
        """
        # Extract data from different potential sources
        bar_data = None

        # Case 1: Standard Event object with data attribute
        if hasattr(event_or_data, 'data'):
            bar_data = event_or_data.data
            logger.debug(f"Extracted data from Event object")

        # Case 2: Already a dict (backward compatibility)
        elif isinstance(event_or_data, dict):
            bar_data = event_or_data
            logger.debug(f"Using dict data directly")

        # Case 3: Custom BarEvent with bar attribute
        elif hasattr(event_or_data, 'bar'):
            bar_data = event_or_data.bar
            logger.debug(f"Extracted data from BarEvent.bar")

        # Case 4: Try to handle other formats
        else:
            logger.warning(f"Unknown data format in on_bar: {type(event_or_data)}")
            try:
                bar_data = dict(event_or_data)
            except Exception as e:
                logger.error(f"Failed to convert to dict: {e}")
                # Create minimal dict with timestamp
                bar_data = {'timestamp': datetime.now()}

        # Log the resulting data 
        logger.debug(f"Final bar_data for processing: {bar_data}")

        # Generate signal
        signal = self.generate_signal(bar_data)

        # Store in history
        if signal is not None:
            self.signals.append(signal)
            if signal.signal_type != SignalType.NEUTRAL:
                logger.info(f"Generated {signal.signal_type} signal")

        return signal



    def update_state(self, key: str, value: Any) -> None:
        """
        Update the rule's internal state.
        
        Args:
            key: State dictionary key
            value: Value to store
        """
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the rule's state.
        
        Args:
            key: State dictionary key
            default: Default value if key is not found
            
        Returns:
            The value from state or default
        """
        return self.state.get(key, default)
    
    def reset(self) -> None:
        """
        Reset the rule's internal state and signal history.
        
        This method should be called when reusing a rule instance
        for a new backtest or trading session.
        """
        self.state = {}
        self.signals = []
    
    def __str__(self) -> str:
        """String representation of the rule."""
        return f"{self.name} (Rule)"
    
    def __repr__(self) -> str:
        """Detailed representation of the rule."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class CompositeRule(Rule):
    """
    A rule composed of multiple sub-rules.
    
    CompositeRule combines signals from multiple rules using a specified
    aggregation method to produce a final signal.
    """
    
    def __init__(self, 
                 name: str, 
                 rules: List[Rule],
                 aggregation_method: str = "majority",
                 params: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        """
        Initialize a composite rule.
        
        Args:
            name: Unique identifier for the rule
            rules: List of component rules
            aggregation_method: Method to combine signals ('majority', 'unanimous', 'weighted')
            params: Dictionary of parameters
            description: Human-readable description
        """
        self.rules = rules
        self.aggregation_method = aggregation_method
        super().__init__(name, params, description)
        
    def _validate_params(self) -> None:
        """Validate the parameters for this composite rule."""
        valid_methods = ['majority', 'unanimous', 'weighted', 'any', 'sequence']
        if self.aggregation_method not in valid_methods:
            raise ValueError(f"Aggregation method must be one of {valid_methods}")
            
        if self.aggregation_method == 'weighted' and 'weights' not in self.params:
            raise ValueError("Weights must be provided for 'weighted' aggregation method")
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a composite signal by combining signals from sub-rules.
        
        Args:
            data: Dictionary containing price data, indicators, features
                 
        Returns:
            Signal object representing the combined decision
        """
        # Generate signals from all sub-rules
        sub_signals = [rule.generate_signal(data) for rule in self.rules]
        
        # Combine signals based on the aggregation method
        if self.aggregation_method == 'majority':
            return self._majority_vote(sub_signals, data)
        elif self.aggregation_method == 'unanimous':
            return self._unanimous_vote(sub_signals, data)
        elif self.aggregation_method == 'weighted':
            return self._weighted_vote(sub_signals, data)
        elif self.aggregation_method == 'any':
            return self._any_vote(sub_signals, data)
        elif self.aggregation_method == 'sequence':
            return self._sequence_vote(sub_signals, data)
        else:
            # Default to majority vote
            return self._majority_vote(sub_signals, data)
    
    def _majority_vote(self, signals: List[Signal], data: Dict[str, Any]) -> Signal:
        """
        Combine signals using a majority vote.
        
        Args:
            signals: List of signals from sub-rules
            data: Original data dictionary
            
        Returns:
            Combined signal
        """
        buy_votes = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_votes = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        neutral_votes = len(signals) - buy_votes - sell_votes
        
        # Determine final signal type
        if buy_votes > sell_votes and buy_votes > neutral_votes:
            signal_type = SignalType.BUY
        elif sell_votes > buy_votes and sell_votes > neutral_votes:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Calculate confidence based on vote distribution
        total_votes = len(signals)
        if total_votes > 0:
            if signal_type == SignalType.BUY:
                confidence = buy_votes / total_votes
            elif signal_type == SignalType.SELL:
                confidence = sell_votes / total_votes
            else:
                confidence = neutral_votes / total_votes
        else:
            confidence = 0.0
        
        # Create combined signal
        timestamp = data.get('timestamp', None)
        price = data.get('Close', None)
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            rule_id=self.name,
            confidence=confidence,
            metadata={'vote_counts': {
                'buy': buy_votes,
                'sell': sell_votes,
                'neutral': neutral_votes
            }}
        )
    
    def _unanimous_vote(self, signals: List[Signal], data: Dict[str, Any]) -> Signal:
        """
        Combine signals requiring unanimous agreement.
        
        Args:
            signals: List of signals from sub-rules
            data: Original data dictionary
            
        Returns:
            Combined signal
        """
        # Check if all signals are the same type
        signal_types = set(s.signal_type for s in signals)
        
        # Determine final signal type
        if len(signal_types) == 1 and SignalType.NEUTRAL not in signal_types:
            signal_type = next(iter(signal_types))
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create combined signal
        timestamp = data.get('timestamp', None)
        price = data.get('Close', None)
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            rule_id=self.name,
            confidence=1.0 if len(signal_types) == 1 else 0.0
        )
    
    def _weighted_vote(self, signals: List[Signal], data: Dict[str, Any]) -> Signal:
        """
        Combine signals using weighted voting.
        
        Args:
            signals: List of signals from sub-rules
            data: Original data dictionary
            
        Returns:
            Combined signal
        """
        weights = self.params.get('weights', [1.0] * len(signals))
        
        # Ensure weights match the number of signals
        if len(weights) != len(signals):
            weights = [1.0] * len(signals)
        
        # Calculate weighted vote
        weighted_sum = 0.0
        total_weight = sum(weights)
        
        for i, signal in enumerate(signals):
            weight = weights[i]
            if signal.signal_type == SignalType.BUY:
                weighted_sum += weight
            elif signal.signal_type == SignalType.SELL:
                weighted_sum -= weight
        
        # Normalize weighted sum to [-1, 1]
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0
        
        # Determine final signal type based on weighted sum
        threshold = self.params.get('threshold', 0.5)
        if normalized_sum > threshold:
            signal_type = SignalType.BUY
        elif normalized_sum < -threshold:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Calculate confidence
        confidence = abs(normalized_sum)
        
        # Create combined signal
        timestamp = data.get('timestamp', None)
        price = data.get('Close', None)
        
        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            rule_id=self.name,
            confidence=confidence,
            metadata={'weighted_sum': normalized_sum}
        )
    
    def _any_vote(self, signals: List[Signal], data: Dict[str, Any]) -> Signal:
        """
        Generate signal if any sub-rule generates a non-neutral signal.
        
        Args:
            signals: List of signals from sub-rules
            data: Original data dictionary
            
        Returns:
            Combined signal
        """
        # Look for the first non-neutral signal
        for signal in signals:
            if signal.signal_type != SignalType.NEUTRAL:
                return Signal(
                    timestamp=data.get('timestamp', None),
                    signal_type=signal.signal_type,
                    price=data.get('Close', None),
                    rule_id=self.name,
                    confidence=signal.confidence,
                    metadata={'triggering_rule': signal.rule_id}
                )
        
        # If all signals are neutral, return neutral
        return Signal(
            timestamp=data.get('timestamp', None),
            signal_type=SignalType.NEUTRAL,
            price=data.get('Close', None),
            rule_id=self.name,
            confidence=1.0
        )
    
    def _sequence_vote(self, signals: List[Signal], data: Dict[str, Any]) -> Signal:
        """
        Generate signal based on a sequence of conditions.
        
        This method requires all sub-rules to generate non-neutral signals
        in sequence for a final signal to be generated.
        
        Args:
            signals: List of signals from sub-rules
            data: Original data dictionary
            
        Returns:
            Combined signal
        """
        # Check if the sequence requirement has been met
        sequence_key = 'sequence_index'
        current_index = self.get_state(sequence_key, 0)
        
        if current_index < len(signals):
            # Check current rule in sequence
            current_signal = signals[current_index]
            
            if current_signal.signal_type != SignalType.NEUTRAL:
                # Move to next rule in sequence
                self.update_state(sequence_key, current_index + 1)
                
                # If this was the last rule, generate the final signal
                if current_index == len(signals) - 1:
                    final_signal_type = current_signal.signal_type
                    self.update_state(sequence_key, 0)  # Reset sequence
                    
                    return Signal(
                        timestamp=data.get('timestamp', None),
                        signal_type=final_signal_type,
                        price=data.get('Close', None),
                        rule_id=self.name,
                        confidence=current_signal.confidence,
                        metadata={'sequence_completed': True}
                    )
            else:
                # If current rule returns neutral, reset sequence
                self.update_state(sequence_key, 0)
        
        # If sequence is not complete, return neutral
        return Signal(
            timestamp=data.get('timestamp', None),
            signal_type=SignalType.NEUTRAL,
            price=data.get('Close', None),
            rule_id=self.name,
            confidence=0.0,
            metadata={'sequence_index': current_index}
        )
    
    def reset(self) -> None:
        """Reset this rule and all sub-rules."""
        super().reset()
        for rule in self.rules:
            rule.reset()


class FeatureBasedRule(Rule):
    """
    A rule that generates signals based on features.
    
    FeatureBasedRule uses a list of features to generate trading signals,
    abstracting away the direct handling of price data and indicators.
    """
    
    def __init__(self, 
                 name: str, 
                 feature_names: List[str],
                 params: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        """
        Initialize a feature-based rule.
        
        Args:
            name: Unique identifier for the rule
            feature_names: List of feature names this rule depends on
            params: Dictionary of parameters
            description: Human-readable description
        """
        self.feature_names = feature_names
        super().__init__(name, params, description)
    
    def generate_signal(self, data: Dict[str, Any]) -> Signal:
        """
        Generate a trading signal based on features.
        
        Args:
            data: Dictionary containing features and other data
                 
        Returns:
            Signal object representing the trading decision
        """
        # Extract features from data
        features = {}
        for feature_name in self.feature_names:
            if feature_name in data:
                features[feature_name] = data[feature_name]
            else:
                # If a required feature is missing, return neutral signal
                timestamp = data.get('timestamp', None)
                price = data.get('Close', None)
                return Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.NEUTRAL,
                    price=price,
                    rule_id=self.name,
                    confidence=0.0,
                    metadata={'error': f"Missing feature: {feature_name}"}
                )
        
        # Call the rule's decision method with the features
        return self.make_decision(features, data)
    
    @abstractmethod
    def make_decision(self, features: Dict[str, Any], data: Dict[str, Any]) -> Signal:
        """
        Make a trading decision based on the features.
        
        This method should be implemented by subclasses to define
        the specific decision logic using features.
        
        Args:
            features: Dictionary of feature values
            data: Original data dictionary
                 
        Returns:
            Signal object representing the trading decision
        """
        pass


    def _validate_param_application(self):
        """
        Validate that parameters were correctly applied to this instance.
        Called after initialization.
        """
        if self.params is None:
            raise ValueError(f"Parameters were not properly applied to {self.name}")

        # Check if any parameter is None when it shouldn't be
        for param_name, param_value in self.params.items():
            if param_value is None:
                default_params = self.default_params()
                if param_name in default_params and default_params[param_name] is not None:
                    raise ValueError(f"Parameter {param_name} is None but should have a value")
