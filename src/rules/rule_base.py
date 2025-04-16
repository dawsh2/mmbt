"""
Rule Base Module

This module defines the base Rule class and related abstractions for the rules layer
of the trading system. Rules transform market data events into trading signals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import datetime
import logging
from collections import deque
import uuid

# Import proper event types
from src.events.event_types import EventType, BarEvent
from src.events.event_base import Event
from src.events.signal_event import SignalEvent

# Set up logging
logger = logging.getLogger(__name__)


class Rule(ABC):
    """
    Base class for all trading rules in the system.
    
    Rules transform market data into trading signals by applying decision logic.
    Each rule encapsulates a specific trading strategy or signal generation logic.
    """
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None, description: str = ""):
        """
        Initialize a rule.
        
        Args:
            name: Unique identifier for this rule
            params: Dictionary of configuration parameters
            description: Human-readable description of the rule
        """
        self.name = name
        self.params = params or self.default_params()
        self.description = description
        self.state = {}
        self.signals = []
        self._validate_params()
        
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
    def generate_signal(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Analyze market data and generate a complete trading signal.
        
        This method is responsible for the entire signal generation process:
        1. Analyzing the bar data 
        2. Determining if a signal should be generated
        3. Creating and returning the proper SignalEvent
        
        Subclasses must implement this method with their specific trading logic.
        
        Args:
            bar_event: BarEvent containing market data
                 
        Returns:
            Complete SignalEvent if conditions warrant a signal, None otherwise
        """
        pass

    def on_bar(self, event: Event) -> Optional[SignalEvent]:
        """
        Process a bar event and generate a trading signal.
        
        This method extracts the bar data from the event and passes it to
        generate_signal() for the actual signal generation logic.
        
        Args:
            event: Event containing a BarEvent in its data attribute
            
        Returns:
            SignalEvent if a signal is generated, None otherwise
        """
        # Extract BarEvent properly with strong type checking
        if not isinstance(event, Event):
            logger.error(f"Expected Event object, got {type(event).__name__}")
            return None

        # Case 1: Event data is a BarEvent
        if isinstance(event.data, BarEvent):
            bar_event = event.data
        # Case 2: Event data is a dict (backward compatibility)
        elif isinstance(event.data, dict) and 'Close' in event.data:
            # Convert dict to BarEvent
            bar_event = BarEvent(event.data)
            logger.warning(f"Rule {self.name}: Received dictionary instead of BarEvent, converting")
        else:
            logger.error(f"Rule {self.name}: Unable to extract BarEvent from {type(event.data).__name__}")
            return None
        
        # Generate signal by delegating to the subclass implementation
        try:
            signal = self.generate_signal(bar_event)
            
            # Store in history
            if signal is not None:
                self.signals.append(signal)
                logger.info(f"Rule {self.name}: Generated {signal.get_signal_name()} signal")
            
            return signal
            
        except Exception as e:
            logger.error(f"Rule {self.name}: Error generating signal: {str(e)}", exc_info=True)
            return None

    def update_state(self, key: str, value: Any) -> None:
        """
        Update the rule's internal state.
        
        Args:
            key: State dictionary key
            value: Value to store
        """
        self.state[key] = value
    
    def get_state(self, key: str = None, default: Any = None) -> Any:
        """
        Get a value from the rule's state or the entire state dict.
        
        Args:
            key: State dictionary key, or None to get the entire state
            default: Default value if key is not found
            
        Returns:
            The value from state or default, or the entire state dict
        """
        if key is None:
            return self.state
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
    
    def generate_signal(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Generate a composite signal by combining signals from sub-rules.
        
        Args:
            bar_event: BarEvent containing market data
                 
        Returns:
            SignalEvent representing the combined decision, or None if no signal
        """
        # Generate signals from all sub-rules
        sub_signals = []
        for rule in self.rules:
            signal = rule.generate_signal(bar_event)
            if signal is not None:
                sub_signals.append(signal)
        
        # If no signals were generated, return None
        if not sub_signals:
            return None
        
        # Combine signals based on the aggregation method
        if self.aggregation_method == 'majority':
            return self._majority_vote(sub_signals, bar_event)
        elif self.aggregation_method == 'unanimous':
            return self._unanimous_vote(sub_signals, bar_event)
        elif self.aggregation_method == 'weighted':
            return self._weighted_vote(sub_signals, bar_event)
        elif self.aggregation_method == 'any':
            return self._any_vote(sub_signals, bar_event)
        elif self.aggregation_method == 'sequence':
            return self._sequence_vote(sub_signals, bar_event)
        else:
            # Default to majority vote
            return self._majority_vote(sub_signals, bar_event)
    
    def _majority_vote(self, signals: List[SignalEvent], bar_event: BarEvent) -> SignalEvent:
        """
        Combine signals using a majority vote.
        
        Args:
            signals: List of signals from sub-rules
            bar_event: Original bar event
            
        Returns:
            Combined signal
        """
        from src.signals import SignalType
        
        # Count votes by signal type
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
        
        # Create metadata with vote information
        metadata = {
            'vote_counts': {
                'buy': buy_votes,
                'sell': sell_votes,
                'neutral': neutral_votes
            },
            'symbol': bar_event.get_symbol()
        }
        
        # Create combined signal
        return SignalEvent(
            signal_type=signal_type, 
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            metadata=metadata,
            confidence=confidence,
            timestamp=bar_event.get_timestamp()
        )
    
    def _unanimous_vote(self, signals: List[SignalEvent], bar_event: BarEvent) -> SignalEvent:
        """
        Combine signals requiring unanimous agreement.
        
        Args:
            signals: List of signals from sub-rules
            bar_event: Original bar event
            
        Returns:
            Combined signal
        """
        from src.signals import SignalType
        
        # Check if all signals are the same type
        signal_types = set(s.signal_type for s in signals)
        
        # Determine final signal type
        if len(signal_types) == 1 and SignalType.NEUTRAL not in signal_types:
            signal_type = next(iter(signal_types))
        else:
            signal_type = SignalType.NEUTRAL
        
        # Create combined signal
        return SignalEvent(
            signal_type=signal_type,
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            confidence=1.0 if len(signal_types) == 1 else 0.0,
            metadata={'unanimous': len(signal_types) == 1},
            timestamp=bar_event.get_timestamp()
        )
    
    def _weighted_vote(self, signals: List[SignalEvent], bar_event: BarEvent) -> SignalEvent:
        """
        Combine signals using weighted voting.
        
        Args:
            signals: List of signals from sub-rules
            bar_event: Original bar event
            
        Returns:
            Combined signal
        """
        from src.signals import SignalType
        
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
        return SignalEvent(
            signal_type=signal_type,
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            confidence=confidence,
            metadata={'weighted_sum': normalized_sum},
            timestamp=bar_event.get_timestamp()
        )
    
    def _any_vote(self, signals: List[SignalEvent], bar_event: BarEvent) -> SignalEvent:
        """
        Generate signal if any sub-rule generates a non-neutral signal.
        
        Args:
            signals: List of signals from sub-rules
            bar_event: Original bar event
            
        Returns:
            Combined signal
        """
        from src.signals import SignalType
        
        # Look for the first non-neutral signal
        for signal in signals:
            if signal.signal_type != SignalType.NEUTRAL:
                return SignalEvent(
                    signal_type=signal.signal_type,
                    price=bar_event.get_price(),
                    symbol=bar_event.get_symbol(),
                    rule_id=self.name,
                    confidence=signal.confidence,
                    metadata={'triggering_rule': signal.rule_id},
                    timestamp=bar_event.get_timestamp()
                )
        
        # If all signals are neutral, return neutral
        return SignalEvent(
            signal_type=SignalType.NEUTRAL,
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            confidence=1.0,
            timestamp=bar_event.get_timestamp()
        )
    
    def _sequence_vote(self, signals: List[SignalEvent], bar_event: BarEvent) -> SignalEvent:
        """
        Generate signal based on a sequence of conditions.
        
        This method requires all sub-rules to generate non-neutral signals
        in sequence for a final signal to be generated.
        
        Args:
            signals: List of signals from sub-rules
            bar_event: Original bar event
            
        Returns:
            Combined signal
        """
        from src.signals import SignalType
        
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
                    
                    return SignalEvent(
                        signal_type=final_signal_type,
                        price=bar_event.get_price(),
                        symbol=bar_event.get_symbol(),
                        rule_id=self.name,
                        confidence=current_signal.confidence,
                        metadata={'sequence_completed': True},
                        timestamp=bar_event.get_timestamp()
                    )
            else:
                # If current rule returns neutral, reset sequence
                self.update_state(sequence_key, 0)
        
        # If sequence is not complete, return neutral
        return SignalEvent(
            signal_type=SignalType.NEUTRAL,
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            confidence=0.0,
            metadata={'sequence_index': current_index},
            timestamp=bar_event.get_timestamp()
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
    
    def generate_signal(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Generate a trading signal based on features.
        
        Args:
            bar_event: BarEvent containing market data
                 
        Returns:
            SignalEvent representing the trading decision, or None if no signal
        """
        # Extract raw bar data
        bar_data = bar_event.get_data()
        
        # Extract features from data
        features = {}
        for feature_name in self.feature_names:
            if feature_name in bar_data:
                features[feature_name] = bar_data[feature_name]
            else:
                # If a required feature is missing, return None
                logger.warning(f"Rule {self.name}: Missing required feature: {feature_name}")
                return None
        
        # Call the rule's decision method with the features
        return self.make_decision(features, bar_event)
    
    @abstractmethod
    def make_decision(self, features: Dict[str, Any], bar_event: BarEvent) -> Optional[SignalEvent]:
        """
        Make a trading decision based on the features.
        
        This method should be implemented by subclasses to define
        the specific decision logic using features.
        
        Args:
            features: Dictionary of feature values
            bar_event: Original bar event
                 
        Returns:
            SignalEvent representing the trading decision, or None if no signal
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


