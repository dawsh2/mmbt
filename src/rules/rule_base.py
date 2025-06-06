"""
Rule Base Module

This module defines the base Rule class and related abstractions for the rules layer
of the trading system. Rules analyze market data to generate trading signals.
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
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None, 
                 description: str = "", event_bus=None):
        """
        Initialize a rule.
        
        Args:
            name: Unique identifier for this rule
            params: Dictionary of configuration parameters
            description: Human-readable description of the rule
            event_bus: Optional event bus for emitting signals
        """
        self.name = name
        self.params = params or self.default_params()
        self.description = description
        self.state = {}
        self.signals = []
        self.event_bus = event_bus
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
        Analyze market data and generate a trading signal.
        
        This method is responsible for the signal generation process:
        1. Analyzing the bar data 
        2. Determining if a signal should be generated
        3. Creating and returning the appropriate SignalEvent
        
        Subclasses must implement this method with their specific trading logic.
        
        Args:
            bar_event: BarEvent containing market data
                 
        Returns:
            SignalEvent if conditions warrant a signal, None otherwise
        """
        pass

    # Update in Rule class
    def on_bar(self, event: Event) -> Optional[SignalEvent]:
            """
            Process a bar event and generate a trading signal.

            Args:
                event: Event containing a BarEvent

            Returns:
                SignalEvent if a signal is generated, None otherwise
            """
            try:
                # Extract BarEvent from event
                if not isinstance(event, Event):
                    logger.warning(f"Expected Event object, got {type(event).__name__}")
                    # Try to handle it directly if it's a BarEvent
                    if isinstance(event, BarEvent):
                        bar_event = event
                    else:
                        return None
                else:
                    # Extract BarEvent from Event.data
                    if isinstance(event.data, BarEvent):
                        bar_event = event.data
                    else:
                        logger.warning(f"Expected BarEvent in event.data, got {type(event.data).__name__}")
                        return None

                # Generate signal
                signal = self.generate_signal(bar_event)

                # If signal was generated and we have an event bus, emit it
                if signal is not None and self.event_bus is not None:
                    if not isinstance(signal, SignalEvent):
                        logger.warning(f"Rule {self.name} returned non-SignalEvent: {type(signal).__name__}")
                        return None

                    # Emit signal event
                    self.event_bus.emit(Event(EventType.SIGNAL, signal))

                    # Log signal
                    signal_type = "BUY" if signal.get_signal_value() > 0 else "SELL" if signal.get_signal_value() < 0 else "NEUTRAL"
                    logger.info(f"Generated {signal_type} signal for {signal.get_symbol()} @ {signal.get_price()}")

                return signal

            except Exception as e:
                logger.error(f"Error in rule {self.name}: {e}", exc_info=True)
                return None

    # def on_bar(self, event: Event) -> Optional[SignalEvent]:
    #     """
    #     Process a bar event and generate a trading signal.

    #     Args:
    #         event: Event containing a BarEvent in its data attribute

    #     Returns:
    #         SignalEvent if a signal is generated, None otherwise
    #     """
    #     # Extract BarEvent with type checking
    #     if not isinstance(event, Event):
    #         logger.error(f"Expected Event object, got {type(event).__name__}")
    #         return None

    #     # Extract BarEvent - ONLY accept BarEvent objects, not dictionaries
    #     if not isinstance(event.data, BarEvent):
    #         logger.error(f"Rule {self.name}: Expected BarEvent in event.data, got {type(event.data).__name__}")
    #         return None

    #     bar_event = event.data

    #     # Generate signal by delegating to the subclass implementation
    #     try:
    #         signal = self.generate_signal(bar_event)

    #         # Validate signal type - STRICT VALIDATION
    #         if signal is not None and not isinstance(signal, SignalEvent):
    #             logger.error(f"Rule {self.name}: Invalid signal type: {type(signal).__name__}, must be SignalEvent")
    #             return None

    #         # If signal was generated, store it and emit it
    #         if signal is not None:
    #             # Store in history
    #             self.signals.append(signal)
    #             logger.info(f"Rule {self.name}: Generated {signal.get_signal_name()} signal")

    #             # Emit signal event if we have an event bus
    #             if self.event_bus is not None:
    #                 try:
    #                     # Create and emit signal event
    #                     signal_event = Event(EventType.SIGNAL, signal)
    #                     self.event_bus.emit(signal_event)
    #                     logger.debug(f"Rule {self.name}: Emitted signal event")
    #                 except Exception as e:
    #                     logger.error(f"Rule {self.name}: Error emitting signal event: {str(e)}", exc_info=True)

    #         return signal

    #     except Exception as e:
    #         logger.error(f"Rule {self.name}: Error generating signal: {str(e)}", exc_info=True)
    #         return None

  
    def set_event_bus(self, event_bus):
        """
        Set the event bus for this rule.

        Args:
            event_bus: Event bus for emitting signals
        """
        self.event_bus = event_bus

        # Auto-register with event bus
        if event_bus is not None:
            # Create a function handler that delegates to on_bar
            def handler_func(event):
                return self.on_bar(event)

            # Register the handler
            event_bus.register(EventType.BAR, handler_func)
        


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
                 description: str = "",
                 event_bus=None):
        """
        Initialize a composite rule.
        
        Args:
            name: Unique identifier for the rule
            rules: List of component rules
            aggregation_method: Method to combine signals ('majority', 'unanimous', 'weighted')
            params: Dictionary of parameters
            description: Human-readable description
            event_bus: Optional event bus for emitting signals
        """
        self.rules = rules
        self.aggregation_method = aggregation_method
        super().__init__(name, params, description, event_bus)
        
        # Set event_bus on all component rules
        for rule in self.rules:
            rule.set_event_bus(None)  # Disable direct emission from sub-rules
        
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
        # Count votes by signal value
        buy_votes = sum(1 for s in signals if s.get_signal_value() == SignalEvent.BUY)
        sell_votes = sum(1 for s in signals if s.get_signal_value() == SignalEvent.SELL)
        neutral_votes = len(signals) - buy_votes - sell_votes
        
        # Determine final signal type
        if buy_votes > sell_votes and buy_votes > neutral_votes:
            signal_value = SignalEvent.BUY
        elif sell_votes > buy_votes and sell_votes > neutral_votes:
            signal_value = SignalEvent.SELL
        else:
            signal_value = SignalEvent.NEUTRAL
        
        # Calculate confidence based on vote distribution
        total_votes = len(signals)
        if total_votes > 0:
            if signal_value == SignalEvent.BUY:
                confidence = buy_votes / total_votes
            elif signal_value == SignalEvent.SELL:
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
            signal_value=signal_value, 
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            metadata=metadata,
            timestamp=bar_event.get_timestamp()
        )
    
    # Other voting methods would go here with similar implementations
    
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
                 description: str = "",
                 event_bus=None):
        """
        Initialize a feature-based rule.
        
        Args:
            name: Unique identifier for the rule
            feature_names: List of feature names this rule depends on
            params: Dictionary of parameters
            description: Human-readable description
            event_bus: Optional event bus for emitting signals
        """
        self.feature_names = feature_names
        super().__init__(name, params, description, event_bus)
    
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
        the specific decision logic using features. It should directly
        return a SignalEvent object when appropriate.
        
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

