"""
Top-N Strategy Module

This module provides the TopNStrategy class that combines signals from top N rules
using a voting mechanism. This version uses standardized SignalEvent objects.
"""

from typing import List, Dict, Any, Optional
import logging

from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry

# Create a logger for this module
logger = logging.getLogger(__name__)

@StrategyRegistry.register(category="ensemble")
class TopNStrategy(Strategy):
    """Strategy that combines signals from top N rules using consensus."""
    
    def __init__(self, rule_objects: List[Any], name: Optional[str] = None, event_bus = None):
        """Initialize the TopN strategy.
        
        Args:
            rule_objects: List of rule objects
            name: Strategy name
            event_bus: Optional event bus for emitting events
        """
        super().__init__(name or "TopNStrategy", event_bus)
        self.rules = rule_objects
        self.last_signal = None

    def generate_signals(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process a bar and generate a consensus signal.
        
        Args:
            bar_event: BarEvent containing market data
            
        Returns:
            SignalEvent if generated, None otherwise
        """
        logger.debug(f"TopNStrategy processing market data")
        
        # Collect signals from all rules
        rule_signals = []
        for rule in self.rules:
            try:
                # Pass bar event to each rule
                if hasattr(rule, 'generate_signal'):
                    signal = rule.generate_signal(bar_event)
                elif hasattr(rule, 'generate_signals'):
                    signal = rule.generate_signals(bar_event)
                elif hasattr(rule, 'on_bar'):
                    # Create event for rules expecting an Event object
                    event = Event(EventType.BAR, bar_event, bar_event.get_timestamp())
                    signal = rule.on_bar(event)
                else:
                    logger.warning(f"Rule {rule} does not have a compatible signal generation method")
                    continue
                
                # Add valid signals to our collection
                if signal is not None and isinstance(signal, SignalEvent):
                    rule_signals.append(signal)
            except Exception as e:
                logger.error(f"Error getting signal from rule: {e}", exc_info=True)
        
        # If no signals were generated, return None
        if not rule_signals:
            return None
            
        # Count votes for each signal type
        buy_votes = sum(1 for s in rule_signals if s.get_signal_value() == SignalEvent.BUY)
        sell_votes = sum(1 for s in rule_signals if s.get_signal_value() == SignalEvent.SELL)
        neutral_votes = len(rule_signals) - buy_votes - sell_votes
        
        # Determine consensus signal using simple majority
        if buy_votes > sell_votes and buy_votes > neutral_votes:
            signal_value = SignalEvent.BUY
            confidence = buy_votes / len(rule_signals)
        elif sell_votes > buy_votes and sell_votes > neutral_votes:
            signal_value = SignalEvent.SELL
            confidence = sell_votes / len(rule_signals)
        else:
            signal_value = SignalEvent.NEUTRAL
            confidence = neutral_votes / len(rule_signals) if neutral_votes > 0 else 0.5
            
        # Create standardized SignalEvent
        signal_event = SignalEvent(
            signal_value=signal_value,
            price=bar_event.get_price(),
            symbol=bar_event.get_symbol(),
            rule_id=self.name,
            metadata={
                'buy_votes': buy_votes,
                'sell_votes': sell_votes,
                'neutral_votes': neutral_votes,
                'total_votes': len(rule_signals),
                'confidence': confidence
            },
            timestamp=bar_event.get_timestamp()
        )

        # Store for reference
        self.last_signal = signal_event

        # Log if non-neutral
        if signal_value != SignalEvent.NEUTRAL:
            logger.info(f"Generated non-neutral signal: {signal_value} for {bar_event.get_symbol()}")

        return signal_event
    
    def reset(self):
        """Reset the strategy state."""
        super().reset()
        
        # Reset each rule
        for rule in self.rules:
            if hasattr(rule, 'reset'):
                rule.reset()
