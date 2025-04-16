"""
Top-N Strategy Module

This module provides the TopNStrategy class that combines signals from top N rules
using a voting mechanism. This updated version uses standardized event objects.
"""

from typing import List, Dict, Any, Optional
import logging

from src.events.event_base import Event
from src.events.event_types import EventType, BarEvent
from src.events.signal_event import SignalEvent
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry
from signals import Signal, SignalRouter, SignalType

# Create a logger for this module
logger = logging.getLogger(__name__)

@StrategyRegistry.register(category="legacy")
class TopNStrategy(Strategy):
    """Strategy that combines signals from top N rules using consensus.
    
    This strategy uses SignalRouter internally for backward compatibility.
    It now handles standardized event objects.
    """
    
    def __init__(self, rule_objects: List[Any], name: Optional[str] = None, event_bus = None):
        """Initialize the TopN strategy.
        
        Args:
            rule_objects: List of rule objects
            name: Strategy name
            event_bus: Optional event bus for emitting events
        """
        super().__init__(name or "TopNStrategy", event_bus)
        self.rules = rule_objects
        self.router = SignalRouter(rule_objects)
        self.last_signal = None

    def generate_signals(self, bar_event: BarEvent) -> Optional[SignalEvent]:
        """Process a bar and generate a consensus signal.
        
        Args:
            bar_event: BarEvent containing market data
            
        Returns:
            SignalEvent if generated, None otherwise
        """
        logger.debug(f"TopNStrategy received bar event")
        
        # For backward compatibility, create an event that the router can handle
        from src.events.event_base import Event
        from src.events.event_types import EventType
        
        # Extract the bar dictionary for the router
        bar_data = bar_event.get_data()
        
        # Create an event for the router to process
        router_event = Event(EventType.BAR, bar_data, bar_event.get_timestamp())
        
        # Process through router
        router_output = self.router.on_bar(router_event)
        
        # If no output, return None
        if not router_output:
            return None
            
        # Get signal collection and consensus
        signal_collection = router_output["signals"]
        consensus_signal_type = signal_collection.get_weighted_consensus()

        # Get symbol from the bar data
        symbol = bar_event.get_symbol()
        
        # Map legacy signal type to StandardSignalEvent type
        if consensus_signal_type == SignalType.BUY:
            signal_value = SignalEvent.BUY
        elif consensus_signal_type == SignalType.SELL:
            signal_value = SignalEvent.SELL
        else:
            signal_value = SignalEvent.NEUTRAL
            
        # Create standardized SignalEvent
        signal_event = SignalEvent(
            signal_type=signal_value,
            price=bar_event.get_price(),
            symbol=symbol,
            rule_id=self.name,
            metadata={
                'consensus': str(consensus_signal_type),
                'component_count': len(self.rules)
            },
            timestamp=bar_event.get_timestamp()
        )

        # Store for reference
        self.last_signal = signal_event

        # Log if non-neutral
        if signal_value != SignalEvent.NEUTRAL:
            logger.info(f"Generated non-neutral signal: {signal_value} for {symbol}")

        return signal_event
    
    def reset(self):
        """Reset the router and signal state."""
        super().reset()
        self.router.reset()
