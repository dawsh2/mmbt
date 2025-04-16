"""
Top-N Strategy Module

This module provides the TopNStrategy class that combines signals from top N rules
using a voting mechanism. This is a migration of the original TopNStrategy to the
new architecture.
"""
from typing import List, Optional, Any
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry
from src.signals import Signal, SignalRouter, SignalType

# At the top of the file with other imports
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

@StrategyRegistry.register(category="legacy")
class TopNStrategy(Strategy):
    """Strategy that combines signals from top N rules using consensus.
    
    This is a migration of the original TopNStrategy to the new architecture.
    It uses SignalRouter internally for backward compatibility.
    """
    
    def __init__(self, rule_objects: List[Any], name: Optional[str] = None):
        """Initialize the TopN strategy.
        
        Args:
            rule_objects: List of rule objects
            name: Strategy name
        """
        super().__init__(name or "TopNStrategy")
        self.rules = rule_objects
        self.router = SignalRouter(rule_objects)
        self.last_signal = None


    def on_bar(self, event):
        """Process a bar and generate a consensus signal."""
        logger.debug(f"TopNStrategy received bar event")
        router_output = self.router.on_bar(event)
        signal_collection = router_output["signals"]
        consensus_signal_type = signal_collection.get_weighted_consensus()

        # Get the symbol from the bar data
        symbol = 'default'
        if hasattr(event, 'data') and hasattr(event.data, 'get'):
            symbol = event.data.get('symbol', 'default')
        elif isinstance(event, dict):
            symbol = event.get('symbol', 'default')
        elif hasattr(event, 'bar') and hasattr(event.bar, 'get'):
            symbol = event.bar.get('symbol', 'default')

        # Create signal with symbol in both the object and metadata
        signal = Signal(
            timestamp=router_output["timestamp"],
            signal_type=consensus_signal_type,
            price=router_output["price"],
            rule_id=self.name,
            confidence=0.8,
            metadata={'symbol': symbol}  # Add symbol to metadata
        )

        # Also add symbol as an attribute for convenience
        setattr(signal, 'symbol', symbol)

        self.last_signal = signal

        if consensus_signal_type != SignalType.NEUTRAL:
            logger.info(f"Generated non-neutral signal: {consensus_signal_type} for {symbol}")

        return self.last_signal
        
 
    
    def reset(self):
        """Reset the router and signal state."""
        self.router.reset()
        self.last_signal = None
