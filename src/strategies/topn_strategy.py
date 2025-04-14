"""
Top-N Strategy Module

This module provides the TopNStrategy class that combines signals from top N rules
using a voting mechanism. This is a migration of the original TopNStrategy to the
new architecture.
"""
from src.signals import Signal, SignalRouter
from typing import List, Optional, Any
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry

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
        """Process a bar and generate a consensus signal.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            Signal: Consensus signal from all rules
        """
        router_output = self.router.on_bar(event)
        signal_collection = router_output["signals"]
        consensus_signal_type = signal_collection.get_weighted_consensus()
        
        self.last_signal = Signal(
            timestamp=router_output["timestamp"],
            signal_type=consensus_signal_type,
            price=router_output["price"],
            rule_id=self.name,
            confidence=0.8  # Add reasonable confidence value
        )
        
        return self.last_signal
    
    def reset(self):
        """Reset the router and signal state."""
        self.router.reset()
        self.last_signal = None
