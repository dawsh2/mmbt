"""
Regime Strategy Module

This module provides the RegimeStrategy class that adapts to different market regimes
by selecting appropriate sub-strategies.
"""

from typing import Dict, Optional, Any, Union
from enum import Enum
from .strategy_base import Strategy
from .strategy_registry import StrategyRegistry
from signals import Signal, SignalType

@StrategyRegistry.register(category="advanced")
class RegimeStrategy(Strategy):
    """Strategy that adapts to different market regimes.
    
    This strategy uses a regime detector to identify the current market regime
    and then delegates to the appropriate strategy for that regime.
    """
    
    def __init__(self, 
                 regime_detector: Any, 
                 regime_strategies: Dict[Enum, Strategy], 
                 default_strategy: Optional[Strategy] = None, 
                 name: Optional[str] = None):
        """Initialize the regime strategy.
        
        Args:
            regime_detector: Object that identifies market regimes
            regime_strategies: Dictionary mapping regime types to strategies
            default_strategy: Strategy to use when no regime-specific strategy is available
            name: Strategy name
        """
        super().__init__(name or "RegimeStrategy")
        self.regime_detector = regime_detector
        self.regime_strategies = regime_strategies
        self.default_strategy = default_strategy
        self.current_regime = None
        self.last_signal = None
    
    def get_strategy_for_regime(self, regime: Enum) -> Strategy:
        """Get the appropriate strategy for a specific regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Strategy: The strategy for this regime, or default if none exists
        """
        return self.regime_strategies.get(regime, self.default_strategy)
    
    def on_bar(self, event):
        """Process a bar and delegate to the appropriate regime-specific strategy.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            Signal: Trading signal from the regime-specific strategy
        """
        bar = event.bar
        
        # Detect the current regime
        self.current_regime = self.regime_detector.detect_regime(bar)
        
        # Get the appropriate strategy for this regime
        strategy = self.get_strategy_for_regime(self.current_regime)
        
        if strategy:
            # Delegate to the regime-specific strategy
            signal = strategy.on_bar(event)
            
            # Add regime information to metadata
            if signal and hasattr(signal, 'metadata'):
                if signal.metadata is None:
                    signal.metadata = {}
                signal.metadata['regime'] = self.current_regime
            
            self.last_signal = signal
            return signal
        else:
            # No strategy available for this regime
            self.last_signal = Signal(
                timestamp=bar["timestamp"],
                signal_type=SignalType.NEUTRAL,
                price=bar["Close"],
                rule_id=self.name,
                confidence=0.0,
                metadata={"regime": self.current_regime, "status": "no_strategy"}
            )
            return self.last_signal
    
    def reset(self):
        """Reset the regime detector and all strategies."""
        if hasattr(self.regime_detector, 'reset'):
            self.regime_detector.reset()
            
        for strategy in self.regime_strategies.values():
            strategy.reset()
            
        if self.default_strategy:
            self.default_strategy.reset()
            
        self.current_regime = None
        self.last_signal = None
