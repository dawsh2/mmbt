"""
Ensemble Strategy Module

This module provides the EnsembleStrategy class that combines signals from multiple
strategies using configurable combination methods.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Any, Union
import numpy as np
from src.strategies.strategy_base import Strategy
from src.strategies.strategy_registry import StrategyRegistry
from signals import Signal, SignalType

@StrategyRegistry.register(category="advanced")
class EnsembleStrategy(Strategy):
    """Strategy that combines signals from multiple sub-strategies.
    
    This strategy implements various methods for combining the signals from
    multiple strategies, including voting, weighted averaging, and consensus.
    """
    
    def __init__(self, 
                 strategies: Dict[str, Strategy], 
                 combination_method: str = 'voting', 
                 weights: Optional[Dict[str, float]] = None, 
                 name: Optional[str] = None):
        """Initialize the ensemble strategy.
        
        Args:
            strategies: Dictionary mapping strategy names to strategy objects
            combination_method: Method to combine signals ('voting', 'weighted', or 'consensus')
            weights: Optional dictionary of weights for each strategy (for 'weighted' method)
            name: Strategy name
        """
        super().__init__(name or "EnsembleStrategy")
        self.strategies = strategies
        self.combination_method = combination_method
        
        # Handle weights
        if combination_method == 'weighted':
            if weights is None:
                # Equal weights if none provided
                weight_value = 1.0 / len(strategies)
                self.weights = {name: weight_value for name in strategies}
            else:
                # Ensure we have weights for each strategy
                missing_strategies = set(strategies.keys()) - set(weights.keys())
                if missing_strategies:
                    print(f"Warning: Missing weights for strategies: {missing_strategies}")
                    weight_value = 1.0 / len(strategies)
                    self.weights = {name: weights.get(name, weight_value) for name in strategies}
                else:
                    self.weights = weights
                
                # Normalize weights to sum to 1
                weight_sum = sum(self.weights.values())
                if weight_sum > 0:
                    self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        else:
            self.weights = None
        
        self.last_signal = None
    
    def on_bar(self, event):
        """Process a bar and generate a signal by combining multiple strategies.
        
        Args:
            event: Bar event containing market data
            
        Returns:
            Signal: Combined signal
        """
        bar = event.bar
        
        # Collect signals from all strategies
        strategy_signals = {}
        strategy_confidences = {}
        
        for name, strategy in self.strategies.items():
            signal = strategy.on_bar(event)
            if signal:
                strategy_signals[name] = signal.signal_type.value
                strategy_confidences[name] = getattr(signal, 'confidence', 1.0)
            else:
                strategy_signals[name] = 0
                strategy_confidences[name] = 0.0
        
        # Combine signals using the specified method
        final_signal_value = 0
        confidence = 0.5  # Default confidence
        
        if self.combination_method == 'voting':
            # Simple majority vote
            votes = defaultdict(int)
            for signal_value in strategy_signals.values():
                votes[signal_value] += 1
            
            # Find signal with most votes
            if votes:
                final_signal_value = max(votes.items(), key=lambda x: x[1])[0]
                confidence = votes[final_signal_value] / len(strategy_signals)
        
        elif self.combination_method == 'weighted':
            # Weighted average of signals
            if self.weights:
                weighted_sum = 0
                total_weight = 0
                
                for name, signal_value in strategy_signals.items():
                    weight = self.weights.get(name, 0.0)
                    weighted_sum += signal_value * weight
                    total_weight += weight
                
                if total_weight > 0:
                    avg_signal = weighted_sum / total_weight
                    
                    # Convert to discrete signal
                    if avg_signal > 0.3:
                        final_signal_value = 1
                    elif avg_signal < -0.3:
                        final_signal_value = -1
                    else:
                        final_signal_value = 0
                    
                    # Confidence based on distance from zero
                    confidence = min(1.0, abs(avg_signal) * 2)  # Scale appropriately
        
        elif self.combination_method == 'consensus':
            # Require all strategies to agree for a signal
            unique_signals = set(strategy_signals.values())
            
            if len(unique_signals) == 1:
                # All strategies agree
                final_signal_value = next(iter(unique_signals))
                confidence = 1.0
            else:
                # No consensus
                final_signal_value = 0
                
                # Calculate confidence based on agreement level
                if 1 in strategy_signals.values() and -1 not in strategy_signals.values():
                    # Some buy signals, no sell signals
                    buy_count = sum(1 for v in strategy_signals.values() if v == 1)
                    confidence = buy_count / len(strategy_signals)
                    final_signal_value = 0  # Still neutral but leaning bullish
                elif -1 in strategy_signals.values() and 1 not in strategy_signals.values():
                    # Some sell signals, no buy signals
                    sell_count = sum(1 for v in strategy_signals.values() if v == -1)
                    confidence = sell_count / len(strategy_signals)
                    final_signal_value = 0  # Still neutral but leaning bearish
                else:
                    # Mixed signals
                    confidence = 0.5
        
        # Convert numeric signal to SignalType
        if final_signal_value == 1:
            final_signal_type = SignalType.BUY
        elif final_signal_value == -1:
            final_signal_type = SignalType.SELL
        else:
            final_signal_type = SignalType.NEUTRAL
        
        # Create signal
        self.last_signal = Signal(
            timestamp=bar["timestamp"],
            signal_type=final_signal_type,
            price=bar["Close"],
            rule_id=self.name,
            confidence=confidence,
            metadata={
                "method": self.combination_method,
                "strategy_signals": strategy_signals,
                "strategy_confidences": strategy_confidences
            }
        )
        
        return self.last_signal
    
    def reset(self):
        """Reset all strategies in the ensemble."""
        for strategy in self.strategies.values():
            strategy.reset()
        self.last_signal = None
