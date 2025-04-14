"""
Component interfaces and factories for the optimization framework.
"""

from abc import ABC, abstractmethod

class OptimizableComponent(ABC):
    """Abstract base class for any component that can be optimized."""
    
    @abstractmethod
    def evaluate(self, data_handler, metric='return'):
        """Evaluate component performance using specified metric."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the component's state."""
        pass

class ComponentFactory(ABC):
    """Factory for creating instances of optimizable components."""
    
    @abstractmethod
    def create(self, component_class, params):
        """Create a component instance with the given parameters."""
        pass

# Rule-specific implementations
class RuleFactory(ComponentFactory):
    """Factory for creating rule instances."""
    
    def create(self, rule_class, params):
        """Create a rule instance with the given parameters."""
        return rule_class(params)



# Regime detector-specific implementations
class RegimeDetectorFactory(ComponentFactory):
    """Factory for creating regime detector instances."""
    
    def create(self, detector_class, params):
        """Create a regime detector instance with the given parameters."""
        return detector_class(params)

# Strategy-specific implementations
class StrategyFactory(ComponentFactory):
    """Factory for creating strategy instances."""
    
    def create(self, strategy_class, params):
        """Create a strategy instance with the given parameters."""
        return strategy_class(params)


# Add to components.py
class WeightedStrategyFactory(ComponentFactory):
    """Factory for creating weighted strategies."""
    
    def create(self, component_class, params):
        """
        Create a weighted strategy with the given components and parameters.
        
        Args:
            component_class: Strategy class to create
            params: Dictionary of parameters including components and weights
            
        Returns:
            WeightedComponentStrategy: A weighted strategy
        """
        components = params.get('components', [])
        weights = params.get('weights', [])
        buy_threshold = params.get('buy_threshold', 0.5)
        sell_threshold = params.get('sell_threshold', -0.5)
        
        return WeightedComponentStrategy(
            components=components,
            weights=weights,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )    
