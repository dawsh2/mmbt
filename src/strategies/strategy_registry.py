"""
Strategy Registry Module

This module provides a registry for strategies that allows dynamic registration
and discovery of strategy implementations.
"""

from typing import Dict, Type, List, Any, Optional

class StrategyRegistry:
    """Registry of available strategies.
    
    This class provides a centralized registry where strategies can be registered
    and retrieved by name. It supports decorator-based registration.
    """
    
    _strategies = {}  # Type: Dict[str, Type[Strategy]]
    _categories = {}  # Type: Dict[str, list]
    
    @classmethod
    def register(cls, category: str = "general"):
        """Decorator to register a strategy class.
        
        Args:
            category: Category to place the strategy in
            
        Returns:
            Decorator function
        """
        def decorator(strategy_class):
            strategy_name = strategy_class.__name__
            cls._strategies[strategy_name] = strategy_class
            
            # Add to category
            if category not in cls._categories:
                cls._categories[category] = []
            cls._categories[category].append(strategy_name)
            
            return strategy_class
        return decorator
    
    @classmethod
    def get_strategy_class(cls, name: str):
        """Get a strategy class by name.
        
        Args:
            name: Name of the strategy class
            
        Returns:
            Type[Strategy]: The strategy class
            
        Raises:
            ValueError: If strategy name is not found
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[name]
    
    @classmethod
    def list_strategies(cls) -> Dict[str, List[str]]:
        """List all registered strategies by category.
        
        Returns:
            Dict[str, list]: Mapping of categories to strategy names
        """
        return cls._categories.copy()
