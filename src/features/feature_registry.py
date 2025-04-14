"""
Feature Registry Module

This module provides a registry system for features, allowing them to be
registered, discovered, and instantiated by name throughout the trading system.
"""

from typing import Dict, Type, Optional, List, Any, Callable, Union
import logging
import inspect

# Set up logging
logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Registry for feature classes in the trading system.
    
    The FeatureRegistry maintains a mapping of feature names to their implementing
    classes, allowing features to be instantiated by name.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for the registry."""
        if cls._instance is None:
            cls._instance = super(FeatureRegistry, cls).__new__(cls)
            cls._instance._features = {}
            cls._instance._categories = {}
        return cls._instance

    def register(self, 
                 feature_class, 
                 name: Optional[str] = None,
                 category: str = "general") -> None:
        """
        Register a feature class with the registry.
        
        Args:
            feature_class: The Feature class to register
            name: Optional name to register the feature under (defaults to class name)
            category: Category to group the feature under
            
        Raises:
            ValueError: If a feature with the same name is already registered
        """
        # Use class name if no name provided
        if name is None:
            name = feature_class.__name__
            
        # Check if feature is already registered
        if name in self._features:
            logger.warning(f"Feature '{name}' is already registered. Overwriting.")
            
        self._features[name] = feature_class
        
        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logger.debug(f"Registered feature '{name}' in category '{category}'")
    
    def get_feature_class(self, name: str):
        """
        Get a feature class by name.
        
        Args:
            name: Name of the feature to retrieve
            
        Returns:
            The feature class
            
        Raises:
            KeyError: If no feature with the given name is registered
        """
        if name not in self._features:
            raise KeyError(f"No feature registered with name '{name}'")
        return self._features[name]
    
    def create_feature(self, 
                       name: str, 
                       params: Optional[Dict[str, Any]] = None,
                       feature_name: Optional[str] = None):
        """
        Create an instance of a feature by name.
        
        Args:
            name: Name of the feature class to instantiate
            params: Parameters to pass to the feature constructor
            feature_name: Optional name for the created feature instance
                        (defaults to the registered name)
            
        Returns:
            An instance of the requested feature
            
        Raises:
            KeyError: If no feature with the given name is registered
        """
        feature_class = self.get_feature_class(name)
        
        # Use registered name if no instance name provided
        if feature_name is None:
            feature_name = name
            
        # Create the feature instance
        return feature_class(name=feature_name, params=params or {})
    
    def list_features(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered features, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of feature names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._features.keys())
    
    def list_categories(self) -> List[str]:
        """
        List all feature categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def clear(self) -> None:
        """Clear all registered features (useful for testing)."""
        self._features = {}
        self._categories = {}
        logger.debug("Cleared feature registry")


# Module-level functions
def get_registry() -> FeatureRegistry:
    """
    Get the global feature registry instance.

    Returns:
        The FeatureRegistry singleton instance
    """
    return FeatureRegistry()


def register_feature(category="general"):
    """
    Decorator to register a feature class in the registry.

    Args:
        category (str): Category for the feature

    Returns:
        decorator function
    """
    def decorator(cls):
        registry = get_registry()
        registry.register(cls, name=None, category=category)
        return cls
    return decorator


def register_features_in_module(module, category: str = "general") -> None:
    """
    Register all Feature classes in a module with the registry.

    Args:
        module: The module object containing feature classes
        category: Category to group the features under
    """
    registry = get_registry()

    # Import Feature inside the function to avoid circular imports
    from src.features.feature_base import Feature
    
    # Find all classes in module that are Feature subclasses
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, Feature) and 
            obj != Feature and
            obj.__module__ == module.__name__):
            registry.register(obj, name, category)
            logger.debug(f"Auto-registered feature '{name}' from module {module.__name__}")
