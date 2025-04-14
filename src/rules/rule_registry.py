"""
Rule Registry Module

This module provides a registry system for rules, allowing them to be
registered, discovered, and instantiated by name throughout the trading system.
"""

from typing import Dict, Type, Optional, List, Any, Callable, Union
import logging
import inspect
from src.rules.rule_base import Rule

# Set up logging
logger = logging.getLogger(__name__)


class RuleRegistry:
    """
    Registry for rule classes in the trading system.
    
    The RuleRegistry maintains a mapping of rule names to their implementing
    classes, allowing rules to be instantiated by name.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for the registry."""
        if cls._instance is None:
            cls._instance = super(RuleRegistry, cls).__new__(cls)
            cls._instance._rules = {}
            cls._instance._categories = {}
        return cls._instance
    
    def register(self, 
                 rule_class: Type[Rule], 
                 name: Optional[str] = None,
                 category: str = "general") -> None:
        """
        Register a rule class with the registry.
        
        Args:
            rule_class: The Rule class to register
            name: Optional name to register the rule under (defaults to class name)
            category: Category to group the rule under
            
        Raises:
            ValueError: If a rule with the same name is already registered
        """
        # Use class name if no name provided
        if name is None:
            name = rule_class.__name__
            
        # Check if rule is already registered
        if name in self._rules:
            logger.warning(f"Rule '{name}' is already registered. Overwriting.")
            
        self._rules[name] = rule_class
        
        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logger.debug(f"Registered rule '{name}' in category '{category}'")
    
    def get_rule_class(self, name: str) -> Type[Rule]:
        """
        Get a rule class by name.
        
        Args:
            name: Name of the rule to retrieve
            
        Returns:
            The rule class
            
        Raises:
            KeyError: If no rule with the given name is registered
        """
        if name not in self._rules:
            raise KeyError(f"No rule registered with name '{name}'")
        return self._rules[name]
    
    def create_rule(self, 
                    name: str, 
                    params: Optional[Dict[str, Any]] = None,
                    rule_name: Optional[str] = None,
                    **kwargs) -> Rule:
        """
        Create an instance of a rule by name.
        
        Args:
            name: Name of the rule class to instantiate
            params: Parameters to pass to the rule constructor
            rule_name: Optional name for the created rule instance
                      (defaults to the registered name)
            **kwargs: Additional keyword arguments to pass to the constructor
            
        Returns:
            An instance of the requested rule
            
        Raises:
            KeyError: If no rule with the given name is registered
        """
        rule_class = self.get_rule_class(name)
        
        # Use registered name if no instance name provided
        if rule_name is None:
            rule_name = name
            
        # Create the rule instance
        return rule_class(name=rule_name, params=params or {}, **kwargs)
    
    def list_rules(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered rules, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of rule names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._rules.keys())
    
    def list_categories(self) -> List[str]:
        """
        List all rule categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def clear(self) -> None:
        """Clear all registered rules (useful for testing)."""
        self._rules = {}
        self._categories = {}
        logger.debug("Cleared rule registry")


# Create decorator for easy rule registration
def register_rule(name: Optional[str] = None, category: str = "general"):
    """
    Decorator for registering rule classes with the registry.
    
    Args:
        name: Optional name for the rule (defaults to class name)
        category: Category to group the rule under
        
    Returns:
        Decorator function that registers the rule class
    """
    def decorator(cls):
        registry = RuleRegistry()
        registry.register(cls, name, category)
        return cls
    return decorator


# Function to register all Rule classes in a module
def register_rules_in_module(module, category: str = "general") -> None:
    """
    Register all Rule classes in a module with the registry.
    
    Args:
        module: The module object containing rule classes
        category: Category to group the rules under
    """
    registry = RuleRegistry()
    
    # Find all classes in module that are Rule subclasses
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, Rule) and 
            obj != Rule and
            obj.__module__ == module.__name__):
            registry.register(obj, name, category)
            logger.debug(f"Auto-registered rule '{name}' from module {module.__name__}")


# Global function to get registry instance
def get_registry() -> RuleRegistry:
    """
    Get the global rule registry instance.
    
    Returns:
        The RuleRegistry singleton instance
    """
    return RuleRegistry()
