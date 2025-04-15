#!/usr/bin/env python3
# rules_wrapper.py - Wrapper for standard rules to handle BarEvent objects

from src.rules import Rule
from src.signals import Signal

class RuleWrapper(Rule):
    """
    A wrapper for rule classes that adds proper handling of BarEvent objects.
    This wrapper ensures that the original rule's generate_signal method receives
    a dictionary, not a BarEvent object.
    """
    
    def __init__(self, original_rule):
        """
        Initialize the wrapper with the original rule.
        
        Args:
            original_rule: The rule instance to wrap
        """
        self.original_rule = original_rule
        # Copy attributes from original rule
        self.name = original_rule.name
        self.params = original_rule.params
        self.description = original_rule.description
        self.state = original_rule.state
        self.signals = original_rule.signals
    
    def _validate_params(self):
        """Delegate to original rule's validation."""
        if hasattr(self.original_rule, '_validate_params'):
            self.original_rule._validate_params()
    
    def generate_signal(self, data):
        """
        Generate a signal, ensuring data is a dictionary, not an event.
        
        Args:
            data: BarEvent object or dictionary
            
        Returns:
            Signal or None
        """
        # Extract data from event if needed
        if hasattr(data, 'data'):
            bar_data = data.data
        else:
            bar_data = data
            
        # Call the original rule's generate_signal method with the extracted data
        return self.original_rule.generate_signal(bar_data)
    
    def reset(self):
        """Reset both the wrapper and the original rule."""
        super().reset()
        self.original_rule.reset()
    
    def __getattr__(self, name):
        """Delegate any other attribute access to the original rule."""
        return getattr(self.original_rule, name)

def wrap_rule(rule):
    """
    Wrap a rule to ensure it can handle BarEvent objects properly.
    
    Args:
        rule: Rule instance to wrap
        
    Returns:
        RuleWrapper: Wrapped rule
    """
    return RuleWrapper(rule)

def wrap_rules(rules):
    """
    Wrap multiple rules.
    
    Args:
        rules: List of rule instances
        
    Returns:
        List of wrapped rules
    """
    return [wrap_rule(rule) for rule in rules]
