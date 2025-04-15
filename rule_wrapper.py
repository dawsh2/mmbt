#!/usr/bin/env python3
# rules_wrapper.py - Wrapper for standard rules to handle BarEvent objects

from src.rules.rule_base import Rule
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
        # Properly initialize the Rule parent class
        super().__init__(
            name=original_rule.name,
            params=original_rule.params,
            description=original_rule.description
        )
        
        # Store the original rule
        self.original_rule = original_rule
        
        # Copy signals from original rule if any exist
        if hasattr(original_rule, 'signals') and original_rule.signals:
            self.signals = list(original_rule.signals)
            
        # Copy state from original rule if any exists
        if hasattr(original_rule, 'state') and original_rule.state:
            self.state = dict(original_rule.state)
    
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
        if hasattr(data, 'data') and isinstance(data.data, dict):
            # If it's an Event with a data dictionary
            bar_data = data.data
        elif isinstance(data, dict):
            # If it's already a dictionary
            bar_data = data
        else:
            # Try to extract common fields from an unknown object
            bar_data = {}
            # Copy timestamp if available
            if hasattr(data, 'timestamp'):
                bar_data['timestamp'] = data.timestamp
                
            # Try to extract OHLCV data
            for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if hasattr(data, field.lower()):
                    bar_data[field] = getattr(data, field.lower())
                elif hasattr(data, field):
                    bar_data[field] = getattr(data, field)
                    
            # If it's a DataFrame row, try to extract by column names
            if hasattr(data, 'get') and callable(data.get):
                for field in ['open', 'high', 'low', 'close', 'volume']:
                    if data.get(field) is not None:
                        bar_data[field.capitalize()] = data.get(field)
            
            # Add symbol if available
            if hasattr(data, 'symbol'):
                bar_data['symbol'] = data.symbol
            
        # Call the original rule's generate_signal method with the extracted data
        return self.original_rule.generate_signal(bar_data)
    
    def on_bar(self, event_or_data):
        """
        Process a bar event and generate a trading signal.
        
        This wrapper properly handles both Event objects and raw data.
        
        Args:
            event_or_data: Event object or dictionary with bar data
            
        Returns:
            Signal object with the trading decision
        """
        # Generate signal using our data extraction method
        signal = self.generate_signal(event_or_data)
        
        # Store the signal in our history
        self.signals.append(signal)
        
        # Also update the original rule's signal history
        if hasattr(self.original_rule, 'signals'):
            self.original_rule.signals.append(signal)
            
        return signal
    
    def reset(self):
        """Reset both the wrapper and the original rule."""
        super().reset()
        self.original_rule.reset()
    
    def update_state(self, key, value):
        """Update state in both wrapper and original rule."""
        super().update_state(key, value)
        if hasattr(self.original_rule, 'update_state'):
            self.original_rule.update_state(key, value)
    
    def get_state(self, key=None, default=None):
        """Get state from the original rule."""
        if hasattr(self.original_rule, 'get_state'):
            return self.original_rule.get_state(key, default)
        return super().get_state(key, default)

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
