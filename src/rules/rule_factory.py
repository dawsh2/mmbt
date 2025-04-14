"""
Rule Factory Module

This module provides factory functions for creating and configuring rule instances
with proper parameter handling and validation.
"""
from typing import Callable, Dict, List, Any, Optional, Union, Tuple, Type
import logging
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

from src.rules.rule_base import Rule, CompositeRule
from src.rules.rule_registry import get_registry, RuleRegistry

# Set up logging
logger = logging.getLogger(__name__)


class RuleFactory:
    """
    Factory for creating rule instances with proper parameter handling.
    
    This class provides methods for creating rule instances with various
    parameter combinations and validation.
    """
    
    def __init__(self, registry: Optional[RuleRegistry] = None):
        """
        Initialize the rule factory.
        
        Args:
            registry: Optional rule registry to use (defaults to global registry)
        """
        self.registry = registry or get_registry()
    
    def create_rule(self, 
                    rule_name: str, 
                    params: Optional[Dict[str, Any]] = None,
                    instance_name: Optional[str] = None,
                    **kwargs) -> Rule:
        """
        Create a rule instance with the given parameters.
        
        Args:
            rule_name: Name of the rule class to instantiate
            params: Parameters for the rule
            instance_name: Optional name for the rule instance
            **kwargs: Additional keyword arguments for the rule constructor
            
        Returns:
            Instantiated rule
            
        Raises:
            KeyError: If rule is not found in registry
        """
        return self.registry.create_rule(
            name=rule_name,
            params=params,
            rule_name=instance_name,
            **kwargs
        )
    
    def create_rule_variants(self, 
                            rule_name: str,
                            param_grid: Dict[str, List[Any]],
                            base_params: Optional[Dict[str, Any]] = None,
                            name_format: str = "{rule}_{param}_{value}") -> List[Rule]:
        """
        Create multiple rule instances with different parameter combinations.
        
        Args:
            rule_name: Name of the rule class to instantiate
            param_grid: Dictionary mapping parameter names to lists of values
            base_params: Base parameters to use for all variants
            name_format: Format string for naming rule instances
            
        Returns:
            List of rule instances with different parameter combinations
        """
        rule_variants = []
        
        # Get all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Calculate all combinations of parameter values
        param_combinations = list(itertools.product(*param_values))
        
        for combo in param_combinations:
            # Create parameter dictionary for this combination
            combo_params = {k: v for k, v in zip(param_names, combo)}
            
            # Merge with base parameters
            if base_params:
                merged_params = base_params.copy()
                merged_params.update(combo_params)
            else:
                merged_params = combo_params
            
            # Create instance name
            instance_name = rule_name
            for param_name, param_value in combo_params.items():
                instance_name = name_format.format(
                    rule=rule_name,
                    param=param_name,
                    value=param_value
                )
            
            # Create rule instance
            try:
                rule = self.create_rule(
                    rule_name=rule_name,
                    params=merged_params,
                    instance_name=instance_name
                )
                rule_variants.append(rule)
            except Exception as e:
                logger.warning(f"Failed to create rule variant {instance_name}: {str(e)}")
        
        return rule_variants
    
    def create_composite_rule(self,
                             name: str,
                             rule_configs: List[Union[str, Dict[str, Any], Rule]],
                             aggregation_method: str = "majority",
                             params: Optional[Dict[str, Any]] = None) -> CompositeRule:
        """
        Create a composite rule from multiple rule configurations.
        
        Args:
            name: Name for the composite rule
            rule_configs: List of rule configurations, which can be:
                          - Rule instance
                          - Rule name (string)
                          - Dict with 'name' and optional 'params' keys
            aggregation_method: Method to combine signals
            params: Parameters for the composite rule
            
        Returns:
            CompositeRule instance
            
        Raises:
            ValueError: If rule_configs is empty
        """
        if not rule_configs:
            raise ValueError("Rule configurations list cannot be empty")
        
        # Create sub-rules
        rules = []
        for config in rule_configs:
            if isinstance(config, Rule):
                # If already a Rule instance, use it directly
                rules.append(config)
            elif isinstance(config, str):
                # If a string, create rule with default parameters
                rules.append(self.create_rule(rule_name=config))
            elif isinstance(config, dict) and 'name' in config:
                # If a dict with 'name', create rule with specified parameters
                rule_name = config['name']
                rule_params = config.get('params', {})
                instance_name = config.get('instance_name', rule_name)
                rules.append(self.create_rule(
                    rule_name=rule_name,
                    params=rule_params,
                    instance_name=instance_name
                ))
            else:
                # Skip invalid configurations
                logger.warning(f"Invalid rule configuration: {config}")
                continue
        
        # Create composite rule
        return CompositeRule(
            name=name,
            rules=rules,
            aggregation_method=aggregation_method,
            params=params or {}
        )
    
    def create_from_config(self, config: Dict[str, Any]) -> Rule:
        """
        Create a rule from a configuration dictionary.
        
        Args:
            config: Dictionary with rule configuration
            
        Returns:
            Rule instance
            
        Raises:
            ValueError: If config is invalid
        """
        if 'type' not in config:
            raise ValueError("Rule configuration must include 'type'")
        
        rule_type = config['type']
        
        if rule_type == 'composite':
            # Create composite rule
            if 'rules' not in config:
                raise ValueError("Composite rule config must include 'rules'")
                
            # Create sub-rules recursively
            sub_rules = []
            for sub_config in config['rules']:
                sub_rule = self.create_from_config(sub_config)
                sub_rules.append(sub_rule)
                
            # Create composite rule
            return CompositeRule(
                name=config.get('name', 'composite_rule'),
                rules=sub_rules,
                aggregation_method=config.get('aggregation_method', 'majority'),
                params=config.get('params', {}),
                description=config.get('description', '')
            )
        else:
            # Create regular rule
            return self.create_rule(
                rule_name=rule_type,
                params=config.get('params', {}),
                instance_name=config.get('name', None),
                description=config.get('description', '')
            )


class RuleOptimizer:
    """
    Optimizer for rule parameters based on performance metrics.
    
    This class provides methods for optimizing rule parameters to
    maximize performance metrics using different optimization methods.
    """
    
    def __init__(self, 
                rule_factory: RuleFactory,
                evaluation_func: Callable[[Rule], float]):
        """
        Initialize the rule optimizer.
        
        Args:
            rule_factory: RuleFactory instance for creating rules
            evaluation_func: Function that takes a rule and returns a performance score
        """
        self.rule_factory = rule_factory
        self.evaluation_func = evaluation_func
    
    def optimize_grid_search(self,
                           rule_name: str,
                           param_grid: Dict[str, List[Any]],
                           base_params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
        """
        Optimize rule parameters using grid search.
        
        Args:
            rule_name: Name of the rule to optimize
            param_grid: Dictionary mapping parameter names to lists of values
            base_params: Base parameters to use for all variants
            
        Returns:
            Tuple of (best parameters, best score)
        """
        best_params = None
        best_score = float('-inf')
        
        # Create all parameter combinations
        variants = self.rule_factory.create_rule_variants(
            rule_name=rule_name,
            param_grid=param_grid,
            base_params=base_params
        )
        
        # Evaluate each variant
        for rule in variants:
            score = self.evaluation_func(rule)
            
            if score > best_score:
                best_score = score
                best_params = rule.params.copy()
        
        return best_params, best_score
    
    def optimize_random_search(self,
                             rule_name: str,
                             param_distributions: Dict[str, Union[List[Any], Callable]],
                             base_params: Optional[Dict[str, Any]] = None,
                             n_iterations: int = 10) -> Tuple[Dict[str, Any], float]:
        """
        Optimize rule parameters using random search.
        
        Args:
            rule_name: Name of the rule to optimize
            param_distributions: Dictionary mapping parameter names to distributions
                               or lists of values
            base_params: Base parameters to use for all variants
            n_iterations: Number of random combinations to try
            
        Returns:
            Tuple of (best parameters, best score)
        """
        best_params = None
        best_score = float('-inf')
        
        for _ in range(n_iterations):
            # Sample parameters from distributions
            params = {}
            for param_name, distribution in param_distributions.items():
                if callable(distribution):
                    # If distribution is a function, call it to get a random value
                    params[param_name] = distribution()
                elif isinstance(distribution, list):
                    # If distribution is a list, sample randomly from it
                    params[param_name] = np.random.choice(distribution)
                else:
                    # Otherwise, use the value directly
                    params[param_name] = distribution
            
            # Merge with base parameters
            if base_params:
                merged_params = base_params.copy()
                merged_params.update(params)
            else:
                merged_params = params
            
            # Create and evaluate rule
            try:
                rule = self.rule_factory.create_rule(
                    rule_name=rule_name,
                    params=merged_params
                )
                
                score = self.evaluation_func(rule)
                
                if score > best_score:
                    best_score = score
                    best_params = merged_params.copy()
            except Exception as e:
                logger.warning(f"Failed to evaluate parameters {merged_params}: {str(e)}")
        
        return best_params, best_score


# Helper functions for creating rules
def create_rule(rule_name: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Rule:
    """
    Create a rule instance using the global registry.
    
    Args:
        rule_name: Name of the rule class to instantiate
        params: Parameters for the rule
        **kwargs: Additional keyword arguments for the rule constructor
        
    Returns:
        Instantiated rule
    """
    factory = RuleFactory()
    return factory.create_rule(rule_name=rule_name, params=params, **kwargs)


def create_composite_rule(name: str, 
                         rule_configs: List[Union[str, Dict[str, Any], Rule]],
                         aggregation_method: str = "majority",
                         params: Optional[Dict[str, Any]] = None) -> CompositeRule:
    """
    Create a composite rule from multiple rule configurations.
    
    Args:
        name: Name for the composite rule
        rule_configs: List of rule configurations
        aggregation_method: Method to combine signals
        params: Parameters for the composite rule
        
    Returns:
        CompositeRule instance
    """
    factory = RuleFactory()
    return factory.create_composite_rule(
        name=name,
        rule_configs=rule_configs,
        aggregation_method=aggregation_method,
        params=params
    )
