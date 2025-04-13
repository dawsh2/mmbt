"""
Grid search optimization module for trading system components.
"""

import itertools
import numpy as np
from abc import ABC, abstractmethod

class GridOptimizer:
    """General-purpose grid search optimizer for any component."""
    
    def __init__(self, component_factory, evaluation_method, top_n=5):
        """
        Initialize the grid optimizer.
        
        Args:
            component_factory: Factory to create component instances
            evaluation_method: Function to evaluate a component
            top_n: Number of top components to select
        """
        self.component_factory = component_factory
        self.evaluate = evaluation_method
        self.top_n = top_n
        self.best_params = {}
        self.best_scores = {}
        self.best_components = {}
    
    def optimize(self, configs, data_handler, metric='return', verbose=True):
        """
        Optimize components using grid search.
        
        Args:
            configs: List of (ComponentClass, param_ranges) tuples
            data_handler: Data handler providing market data
            metric: Metric to optimize for ('return', 'sharpe', etc.)
            verbose: Whether to print progress
            
        Returns:
            dict: Mapping of component indices to optimized instances
        """
        # Expand parameter grids
        expanded_configs = self._expand_param_grid(configs, verbose)
        
        all_performances = {}
        
        # Evaluate each component with each parameter set
        for i, (component_class, param_sets) in enumerate(expanded_configs):
            component_performances = {}
            
            if verbose:
                print(f"Optimizing {component_class.__name__} with {len(param_sets)} parameter sets...")
            
            for params in param_sets:
                if verbose:
                    print(f"  Testing params: {params}")
                
                # Create and evaluate component
                component = self.component_factory.create(component_class, params)
                score = self.evaluate(component, data_handler, metric)
                component_performances[tuple(sorted(params.items()))] = score
                
                # Clean up
                if hasattr(component, 'reset'):
                    component.reset()
            
            # Find best parameters for this component
            if component_performances:
                best_params_tuple = max(component_performances, key=component_performances.get)
                best_score = component_performances[best_params_tuple]
                best_params = dict(best_params_tuple)
                all_performances[i] = (best_params, best_score, component_class)
                
                if verbose:
                    print(f"  Best params: {best_params}, Score: {best_score:.4f}")
            elif verbose:
                print(f"  No valid parameter sets found.")
        
        # Select top components
        sorted_components = sorted(all_performances.items(), key=lambda item: item[1][1], reverse=True)
        top_components = sorted_components[:self.top_n]
        
        # Store results
        self.best_params = {idx: data[0] for idx, data in top_components}
        self.best_scores = {idx: data[1] for idx, data in top_components}
        self.best_components = {
            idx: self.component_factory.create(data[2], self.best_params[idx]) 
            for idx, data in top_components
        }
        
        return self.best_components
    
    def _expand_param_grid(self, configs, verbose=False):
        """Expand parameter ranges into a grid of parameter sets."""
        expanded_configs = []
        
        for component_class, param_ranges in configs:
            if verbose:
                print(f"Expanding parameters for {component_class.__name__}:")
                print(f"  Original param ranges: {param_ranges}")
            
            param_names = param_ranges.keys()
            param_values = param_ranges.values()
            param_combinations = list(itertools.product(*param_values))
            parameter_sets = [dict(zip(param_names, combo)) for combo in param_combinations]
            
            if verbose:
                print(f"  Generated {len(parameter_sets)} parameter sets")
                print(f"  First few sets: {parameter_sets[:3]}")
            
            expanded_configs.append((component_class, parameter_sets))
        
        return expanded_configs
