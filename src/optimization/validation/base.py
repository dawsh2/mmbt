"""
Base validator interface for the optimization framework.
"""

from abc import ABC, abstractmethod

class Validator(ABC):
    """Base abstract class for validation components."""
    
    @abstractmethod
    def validate(self, component_factory, optimization_method, data_handler, configs=None, 
                 metric='sharpe', verbose=True, **kwargs):
        """
        Validate a component or strategy using the specified method.
        
        Args:
            component_factory: Factory for creating component instances
            optimization_method: Method to use for optimization
            data_handler: Data handler providing market data
            configs: Component configuration for optimization
            metric: Performance metric to optimize
            verbose: Whether to print progress information
            **kwargs: Additional parameters
            
        Returns:
            dict: Validation results
        """
        pass
