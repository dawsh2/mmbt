"""
Factory for creating market regime detectors.
"""

from .detector_registry import DetectorRegistry

class DetectorFactory:
    """
    Factory for creating regime detector instances.
    
    This class provides methods to instantiate detectors from their registered classes,
    either by name or from configuration dictionaries.
    """
    
    def __init__(self, registry=None):
        """
        Initialize the detector factory.
        
        Args:
            registry: Optional DetectorRegistry instance
        """
        self.registry = registry or DetectorRegistry()
    
    def create_detector(self, name_or_class, config=None):
        """
        Create a detector instance.
        
        Args:
            name_or_class: String name or class reference
            config: Optional configuration dictionary
            
        Returns:
            DetectorBase: Instantiated detector
        """
        # Handle the case where a class is provided directly
        if isinstance(name_or_class, type):
            return name_or_class(config=config)
        
        # Get the class from the registry
        detector_class = self.registry.get_detector_class(name_or_class)
        if detector_class is None:
            raise ValueError(f"Unknown detector: {name_or_class}")
        
        # Create and return the detector instance
        return detector_class(config=config)
    
    def create_from_config(self, config):
        """
        Create a detector from a configuration dictionary.
        
        Args:
            config: Dictionary with 'type' and optional 'params'
            
        Returns:
            DetectorBase: Instantiated detector
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        
        detector_type = config.get('type')
        if not detector_type:
            raise ValueError("Config must contain 'type' key")
        
        params = config.get('params', {})
        
        return self.create_detector(detector_type, config=params)
    
    def create_composite(self, configs, combination_method='majority'):
        """
        Create a composite detector from multiple configurations.
        
        Args:
            configs: List of detector configurations
            combination_method: Method to combine detector outputs
            
        Returns:
            CompositeDetector: Composite detector instance
        """
        from .detectors.composite_detectors import CompositeDetector
        
        detectors = [self.create_from_config(cfg) for cfg in configs]
        return CompositeDetector(detectors=detectors, combination_method=combination_method)
