"""
Registry for market regime detectors.
"""

class DetectorRegistry:
    """
    Registry for market regime detectors.
    
    This class maintains a registry of available detector classes organized by
    category. It provides decorator-based registration and methods to retrieve
    detectors by name.
    """
    
    def __init__(self):
        """Initialize an empty detector registry."""
        self._detectors = {}
        self._categories = {}
    
    def register(self, category="general"):
        """
        Decorator to register a detector class.
        
        Args:
            category: Category to register the detector under
            
        Returns:
            function: Decorator function
        """
        def decorator(cls):
            # Store the detector class in the registry
            self._detectors[cls.__name__] = cls
            
            # Add to category
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(cls.__name__)
            
            return cls
        return decorator
    
    def get_detector_class(self, name):
        """
        Get a detector class by name.
        
        Args:
            name: Name of the detector class
            
        Returns:
            class: The detector class, or None if not found
        """
        return self._detectors.get(name)
    
    def list_detectors(self, category=None):
        """
        List available detectors, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            list: Names of available detectors
        """
        if category is None:
            return list(self._detectors.keys())
        return self._categories.get(category, [])
    
    def list_categories(self):
        """
        List available detector categories.
        
        Returns:
            list: Names of available categories
        """
        return list(self._categories.keys())
