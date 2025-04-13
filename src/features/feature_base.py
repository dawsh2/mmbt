"""
Base Feature Module

This module defines the base Feature class and related abstractions for the feature layer
of the trading system. Features transform raw price data and indicators into meaningful
inputs for trading rules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd


class Feature(ABC):
    """
    Base class for all features in the trading system.
    
    Features transform raw price data and indicators into a format suitable for 
    use by trading rules. They encapsulate the logic for calculating derived values
    from market data while maintaining a consistent interface.
    """
    
    def __init__(self, 
                 name: str, 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        """
        Initialize a feature with parameters.
        
        Args:
            name: Unique identifier for the feature
            params: Dictionary of parameters for feature calculation
            description: Human-readable description of what the feature measures
        """
        self.name = name
        self.params = params or {}
        self.description = description
        self._validate_params()
        
    def _validate_params(self) -> None:
        """
        Validate the parameters provided to the feature.
        
        This method should be overridden by subclasses to provide
        specific parameter validation logic.
        
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> Any:
        """
        Calculate the feature value from the provided data.
        
        Args:
            data: Dictionary containing price data and calculated indicators
                 required by this feature
                 
        Returns:
            The calculated feature value (can be scalar, array, or any type)
        """
        pass
    
    @property
    def default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for this feature.
        
        Returns:
            Dictionary of default parameter values
        """
        return {}
    
    def __str__(self) -> str:
        """String representation of the feature."""
        return f"{self.name} (Feature)"
    
    def __repr__(self) -> str:
        """Detailed representation of the feature."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class FeatureSet:
    """
    A collection of features with convenience methods for batch calculation.
    
    FeatureSet provides a way to organize related features and calculate them
    together efficiently on the same dataset.
    """
    
    def __init__(self, features: Optional[List[Feature]] = None, name: str = ""):
        """
        Initialize a feature set with a list of features.
        
        Args:
            features: List of Feature objects to include in the set
            name: Optional name for the feature set
        """
        self.features = features or []
        self.name = name
        
    def add_feature(self, feature: Feature) -> None:
        """
        Add a feature to the set.
        
        Args:
            feature: Feature object to add
        """
        self.features.append(feature)
        
    def calculate_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all features in the set on the provided data.
        
        Args:
            data: Dictionary containing price data and indicators
            
        Returns:
            Dictionary mapping feature names to calculated values
        """
        result = {}
        for feature in self.features:
            result[feature.name] = feature.calculate(data)
        return result
    
    def to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate all features and return as a DataFrame.
        
        Args:
            data: Dictionary containing price data and indicators
            
        Returns:
            DataFrame with columns for each feature
        """
        features_dict = self.calculate_all(data)
        
        # Handle features that return arrays or Series of different lengths
        result_dict = {}
        
        # First identify the longest result to determine DataFrame length
        max_length = 0
        for feature_name, value in features_dict.items():
            if isinstance(value, (list, np.ndarray, pd.Series)):
                max_length = max(max_length, len(value))
        
        # If no sequence found, just return the dictionary as a single-row DataFrame
        if max_length == 0:
            return pd.DataFrame(features_dict, index=[0])
        
        # Convert each feature value to a Series of the right length
        for feature_name, value in features_dict.items():
            if isinstance(value, (list, np.ndarray)):
                # For list or array, pad with NaNs or truncate
                value_array = np.array(value)
                if len(value_array) < max_length:
                    padded = np.full(max_length, np.nan)
                    padded[-len(value_array):] = value_array
                    result_dict[feature_name] = padded
                else:
                    result_dict[feature_name] = value_array[-max_length:]
            elif isinstance(value, pd.Series):
                # For Series, reindex or truncate
                if len(value) < max_length:
                    result_dict[feature_name] = value.reindex(range(max_length), fill_value=np.nan)
                else:
                    result_dict[feature_name] = value.iloc[-max_length:]
            else:
                # For scalar values, repeat to fill the DataFrame
                result_dict[feature_name] = np.full(max_length, value)
        
        return pd.DataFrame(result_dict)
    
    def __len__(self) -> int:
        """Get the number of features in the set."""
        return len(self.features)
    
    def __getitem__(self, index) -> Feature:
        """Get a feature by index."""
        return self.features[index]
    
    def __iter__(self):
        """Iterate through features."""
        return iter(self.features)


class CompositeFeature(Feature):
    """
    A feature composed of multiple sub-features.
    
    CompositeFeature allows combining multiple features into a single feature,
    making it easier to create complex feature hierarchies.
    """
    
    def __init__(self, 
                 name: str, 
                 features: List[Feature],
                 combiner_func: callable,
                 params: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        """
        Initialize a composite feature.
        
        Args:
            name: Unique identifier for the feature
            features: List of component features
            combiner_func: Function that combines component feature values
            params: Dictionary of parameters
            description: Human-readable description
        """
        super().__init__(name, params, description)
        self.features = features
        self.combiner_func = combiner_func
        
    def calculate(self, data: Dict[str, Any]) -> Any:
        """
        Calculate the composite feature by combining sub-feature values.
        
        Args:
            data: Dictionary containing price data and indicators
                 
        Returns:
            The calculated composite feature value
        """
        # Calculate all component features
        feature_values = [feature.calculate(data) for feature in self.features]
        
        # Combine the feature values using the provided function
        return self.combiner_func(feature_values, **self.params)
    
    def __repr__(self) -> str:
        """Detailed representation of the composite feature."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"features=[{', '.join(f.name for f in self.features)}], "
                f"params={self.params})")


class StatefulFeature(Feature):
    """
    A feature that maintains internal state between calculations.
    
    StatefulFeature is useful for features that depend on historical values
    or require incremental updates, such as EMA-based features.
    """
    
    def __init__(self, 
                 name: str, 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "",
                 max_history: int = 100):
        """
        Initialize a stateful feature.
        
        Args:
            name: Unique identifier for the feature
            params: Dictionary of parameters
            description: Human-readable description
            max_history: Maximum history length to maintain
        """
        super().__init__(name, params, description)
        self.max_history = max_history
        self.history = []
        self.state = {}
        
    def update(self, data: Dict[str, Any]) -> Any:
        """
        Update the feature state with new data and return the new value.
        
        Args:
            data: Dictionary containing price data and indicators
            
        Returns:
            The newly calculated feature value
        """
        value = self.calculate(data)
        
        # Add to history, maintaining max length
        self.history.append(value)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return value
    
    def reset(self) -> None:
        """Reset the feature's internal state."""
        self.history = []
        self.state = {}
