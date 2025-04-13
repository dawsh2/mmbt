"""
Feature Utilities Module

This module provides utility functions for combining and transforming
features in various ways to create complex feature compositions.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from .feature_base import Feature, CompositeFeature


def combine_features(features: List[Feature], 
                     combiner_func: Callable,
                     name: str = "composite_feature", 
                     params: Optional[Dict[str, Any]] = None,
                     description: str = "Combined features") -> CompositeFeature:
    """
    Combine multiple features into a composite feature.
    
    Args:
        features: List of feature objects to combine
        combiner_func: Function that takes a list of feature values and returns a combined value
        name: Name for the new composite feature
        params: Parameters to pass to the combiner function
        description: Description of the new feature
        
    Returns:
        CompositeFeature object
    """
    return CompositeFeature(
        name=name,
        features=features,
        combiner_func=combiner_func,
        params=params or {},
        description=description
    )


def weighted_average_combiner(feature_values: List[Any], **params) -> float:
    """
    Combine feature values using a weighted average.
    
    Args:
        feature_values: List of feature values to combine
        params: Must include 'weights' as a list of weights
        
    Returns:
        Weighted average of feature values
    """
    weights = params.get('weights', None)
    
    # Ensure all values are numeric
    numeric_values = []
    for value in feature_values:
        if isinstance(value, (int, float)):
            numeric_values.append(value)
        elif isinstance(value, dict) and 'value' in value:
            numeric_values.append(value['value'])
        elif isinstance(value, dict) and 'signal' in value:
            numeric_values.append(value['signal'])
        else:
            numeric_values.append(0)  # Default for non-numeric
    
    # If weights not provided, use equal weights
    if not weights or len(weights) != len(numeric_values):
        weights = [1.0 / len(numeric_values)] * len(numeric_values)
    
    # Calculate weighted average
    return sum(w * v for w, v in zip(weights, numeric_values)) / sum(weights)


def logical_combiner(feature_values: List[Any], **params) -> bool:
    """
    Combine feature values using logical operations.
    
    Args:
        feature_values: List of feature values to combine
        params: Must include 'operation' as one of 'and', 'or', 'majority'
        
    Returns:
        Boolean result of logical operation
    """
    operation = params.get('operation', 'and')
    threshold = params.get('threshold', 0.5)  # For majority voting
    
    # Convert feature values to boolean
    bool_values = []
    for value in feature_values:
        if isinstance(value, bool):
            bool_values.append(value)
        elif isinstance(value, (int, float)):
            bool_values.append(bool(value))
        elif isinstance(value, dict) and 'signal' in value:
            bool_values.append(value['signal'] > 0)
        else:
            bool_values.append(False)  # Default for non-convertible
    
    if operation == 'and':
        return all(bool_values)
    elif operation == 'or':
        return any(bool_values)
    elif operation == 'majority':
        if bool_values:
            return sum(bool_values) / len(bool_values) >= threshold
        return False
    else:
        raise ValueError(f"Unknown logical operation: {operation}")


def threshold_combiner(feature_values: List[Any], **params) -> int:
    """
    Combine feature values using thresholds to generate a signal.
    
    Args:
        feature_values: List of feature values to combine
        params: Dictionary containing:
            - 'thresholds': List of thresholds for each feature
            - 'directions': List of expected directions (1 or -1) for each feature
            - 'min_agreements': Minimum number of agreements needed for a signal
        
    Returns:
        Signal value: 1 (buy), -1 (sell), or 0 (neutral)
    """
    thresholds = params.get('thresholds', [0] * len(feature_values))
    directions = params.get('directions', [1] * len(feature_values))
    min_agreements = params.get('min_agreements', len(feature_values))
    
    # Extract signal values
    signal_values = []
    for i, value in enumerate(feature_values):
        if isinstance(value, (int, float)):
            signal_values.append(value)
        elif isinstance(value, dict) and 'value' in value:
            signal_values.append(value['value'])
        elif isinstance(value, dict) and 'signal' in value:
            signal_values.append(value['signal'])
        else:
            signal_values.append(0)  # Default for non-extractable
    
    # Count agreements
    buy_agreements = 0
    sell_agreements = 0
    
    for i, value in enumerate(signal_values):
        threshold = thresholds[i] if i < len(thresholds) else 0
        direction = directions[i] if i < len(directions) else 1
        
        if direction > 0:  # Looking for values above threshold
            if value > threshold:
                buy_agreements += 1
            elif value < -threshold:
                sell_agreements += 1
        else:  # Looking for values below threshold
            if value < -threshold:
                buy_agreements += 1
            elif value > threshold:
                sell_agreements += 1
    
    # Generate signal based on agreements
    if buy_agreements >= min_agreements:
        return 1
    elif sell_agreements >= min_agreements:
        return -1
    else:
        return 0


def cross_feature_indicator(feature_values: List[Any], **params) -> Dict[str, Any]:
    """
    Create a crossover indicator from two features.
    
    Args:
        feature_values: List of exactly two feature values
        params: Dictionary containing:
            - 'direction': Expected crossover direction (1 for first > second, -1 for first < second)
        
    Returns:
        Dictionary with crossover status information
    """
    if len(feature_values) != 2:
        raise ValueError("Cross feature indicator requires exactly two input features")
    
    direction = params.get('direction', 1)
    smooth = params.get('smooth', False)
    
    # Extract numeric values from features
    values = []
    for value in feature_values:
        if isinstance(value, (int, float)):
            values.append(value)
        elif isinstance(value, dict) and 'value' in value:
            values.append(value['value'])
        elif isinstance(value, dict) and 'signal' in value:
            values.append(value['signal'])
        else:
            values.append(0)  # Default for non-extractable
    
    if len(values) < 2:
        return {
            'state': 0,
            'direction': 0,
            'signal': 0
        }
    
    # Determine the state based on the relationship between values
    if values[0] > values[1]:
        state = 1  # First value above second
    elif values[0] < values[1]:
        state = -1  # First value below second
    else:
        state = 0  # Equal
    
    # Calculate actual difference
    difference = values[0] - values[1]
    
    # Determine if this is the desired direction
    if (direction > 0 and state > 0) or (direction < 0 and state < 0):
        signal = 1 if smooth else state  # Buy signal
    elif (direction > 0 and state < 0) or (direction < 0 and state > 0):
        signal = -1 if smooth else state  # Sell signal
    else:
        signal = 0  # Neutral
    
    return {
        'state': state,
        'difference': difference,
        'signal': signal
    }


def z_score_normalize(feature_values: List[Any], **params) -> List[float]:
    """
    Normalize feature values using Z-score normalization.
    
    Args:
        feature_values: List of feature values to normalize
        params: Dictionary containing additional parameters (not used)
        
    Returns:
        List of normalized feature values
    """
    # Extract numeric values
    numeric_values = []
    for value in feature_values:
        if isinstance(value, (int, float)):
            numeric_values.append(value)
        elif isinstance(value, dict) and 'value' in value:
            numeric_values.append(value['value'])
        elif isinstance(value, dict) and 'signal' in value:
            numeric_values.append(value['signal'])
        else:
            numeric_values.append(0)  # Default for non-numeric
    
    if not numeric_values:
        return []
    
    # Calculate mean and standard deviation
    mean = sum(numeric_values) / len(numeric_values)
    variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
    std_dev = np.sqrt(variance) if variance > 0 else 1.0
    
    # Normalize values
    normalized = [(x - mean) / std_dev for x in numeric_values]
    
    return normalized


def create_feature_vector(features: List[Feature], data: Dict[str, Any]) -> np.ndarray:
    """
    Create a feature vector from multiple features.
    
    Args:
        features: List of feature objects
        data: Dictionary containing the data to calculate features from
        
    Returns:
        Numpy array containing feature values
    """
    feature_values = []
    
    for feature in features:
        value = feature.calculate(data)
        
        # Handle different feature value types
        if isinstance(value, (int, float)):
            feature_values.append(value)
        elif isinstance(value, dict):
            # Try to extract a numeric value from the dictionary
            if 'value' in value:
                feature_values.append(value['value'])
            elif 'signal' in value:
                feature_values.append(value['signal'])
            else:
                # Use the first numeric value found
                for v in value.values():
                    if isinstance(v, (int, float)):
                        feature_values.append(v)
                        break
                else:
                    feature_values.append(0)  # Default if no numeric value found
        elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
            # Use the last value from a sequence
            if isinstance(value[-1], (int, float)):
                feature_values.append(value[-1])
            else:
                feature_values.append(0)  # Default for non-numeric
        else:
            feature_values.append(0)  # Default for other types
    
    return np.array(feature_values)


def combine_time_series_features(features: List[Feature], 
                                 data: Dict[str, Any],
                                 lookback: int = 10) -> np.ndarray:
    """
    Create a time series of feature vectors from multiple features.
    
    Args:
        features: List of feature objects
        data: Dictionary containing historical data (with each value as a list/array)
        lookback: Number of historical time steps to include
        
    Returns:
        2D numpy array with shape (lookback, num_features)
    """
    num_features = len(features)
    result = np.zeros((lookback, num_features))
    
    # Ensure data dictionary has historical values
    has_history = False
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)) and len(value) >= lookback:
            has_history = True
            break
    
    if not has_history:
        return result
    
    # Create a time series of data dictionaries
    data_series = []
    for i in range(lookback):
        time_step_data = {}
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > i:
                time_step_data[key] = value[-(i+1)]
            else:
                time_step_data[key] = value
        data_series.append(time_step_data)
    
    # Calculate features for each time step
    for t, time_step_data in enumerate(data_series):
        for f, feature in enumerate(features):
            value = feature.calculate(time_step_data)
            
            # Handle different feature value types
            if isinstance(value, (int, float)):
                result[t, f] = value
            elif isinstance(value, dict):
                # Try to extract a numeric value from the dictionary
                if 'value' in value:
                    result[t, f] = value['value']
                elif 'signal' in value:
                    result[t, f] = value['signal']
                else:
                    # Use the first numeric value found
                    for v in value.values():
                        if isinstance(v, (int, float)):
                            result[t, f] = v
                            break
    
    return result
