"""
Features Module

This module provides the feature layer of the trading system, which transforms
raw price data and technical indicators into meaningful trading signals.
"""

# Import base classes and registry
from src.features.feature_base import Feature, FeatureSet, CompositeFeature, StatefulFeature
from src.features.feature_registry import (
    FeatureRegistry, register_feature, register_features_in_module, get_registry
)

# Import feature modules so they register themselves
from src.features import price_features
from src.features import technical_features
from src.features import time_features

# Import utilities
from src.features.feature_utils import (
    combine_features,
    weighted_average_combiner,
    logical_combiner,
    threshold_combiner,
    cross_feature_indicator,
    create_feature_vector
)

# Import specific features for direct access
from src.features.price_features import (
    ReturnFeature,
    NormalizedPriceFeature,
    PricePatternFeature,
    VolumeProfileFeature,
    PriceDistanceFeature
)

from src.features.technical_features import (
    MACrossoverFeature,
    OscillatorStateFeature,
    TrendStrengthFeature,
    VolatilityFeature,
    SupportResistanceFeature,
    SignalAgreementFeature,
    DivergenceFeature
)

from src.features.time_features import (
    TimeOfDayFeature,
    DayOfWeekFeature,
    MonthFeature,
    SeasonalityFeature,
    EventFeature
)

# Export key functions and classes
__all__ = [
    # Base classes
    'Feature', 'FeatureSet', 'CompositeFeature', 'StatefulFeature',
    
    # Registry
    'FeatureRegistry', 'register_feature', 'register_features_in_module', 'get_registry',
    
    # Utilities
    'combine_features', 'weighted_average_combiner', 'logical_combiner',
    'threshold_combiner', 'cross_feature_indicator', 'create_feature_vector',
    
    # Price features
    'ReturnFeature', 'NormalizedPriceFeature', 'PricePatternFeature',
    'VolumeProfileFeature', 'PriceDistanceFeature',
    
    # Technical features
    'MACrossoverFeature', 'OscillatorStateFeature', 'TrendStrengthFeature',
    'VolatilityFeature', 'SupportResistanceFeature', 'SignalAgreementFeature',
    'DivergenceFeature',
    
    # Time features
    'TimeOfDayFeature', 'DayOfWeekFeature', 'MonthFeature',
    'SeasonalityFeature', 'EventFeature'
]

# Helper function to create a set of common features
def create_default_feature_set(name="default_features"):
    """
    Create a default set of commonly used features.
    
    Args:
        name: Name for the feature set
        
    Returns:
        FeatureSet: A set of common features
    """
    feature_set = FeatureSet(name=name)
    
    # Add default price features
    feature_set.add_feature(ReturnFeature(name="return_1d", params={'periods': [1]}))
    feature_set.add_feature(ReturnFeature(name="return_5d", params={'periods': [5]}))
    feature_set.add_feature(NormalizedPriceFeature(name="norm_price_zscore", params={'method': 'z-score', 'window': 20}))
    
    # Add default technical features
    feature_set.add_feature(MACrossoverFeature(name="sma_crossover", params={'fast_ma': 'SMA_10', 'slow_ma': 'SMA_30'}))
    feature_set.add_feature(OscillatorStateFeature(name="rsi_state", params={'oscillator': 'RSI'}))
    feature_set.add_feature(VolatilityFeature(name="volatility", params={'method': 'atr'}))
    
    # Add default time features
    feature_set.add_feature(DayOfWeekFeature(name="day_of_week"))
    feature_set.add_feature(SeasonalityFeature(name="monthly_pattern", params={'period': 'month'}))
    
    return feature_set
