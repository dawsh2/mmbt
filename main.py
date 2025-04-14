"""
Rules Module

This module provides rule classes that generate trading signals based on technical indicators,
price patterns, and other market conditions. Rules are the building blocks of trading strategies
and encapsulate specific trading logic.
"""

# Import base classes and registry
from .rule_base import Rule
from .rule_registry import RuleRegistry
from .rule_factory import RuleFactory

# Create a global registry instance for convenience
registry = RuleRegistry()

# Import rule implementations
from .ma_rules import (
    SMARule,
    EMARule,
    MACrossoverRule,
    PriceToMARule
)

from .oscillator_rules import (
    RSIRule,
    StochasticRule,
    MACDRule,
    CCIRule
)

from .volatility_rules import (
    BollingerBandRule,
    ATRRule,
    DonchianChannelRule,
    KeltnerChannelRule
)

from .trend_rules import (
    ADXRule,
    AroonRule,
    ParabolicSARRule,
    IchimokuRule
)

from .volume_rules import (
    VolumeBreakoutRule,
    OBVRule,
    VolumeProfileRule,
    ChaikinOscillatorRule
)

from .pattern_rules import (
    DoubleTopsBottomsRule,
    HeadAndShouldersRule,
    SupportResistanceRule,
    TrendlineRule
)

from .multi_timeframe_rules import (
    MultiTimeframeRule,
    TimeFrameAlignmentRule
)

# Helper function to create a default set of rules
def create_default_rule_set():
    """
    Create a default set of common rules.
    
    Returns:
        list: List of rule instances with default parameters
    """
    factory = RuleFactory()
    
    return [
        factory.create_rule('SMARule', {'periods': [10, 30], 'price_key': 'Close'}),
        factory.create_rule('MACrossoverRule', {'fast_period': 10, 'slow_period': 30, 'price_key': 'Close'}),
        factory.create_rule('RSIRule', {'period': 14, 'overbought': 70, 'oversold': 30}),
        factory.create_rule('BollingerBandRule', {'period': 20, 'std_dev': 2}),
        factory.create_rule('ADXRule', {'period': 14, 'threshold': 25}),
        factory.create_rule('VolumeBreakoutRule', {'volume_factor': 2.0, 'price_change': 0.01})
    ]

# Export key components
__all__ = [
    'Rule',
    'RuleRegistry',
    'RuleFactory',
    'registry',
    'create_default_rule_set',
    
    # MA Rules
    'SMARule',
    'EMARule',
    'MACrossoverRule',
    'PriceToMARule',
    
    # Oscillator Rules
    'RSIRule',
    'StochasticRule',
    'MACDRule',
    'CCIRule',
    
    # Volatility Rules
    'BollingerBandRule',
    'ATRRule',
    'DonchianChannelRule',
    'KeltnerChannelRule',
    
    # Trend Rules
    'ADXRule',
    'AroonRule',
    'ParabolicSARRule',
    'IchimokuRule',
    
    # Volume Rules
    'VolumeBreakoutRule',
    'OBVRule',
    'VolumeProfileRule',
    'ChaikinOscillatorRule',
    
    # Pattern Rules
    'DoubleTopsBottomsRule',
    'HeadAndShouldersRule',
    'SupportResistanceRule',
    'TrendlineRule',
    
    # Multi-Timeframe Rules
    'MultiTimeframeRule',
    'TimeFrameAlignmentRule'
]
