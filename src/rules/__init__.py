"""
Rules Module

This module provides a framework for creating trading rules that generate signals
based on technical indicators.
"""

# Import base classes
from .rule_base import Rule
from .rule_registry import RuleRegistry, register_rule
from .rule_factory import RuleFactory, create_rule, create_composite_rule

# Import Crossover Rules
from .crossover_rules import (
    SMAcrossoverRule,
    ExponentialMACrossoverRule,
    MACDCrossoverRule, 
    PriceMACrossoverRule,
    BollingerBandsCrossoverRule,
    StochasticCrossoverRule
)

# Import Oscillator Rules
from .oscillator_rules import (
    RSIRule,
    StochasticRule,
    CCIRule,
    MACDHistogramRule
)

# Import Trend Rules
from .trend_rules import (
    ADXRule,
    IchimokuRule,
    VortexRule
)

# Import Volatility Rules
from .volatility_rules import (
    BollingerBandRule,
    ATRTrailingStopRule,
    VolatilityBreakoutRule,
    KeltnerChannelRule
)

# Create aliases for common rules
SMARule = SMAcrossoverRule
RSIRule = RSIRule

# Register all rules in the registry
rule_registry = RuleRegistry()

__all__ = [
    # Base classes
    'Rule', 'RuleRegistry', 'register_rule', 'RuleFactory',
    'create_rule', 'create_composite_rule',
    
    # Crossover Rules
    'SMAcrossoverRule', 'SMARule',
    'ExponentialMACrossoverRule',
    'MACDCrossoverRule',
    'PriceMACrossoverRule',
    'BollingerBandsCrossoverRule',
    'StochasticCrossoverRule',
    
    # Oscillator Rules
    'RSIRule',
    'StochasticRule',
    'CCIRule',
    'MACDHistogramRule',
    
    # Trend Rules
    'ADXRule',
    'IchimokuRule',
    'VortexRule',
    
    # Volatility Rules
    'BollingerBandRule',
    'ATRTrailingStopRule',
    'VolatilityBreakoutRule',
    'KeltnerChannelRule',
    
    # Registry
    'rule_registry'
]
