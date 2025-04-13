# src/__init__.py
"""
Trading system package.

This package contains components for building, optimizing, and validating
algorithmic trading strategies.
"""

__version__ = '0.1.0'


# src/indicators/__init__.py
"""
Indicators module.

This module provides pure functions for calculating technical indicators
from price data. These functions are stateless and don't retain any
historical data.
"""

# Import commonly used indicators for easier access
from .indicators.moving_averages import (
    simple_moving_average,
    exponential_moving_average,
    double_exponential_moving_average,
    triple_exponential_moving_average
)

from .indicators.oscillators import (
    relative_strength_index,
    stochastic_oscillator,
    commodity_channel_index
)

from .indicators.volatility import (
    bollinger_bands,
    average_true_range
)

from .indicators.trend import (
    average_directional_index,
    moving_average_convergence_divergence
)


# src/features/__init__.py
"""
Features module.

This module provides feature classes that transform raw price data and
indicators into normalized inputs for rules. Features are stateful objects
that can maintain history.
"""


from .features.feature_base import Feature
from .features.feature_registry import get_registry, register_feature


# src/rules/__init__.py
"""
Rules module.

This module provides rule classes that make trading decisions based on
feature values. Rules combine one or more features to generate buy/sell signals.
"""

from .rules.rule_base import Rule
from .rules.rule_registry import RuleRegistry
from .rules.rule_factory import RuleFactory


# src/strategies/__init__.py
"""
Strategies module.

This module provides strategy classes that combine rules to make trading
decisions. Strategies can use different methods to combine rule signals.
"""

from .strategies.strategy_base import Strategy
from .strategies.strategy_factory import StrategyFactory


# src/utils/__init__.py
"""
Utilities module.

This module provides utility functions and classes for data processing,
performance measurement, and other common tasks.
"""
