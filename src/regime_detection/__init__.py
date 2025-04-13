"""
Market regime detection module for algorithmic trading.

This module provides tools for identifying market regimes and adapting 
trading strategies accordingly.
"""

from .regime_type import RegimeType
from .detector_base import DetectorBase
from .detector_registry import DetectorRegistry
from .detector_factory import DetectorFactory
from .regime_manager import RegimeManager

# Import and register all detector implementations
from .detectors import *

# Create a global registry instance for convenience
registry = DetectorRegistry()

__all__ = [
    'RegimeType',
    'DetectorBase',
    'DetectorRegistry',
    'DetectorFactory',
    'RegimeManager',
    'registry'
]
