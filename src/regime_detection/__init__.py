"""
Market regime detection module for algorithmic trading.

This module provides tools for identifying market regimes and adapting 
trading strategies accordingly.
"""

from src.regime_detection.regime_type import RegimeType
from src.regime_detection.detector_base import DetectorBase
from src.regime_detection.detector_registry import DetectorRegistry
from src.regime_detection.detector_factory import DetectorFactory
from src.regime_detection.regime_manager import RegimeManager

# Import and register all detector implementations
from src.regime_detection.detectors import *

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
