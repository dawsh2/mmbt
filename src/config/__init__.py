# Directory structure: config/__init__.py

"""
Configuration management for the trading system.

This module provides a centralized way to manage configuration for all components
of the trading system, with validation and defaults.
"""

from .config_manager import ConfigManager

__all__ = ['ConfigManager']
