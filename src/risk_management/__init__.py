"""
Risk management package for algorithmic trading systems.

This package provides a comprehensive framework for implementing advanced risk 
management using MAE, MFE, and ETD analysis to derive data-driven risk parameters.

Main components:
- RiskManager: Applies risk rules to trades
- RiskMetricsCollector: Collects risk metrics from trades
- RiskAnalysisEngine: Analyzes collected metrics
- RiskParameterOptimizer: Derives optimal risk parameters
"""

# Export main classes
from src.risk_management.risk_manager import RiskManager
from src.risk_management.collector import RiskMetricsCollector
from src.risk_management.analyzer import RiskAnalysisEngine
from src.risk_management.parameter_optimizer import RiskParameterOptimizer

# Export type definitions
from src.risk_management.types import (
    RiskParameters, 
    RiskToleranceLevel, 
    ExitReason, 
    TradeMetrics,
    RiskAnalysisResults
)

__all__ = [
    # Main classes
    'RiskManager',
    'RiskMetricsCollector',
    'RiskAnalysisEngine',
    'RiskParameterOptimizer',
    
    # Types
    'RiskParameters',
    'RiskToleranceLevel',
    'ExitReason',
    'TradeMetrics',
    'RiskAnalysisResults',
]
