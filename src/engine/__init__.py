# src/engine/__init__.py
from src.engine.execution_engine import ExecutionEngine
from src.engine.market_simulator import MarketSimulator
from src.engine.backtester import Backtester

# Import DefaultPositionManager from backtester
# from src.engine.backtester import DefaultPositionManager

__all__ = [
    'ExecutionEngine',
    'MarketSimulator',
    'Backtester',
#    'DefaultPositionManager'  # Export the simple position manager
]
