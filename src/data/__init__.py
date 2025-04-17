"""
Data Module Initialization

This module provides components for market data acquisition, preprocessing, and management.
"""

# Import core components
from .data_handler import DataHandler
from .data_sources import CSVDataSource

# Try to import connectors if available
try:
    # If DatabaseConnector exists in data_connectors.py
    from .data_connectors import DatabaseConnector, SQLiteConnector, APIConnector
except ImportError:
    # Provide stub/base class if import fails
    from abc import ABC, abstractmethod
    
    class DatabaseConnector(ABC):
        """Base stub class for database connectors."""
        @abstractmethod
        def connect(self):
            pass
            
        @abstractmethod
        def close(self):
            pass
            
        @abstractmethod
        def save_data(self, data, symbol, timeframe):
            pass
            
        @abstractmethod
        def load_data(self, symbol, timeframe, start_date, end_date):
            pass
            
# Make these classes available at the module level
__all__ = [
    'DataHandler',
    'CSVDataSource',
    'DatabaseConnector',
]
