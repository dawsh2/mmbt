"""
Logging Module for Trading System.

This module provides a unified logging interface for all components 
of the trading system with configurable formatters and handlers.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from .log_config import load_config_from_file, apply_config
from .log_formatters import JsonFormatter
from .log_handlers import (
    LogHandler, 
    ContextFilter, 
    LogContext, 
    AlertHandler, 
    DatabaseHandler
)

class TradeLogger:
    """
    Main interface for logging in the trading system.
    
    This class provides a unified approach for getting loggers,
    configuring them, and setting up handlers for different components.
    """
    
    @staticmethod
    def get_logger(name):
        """
        Get a logger with the specified name.
        
        Args:
            name (str): Logger name, typically using dot notation
                        (e.g., 'trading.strategy')
                        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        if len(logger.handlers) == 0:
            # Add context filter by default
            logger.addFilter(ContextFilter())
        
        return logger
    
    @staticmethod
    def configure_from_file(config_file):
        """
        Configure logging from a configuration file.
        
        Args:
            config_file (str): Path to configuration file (JSON or YAML)
        """
        config = load_config_from_file(config_file)
        TradeLogger.configure(config)
    
    @staticmethod
    def configure(config=None):
        """
        Configure logging with the provided configuration.
        
        Args:
            config (dict, optional): Logging configuration dictionary
        """
        if config is None:
            # Default configuration with console output
            config = {
                'version': 1,
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                        'datefmt': '%Y-%m-%d %H:%M:%S'
                    }
                },
                'handlers': {
                    'console': {
                        'class': 'logging.StreamHandler',
                        'level': 'INFO',
                        'formatter': 'standard',
                        'stream': 'ext://sys.stdout'
                    }
                },
                'root': {
                    'level': 'INFO',
                    'handlers': ['console'],
                    'propagate': True
                }
            }
        
        apply_config(config)
    
    @staticmethod
    def set_level(name, level):
        """
        Set the log level for a specific logger.
        
        Args:
            name (str): Logger name
            level (str|int): Log level (e.g., 'DEBUG', 'INFO', logging.DEBUG)
        """
        logger = logging.getLogger(name)
        
        # Convert string levels to integers if necessary
        if isinstance(level, str):
            level = getattr(logging, level.upper())
            
        logger.setLevel(level)
    
    @staticmethod
    def add_file_handler(name, filename, level='DEBUG', formatter='detailed',
                         max_bytes=10485760, backup_count=5):
        """
        Add a file handler to a logger.
        
        Args:
            name (str): Logger name
            filename (str): Log file path
            level (str|int, optional): Log level for the handler
            formatter (str, optional): Formatter name
            max_bytes (int, optional): Maximum file size before rotation
            backup_count (int, optional): Number of backup files to keep
        """
        logger = logging.getLogger(name)
        
        # Convert string levels to integers if necessary
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Create handler
        handler = RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(level)
        
        # Set formatter if available in logging config
        formatter_obj = logging.getLogger().handlers[0].formatter
        for h in logging.getLogger().handlers:
            if getattr(h, 'name', '') == formatter:
                formatter_obj = h.formatter
                break
        
        handler.setFormatter(formatter_obj)
        logger.addHandler(handler)
    
    @staticmethod
    def add_console_handler(name, level='INFO', formatter='standard'):
        """
        Add a console handler to a logger.
        
        Args:
            name (str): Logger name
            level (str|int, optional): Log level for the handler
            formatter (str, optional): Formatter name
        """
        logger = logging.getLogger(name)
        
        # Convert string levels to integers if necessary
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Create handler
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        # Set formatter if available in logging config
        formatter_obj = logging.getLogger().handlers[0].formatter
        for h in logging.getLogger().handlers:
            if getattr(h, 'name', '') == formatter:
                formatter_obj = h.formatter
                break
        
        handler.setFormatter(formatter_obj)
        logger.addHandler(handler)


# Utility decorators
def log_execution_time(logger=None, level=logging.DEBUG):
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger (Logger|str, optional): Logger to use (or logger name)
        level (int, optional): Log level to use
    
    Returns:
        callable: Decorated function
    """
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            log = logger
            if isinstance(logger, str):
                log = TradeLogger.get_logger(logger)
            elif logger is None:
                log = TradeLogger.get_logger(func.__module__)
            
            # Log execution time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.log(level, f"{func.__name__} executed in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                log.log(level, f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def log_method_calls(logger=None, entry_level=logging.DEBUG, 
                    exit_level=logging.DEBUG, arg_level=logging.DEBUG):
    """
    Decorator to log method entry and exit with arguments.
    
    Args:
        logger (Logger|str, optional): Logger to use (or logger name)
        entry_level (int, optional): Log level for method entry
        exit_level (int, optional): Log level for method exit
        arg_level (int, optional): Log level for arguments
    
    Returns:
        callable: Decorated function
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            log = logger
            if isinstance(logger, str):
                log = TradeLogger.get_logger(logger)
            elif logger is None:
                log = TradeLogger.get_logger(func.__module__)
            
            # Log method entry
            log.log(entry_level, f"Entering {func.__name__}")
            
            # Log arguments if needed
            if log.isEnabledFor(arg_level):
                # Skip self or cls for methods
                arg_str = str(args[1:]) if args and hasattr(args[0], func.__name__) else str(args)
                kwarg_str = str(kwargs)
                log.log(arg_level, f"Args: {arg_str}, Kwargs: {kwarg_str}")
            
            # Execute method
            try:
                result = func(*args, **kwargs)
                log.log(exit_level, f"Exiting {func.__name__}")
                return result
            except Exception as e:
                log.log(exit_level, f"Exiting {func.__name__} with exception: {str(e)}")
                raise
        
        return wrapper
    
    return decorator

# Initialize logging with default configuration
TradeLogger.configure()
