"""
Logging system for the trading application.

This module provides a structured logging framework for different
components of the trading system with configurable log levels,
formatting, and output destinations.
"""

import logging
import logging.handlers
import os
import sys
import json
import yaml
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import threading
from functools import wraps
import time


class TradeLogger:
    """
    Main logger interface for the trading system.
    
    This class provides a unified interface for logging across
    different components with support for multiple output formats
    and destinations.
    """
    
    # Class-level dictionary to store logger instances
    _loggers = {}
    
    # Default configuration
    _default_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                'class': 'logging_system.log_formatters.JsonFormatter'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/trading.log',
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 10
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
            'trading.strategy': {
                'level': 'DEBUG',
                'propagate': True
            },
            'trading.execution': {
                'level': 'DEBUG',
                'propagate': True
            }
        }
    }
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name: Logger name, typically using dot notation (e.g., 'trading.strategy')
            
        Returns:
            Logger instance
        """
        # Check if logger already exists
        if name in cls._loggers:
            return cls._loggers[name]
            
        # Create a new logger
        logger = logging.getLogger(name)
        cls._loggers[name] = logger
        
        return logger
        
    @classmethod
    def configure_from_file(cls, config_file: str) -> None:
        """
        Configure logging from a configuration file.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
        """
        # Determine file type from extension
        _, ext = os.path.splitext(config_file)
        
        # Load configuration
        if ext.lower() == '.json':
            with open(config_file, 'r') as f:
                config = json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
            
        # Apply configuration
        cls.configure(config)
        
    @classmethod
    def configure(cls, config: Dict[str, Any] = None) -> None:
        """
        Configure logging with the provided configuration.
        
        Args:
            config: Logging configuration dictionary
        """
        if config is None:
            config = cls._default_config
            
        # Create logs directory if it doesn't exist
        for handler_config in config.get('handlers', {}).values():
            if 'filename' in handler_config:
                log_dir = os.path.dirname(handler_config['filename'])
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            
        # Apply configuration using dictConfig
        logging.config.dictConfig(config)
        
    @classmethod
    def set_level(cls, name: str, level: Union[str, int]) -> None:
        """
        Set the log level for a specific logger.
        
        Args:
            name: Logger name
            level: Log level (e.g., 'DEBUG', 'INFO', logging.DEBUG)
        """
        logger = cls.get_logger(name)
        logger.setLevel(level)
        
    @classmethod
    def add_file_handler(cls, name: str, filename: str, level: Union[str, int] = 'DEBUG',
                        formatter: str = 'detailed', max_bytes: int = 10485760, 
                        backup_count: int = 5) -> None:
        """
        Add a file handler to a logger.
        
        Args:
            name: Logger name
            filename: Log file path
            level: Log level for the handler
            formatter: Formatter name
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        logger = cls.get_logger(name)
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create handler
        handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        # Set level
        handler.setLevel(level)
        
        # Set formatter
        if formatter == 'standard':
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
        elif formatter == 'detailed':
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
        elif formatter == 'json':
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(formatter))
            
        # Add handler to logger
        logger.addHandler(handler)
        
    @classmethod
    def add_console_handler(cls, name: str, level: Union[str, int] = 'INFO',
                           formatter: str = 'standard') -> None:
        """
        Add a console handler to a logger.
        
        Args:
            name: Logger name
            level: Log level for the handler
            formatter: Formatter name
        """
        logger = cls.get_logger(name)
        
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Set level
        handler.setLevel(level)
        
        # Set formatter
        if formatter == 'standard':
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
        elif formatter == 'detailed':
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(threadName)s] %(message)s',
                '%Y-%m-%d %H:%M:%S'
            ))
        elif formatter == 'json':
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(formatter))
            
        # Add handler to logger
        logger.addHandler(handler)


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.
    
    This enables structured logging which is useful for log parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log message
        """
        # Create base log object
        log_object = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno,
            'thread': record.threadName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_object['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
            
        # Add extra attributes if present
        if hasattr(record, 'extra') and record.extra:
            log_object.update(record.extra)
            
        return json.dumps(log_object)


class LogContext:
    """
    Context manager for adding contextual information to logs.
    
    This allows for adding extra information to log records within a specific context.
    """
    
    _context_data = threading.local()
    
    def __init__(self, **kwargs):
        """
        Initialize the log context.
        
        Args:
            **kwargs: Contextual information to add to logs
        """
        self.data = kwargs
        self.previous_data = None
        
    def __enter__(self):
        """Enter the context and set contextual data."""
        # Save previous context data
        self.previous_data = getattr(self._context_data, 'data', {})
        
        # Set new context data
        if not hasattr(self._context_data, 'data'):
            self._context_data.data = {}
            
        # Update context data with new values
        new_data = self.previous_data.copy()
        new_data.update(self.data)
        self._context_data.data = new_data
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore previous data."""
        self._context_data.data = self.previous_data
        
    @classmethod
    def get_context_data(cls) -> Dict[str, Any]:
        """
        Get the current context data.
        
        Returns:
            Dictionary containing context data
        """
        return getattr(cls._context_data, 'data', {})
        
    @classmethod
    def add_context_data(cls, **kwargs) -> None:
        """
        Add data to the current context.
        
        Args:
            **kwargs: Data to add to context
        """
        if not hasattr(cls._context_data, 'data'):
            cls._context_data.data = {}
            
        cls._context_data.data.update(kwargs)
        
    @classmethod
    def clear_context_data(cls) -> None:
        """Clear all context data."""
        if hasattr(cls._context_data, 'data'):
            cls._context_data.data = {}


class ContextFilter(logging.Filter):
    """
    Filter that adds context data to log records.
    
    This filter adds any context data from LogContext to log records
    as they are emitted.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context data to the log record.
        
        Args:
            record: Log record to add context data to
            
        Returns:
            True (always passes the filter)
        """
        # Get context data
        context_data = LogContext.get_context_data()
        
        # Add context data to record
        if context_data:
            record.extra = getattr(record, 'extra', {})
            record.extra.update(context_data)
            
        return True


def log_execution_time(logger=None, level=logging.DEBUG):
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Logger to use (or logger name)
        level: Log level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            if logger is None:
                log = logging.getLogger(func.__module__)
            elif isinstance(logger, str):
                log = logging.getLogger(logger)
            else:
                log = logger
                
            # Log start
            log.log(level, f"Starting {func.__name__}")
            
            # Execute function and time it
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log.log(level, f"Completed {func.__name__} in {elapsed:.3f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                log.log(logging.ERROR, f"Exception in {func.__name__} after {elapsed:.3f} seconds: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_method_calls(logger=None, entry_level=logging.DEBUG, exit_level=logging.DEBUG,
                    arg_level=logging.DEBUG):
    """
    Decorator to log method entry and exit with arguments.
    
    Args:
        logger: Logger to use (or logger name)
        entry_level: Log level for method entry
        exit_level: Log level for method exit
        arg_level: Log level for arguments
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            if logger is None:
                log = logging.getLogger(func.__module__)
            elif isinstance(logger, str):
                log = logging.getLogger(logger)
            else:
                log = logger
                
            # Format arguments (skip self/cls for methods)
            arg_str = ""
            if args:
                if func.__name__ in ['__init__', '__call__'] or not args:
                    arg_list = [repr(a) for a in args]
                else:
                    arg_list = [repr(a) for a in args[1:]]  # Skip self/cls
                arg_str += ", ".join(arg_list)
                
            if kwargs:
                if arg_str:
                    arg_str += ", "
                arg_str += ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
                
            # Log method entry with arguments
            log.log(entry_level, f"Entering {func.__name__}()")
            if arg_str and log.isEnabledFor(arg_level):
                log.log(arg_level, f"Arguments: {arg_str}")
                
            # Execute method
            try:
                result = func(*args, **kwargs)
                
                # Log method exit
                log.log(exit_level, f"Exiting {func.__name__}()")
                
                return result
            except Exception as e:
                log.log(logging.ERROR, f"Exception in {func.__name__}(): {str(e)}")
                raise
                
        return wrapper
    return decorator


class LogHandler:
    """
    Base class for custom log handlers with specialized processing.
    
    This class serves as a base for creating custom handlers that
    can perform additional processing on log records.
    """
    
    def __init__(self, logger_name: str = None, level: Union[str, int] = logging.INFO):
        """
        Initialize the log handler.
        
        Args:
            logger_name: Name of logger to attach to
            level: Minimum log level to process
        """
        self.logger_name = logger_name
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.enabled = True
        
        # Attach to logger if specified
        if logger_name:
            self.attach()
            
    def handle(self, record: logging.LogRecord) -> None:
        """
        Handle a log record.
        
        Args:
            record: Log record to handle
        """
        if not self.enabled or record.levelno < self.level:
            return
            
        self.process(record)
        
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record.
        
        Args:
            record: Log record to process
        """
        raise NotImplementedError("Subclasses must implement process()")
        
    def attach(self) -> None:
        """Attach this handler to the logger."""
        if not self.logger_name:
            return
            
        # Get logger
        logger = logging.getLogger(self.logger_name)
        
        # Create and add handler
        class CustomHandler(logging.Handler):
            def __init__(self, callback, level):
                super().__init__(level)
                self.callback = callback
                
            def emit(self, record):
                self.callback(record)
                
        handler = CustomHandler(self.handle, self.level)
        logger.addHandler(handler)
        
    def enable(self) -> None:
        """Enable the handler."""
        self.enabled = True
        
    def disable(self) -> None:
        """Disable the handler."""
        self.enabled = False


class AlertHandler(LogHandler):
    """
    Handler for generating alerts based on log messages.
    
    This handler monitors log records and triggers alerts for
    records that meet specific criteria.
    """
    
    def __init__(self, logger_name: str = None, level: Union[str, int] = logging.ERROR,
                alert_callback=None, alert_keywords: List[str] = None):
        """
        Initialize the alert handler.
        
        Args:
            logger_name: Name of logger to attach to
            level: Minimum log level for alerts
            alert_callback: Function to call for alerts
            alert_keywords: Keywords to trigger alerts on
        """
        super().__init__(logger_name, level)
        self.alert_callback = alert_callback
        self.alert_keywords = alert_keywords or []
        
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record and generate alerts if needed.
        
        Args:
            record: Log record to process
        """
        # Check if record matches alert criteria
        message = record.getMessage().lower()
        
        # Check for keywords
        keyword_match = any(keyword.lower() in message for keyword in self.alert_keywords)
        
        # Generate alert if needed
        if keyword_match or record.levelno >= logging.ERROR:
            self._generate_alert(record)
            
    def _generate_alert(self, record: logging.LogRecord) -> None:
        """
        Generate an alert for a log record.
        
        Args:
            record: Log record to generate alert for
        """
        # Create alert object
        alert = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'source': f"{record.module}:{record.lineno}"
        }
        
        # Add exception info if present
        if record.exc_info:
            alert['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
            
        # Call alert callback if provided
        if self.alert_callback:
            self.alert_callback(alert)
            
        # Always log the alert
        alert_logger = logging.getLogger('trading.alerts')
        alert_logger.warning(f"ALERT: {record.getMessage()}")


class DatabaseHandler(LogHandler):
    """
    Handler for storing log records in a database.
    
    This handler saves log records to a database for persistent storage
    and later analysis.
    """
    
    def __init__(self, logger_name: str = None, level: Union[str, int] = logging.INFO,
                db_connection=None, table_name: str = 'logs'):
        """
        Initialize the database handler.
        
        Args:
            logger_name: Name of logger to attach to
            level: Minimum log level to store
            db_connection: Database connection
            table_name: Table name for logs
        """
        super().__init__(logger_name, level)
        self.db_connection = db_connection
        self.table_name = table_name
        
        # Create table if it doesn't exist
        self._create_table()
        
    def _create_table(self) -> None:
        """Create the log table if it doesn't exist."""
        if not self.db_connection:
            return
            
        # This is a simplistic example - adapt for your specific database
        cursor = self.db_connection.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                level TEXT,
                logger TEXT,
                message TEXT,
                module TEXT,
                line INTEGER,
                thread TEXT,
                exception TEXT
            )
        """)
        self.db_connection.commit()
        
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record and store it in the database.
        
        Args:
            record: Log record to process
        """
        if not self.db_connection:
            return
            
        # Extract exception info if present
        exception_info = None
        if record.exc_info:
            exception_info = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': ''.join(traceback.format_exception(*record.exc_info))
            }
            exception_info = json.dumps(exception_info)
            
        # Insert record into database
        cursor = self.db_connection.cursor()
        cursor.execute(f"""
            INSERT INTO {self.table_name}
            (timestamp, level, logger, message, module, line, thread, exception)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            record.levelname,
            record.name,
            record.getMessage(),
            record.module,
            record.lineno,
            record.threadName,
            exception_info
        ))
        self.db_connection.commit()
