"""
Custom log handlers for the logging system.

This module provides specialized log handlers for different outputs
and processing of log records.
"""
import logging
import threading
import weakref
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union

class LogContext:
    """
    Context manager for adding contextual information to logs.
    
    This class allows adding context data to log records, which can be
    used by log formatters to include additional information.
    """
    
    # Thread-local storage for context data
    _context_data = threading.local()
    
    def __init__(self, **kwargs):
        """
        Initialize with contextual data.
        
        Args:
            **kwargs: Contextual information to add to logs
        """
        self.data = kwargs
        self._previous_data = None
        
    def __enter__(self):
        """
        Enter the context and set contextual data.
        
        Returns:
            LogContext: Self for context manager
        """
        self._previous_data = getattr(LogContext._context_data, 'data', {})
        
        # Merge with existing context data
        new_data = self._previous_data.copy()
        new_data.update(self.data)
        LogContext._context_data.data = new_data
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context and restore previous data.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        LogContext._context_data.data = self._previous_data
    
    @classmethod
    def get_context_data(cls) -> Dict[str, Any]:
        """
        Get the current context data.
        
        Returns:
            dict: Dictionary containing context data
        """
        return getattr(cls._context_data, 'data', {}).copy()
    
    @classmethod
    def add_context_data(cls, **kwargs):
        """
        Add data to the current context.
        
        Args:
            **kwargs: Data to add to context
        """
        data = getattr(cls._context_data, 'data', {}).copy()
        data.update(kwargs)
        cls._context_data.data = data
    
    @classmethod
    def clear_context_data(cls):
        """Clear all context data."""
        cls._context_data.data = {}


class ContextFilter(logging.Filter):
    """
    Filter that adds context data to log records.
    
    This filter adds context data from LogContext to each log record
    it processes.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context data to the log record.
        
        Args:
            record (LogRecord): Log record to add context data to
            
        Returns:
            bool: True (always passes the filter)
        """
        # Add context data to record
        record.context_data = LogContext.get_context_data()
        
        # Always return True to pass the record
        return True


class LogHandler:
    """
    Base class for custom log handlers.
    
    This class provides a foundation for creating custom log handlers
    that process log records in specific ways.
    """
    
    def __init__(self, logger_name: Optional[str] = None, 
                level: Union[str, int] = logging.INFO):
        """
        Initialize handler.
        
        Args:
            logger_name (str, optional): Name of logger to attach to
            level (str|int, optional): Minimum log level to process
        """
        self.logger_name = logger_name
        
        # Convert string level to int if needed
        if isinstance(level, str):
            self.level = getattr(logging, level.upper())
        else:
            self.level = level
            
        self.enabled = True
        self._handler = None
    
    def handle(self, record: logging.LogRecord) -> None:
        """
        Handle a log record.
        
        Args:
            record (LogRecord): Log record to handle
        """
        if not self.enabled:
            return
            
        if record.levelno >= self.level:
            self.process(record)
    
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record (abstract method).
        
        Args:
            record (LogRecord): Log record to process
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def attach(self) -> None:
        """Attach this handler to the logger."""
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
            
            # Create a custom handler that calls our handle method
            handler = logging.Handler()
            handler.emit = lambda record: self.handle(record)
            
            # Store handler reference
            self._handler = handler
            logger.addHandler(handler)
    
    def detach(self) -> None:
        """Detach this handler from the logger."""
        if self.logger_name and self._handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self._handler)
            self._handler = None
    
    def enable(self) -> None:
        """Enable the handler."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the handler."""
        self.enabled = False


class AlertHandler(LogHandler):
    """
    Handler for generating alerts based on log messages.
    
    This handler can generate alerts when log messages match certain
    criteria, and can call a callback function to handle the alerts.
    """
    
    def __init__(self, 
                logger_name: Optional[str] = None, 
                level: Union[str, int] = logging.ERROR,
                alert_callback: Optional[Callable] = None,
                alert_keywords: Optional[List[str]] = None):
        """
        Initialize alert handler.
        
        Args:
            logger_name (str, optional): Name of logger to attach to
            level (str|int, optional): Minimum log level for alerts
            alert_callback (callable, optional): Function to call for alerts
            alert_keywords (list, optional): Keywords to trigger alerts on
        """
        super().__init__(logger_name, level)
        self.alert_callback = alert_callback
        self.alert_keywords = alert_keywords or []
    
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record and generate alerts if needed.
        
        Args:
            record (LogRecord): Log record to process
        """
        # Check if message contains any keywords
        message = record.getMessage()
        if self.alert_keywords and not any(kw in message for kw in self.alert_keywords):
            return
        
        # Create alert info
        alert_info = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'source': f"{record.module}.{record.funcName}:{record.lineno}",
            'message': message
        }
        
        # Add exception info if present
        if record.exc_info:
            alert_info['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1])
            }
        
        # Add context data if present
        if hasattr(record, 'context_data') and record.context_data:
            alert_info.update(record.context_data)
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert_info)
            except Exception as e:
                # Log error but don't raise
                logging.getLogger(__name__).error(
                    f"Error in alert callback: {str(e)}", exc_info=True)
        else:
            # Default behavior: print alert
            print(f"ALERT: {alert_info['level']} - {alert_info['message']}")


class DatabaseHandler(LogHandler):
    """
    Handler for storing log records in a database.
    
    This handler stores log records in a database table, which can be
    useful for long-term storage and analysis.
    """
    
    def __init__(self, 
                logger_name: Optional[str] = None, 
                level: Union[str, int] = logging.INFO,
                db_connection: Optional[Any] = None,
                table_name: str = 'logs'):
        """
        Initialize database handler.
        
        Args:
            logger_name (str, optional): Name of logger to attach to
            level (str|int, optional): Minimum log level to store
            db_connection (object, optional): Database connection
            table_name (str, optional): Table name for logs
        """
        super().__init__(logger_name, level)
        self.db_connection = db_connection
        self.table_name = table_name
        
        # Connection cache to avoid creating many connections
        self._connections = weakref.WeakValueDictionary()
        
        # Initialize database table if connection provided
        if db_connection:
            self._initialize_table()
    
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record and store it in the database.
        
        Args:
            record (LogRecord): Log record to process
        """
        if not self.db_connection:
            return
            
        try:
            # Get connection for current thread
            conn = self._get_connection()
            
            # Create cursor
            cursor = conn.cursor()
            
            # Format record for database
            log_data = self._format_record(record)
            
            # Insert into database
            columns = ', '.join(log_data.keys())
            placeholders = ', '.join(['?'] * len(log_data))
            
            sql = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, list(log_data.values()))
            
            # Commit transaction
            conn.commit()
        except Exception as e:
            # Log error but don't raise
            logging.getLogger(__name__).error(
                f"Error storing log in database: {str(e)}", exc_info=True)
    
    def _initialize_table(self) -> None:
        """Initialize database table if it doesn't exist."""
        try:
            # Get connection
            conn = self._get_connection()
            
            # Create cursor
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                level TEXT,
                logger TEXT,
                message TEXT,
                module TEXT,
                function TEXT,
                line INTEGER,
                exception TEXT,
                context TEXT
            )
            """
            cursor.execute(sql)
            
            # Create index on timestamp
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp "
                          f"ON {self.table_name} (timestamp)")
            
            # Commit transaction
            conn.commit()
        except Exception as e:
            # Log error but don't raise
            logging.getLogger(__name__).error(
                f"Error initializing database table: {str(e)}", exc_info=True)
    
    def _get_connection(self) -> Any:
        """
        Get a database connection for the current thread.
        
        Returns:
            object: Database connection
        """
        # Get current thread ID
        thread_id = threading.get_ident()
        
        # Return cached connection if available
        if thread_id in self._connections:
            return self._connections[thread_id]
        
        # Create new connection if needed
        conn = self.db_connection
        self._connections[thread_id] = conn
        return conn
    
    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Format a log record for database storage.
        
        Args:
            record (LogRecord): Log record to format
            
        Returns:
            dict: Formatted log data
        """
        # Basic log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = f"{record.exc_info[0].__name__}: {str(record.exc_info[1])}"
        else:
            log_data['exception'] = None
        
        # Add context data if present
        if hasattr(record, 'context_data') and record.context_data:
            import json
            log_data['context'] = json.dumps(record.context_data)
        else:
            log_data['context'] = None
        
        return log_data


class MemoryHandler(LogHandler):
    """
    Handler that stores log records in memory.
    
    This handler is useful for testing and for capturing logs for
    later analysis within the application.
    """
    
    def __init__(self, 
                logger_name: Optional[str] = None, 
                level: Union[str, int] = logging.INFO,
                max_records: int = 1000):
        """
        Initialize memory handler.
        
        Args:
            logger_name (str, optional): Name of logger to attach to
            level (str|int, optional): Minimum log level to store
            max_records (int, optional): Maximum number of records to store
        """
        super().__init__(logger_name, level)
        self.max_records = max_records
        self.records = []
    
    def process(self, record: logging.LogRecord) -> None:
        """
        Process a log record and store it in memory.
        
        Args:
            record (LogRecord): Log record to process
        """
        # Make a copy of the record to avoid modification
        import copy
        record_copy = copy.copy(record)
        
        # Add to records
        self.records.append(record_copy)
        
        # Trim if needed
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
    
    def get_records(self, level: Optional[Union[str, int]] = None, 
                   logger: Optional[str] = None) -> List[logging.LogRecord]:
        """
        Get stored records with optional filtering.
        
        Args:
            level (str|int, optional): Filter by level
            logger (str, optional): Filter by logger name
            
        Returns:
            list: Filtered log records
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Filter records
        filtered = self.records
        
        if level is not None:
            filtered = [r for r in filtered if r.levelno >= level]
            
        if logger is not None:
            filtered = [r for r in filtered if r.name.startswith(logger)]
            
        return filtered
    
    def clear(self) -> None:
        """Clear all stored records."""
        self.records = []
