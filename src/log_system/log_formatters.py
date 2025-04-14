"""
Custom log formatters for the logging system.

This module provides formatters for different logging output formats,
including JSON and structured formats.
"""
import json
import logging
import datetime
import traceback
from typing import Any, Dict, Optional, List

class JsonFormatter(logging.Formatter):
    """
    Format log records as JSON.
    
    This formatter converts log records to JSON format, which is useful
    for structured logging and log analysis.
    """
    
    def __init__(self, include_extra: bool = True):
        """
        Initialize formatter.
        
        Args:
            include_extra (bool, optional): Whether to include extra fields
                                           from the log record
        """
        super().__init__()
        self.include_extra = include_extra
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record (LogRecord): Log record to format
            
        Returns:
            str: JSON-formatted log message
        """
        # Create base log data
        log_data = {
            'timestamp': self.format_time(record.created),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add location info
        log_data.update({
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        })
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.format_traceback(record.exc_info[2])
            }
        
        # Add context fields
        self._add_context_fields(record, log_data)
        
        # Add extra fields if enabled
        if self.include_extra:
            self._add_extra_fields(record, log_data)
        
        # Convert to JSON
        return json.dumps(log_data)
        
    def format_time(self, timestamp: float) -> str:
        """
        Format a timestamp as ISO 8601.
        
        Args:
            timestamp (float): UNIX timestamp
            
        Returns:
            str: Formatted timestamp
        """
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    def format_traceback(self, tb) -> List[str]:
        """
        Format a traceback as a list of strings.
        
        Args:
            tb: Traceback object
            
        Returns:
            list: List of traceback lines
        """
        if tb:
            return traceback.format_tb(tb)
        return []
    
    def _add_context_fields(self, record: logging.LogRecord, log_data: Dict[str, Any]) -> None:
        """
        Add context fields to log data.
        
        Args:
            record (LogRecord): Log record
            log_data (dict): Log data dictionary to update
        """
        # Check for context data
        if hasattr(record, 'context_data') and record.context_data:
            # Add context data directly to log data (flat structure)
            log_data.update(record.context_data)
    
    def _add_extra_fields(self, record: logging.LogRecord, log_data: Dict[str, Any]) -> None:
        """
        Add extra fields from record to log data.
        
        Args:
            record (LogRecord): Log record
            log_data (dict): Log data dictionary to update
        """
        # Get all attributes of the record
        for key, value in record.__dict__.items():
            # Skip standard fields and context_data
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text',
                          'filename', 'funcName', 'id', 'levelname', 'levelno',
                          'lineno', 'module', 'msecs', 'message', 'msg',
                          'name', 'pathname', 'process', 'processName',
                          'relativeCreated', 'stack_info', 'thread', 'threadName',
                          'context_data'):
                try:
                    # Convert value to JSON-serializable form
                    json.dumps({key: value})
                    log_data[key] = value
                except (TypeError, OverflowError):
                    # Skip values that can't be JSON serialized
                    log_data[key] = str(value)

class StructuredFormatter(logging.Formatter):
    """
    Format log records with a structured format string.
    
    This formatter extends the standard formatter with support for
    context data and better exception formatting.
    """
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Initialize formatter.
        
        Args:
            fmt (str, optional): Format string
            datefmt (str, optional): Date format string
        """
        if fmt is None:
            fmt = ('%(asctime)s [%(levelname)s] [%(name)s] '
                  '%(message)s %(context)s')
        super().__init__(fmt, datefmt)
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with structured data.
        
        Args:
            record (LogRecord): Log record to format
            
        Returns:
            str: Formatted log message
        """
        # Format context data if present
        record.context = self._format_context(record)
        
        # Call parent format method
        return super().format(record)
    
    def formatException(self, ei) -> str:
        """
        Format an exception for structured output.
        
        Args:
            ei: Exception info tuple
            
        Returns:
            str: Formatted exception
        """
        # Call parent formatException method
        formatted = super().formatException(ei)
        
        # Add custom formatting if needed
        return f"\n{formatted}"
    
    def _format_context(self, record: logging.LogRecord) -> str:
        """
        Format context data as a string.
        
        Args:
            record (LogRecord): Log record
            
        Returns:
            str: Formatted context data
        """
        if hasattr(record, 'context_data') and record.context_data:
            # Format as key=value pairs
            pairs = []
            for key, value in record.context_data.items():
                if isinstance(value, str) and ' ' in value:
                    # Quote string values with spaces
                    pairs.append(f'{key}="{value}"')
                else:
                    pairs.append(f'{key}={value}')
            
            return '[' + ' '.join(pairs) + ']'
        
        return ''
