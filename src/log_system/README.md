# Log_system Module

Logging Module for Trading System.

This module provides a unified logging interface for all components 
of the trading system with configurable formatters and handlers.

## Contents

- [log_config](#log_config)
- [log_formatters](#log_formatters)
- [log_handlers](#log_handlers)
- [logging](#logging)

## log_config

Configuration handling for the logging system.

This module provides functions for loading, validating, and applying
logging configurations.

### Functions

#### `load_config_from_file(config_file)`

*Returns:* `Dict[str, Any]`

Load logging configuration from a file.

Args:
    config_file (str): Path to configuration file (JSON or YAML)
    
Returns:
    dict: Loaded configuration
    
Raises:
    FileNotFoundError: If the file doesn't exist
    ValueError: If the file format is invalid

#### `validate_config(config)`

*Returns:* `None`

Validate logging configuration.

Args:
    config (dict): Logging configuration dictionary
    
Raises:
    ValueError: If the configuration is invalid

#### `apply_config(config)`

*Returns:* `None`

Apply logging configuration.

Args:
    config (dict): Logging configuration dictionary

#### `get_default_config()`

*Returns:* `Dict[str, Any]`

Get default logging configuration.

Returns:
    dict: Default configuration

#### `create_logger_config(name, level='INFO', handlers=None, propagate=False)`

*Returns:* `Dict[str, Any]`

Create a logger configuration.

Args:
    name (str): Logger name
    level (str, optional): Log level
    handlers (list, optional): List of handler names
    propagate (bool, optional): Whether to propagate to parent
    
Returns:
    dict: Logger configuration

#### `create_handler_config(handler_type, level='INFO', formatter='standard')`

*Returns:* `Dict[str, Any]`

Create a handler configuration.

Args:
    handler_type (str): Handler type ('console', 'file', etc.)
    level (str, optional): Log level
    formatter (str, optional): Formatter name
    **kwargs: Additional handler-specific parameters
    
Returns:
    dict: Handler configuration

## log_formatters

Custom log formatters for the logging system.

This module provides formatters for different logging output formats,
including JSON and structured formats.

### Classes

#### `JsonFormatter`

Format log records as JSON.

This formatter converts log records to JSON format, which is useful
for structured logging and log analysis.

##### Methods

###### `__init__(include_extra=True)`

Initialize formatter.

Args:
    include_extra (bool, optional): Whether to include extra fields
                                   from the log record

###### `format(record)`

*Returns:* `str`

Format a log record as JSON.

Args:
    record (LogRecord): Log record to format
    
Returns:
    str: JSON-formatted log message

###### `format_time(timestamp)`

*Returns:* `str`

Format a timestamp as ISO 8601.

Args:
    timestamp (float): UNIX timestamp
    
Returns:
    str: Formatted timestamp

###### `format_traceback(tb)`

*Returns:* `List[str]`

Format a traceback as a list of strings.

Args:
    tb: Traceback object
    
Returns:
    list: List of traceback lines

###### `_add_context_fields(record, log_data)`

*Returns:* `None`

Add context fields to log data.

Args:
    record (LogRecord): Log record
    log_data (dict): Log data dictionary to update

###### `_add_extra_fields(record, log_data)`

*Returns:* `None`

Add extra fields from record to log data.

Args:
    record (LogRecord): Log record
    log_data (dict): Log data dictionary to update

#### `StructuredFormatter`

Format log records with a structured format string.

This formatter extends the standard formatter with support for
context data and better exception formatting.

##### Methods

###### `__init__(fmt=None, datefmt=None)`

Initialize formatter.

Args:
    fmt (str, optional): Format string
    datefmt (str, optional): Date format string

###### `format(record)`

*Returns:* `str`

Format a log record with structured data.

Args:
    record (LogRecord): Log record to format
    
Returns:
    str: Formatted log message

###### `formatException(ei)`

*Returns:* `str`

Format an exception for structured output.

Args:
    ei: Exception info tuple
    
Returns:
    str: Formatted exception

###### `_format_context(record)`

*Returns:* `str`

Format context data as a string.

Args:
    record (LogRecord): Log record
    
Returns:
    str: Formatted context data

## log_handlers

Custom log handlers for the logging system.

This module provides specialized log handlers for different outputs
and processing of log records.

### Classes

#### `LogContext`

Context manager for adding contextual information to logs.

This class allows adding context data to log records, which can be
used by log formatters to include additional information.

##### Methods

###### `__init__()`

Initialize with contextual data.

Args:
    **kwargs: Contextual information to add to logs

###### `__enter__()`

Enter the context and set contextual data.

Returns:
    LogContext: Self for context manager

*Returns:* LogContext: Self for context manager

###### `__exit__(exc_type, exc_val, exc_tb)`

Exit the context and restore previous data.

Args:
    exc_type: Exception type
    exc_val: Exception value
    exc_tb: Exception traceback

###### `get_context_data(cls)`

*Returns:* `Dict[str, Any]`

Get the current context data.

Returns:
    dict: Dictionary containing context data

###### `add_context_data(cls)`

Add data to the current context.

Args:
    **kwargs: Data to add to context

###### `clear_context_data(cls)`

Clear all context data.

#### `ContextFilter`

Filter that adds context data to log records.

This filter adds context data from LogContext to each log record
it processes.

##### Methods

###### `filter(record)`

*Returns:* `bool`

Add context data to the log record.

Args:
    record (LogRecord): Log record to add context data to
    
Returns:
    bool: True (always passes the filter)

#### `LogHandler`

Base class for custom log handlers.

This class provides a foundation for creating custom log handlers
that process log records in specific ways.

##### Methods

###### `__init__(logger_name=None, level)`

Initialize handler.

Args:
    logger_name (str, optional): Name of logger to attach to
    level (str|int, optional): Minimum log level to process

###### `handle(record)`

*Returns:* `None`

Handle a log record.

Args:
    record (LogRecord): Log record to handle

###### `process(record)`

*Returns:* `None`

Process a log record (abstract method).

Args:
    record (LogRecord): Log record to process

###### `attach()`

*Returns:* `None`

Attach this handler to the logger.

###### `detach()`

*Returns:* `None`

Detach this handler from the logger.

###### `enable()`

*Returns:* `None`

Enable the handler.

###### `disable()`

*Returns:* `None`

Disable the handler.

#### `AlertHandler`

Handler for generating alerts based on log messages.

This handler can generate alerts when log messages match certain
criteria, and can call a callback function to handle the alerts.

##### Methods

###### `__init__(logger_name=None, level, alert_callback=None, alert_keywords=None)`

Initialize alert handler.

Args:
    logger_name (str, optional): Name of logger to attach to
    level (str|int, optional): Minimum log level for alerts
    alert_callback (callable, optional): Function to call for alerts
    alert_keywords (list, optional): Keywords to trigger alerts on

###### `process(record)`

*Returns:* `None`

Process a log record and generate alerts if needed.

Args:
    record (LogRecord): Log record to process

#### `DatabaseHandler`

Handler for storing log records in a database.

This handler stores log records in a database table, which can be
useful for long-term storage and analysis.

##### Methods

###### `__init__(logger_name=None, level, db_connection=None, table_name='logs')`

Initialize database handler.

Args:
    logger_name (str, optional): Name of logger to attach to
    level (str|int, optional): Minimum log level to store
    db_connection (object, optional): Database connection
    table_name (str, optional): Table name for logs

###### `process(record)`

*Returns:* `None`

Process a log record and store it in the database.

Args:
    record (LogRecord): Log record to process

###### `_initialize_table()`

*Returns:* `None`

Initialize database table if it doesn't exist.

###### `_get_connection()`

*Returns:* `Any`

Get a database connection for the current thread.

Returns:
    object: Database connection

###### `_format_record(record)`

*Returns:* `Dict[str, Any]`

Format a log record for database storage.

Args:
    record (LogRecord): Log record to format
    
Returns:
    dict: Formatted log data

#### `MemoryHandler`

Handler that stores log records in memory.

This handler is useful for testing and for capturing logs for
later analysis within the application.

##### Methods

###### `__init__(logger_name=None, level, max_records=1000)`

Initialize memory handler.

Args:
    logger_name (str, optional): Name of logger to attach to
    level (str|int, optional): Minimum log level to store
    max_records (int, optional): Maximum number of records to store

###### `process(record)`

*Returns:* `None`

Process a log record and store it in memory.

Args:
    record (LogRecord): Log record to process

###### `get_records(level=None, logger=None)`

*Returns:* `List[logging.LogRecord]`

Get stored records with optional filtering.

Args:
    level (str|int, optional): Filter by level
    logger (str, optional): Filter by logger name
    
Returns:
    list: Filtered log records

###### `clear()`

*Returns:* `None`

Clear all stored records.

## logging

Logging system for the trading application.

This module provides a structured logging framework for different
components of the trading system with configurable log levels,
formatting, and output destinations.

### Functions

#### `log_execution_time(logger=None, level)`

Decorator to log the execution time of a function.

Args:
    logger: Logger to use (or logger name)
    level: Log level to use
    
Returns:
    Decorator function

*Returns:* Decorator function

#### `log_method_calls(logger=None, entry_level, exit_level, arg_level)`

Decorator to log method entry and exit with arguments.

Args:
    logger: Logger to use (or logger name)
    entry_level: Log level for method entry
    exit_level: Log level for method exit
    arg_level: Log level for arguments
    
Returns:
    Decorator function

*Returns:* Decorator function

### Classes

#### `TradeLogger`

Main logger interface for the trading system.

This class provides a unified interface for logging across
different components with support for multiple output formats
and destinations.

##### Methods

###### `get_logger(cls, name)`

*Returns:* `logging.Logger`

Get a logger instance with the specified name.

Args:
    name: Logger name, typically using dot notation (e.g., 'trading.strategy')
    
Returns:
    Logger instance

###### `configure_from_file(cls, config_file)`

*Returns:* `None`

Configure logging from a configuration file.

Args:
    config_file: Path to configuration file (JSON or YAML)

###### `configure(cls, config=None)`

*Returns:* `None`

Configure logging with the provided configuration.

Args:
    config: Logging configuration dictionary

###### `set_level(cls, name, level)`

*Returns:* `None`

Set the log level for a specific logger.

Args:
    name: Logger name
    level: Log level (e.g., 'DEBUG', 'INFO', logging.DEBUG)

###### `add_file_handler(cls, name, filename, level='DEBUG', formatter='detailed', max_bytes=10485760, backup_count=5)`

*Returns:* `None`

Add a file handler to a logger.

Args:
    name: Logger name
    filename: Log file path
    level: Log level for the handler
    formatter: Formatter name
    max_bytes: Maximum file size before rotation
    backup_count: Number of backup files to keep

###### `add_console_handler(cls, name, level='INFO', formatter='standard')`

*Returns:* `None`

Add a console handler to a logger.

Args:
    name: Logger name
    level: Log level for the handler
    formatter: Formatter name

#### `JsonFormatter`

Custom formatter that outputs log records as JSON.

This enables structured logging which is useful for log parsing and analysis.

##### Methods

###### `format(record)`

*Returns:* `str`

Format a log record as JSON.

Args:
    record: Log record to format
    
Returns:
    JSON-formatted log message

#### `LogContext`

Context manager for adding contextual information to logs.

This allows for adding extra information to log records within a specific context.

##### Methods

###### `__init__()`

Initialize the log context.

Args:
    **kwargs: Contextual information to add to logs

###### `__enter__()`

Enter the context and set contextual data.

###### `__exit__(exc_type, exc_val, exc_tb)`

Exit the context and restore previous data.

###### `get_context_data(cls)`

*Returns:* `Dict[str, Any]`

Get the current context data.

Returns:
    Dictionary containing context data

###### `add_context_data(cls)`

*Returns:* `None`

Add data to the current context.

Args:
    **kwargs: Data to add to context

###### `clear_context_data(cls)`

*Returns:* `None`

Clear all context data.

#### `ContextFilter`

Filter that adds context data to log records.

This filter adds any context data from LogContext to log records
as they are emitted.

##### Methods

###### `filter(record)`

*Returns:* `bool`

Add context data to the log record.

Args:
    record: Log record to add context data to
    
Returns:
    True (always passes the filter)

#### `LogHandler`

Base class for custom log handlers with specialized processing.

This class serves as a base for creating custom handlers that
can perform additional processing on log records.

##### Methods

###### `__init__(logger_name=None, level)`

Initialize the log handler.

Args:
    logger_name: Name of logger to attach to
    level: Minimum log level to process

###### `handle(record)`

*Returns:* `None`

Handle a log record.

Args:
    record: Log record to handle

###### `process(record)`

*Returns:* `None`

Process a log record.

Args:
    record: Log record to process

###### `attach()`

*Returns:* `None`

Attach this handler to the logger.

###### `enable()`

*Returns:* `None`

Enable the handler.

###### `disable()`

*Returns:* `None`

Disable the handler.

#### `AlertHandler`

Handler for generating alerts based on log messages.

This handler monitors log records and triggers alerts for
records that meet specific criteria.

##### Methods

###### `__init__(logger_name=None, level, alert_callback=None, alert_keywords=None)`

Initialize the alert handler.

Args:
    logger_name: Name of logger to attach to
    level: Minimum log level for alerts
    alert_callback: Function to call for alerts
    alert_keywords: Keywords to trigger alerts on

###### `process(record)`

*Returns:* `None`

Process a log record and generate alerts if needed.

Args:
    record: Log record to process

###### `_generate_alert(record)`

*Returns:* `None`

Generate an alert for a log record.

Args:
    record: Log record to generate alert for

#### `DatabaseHandler`

Handler for storing log records in a database.

This handler saves log records to a database for persistent storage
and later analysis.

##### Methods

###### `__init__(logger_name=None, level, db_connection=None, table_name='logs')`

Initialize the database handler.

Args:
    logger_name: Name of logger to attach to
    level: Minimum log level to store
    db_connection: Database connection
    table_name: Table name for logs

###### `_create_table()`

*Returns:* `None`

Create the log table if it doesn't exist.

###### `process(record)`

*Returns:* `None`

Process a log record and store it in the database.

Args:
    record: Log record to process
