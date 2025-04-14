# Logging Module Documentation

The Logging module provides a structured logging framework for different components of the trading system with configurable log levels, formatting, and output destinations.

## Core Concepts

**TradeLogger**: Main logger interface that provides unified logging across system components.  
**JsonFormatter**: Custom formatter that outputs structured JSON log records.  
**LogContext**: Context manager for adding contextual information to logs.  
**LogHandler**: Base class for custom log handlers with specialized processing.

## Basic Usage

```python
from logging_system import TradeLogger

# Get a logger
logger = TradeLogger.get_logger('trading.strategy')

# Log at different levels
logger.info("Strategy initialized")
logger.debug("Processing bar data: %s", bar_data)
logger.warning("Potential signal conflict detected")
logger.error("Failed to execute trade: %s", error_message)

# Configure from file
TradeLogger.configure_from_file('logging_config.json')

# Add context to logs
from logging_system import LogContext

with LogContext(strategy_id='ma_crossover', symbol='AAPL'):
    logger.info("Signal generated")  # Will include strategy_id and symbol
```

## API Reference

### TradeLogger

Main logger interface for the trading system, providing a unified approach for logging across different components.

**Class Methods:**
- `get_logger(name)`: Get a logger instance with the specified name
  - `name` (str): Logger name, typically using dot notation (e.g., 'trading.strategy')
  - Returns: Logger instance

- `configure_from_file(config_file)`: Configure logging from a configuration file
  - `config_file` (str): Path to configuration file (JSON or YAML)

- `configure(config=None)`: Configure logging with the provided configuration
  - `config` (dict, optional): Logging configuration dictionary

- `set_level(name, level)`: Set the log level for a specific logger
  - `name` (str): Logger name
  - `level` (str|int): Log level (e.g., 'DEBUG', 'INFO', logging.DEBUG)

- `add_file_handler(name, filename, level='DEBUG', formatter='detailed', max_bytes=10485760, backup_count=5)`: Add a file handler to a logger
  - `name` (str): Logger name
  - `filename` (str): Log file path
  - `level` (str|int, optional): Log level for the handler
  - `formatter` (str, optional): Formatter name
  - `max_bytes` (int, optional): Maximum file size before rotation
  - `backup_count` (int, optional): Number of backup files to keep

- `add_console_handler(name, level='INFO', formatter='standard')`: Add a console handler to a logger
  - `name` (str): Logger name
  - `level` (str|int, optional): Log level for the handler
  - `formatter` (str, optional): Formatter name

**Example:**
```python
# Get and configure a logger
logger = TradeLogger.get_logger('trading.strategy')
TradeLogger.set_level('trading.strategy', 'DEBUG')
TradeLogger.add_file_handler('trading.strategy', 'logs/strategy.log')

# Log messages
logger.info("Starting strategy")
logger.debug("Processing data: %s", data)
```

### JsonFormatter

Custom formatter that outputs log records as JSON for structured logging.

**Constructor Parameters:**
- No parameters required

**Methods:**
- `format(record)`: Format a log record as JSON
  - `record` (LogRecord): Log record to format
  - Returns: JSON-formatted log message

**Example:**
```python
import logging
from logging_system import JsonFormatter

# Create a formatter
formatter = JsonFormatter()

# Create a handler with the formatter
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Add handler to logger
logger = logging.getLogger('trading.strategy')
logger.addHandler(handler)
```

### LogContext

Context manager for adding contextual information to logs.

**Constructor Parameters:**
- `**kwargs`: Contextual information to add to logs

**Methods:**
- `__enter__()`: Enter the context and set contextual data
- `__exit__(exc_type, exc_val, exc_tb)`: Exit the context and restore previous data

**Class Methods:**
- `get_context_data()`: Get the current context data
  - Returns: Dictionary containing context data

- `add_context_data(**kwargs)`: Add data to the current context
  - `**kwargs`: Data to add to context

- `clear_context_data()`: Clear all context data

**Example:**
```python
from logging_system import LogContext, TradeLogger

logger = TradeLogger.get_logger('trading.execution')

# Use as context manager
with LogContext(trade_id='123', symbol='AAPL'):
    logger.info("Executing order")  # Will include trade_id and symbol in log

# Add context directly
LogContext.add_context_data(portfolio_id='main')
logger.info("Updating positions")  # Will include portfolio_id in log

# Clear context
LogContext.clear_context_data()
```

### ContextFilter

Filter that adds context data to log records.

**Methods:**
- `filter(record)`: Add context data to the log record
  - `record` (LogRecord): Log record to add context data to
  - Returns: True (always passes the filter)

### LogHandler

Base class for custom log handlers with specialized processing.

**Constructor Parameters:**
- `logger_name` (str, optional): Name of logger to attach to
- `level` (str|int, optional): Minimum log level to process (default: logging.INFO)

**Methods:**
- `handle(record)`: Handle a log record
  - `record` (LogRecord): Log record to handle

- `process(record)`: Process a log record (abstract method)
  - `record` (LogRecord): Log record to process

- `attach()`: Attach this handler to the logger

- `enable()`: Enable the handler

- `disable()`: Disable the handler

**Example:**
```python
from logging_system import LogHandler

class CustomHandler(LogHandler):
    def __init__(self, logger_name="trading"):
        super().__init__(logger_name, "WARNING")
        
    def process(self, record):
        # Custom processing logic
        if record.levelno >= logging.ERROR:
            # Send notification for errors
            send_notification(record.getMessage())

# Create and attach handler
handler = CustomHandler()
handler.attach()
```

### AlertHandler

Handler for generating alerts based on log messages.

**Constructor Parameters:**
- `logger_name` (str, optional): Name of logger to attach to
- `level` (str|int, optional): Minimum log level for alerts (default: logging.ERROR)
- `alert_callback` (callable, optional): Function to call for alerts
- `alert_keywords` (list, optional): Keywords to trigger alerts on

**Methods:**
- `process(record)`: Process a log record and generate alerts if needed
  - `record` (LogRecord): Log record to process

**Example:**
```python
from logging_system import AlertHandler

def send_alert(alert_info):
    # Send alert via email, SMS, etc.
    print(f"ALERT: {alert_info['message']}")

# Create alert handler
alert_handler = AlertHandler(
    logger_name="trading.strategy",
    level="WARNING",
    alert_callback=send_alert,
    alert_keywords=["margin call", "connection lost", "timeout"]
)

# Attach it
alert_handler.attach()
```

### DatabaseHandler

Handler for storing log records in a database.

**Constructor Parameters:**
- `logger_name` (str, optional): Name of logger to attach to
- `level` (str|int, optional): Minimum log level to store (default: logging.INFO)
- `db_connection` (object, optional): Database connection
- `table_name` (str, optional): Table name for logs (default: 'logs')

**Methods:**
- `process(record)`: Process a log record and store it in the database
  - `record` (LogRecord): Log record to process

**Example:**
```python
import sqlite3
from logging_system import DatabaseHandler

# Create database connection
conn = sqlite3.connect('trading_logs.db')

# Create handler
db_handler = DatabaseHandler(
    logger_name="trading",
    level="INFO",
    db_connection=conn,
    table_name="system_logs"
)

# Attach it
db_handler.attach()
```

### Utility Decorators

#### log_execution_time

Decorator to log the execution time of a function.

**Parameters:**
- `logger` (Logger|str, optional): Logger to use (or logger name)
- `level` (int, optional): Log level to use (default: logging.DEBUG)

**Example:**
```python
from logging_system import log_execution_time, TradeLogger

logger = TradeLogger.get_logger('trading.performance')

@log_execution_time(logger=logger, level=logging.INFO)
def process_market_data(data):
    # Process data
    return processed_data
```

#### log_method_calls

Decorator to log method entry and exit with arguments.

**Parameters:**
- `logger` (Logger|str, optional): Logger to use (or logger name)
- `entry_level` (int, optional): Log level for method entry (default: logging.DEBUG)
- `exit_level` (int, optional): Log level for method exit (default: logging.DEBUG)
- `arg_level` (int, optional): Log level for arguments (default: logging.DEBUG)

**Example:**
```python
from logging_system import log_method_calls

@log_method_calls(logger="trading.strategy", entry_level=logging.INFO)
def execute_trade(symbol, price, quantity):
    # Execute trade
    return trade_id
```

## Advanced Usage

### Custom Configuration

```python
from logging_system import TradeLogger
import logging

# Custom configuration
config = {
    'version': 1,
    'formatters': {
        'brief': {
            'format': '%(asctime)s [%(levelname)s] %(message)s',
            'datefmt': '%H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'brief',
            'stream': 'ext://sys.stdout'
        },
        'strategy_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/strategy.log',
            'maxBytes': 10485760,
            'backupCount': 3
        },
        'trade_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'logs/trades.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'loggers': {
        'trading.strategy': {
            'handlers': ['console', 'strategy_file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'trading.execution': {
            'handlers': ['console', 'trade_file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Apply configuration
TradeLogger.configure(config)
```

### Creating Custom Log Handlers

```python
from logging_system import LogHandler
import logging
import json
import requests

class WebhookHandler(LogHandler):
    """Handler that sends log events to a webhook."""
    
    def __init__(self, webhook_url, logger_name=None, level=logging.ERROR):
        super().__init__(logger_name, level)
        self.webhook_url = webhook_url
        
    def process(self, record):
        # Format record as JSON
        log_data = {
            'timestamp': record.created,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1])
            }
            
        # Send to webhook
        try:
            requests.post(
                self.webhook_url,
                json=log_data,
                headers={'Content-Type': 'application/json'},
                timeout=1  # Don't block for too long
            )
        except Exception as e:
            # Log to standard error but don't raise
            import sys
            print(f"Failed to send log to webhook: {e}", file=sys.stderr)

# Usage
webhook_handler = WebhookHandler(
    webhook_url='https://example.com/log_webhook',
    logger_name='trading.strategy',
    level=logging.WARNING
)
webhook_handler.attach()
```

### Structured Logging with Context

```python
from logging_system import TradeLogger, LogContext, JsonFormatter
import logging

# Configure logging for JSON output
logger = TradeLogger.get_logger('trading.execution')

# Add JSON handler
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# Log structured events during trade execution
def execute_order(order):
    # Create a context with common attributes
    with LogContext(
        order_id=order.id,
        symbol=order.symbol,
        quantity=order.quantity,
        side=order.side
    ):
        logger.info("Processing order")
        
        try:
            # Execute order logic here
            price = get_market_price(order.symbol)
            
            # Add more context for fill
            LogContext.add_context_data(
                price=price,
                execution_time=get_current_time()
            )
            
            logger.info("Order filled")
            
            # All logs include the full context:
            # {
            #   "timestamp": "2023-06-15 10:30:45.123",
            #   "level": "INFO",
            #   "logger": "trading.execution",
            #   "message": "Order filled",
            #   "order_id": "12345",
            #   "symbol": "AAPL",
            #   "quantity": 100,
            #   "side": "BUY",
            #   "price": 152.50,
            #   "execution_time": "2023-06-15 10:30:45.000"
            # }
            
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}", exc_info=True)
```

### Real-time Monitoring with AlertHandler

```python
from logging_system import TradeLogger, AlertHandler
import logging
import smtplib
from email.message import EmailMessage

def email_alert(alert_info):
    """Send email alert for critical issues."""
    msg = EmailMessage()
    msg['Subject'] = f"TRADING ALERT: {alert_info['level']}"
    msg['From'] = "alerts@example.com"
    msg['To'] = "trader@example.com"
    
    # Format message body
    body = f"""
    Alert Time: {alert_info['timestamp']}
    Level: {alert_info['level']}
    Source: {alert_info['logger']} ({alert_info['source']})
    
    Message: {alert_info['message']}
    """
    
    if 'exception' in alert_info:
        body += f"""
        Exception: {alert_info['exception']['type']}
        Error: {alert_info['exception']['message']}
        """
    
    msg.set_content(body)
    
    # Send email
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('alerts@example.com', 'password')
        server.send_message(msg)

# Create alert handler
alert_handler = AlertHandler(
    logger_name="trading",
    level=logging.ERROR,
    alert_callback=email_alert,
    alert_keywords=["connection lost", "API error", "margin call"]
)

# Attach to logger
alert_handler.attach()
```

## Best Practices

1. **Use hierarchical logger names** to organize logs by component (e.g., 'trading.strategy.momentum', 'trading.execution.api')

2. **Set appropriate log levels** for development (DEBUG) vs. production (INFO/WARNING)

3. **Use LogContext for grouping related logs** rather than repeating the same information in each log message

4. **Include structured data** to make logs easier to search and analyze

5. **Configure different handlers for different purposes**:
   - Console for interactive monitoring
   - Files for historical records
   - Specialized handlers for alerts and metrics

6. **Use decorators** to consistently log method calls and execution times across the application

7. **Implement error handling** in custom handlers to prevent logging failures from affecting the application

8. **Periodically rotate log files** to manage disk space and make logs easier to archive

9. **Be mindful of performance** when logging in high-frequency components

10. **Create consistent log message formats** to make logs easier to read and parse