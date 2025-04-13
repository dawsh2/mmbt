"""
Time Features Module

This module provides features derived from time and date information, such as
seasonality patterns, day of week effects, and other calendar-based features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import calendar

from .feature_base import Feature
from .feature_registry import register_feature


@register_feature(category="time")
class TimeOfDayFeature(Feature):
    """
    Time of Day feature.
    
    This feature extracts time of day information and identifies patterns
    based on the hour of the day.
    """
    
    def __init__(self, 
                 name: str = "time_of_day", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Time of day analysis"):
        """
        Initialize the time of day feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
                - trading_hours: List of trading hour ranges (default: [(9, 16)])
                - zones: Custom time zones to create (default: None)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for time of day."""
        return {
            'format': '%Y-%m-%d %H:%M:%S',
            'trading_hours': [(9, 16)],  # 9:00 AM to 4:00 PM by default
            'zones': None
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract time of day features.
        
        Args:
            data: Dictionary containing timestamp information
        
        Returns:
            Dictionary with time of day features
        """
        timestamp_format = self.params.get('format', '%Y-%m-%d %H:%M:%S')
        trading_hours = self.params.get('trading_hours', [(9, 16)])
        custom_zones = self.params.get('zones', None)
        
        # Check for required timestamp data
        if 'timestamp' not in data:
            return {
                'hour': None,
                'minute': None,
                'session': None,
                'is_trading_hours': False,
                'normalized_time': 0.0
            }
            
        # Parse timestamp
        timestamp = data['timestamp']
        
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, timestamp_format)
            except ValueError:
                # Try some common formats if the specified one fails
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    return {
                        'hour': None,
                        'minute': None,
                        'session': None,
                        'is_trading_hours': False,
                        'normalized_time': 0.0
                    }
        
        # Extract time components
        hour = timestamp.hour
        minute = timestamp.minute
        
        # Check if current time is within trading hours
        is_trading_hours = False
        for start_hour, end_hour in trading_hours:
            if start_hour <= hour < end_hour:
                is_trading_hours = True
                break
        
        # Determine trading session
        if custom_zones:
            session = 'unknown'
            for zone_name, (zone_start, zone_end) in custom_zones.items():
                if zone_start <= hour < zone_end:
                    session = zone_name
                    break
        else:
            # Default sessions
            if hour < 9:
                session = 'pre_market'
            elif 9 <= hour < 12:
                session = 'morning'
            elif 12 <= hour < 13:
                session = 'lunch'
            elif 13 <= hour < 16:
                session = 'afternoon'
            else:  # hour >= 16
                session = 'after_hours'
        
        # Calculate normalized time of day (0.0 to 1.0)
        minutes_since_midnight = hour * 60 + minute
        normalized_time = minutes_since_midnight / (24 * 60)
        
        return {
            'hour': hour,
            'minute': minute,
            'session': session,
            'is_trading_hours': is_trading_hours,
            'normalized_time': normalized_time
        }


@register_feature(category="time")
class DayOfWeekFeature(Feature):
    """
    Day of Week feature.
    
    This feature extracts day of week information and identifies patterns
    based on the day of the week.
    """
    
    def __init__(self, 
                 name: str = "day_of_week", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Day of week analysis"):
        """
        Initialize the day of week feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
                - trading_days: List of trading days (default: [0, 1, 2, 3, 4])
                - with_cyclical: Whether to include cyclical encoding (default: True)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for day of week."""
        return {
            'format': '%Y-%m-%d %H:%M:%S',
            'trading_days': [0, 1, 2, 3, 4],  # Monday to Friday (0-based)
            'with_cyclical': True
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract day of week features.
        
        Args:
            data: Dictionary containing timestamp information
        
        Returns:
            Dictionary with day of week features
        """
        timestamp_format = self.params.get('format', '%Y-%m-%d %H:%M:%S')
        trading_days = self.params.get('trading_days', [0, 1, 2, 3, 4])
        with_cyclical = self.params.get('with_cyclical', True)
        
        # Check for required timestamp data
        if 'timestamp' not in data:
            return {
                'day_of_week': None,
                'is_trading_day': False,
                'day_name': None
            }
            
        # Parse timestamp
        timestamp = data['timestamp']
        
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, timestamp_format)
            except ValueError:
                # Try some common formats if the specified one fails
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    return {
                        'day_of_week': None,
                        'is_trading_day': False,
                        'day_name': None
                    }
        
        # Extract day of week (0 = Monday, 6 = Sunday)
        day_of_week = timestamp.weekday()
        
        # Check if current day is a trading day
        is_trading_day = day_of_week in trading_days
        
        # Get day name
        day_name = calendar.day_name[day_of_week]
        
        # Create result dictionary
        result = {
            'day_of_week': day_of_week,
            'is_trading_day': is_trading_day,
            'day_name': day_name
        }
        
        # Add cyclical encoding if requested
        if with_cyclical:
            # Encode day of week as a point on a circle
            angle = 2 * np.pi * day_of_week / 7
            result['day_sin'] = np.sin(angle)
            result['day_cos'] = np.cos(angle)
        
        return result


@register_feature(category="time")
class MonthFeature(Feature):
    """
    Month feature.
    
    This feature extracts month information and identifies patterns
    based on the month of the year.
    """
    
    def __init__(self, 
                 name: str = "month", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Month analysis"):
        """
        Initialize the month feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
                - with_cyclical: Whether to include cyclical encoding (default: True)
                - with_quarters: Whether to include quarter information (default: True)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for month."""
        return {
            'format': '%Y-%m-%d %H:%M:%S',
            'with_cyclical': True,
            'with_quarters': True
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract month features.
        
        Args:
            data: Dictionary containing timestamp information
        
        Returns:
            Dictionary with month features
        """
        timestamp_format = self.params.get('format', '%Y-%m-%d %H:%M:%S')
        with_cyclical = self.params.get('with_cyclical', True)
        with_quarters = self.params.get('with_quarters', True)
        
        # Check for required timestamp data
        if 'timestamp' not in data:
            return {
                'month': None,
                'month_name': None
            }
            
        # Parse timestamp
        timestamp = data['timestamp']
        
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, timestamp_format)
            except ValueError:
                # Try some common formats if the specified one fails
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    return {
                        'month': None,
                        'month_name': None
                    }
        
        # Extract month information
        month = timestamp.month
        month_name = calendar.month_name[month]
        
        # Create result dictionary
        result = {
            'month': month,
            'month_name': month_name
        }
        
        # Add cyclical encoding if requested
        if with_cyclical:
            # Encode month as a point on a circle
            angle = 2 * np.pi * (month - 1) / 12
            result['month_sin'] = np.sin(angle)
            result['month_cos'] = np.cos(angle)
        
        # Add quarter information if requested
        if with_quarters:
            quarter = (month - 1) // 3 + 1
            result['quarter'] = quarter
            
            # Financial quarter (assuming fiscal year starts in January)
            fiscal_quarter = quarter
            result['fiscal_quarter'] = fiscal_quarter
        
        return result


@register_feature(category="time")
class SeasonalityFeature(Feature):
    """
    Seasonality feature.
    
    This feature detects seasonal patterns in price data based on time periods
    such as day of week, month of year, etc.
    """
    
    def __init__(self, 
                 name: str = "seasonality", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Seasonality pattern detection"):
        """
        Initialize the seasonality feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - period: Seasonality period to analyze ('day', 'week', 'month', 'quarter')
                - lookback: Lookback period for pattern analysis (default: 252)
                - min_pattern_strength: Minimum strength for pattern detection (default: 0.6)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for seasonality."""
        return {
            'period': 'month',
            'lookback': 252,
            'min_pattern_strength': 0.6
        }
    
    def _validate_params(self) -> None:
        """Validate the parameters for this feature."""
        valid_periods = ['day', 'week', 'month', 'quarter']
        if self.params.get('period') not in valid_periods:
            raise ValueError(f"Period must be one of {valid_periods}")
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect seasonality patterns in price data.
        
        Args:
            data: Dictionary containing price history and timestamp information
        
        Returns:
            Dictionary with seasonality information
        """
        period = self.params.get('period', 'month')
        lookback = self.params.get('lookback', 252)
        min_pattern_strength = self.params.get('min_pattern_strength', 0.6)
        
        # Check for required data
        if 'timestamp' not in data or 'Close' not in data:
            return {
                'pattern_detected': False,
                'pattern_strength': 0,
                'expected_direction': 0,
                'historical_performance': 0
            }
            
        # Extract timestamp and price data
        timestamps = data['timestamp']
        prices = data['Close']
        
        # Ensure we have enough data
        if not isinstance(timestamps, (list, np.ndarray, pd.Series)) or len(timestamps) < lookback:
            return {
                'pattern_detected': False,
                'pattern_strength': 0,
                'expected_direction': 0,
                'historical_performance': 0
            }
            
        if not isinstance(prices, (list, np.ndarray, pd.Series)) or len(prices) < lookback:
            return {
                'pattern_detected': False,
                'pattern_strength': 0,
                'expected_direction': 0,
                'historical_performance': 0
            }
        
        # Get the timestamp for the current bar
        current_timestamp = timestamps[-1]
        
        # Convert to datetime if needed
        if isinstance(current_timestamp, str):
            try:
                current_timestamp = pd.to_datetime(current_timestamp)
            except:
                return {
                    'pattern_detected': False,
                    'pattern_strength': 0,
                    'expected_direction': 0,
                    'historical_performance': 0
                }
        
        # Convert all timestamps to datetime if needed
        if isinstance(timestamps[0], str):
            try:
                timestamps = [pd.to_datetime(ts) for ts in timestamps]
            except:
                return {
                    'pattern_detected': False,
                    'pattern_strength': 0,
                    'expected_direction': 0,
                    'historical_performance': 0
                }
        
        # Get the current period identifier
        if period == 'day':
            current_period = current_timestamp.weekday()  # 0-6 (Monday-Sunday)
        elif period == 'week':
            current_period = (current_timestamp.day - 1) // 7  # 0-4 (weeks of month)
        elif period == 'month':
            current_period = current_timestamp.month - 1  # 0-11 (Jan-Dec)
        elif period == 'quarter':
            current_period = (current_timestamp.month - 1) // 3  # 0-3 (Q1-Q4)
        else:
            return {
                'pattern_detected': False,
                'pattern_strength': 0,
                'expected_direction': 0,
                'historical_performance': 0
            }
        
        # Find historical periods matching the current one
        similar_periods = []
        
        for i in range(lookback):
            if i >= len(timestamps) - 1:
                continue
                
            ts = timestamps[-(i+2)]  # Previous timestamps
            
            # Extract period identifier
            if period == 'day':
                period_id = ts.weekday()
            elif period == 'week':
                period_id = (ts.day - 1) // 7
            elif period == 'month':
                period_id = ts.month - 1
            elif period == 'quarter':
                period_id = (ts.month - 1) // 3
            else:
                continue
                
            # Check if this is a similar period
            if period_id == current_period:
                # Calculate return for this period
                if i < len(prices) - 1:
                    period_return = (prices[-(i+1)] - prices[-(i+2)]) / prices[-(i+2)]
                    similar_periods.append(period_return)
        
        # Calculate pattern strength and expected direction
        if similar_periods:
            positive_count = sum(1 for r in similar_periods if r > 0)
            negative_count = len(similar_periods) - positive_count
            
            if positive_count > negative_count:
                pattern_strength = positive_count / len(similar_periods)
                expected_direction = 1  # Bullish
            elif negative_count > positive_count:
                pattern_strength = negative_count / len(similar_periods)
                expected_direction = -1  # Bearish
            else:
                pattern_strength = 0.5
                expected_direction = 0  # Neutral
                
            avg_performance = sum(similar_periods) / len(similar_periods) * 100  # Percentage
            
            # Determine if a significant pattern exists
            pattern_detected = pattern_strength >= min_pattern_strength
            
            return {
                'pattern_detected': pattern_detected,
                'pattern_strength': pattern_strength,
                'expected_direction': expected_direction,
                'historical_performance': avg_performance,
                'sample_size': len(similar_periods)
            }
        
        return {
            'pattern_detected': False,
            'pattern_strength': 0,
            'expected_direction': 0,
            'historical_performance': 0,
            'sample_size': 0
        }


@register_feature(category="time")
class EventFeature(Feature):
    """
    Event detection feature.
    
    This feature identifies special events like earnings releases, holidays,
    economic announcements, etc. based on the calendar date.
    """
    
    def __init__(self, 
                 name: str = "event", 
                 params: Optional[Dict[str, Any]] = None,
                 description: str = "Calendar event detection"):
        """
        Initialize the event feature.
        
        Args:
            name: Feature name
            params: Dictionary containing:
                - format: Format string for timestamps (default: '%Y-%m-%d %H:%M:%S')
                - holidays: List or dictionary of holiday dates
                - events: Dictionary of other special events/dates
                - event_window: Number of days to consider around an event (default: 1)
            description: Feature description
        """
        super().__init__(name, params or self.default_params, description)
        
    @property
    def default_params(self) -> Dict[str, Any]:
        """Default parameters for event detection."""
        return {
            'format': '%Y-%m-%d %H:%M:%S',
            'holidays': None,
            'events': None,
            'event_window': 1
        }
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect calendar events based on date.
        
        Args:
            data: Dictionary containing timestamp information
        
        Returns:
            Dictionary with event information
        """
        timestamp_format = self.params.get('format', '%Y-%m-%d %H:%M:%S')
        holidays = self.params.get('holidays', None)
        events = self.params.get('events', None)
        event_window = self.params.get('event_window', 1)
        
        # Check for required timestamp data
        if 'timestamp' not in data:
            return {
                'is_holiday': False,
                'holiday_name': None,
                'is_event': False,
                'event_name': None,
                'days_to_event': None
            }
            
        # Parse timestamp
        timestamp = data['timestamp']
        
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(timestamp, timestamp_format)
            except ValueError:
                # Try some common formats if the specified one fails
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    return {
                        'is_holiday': False,
                        'holiday_name': None,
                        'is_event': False,
                        'event_name': None,
                        'days_to_event': None
                    }
        
        # Extract date components
        date_str = timestamp.strftime('%Y-%m-%d')
        month_day = timestamp.strftime('%m-%d')
        
        # Check if today is a holiday
        is_holiday = False
        holiday_name = None
        
        if holidays:
            if isinstance(holidays, dict):
                # Check exact dates (with year)
                if date_str in holidays:
                    is_holiday = True
                    holiday_name = holidays[date_str]
                    
                # Check recurring dates (no year)
                elif month_day in holidays:
                    is_holiday = True
                    holiday_name = holidays[month_day]
            elif isinstance(holidays, list):
                # Check if date is in the list
                is_holiday = date_str in holidays or month_day in holidays
                if is_holiday:
                    holiday_name = "Holiday"
        
        # Check for special events
        is_event = False
        event_name = None
        days_to_event = None
        
        if events:
            # First check if today is an event
            if date_str in events:
                is_event = True
                event_name = events[date_str]
                days_to_event = 0
            else:
                # Check if we're within the event window of any event
                current_date = timestamp.date()
                
                for event_date_str, name in events.items():
                    try:
                        # Parse event date
                        event_date = datetime.strptime(event_date_str, '%Y-%m-%d').date()
                        
                        # Calculate days difference
                        delta = (event_date - current_date).days
                        
                        # Check if within window
                        if abs(delta) <= event_window:
                            is_event = True
                            event_name = name
                            days_to_event = delta
                            break
                    except ValueError:
                        # Skip invalid date formats
                        continue
        
        return {
            'is_holiday': is_holiday,
            'holiday_name': holiday_name,
            'is_event': is_event,
            'event_name': event_name,
            'days_to_event': days_to_event
        }
