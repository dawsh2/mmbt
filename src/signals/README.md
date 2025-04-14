# Signals Module Documentation

The Signals module provides a framework for generating, processing, and filtering trading signals. It enables signal quality improvement through filtering, transformation, and confidence scoring to enhance trading decision reliability.

## Core Concepts

**Signal**: Standardized container for trading signals with signal type, confidence score, and metadata.  
**SignalType**: Enumeration of different signal types (BUY, SELL, NEUTRAL).  
**SignalFilter**: Base class for signal filtering algorithms that remove noise from raw signals.  
**SignalTransform**: Base class for transformations that process signal data to extract additional insights.  
**SignalProcessor**: Coordinates signal processing operations including filtering, transformation, and confidence scoring.

## Basic Usage

```python
from signals import Signal, SignalType, SignalProcessor
from signals.signal_processing import MovingAverageFilter

# Create a raw signal
signal = Signal(
    timestamp="2023-06-15 10:30:00",
    signal_type=SignalType.BUY,
    price=150.75,
    rule_id="sma_crossover",
    confidence=0.8,
    metadata={"ma_fast": 10, "ma_slow": 30}
)

# Create signal processor with filtering
processor = SignalProcessor({
    'signals': {
        'processing': {
            'use_filtering': True,
            'filter_type': 'moving_average',
            'window_size': 5
        }
    }
})

# Process the signal
processed_signal = processor.process_signal(signal)

# Check processed signal
print(f"Signal type: {processed_signal.signal_type}")
print(f"Confidence: {processed_signal.confidence:.2f}")
print(f"Filtered: {processed_signal.metadata.get('filtered', False)}")
```

## API Reference

### Signal

Class representing a trading signal.

**Constructor Parameters:**
- `timestamp`: Signal timestamp
- `signal_type` (SignalType): Type of signal (BUY, SELL, NEUTRAL)
- `price` (float): Price at signal generation
- `rule_id` (str, optional): Identifier of the rule that generated the signal
- `confidence` (float, optional): Confidence score (0-1) (default: 1.0)
- `metadata` (dict, optional): Additional signal metadata
- `symbol` (str, optional): Instrument symbol (default: 'default')

**Methods:**
- `copy()`: Create a copy of the signal
  - Returns: New Signal object with the same attributes

**Class Methods:**
- `from_numeric(timestamp, signal_value, price, rule_id=None, metadata=None, symbol='default')`: Create a Signal object from a numeric signal value
  - `timestamp`: Signal timestamp
  - `signal_value` (int): Numeric signal value (-1, 0, 1)
  - `price` (float): Price at signal generation
  - `rule_id` (str, optional): Optional rule identifier
  - `metadata` (dict, optional): Additional signal data
  - `symbol` (str, optional): Instrument symbol
  - Returns: A new Signal object

**Example:**
```python
from signals import Signal, SignalType

# Create a signal directly
buy_signal = Signal(
    timestamp="2023-06-15 10:30:00",
    signal_type=SignalType.BUY,
    price=150.75,
    rule_id="rsi_oversold",
    confidence=0.9,
    metadata={"rsi_value": 28.5}
)

# Create from numeric value
sell_signal = Signal.from_numeric(
    timestamp="2023-06-15 14:45:00",
    signal_value=-1,  # -1 = SELL
    price=152.50,
    rule_id="bb_upper_touch"
)
```

### SignalType

Enumeration of different signal types.

```python
class SignalType(Enum):
    BUY = 1
    SELL = -1
    NEUTRAL = 0
```

### SignalFilter

Base class for signal filters.

**Methods:**
- `filter(signal, history=None)`: Filter a signal, potentially using history
  - `signal` (Signal): Input signal to filter
  - `history` (list, optional): Optional list of historical signals
  - Returns: Filtered signal
- `reset()`: Reset filter state

**Example:**
```python
from signals.signal_processing import SignalFilter

class CustomFilter(SignalFilter):
    def __init__(self):
        self.prev_signal = None
        
    def filter(self, signal, history=None):
        # Only allow changes in signal direction
        if self.prev_signal is None:
            self.prev_signal = signal
            return signal
            
        if signal.signal_type == self.prev_signal.signal_type:
            # No change, return neutral
            filtered = signal.copy()
            filtered.signal_type = SignalType.NEUTRAL
            filtered.metadata['filtered'] = True
            filtered.metadata['filter_type'] = 'custom'
            filtered.metadata['pre_filter_type'] = signal.signal_type
        else:
            # Signal change, pass through
            filtered = signal
            self.prev_signal = signal
            
        return filtered
        
    def reset(self):
        self.prev_signal = None
```

### MovingAverageFilter

Filter signals using a moving average.

**Constructor Parameters:**
- `window_size` (int, optional): Size of moving average window (default: 5)

**Methods:**
- `filter(signal, history=None)`: Apply moving average filter to signal
  - `signal` (Signal): Input signal to filter
  - `history` (list, optional): Optional list of historical signals
  - Returns: Filtered signal
- `reset()`: Reset filter state

**Example:**
```python
from signals.signal_processing import MovingAverageFilter

# Create filter with window of 3
ma_filter = MovingAverageFilter(window_size=3)

# Process a sequence of signals
filtered_signals = []
for signal in raw_signals:
    filtered = ma_filter.filter(signal)
    filtered_signals.append(filtered)
    
# Reset the filter for a new symbol
ma_filter.reset()
```

### ExponentialFilter

Filter signals using exponential smoothing.

**Constructor Parameters:**
- `alpha` (float, optional): Smoothing factor (0-1) (default: 0.2)
  Higher alpha gives more weight to recent signals

**Methods:**
- `filter(signal, history=None)`: Apply exponential filter to signal
  - `signal` (Signal): Input signal to filter
  - `history` (list, optional): Optional list of historical signals
  - Returns: Filtered signal
- `reset()`: Reset filter state

**Example:**
```python
from signals.signal_processing import ExponentialFilter

# Create filter with alpha of 0.3
exp_filter = ExponentialFilter(alpha=0.3)

# Process a sequence of signals
filtered_signals = []
for signal in raw_signals:
    filtered = exp_filter.filter(signal)
    filtered_signals.append(filtered)
```

### KalmanFilter

Apply Kalman filtering to signals.

**Constructor Parameters:**
- `process_variance` (float, optional): Process noise variance (Q) (default: 1e-5)
- `measurement_variance` (float, optional): Measurement noise variance (R) (default: 1e-2)

**Methods:**
- `filter(signal, history=None)`: Apply Kalman filter to signal
  - `signal` (Signal): Input signal to filter
  - `history` (list, optional): Optional list of historical signals
  - Returns: Filtered signal
- `reset()`: Reset filter state

**Example:**
```python
from signals.signal_processing import KalmanFilter

# Create Kalman filter
kalman_filter = KalmanFilter(
    process_variance=1e-4,
    measurement_variance=1e-2
)

# Process a sequence of signals
filtered_signals = []
for signal in raw_signals:
    filtered = kalman_filter.filter(signal)
    filtered_signals.append(filtered)
```

### SignalTransform

Base class for signal transformations.

**Methods:**
- `transform(data)`: Apply transformation to data
  - `data`: Input data to transform
  - Returns: Transformed data

**Example:**
```python
from signals.signal_processing import SignalTransform

class SimpleTransform(SignalTransform):
    def transform(self, data):
        # Example transformation: scale the data
        return [x * 2 for x in data]
```

### WaveletTransform

Apply wavelet transform to price data for multi-scale analysis.

**Constructor Parameters:**
- `wavelet` (str, optional): Wavelet type ('db1', 'haar', etc.) (default: 'db1')
- `level` (int, optional): Decomposition level (default: 3)

**Methods:**
- `add_price(price)`: Add price to history
  - `price` (float): Price to add
- `transform(data=None)`: Apply wavelet transform
  - `data` (list, optional): Optional data to transform (uses price history if None)
  - Returns: Wavelet coefficients at different scales
- `analyze_trend(coeffs=None)`: Analyze trend at different scales
  - `coeffs` (list, optional): Optional pre-computed wavelet coefficients
  - Returns: Dictionary with trend analysis
- `denoise(coeffs=None, threshold=0.1)`: Denoise data using wavelet thresholding
  - `coeffs` (list, optional): Optional pre-computed wavelet coefficients
  - `threshold` (float, optional): Threshold for coefficient removal (default: 0.1)
  - Returns: Denoised data

**Example:**
```python
from signals.signal_processing import WaveletTransform

# Create wavelet transform
wavelet = WaveletTransform(wavelet='db4', level=3)

# Add price data
for price in price_data:
    wavelet.add_price(price)
    
# Transform and analyze
coeffs = wavelet.transform()
trend_analysis = wavelet.analyze_trend(coeffs)
denoised_data = wavelet.denoise(coeffs, threshold=0.05)

print(f"Main trend: {trend_analysis['main_trend']}")
```

### BayesianConfidenceScorer

Calculate confidence scores for signals using Bayesian methods.

**Constructor Parameters:**
- `prior_accuracy` (float, optional): Initial belief about signal accuracy (0-1) (default: 0.5)
- `smoothing` (int, optional): Controls how quickly scores adapt to new data (default: 10)

**Methods:**
- `record_signal_result(signal, was_correct)`: Record whether a signal led to a profitable trade
  - `signal` (Signal): The original signal
  - `was_correct` (bool): Whether the signal led to a profitable trade
- `calculate_confidence(signal, context=None)`: Calculate confidence score for a signal
  - `signal` (Signal): The signal to score
  - `context` (dict, optional): Optional context dict (regime, market conditions, etc.)
  - Returns: Confidence score between 0-1

**Example:**
```python
from signals.signal_processing import BayesianConfidenceScorer

# Create confidence scorer
scorer = BayesianConfidenceScorer(prior_accuracy=0.6, smoothing=5)

# Calculate confidence for a new signal
context = {"regime": "trending_up", "volatility": "low"}
confidence = scorer.calculate_confidence(signal, context)

# After signal results are known
was_profitable = True  # or False
scorer.record_signal_result(signal, was_profitable)
```

### SignalProcessor

Coordinates signal processing operations including filtering, transformation, and confidence scoring.

**Constructor Parameters:**
- `config` (dict|object, optional): Configuration dictionary or object

**Methods:**
- `process_signal(signal, context=None)`: Process a signal through filtering, transformation, and confidence scoring
  - `signal` (Signal): The signal to process
  - `context` (dict, optional): Optional context information
  - Returns: Processed signal
- `process_price_data(prices, symbol='default')`: Process price data for additional insights
  - `prices` (list): List or array of price data
  - `symbol` (str, optional): Instrument symbol
  - Returns: Dictionary with analysis results
- `record_trade_result(signal, was_profitable)`: Record trade result for confidence model learning
  - `signal` (Signal): The signal that led to the trade
  - `was_profitable` (bool): Whether the trade was profitable
- `reset()`: Reset all signal processing components

**Example:**
```python
from signals import SignalProcessor
from config import ConfigManager

# Create configuration
config = ConfigManager()
config.set('signals.processing.use_filtering', True)
config.set('signals.processing.filter_type', 'exponential')
config.set('signals.confidence.use_confidence_score', True)

# Create signal processor
processor = SignalProcessor(config)

# Process a signal
processed_signal = processor.process_signal(signal, context={"regime": "trending_up"})

# Process price data for insights
analysis = processor.process_price_data(price_history, symbol='AAPL')

# Record results for learning
processor.record_trade_result(processed_signal, was_profitable=True)
```

## Advanced Usage

### Building a Signal Processing Pipeline

```python
from signals import Signal, SignalType, SignalProcessor
from signals.signal_processing import (
    MovingAverageFilter, KalmanFilter, BayesianConfidenceScorer
)

class SignalPipeline:
    """Custom signal processing pipeline with multiple filters."""
    
    def __init__(self, config=None):
        # Initialize components
        self.ma_filter = MovingAverageFilter(window_size=5)
        self.kalman_filter = KalmanFilter()
        self.confidence_scorer = BayesianConfidenceScorer(prior_accuracy=0.55)
        
        # Settings
        self.min_confidence = config.get('min_confidence', 0.6) if config else 0.6
        
    def process(self, signal, context=None):
        """
        Process a signal through the pipeline.
        
        Args:
            signal: Raw signal to process
            context: Optional context information
            
        Returns:
            Processed signal and action flag
        """
        # Apply moving average filter
        filtered_signal = self.ma_filter.filter(signal)
        
        # Apply Kalman filter for additional smoothing
        filtered_signal = self.kalman_filter.filter(filtered_signal)
        
        # Calculate confidence score
        confidence = self.confidence_scorer.calculate_confidence(filtered_signal, context)
        filtered_signal.confidence = confidence
        
        # Determine if signal should generate action
        should_act = (
            filtered_signal.signal_type != SignalType.NEUTRAL and 
            filtered_signal.confidence >= self.min_confidence
        )
        
        return filtered_signal, should_act
        
    def record_result(self, signal, was_profitable):
        """Record signal result for confidence learning."""
        self.confidence_scorer.record_signal_result(signal, was_profitable)
        
    def reset(self):
        """Reset pipeline state."""
        self.ma_filter.reset()
        self.kalman_filter.reset()

# Usage
pipeline = SignalPipeline()

# Process signals
for raw_signal in incoming_signals:
    context = {"market_regime": current_regime, "volatility": current_volatility}
    processed_signal, should_act = pipeline.process(raw_signal, context)
    
    if should_act:
        # Execute trade based on signal
        execute_trade(processed_signal)
```

### Signal Transformation with Wavelets

```python
from signals.signal_processing import WaveletTransform
import numpy as np
import matplotlib.pyplot as plt

class WaveletAnalyzer:
    """Analyze price data using wavelet transforms for signal denoising and feature extraction."""
    
    def __init__(self, wavelet='db4', level=3):
        self.transform = WaveletTransform(wavelet=wavelet, level=level)
        self.price_history = []
        
    def add_data(self, prices):
        """Add price data for analysis."""
        for price in prices:
            self.transform.add_price(price)
            self.price_history.append(price)
    
    def denoise_prices(self, threshold=0.1):
        """Remove noise from price data using wavelets."""
        # Transform data
        coeffs = self.transform.transform()
        
        # Apply denoising
        denoised = self.transform.denoise(coeffs, threshold=threshold)
        
        return denoised
        
    def extract_features(self):
        """Extract trend features at different scales."""
        # Transform data
        coeffs = self.transform.transform()
        
        # Analyze trend
        trend_info = self.transform.analyze_trend(coeffs)
        
        # Extract additional features
        features = {
            'main_trend': trend_info['main_trend'],
            'detail_trends': trend_info['detail_trends'],
            'trend_strength': abs(trend_info['main_trend']),
            'noise_level': self._calculate_noise_level(coeffs)
        }
        
        return features
    
    def _calculate_noise_level(self, coeffs):
        """Calculate noise level from wavelet coefficients."""
        # Use highest frequency detail coefficients
        detail_coeffs = coeffs[-1]
        
        # Calculate noise as normalized energy in highest frequency band
        noise = np.sum(np.square(detail_coeffs)) / len(detail_coeffs)
        
        return noise
        
    def plot_analysis(self):
        """Plot original, denoised prices and trend components."""
        if not self.price_history:
            return
            
        # Get denoised data
        denoised = self.denoise_prices()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot original and denoised prices
        plt.subplot(2, 1, 1)
        plt.plot(self.price_history, label='Original Prices', alpha=0.7)
        plt.plot(denoised, label='Denoised Prices', linewidth=2)
        plt.legend()
        plt.title('Price Denoising using Wavelets')
        
        # Get wavelet coefficients
        coeffs = self.transform.transform()
        
        # Plot approximation and details
        plt.subplot(2, 1, 2)
        plt.plot(coeffs[0], label='Approximation (Trend)', linewidth=2)
        for i, detail in enumerate(coeffs[1:], 1):
            plt.plot(detail, label=f'Detail Level {i}', alpha=0.5)
        plt.legend()
        plt.title('Wavelet Decomposition')
        
        plt.tight_layout()
        return plt.gcf()

# Usage
analyzer = WaveletAnalyzer(wavelet='db4', level=3)
analyzer.add_data(price_history)

# Get denoised prices for trading
denoised_prices = analyzer.denoise_prices(threshold=0.08)

# Extract features for signal generation
features = analyzer.extract_features()
print(f"Main trend: {features['main_trend']}")
print(f"Trend strength: {features['trend_strength']}")
print(f"Noise level: {features['noise_level']}")

# Generate signal based on wavelet features
if features['main_trend'] > 0 and features['trend_strength'] > 0.5 and features['noise_level'] < 0.2:
    signal_type = SignalType.BUY
elif features['main_trend'] < 0 and features['trend_strength'] > 0.5 and features['noise_level'] < 0.2:
    signal_type = SignalType.SELL
else:
    signal_type = SignalType.NEUTRAL
```

### Bayesian Confidence Scoring System

```python
from signals.signal_processing import BayesianConfidenceScorer
from signals import Signal, SignalType
import numpy as np

class AdvancedConfidenceScorer:
    """
    Advanced confidence scoring system with regime-specific learning
    and signal feature analysis.
    """
    
    def __init__(self):
        # Create regime-specific scorers
        self.scorers = {
            'trending_up': BayesianConfidenceScorer(prior_accuracy=0.6),
            'trending_down': BayesianConfidenceScorer(prior_accuracy=0.6),
            'range_bound': BayesianConfidenceScorer(prior_accuracy=0.5),
            'volatile': BayesianConfidenceScorer(prior_accuracy=0.4),
            'default': BayesianConfidenceScorer(prior_accuracy=0.5)
        }
        
        # Feature importance weights
        self.feature_weights = {
            'signal_strength': 0.3,
            'historical_accuracy': 0.5,
            'confluence': 0.2
        }
    
    def calculate_confidence(self, signal, context=None):
        """
        Calculate confidence score for a signal.
        
        Args:
            signal: The signal to score
            context: Market context (regime, volatility, etc.)
            
        Returns:
            float: Confidence score between 0-1
        """
        # Get current regime
        regime = context.get('regime', 'default') if context else 'default'
        
        # Get appropriate scorer
        scorer = self.scorers.get(regime, self.scorers['default'])
        
        # Get base confidence from Bayesian model
        base_confidence = scorer.calculate_confidence(signal)
        
        # Calculate additional confidence factors
        signal_strength = self._calculate_signal_strength(signal)
        confluence = self._calculate_confluence(signal, context)
        
        # Combine factors with weights
        weighted_confidence = (
            base_confidence * self.feature_weights['historical_accuracy'] +
            signal_strength * self.feature_weights['signal_strength'] +
            confluence * self.feature_weights['confluence']
        )
        
        # Ensure confidence is in 0-1 range
        return min(max(weighted_confidence, 0.0), 1.0)
    
    def record_trade_result(self, signal, was_profitable, context=None):
        """
        Record trade result for learning.
        
        Args:
            signal: The signal that led to the trade
            was_profitable: Whether the trade was profitable
            context: Market context when signal was generated
        """
        # Get regime
        regime = context.get('regime', 'default') if context else 'default'
        
        # Record with appropriate scorer
        scorer = self.scorers.get(regime, self.scorers['default'])
        scorer.record_signal_result(signal, was_profitable)
    
    def _calculate_signal_strength(self, signal):
        """Calculate strength factor based on signal metadata."""
        strength = 0.5  # Default value
        
        # Check for signal strength indicators in metadata
        if hasattr(signal, 'metadata') and signal.metadata:
            # RSI extremes indicate stronger signals
            if 'rsi' in signal.metadata:
                rsi = signal.metadata['rsi']
                if signal.signal_type == SignalType.BUY and rsi < 30:
                    # Stronger buy signal when RSI is lower
                    strength = 0.5 + (30 - rsi) / 60  # 0.5-0.8 range
                elif signal.signal_type == SignalType.SELL and rsi > 70:
                    # Stronger sell signal when RSI is higher
                    strength = 0.5 + (rsi - 70) / 60  # 0.5-0.8 range
            
            # Crossover distance indicates signal strength
            if 'crossover_distance' in signal.metadata:
                # Normalize to 0-0.3 range
                dist = min(signal.metadata['crossover_distance'] / 0.05, 1.0) * 0.3
                strength = max(strength, 0.5 + dist)
        
        return strength
    
    def _calculate_confluence(self, signal, context=None):
        """Calculate confluence factor based on multiple indicators."""
        if not (hasattr(signal, 'metadata') and signal.metadata):
            return 0.5
            
        # Check if metadata contains confluence data
        if 'indicator_agreement' in signal.metadata:
            return signal.metadata['indicator_agreement']
            
        # Calculate from individual indicators if available
        indicators = []
        
        # Check for common indicators in metadata
        indicator_keys = ['rsi', 'macd', 'bb', 'adx', 'sma_cross']
        for key in indicator_keys:
            if key in signal.metadata:
                indicators.append(signal.metadata[key])
        
        if not indicators:
            return 0.5
            
        # Convert indicator values to -1, 0, 1 signals
        signals = []
        for indicator in indicators:
            if isinstance(indicator, bool):
                signals.append(1 if indicator else 0)
            elif isinstance(indicator, (int, float)):
                # Assuming standardized indicator values
                if indicator > 0.5:
                    signals.append(1)
                elif indicator < -0.5:
                    signals.append(-1)
                else:
                    signals.append(0)
            elif indicator in ['buy', 'long', 'bullish']:
                signals.append(1)
            elif indicator in ['sell', 'short', 'bearish']:
                signals.append(-1)
            else:
                signals.append(0)
        
        # Calculate agreement with signal direction
        if signals:
            direction = signal.signal_type.value
            agreements = [1 if s * direction > 0 else 0 for s in signals]
            return sum(agreements) / len(agreements)
            
        return 0.5

# Usage
confidence_system = AdvancedConfidenceScorer()

# Calculate confidence for a signal
context = {
    'regime': 'trending_up',
    'volatility': 'low',
    'market_conditions': 'normal'
}

signal = Signal(
    timestamp="2023-06-15 10:30:00",
    signal_type=SignalType.BUY,
    price=150.75,
    metadata={
        'rsi': 28,
        'macd': 'bullish',
        'bb': True,
        'crossover_distance': 0.03
    }
)

confidence = confidence_system.calculate_confidence(signal, context)
print(f"Signal confidence: {confidence:.2f}")

# After trade completes
confidence_system.record_trade_result(signal, was_profitable=True, context=context)
```

### Integration with the Configuration System

```python
from signals import SignalProcessor
from config import ConfigManager

def create_signal_processor_from_config(config_file=None):
    """
    Create a signal processor with configuration from file or defaults.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured SignalProcessor instance
    """
    # Create config manager
    config = ConfigManager()
    
    # Load from file if provided
    if config_file:
        config.load_from_file(config_file)
    
    # Default configuration for different market conditions
    default_configs = {
        'normal': {
            'signals.processing.use_filtering': True,
            'signals.processing.filter_type': 'moving_average',
            'signals.processing.window_size': 5,
            'signals.confidence.use_confidence_score': True,
            'signals.confidence.min_confidence': 0.6
        },
        'volatile': {
            'signals.processing.use_filtering': True,
            'signals.processing.filter_type': 'kalman',
            'signals.confidence.use_confidence_score': True,
            'signals.confidence.min_confidence': 0.7,
            'signals.processing.use_transformations': True
        },
        'trending': {
            'signals.processing.use_filtering': True,
            'signals.processing.filter_type': 'exponential',
            'signals.processing.window_size': 3,
            'signals.confidence.use_confidence_score': True,
            'signals.confidence.min_confidence': 0.5
        }
    }
    
    # Get market condition from config or use default
    market_condition = config.get('market.condition', 'normal')
    
    # Apply appropriate default config based on market condition
    default_config = default_configs.get(market_condition, default_configs['normal'])
    for key, value in default_config.items():
        if not config.has_key(key):
            config.set(key, value)
    
    # Create signal processor with config
    processor = SignalProcessor(config)
    
    return processor

# Usage
processor = create_signal_processor_from_config('trading_config.yaml')
processed_signal = processor.process_signal(signal)
```

## Best Practices

1. **Filter appropriately for timeframe**: Use less aggressive filtering for higher timeframes, more aggressive for lower timeframes

2. **Tune filter parameters**: Test different filter parameters to balance responsiveness and noise reduction

3. **Combine multiple filters**: Different filters excel at removing different types of noise; consider combining them

4. **Use confidence scoring**: Quantify signal reliability to make better trading decisions

5. **Start conservative**: Begin with higher confidence thresholds and relax after validation

6. **Record and analyze signal performance**: Track signal outcomes to continuously improve confidence models

7. **Consider market regimes**: Different filtering and transformation approaches work best in different market conditions

8. **Use transformations for additional insights**: Wavelets and other transforms can reveal patterns not visible in raw data

9. **Implement proper state management**: Always reset filter states when switching symbols or timeframes

10. **Balance robustness and responsiveness**: Excessive filtering reduces noise but can introduce harmful lag in signal generation