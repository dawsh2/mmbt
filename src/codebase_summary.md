# Codebase Summary for LLM

### `__init__.py`
📘 Module: Trading system package.


### `config/config_manager.py`
📘 Module: Main configuration manager for the trading system.
📦 Classes:
- Class `ConfigManager`: Central configuration management for the trading system.

### `config/validators.py`
📘 Module: Validation utilities for configuration.
🔧 Functions:
- Function `validate_config`: Validate the configuration against schema.
- Function `validate_section`: Recursively validate a configuration section.

### `config/__init__.py`
📘 Module: Configuration management for the trading system.

### `config/example_usage.py`
📘 Module: Example usage of the ConfigManager.
🔧 Functions:
- Function `main`: No docstring

### `config/defaults.py`
📘 Module: Default configuration values.

### `config/schema.py`
📘 Module: Configuration schema for validation.

### `regime_detection/__init__.py`
📘 Module: Market regime detection module for algorithmic trading.

### `regime_detection/regime_type.py`
📘 Module: Enumeration of market regime types for regime detection.
📦 Classes:
- Class `RegimeType`: Enumeration of different market regime types.

### `regime_detection/detector_registry.py`
📘 Module: Registry for market regime detectors.
📦 Classes:
- Class `DetectorRegistry`: Registry for market regime detectors.

### `regime_detection/detector_factory.py`
📘 Module: Factory for creating market regime detectors.
📦 Classes:
- Class `DetectorFactory`: Factory for creating regime detector instances.

### `regime_detection/regime_manager.py`
📘 Module: Regime manager for adapting trading strategies based on market regimes.
📦 Classes:
- Class `RegimeManager`: Manages trading strategies based on detected market regimes.

### `regime_detection/detector_base.py`
📘 Module: Base class for market regime detectors.
📦 Classes:
- Class `DetectorBase`: Abstract base class for regime detection algorithms.

### `regime_detection/detectors/trend_detectors.py`
📘 Module: Trend-based regime detectors.
📦 Classes:
- Class `TrendStrengthRegimeDetector`: Regime detector based on trend strength using the ADX indicator.

### `regime_detection/detectors/volatility_detectors.py`
📘 Module: Volatility-based regime detectors.
📦 Classes:
- Class `VolatilityRegimeDetector`: Regime detector based on market volatility.

### `regime_detection/detectors/composite_detectors.py`
📘 Module: Composite regime detectors.
📦 Classes:
- Class `CompositeDetector`: Composite detector that combines multiple regime detectors.

### `strategies/strategy_base.py`
📘 Module: Base Strategy Module
📦 Classes:
- Class `Strategy`: Base class for all trading strategies.

### `strategies/ensemble_strategy.py`
📘 Module: Ensemble Strategy Module
📦 Classes:
- Class `EnsembleStrategy`: Strategy that combines signals from multiple sub-strategies.

### `strategies/strategy_factory.py`
📘 Module: Strategy Factory Module
📦 Classes:
- Class `StrategyFactory`: Factory for creating strategy instances.

### `strategies/weighted_strategy.py`
📘 Module: Weighted Strategy Module
📦 Classes:
- Class `WeightedStrategy`: Strategy that combines signals from multiple rules using weights.

### `strategies/regime_strategy.py`
📘 Module: Regime Strategy Module
📦 Classes:
- Class `RegimeStrategy`: Strategy that adapts to different market regimes.

### `strategies/__init__.py`
📘 Module: Strategies Module

### `strategies/topn_strategy.py`
📘 Module: Top-N Strategy Module
📦 Classes:
- Class `TopNStrategy`: Strategy that combines signals from top N rules using consensus.

### `strategies/strategy_registry.py`
📘 Module: Strategy Registry Module
📦 Classes:
- Class `StrategyRegistry`: Registry of available strategies.

### `indicators/moving_averages.py`
📘 Module: Moving average indicators.
🔧 Functions:
- Function `simple_moving_average`: Calculate Simple Moving Average.
- Function `weighted_moving_average`: Calculate Weighted Moving Average.
- Function `exponential_moving_average`: Calculate Exponential Moving Average.
- Function `double_exponential_moving_average`: Calculate Double Exponential Moving Average.
- Function `triple_exponential_moving_average`: Calculate Triple Exponential Moving Average.
- Function `hull_moving_average`: Calculate Hull Moving Average.
- Function `kaufman_adaptive_moving_average`: Calculate Kaufman's Adaptive Moving Average (KAMA).
- Function `variable_index_dynamic_average`: Calculate Variable Index Dynamic Average (VIDYA).

### `indicators/trend.py`
📘 Module: Trend indicators.
🔧 Functions:
- Function `average_directional_index`: Calculate Average Directional Index (ADX).
- Function `moving_average_convergence_divergence`: Calculate Moving Average Convergence Divergence (MACD).
- Function `parabolic_sar`: Calculate Parabolic SAR (Stop and Reverse).
- Function `aroon`: Calculate Aroon indicators.
- Function `ichimoku_cloud`: Calculate Ichimoku Cloud components.
- Function `trix`: Calculate the TRIX indicator (Triple Exponential Average).
- Function `vortex_indicator`: Calculate Vortex Indicator.
- Function `supertrend`: Calculate SuperTrend indicator.

### `indicators/oscillators.py`
📘 Module: Oscillator indicators.
🔧 Functions:
- Function `relative_strength_index`: Calculate Relative Strength Index (RSI).
- Function `stochastic_oscillator`: Calculate Stochastic Oscillator.
- Function `commodity_channel_index`: Calculate Commodity Channel Index (CCI).
- Function `williams_r`: Calculate Williams %R.
- Function `money_flow_index`: Calculate Money Flow Index (MFI).
- Function `macd`: Calculate Moving Average Convergence Divergence (MACD).
- Function `rate_of_change`: Calculate Rate of Change (ROC).
- Function `awesome_oscillator`: Calculate Awesome Oscillator.

### `indicators/volume.py`
📘 Module: Volume indicators.
🔧 Functions:
- Function `on_balance_volume`: Calculate On-Balance Volume (OBV).
- Function `volume_price_trend`: Calculate Volume-Price Trend (VPT).
- Function `accumulation_distribution`: Calculate Accumulation/Distribution Line.
- Function `chaikin_oscillator`: Calculate Chaikin Oscillator.
- Function `money_flow_index`: Calculate Money Flow Index (MFI).
- Function `ease_of_movement`: Calculate Ease of Movement (EOM).
- Function `volume_weighted_average_price`: Calculate Volume Weighted Average Price (VWAP).
- Function `negative_volume_index`: Calculate Negative Volume Index (NVI).
- Function `positive_volume_index`: Calculate Positive Volume Index (PVI).

### `indicators/volatility.py`
📘 Module: Volatility indicators.
🔧 Functions:
- Function `average_true_range`: Calculate Average True Range (ATR).
- Function `bollinger_bands`: Calculate Bollinger Bands.
- Function `keltner_channels`: Calculate Keltner Channels.
- Function `donchian_channels`: Calculate Donchian Channels.
- Function `volatility_index`: Calculate a simple Volatility Index based on standard deviation.
- Function `chaikin_volatility`: Calculate Chaikin Volatility.
- Function `historical_volatility`: Calculate Historical Volatility.
- Function `standard_deviation`: Calculate Rolling Standard Deviation.

### `optimization/strategies.py`
📘 Module: Strategy implementations for the optimization framework.
📦 Classes:
- Class `WeightedComponentStrategy`: Strategy that combines signals from multiple components using weights.

### `optimization/__init__.py`
📘 Module: Unified optimization framework for trading systems.
📦 Classes:
- Class `OptimizationSequence`: Enumeration of optimization sequencing strategies.
- Class `ValidationMethod`: Enumeration of validation methods.

### `optimization/optimizer_manager.py`
📘 Module: Enhanced OptimizerManager with integrated grid search capabilities.
📦 Classes:
- Class `OptimizationMethod`: Enumeration of supported optimization methods.
- Class `OptimizerManager`: Enhanced manager for coordinating different optimization approaches.

### `optimization/example.py`
📘 Module: Example usage of the unified optimization framework.
🔧 Functions:
- Function `main`: No docstring

### `optimization/grid_search.py`
📘 Module: Grid search optimization module for trading system components.
📦 Classes:
- Class `GridOptimizer`: General-purpose grid search optimizer for any component.

### `optimization/components.py`
📘 Module: Component interfaces and factories for the optimization framework.
📦 Classes:
- Class `OptimizableComponent`: Abstract base class for any component that can be optimized.
- Class `ComponentFactory`: Factory for creating instances of optimizable components.
- Class `RuleFactory`: Factory for creating rule instances.
- Class `RegimeDetectorFactory`: Factory for creating regime detector instances.
- Class `StrategyFactory`: Factory for creating strategy instances.
- Class `WeightedStrategyFactory`: Factory for creating weighted strategies.

### `optimization/genetic_search.py`
📘 Module: Genetic optimization module for trading system components.
📦 Classes:
- Class `GeneticOptimizer`: Optimizes components using a genetic algorithm approach with regularization

### `optimization/evaluators.py`
📘 Module: Evaluator classes for different component types in the optimization framework.
📦 Classes:
- Class `RuleEvaluator`: Evaluator for trading rules.
- Class `RegimeDetectorEvaluator`: Evaluator for regime detectors.
- Class `StrategyEvaluator`: Evaluator for complete trading strategies.

### `optimization/validation/__init__.py`
📘 Module: Validation components for the optimization framework.

### `optimization/validation/utils.py`
📘 Module: Utility functions and classes for validation components.
📦 Classes:
- Class `WindowDataHandler`: Data handler for a specific window or fold.
🔧 Functions:
- Function `create_train_test_windows`: Create training and testing windows for walk-forward validation.

### `optimization/validation/walk_forward.py`
📦 Classes:
- Class `WalkForwardValidator`: Walk-Forward Validation for trading strategies.

### `optimization/validation/cross_val.py`
📘 Module: Cross-validation implementation for trading system optimization.
📦 Classes:
- Class `CrossValidator`: Cross-Validation for trading strategies.

### `optimization/validation/base.py`
📘 Module: Base validator interface for the optimization framework.
📦 Classes:
- Class `Validator`: Base abstract class for validation components.

### `optimization/validation/nested_cv.py`
📘 Module: Nested cross-validation implementation for trading system optimization.
📦 Classes:
- Class `NestedCrossValidator`: Nested Cross-Validation for more robust evaluation of trading strategies.

### `signals/signal_processing.py`
📘 Module: Signal Processing Module for Trading System
📦 Classes:
- Class `SignalType`: Enumeration of different signal types.
- Class `Signal`: Class representing a trading signal.
- Class `SignalFilter`: Base class for signal filters.
- Class `MovingAverageFilter`: Filter signals using a moving average.
- Class `ExponentialFilter`: Filter signals using exponential smoothing.
- Class `KalmanFilter`: Apply Kalman filtering to signals.
- Class `SignalTransform`: Base class for signal transformations.
- Class `WaveletTransform`: Apply wavelet transform to price data for multi-scale analysis.
- Class `BayesianConfidenceScorer`: Calculate confidence scores for signals using Bayesian methods.
- Class `SignalProcessor`: Coordinates signal processing operations including filtering, 

### `features/feature_registry.py`
📘 Module: Feature Registry Module
📦 Classes:
- Class `FeatureRegistry`: Registry for feature classes in the trading system.
🔧 Functions:
- Function `register_feature`: Decorator for registering feature classes with the registry.
- Function `register_features_in_module`: Register all Feature classes in a module with the registry.
- Function `get_registry`: Get the global feature registry instance.

### `features/price_features.py`
📘 Module: Price Features Module
📦 Classes:
- Class `ReturnFeature`: Calculate price returns over specified periods.
- Class `NormalizedPriceFeature`: Normalize price data relative to a reference point.
- Class `PricePatternFeature`: Detect specific price patterns in the data.
- Class `VolumeProfileFeature`: Calculate volume profile features based on price and volume data.
- Class `PriceDistanceFeature`: Calculate distance of price from various reference points.

### `features/time_features.py`
📘 Module: Time Features Module
📦 Classes:
- Class `TimeOfDayFeature`: Time of Day feature.
- Class `DayOfWeekFeature`: Day of Week feature.
- Class `MonthFeature`: Month feature.
- Class `SeasonalityFeature`: Seasonality feature.
- Class `EventFeature`: Event detection feature.

### `features/__init__.py`
📘 Module: Features Module
🔧 Functions:
- Function `create_default_feature_set`: Create a default set of commonly used features.

### `features/feature_utils.py`
📘 Module: Feature Utilities Module
🔧 Functions:
- Function `combine_features`: Combine multiple features into a composite feature.
- Function `weighted_average_combiner`: Combine feature values using a weighted average.
- Function `logical_combiner`: Combine feature values using logical operations.
- Function `threshold_combiner`: Combine feature values using thresholds to generate a signal.
- Function `cross_feature_indicator`: Create a crossover indicator from two features.
- Function `z_score_normalize`: Normalize feature values using Z-score normalization.
- Function `create_feature_vector`: Create a feature vector from multiple features.
- Function `combine_time_series_features`: Create a time series of feature vectors from multiple features.

### `features/feature_base.py`
📘 Module: Base Feature Module
📦 Classes:
- Class `Feature`: Base class for all features in the trading system.
- Class `FeatureSet`: A collection of features with convenience methods for batch calculation.
- Class `CompositeFeature`: A feature composed of multiple sub-features.
- Class `StatefulFeature`: A feature that maintains internal state between calculations.

### `tests/conftest.py`
📘 Module: Configuration for pytest.
🔧 Functions:
- Function `sample_price_data`: Create sample price data for testing.
- Function `sample_bar_data`: Create sample bar data for testing.

### `tests/test_features.py`
📦 Classes:
- Class `MockFeature`: Mock feature for testing.
- Class `TestFeatures`: Test suite for features.

### `tests/test_indicators.py`
📦 Classes:
- Class `TestMovingAverages`: Test suite for moving average indicators.

### `tests/test_rules.py`
📦 Classes:
- Class `MockFeature`: No docstring
- Class `MockRule`: No docstring
- Class `TestRules`: Test suite for rules.

### `tests/test_strategies.py`
📦 Classes:
- Class `MockEvent`: No docstring
- Class `MockStrategy`: No docstring
- Class `TestStrategies`: Test suite for strategies.

### `risk_management/analyzer.py`
📘 Module: Analysis engine for risk management metrics.
📦 Classes:
- Class `RiskAnalysisEngine`: Analyzes MAE, MFE, and ETD metrics to derive insights for risk management.

### `risk_management/collector.py`
📘 Module: Data collection module for risk management metrics.
📦 Classes:
- Class `RiskMetricsCollector`: Collects and stores MAE, MFE, and ETD metrics for analyzing trading performance.

### `risk_management/__init__.py`
📘 Module: Risk management package for algorithmic trading systems.

### `risk_management/types.py`
📘 Module: Type definitions and enums for the risk management module.
📦 Classes:
- Class `RiskToleranceLevel`: Enum for different risk tolerance levels.
- Class `ExitReason`: Enum for different exit reasons.
- Class `TradeMetrics`: Data class for storing trade metrics.
- Class `RiskParameters`: Data class for storing risk management parameters.
- Class `RiskAnalysisResults`: Data class for storing risk analysis results.

### `risk_management/parameter_optimizer.py`
📘 Module: Parameter optimization for risk management rules.
📦 Classes:
- Class `RiskParameterOptimizer`: Derives optimal risk management parameters from analyzed trade metrics.

### `risk_management/risk_manager.py`
📘 Module: Risk manager module for implementing MAE, MFE, and ETD based risk management.
📦 Classes:
- Class `RiskManager`: Applies risk management rules based on MAE, MFE, and ETD analysis.

### `rules/rule_base.py`
📘 Module: Rule Base Module
📦 Classes:
- Class `Rule`: Base class for all trading rules in the system.
- Class `CompositeRule`: A rule composed of multiple sub-rules.
- Class `FeatureBasedRule`: A rule that generates signals based on features.

### `rules/rule_registry.py`
📘 Module: Rule Registry Module
📦 Classes:
- Class `RuleRegistry`: Registry for rule classes in the trading system.
🔧 Functions:
- Function `register_rule`: Decorator for registering rule classes with the registry.
- Function `register_rules_in_module`: Register all Rule classes in a module with the registry.
- Function `get_registry`: Get the global rule registry instance.

### `rules/crossover_rules.py`
📘 Module: Crossover Rules Module
📦 Classes:
- Class `SMAcrossoverRule`: Simple Moving Average (SMA) Crossover Rule.
- Class `ExponentialMACrossoverRule`: Exponential Moving Average (EMA) Crossover Rule.
- Class `MACDCrossoverRule`: Moving Average Convergence Divergence (MACD) Crossover Rule.
- Class `PriceMACrossoverRule`: Price-Moving Average Crossover Rule.
- Class `BollingerBandsCrossoverRule`: Bollinger Bands Crossover Rule.
- Class `StochasticCrossoverRule`: Stochastic Oscillator Crossover Rule.

### `rules/trend_rules.py`
📘 Module: Trend Rules Module
📦 Classes:
- Class `ADXRule`: Average Directional Index (ADX) Rule.
- Class `IchimokuRule`: Ichimoku Cloud Rule.
- Class `VortexRule`: Vortex Indicator Rule.

### `rules/oscillator_rules.py`
📘 Module: Oscillator Rules Module
📦 Classes:
- Class `RSIRule`: Relative Strength Index (RSI) Rule.
- Class `StochasticRule`: Stochastic Oscillator Rule.
- Class `CCIRule`: Commodity Channel Index (CCI) Rule.
- Class `MACDHistogramRule`: MACD Histogram Rule.

### `rules/rule_factory.py`
📘 Module: Rule Factory Module
📦 Classes:
- Class `RuleFactory`: Factory for creating rule instances with proper parameter handling.
- Class `RuleOptimizer`: Optimizer for rule parameters based on performance metrics.
🔧 Functions:
- Function `create_rule`: Create a rule instance using the global registry.
- Function `create_composite_rule`: Create a composite rule from multiple rule configurations.

### `rules/volatility_rules.py`
📘 Module: Volatility Rules Module
📦 Classes:
- Class `BollingerBandRule`: Bollinger Bands Rule.
- Class `ATRTrailingStopRule`: Average True Range (ATR) Trailing Stop Rule.
- Class `VolatilityBreakoutRule`: Volatility Breakout Rule.
- Class `KeltnerChannelRule`: Keltner Channel Rule.

### `events/event.py`
📘 Module: Event system for the trading application.
📦 Classes:
- Class `EventType`: Enumeration of event types in the trading system.
- Class `Event`: Base class for all events in the trading system.
- Class `EventHandler`: Base class for event handlers.
- Class `FunctionEventHandler`: Event handler that delegates processing to a function.
- Class `EventBus`: Central event bus for routing events between system components.
- Class `EventEmitter`: Mixin class for components that emit events.
- Class `MarketDataHandler`: Event handler for market data events.
- Class `SignalHandler`: Event handler for signal events.
- Class `OrderHandler`: Event handler for order events.

### `engine/execution_engine.py`
📘 Module: Execution Engine for Trading System
📦 Classes:
- Class `EventType`: No docstring
- Class `Event`: Base class for all events in the system.
- Class `Order`: Represents a trading order.
- Class `Fill`: Represents an order fill (execution).
- Class `Position`: Represents a single position in a financial instrument.
- Class `Portfolio`: Manages a collection of positions and overall portfolio state.
- Class `ExecutionEngine`: Handles order execution, position tracking, and portfolio management.

### `engine/position_manager.py`
📘 Module: Position Management Module for Trading System
📦 Classes:
- Class `SizingStrategy`: Base class for position sizing strategies.
- Class `FixedSizingStrategy`: Position sizing using a fixed number of units.
- Class `PercentOfEquitySizing`: Size position as a percentage of portfolio equity.
- Class `VolatilityBasedSizing`: Size positions based on asset volatility.
- Class `KellySizingStrategy`: Position sizing based on the Kelly Criterion.
- Class `RiskManager`: Manages risk controls and limits for trading.
- Class `AllocationStrategy`: Base class for portfolio allocation strategies.
- Class `EqualAllocationStrategy`: Allocate capital equally across instruments.
- Class `VolatilityParityAllocation`: Allocate capital based on relative volatility of instruments.
- Class `PositionManager`: Manages position sizing, risk, and allocation decisions.

### `engine/backtester.py`
📘 Module: Backtester for Trading System
📦 Classes:
- Class `EventBus`: Simple event bus for handling and routing events in the system.
- Class `Backtester`: Main orchestration class that coordinates the backtest execution.
- Class `DefaultPositionManager`: Default position manager that implements basic functionality.
- Class `EventType`: No docstring
- Class `Event`: No docstring
- Class `Order`: No docstring

### `engine/market_simulator.py`
📘 Module: Market Simulator for Trading System
📦 Classes:
- Class `SlippageModel`: Base class for slippage models.
- Class `NoSlippageModel`: No slippage model - returns the base price unchanged.
- Class `FixedSlippageModel`: Fixed slippage model - applies a fixed basis point slippage to the price.
- Class `VolumeBasedSlippageModel`: Volume-based slippage model that scales with order size relative to volume.
- Class `FeeModel`: Base class for fee models.
- Class `NoFeeModel`: No fee model - returns zero fees.
- Class `FixedFeeModel`: Fixed fee model - applies a fixed basis point fee to the transaction value.
- Class `TieredFeeModel`: Tiered fee model with different fees based on transaction value.
- Class `MarketSimulator`: Simulates market effects like slippage, delays, and transaction costs.

### `logging/logging.py`
📘 Module: Logging system for the trading application.
📦 Classes:
- Class `TradeLogger`: Main logger interface for the trading system.
- Class `JsonFormatter`: Custom formatter that outputs log records as JSON.
- Class `LogContext`: Context manager for adding contextual information to logs.
- Class `ContextFilter`: Filter that adds context data to log records.
- Class `LogHandler`: Base class for custom log handlers with specialized processing.
- Class `AlertHandler`: Handler for generating alerts based on log messages.
- Class `DatabaseHandler`: Handler for storing log records in a database.
🔧 Functions:
- Function `log_execution_time`: Decorator to log the execution time of a function.
- Function `log_method_calls`: Decorator to log method entry and exit with arguments.

### `analytics/metrics.py`
📘 Module: Performance Metrics Module for Trading System
🔧 Functions:
- Function `calculate_metrics_from_trades`: Calculate key performance metrics from a list of trades.
- Function `calculate_max_drawdown`: Calculate the maximum drawdown from an equity curve.
- Function `calculate_consecutive_winloss`: Calculate maximum consecutive winning and losing trades.
- Function `calculate_monthly_returns`: Calculate monthly returns from trades.
- Function `calculate_drawdown_periods`: Identify significant drawdown periods in an equity curve.
- Function `calculate_regime_performance`: Calculate performance metrics by market regime.
- Function `analyze_trade_durations`: Analyze trade durations.

### `analytics/visualization.py`
📘 Module: Visualization Module for Trading System
📦 Classes:
- Class `TradeVisualizer`: Class for creating trading system visualizations.
