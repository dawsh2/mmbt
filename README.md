# Event-Driven Trading System Framework

A modular, event-driven backtester and optimization framework for algorithmic trading strategies.

## Overview

This framework provides a robust platform for developing, testing, and optimizing algorithmic trading strategies. It uses an event-driven architecture where components communicate through well-defined events, allowing for flexible and extensible design.

Key features include:
- Event-driven architecture for realistic simulation
- Modular design with decoupled components
- Multiple optimization approaches (genetic algorithms, walk-forward validation)
- Regime detection for market condition adaptation
- Ensemble strategy capabilities
- Comprehensive performance analytics

## Architecture

The system is organized around the following core components:

### Data Handling

- `CSVDataHandler`: Loads and manages time series data from CSV files, providing train/test data separation

### Event System

- `BarEvent`: Represents a market data update
- `MarketEvent`: Encapsulates market data updates
- `Signal`: Standardized format for trading signals with metadata

### Rule System

- `EventDrivenRuleSystem`: Manages and trains individual trading rules
- Various rule implementations (e.g., `Rule0` through `Rule15`) implementing different technical indicators and strategies

### Strategy Framework

- `Strategy`: Abstract base class for all trading strategies
- `TopNStrategy`: Combines signals from multiple rules using voting
- `WeightedRuleStrategy`: Combines rule signals using optimized weights
- `EnsembleStrategy`: Advanced strategy that combines multiple optimized strategies

### Backtesting

- `Backtester`: Simulates strategy execution on historical data

### Optimization

- `GeneticOptimizer`: Implements genetic algorithm for weight optimization
- `OptimizerManager`: Coordinates different optimization methods and sequences
- `WalkForwardValidator`: Implements walk-forward validation
- `CrossValidator`: Implements k-fold cross-validation
- `ValidatedEnsemble`: Combines validation and ensemble strategies

### Regime Detection

- `RegimeDetector`: Abstract base class for market regime detection
- `TrendStrengthRegimeDetector`: Detects market regimes based on trend strength
- `VolatilityRegimeDetector`: Detects market regimes based on volatility
- `RegimeManager`: Manages regime-specific strategies

### Analysis

- `TradeAnalyzer`: Comprehensive analysis of trading results

## Optimization Approaches

The framework supports several optimization approaches:

1. **Rules-First**: Optimize rule weights first, then apply to different regimes
2. **Regimes-First**: Identify regimes first, then optimize rule weights per regime
3. **Iterative**: Alternate between optimizing rules and regimes
4. **Joint**: Optimize rules and regime parameters simultaneously
5. **Walk-Forward**: Continuously re-optimize on rolling windows

## Modular Design Example

The `ensemble_validated.py` module demonstrates how the framework's modularity allows for advanced strategy development. It implements a validated ensemble approach that:

1. Optimizes multiple strategies using different metrics (Sharpe ratio, return, win rate)
2. Performs walk-forward validation to ensure robustness
3. Combines the strategies using weighted ensemble methods
4. Adapts to different market regimes

```python
# Example of creating a validated ensemble
validator = ValidatedEnsemble(
    data_filepath=DATA_FILE,
    rules_config=rules_config,
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    train_pct=TRAIN_PCT,
    top_n=TOP_N_RULES,
    optimization_metrics=['sharpe', 'return', 'win_rate'],
    ensemble_method='weighted',
    verbose=True
)

# Run validation
results = validator.run_validation(plot_results=True)

# Get best ensemble strategy
best_ensemble = validator.get_best_ensemble()
```

## Usage Examples

### 1. Basic Backtesting

```python
# Load data
data_handler = CSVDataHandler("data.csv", train_fraction=0.8)

# Create rules and strategy
rules_config = [
    (Rule0, {'fast_window': [5, 10], 'slow_window': [20, 30, 50]}),
    (Rule1, {'ma1': [10, 20], 'ma2': [30, 50]})
]
rule_system = EventDrivenRuleSystem(rules_config=rules_config, top_n=2)
rule_system.train_rules(data_handler)
strategy = rule_system.get_top_n_strategy()

# Run backtest
backtester = Backtester(data_handler, strategy)
results = backtester.run(use_test_data=True)
```

### 2. Genetic Optimization

```python
# Get top rule objects from rule system
top_rule_objects = list(rule_system.trained_rule_objects.values())

# Create and run genetic optimizer
optimizer = GeneticOptimizer(
    data_handler=data_handler,
    rule_objects=top_rule_objects,
    population_size=20,
    num_generations=50,
    optimization_metric='sharpe'
)
optimal_weights = optimizer.optimize()

# Create optimized strategy
optimized_strategy = WeightedRuleStrategy(
    rule_objects=top_rule_objects,
    weights=optimal_weights
)
```

### 3. Regime-Based Strategy

```python
# Create regime detector
regime_detector = TrendStrengthRegimeDetector(adx_period=14, adx_threshold=25)

# Create regime manager
regime_manager = RegimeManager(
    regime_detector=regime_detector,
    rule_objects=top_rule_objects,
    data_handler=data_handler
)

# Optimize strategies for different regimes
regime_manager.optimize_regime_strategies()

# Backtest the regime-based strategy
regime_backtester = Backtester(data_handler, regime_manager)
regime_results = regime_backtester.run(use_test_data=True)
```

## Advanced Features

- **Modular Optimization Pipeline**: Customize optimization sequences to fit your strategy requirements
- **Signal Router**: Standardizes signal flow between components, ensuring consistent formats
- **Comprehensive Analytics**: Detailed performance metrics and visualizations
- **Early Stopping**: Optimization routines include early stopping to improve efficiency
- **Multi-objective Optimization**: Optimize for multiple objectives like Sharpe ratio, return, win rate
- **Efficient Batch Processing**: Support for batch processing to handle large datasets

## Getting Started

1. Install required dependencies:
   ```
   pip install numpy pandas matplotlib
   ```

2. Prepare your historical data in CSV format

3. Create a configuration with your desired trading rules

4. Run a basic backtest or optimization using the example scripts

## Example Files

- `main.py`: Basic backtesting example
- `run_ga_optimization.py`: Genetic algorithm optimization example
- `run_regime_filter.py`: Regime-based strategy example
- `run_combined_optimization.py`: Combined optimization approaches
- `ensemble_validated.py`: Advanced ensemble strategy with validation

## Contributing

Contributions to improve the framework are welcome. Please feel free to submit issues or pull requests.



## OUTDATAED?? PRoofread rhtis
# Trading Strategy Backtesting Engine

A simple and modular backtesting engine for trading strategies with a focus on technical indicators and genetic algorithm optimization.

## Features

- Modular architecture with clean separation of concerns
- 16 built-in technical trading rules
- Rule parameter optimization
- Genetic algorithm weight optimization
- Performance metrics calculation
- Regime filtering
- Multiple strategy comparison

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: numpy, pandas, matplotlib

### Installation

1. Clone the repository or download the source code
2. Install the required dependencies:

```bash
pip install numpy pandas matplotlib
```

### Basic Usage

To run a backtest with default settings:

```bash
python main.py --data your_data.csv
```

The data file should contain OHLC (Open, High, Low, Close) price data with a Date column.

## Command Line Options

- `--train`: Train rule parameters
- `--test`: Test using trained parameters
- `--backtest`: Run both training and testing (default)
- `--data`: Path to the data file (CSV format)
- `--train-size`: Proportion of data to use for training (default: 0.6)
- `--top-n`: Number of top-performing rules to use for unweighted strategy (default: 5)
- `--no-weights`: Use unweighted strategy (top-n rules)
- `--ga-pop-size`: Population size for genetic algorithm (default: 8)
- `--ga-generations`: Number of generations for genetic algorithm (default: 100)
- `--ga-parents`: Number of parents for genetic algorithm (default: 4)
- `--filter-regime`: Apply regime filtering
- `--output`: Output file for results (default: results.json)
- `--save-params`: File to save/load trained parameters (default: params.json)

## Examples

### Train and test using weighted strategy (default)

```bash
python main.py --data your_data.csv --train-size 0.7
```

### Use the top 3 rules without GA weight optimization

```bash
python main.py --data your_data.csv --no-weights --top-n 3
```

### Train only (for parameter optimization)

```bash
python main.py --data your_data.csv --train
```

### Test only (using previously trained parameters)

```bash
python main.py --data your_data.csv --test
```

### Apply regime filtering

```bash
python main.py --data your_data.csv --filter-regime
```

## Architecture

The backtesting engine consists of the following components:

1. **Data Module** (`data.py`): Handles data loading, preprocessing, and train/test splitting
2. **Trading Rules** (`rules.py` & `ta_functions.py`): Implements technical trading rules and indicators
3. **Strategy Module** (`strategy.py`): Defines strategy types (TopN and Weighted)
4. **Backtester** (`backtester.py`): Core backtesting engine and performance evaluation
5. **Genetic Algorithm** (`ga.py`): Optimization of rule weights
6. **Metrics** (`metrics.py`): Performance metrics calculation
7. **Configuration** (`config.py`): Configuration management
8. **Main** (`main.py`): Entry point with CLI interface

## Adding New Rules

To add a new trading rule:

1. Add the rule function to the `TradingRules` class in `rules.py`
2. Update the `rule_functions` list in the `__init__` method
3. Make sure your rule function follows the same signature as the existing rules

## Customizing the GA Optimization

The genetic algorithm parameters can be customized through command line arguments:

```bash
python main.py --ga-pop-size 20 --ga-generations 200 --ga-parents 8
```

You can also modify the fitness function in `ga.py` to optimize for different objectives.