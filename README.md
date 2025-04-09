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