# Documentation Style Guide for LLM Analysis

## Overview
This document outlines our documentation standards optimized for LLM analysis. Following these guidelines will ensure our codebase can be efficiently analyzed by LLMs while maintaining human readability.

## Docstring Guidelines

### When to Include Docstrings
- **Required for:**
  - All modules
  - All classes
  - Public methods and functions with non-trivial logic
  - Functions with multiple parameters or return values
  - Complex algorithms or business logic

- **Not required for:**
  - Simple, self-explanatory functions (e.g., `is_even()`, `get_name()`)
  - Private helper methods with clear naming
  - Trivial property accessors

### Docstring Format
- Use Google-style docstring format for consistency
- Keep summaries concise (1-2 lines)
- Include parameters, return values, and exceptions when relevant
- Add examples for non-obvious usage

```python
def process_data(input_data, normalize=False):
    """Transform raw data into processed format.
    
    Args:
        input_data: The data to process (dict or DataFrame)
        normalize: Whether to normalize values (default: False)
        
    Returns:
        Processed data structure ready for analysis
        
    Raises:
        ValueError: If input_data is empty
    """
```

## Code Organization

### Modularity
- Keep files short (<300 lines when possible)
- Use Python packages (directories with `__init__.py` files) to organize related modules
- Structure your imports with the package system in mind (e.g., `import optimization` rather than individual files)
- One class per file for major components
- Group related functionality in appropriately named modules

### Imports
- List imports at the top of the file in alphabetical order
- Group standard library, third-party, and local imports

## Error Handling and Fallbacks

### No Mock Fallbacks
- **Mock fallbacks are strictly prohibited** - if a function or module fails, it must fail explicitly
- Do not use placeholder/dummy data when real data is unavailable
- Never silently degrade functionality with default values when critical operations fail
- Always raise appropriate exceptions with clear error messages when failures occur
- Example of what NOT to do:
  ```python
  # BAD PRACTICE - DO NOT DO THIS
  def calculate_returns(data):
      try:
          # Real calculation
          return data['price'].pct_change()
      except (KeyError, TypeError):
          # Mock fallback - THIS IS NOT ALLOWED
          print("Warning: Using dummy data")
          return pd.Series([0.001, -0.002, 0.003])  # Dummy values
  ```

### Standardized Calculations
- **All standardized financial calculations must use imported modules**
- Never reimplement standard calculations (returns, Sharpe ratio, drawdowns, etc.)
- Always import and use the appropriate utility functions:
  ```python
  # CORRECT APPROACH
  from utils.return_utils import calculate_returns, calculate_sharpe_ratio

  def analyze_performance(data):
      returns = calculate_returns(data)
      sharpe = calculate_sharpe_ratio(returns)
      return {'returns': returns, 'sharpe': sharpe}
  ```
- Document which utility modules should be used for which calculations
- Ensure consistent calculation methods across the entire codebase
- Benefits: Ensures reproducibility, consistency, and reduces bugs in critical calculations

## Architecture Documentation

### High-Level Architecture
- Maintain an `ARCHITECTURE.md` file in the project root
- Update with each significant change
- Include:
  - Component responsibilities
  - Data flow diagrams (text-based is acceptable)
  - Key interfaces between components

### Relationship Maps
- Document key relationships between components
- Use consistent notation:
  ```
  ComponentA → method_call → ComponentB
  ModelX ← depends on ← UtilityY
  ```

### Call Graphs
- Include token-efficient call graphs to show function relationships
- Use simple text-based formats:
  ```
  # Indented format
  main()
    ├─ process_data()
    │    └─ validate()
    └─ generate_report()
         ├─ format_output()
         └─ send_email()
  
  # Arrow format
  main() -> process_data() -> validate()
         -> generate_report() -> format_output()
                              -> send_email()
  ```
- Focus on high-level functions rather than every function call
- Include call graphs in module documentation or ARCHITECTURE.md
- Benefits: Helps LLMs understand control flow and function relationships with minimal tokens

### Progressive Disclosure
- Organize documentation hierarchically:
  1. High-level overview (ARCHITECTURE.md)
  2. Package-level documentation
  3. Module-level documentation
  4. Class/function-level documentation

## Documentation Maintenance

### Update Frequency
- Update architecture documentation with significant changes
- Update docstrings when function signatures or behavior changes

### Automation
- Use tools like Sphinx to auto-generate documentation
- Consider custom scripts to generate relationship maps and architecture overviews

### Optional Automation Hooks
- Generate architecture docs, dependency graphs, and package READMEs automatically using:
  - Custom Python scripts
  - Sphinx, MkDocs, Mermaid.js, or PlantUML
- Benefits: Supports documentation that stays current with code changes

### Centralized Configuration
- Consolidate all configuration into YAML or TOML files:
  ```yaml
  optimization:
    strategy: genetic
    population_size: 100
    mutation_rate: 0.05
  ```
- Benefits: Simplifies parameter management and enables LLMs/tools to reason about runtime behavior

## Example Documentation Structure

```
project/
├── ARCHITECTURE.md         # High-level overview
├── GLOSSARY.md             # Domain-specific terminology definitions
├── docs/
│   ├── architecture/       # Detailed architecture documents
│   │   ├── data_flow.md    # Data flow diagrams
│   │   └── components.md   # Component interactions
│   └── api/                # Auto-generated API docs
├── src/
│   ├── optimization/       # Package for optimization algorithms
│   │   ├── __init__.py     # Package exports and docstring with overview
│   │   ├── genetic.py      # Genetic algorithm implementation
│   │   └── bayesian.py     # Bayesian optimization implementation
│   ├── data_processing/    # Another package
│   │   ├── __init__.py     # Makes the directory a package
│   │   └── ...
│   └── utils/              # Utility functions package
│       ├── __init__.py     # Makes the directory a package
│       └── ...
├── tests/                  # Test directory
│   └── ...
└── config.yaml             # Centralized configuration
```

### Package Structure Example

In the `__init__.py` file, expose the package's public interface:

```python
# src/optimization/__init__.py
"""Optimization algorithms for finding optimal solutions.

This package provides various optimization strategies including
genetic algorithms and Bayesian optimization approaches.
"""

from .genetic import GeneticOptimizer, run_genetic_optimization
from .bayesian import BayesianOptimizer, optimize_with_bayesian

# This allows:
# from optimization import GeneticOptimizer
# instead of:
# from optimization.genetic import GeneticOptimizer
```

## LLM Optimization Techniques

### Token-Aware Naming & Layering
- Use descriptive, semantically rich names for functions, classes, and modules
- Avoid generic names like `do()` or `process()`
- Choose names that convey purpose and context
- Benefits: Improves code readability and enhances LLM token embeddings and suggestions

### Module Dependency Tags
- Add metadata at the top of each file to indicate relationships:
  ```python
  # @component: optimization
  # @depends_on: data_processing.normalizer
  # @used_by: trading_system.backtester
  ```
- Benefits: Enables tooling and LLMs to map and explain inter-module dependencies

### Glossary & Concept Indexing
- Create a `GLOSSARY.md` file with domain-specific terms
- Optionally annotate terms inline:
  ```python
  # glossary: "regime filter" = logic for adjusting strategy by market state
  ```
- Benefits: Helps LLMs and developers understand custom domain language

### Token Efficiency
- Front-load important information in summaries
- Use bulleted lists for parameters rather than lengthy paragraphs
- Include concrete types for parameters and return values

### Relationship Context
- Consider including call relationships in function docstrings (optional):
  ```python
  def process_data(input_data):
      """Process the input data.
      
      Called by: main(), batch_processor()
      Calls: validate(), transform_data()
      """
  ```
- Most valuable for core/complex functions and less necessary for utility functions
- Can be automated with documentation generators rather than maintained manually
- Document dependencies and the purpose of each dependency at the module level

### Semantic Structure
- Use consistent terminology throughout documentation
- Structure similar components with similar documentation patterns
- Include domain-specific terminology explanations in a glossary

### Unit Test Context Integration
- Annotate test files with references to modules they validate:
  ```python
  # tests/test_backtest.py
  # tests: src/trading/backtest_engine.py
  ```
- Benefits: Enhances traceability between tests and source files

### Literate Programming Support
- Consider org-mode or similar literate formats with `:tangle yes` for code blocks
- Maintain narrative structure around logic
- Benefits: Produces executable documentation and improves comprehension

By following these guidelines, our documentation will be optimized for LLM analysis while remaining useful for human developers.
