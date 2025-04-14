# API Documentation Generator Guide

This guide provides instructions for generating concise, comprehensive API documentation for code modules. The goal is to create documentation that allows users to effectively use a module's API without needing access to the implementation code.

## Documentation Objectives

When generating API documentation, aim to:

1. **Be concise but complete** - Document all public interfaces without unnecessary verbosity
2. **Focus on usage** - Prioritize "how to use" over "how it works"
3. **Include all parameters** - Document all parameters with types and default values
4. **Provide examples** - Include practical examples for common use cases
5. **Structure logically** - Organize by concepts, classes, or functions

## Documentation Template

### 1. Module Overview
Start with a brief (1-3 sentence) description of what the module does and its primary purpose.

### 2. Core Concepts
Explain fundamental concepts and terms used throughout the module. Keep this section brief (1-2 paragraphs).

### 3. Basic Usage
Provide a minimal working example that demonstrates the most common use case. This should be complete and runnable.

### 4. API Reference
Document all public classes, functions, and methods with:
- Function/class signatures
- Parameter descriptions and default values
- Return types and descriptions
- Example usage

### 5. Advanced Usage
Include examples of advanced patterns, configurations, or combinations.

## Example Structure

```markdown
# Module Name Documentation

Brief description of the module's purpose and functionality.

## Core Concepts

Explanation of key concepts and terminology.

## Basic Usage

```python
# Simple example of most common use case
from module import primary_class

instance = primary_class(param1=value1)
result = instance.main_method()
```

## API Reference

### ClassName

Description of the class.

**Constructor Parameters:**
- `param1` (type): Description (default: default_value)
- `param2` (type): Description (default: default_value)

**Methods:**
- `method_name(param1, param2)`: Description
  - `param1` (type): Description
  - `param2` (type): Description
  - Returns: Description of return value

**Example:**
```python
# Example usage
```

### function_name(param1, param2)

Description of the function.

**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Returns:**
- (type): Description

**Example:**
```python
# Example usage
```

## Advanced Usage

Examples of advanced usage patterns.
```

## Documentation Generation Process

When generating documentation:

1. **Analyze the code** - Examine imports, class definitions, function signatures, docstrings, and usage patterns
2. **Identify public interfaces** - Focus on what users are expected to interact with
3. **Extract parameters** - List all parameters, their types, defaults, and purposes
4. **Create examples** - Write concise, realistic examples covering common use cases
5. **Organize hierarchically** - Present information in order of increasing complexity
6. **Edit for clarity** - Remove redundancy and ensure explanations are clear and concise

## Module-Specific Guidelines

### For Rule-Based Systems
- Document all rule types and their specific parameters
- Clarify how rules interact and can be combined
- Explain rule evaluation and signal generation

### For Data Processing
- Document expected input and output formats
- Clarify transformation pipelines and ordering requirements
- Include examples with sample data

### For Configuration Systems
- Document all configuration options and formats
- Show complete configuration examples
- Explain default behaviors and fallbacks

## Final Checklist

Before finalizing documentation, verify:

- [ ] All public classes and functions are documented
- [ ] All parameters are listed with types and defaults
- [ ] Return values are documented
- [ ] Examples cover basic and advanced usage
- [ ] Documentation is concise without unnecessary explanation
- [ ] Users can understand the API without seeing implementation
- [ ] Edge cases and common pitfalls are mentioned

## Example Documentation

For reference, see the example documentation of the Rules module, which follows this template and demonstrates effective API documentation that allows users to utilize the module without requiring access to implementation details.