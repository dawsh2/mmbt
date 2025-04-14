"""
Validation components for the optimization framework.

This package provides different validation strategies for evaluating the robustness
of optimized trading components, including:
- Walk-forward validation
- K-fold cross-validation 
- Nested cross-validation
"""

# Import the base validator interface
from src.optimization.validation.base import Validator

# Import specific validator implementations
from src.optimization.validation.walk_forward import WalkForwardValidator
from src.optimization.validation.cross_val import CrossValidator
from src.optimization.validation.nested_cv import NestedCrossValidator

# Import utility classes
from src.optimization.validation.utils import WindowDataHandler, create_train_test_windows

__all__ = [
    'Validator',
    'WalkForwardValidator', 
    'CrossValidator',
    'NestedCrossValidator',
    'WindowDataHandler',
    'create_train_test_windows'
]
