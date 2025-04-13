"""
Validation components for the optimization framework.

This package provides different validation strategies for evaluating the robustness
of optimized trading components, including:
- Walk-forward validation
- K-fold cross-validation 
- Nested cross-validation
"""

# Import the base validator interface
from .base import Validator

# Import specific validator implementations
from .walk_forward import WalkForwardValidator
from .cross_val import CrossValidator
from .nested_cv import NestedCrossValidator

# Import utility classes
from .utils import WindowDataHandler, create_train_test_windows

__all__ = [
    'Validator',
    'WalkForwardValidator', 
    'CrossValidator',
    'NestedCrossValidator',
    'WindowDataHandler',
    'create_train_test_windows'
]
