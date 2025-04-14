#!/usr/bin/env python
"""
A direct script to fix the import issues by manipulating the import system.
"""

import os
import sys

# Force Python to skip your custom logging module
class CustomImportFinder:
    def find_spec(self, fullname, path, target=None):
        # Skip src.logging to force Python to use the standard library
        if fullname == 'src.logging' or fullname == 'logging' and 'src' in sys.path[0]:
            return None
        return None

# Install the custom finder
sys.meta_path.insert(0, CustomImportFinder())

# Add src to the path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

# Now try to import and use your modules
try:
    from rules import SMARule, RSIRule, create_composite_rule
    from optimization import OptimizerManager, OptimizationMethod
    from strategies import WeightedStrategy
    from engine import Backtester
    from config import ConfigManager
    
    print("Successfully imported all modules!")
    
    # Your actual code can go here
    print("Ready to run your optimization code")
    
except ImportError as e:
    print(f"Import error: {e}")
    
    # Suggest removing the problematic module
    if 'logging.handlers' in str(e):
        print("\nRecommended Fix:")
        print("The issue is with your custom logging module conflicting with Python's standard library.")
        print("Please try either:")
        print("1. Rename the src/logging directory: mv src/logging src/log_system")
        print("2. Remove it temporarily: mv src/logging src/logging_backup")
        print("3. Modify src/logging/__init__.py to avoid the conflicting import")
