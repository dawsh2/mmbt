#!/usr/bin/env python
"""
Script to fix all imports in the codebase.
"""

import os
import re
import sys

def fix_file(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Replace relative imports with absolute ones
    modified_content = content
    
    # Fix "from . import X" -> "from src.current_package import X"
    rel_path = os.path.relpath(file_path, os.path.join(os.getcwd(), 'src'))
    package_path = os.path.dirname(rel_path).replace(os.path.sep, '.')
    modified_content = re.sub(r'from\s+\.\s+import', f'from src.{package_path} import', modified_content)
    
    # Fix "from .module import X" -> "from src.current_package.module import X"
    modified_content = re.sub(r'from\s+\.([a-zA-Z0-9_]+)', f'from src.{package_path}.\\1', modified_content)
    
    # Fix "from .. import X" -> "from src import X"
    modified_content = re.sub(r'from\s+\.\.\s+import', 'from src import', modified_content)
    
    # Fix "from ..module import X" -> "from src.module import X"
    modified_content = re.sub(r'from\s+\.\.([a-zA-Z0-9_]+)', 'from src.\\1', modified_content)
    
    # Fix "from logging import X" -> "from log_system import X"
    modified_content = re.sub(r'from\s+logging\s+import\s+(?!handlers|config|basicConfig)',
                           'from src.log_system import ', modified_content)
    
    # Fix "import logging" -> "import src.log_system as logging"
    modified_content = re.sub(r'import\s+logging\s*$', 'import src.log_system as logging', modified_content)
    
    # Write changes back if needed
    if modified_content != content:
        try:
            with open(file_path, 'w') as f:
                f.write(modified_content)
            print(f"Updated imports in {file_path}")
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False
    
    return False

def process_directory(directory):
    """Process all Python files in a directory."""
    files_changed = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.startswith('.#'):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    files_changed += 1
    
    return files_changed

if __name__ == "__main__":
    src_dir = os.path.join(os.getcwd(), 'src')
    
    if not os.path.isdir(src_dir):
        print(f"Error: {src_dir} not found. Run this from project root.")
        sys.exit(1)
    
    print(f"Processing Python files in {src_dir}...")
    files_changed = process_directory(src_dir)
    print(f"Updated imports in {files_changed} files")
