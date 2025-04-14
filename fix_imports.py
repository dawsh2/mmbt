#!/usr/bin/env python
"""
Script to fix relative imports in the codebase.
Run this script from your project root directory.
"""

import os
import re
import sys
from pathlib import Path

def find_python_files(root_dir):
    """Find all Python files in the directory tree."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # Skip temporary or backup files
            if file.startswith('.#') or file.startswith('#') or file.endswith('~'):
                continue
                
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def convert_relative_imports(file_path, src_dir):
    """Convert relative imports to absolute imports in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error opening file {file_path}: {e}")
        return False
    
    # Get the relative path from src_dir
    rel_path = os.path.relpath(file_path, src_dir)
    
    # Calculate the module path (dot notation)
    module_path = os.path.dirname(rel_path).replace(os.path.sep, '.')
    
    # Regular expression to match relative imports
    # Matches patterns like: from ..module import X or from . import X
    relative_import_pattern = r'^(\s*from\s+)(\.*)([\w\.]+)(\s+import\s+.+)$'
    
    modified_lines = []
    changes_made = False
    
    for line in content.split('\n'):
        match = re.match(relative_import_pattern, line)
        if match:
            from_part, dots, module_name, import_part = match.groups()
            
            # Skip if it's not actually a relative import (no dots)
            if not dots:
                modified_lines.append(line)
                continue
                
            # Calculate the absolute import path
            if dots == '.':
                # Same package import
                if module_name:
                    # from .module import X -> from src.module import X
                    new_import = f"{from_part}src.{module_path}.{module_name}{import_part}"
                else:
                    # from . import X -> from src.parent import X
                    new_import = f"{from_part}src.{module_path}{import_part}"
            else:
                # Parent package import
                parts = module_path.split('.')
                levels_up = len(dots) - 1
                
                if levels_up >= len(parts):
                    print(f"Warning: Cannot resolve relative import in {file_path}: {line}")
                    modified_lines.append(line)
                    continue
                    
                if module_name:
                    # from ..module import X -> from src.module import X
                    parent_path = '.'.join(parts[:-levels_up]) if levels_up < len(parts) else ''
                    if parent_path:
                        new_import = f"{from_part}src.{parent_path}.{module_name}{import_part}"
                    else:
                        new_import = f"{from_part}src.{module_name}{import_part}"
                else:
                    # from .. import X -> from src import X
                    parent_path = '.'.join(parts[:-levels_up]) if levels_up < len(parts) else ''
                    if parent_path:
                        new_import = f"{from_part}src.{parent_path}{import_part}"
                    else:
                        new_import = f"{from_part}src{import_part}"
            
            modified_lines.append(new_import)
            changes_made = True
            print(f"  Changed: {line} -> {new_import}")
        else:
            modified_lines.append(line)
    
    if changes_made:
        try:
            with open(file_path, 'w') as f:
                f.write('\n'.join(modified_lines))
            return True
        except (PermissionError, IOError) as e:
            print(f"Error writing to file {file_path}: {e}")
            return False
    
    return False

def main():
    # Get the src directory
    src_dir = os.path.join(os.getcwd(), 'src')
    
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        print("Make sure you're running this script from the project root")
        return
    
    print(f"Fixing imports in {src_dir}")
    
    # Find all Python files
    python_files = find_python_files(src_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Process each file
    modified_files = 0
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, os.getcwd())
        print(f"Processing {rel_path}")
        if convert_relative_imports(file_path, src_dir):
            modified_files += 1
    
    print(f"\nDone! Modified {modified_files} files")

if __name__ == "__main__":
    main()
