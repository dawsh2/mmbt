#!/usr/bin/env python
"""
Script to rename the logging module and update references.
Run this script from your project root directory.
"""

import os
import re
import shutil
import sys

def rename_directory(src_path, new_name):
    """Rename a directory."""
    dir_path = os.path.dirname(src_path)
    new_path = os.path.join(dir_path, new_name)
    
    print(f"Renaming directory: {src_path} -> {new_path}")
    
    try:
        # Create new directory if it doesn't exist
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            
        # Copy all files
        for item in os.listdir(src_path):
            item_path = os.path.join(src_path, item)
            if os.path.isfile(item_path):
                shutil.copy2(item_path, os.path.join(new_path, item))
        
        print(f"Successfully copied files from {src_path} to {new_path}")
        return True
    except Exception as e:
        print(f"Error renaming directory: {e}")
        return False

def update_imports(directory, old_module, new_module):
    """Update import statements in all Python files."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_imports_in_file(file_path, old_module, new_module)

def update_imports_in_file(file_path, old_module, new_module):
    """Update imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error opening file {file_path}: {e}")
        return False
    
    # Pattern for import statements
    import_patterns = [
        # from src.logging import X -> from src.log_system import X
        rf"from\s+src\.{old_module}([\.\s])",
        # import src.logging -> import src.log_system
        rf"import\s+src\.{old_module}([\.\s])",
        # from logging import X -> from log_system import X (be careful with this one)
        rf"from\s+{old_module}([\.\s])",
        # import logging (be careful with this one)
        rf"import\s+{old_module}([\.\s])"
    ]
    
    modified_content = content
    changes_made = False
    
    for pattern in import_patterns:
        # For simple replacements that don't involve the standard library
        if 'src.' in pattern:
            replacement = rf"from src.{new_module}\1" if "from" in pattern else rf"import src.{new_module}\1"
            new_content = re.sub(pattern, replacement, modified_content)
            if new_content != modified_content:
                changes_made = True
                print(f"  Updated in {file_path}: {pattern} -> {replacement}")
                modified_content = new_content
        else:
            # For patterns without 'src.', we need to be careful not to replace standard library imports
            # Look for lines matching the pattern
            matches = re.finditer(pattern, modified_content, re.MULTILINE)
            
            # Check each match to see if it's a reference to our module
            new_content = modified_content
            for match in matches:
                line = modified_content[match.start():match.end()]
                
                # Check if this is likely referring to our module vs. standard library
                # This is a heuristic and might need refinement
                if any(hint in line for hint in ['TradeLogger', 'LogContext', 'log_config']):
                    replacement = line.replace(old_module, new_module)
                    new_content = new_content.replace(line, replacement)
                    changes_made = True
                    print(f"  Updated in {file_path}: {line} -> {replacement}")
            
            modified_content = new_content
    
    if changes_made:
        try:
            with open(file_path, 'w') as f:
                f.write(modified_content)
            return True
        except (PermissionError, IOError) as e:
            print(f"Error writing to file {file_path}: {e}")
            return False
    
    return False

def main():
    src_dir = os.path.join(os.getcwd(), 'src')
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        print("Make sure you're running this script from the project root")
        return
    
    # Old and new module names
    old_module = 'logging'
    new_module = 'log_system'
    
    # Path to the logging module
    logging_path = os.path.join(src_dir, old_module)
    
    if not os.path.isdir(logging_path):
        print(f"Error: Logging module not found at {logging_path}")
        return
    
    # 1. Rename the directory
    success = rename_directory(logging_path, new_module)
    if not success:
        print("Failed to rename directory. Exiting.")
        return
    
    # 2. Update import references throughout the codebase
    print(f"\nUpdating import references from '{old_module}' to '{new_module}'...")
    update_imports(src_dir, old_module, new_module)
    
    print("\nDone! Remember to update any references to the logging module in your scripts.")

if __name__ == "__main__":
    main()
