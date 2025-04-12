#!/usr/bin/env python3
"""
Attempts to fix the duplicated __init__ methods in Rule3 through Rule15
in strategy.py. This script also tries to make a best-guess correction
to the history attribute access based on common patterns.

WARNING: This script makes assumptions and requires careful review
of the modified strategy.py file. It might not be correct for all rules.
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the file before modifying it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.auto_fix_backup.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def fix_rule_init_and_history(file_content):
    """Fixes duplicated __init__ and attempts history access correction."""
    rules_to_fix = range(3, 16)
    updated_content = file_content

    for rule_num in rules_to_fix:
        rule_name = f"Rule{rule_num}"
        class_pattern = re.compile(
            r'class\s+' + re.escape(rule_name) + r'\s*:\n(\s*)"""[\s\S]*?"""\n(\s*)def __init__\(self, params\):\n(\s*).*?\n(\s*)"""[\s\S]*?"""\n(\s*)def __init__\(self, params\):\n(\s*pass)',
            re.MULTILINE | re.DOTALL
        )
        match = class_pattern.search(updated_content)
        if match:
            indent1 = match.group(3)
            init_content = match.group(4)
            full_match = match.group(0)
            replacement = f'class {rule_name}:\n{match.group(2)}"""[\s\S]*?"""\n{match.group(2)}def __init__(self, params):\n{indent1}{init_content.strip()}\n'
            updated_content = updated_content.replace(full_match, replacement)
            print(f"Fixed duplicated __init__ in {rule_name}")
        else:
            class_pattern_simple = re.compile(
                r'class\s+' + re.escape(rule_name) + r'\s*:\n(\s*)def __init__\(self, params\):\n(\s*pass)\n(\s*)"""',
                re.MULTILINE | re.DOTALL
            )
            match_simple = class_pattern_simple.search(updated_content)
            if match_simple:
                full_match_simple = match_simple.group(0)
                replacement_simple = f'class {rule_name}:\n{match_simple.group(1)}def __init__(self, params):\n{match_simple.group(2)}\n{match_simple.group(3)}"""'
                updated_content = updated_content.replace(full_match_simple, replacement_simple)
                print(f"Cleaned simple pass __init__ in {rule_name}")

        # Attempt to fix history access (assuming it should be a direct append)
        history_access_pattern = re.compile(
            r'self\.history\[\'(.*?)\'\]\.append\(',
            re.MULTILINE
        )
        updated_content = history_access_pattern.sub(r'self.history.append(', updated_content)
        print(f"Attempted to fix history access in {rule_name}")

        # Attempt to initialize missing history-like attributes based on error messages
        attribute_map = {
            'Rule3': ['history'],
            'Rule4': ['history'],
            'Rule5': ['history'],
            'Rule6': ['history'],
            'Rule7': ['high_history', 'low_history', 'stoch_history', 'stoch_ma_history'],
            'Rule8': ['history'],
            'Rule9': ['high_history', 'low_history'],
            'Rule10': ['history'],
            'Rule11': ['history'],
            'Rule12': ['history', 'high_history', 'low_history'],
            'Rule13': ['high_history', 'low_history', 'stoch_history', 'cci_history'],
            'Rule14': ['high_history', 'low_history', 'close_history', 'atr_history'],
            'Rule15': ['close_history', 'bb_high_history', 'bb_mid_history', 'bb_low_history'], # More explicit names
        }

        class_def_pattern = re.compile(r'class\s+' + re.escape(rule_name) + r'\s*:\n', re.MULTILINE)
        class_match = class_def_pattern.search(updated_content)

        if rule_name in attribute_map and class_match:
            insertion_point = class_match.end()
            init_insertion = ""
            init_pattern = re.compile(r'def __init__\(self, params\):', re.MULTILINE)
            init_match = init_pattern.search(updated_content[insertion_point:])

            if init_match:
                init_start = insertion_point + init_match.end() + 1
                indentation_match = re.search(r'\n(\s+)', updated_content[init_start:])
                indentation = indentation_match.group(1) if indentation_match else "    "

                for attr in attribute_map[rule_name]:
                    if not re.search(rf'self\.{attr}\s*=', updated_content[init_start:], re.MULTILINE):
                        if 'history' in attr:
                            init_insertion += f"{indentation}self.{attr} = deque(maxlen=200)\n"
                        else:
                            init_insertion += f"{indentation}self.{attr} = deque(maxlen=200)\n" # Default to deque
                if init_insertion:
                    updated_content = updated_content[:init_start] + init_insertion + updated_content[init_start:]
                    print(f"Attempted to add missing attribute initializations in {rule_name}")


    return updated_content

def main():
    """Main function to run the script."""
    strategy_py = "strategy.py"

    if not os.path.exists(strategy_py):
        print(f"Error: {strategy_py} not found")
        return False

    backup_file(strategy_py)

    with open(strategy_py, 'r') as f:
        content = f.read()

    updated_content = fix_rule_init_and_history(content)

    with open(strategy_py, 'w') as f:
        f.write(updated_content)

    print("\nAttempted to fix duplicated __init__ methods and history access in Rule3 through Rule15.")
    print("Please carefully review the changes in strategy.py and adjust the __init__")
    print("methods and on_bar methods to correctly implement the logic for each rule.")
    print("A backup of the original file was created.")
    return True

if __name__ == "__main__":
    main()
