#!/usr/bin/env python3
"""
Attempts to automatically fix the "no attribute" errors encountered so far in
strategy.py. This includes:

- ema1_value in Rule3
- close_history in Rule7, Rule9, Rule13
- gain_history in Rule10, Rule12
- bb_period in Rule15
- _calculate_tr in Rule14

It also adds initializations for attributes that seem to be missing based on the
provided code for Rules 0-9, and attempts basic 'NoneType' checks in Rules 4, 5, and 6.

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
    backup_path = f"{file_path}.{timestamp}.auto_fix_attr_v3_backup.bak"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def fix_missing_attributes(file_content):
    """Adds initializations for missing attributes in specified rule classes."""
    attributes_to_add = {
        'Rule3': {'ema1_value': 'np.nan', 'ema1_history': 'deque()', 'ema2_value': 'np.nan', 'ema2_history': 'deque()', 'alpha_ema2': '2 / (params.get(\'ema2_period\', 40) + 1) if params.get(\'ema2_period\', 40) > 0 else 0', 'ema1_period': 'params.get(\'ema1_period\', 20)', 'ema2_period': 'params.get(\'ema2_period\', 40)'},
        'Rule4': {'ema1_value': 'np.nan', 'ema2_value': 'np.nan', 'dema1_value': 'np.nan', 'ma2_sum': '0', 'ma2_count': '0', 'dema1_period': 'params.get(\'dema1_period\', 20)', 'ma2_period': 'params.get(\'ma2_period\', 50)', 'alpha1': '2 / (params.get(\'dema1_period\', 20) + 1) if params.get(\'dema1_period\', 20) > 0 else 0'},
        'Rule5': {'ema1_value': 'np.nan', 'ema1_history': 'deque()', 'ema2_value': 'np.nan', 'dema1_value': 'np.nan', 'dema1_history': 'deque()', 'ema3_value': 'np.nan', 'ema3_history': 'deque()', 'ema4_value': 'np.nan', 'dema2_value': 'np.nan', 'dema2_history': 'deque()', 'dema1_period': 'params.get(\'dema1_period\', 15)', 'dema2_period': 'params.get(\'dema2_period\', 30)', 'alpha1': '2 / (params.get(\'dema1_period\', 15) + 1) if params.get(\'dema1_period\', 15) > 0 else 0', 'alpha2': '2 / (params.get(\'dema2_period\', 30) + 1) if params.get(\'dema2_period\', 30) > 0 else 0'},
        'Rule6': {'ema1_value': 'np.nan', 'ema2_value': 'np.nan', 'ema3_value': 'np.nan', 'tema1_value': 'np.nan', 'ma2_sum': '0', 'ma2_count': '0', 'tema1_period': 'params.get(\'tema1_period\', 10)', 'ma2_period': 'params.get(\'ma2_period\', 30)', 'alpha1': '2 / (params.get(\'tema1_period\', 10) + 1) if params.get(\'tema1_period\', 10) > 0 else 0', 'close_history': 'deque(maxlen=200)'},
        'Rule7': {'close_history': 'deque(maxlen=200)', 'stoch1_period': 'params.get(\'stoch1_period\', 14)', 's1_history': 'deque()', 's2_sum': '0', 's2_count': '0', 's2_value': 'np.nan', 'stochma2_period': 'params.get(\'stochma2_period\', 3)'},
        'Rule9': {'close_history': 'deque(maxlen=200)', 'ichimoku_tenkan_period': 'params.get(\'ichimoku_tenkan_period\', 9)', 'ichimoku_kijun_period': 'params.get(\'ichimoku_kijun_period\', 26)', 'ichimoku_senkou_b_period': 'params.get(\'ichimoku_senkou_b_period\', 52)'},
        'Rule10': {'gain_history': 'deque(maxlen=200)'},
        'Rule12': {'gain_history': 'deque(maxlen=200)'},
        'Rule13': {'close_history': 'deque(maxlen=200)'},
        'Rule15': {'bb_period': 'params.get(\'bb_period\', 20)'},
        'Rule14': {'_calculate_tr': 'self._calculate_true_range  # Assuming it\'s a method', 'atr_period': 'params.get(\'atr_period\', 14)', 'atr_multiplier': 'params.get(\'atr_multiplier\', 3)', 'atr_value': 'np.nan', 'trailing_stop': 'np.nan', 'position': '0', 'trend_direction': '0'},
    }

    updated_content = file_content
    for rule_name, attrs in attributes_to_add.items():
        class_pattern = re.compile(r'class\s+' + re.escape(rule_name) + r'\s*:\n', re.MULTILINE)
        class_match = class_pattern.search(updated_content)

        if class_match:
            insertion_point = class_match.end()
            init_pattern = re.compile(r'def __init__\(self, params\):', re.MULTILINE)
            init_match = init_pattern.search(updated_content[insertion_point:])

            if init_match:
                init_start = insertion_point + init_match.end() + 1
                indentation_match = re.search(r'\n(\s+)', updated_content[init_start:])
                indentation = indentation_match.group(1) if indentation_match else "    "

                for attr_name, default_value in attrs.items():
                    if not re.search(rf'self\.{attr_name}\s*=', updated_content[init_start:], re.MULTILINE):
                        updated_content = updated_content[:init_start] + f"{indentation}self.{attr_name} = {default_value}\n" + updated_content[init_start:]
                        print(f"Added initialization for '{attr_name}' in {rule_name}")
            else:
                print(f"Warning: Could not find __init__ method in {rule_name}")
        else:
            print(f"Warning: Could not find class definition for {rule_name}")

    # Add basic NoneType checks (manual review of logic is crucial)
    rules_to_check_none = ['Rule4', 'Rule5', 'Rule6']
    for rule_name in rules_to_check_none:
        class_pattern = re.compile(r'class\s+' + re.escape(rule_name) + r'\s*:\n', re.MULTILINE)
        class_match = class_pattern.search(updated_content)
        if class_match:
            on_bar_pattern = re.compile(r'def on_bar\(self, bar\):\n(.*?)(?:def|\Z)', re.MULTILINE | re.DOTALL)
            on_bar_match = on_bar_pattern.search(updated_content[class_match.end():])
            if on_bar_match:
                on_bar_code = on_bar_match.group(1)
                lines = on_bar_code.splitlines()
                updated_on_bar_lines = []
                for line in lines:
                    comparison_match = re.search(r'(if|elif)\s+(.+?)\s*(>|<|>=|<=|==|!=)\s*(\d+)', line)
                    if comparison_match:
                        condition = comparison_match.group(2).strip()
                        if "self." in condition:
                            updated_line = re.sub(r'(self\.\w+)', r'(\1 is not None and \1)', line)
                            updated_on_bar_lines.append(updated_line)
                            print(f"Attempted to add None check in {rule_name}: {line.strip()} -> {updated_line.strip()}")
                        else:
                            updated_on_bar_lines.append(line)
                    else:
                        updated_on_bar_lines.append(line)
                updated_content = updated_content[:class_match.end() + on_bar_match.start()] + "\n".join(updated_on_bar_lines) + updated_content[class_match.end() + on_bar_match.end():]
            else:
                print(f"Warning: Could not find 'on_bar' method in {rule_name}")
        else:
            print(f"Warning: Could not find class definition for {rule_name}")

    # Add _calculate_true_range method to Rule14 if not present
    rule14_pattern = re.compile(r'class\s+Rule14\s*:\n', re.MULTILINE)
    rule14_match = rule14_pattern.search(updated_content)
    if rule14_match:
        method_pattern = re.compile(r'def\s+_calculate_true_range\(self,', re.MULTILINE)
        method_match = method_pattern.search(updated_content[rule14_match.end():])
        if not method_match:
            init_pattern = re.compile(r'def __init__\(self, params\):', re.MULTILINE)
            init_match = init_pattern.search(updated_content[rule14_match.end():])
            if init_match:
                insertion_point = rule14_match.end() + init_match.end() + 1
                indentation_match = re.search(r'\n(\s+)', updated_content[insertion_point:])
                indentation = indentation_match.group(1) if indentation_match else "    "
                true_range_method = f"""
{indentation}def _calculate_true_range(self, current_high, current_low, previous_close):
{indentation}    return max(current_high - current_low,
{indentation}               abs(current_high - previous_close),
{indentation}               abs(current_low - previous_close))
"""
                updated_content = updated_content[:insertion_point] + true_range_method + updated_content[insertion_point:]
                print("Added _calculate_true_range method to Rule14")
            else:
                print("Warning: Could not find __init__ method in Rule14 to add _calculate_true_range")
        else:
            print("_calculate_true_range method already exists in Rule14")

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

    updated_content = fix_missing_attributes(content)

    with open(strategy_py, 'w') as f:
        f.write(updated_content)

    print("\nAttempted to automatically add initializations for missing attributes.")
    print("Also attempted to add basic 'None' checks in Rules 4, 5, and 6.")
    print("And attempted to ensure _calculate_true_range exists in Rule14.")
    print("Please carefully review the changes in strategy.py and ensure they are correct.")
    print("MANUAL REVIEW IS ESPECIALLY NEEDED FOR RULES 4, 5, AND 6 TO ENSURE THE LOGIC IS CORRECTLY HANDLED.")
    print("A backup of the original file was created.")
    return True

if __name__ == "__main__":
    main()
