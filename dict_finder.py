#!/usr/bin/env python3
"""
Standardization Finder Script

This script searches your codebase for patterns that need to be updated
as part of the signal and BarEvent standardization effort.
"""

import os
import re
import argparse
from typing import List, Dict, Any, Tuple

# Patterns to search for
PATTERNS = {
    "generate_signals_method": r"def\s+generate_signals\s*\(",
    "dict_bar_data": r"bar_data\s*:\s*Dict\s*\[\s*str\s*,\s*Any\s*\]",
    "dict_access": r"(?:bar_data|data)\s*\[\s*['\"](?:Open|High|Low|Close|Volume|symbol|timestamp)['\"]",
    "create_signal_helper": r"create_signal\s*\(",
    "direct_signal_creation": r"Signal\s*\(",
    "direct_bar_event_creation": r"BarEvent\s*\(",
    "dict_to_barevent": r"BarEvent\s*\(\s*(?:bar_data|data)\s*\)",
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Find code that needs standardization")
    parser.add_argument("--src", default="src", help="Source directory to search")
    parser.add_argument("--ext", default=".py", help="File extension to search")
    return parser.parse_args()

def find_matches(file_path: str, patterns: Dict[str, str]) -> Dict[str, List[Tuple[int, str]]]:
    """Find all pattern matches in a file."""
    matches = {name: [] for name in patterns}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                for name, pattern in patterns.items():
                    if re.search(pattern, line):
                        matches[name].append((i, line.strip()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return matches

def search_directory(directory: str, extension: str, patterns: Dict[str, str]) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
    """Search directory recursively for files matching patterns."""
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                matches = find_matches(file_path, patterns)
                
                # Only add files with matches
                if any(matches.values()):
                    results[rel_path] = matches
    
    return results

def print_results(results: Dict[str, Dict[str, List[Tuple[int, str]]]]) -> None:
    """Print search results in a readable format."""
    print("\n==== Code Standardization Finder Results ====\n")
    
    # Summary counts
    pattern_counts = {name: 0 for name in PATTERNS}
    file_count = 0
    
    for file_path, matches in results.items():
        has_matches = False
        for pattern_name, occurrences in matches.items():
            if occurrences:
                has_matches = True
                pattern_counts[pattern_name] += len(occurrences)
        
        if has_matches:
            file_count += 1
    
    print("Summary:")
    print(f"  Found issues in {file_count} files")
    for pattern_name, count in pattern_counts.items():
        print(f"  {pattern_name}: {count} occurrences")
    
    print("\nDetailed Results:")
    for file_path, matches in sorted(results.items()):
        has_matches = any(matches.values())
        if not has_matches:
            continue
            
        print(f"\n== {file_path} ==")
        
        for pattern_name, occurrences in matches.items():
            if not occurrences:
                continue
                
            print(f"  {pattern_name}:")
            for line_num, line_text in occurrences:
                print(f"    Line {line_num}: {line_text}")

def generate_todo_list(results: Dict[str, Dict[str, List[Tuple[int, str]]]]) -> None:
    """Generate a TODO list of files to update."""
    print("\n==== Standardization TODO List ====\n")
    
    # Group files by type of issue
    files_with_generate_signals = []
    files_with_dict_usage = []
    files_with_signal_helpers = []
    
    for file_path, matches in results.items():
        if matches["generate_signals_method"]:
            files_with_generate_signals.append(file_path)
        
        if (matches["dict_bar_data"] or matches["dict_access"] or 
            matches["dict_to_barevent"]):
            files_with_dict_usage.append(file_path)
            
        if matches["create_signal_helper"] or matches["direct_signal_creation"]:
            files_with_signal_helpers.append(file_path)
    
    # Method renaming
    if files_with_generate_signals:
        print("1. Rename 'generate_signals' to 'process_signals' in these files:")
        for file in sorted(files_with_generate_signals):
            print(f"   - {file}")
    
    # Dictionary usage
    if files_with_dict_usage:
        print("\n2. Replace dictionary usage with BarEvent in these files:")
        for file in sorted(files_with_dict_usage):
            print(f"   - {file}")
    
    # Signal creation helpers
    if files_with_signal_helpers:
        print("\n3. Replace signal helper methods with direct SignalEvent creation:")
        for file in sorted(files_with_signal_helpers):
            print(f"   - {file}")

def main():
    """Main function."""
    args = parse_args()
    results = search_directory(args.src, args.ext, PATTERNS)
    print_results(results)
    generate_todo_list(results)

if __name__ == "__main__":
    main()
