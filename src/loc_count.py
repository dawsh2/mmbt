import os
from collections import defaultdict

def count_lines_by_directory(root_dir='.'):
    line_counts = defaultdict(int)

    for dirpath, _, filenames in os.walk(root_dir):
        py_files = [f for f in filenames if f.endswith('.py')]
        for filename in py_files:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    relative_dir = os.path.relpath(dirpath, root_dir)
                    line_counts[relative_dir] += len(lines)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")

    # Sort by number of lines, descending
    sorted_counts = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Directory':<40} | {'Lines of Code'}")
    print("-" * 60)
    for directory, lines in sorted_counts:
        print(f"{directory:<40} | {lines}")
    total = sum(line_counts.values())
    print("-" * 60)
    print(f"{'Total':<40} | {total}")

if __name__ == "__main__":
    count_lines_by_directory('.')
