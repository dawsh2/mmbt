import os

def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except Exception as e:
        print(f"Could not read {filepath}: {e}")
        return 0

def walk_directory_and_count_lines(directory, file_extensions=None):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for name in files:
            if file_extensions:
                if not any(name.endswith(ext) for ext in file_extensions):
                    continue
            full_path = os.path.join(root, name)
            lines = count_lines_in_file(full_path)
            total_lines += lines
            print(f"{full_path}: {lines} lines")
    print(f"\nTotal lines: {total_lines}")

# Usage
if __name__ == "__main__":
    target_directory = "."  # Change this to your target directory
    file_types = ['.py']    # Filter by extension, or set to None for all files
    walk_directory_and_count_lines(target_directory, file_types)
