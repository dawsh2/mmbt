#!/usr/bin/env python3
"""
Auto-documentation generator for Python modules.

This script generates markdown documentation for Python modules by extracting:
1. Module overview from __init__.py
2. Function signatures, docstrings, and return values from all Python files

The documentation is saved as README.md within each module directory.
"""

import os
import ast
import re
from pathlib import Path


def extract_module_overview(init_path):
    """Extract module overview from __init__.py docstring."""
    if not os.path.exists(init_path):
        return "No module overview available."
    
    try:
        with open(init_path, 'r') as file:
            module_content = file.read()
        
        # Parse the module
        module_ast = ast.parse(module_content)
        
        # Extract the module docstring
        if (module_ast.body and isinstance(module_ast.body[0], ast.Expr) and 
                isinstance(module_ast.body[0].value, ast.Str)):
            return module_ast.body[0].value.s.strip()
        
        return "No module overview available."
    except Exception as e:
        return f"Failed to extract module overview: {str(e)}"


def parse_file(file_path):
    """Parse a Python file and extract functions, classes, methods, and docstrings."""
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Parse the file
        tree = ast.parse(file_content)
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree)
        
        # Functions and classes
        functions = []
        classes = []
        
        for node in tree.body:
            # Handle functions
            if isinstance(node, ast.FunctionDef):
                functions.append(extract_function_info(node, file_content))
            
            # Handle classes
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or "No docstring provided.",
                    'methods': []
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info['methods'].append(extract_function_info(item, file_content))
                
                classes.append(class_info)
        
        return {
            'module_docstring': module_docstring,
            'functions': functions,
            'classes': classes
        }
    except Exception as e:
        return {
            'error': f"Failed to parse file: {str(e)}",
            'module_docstring': None,
            'functions': [],
            'classes': []
        }


def extract_function_info(node, source):
    """Extract information about a function."""
    # Get function signature
    args = []
    
    # Process arguments
    for arg in node.args.args:
        if arg.arg != 'self':
            args.append(arg.arg)
    
    # Process keyword arguments with defaults
    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
    for i, arg in enumerate(node.args.args):
        if arg.arg != 'self' and defaults[i] is not None:
            # Try to get the default value as a string
            if isinstance(defaults[i], ast.Constant):
                default_val = defaults[i].value
                if isinstance(default_val, str):
                    default_val = f"'{default_val}'"
                args[args.index(arg.arg)] = f"{arg.arg}={default_val}"
    
    # Get the function's return annotation if it exists
    returns = None
    if node.returns:
        returns = ast.unparse(node.returns).strip()
    
    # Get function docstring and extract return info from it
    docstring = ast.get_docstring(node) or "No docstring provided."
    return_info = extract_return_info(docstring)
    
    return {
        'name': node.name,
        'signature': f"{node.name}({', '.join(args)})",
        'docstring': docstring,
        'returns': returns,
        'return_info': return_info
    }


def extract_return_info(docstring):
    """Extract return information from a docstring."""
    if not docstring:
        return None
    
    # Look for Returns: section in docstring
    returns_match = re.search(r'Returns[:\n]\s*(.*?)(?:\n\s*\n|\Z)', docstring, re.DOTALL)
    if returns_match:
        return returns_match.group(1).strip()
    
    return None


def is_editor_file(path):
    """Check if a file is likely an editor temporary file."""
    patterns = [
        '__pycache__',
        '.pyc',
        '.pyo',
        '.swp',
        '.swo', 
        '.DS_Store',
        '~',      # backup files ending with tilde
        '#'       # emacs backup files often contain hash
    ]
    path_str = str(path)
    return any(pattern in path_str for pattern in patterns)


def generate_markdown():
    """Generate markdown documentation for all modules in the hardcoded source directory."""
    # Hardcoded source directory
    src_path = Path(os.path.expanduser("~/mmbt/src/"))
    
    if not src_path.exists():
        print(f"Error: Source directory {src_path} does not exist.")
        return
    
    print(f"Generating documentation for modules in {src_path}...")
    
    # Find all modules (directories with __init__.py)
    modules = [d for d in src_path.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    
    # Generate documentation for each module
    for module in modules:
        module_name = module.name
        
        # Skip hidden directories
        if module_name.startswith('.'):
            continue
        
        print(f"Generating documentation for {module_name}...")
        
        # Extract module overview
        init_path = module / "__init__.py"
        module_overview = extract_module_overview(init_path)
        
        # Get all Python files in the module, excluding editor files
        python_files = sorted([
            f for f in module.glob('**/*.py') 
            if f.name != '__init__.py' and not is_editor_file(f)
        ])
        
        # Generate markdown content
        markdown_content = [
            f"# {module_name.capitalize()} Module",
            "",
            module_overview,
            "",
            "## Contents",
            ""
        ]
        
        # Add table of contents
        for py_file in python_files:
            relative_path = py_file.relative_to(module)
            file_name = relative_path.stem
            markdown_content.append(f"- [{file_name}](#{file_name.lower()})")
        
        markdown_content.append("")
        
        # Process each Python file
        for py_file in python_files:
            relative_path = py_file.relative_to(module)
            file_name = relative_path.stem
            
            markdown_content.extend([
                f"## {file_name}",
                ""
            ])
            
            # Parse the file
            file_info = parse_file(py_file)
            
            # Add module docstring if available
            if file_info.get('module_docstring'):
                markdown_content.extend([
                    file_info['module_docstring'],
                    ""
                ])
            
            # Add functions
            if file_info['functions']:
                markdown_content.append("### Functions")
                markdown_content.append("")
                
                for func in file_info['functions']:
                    markdown_content.extend([
                        f"#### `{func['signature']}`",
                        ""
                    ])
                    
                    if func['returns']:
                        markdown_content.extend([
                            f"*Returns:* `{func['returns']}`",
                            ""
                        ])
                    
                    markdown_content.extend([
                        func['docstring'],
                        ""
                    ])
                    
                    if func['return_info'] and not func['returns']:
                        markdown_content.extend([
                            f"*Returns:* {func['return_info']}",
                            ""
                        ])
            
            # Add classes
            if file_info['classes']:
                markdown_content.append("### Classes")
                markdown_content.append("")
                
                for class_info in file_info['classes']:
                    markdown_content.extend([
                        f"#### `{class_info['name']}`",
                        "",
                        class_info['docstring'],
                        ""
                    ])
                    
                    if class_info['methods']:
                        markdown_content.append("##### Methods")
                        markdown_content.append("")
                        
                        for method in class_info['methods']:
                            markdown_content.extend([
                                f"###### `{method['signature']}`",
                                ""
                            ])
                            
                            if method['returns']:
                                markdown_content.extend([
                                    f"*Returns:* `{method['returns']}`",
                                    ""
                                ])
                            
                            markdown_content.extend([
                                method['docstring'],
                                ""
                            ])
                            
                            if method['return_info'] and not method['returns']:
                                markdown_content.extend([
                                    f"*Returns:* {method['return_info']}",
                                    ""
                                ])
        
        # Write the markdown file inside the module directory
        output_file = module / f"README.md"
        with open(output_file, 'w') as f:
            f.write('\n'.join(markdown_content))
        
        print(f"Documentation generated: {output_file}")


if __name__ == "__main__":
    generate_markdown()
    print("Documentation generation complete.")
