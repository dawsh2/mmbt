import os
import ast
import sys

def extract_summary(file_path):
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, FileNotFoundError, UnicodeDecodeError):
        return None, [], []

    module_docstring = ast.get_docstring(tree)
    functions = []
    classes = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or "No docstring"
            functions.append(f"Function `{node.name}`: {doc.splitlines()[0]}")
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or "No docstring"
            classes.append(f"Class `{node.name}`: {doc.splitlines()[0]}")

    return module_docstring, functions, classes

def is_editor_dropping(filename):
    return (
        filename.startswith(".#") or
        filename.startswith("#") or
        filename.endswith("~")
    )

def summarize_codebase(root='.'):
    summaries = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".py") or is_editor_dropping(filename):
                continue
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path_


