import os
import ast

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
            rel_path = os.path.relpath(full_path, root)
            mod_doc, funcs, clss = extract_summary(full_path)
            if mod_doc or funcs or clss:
                summaries.append(f"### `{rel_path}`")
                if mod_doc:
                    summaries.append(f"📘 Module: {mod_doc.splitlines()[0]}")
                if clss:
                    summaries.append("📦 Classes:")
                    summaries.extend([f"- {c}" for c in clss])
                if funcs:
                    summaries.append("🔧 Functions:")
                    summaries.extend([f"- {f}" for f in funcs])
                summaries.append("")  # Spacer
    return "\n".join(summaries)

if __name__ == "__main__":
    out = summarize_codebase(".")
    with open("codebase_summary.md", "w") as f:
        f.write("# Codebase Summary for LLM\n\n")
        f.write(out)
    print("✅ Summary saved to `codebase_summary.md`")

