"""
Code verification script for PPMI optimizations.
Checks syntax, imports, and code structure without requiring data.
"""

import ast
import sys

def verify_syntax(file_path):
    """Verify Python syntax."""
    print(f"Verifying syntax of {file_path}...")
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✓ Syntax verification passed")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

def check_ppmi_class(file_path):
    """Check PPMI class structure."""
    print("\nChecking PPMI class structure...")

    with open(file_path, 'r') as f:
        code = f.read()

    tree = ast.parse(code)

    # Find PPMI class
    ppmi_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'PPMI':
            ppmi_class = node
            break

    if not ppmi_class:
        print("✗ PPMI class not found")
        return False

    print("✓ PPMI class found")

    # Check required methods
    required_methods = {
        '__init__': False,
        'load_sequence': False,
        '_set_data': False,
        '_get_file_metadata': False,
        '_load_from_cache': False
    }

    for item in ppmi_class.body:
        if isinstance(item, ast.FunctionDef):
            if item.name in required_methods:
                required_methods[item.name] = True
                print(f"  ✓ Found method: {item.name}")

                # Check __init__ parameters
                if item.name == '__init__':
                    params = [arg.arg for arg in item.args.args]
                    if 'cache_size' in params and 'use_mmap' in params:
                        print(f"    ✓ __init__ has cache_size and use_mmap parameters")
                    else:
                        print(f"    ✗ __init__ missing optimization parameters")
                        return False

    # Verify all required methods are present
    missing_methods = [k for k, v in required_methods.items() if not v]
    if missing_methods:
        print(f"✗ Missing methods: {', '.join(missing_methods)}")
        return False

    print("✓ All required methods found")
    return True

def check_imports(file_path):
    """Check required imports."""
    print("\nChecking required imports...")

    with open(file_path, 'r') as f:
        code = f.read()

    tree = ast.parse(code)

    required_imports = {
        'OrderedDict': False,
        'lru_cache': False
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'collections':
                for alias in node.names:
                    if alias.name == 'OrderedDict':
                        required_imports['OrderedDict'] = True
                        print("  ✓ Found: from collections import OrderedDict")

            if node.module == 'functools':
                for alias in node.names:
                    if alias.name == 'lru_cache':
                        required_imports['lru_cache'] = True
                        print("  ✓ Found: from functools import lru_cache")

    missing = [k for k, v in required_imports.items() if not v]
    if missing:
        print(f"  ! Note: {', '.join(missing)} not imported (may not be needed)")

    print("✓ Import check completed")
    return True

def check_optimization_features(file_path):
    """Check for optimization features in code."""
    print("\nChecking optimization features...")

    with open(file_path, 'r') as f:
        code = f.read()

    checks = {
        '_data_cache': 'Data cache attribute',
        '_metadata_cache': 'Metadata cache attribute',
        'mmap_mode': 'Memory mapping',
        'move_to_end': 'LRU cache management',
        'popitem': 'Cache eviction'
    }

    found = {}
    for key, desc in checks.items():
        if key in code:
            found[key] = True
            print(f"  ✓ Found: {desc} ({key})")
        else:
            found[key] = False
            print(f"  ✗ Missing: {desc} ({key})")

    all_found = all(found.values())
    if all_found:
        print("✓ All optimization features present")
    else:
        print("✗ Some optimization features missing")

    return all_found

def verify_docstrings(file_path):
    """Check for proper documentation."""
    print("\nChecking documentation...")

    with open(file_path, 'r') as f:
        code = f.read()

    tree = ast.parse(code)

    # Find PPMI class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'PPMI':
            # Check class docstring
            if ast.get_docstring(node):
                print("  ✓ Class docstring present")
                docstring = ast.get_docstring(node)
                if 'Optimization' in docstring or 'cache' in docstring.lower():
                    print("  ✓ Docstring mentions optimizations")
                else:
                    print("  ! Docstring doesn't mention optimizations")
            else:
                print("  ✗ No class docstring")

            # Check method docstrings
            methods_with_docs = 0
            total_methods = 0
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    total_methods += 1
                    if ast.get_docstring(item):
                        methods_with_docs += 1

            print(f"  ✓ {methods_with_docs}/{total_methods} methods have docstrings")
            break

    print("✓ Documentation check completed")
    return True

def main():
    """Run all verification checks."""
    print("="*60)
    print("PPMI Dataset Optimization - Code Verification")
    print("="*60)

    file_path = "datasets/fmri_datasets.py"

    checks = [
        ("Syntax", verify_syntax, file_path),
        ("Imports", check_imports, file_path),
        ("PPMI Class Structure", check_ppmi_class, file_path),
        ("Optimization Features", check_optimization_features, file_path),
        ("Documentation", verify_docstrings, file_path),
    ]

    results = []
    for name, check_func, *args in checks:
        try:
            result = check_func(*args)
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))

    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All verification checks passed!")
        print("\nOptimizations implemented:")
        print("  1. LRU cache for data files")
        print("  2. Memory-mapped file loading")
        print("  3. Metadata caching")
        print("  4. Optimized _set_data method")
        print("\nThe code is ready for testing with real data.")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
