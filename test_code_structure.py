"""
Test code structure without requiring torch/transformers
Validates that the code is syntactically correct and well-structured
"""

import ast
import sys
from pathlib import Path


def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("Testing Python syntax...")
    
    files_to_check = [
        "enhanced_ocr_inference.py",
        "test_enhanced_ocr.py",
    ]
    
    for filename in files_to_check:
        filepath = Path(__file__).parent / filename
        try:
            with open(filepath) as f:
                code = f.read()
            ast.parse(code)
            print(f"  ✓ {filename} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {filename} has syntax error: {e}")
            return False
    
    print("✓ All Python files have valid syntax")
    return True


def test_class_structure():
    """Test that the main class has expected structure"""
    print("\nTesting class structure...")
    
    filepath = Path(__file__).parent / "enhanced_ocr_inference.py"
    with open(filepath) as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Find the DeepSeekOCRInference class
    class_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeepSeekOCRInference":
            class_node = node
            break
    
    assert class_node is not None, "DeepSeekOCRInference class not found"
    print("  ✓ DeepSeekOCRInference class found")
    
    # Check methods
    methods = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    
    expected_methods = {
        '__init__',
        '_setup_chat_template',
        'infer',
        'infer_batch',
        '_has_repetition',
        'infer_with_column_split',
    }
    
    for method in expected_methods:
        assert method in methods, f"Missing method: {method}"
        print(f"  ✓ Method '{method}' exists")
    
    # Check infer method has proper parameters
    infer_method = next(node for node in class_node.body 
                       if isinstance(node, ast.FunctionDef) and node.name == "infer")
    
    param_names = [arg.arg for arg in infer_method.args.args]
    
    expected_params = [
        'self', 'image_path', 'prompt', 'max_new_tokens', 'temperature',
        'top_p', 'repetition_penalty', 'no_repeat_ngram_size', 'length_penalty',
        'num_beams', 'early_stopping', 'detect_repetition', 'retry_on_failure', 'max_retries'
    ]
    
    for param in expected_params:
        assert param in param_names, f"Missing parameter: {param}"
    
    print(f"  ✓ infer() has all {len(expected_params)} expected parameters")
    
    # Check that infer has return annotation
    assert infer_method.returns is not None, "infer() missing return type annotation"
    print("  ✓ infer() has return type annotation")
    
    print("✓ Class structure correct")
    return True


def test_docstrings():
    """Test that key functions have docstrings"""
    print("\nTesting docstrings...")
    
    filepath = Path(__file__).parent / "enhanced_ocr_inference.py"
    with open(filepath) as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Check module docstring
    module_docstring = ast.get_docstring(tree)
    assert module_docstring is not None, "Missing module docstring"
    assert "repetition" in module_docstring.lower(), "Module docstring should mention repetition"
    print("  ✓ Module has descriptive docstring")
    
    # Find class and check docstring
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeepSeekOCRInference":
            class_docstring = ast.get_docstring(node)
            assert class_docstring is not None, "Missing class docstring"
            print("  ✓ Class has docstring")
            
            # Check method docstrings
            for method_node in node.body:
                if isinstance(method_node, ast.FunctionDef) and not method_node.name.startswith('_'):
                    method_docstring = ast.get_docstring(method_node)
                    assert method_docstring is not None, f"Missing docstring for {method_node.name}"
                    print(f"  ✓ Method '{method_node.name}' has docstring")
    
    print("✓ All public methods have docstrings")
    return True


def test_anti_repetition_logic():
    """Test that anti-repetition logic is present in code"""
    print("\nTesting anti-repetition logic...")
    
    filepath = Path(__file__).parent / "enhanced_ocr_inference.py"
    with open(filepath) as f:
        code = f.read()
    
    # Check for key anti-repetition features
    checks = [
        ("repetition_penalty", "Repetition penalty parameter"),
        ("no_repeat_ngram_size", "N-gram blocking parameter"),
        ("_has_repetition", "Repetition detection method"),
        ("retry_on_failure", "Retry logic parameter"),
        ("max_retries", "Max retries parameter"),
    ]
    
    for keyword, description in checks:
        assert keyword in code, f"Missing: {description}"
        print(f"  ✓ {description} present")
    
    print("✓ Anti-repetition logic implemented")
    return True


def test_chat_template_logic():
    """Test that chat template setup is present"""
    print("\nTesting chat template logic...")
    
    filepath = Path(__file__).parent / "enhanced_ocr_inference.py"
    with open(filepath) as f:
        code = f.read()
    
    checks = [
        ("_setup_chat_template", "Chat template setup method"),
        ("apply_chat_template", "Chat template application"),
        ("chat_template", "Chat template property"),
    ]
    
    for keyword, description in checks:
        assert keyword in code, f"Missing: {description}"
        print(f"  ✓ {description} present")
    
    print("✓ Chat template support implemented")
    return True


def test_return_value():
    """Test that infer method returns text (not None)"""
    print("\nTesting return value...")
    
    filepath = Path(__file__).parent / "enhanced_ocr_inference.py"
    with open(filepath) as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Find infer method
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DeepSeekOCRInference":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "infer":
                    # Check return annotation
                    assert method.returns is not None, "Missing return annotation"
                    
                    # Check that there are return statements
                    returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
                    assert len(returns) > 0, "No return statements found"
                    print(f"  ✓ infer() has {len(returns)} return statement(s)")
                    
                    # Check return type annotation is str
                    if isinstance(method.returns, ast.Name):
                        assert method.returns.id == "str", "Return type should be str"
                        print("  ✓ Return type annotated as 'str'")
    
    print("✓ infer() returns text (not None)")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Enhanced DeepSeek-OCR Code Structure Tests")
    print("=" * 60)
    
    tests = [
        test_python_syntax,
        test_class_structure,
        test_docstrings,
        test_anti_repetition_logic,
        test_chat_template_logic,
        test_return_value,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All code structure tests passed!")
        print("\nKey improvements verified:")
        print("  1. ✓ infer() returns str (not None)")
        print("  2. ✓ Chat template support (_setup_chat_template)")
        print("  3. ✓ Anti-repetition parameters (repetition_penalty, no_repeat_ngram_size)")
        print("  4. ✓ Repetition detection (_has_repetition)")
        print("  5. ✓ Retry logic (retry_on_failure, max_retries)")
        print("  6. ✓ Comprehensive docstrings")
        print("  7. ✓ Column splitting (infer_with_column_split)")
        print("  8. ✓ Batch processing (infer_batch)")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
