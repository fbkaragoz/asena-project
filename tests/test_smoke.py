"""Sanity smoke test — verifies pytest harness and core imports work."""
import importlib


def test_python_version_ok():
    import sys
    assert sys.version_info >= (3, 11)


def test_torch_imports():
    torch = importlib.import_module("torch")
    assert hasattr(torch, "__version__")


def test_tokenizers_imports():
    importlib.import_module("tokenizers")


def test_pyarrow_imports():
    importlib.import_module("pyarrow")
