"""Load quantization code directly from the nested upstream repositories."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEANQUANT_ROOT = REPO_ROOT / "LeanQuant"
SQUEEZELLM_ROOT = REPO_ROOT / "SqueezeLLM"
SQUEEZELLM_GRADIENTS_ROOT = REPO_ROOT / "SqueezeLLM-gradients"
SQUEEZELLM_GRADIENTS_REVISION = "5f2a16698b93cddddf858a78bef61fd5c6271055"
SQUEEZELLM_GRADIENTS_SOURCE = (
    f"SqueezeLLM-gradients@{SQUEEZELLM_GRADIENTS_REVISION}"
)


@contextmanager
def _python_path(path: Path):
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def _verify_source(module, root: Path):
    source = Path(module.__file__).resolve()
    if root.resolve() not in source.parents:
        raise ImportError(f"Loaded {module.__name__} from {source}, expected under {root}")


def _load_source_module(module_name: str, source: Path):
    if not source.is_file():
        raise ImportError(
            f"Missing upstream source {source}. Run "
            "'git submodule update --init --recursive'."
        )
    cached = sys.modules.get(module_name)
    if cached is not None:
        _verify_source(cached, source.parent)
        return cached
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _verify_source(module, source.parent)
    return module


def _load_source_function(source: Path, function_name: str):
    """Compile one function verbatim from an upstream file without importing its CLI."""

    if not source.is_file():
        raise ImportError(
            f"Missing upstream source {source}. Run "
            "'git submodule update --init --recursive'."
        )
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function_node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function_node is None:
        raise ImportError(f"Function {function_name!r} is missing from {source}")
    namespace = {}
    function_module = ast.Module(body=[function_node], type_ignores=[])
    exec(compile(function_module, str(source), "exec"), namespace)
    function = namespace[function_name]
    function.__upstream_source__ = str(source)
    return function


def load_leanquant_upstream():
    with _python_path(LEANQUANT_ROOT):
        module = importlib.import_module("lean_quantizer")
    _verify_source(module, LEANQUANT_ROOT)
    return module.LeanQuant, module.Quantizer


def load_squeezellm_kmeans():
    with _python_path(SQUEEZELLM_ROOT):
        module = importlib.import_module("quantization.nuq")
    _verify_source(module, SQUEEZELLM_ROOT)
    return module.kmeans_fit


def load_squeezellm_model_parse():
    with _python_path(SQUEEZELLM_ROOT):
        module = importlib.import_module("squeezellm.model_parse")
    _verify_source(module, SQUEEZELLM_ROOT)
    return module


def load_squeezellm_remove_outliers():
    with _python_path(SQUEEZELLM_ROOT):
        module = importlib.import_module("squeezellm.outliers")
    _verify_source(module, SQUEEZELLM_ROOT)
    return module.remove_outliers


def load_squeezellm_gradients():
    """Load the exact C4 loader and Fisher helpers from SqueezeLLM-gradients."""

    datautils = _load_source_module(
        "_rbvt_squeezellm_gradients_datautils",
        SQUEEZELLM_GRADIENTS_ROOT / "datautils.py",
    )
    run_source = SQUEEZELLM_GRADIENTS_ROOT / "run.py"
    return (
        datautils.get_loaders,
        _load_source_function(run_source, "get_modules"),
        _load_source_function(run_source, "square_grad_hook"),
    )


__all__ = [
    "LEANQUANT_ROOT",
    "SQUEEZELLM_GRADIENTS_ROOT",
    "SQUEEZELLM_GRADIENTS_REVISION",
    "SQUEEZELLM_GRADIENTS_SOURCE",
    "SQUEEZELLM_ROOT",
    "load_leanquant_upstream",
    "load_squeezellm_gradients",
    "load_squeezellm_kmeans",
    "load_squeezellm_model_parse",
    "load_squeezellm_remove_outliers",
]
