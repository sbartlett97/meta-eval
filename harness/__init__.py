"""Harness package: model loading, in-process serving, and test execution.

Public surface:
    * ``ModelLoader``   -- load models under evaluation (llama.cpp / Ollama / API).
    * ``TestRunner``    -- run a test suite against models, collect outputs.
    * ``get_engine``    -- process-wide cache of in-process llama.cpp GGUF engines.
    * ``hydrate_weights`` -- download open weights from HuggingFace at startup.
"""

from harness.hydrate import hydrate_weights
from harness.llamacpp_engine import get_engine
from harness.model_loader import ModelLoader
from harness.test_runner import TestRunner

__all__ = ["ModelLoader", "TestRunner", "get_engine", "hydrate_weights"]
