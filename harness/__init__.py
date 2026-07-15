"""Harness package: model serving, loading, and test execution.

Public surface:
    * ``vLLMServerManager`` -- start/stop/health-check local vLLM servers.
    * ``ModelLoader``       -- load models under evaluation (Ollama / vLLM / API).
    * ``TestRunner``        -- run a test suite against models, collect outputs.
"""

from harness.vllm_server_manager import vLLMServerManager
from harness.model_loader import ModelLoader
from harness.test_runner import TestRunner

__all__ = ["vLLMServerManager", "ModelLoader", "TestRunner"]
