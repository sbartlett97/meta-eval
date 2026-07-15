"""vLLM Server Manager (PRD v3.1).

Responsibilities
----------------
1. Resolve pre-quantized checkpoints from HuggingFace (e.g.
   ``unsloth/Mistral-7B-v0.3-GGUF``). vLLM downloads/caches on first launch.
2. Start a vLLM OpenAI-compatible server per local model (Mistral :8000,
   Llama :8001, ...).
3. Health-check each server by polling ``/health`` until ready.
4. Manage ports and process handles; clean up on error.
5. Graceful shutdown (SIGTERM, then SIGKILL after a grace period).

Status: AI-scaffolded -- SAM MUST REVIEW before trusting on real hardware.
Review checklist lives in the PRD ("Sam Must Review"). In particular verify the
vLLM CLI flags against the installed vLLM version + Apple Metal, and confirm the
memory budget (2x 7B + judges) fits in 32GB.

This module shells out to the ``vllm serve`` CLI via ``subprocess`` rather than
importing vLLM in-process, so a crash in one server cannot take down the harness
and each model gets an isolated process (PRD "Why Separate vLLM Servers?").
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)


# Default launch parameters. Overridden by config/hardware_profile.yaml when a
# path is supplied to vLLMServerManager.
_DEFAULT_VLLM: Dict[str, object] = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 2048,
    "dtype": "auto",
    "enable_prefix_caching": True,
    "extra_args": [],
}
_DEFAULT_LIFECYCLE: Dict[str, float] = {
    "health_check_timeout_s": 90.0,
    "health_poll_interval_s": 2.0,
    "startup_grace_s": 5.0,
    "shutdown_timeout_s": 20.0,
}


@dataclass
class ServerHandle:
    """Bookkeeping for one running vLLM server."""

    model_id: str
    port: int
    process: subprocess.Popen
    log_file: Optional["object"] = None  # open file handle for the server log

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"


@dataclass
class vLLMServerManager:
    """Start, health-check, and stop vLLM servers for local models.

    Args:
        models_config: Parsed ``config/models.yaml`` dict, or a path to it. Only
            entries with ``serving.engine == "vllm"`` are managed here.
        hardware_profile: Parsed ``config/hardware_profile.yaml`` dict, or a path
            to it. Supplies vLLM launch defaults + lifecycle timeouts.
        max_model_len: Convenience override for the served context window.
        log_dir: Directory for per-server stdout/stderr logs.
    """

    models_config: Dict = field(default_factory=dict)
    hardware_profile: Dict = field(default_factory=dict)
    max_model_len: Optional[int] = None
    log_dir: str = "logs/vllm"

    _vllm: Dict = field(default_factory=dict, init=False)
    _lifecycle: Dict = field(default_factory=dict, init=False)
    _servers: Dict[int, ServerHandle] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.models_config = _as_dict(self.models_config)
        self.hardware_profile = _as_dict(self.hardware_profile)

        self._vllm = {**_DEFAULT_VLLM, **self.hardware_profile.get("vllm_defaults", {})}
        self._lifecycle = {
            **_DEFAULT_LIFECYCLE,
            **self.hardware_profile.get("lifecycle", {}),
        }
        if self.max_model_len is not None:
            self._vllm["max_model_len"] = self.max_model_len

        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start_all(self) -> Dict[int, ServerHandle]:
        """Start a server for every vLLM-served local model in the config.

        Returns a mapping ``port -> ServerHandle``. Raises after cleaning up any
        already-started servers if one fails to come up.
        """
        for model in self._vllm_models():
            port = int(model["serving"]["vllm_port"])
            try:
                self.start_server(model["checkpoint"], port)
            except Exception:
                logger.error("Failed to start %s on :%s; stopping all", model["id"], port)
                self.stop_all()
                raise
        return dict(self._servers)

    def start_server(self, model_id: str, port: int) -> subprocess.Popen:
        """Start a vLLM server for a pre-quantized checkpoint.

        Args:
            model_id: HuggingFace model id (e.g. ``unsloth/Mistral-7B-v0.3-GGUF``).
            port: Port to serve on (8000, 8001, ...).

        Returns:
            The ``subprocess.Popen`` handle for the server.

        Raises:
            RuntimeError: if the port is already managed, or the server does not
                become healthy within the configured timeout.
        """
        if port in self._servers:
            raise RuntimeError(f"Port {port} already has a managed server")
        if _port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use by another process; free it first "
                f"(`lsof -i :{port}`)"
            )

        cmd = self._build_command(model_id, port)
        log_path = os.path.join(self.log_dir, f"vllm_{port}.log")
        logger.info("Starting vLLM: %s (port %s)\n  %s", model_id, port, " ".join(cmd))
        log_file = open(log_path, "w")  # noqa: SIM115 -- kept open for the process lifetime
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                # New process group so we can signal the whole vLLM tree cleanly.
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            log_file.close()
            raise RuntimeError(
                "`vllm` CLI not found. Install with `pip install vllm` "
                "(see requirements.txt)."
            ) from exc

        handle = ServerHandle(model_id=model_id, port=port, process=proc, log_file=log_file)
        self._servers[port] = handle

        if not self.health_check(port, timeout=self._lifecycle["health_check_timeout_s"]):
            self.stop_server(port)
            raise RuntimeError(
                f"vLLM server for {model_id} on :{port} did not become healthy "
                f"within {self._lifecycle['health_check_timeout_s']}s. See {log_path}."
            )
        logger.info("vLLM server ready: %s (port %s)", model_id, port)
        return proc

    def stop_server(self, port: int) -> None:
        """Gracefully shut down the vLLM server on ``port`` (SIGTERM -> SIGKILL)."""
        handle = self._servers.pop(port, None)
        if handle is None:
            logger.warning("stop_server: no managed server on port %s", port)
            return

        proc = handle.process
        if proc.poll() is None:  # still running
            logger.info("Stopping vLLM server on :%s (pid %s)", port, proc.pid)
            _terminate_group(proc, timeout=self._lifecycle["shutdown_timeout_s"])
        if handle.log_file is not None:
            handle.log_file.close()

    def stop_all(self) -> None:
        """Stop every managed server. Safe to call multiple times."""
        for port in list(self._servers.keys()):
            self.stop_server(port)

    def is_ready(self, port: int) -> bool:
        """Return True if the vLLM server on ``port`` answers ``/health`` with 200."""
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def health_check(self, port: int, timeout: float = 60.0) -> bool:
        """Poll ``/health`` until ready or ``timeout`` seconds elapse.

        Also fails fast if the underlying process has already exited.
        """
        interval = self._lifecycle["health_poll_interval_s"]
        deadline = time.monotonic() + timeout
        time.sleep(min(self._lifecycle["startup_grace_s"], timeout))

        handle = self._servers.get(port)
        while time.monotonic() < deadline:
            if handle is not None and handle.process.poll() is not None:
                logger.error(
                    "vLLM process on :%s exited early (code %s)",
                    port,
                    handle.process.returncode,
                )
                return False
            if self.is_ready(port):
                return True
            time.sleep(interval)
        return False

    # ------------------------------------------------------------------ #
    # Context-manager sugar -- ensures cleanup on exceptions.
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "vLLMServerManager":
        return self

    def __exit__(self, *_exc) -> None:
        self.stop_all()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _vllm_models(self) -> List[Dict]:
        models = self.models_config.get("local_models", [])
        return [m for m in models if m.get("serving", {}).get("engine") == "vllm"]

    def _build_command(self, model_id: str, port: int) -> List[str]:
        v = self._vllm
        cmd: List[str] = [
            "vllm",
            "serve",
            model_id,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(v["tensor_parallel_size"]),
            "--gpu-memory-utilization",
            str(v["gpu_memory_utilization"]),
            "--max-model-len",
            str(v["max_model_len"]),
            "--dtype",
            str(v["dtype"]),
        ]
        if v.get("enable_prefix_caching"):
            cmd.append("--enable-prefix-caching")
        cmd.extend(str(a) for a in v.get("extra_args", []))
        return cmd


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #
def _as_dict(value) -> Dict:
    """Accept either an already-parsed dict or a path to a YAML file."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        with open(value, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _terminate_group(proc: subprocess.Popen, timeout: float) -> None:
    """SIGTERM the process group, then SIGKILL if it outlives ``timeout``."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Server pid %s did not exit; sending SIGKILL", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------- #
# CLI: `python harness/vllm_server_manager.py start`
# ---------------------------------------------------------------------- #
def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage local vLLM servers.")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--models", default="config/models.yaml")
    parser.add_argument("--hardware", default="config/hardware_profile.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    manager = vLLMServerManager(models_config=args.models, hardware_profile=args.hardware)

    if args.action == "status":
        for model in manager._vllm_models():
            port = int(model["serving"]["vllm_port"])
            state = "READY" if manager.is_ready(port) else "down"
            print(f"{model['id']:<20} :{port}  {state}")
        return 0

    if args.action == "stop":
        # Best-effort: without persisted PIDs we can only report. Real shutdown
        # happens within a live manager process (or via `lsof`/`kill`).
        print("No managed servers in this process. Stop the process that ran `start`.")
        return 0

    # start
    manager.start_all()
    print("All vLLM servers started. Ctrl-C to stop.")
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
