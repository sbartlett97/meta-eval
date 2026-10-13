"""Tests for harness.env.load_env (network-free)."""

import os
import sys

import pytest

from harness.env import load_env

pytest.importorskip("dotenv")


def test_load_env_sets_missing_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("META_EVAL_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("META_EVAL_TEST_KEY=from-file\n")

    assert load_env(str(env)) is True
    assert os.environ["META_EVAL_TEST_KEY"] == "from-file"
    monkeypatch.delenv("META_EVAL_TEST_KEY")


def test_load_env_does_not_override_existing_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("META_EVAL_TEST_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("META_EVAL_TEST_KEY=from-file\n")

    load_env(str(env))
    assert os.environ["META_EVAL_TEST_KEY"] == "from-shell"


def test_load_env_missing_file_is_a_noop(tmp_path):
    assert load_env(str(tmp_path / "nope.env")) is False


def test_load_env_without_python_dotenv_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("META_EVAL_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("META_EVAL_TEST_KEY=from-file\n")
    monkeypatch.setitem(sys.modules, "dotenv", None)  # makes `import dotenv` raise

    assert load_env(str(env)) is False
    assert "META_EVAL_TEST_KEY" not in os.environ
