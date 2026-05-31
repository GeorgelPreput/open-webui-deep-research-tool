"""Regression tests for config/env bug fixes.

Covers:
  BUG 42 - DR_MODELS_EMBEDDING_MODEL and DR_ADVANCED_PDF_LEGACY_TLS_VERIFY are
           wired into the env-var map.
  BUG 34 - malformed numeric env vars log a warning (instead of being silently
           swallowed) and fall back to the default.
"""
import logging

from deep_research.config.env import load_valves_from_env
from deep_research.config.valves import Valves


def test_embedding_model_env_override(monkeypatch):
    monkeypatch.setenv("DR_MODELS_EMBEDDING_MODEL", "my-embed-model")
    valves = load_valves_from_env()
    assert valves.models.embedding_model == "my-embed-model"


def test_pdf_legacy_tls_verify_env_override(monkeypatch):
    monkeypatch.setenv("DR_ADVANCED_PDF_LEGACY_TLS_VERIFY", "false")
    valves = load_valves_from_env()
    assert valves.advanced.pdf_legacy_tls_verify is False


def test_pdf_legacy_tls_verify_defaults_true(monkeypatch):
    monkeypatch.delenv("DR_ADVANCED_PDF_LEGACY_TLS_VERIFY", raising=False)
    valves = load_valves_from_env()
    assert valves.advanced.pdf_legacy_tls_verify is True


def test_malformed_numeric_env_logs_warning_and_uses_default(monkeypatch, caplog):
    monkeypatch.setenv("DR_WEB_MAX_RESULT_TOKENS", "notanumber")
    with caplog.at_level(logging.WARNING, logger="deep_research.config.env"):
        valves = load_valves_from_env()
    # Default preserved (no crash).
    assert valves.web.max_result_tokens == Valves().web.max_result_tokens
    # A diagnostic naming the offending var was emitted.
    assert any("MAX_RESULT_TOKENS" in r.message for r in caplog.records)
