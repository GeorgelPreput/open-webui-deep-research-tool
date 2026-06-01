import json
import logging

import pytest

from deep_research.config.logging import (
    _LOGGER_NAME,
    _MANAGED_FLAG,
    configure_logging,
    redact_secret,
    reset_log_context,
    set_log_context,
)
from deep_research.config.valves import Valves


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("DR_LOG_LEVEL", "DR_LOG_FORMAT", "DR_LOG_INCLUDE_TRACEBACKS"):
        monkeypatch.delenv(k, raising=False)
    # Strip managed + non-managed handlers so each test starts clean.
    root = logging.getLogger(_LOGGER_NAME)
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)


def _managed_handlers():
    return [
        h
        for h in logging.getLogger(_LOGGER_NAME).handlers
        if getattr(h, _MANAGED_FLAG, False)
    ]


def test_configure_logging_sets_level_from_env(monkeypatch):
    monkeypatch.setenv("DR_LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger(_LOGGER_NAME).level == logging.DEBUG


def test_configure_logging_idempotent():
    configure_logging()
    configure_logging()
    assert len(_managed_handlers()) == 1


def test_redact_secret():
    assert redact_secret(None) == ""
    assert redact_secret("") == ""
    assert redact_secret("short") == "********"
    assert redact_secret("sk-abcdef1234567890xyz") == "sk-a…0xyz"


def test_correlation_context_in_record(capsys):
    configure_logging()
    handle = set_log_context(
        conversation_id="conv-1",
        chat_id="chat-2",
        run_id="run-3",
        request_id="req-4",
    )
    try:
        logging.getLogger("deep_research.test").warning("hello")
    finally:
        reset_log_context(handle)
    err = capsys.readouterr().err
    assert "conv=conv-1" in err
    assert "run=run-3" in err
    assert "req=req-4" in err
    assert "hello" in err


def test_correlation_defaults_to_dash(capsys):
    configure_logging()
    logging.getLogger("deep_research.test").warning("bare")
    err = capsys.readouterr().err
    assert "conv=-" in err
    assert "run=-" in err
    assert "req=-" in err


def test_json_format_emits_valid_json(monkeypatch, capsys):
    monkeypatch.setenv("DR_LOG_FORMAT", "json")
    configure_logging()
    logging.getLogger("deep_research.test").warning("payload")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    obj = json.loads(line)
    assert obj["level"] == "WARNING"
    assert obj["message"] == "payload"
    assert obj["logger"] == "deep_research.test"
    assert "conversation_id" in obj
    assert "run_id" in obj


def test_include_tracebacks_false_strips_traceback(monkeypatch, capsys):
    monkeypatch.setenv("DR_LOG_INCLUDE_TRACEBACKS", "false")
    configure_logging()
    log = logging.getLogger("deep_research.test")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("oops")
    err = capsys.readouterr().err
    assert "oops" in err
    assert "Traceback" not in err
    assert "ValueError" not in err


def test_valves_override_env_when_changed(monkeypatch):
    monkeypatch.setenv("DR_LOG_LEVEL", "INFO")
    valves = Valves()
    valves.logging.level = "DEBUG"
    configure_logging(valves)
    assert logging.getLogger(_LOGGER_NAME).level == logging.DEBUG


def test_default_when_nothing_set():
    configure_logging()
    assert logging.getLogger(_LOGGER_NAME).level == logging.INFO
