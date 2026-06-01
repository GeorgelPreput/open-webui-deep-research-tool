"""Centralized logging configuration for the deep_research package.

Reads config from (highest first):
1. Explicit ``valves.logging.*`` fields (when ``valves`` is passed and the
   field differs from its default).
2. Bare env vars ``DR_LOG_LEVEL`` / ``DR_LOG_FORMAT`` /
   ``DR_LOG_INCLUDE_TRACEBACKS``.
3. Defaults: ``INFO`` / ``text`` / ``True``.

Handlers are attached to the ``"deep_research"`` logger (not root) with
``propagate = False`` so we coexist cleanly with uvicorn / OWUI root config.
``configure_logging()`` is idempotent — repeated calls replace the prior
managed handler instead of stacking duplicates.
"""

from __future__ import annotations

import contextvars
import json as _json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deep_research.config.valves import Valves

_LOGGER_NAME = "deep_research"
_MANAGED_FLAG = "_dr_managed"

_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "text"
_DEFAULT_INCLUDE_TRACEBACKS = True

_TEXT_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[conv=%(conversation_id)s run=%(run_id)s req=%(request_id)s] "
    "%(message)s"
)

_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dr_conversation_id", default="-"
)
_chat_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dr_chat_id", default="-"
)
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dr_run_id", default="-"
)
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dr_request_id", default="-"
)


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.conversation_id = _conversation_id.get()
        record.chat_id = _chat_id.get()
        record.run_id = _run_id.get()
        record.request_id = _request_id.get()
        return True


class _JSONFormatter(logging.Formatter):
    def __init__(self, *, include_tracebacks: bool) -> None:
        super().__init__()
        self._include_tracebacks = include_tracebacks

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "conversation_id": getattr(record, "conversation_id", "-"),
            "chat_id": getattr(record, "chat_id", "-"),
            "run_id": getattr(record, "run_id", "-"),
            "request_id": getattr(record, "request_id", "-"),
        }
        if self._include_tracebacks and record.exc_info:
            payload["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )
        return _json.dumps(payload, ensure_ascii=False, default=str)


class _StripTracebackFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info or record.exc_text:
            original_exc_info = record.exc_info
            original_exc_text = record.exc_text
            record.exc_info = None
            record.exc_text = None
            try:
                return super().format(record)
            finally:
                record.exc_info = original_exc_info
                record.exc_text = original_exc_text
        return super().format(record)


def _resolve_level(value: str | None) -> int:
    if not value:
        return logging.INFO
    upper = value.strip().upper()
    return logging.getLevelName(upper) if isinstance(logging.getLevelName(upper), int) else logging.INFO


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _resolve_settings(valves: "Valves | None") -> tuple[int, str, bool]:
    env_level = os.environ.get("DR_LOG_LEVEL")
    env_format = os.environ.get("DR_LOG_FORMAT")
    env_tb = os.environ.get("DR_LOG_INCLUDE_TRACEBACKS")

    level_str = env_level if env_level else _DEFAULT_LEVEL
    format_str = env_format if env_format else _DEFAULT_FORMAT
    include_tb = _coerce_bool(env_tb, _DEFAULT_INCLUDE_TRACEBACKS)

    if valves is not None:
        lv = getattr(valves, "logging", None)
        if lv is not None:
            # Valves win when explicitly set away from default.
            if lv.level and lv.level != _DEFAULT_LEVEL:
                level_str = lv.level
            elif not env_level and lv.level:
                level_str = lv.level
            if lv.format and lv.format != _DEFAULT_FORMAT:
                format_str = lv.format
            elif not env_format and lv.format:
                format_str = lv.format
            if env_tb is None:
                include_tb = lv.include_tracebacks

    return _resolve_level(level_str), (format_str or _DEFAULT_FORMAT).lower(), include_tb


def configure_logging(
    valves: "Valves | None" = None,
    *,
    force: bool = False,
) -> None:
    level, fmt, include_tb = _resolve_settings(valves)

    root = logging.getLogger(_LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False

    for h in list(root.handlers):
        if getattr(h, _MANAGED_FLAG, False) or force:
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _MANAGED_FLAG, True)
    handler.setLevel(level)
    handler.addFilter(CorrelationFilter())

    if fmt == "json":
        handler.setFormatter(_JSONFormatter(include_tracebacks=include_tb))
    else:
        if include_tb:
            handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
        else:
            handler.setFormatter(_StripTracebackFormatter(_TEXT_FORMAT))

    root.addHandler(handler)


def set_log_context(
    *,
    conversation_id: str | None = None,
    chat_id: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
) -> object:
    snapshot = {
        "conversation_id": _conversation_id.set(conversation_id)
        if conversation_id is not None
        else None,
        "chat_id": _chat_id.set(chat_id) if chat_id is not None else None,
        "run_id": _run_id.set(run_id) if run_id is not None else None,
        "request_id": _request_id.set(request_id) if request_id is not None else None,
    }
    return snapshot


def reset_log_context(handle: object) -> None:
    if not isinstance(handle, dict):
        return
    tok = handle.get("conversation_id")
    if tok is not None:
        _conversation_id.reset(tok)
    tok = handle.get("chat_id")
    if tok is not None:
        _chat_id.reset(tok)
    tok = handle.get("run_id")
    if tok is not None:
        _run_id.reset(tok)
    tok = handle.get("request_id")
    if tok is not None:
        _request_id.reset(tok)


def redact_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) < 9:
        return "********"
    return f"{value[:4]}…{value[-4:]}"
