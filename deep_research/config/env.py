import logging
import os
from typing import Any

from deep_research.config.valves import Valves

logger = logging.getLogger("deep_research.config.env")


def load_valves_from_env(prefix: str = "DR_") -> Valves:
    prefix_upper = prefix.upper()
    env_data: dict[str, Any] = {}

    for key, raw in os.environ.items():
        if not key.startswith(prefix_upper):
            continue
        rest = key[len(prefix_upper) :]
        parts = rest.lower().split("_", 1)
        if len(parts) != 2:
            continue
        group, field = parts
        if group not in VALVES_GROUP_MAP:
            continue
        if field not in VALVES_GROUP_MAP[group]:
            continue
        target_type = VALVES_GROUP_MAP[group][field]
        try:
            env_data.setdefault(group, {})[field] = _coerce(raw, target_type)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Ignoring env var {key}={raw!r}: cannot coerce to "
                f"{target_type.__name__} ({e}); using default"
            )

    raw_valves = {}
    for group, fields in env_data.items():
        raw_valves[group] = fields

    if env_data:
        logger.debug("Loaded DR_* env overrides: %s", _summarize(env_data))

    return Valves.model_validate(raw_valves)


_SENSITIVE_FIELD_HINTS = ("key", "token", "secret", "password")


def _summarize(env_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from deep_research.config.logging import redact_secret

    out: dict[str, dict[str, Any]] = {}
    for group, fields in env_data.items():
        out[group] = {}
        for field, value in fields.items():
            lower = field.lower()
            if any(hint in lower for hint in _SENSITIVE_FIELD_HINTS):
                out[group][field] = redact_secret(str(value))
            else:
                out[group][field] = value
    return out


def _coerce(value: str, target: type) -> Any:
    if target is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    if target is str:
        return value
    return value


VALVES_GROUP_MAP: dict[str, dict[str, type]] = {
    "models": {
        "research_model": str,
        "synthesis_model": str,
        "quality_filter_model": str,
        "embedding_model": str,
        "research_context_window": int,
        "synthesis_context_window": int,
        "temperature": float,
        "synthesis_temperature": float,
    },
    "cycles": {
        "min_cycles": int,
        "max_cycles": int,
        "gap_exploration_weight": float,
        "trajectory_momentum": float,
        "followup_weight": float,
    },
    "web": {
        "search_results_per_query": int,
        "successful_results_per_query": int,
        "extra_results_per_query": int,
        "repeats_before_expansion": int,
        "max_result_tokens": int,
        "domain_priority": str,
        "content_priority": str,
        "quality_filter_enabled": bool,
        "quality_similarity_threshold": float,
        "fetch_concurrency": int,
        "search_concurrency": int,
    },
    "compression": {
        "chunk_level": int,
        "compression_level": int,
        "stepped_synthesis_compression": bool,
    },
    "persistence": {
        "export_research_data": bool,
        "interactive_research": bool,
        "user_preference_throughout": bool,
    },
    "events": {
        "enable_progress_embed": bool,
        "flush_interval_ms": int,
        "quiet_chat_mode": bool,
    },
    "advanced": {
        "query_weight": float,
        "llm_concurrency": int,
        "embedding_concurrency": int,
        "executor_workers": int,
        "http_timeout_seconds": int,
        "http_max_retries": int,
        "pdf_legacy_tls_verify": bool,
    },
    "logging": {
        "level": str,
        "format": str,
        "include_tracebacks": bool,
    },
}
