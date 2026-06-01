import os

import pytest

from deep_research.config.env import load_valves_from_env
from deep_research.config.valves import Valves


def test_env_loader_defaults_match_valves(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DR_"):
            monkeypatch.delenv(key, raising=False)
    valves = load_valves_from_env()
    default = Valves()
    assert valves.models.research_model == default.models.research_model
    assert valves.cycles.min_cycles == default.cycles.min_cycles


def test_env_loader_overrides_known_fields(monkeypatch):
    monkeypatch.setenv("DR_MODELS_RESEARCH_MODEL", "test-model")
    monkeypatch.setenv("DR_CYCLES_MIN_CYCLES", "3")
    monkeypatch.setenv("DR_WEB_QUALITY_FILTER_ENABLED", "false")
    monkeypatch.setenv("DR_ADVANCED_LLM_CONCURRENCY", "16")

    valves = load_valves_from_env()
    assert valves.models.research_model == "test-model"
    assert valves.cycles.min_cycles == 3
    assert valves.web.quality_filter_enabled is False
    assert valves.advanced.llm_concurrency == 16


def test_env_loader_ignores_unknown(monkeypatch):
    monkeypatch.setenv("DR_NOTAGROUP_BLAH", "1")
    monkeypatch.setenv("DR_MODELS_NOTAFIELD", "1")
    monkeypatch.setenv("UNRELATED_VAR", "stuff")
    # Should not raise
    load_valves_from_env()


def test_env_loader_swallows_bad_values(monkeypatch):
    monkeypatch.setenv("DR_CYCLES_MIN_CYCLES", "not-an-int")
    valves = load_valves_from_env()
    # Bad value silently skipped, default preserved
    assert valves.cycles.min_cycles == Valves().cycles.min_cycles


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_env_loader_bool_coercion(monkeypatch, raw, expected):
    monkeypatch.setenv("DR_PERSISTENCE_INTERACTIVE_RESEARCH", raw)
    valves = load_valves_from_env()
    assert valves.persistence.interactive_research is expected


def test_env_loader_picks_up_logging_group(monkeypatch):
    monkeypatch.setenv("DR_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("DR_LOGGING_FORMAT", "json")
    monkeypatch.setenv("DR_LOGGING_INCLUDE_TRACEBACKS", "false")
    valves = load_valves_from_env()
    assert valves.logging.level == "DEBUG"
    assert valves.logging.format == "json"
    assert valves.logging.include_tracebacks is False
