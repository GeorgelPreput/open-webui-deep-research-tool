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


# ---- LLM provider env loading ----

def test_env_loader_picks_up_llm_group(monkeypatch):
    monkeypatch.setenv("DR_LLM_BASE_URL", "http://my-llm:8000")
    monkeypatch.setenv("DR_LLM_API_KEY", "sk-abc123")
    monkeypatch.setenv("DR_LLM_CHAT_PATH", "/v1/chat/completions")
    valves = load_valves_from_env()
    assert valves.llm.base_url == "http://my-llm:8000"
    assert valves.llm.api_key == "sk-abc123"
    assert valves.llm.chat_path == "/v1/chat/completions"


def test_env_loader_llm_defaults(monkeypatch):
    for key in ("DR_LLM_BASE_URL", "DR_LLM_API_KEY", "DR_LLM_CHAT_PATH"):
        monkeypatch.delenv(key, raising=False)
    valves = load_valves_from_env()
    assert valves.llm.base_url == ""
    assert valves.llm.api_key == ""
    assert valves.llm.chat_path == "/chat/completions"


def test_env_loader_picks_up_embeddings_group(monkeypatch):
    monkeypatch.setenv("DR_EMBEDDINGS_BASE_URL", "http://my-emb:9000")
    monkeypatch.setenv("DR_EMBEDDINGS_API_KEY", "sk-emb")
    monkeypatch.setenv("DR_EMBEDDINGS_EMBEDDINGS_PATH", "/v1/embeddings")
    valves = load_valves_from_env()
    assert valves.embeddings.base_url == "http://my-emb:9000"
    assert valves.embeddings.api_key == "sk-emb"
    assert valves.embeddings.embeddings_path == "/v1/embeddings"


def test_env_loader_embeddings_defaults(monkeypatch):
    for key in ("DR_EMBEDDINGS_BASE_URL", "DR_EMBEDDINGS_API_KEY", "DR_EMBEDDINGS_EMBEDDINGS_PATH"):
        monkeypatch.delenv(key, raising=False)
    valves = load_valves_from_env()
    assert valves.embeddings.base_url == ""
    assert valves.embeddings.api_key == ""
    assert valves.embeddings.embeddings_path == "/embeddings"


def test_embeddings_api_key_redacted_in_summary(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("DR_EMBEDDINGS_BASE_URL", "http://emb:9000")
    monkeypatch.setenv("DR_EMBEDDINGS_API_KEY", "sk-emb-secret")
    with caplog.at_level(logging.DEBUG, logger="deep_research.config.env"):
        load_valves_from_env()
    assert "sk-emb-secret" not in caplog.text
    assert any("embeddings" in r.message.lower() for r in caplog.records)


def test_llm_api_key_redacted_in_summary(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("DR_LLM_BASE_URL", "http://llm:8000")
    monkeypatch.setenv("DR_LLM_API_KEY", "sk-supersecret")
    with caplog.at_level(logging.DEBUG, logger="deep_research.config.env"):
        load_valves_from_env()
    assert "sk-supersecret" not in caplog.text
    assert any("llm" in r.message.lower() for r in caplog.records)


# ---- Coordinator fail-fast on missing LLM config ----

@pytest.mark.asyncio
async def test_coordinator_fails_fast_without_llm_base_url():
    from deep_research import Coordinator
    from deep_research.config.valves import Valves
    from deep_research.orchestrator.coordinator import RuntimeConfig

    config = RuntimeConfig(
        data_dir="/tmp/dr_test",
        base_url="http://owui:8080",
        llm_base_url="",
        llm_api_key="sk-x",
        embeddings_base_url="http://emb:9000",
        embeddings_api_key="sk-emb",
    )
    coord = Coordinator(valves=Valves(), config=config)
    with pytest.raises(ValueError, match="llm_base_url"):
        await coord.start()


@pytest.mark.asyncio
async def test_coordinator_fails_fast_without_llm_api_key():
    from deep_research import Coordinator
    from deep_research.config.valves import Valves
    from deep_research.orchestrator.coordinator import RuntimeConfig

    config = RuntimeConfig(
        data_dir="/tmp/dr_test",
        base_url="http://owui:8080",
        llm_base_url="http://llm:8000",
        llm_api_key="",
        embeddings_base_url="http://emb:9000",
        embeddings_api_key="sk-emb",
    )
    coord = Coordinator(valves=Valves(), config=config)
    with pytest.raises(ValueError, match="llm_api_key"):
        await coord.start()


@pytest.mark.asyncio
async def test_coordinator_fails_fast_without_embeddings_base_url():
    from deep_research import Coordinator
    from deep_research.config.valves import Valves
    from deep_research.orchestrator.coordinator import RuntimeConfig

    config = RuntimeConfig(
        data_dir="/tmp/dr_test",
        base_url="http://owui:8080",
        llm_base_url="http://llm:8000",
        llm_api_key="sk-x",
        embeddings_base_url="",
        embeddings_api_key="sk-emb",
    )
    coord = Coordinator(valves=Valves(), config=config)
    with pytest.raises(ValueError, match="embeddings_base_url"):
        await coord.start()


@pytest.mark.asyncio
async def test_coordinator_fails_fast_without_embeddings_api_key():
    from deep_research import Coordinator
    from deep_research.config.valves import Valves
    from deep_research.orchestrator.coordinator import RuntimeConfig

    config = RuntimeConfig(
        data_dir="/tmp/dr_test",
        base_url="http://owui:8080",
        llm_base_url="http://llm:8000",
        llm_api_key="sk-x",
        embeddings_base_url="http://emb:9000",
        embeddings_api_key="",
    )
    coord = Coordinator(valves=Valves(), config=config)
    with pytest.raises(ValueError, match="embeddings_api_key"):
        await coord.start()


