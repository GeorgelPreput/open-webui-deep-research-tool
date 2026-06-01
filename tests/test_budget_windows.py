import asyncio
import time
import uuid

from deep_research.adapter.auth import StaticToken
from deep_research.adapter.client import OWUIClient
from deep_research.adapter.models import ModelInfo
from deep_research.budget.windows import get_model_context_window
from deep_research.config.valves import Valves
from deep_research.core.state import ResearchStateManager
from deep_research.core.types import ResearchMode, RunContext, RunUser
from deep_research.orchestrator.coordinator import CacheBundle, RuntimeConfig


def _build_ctx(valves=None, models_cache=None):
    v = valves or Valves()
    cfg = RuntimeConfig()
    cb = CacheBundle.create(v, cfg)
    if models_cache:
        cb.models = models_cache
    client = OWUIClient(
        base_url="http://x", token_provider=StaticToken("t"), timeout_seconds=1,
        max_retries=1,
        search_semaphore=asyncio.Semaphore(1),
        fetch_semaphore=asyncio.Semaphore(1),
    )
    return RunContext(
        user=RunUser(id="u", name="u"),
        conversation_id=str(uuid.uuid4()),
        chat_id=None, request_id="r", run_id="r",
        valves=v, config=cfg, client=client, llm=None, embeddings=None,
        events=None, caches=cb, state=ResearchStateManager(),
        executor=None, mode=ResearchMode.FRESH, started_at=time.monotonic(),
    )


def test_window_uses_research_override():
    v = Valves()
    v.models.research_context_window = 32000
    ctx = _build_ctx(v)
    assert get_model_context_window(ctx, v.models.research_model) == 32000


def test_window_uses_synthesis_override():
    v = Valves()
    v.models.synthesis_context_window = 65536
    ctx = _build_ctx(v)
    assert get_model_context_window(ctx, v.models.synthesis_model) == 65536


def test_window_reads_models_cache_modelinfo():
    v = Valves()
    cache = {v.models.research_model: ModelInfo(
        id=v.models.research_model, name="r", context_window=16384, meta={}
    )}
    ctx = _build_ctx(v, models_cache=cache)
    assert get_model_context_window(ctx, v.models.research_model) == 16384


def test_window_falls_back_to_8192():
    v = Valves()
    ctx = _build_ctx(v)
    assert get_model_context_window(ctx, "unknown-model") == 8192
