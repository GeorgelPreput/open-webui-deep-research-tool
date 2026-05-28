import asyncio
import time
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from deep_research.adapter.auth import StaticToken
from deep_research.adapter.client import OWUIClient
from deep_research.config.valves import Valves
from deep_research.core.state import ResearchStateManager
from deep_research.core.types import ResearchMode, RunContext, RunUser
from deep_research.orchestrator.coordinator import CacheBundle, RuntimeConfig
from deep_research.progress.events import EventBus


@pytest.fixture
def valves() -> Valves:
    return Valves()


@pytest.fixture
def runtime_config(tmp_path: Path) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.data_dir = tmp_path
    return cfg


@pytest.fixture
def cache_bundle(valves: Valves, runtime_config: RuntimeConfig) -> CacheBundle:
    return CacheBundle.create(valves, runtime_config)


@pytest.fixture
def state_manager() -> ResearchStateManager:
    return ResearchStateManager()


@pytest_asyncio.fixture
async def owui_client(valves: Valves):
    client = OWUIClient(
        base_url="http://mock-owui:8080",
        token_provider=StaticToken("mock-token"),
        timeout_seconds=10,
        max_retries=1,
        llm_semaphore=asyncio.Semaphore(4),
        embedding_semaphore=asyncio.Semaphore(8),
        search_semaphore=asyncio.Semaphore(2),
        fetch_semaphore=asyncio.Semaphore(4),
    )
    await client.start()
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def event_bus():
    sink_calls: list = []

    async def sink(event):
        sink_calls.append(event)

    bus = EventBus(sink, flush_interval_ms=50)
    bus.sink_calls = sink_calls  # type: ignore[attr-defined]
    await bus.start()
    try:
        yield bus
    finally:
        await bus.stop()


@pytest_asyncio.fixture
async def run_context(
    valves: Valves,
    runtime_config: RuntimeConfig,
    cache_bundle: CacheBundle,
    state_manager: ResearchStateManager,
    owui_client: OWUIClient,
    event_bus: EventBus,
):
    user = RunUser(id="test-user", name="Test", email="test@example.com")
    yield RunContext(
        user=user,
        conversation_id=f"conv-{uuid.uuid4()}",
        chat_id=None,
        request_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        valves=valves,
        config=runtime_config,
        client=owui_client,
        events=event_bus,
        caches=cache_bundle,
        state=state_manager,
        executor=None,
        mode=ResearchMode.FRESH,
        started_at=time.monotonic(),
    )
