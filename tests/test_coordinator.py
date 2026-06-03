import asyncio

import pytest

from deep_research import Coordinator, Valves
from deep_research.adapter.auth import StaticToken
from deep_research.core.types import RunUser
from deep_research.orchestrator.coordinator import AlreadyRunningError, RuntimeConfig


@pytest.fixture
def coord_factory(tmp_path):
    coords: list[Coordinator] = []

    def _make() -> Coordinator:
        cfg = RuntimeConfig(
        llm_base_url="http://mock-llm:9090",
        llm_api_key="sk-test",
        embeddings_base_url="http://mock-emb:9091",
        embeddings_api_key="sk-emb",
    )
        cfg.data_dir = tmp_path
        c = Coordinator(valves=Valves(), config=cfg)
        coords.append(c)
        return c

    yield _make
    # Synchronously close any coordinator that started a client
    for c in coords:
        if c._started:
            asyncio.get_event_loop().run_until_complete(c.close())


@pytest.mark.asyncio
async def test_coordinator_start_close(tmp_path):
    cfg = RuntimeConfig(
        llm_base_url="http://mock-llm:9090",
        llm_api_key="sk-test",
        embeddings_base_url="http://mock-emb:9091",
        embeddings_api_key="sk-emb",
    )
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start()
    assert coord._client is not None
    # No writeback_token passed → writeback_client stays unset
    assert coord.writeback_client is None
    await coord.close()
    assert coord._client is None


@pytest.mark.asyncio
async def test_coordinator_start_with_writeback_token(tmp_path):
    cfg = RuntimeConfig(
        llm_base_url="http://mock-llm:9090",
        llm_api_key="sk-test",
        embeddings_base_url="http://mock-emb:9091",
        embeddings_api_key="sk-emb",
    )
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start(writeback_token="sk-admin")
    assert coord.writeback_client is not None
    assert coord.writeback_client is not coord._client
    # Token provider is StaticToken("sk-admin")
    token = await coord.writeback_client._token_provider.get_token()
    assert token == "sk-admin"
    await coord.close()
    assert coord.writeback_client is None


@pytest.mark.asyncio
async def test_inflight_dedupe_rejects_duplicate(tmp_path, monkeypatch):
    cfg = RuntimeConfig(
        llm_base_url="http://mock-llm:9090",
        llm_api_key="sk-test",
        embeddings_base_url="http://mock-emb:9091",
        embeddings_api_key="sk-emb",
    )
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start()
    try:
        # Stub _run_phases to block until released so we can test concurrency
        release = asyncio.Event()

        async def fake_run_phases(ctx):
            await release.wait()
            from deep_research.core.types import Report
            return Report(content="ok", conversation_id=ctx.conversation_id)

        monkeypatch.setattr(coord, "_run_phases", fake_run_phases)

        user = RunUser(id="u1", name="u1")
        token = StaticToken("t")
        sink_calls: list = []

        async def sink(event):
            sink_calls.append(event)

        first = asyncio.create_task(
            coord.run(
                user=user,
                conversation_id="cv1",
                chat_id=None,
                token=token,
                prompt="q",
                history=[],
                sink=sink,
            )
        )
        # Give the first invocation a tick to claim the inflight slot
        await asyncio.sleep(0.05)

        with pytest.raises(AlreadyRunningError):
            await coord.run(
                user=user,
                conversation_id="cv1",
                chat_id=None,
                token=token,
                prompt="q2",
                history=[],
                sink=sink,
            )

        release.set()
        result = await first
        assert result.content == "ok"
    finally:
        await coord.close()


@pytest.mark.asyncio
async def test_distinct_conversations_run_concurrently(tmp_path, monkeypatch):
    cfg = RuntimeConfig(
        llm_base_url="http://mock-llm:9090",
        llm_api_key="sk-test",
        embeddings_base_url="http://mock-emb:9091",
        embeddings_api_key="sk-emb",
    )
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start()
    try:
        async def fake_run_phases(ctx):
            from deep_research.core.types import Report
            return Report(content=ctx.conversation_id, conversation_id=ctx.conversation_id)

        monkeypatch.setattr(coord, "_run_phases", fake_run_phases)
        token = StaticToken("t")
        user = RunUser(id="u1", name="u1")

        async def sink(event):
            pass

        a, b = await asyncio.gather(
            coord.run(user=user, conversation_id="A", chat_id=None,
                      token=token, prompt="x", history=[], sink=sink),
            coord.run(user=user, conversation_id="B", chat_id=None,
                      token=token, prompt="x", history=[], sink=sink),
        )
        assert {a.content, b.content} == {"A", "B"}
    finally:
        await coord.close()
