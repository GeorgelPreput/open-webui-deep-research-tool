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
        cfg = RuntimeConfig()
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
    cfg = RuntimeConfig()
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start()
    assert coord._client is not None
    await coord.close()
    assert coord._client is None


@pytest.mark.asyncio
async def test_inflight_dedupe_rejects_duplicate(tmp_path, monkeypatch):
    cfg = RuntimeConfig()
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
    cfg = RuntimeConfig()
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
