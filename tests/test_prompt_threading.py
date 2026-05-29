import asyncio

import pytest

import deep_research.orchestrator.phases.cycles as cycles_mod
import deep_research.orchestrator.phases.rehydrate as rehydrate_mod
import deep_research.persistence.sources as sources_mod
from deep_research import Coordinator, Valves
from deep_research.adapter.auth import StaticToken
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig


@pytest.mark.asyncio
async def test_build_context_stores_prompt_and_history(tmp_path):
    cfg = RuntimeConfig()
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start()
    try:
        user = RunUser(id="u1", name="u1")
        token = StaticToken("t")
        history = [ChatMessage(role="user", content="earlier")]

        async def sink(event):
            pass

        ctx = await coord._build_context(
            user, "conv1", None, token, "what is mamba?", history, sink
        )
        assert ctx.prompt == "what is mamba?"
        assert ctx.history == history
    finally:
        await coord.close()


@pytest.mark.asyncio
async def test_post_report_branch_uses_prompt_and_skips_cycles(tmp_path, monkeypatch):
    cfg = RuntimeConfig()
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    await coord.start()
    try:
        # rehydrate flags this run as post-report QA
        async def fake_rehydrate(ctx, state):
            state["post_report_mode"] = True
            return state

        captured: dict = {}

        async def fake_qa(ctx, body):
            captured["body"] = body
            return "answer-from-kb"

        # cycles must NOT run in post-report mode
        async def boom_cycles(ctx, ps):
            raise AssertionError("run_cycles must not be called in post-report mode")

        monkeypatch.setattr(rehydrate_mod, "run_rehydrate", fake_rehydrate)
        monkeypatch.setattr(cycles_mod, "run_cycles", boom_cycles)
        monkeypatch.setattr(sources_mod, "answer_post_report_user_qa", fake_qa)

        user = RunUser(id="u1", name="u1")
        token = StaticToken("t")

        async def sink(event):
            pass

        report = await coord.run(
            user=user,
            conversation_id="cv-pr",
            chat_id=None,
            token=token,
            prompt="follow-up question",
            history=[],
            sink=sink,
        )
        assert report.content == "answer-from-kb"
        messages = captured["body"]["messages"]
        assert messages[-1]["content"] == "follow-up question"
    finally:
        await coord.close()


@pytest.mark.asyncio
async def test_concurrent_start_creates_single_client(tmp_path):
    cfg = RuntimeConfig()
    cfg.data_dir = tmp_path
    coord = Coordinator(valves=Valves(), config=cfg)
    try:
        await asyncio.gather(*(coord.start() for _ in range(8)))
        # Double-checked locking must produce exactly one client.
        assert coord._client is not None
        assert coord._started is True
    finally:
        await coord.close()
