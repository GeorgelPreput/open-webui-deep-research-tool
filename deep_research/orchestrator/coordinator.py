import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from deep_research.adapter.auth import (
    BearerTokenProvider,
    ContextTokenProvider,
    reset_current_token,
    set_current_token,
)
from deep_research.adapter.client import OWUIClient
from deep_research.config.logging import reset_log_context, set_log_context
from deep_research.config.valves import Valves
from deep_research.core.caches import EmbeddingCache, LRUBytesBoundedCache, TransformationCache
from deep_research.core.state import ResearchStateManager
from deep_research.core.types import ChatMessage, Report, ResearchMode, RunContext, RunUser
from deep_research.progress.events import (
    EmbedEvent,
    Event,
    EventBus,
    MessageEvent,
    Sink,
    StatusEvent,
)

logger = logging.getLogger("deep_research.orchestrator")


class AlreadyRunningError(RuntimeError):
    pass


class RuntimeConfig:
    def __init__(
        self,
        data_dir: str = "/tmp/deep_research",
        base_url: str = "http://localhost:8080",
    ):
        # Coerce to Path: the runtime shims pass str (from env), but the
        # vocabulary disk-cache code uses `data_dir / "deep_research"`.
        self.data_dir = Path(data_dir)
        self.base_url = base_url


class CacheBundle:
    def __init__(self, valves: Valves, config: RuntimeConfig):
        from deep_research.config.constants import (
            EMBEDDING_CACHE_MAX_MB,
            TRANSFORMATION_CACHE_MAX_MB,
        )
        self.embedding = EmbeddingCache(max_bytes=EMBEDDING_CACHE_MAX_MB * 1024 * 1024)
        self.transformation = TransformationCache(max_bytes=TRANSFORMATION_CACHE_MAX_MB * 1024 * 1024)
        self.vocabulary = LRUBytesBoundedCache(max_bytes=64 * 1024 * 1024)
        self.vocabulary_embeddings = LRUBytesBoundedCache(max_bytes=128 * 1024 * 1024)
        self.models: dict[str, Any] = {}

    @classmethod
    def create(cls, valves: Valves, config: RuntimeConfig) -> "CacheBundle":
        return cls(valves, config)


class Coordinator:
    def __init__(self, *, valves: Valves, config: RuntimeConfig):
        self._valves = valves
        self._config = config
        self._inflight: set[str] = set()
        self._inflight_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._shared_caches = CacheBundle.create(valves, config)
        self._state_manager = ResearchStateManager()
        self._executor = ThreadPoolExecutor(max_workers=valves.advanced.executor_workers)
        self._client: OWUIClient | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            logger.info(
                "Coordinator starting: base_url=%s data_dir=%s "
                "llm_concurrency=%d embedding_concurrency=%d "
                "http_timeout=%ds http_max_retries=%d",
                self._config.base_url,
                self._config.data_dir,
                self._valves.advanced.llm_concurrency,
                self._valves.advanced.embedding_concurrency,
                self._valves.advanced.http_timeout_seconds,
                self._valves.advanced.http_max_retries,
            )
            self._client = OWUIClient(
                base_url=self._config.base_url,
                token_provider=ContextTokenProvider(),
                timeout_seconds=self._valves.advanced.http_timeout_seconds,
                max_retries=self._valves.advanced.http_max_retries,
                llm_semaphore=asyncio.Semaphore(self._valves.advanced.llm_concurrency),
                embedding_semaphore=asyncio.Semaphore(self._valves.advanced.embedding_concurrency),
                search_semaphore=asyncio.Semaphore(self._valves.web.search_concurrency),
                fetch_semaphore=asyncio.Semaphore(self._valves.web.fetch_concurrency),
            )
            await self._client.start()
            self._started = True
            logger.info("Coordinator started")

    async def close(self) -> None:
        logger.info("Coordinator shutting down")
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._executor.shutdown(wait=False)
        self._started = False

    async def run(
        self,
        *,
        user: RunUser,
        conversation_id: str,
        chat_id: str | None,
        token: BearerTokenProvider,
        prompt: str,
        history: list[ChatMessage],
        sink: Sink,
    ) -> Report:
        inflight_key = f"{user.id}:{conversation_id}"
        async with self._inflight_lock:
            if inflight_key in self._inflight:
                raise AlreadyRunningError(
                    f"Research already running for conversation {conversation_id}"
                )
            self._inflight.add(inflight_key)

        bearer = await token.get_token()
        token_handle = set_current_token(bearer)
        run_started = time.monotonic()
        log_handle = None
        try:
            ctx = await self._build_context(user, conversation_id, chat_id, token, prompt, history, sink)
            log_handle = set_log_context(
                conversation_id=ctx.conversation_id,
                chat_id=ctx.chat_id or "-",
                run_id=ctx.run_id,
                request_id=ctx.request_id,
            )
            logger.info(
                "Run started: conversation_id=%s prompt_chars=%d",
                ctx.conversation_id,
                len(prompt or ""),
            )
            await ctx.events.start()
            try:
                return await self._run_phases(ctx)
            finally:
                await ctx.events.stop()
        finally:
            elapsed = time.monotonic() - run_started
            logger.info(
                "Run finished: conversation_id=%s elapsed_s=%.2f",
                conversation_id,
                elapsed,
            )
            if log_handle is not None:
                reset_log_context(log_handle)
            reset_current_token(token_handle)
            async with self._inflight_lock:
                self._inflight.discard(inflight_key)

    async def stream(self, **kwargs) -> AsyncIterator[Event]:
        sink_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def sink(event: Event) -> None:
            if isinstance(event, (StatusEvent, MessageEvent, EmbedEvent)):
                await sink_queue.put(event)

        async def runner():
            try:
                result = await self.run(**kwargs, sink=sink)
                await sink_queue.put(MessageEvent(content=result.content))
            except Exception as e:
                await sink_queue.put(StatusEvent(description=str(e), level="error", done=True))
            finally:
                await sink_queue.put(None)

        task = asyncio.create_task(runner())
        while True:
            event = await sink_queue.get()
            if event is None:
                break
            yield event
        await task

    async def _build_context(
        self,
        user: RunUser,
        conversation_id: str,
        chat_id: str | None,
        token: BearerTokenProvider,
        prompt: str,
        history: list[ChatMessage],
        sink: Sink,
    ) -> RunContext:
        event_bus = EventBus(sink, flush_interval_ms=self._valves.events.flush_interval_ms)
        ctx = RunContext(
            user=user,
            conversation_id=conversation_id,
            chat_id=chat_id,
            request_id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
            valves=self._valves,
            config=self._config,
            client=self._client,
            events=event_bus,
            caches=self._shared_caches,
            state=self._state_manager,
            executor=self._executor,
            mode=ResearchMode.FRESH,
            started_at=time.time(),
            research_date=datetime.now().strftime("%Y-%m-%d"),
            prompt=prompt,
            history=history,
        )
        return ctx

    async def _run_phases(self, ctx: RunContext) -> Report:
        from deep_research.orchestrator.phases import (
            compress,
            cycles,
            finalize,
            front_back,
            initial_queries,
            outline,
            rehydrate,
            synthesize,
        )
        from deep_research.orchestrator.phases import (
            outline_feedback as of_phase,
        )

        async def _phase(name: str, coro):
            logger.info("Phase start: %s", name)
            t0 = time.monotonic()
            try:
                result = await coro
                logger.info(
                    "Phase done: %s elapsed_s=%.2f",
                    name,
                    time.monotonic() - t0,
                )
                return result
            except Exception:
                logger.exception("Phase failed: %s", name)
                raise

        phase_state: dict[str, Any] = {
            "user_message": ctx.prompt,
            "history": ctx.history,
        }
        ctx.state.update_state(ctx.conversation_id, "last_user_message", ctx.prompt)
        phase_state = await _phase("rehydrate", rehydrate.run_rehydrate(ctx, phase_state))

        if phase_state.get("post_report_mode"):
            from deep_research.persistence.sources import answer_post_report_user_qa
            logger.info("Phase start: post_report_qa")
            t0 = time.monotonic()
            answer = await answer_post_report_user_qa(
                ctx,
                body={"messages": [{"role": "user", "content": ctx.prompt}]},
            )
            logger.info(
                "Phase done: post_report_qa elapsed_s=%.2f",
                time.monotonic() - t0,
            )
            return Report(content=answer, conversation_id=ctx.conversation_id)

        phase_state = await _phase(
            "outline_feedback", of_phase.run_outline_feedback(ctx, phase_state)
        )
        phase_state = await _phase(
            "initial_queries", initial_queries.run_initial_queries(ctx, phase_state)
        )

        if phase_state.get("awaiting_outline_feedback"):
            return Report(
                content=phase_state.get("outline_report", ""),
                conversation_id=ctx.conversation_id,
            )

        phase_state = await _phase("outline", outline.run_outline(ctx, phase_state))
        phase_state = await _phase("cycles", cycles.run_cycles(ctx, phase_state))
        phase_state = await _phase("compress", compress.run_compress(ctx, phase_state))
        phase_state = await _phase("synthesize", synthesize.run_synthesize(ctx, phase_state))
        phase_state = await _phase("front_back", front_back.run_front_back(ctx, phase_state))
        report = await _phase("finalize", finalize.run_finalize(ctx, phase_state))
        return report
