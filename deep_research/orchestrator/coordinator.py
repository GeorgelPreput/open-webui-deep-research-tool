import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from deep_research.adapter.auth import (
    BearerTokenProvider,
    ContextTokenProvider,
    StaticToken,
    reset_current_token,
    set_current_token,
)
from deep_research.adapter.client import OWUIClient
from deep_research.adapter.llm_provider import EmbeddingProviderClient, LLMProviderClient
from deep_research.adapter.throttle import HttpThrottle
from deep_research.config.logging import reset_log_context, set_log_context
from deep_research.config.valves import Valves
from deep_research.core.caches import EmbeddingCache, LRUBytesBoundedCache, TransformationCache
from deep_research.core.state import ResearchStateManager
from deep_research.core.types import ChatMessage, Report, ResearchMode, RunContext, RunUser
from deep_research.progress.events import EventBus, Sink, StatusEvent

logger = logging.getLogger("deep_research.orchestrator")


def _dr_version() -> str:
    """Best-effort lookup of the installed ``deep-research`` package version.

    Falls back to ``"unknown"`` if the package is not installed in a way
    importlib can introspect (e.g. ad-hoc editable run from a source tree
    that was never ``pip install``ed). Logged once per process so operators
    can correlate behaviour with a deployed build.
    """
    try:
        return _pkg_version("deep-research")
    except PackageNotFoundError:
        return "unknown"


class AlreadyRunningError(RuntimeError):
    pass


class RuntimeConfig:
    def __init__(
        self,
        data_dir: str = "/tmp/deep_research",
        base_url: str = "http://localhost:8080",
        # Chat provider
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_chat_path: str = "/chat/completions",
        # Embeddings provider — fully independent of the chat provider
        embeddings_base_url: str = "",
        embeddings_api_key: str = "",
        embeddings_path: str = "/embeddings",
    ):
        # Coerce to Path: the runtime shims pass str (from env), but the
        # vocabulary disk-cache code uses `data_dir / "deep_research"`.
        self.data_dir = Path(data_dir)
        self.base_url = base_url
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_chat_path = llm_chat_path
        self.embeddings_base_url = embeddings_base_url
        self.embeddings_api_key = embeddings_api_key
        self.embeddings_path = embeddings_path


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
        self._writeback_client: OWUIClient | None = None
        self._llm: LLMProviderClient | None = None
        self._embeddings: EmbeddingProviderClient | None = None
        self._llm_throttle: HttpThrottle | None = None
        self._embeddings_throttle: HttpThrottle | None = None
        self._writeback_throttle: HttpThrottle | None = None
        self._started = False

    async def start(self, *, writeback_token: str | None = None) -> None:
        """Bring the Coordinator's adapter clients online.

        ``writeback_token``, when provided, instantiates a *second*
        OWUIClient bound to a static admin token. Only the OpenAPI Tool
        Server entrypoint needs this — it POSTs to OWUI's per-message
        ``/event`` endpoint, which admin-bypasses chat ownership. Other
        entrypoints leave the kwarg unset and the second client is
        never created.
        """
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if not self._config.base_url:
                raise ValueError(
                    "RuntimeConfig.base_url is required (OWUI base URL for "
                    "retrieval/files/KB/chats)"
                )
            if not self._config.llm_base_url:
                raise ValueError(
                    "RuntimeConfig.llm_base_url is required. Set DR_LLM_BASE_URL "
                    "or valves.llm.base_url to the OpenAI-compatible chat provider base."
                )
            if not self._config.llm_api_key:
                raise ValueError(
                    "RuntimeConfig.llm_api_key is required. Set DR_LLM_API_KEY "
                    "or valves.llm.api_key to the chat provider bearer token."
                )
            if not self._config.embeddings_base_url:
                raise ValueError(
                    "RuntimeConfig.embeddings_base_url is required. Set "
                    "DR_EMBEDDINGS_BASE_URL or valves.embeddings.base_url to the "
                    "OpenAI-compatible embedding provider base."
                )
            if not self._config.embeddings_api_key:
                raise ValueError(
                    "RuntimeConfig.embeddings_api_key is required. Set "
                    "DR_EMBEDDINGS_API_KEY or valves.embeddings.api_key to the "
                    "embedding provider bearer token."
                )
            logger.info("Deep Research version=%s", _dr_version())
            # Log presence-only for API keys: confirming a key was loaded is
            # sufficient for operators, and "set"/"unset" keeps any portion of
            # the credential out of log archives.
            logger.info(
                "Coordinator starting: owui_base=%s llm_base=%s llm_chat=%s "
                "llm_key=%s embeddings_base=%s embeddings_path=%s embeddings_key=%s "
                "data_dir=%s llm_concurrency=%d embedding_concurrency=%d "
                "http_timeout=%ds http_max_retries=%d",
                self._config.base_url,
                self._config.llm_base_url,
                self._config.llm_chat_path,
                "set" if self._config.llm_api_key else "unset",
                self._config.embeddings_base_url,
                self._config.embeddings_path,
                "set" if self._config.embeddings_api_key else "unset",
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
                search_semaphore=asyncio.Semaphore(self._valves.web.search_concurrency),
                fetch_semaphore=asyncio.Semaphore(self._valves.web.fetch_concurrency),
            )
            if writeback_token:
                # Writeback path uses a static admin token to POST OWUI's
                # per-message ``/event`` endpoint on behalf of arbitrary
                # chats — only viable because that endpoint admin-bypasses
                # the chat-owner filter. A separate client keeps the static
                # admin token isolated from the per-request user-token path.
                # The throttle prevents a drain burst (status pills,
                # citations, final report, iframe replaces) and KB ingestion
                # uploads from hammering OWUI and its downstream embedding
                # pipeline.
                wb_t = self._valves.writeback_throttle
                self._writeback_throttle = HttpThrottle(
                    label="owui_writeback",
                    max_rps=wb_t.max_requests_per_second,
                    min_interval_ms=wb_t.min_interval_ms,
                    max_delay_seconds=wb_t.max_delay_seconds,
                )
                self._writeback_client = OWUIClient(
                    base_url=self._config.base_url,
                    token_provider=StaticToken(writeback_token),
                    timeout_seconds=self._valves.advanced.http_timeout_seconds,
                    max_retries=self._valves.advanced.http_max_retries,
                    throttle=self._writeback_throttle,
                )
            llm_t = self._valves.llm_throttle
            emb_t = self._valves.embeddings_throttle
            self._llm_throttle = HttpThrottle(
                label="llm",
                max_rps=llm_t.max_requests_per_second,
                min_interval_ms=llm_t.min_interval_ms,
                max_delay_seconds=llm_t.max_delay_seconds,
            )
            self._embeddings_throttle = HttpThrottle(
                label="embeddings",
                max_rps=emb_t.max_requests_per_second,
                min_interval_ms=emb_t.min_interval_ms,
                max_delay_seconds=emb_t.max_delay_seconds,
            )
            self._llm = LLMProviderClient(
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
                chat_path=self._config.llm_chat_path,
                timeout_seconds=self._valves.advanced.http_timeout_seconds,
                max_retries=llm_t.max_retries,
                llm_semaphore=asyncio.Semaphore(self._valves.advanced.llm_concurrency),
                throttle=self._llm_throttle,
                base_delay_seconds=llm_t.base_delay_seconds,
                max_delay_seconds=llm_t.max_delay_seconds,
            )
            self._embeddings = EmbeddingProviderClient(
                base_url=self._config.embeddings_base_url,
                api_key=self._config.embeddings_api_key,
                embeddings_path=self._config.embeddings_path,
                timeout_seconds=self._valves.advanced.http_timeout_seconds,
                max_retries=emb_t.max_retries,
                embedding_semaphore=asyncio.Semaphore(self._valves.advanced.embedding_concurrency),
                throttle=self._embeddings_throttle,
                base_delay_seconds=emb_t.base_delay_seconds,
                max_delay_seconds=emb_t.max_delay_seconds,
                batch_max_inputs=emb_t.batch_max_inputs,
            )
            await self._client.start()
            if self._writeback_client is not None:
                await self._writeback_client.start()
            await self._llm.start()
            await self._embeddings.start()
            self._started = True
            logger.info(
                "Coordinator started writeback_client=%s",
                "set" if self._writeback_client is not None else "unset",
            )

    @property
    def state_manager(self) -> ResearchStateManager:
        """Process-shared per-conversation state. Used by the OpenAPI runner
        to peek at ``waiting_for_outline_feedback`` between two
        ``run()`` invocations on the same conversation_id."""
        return self._state_manager

    @property
    def writeback_client(self) -> OWUIClient | None:
        """OWUIClient bound to a static admin token, or None if
        ``writeback_token`` was not passed to :meth:`start`.

        Used by the OpenAPI Tool Server's outbox worker to POST events
        to OWUI's per-message ``/event`` endpoint. Only that endpoint
        admin-bypasses the chat-owner filter, so the admin token isn't
        useful for general OWUI calls (use ``self._client`` for those).
        """
        return self._writeback_client

    async def close(self) -> None:
        logger.info("Coordinator shutting down")
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._writeback_client is not None:
            await self._writeback_client.close()
            self._writeback_client = None
        if self._llm is not None:
            await self._llm.close()
            self._llm = None
        if self._embeddings is not None:
            await self._embeddings.close()
            self._embeddings = None
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
        cancellation_token: Any = None,
        target_message_id: str | None = None,
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
            ctx = await self._build_context(
                user, conversation_id, chat_id, token, prompt, history, sink,
                cancellation_token=cancellation_token,
                target_message_id=target_message_id,
            )
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
                await self._emit_degraded_warnings(ctx)
                await self._emit_throttle_diagnostics(ctx)
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

    async def prewarm_vocabulary(
        self,
        *,
        user: RunUser,
        conversation_id: str,
        chat_id: str | None,
        token: BearerTokenProvider,
        sink: Sink,
        target_message_id: str | None = None,
    ) -> None:
        """Warm the process-shared vocabulary embedding cache.

        Spawned by the OpenAPI runner at ``AWAITING_OUTLINE_FEEDBACK`` so
        the ~10-minute cold-cache embedding load happens during the
        otherwise-idle gate-wait window, not during the user-visible
        ``generate_replacement_topics`` step on feedback resume. The work
        serialises on ``semantics.vocabulary._vocab_emb_load_lock``: if
        the user submits feedback before this finishes, the engine's own
        ``load_vocabulary_embeddings`` call waits on the same lock and
        reuses the partial work — no duplicate batches.

        Progress is emitted to ``sink`` via ``StatusEvent`` (see
        ``semantics.vocabulary._emit_vocab_progress``) so the iframe
        revision keeps advancing.
        """
        from deep_research.semantics.vocabulary import load_vocabulary_embeddings

        bearer = await token.get_token()
        token_handle = set_current_token(bearer)
        log_handle = None
        try:
            ctx = await self._build_context(
                user, conversation_id, chat_id, token, "", [], sink,
                target_message_id=target_message_id,
            )
            log_handle = set_log_context(
                conversation_id=ctx.conversation_id,
                chat_id=ctx.chat_id or "-",
                run_id=ctx.run_id,
                request_id=ctx.request_id,
            )
            await ctx.events.start()
            try:
                await load_vocabulary_embeddings(ctx)
            finally:
                await ctx.events.stop()
        finally:
            if log_handle is not None:
                reset_log_context(log_handle)
            reset_current_token(token_handle)

    async def _emit_degraded_warnings(self, ctx: RunContext) -> None:
        """Emit a one-shot warning the first time each throttle goes degraded.

        Called periodically (currently from the diagnostics-emit path); flag
        guards on ``ctx`` keep it at most once per side per run.
        """
        emb = ctx.embeddings_diagnostics
        llm = ctx.llm_diagnostics
        if emb is not None and emb.degraded and not ctx.embeddings_degraded_warned:
            ctx.embeddings_degraded_warned = True
            await ctx.events.emit(
                StatusEvent(
                    description=(
                        "Degraded mode: embedding rate limits detected — "
                        "dimension tracking and per-result similarity scoring "
                        "are temporarily reduced."
                    ),
                    level="warning",
                    done=False,
                )
            )
        if llm is not None and llm.degraded and not ctx.llm_degraded_warned:
            ctx.llm_degraded_warned = True
            await ctx.events.emit(
                StatusEvent(
                    description="Degraded mode: LLM provider rate limits detected.",
                    level="warning",
                    done=False,
                )
            )

    async def _emit_throttle_diagnostics(self, ctx: RunContext) -> None:
        """Emit a single end-of-run StatusEvent summarising both throttles.

        Surfaces attempts / retries / 429s / degraded activations so an
        operator can spot quota pressure without scraping logs. Best-effort:
        any failure here must not mask the original phase outcome.
        """
        try:
            lines: list[str] = []
            emb = ctx.embeddings_diagnostics
            llm = ctx.llm_diagnostics
            if emb is not None:
                lines.append(emb.snapshot().format_line())
            if llm is not None:
                lines.append(llm.snapshot().format_line())
            if not lines:
                return
            await ctx.events.emit(
                StatusEvent(
                    description="Throttle diagnostics: " + " | ".join(lines),
                    level="info",
                    done=False,
                )
            )
        except Exception:
            logger.exception("Failed to emit throttle diagnostics")

    async def _build_context(
        self,
        user: RunUser,
        conversation_id: str,
        chat_id: str | None,
        token: BearerTokenProvider,
        prompt: str,
        history: list[ChatMessage],
        sink: Sink,
        *,
        cancellation_token: Any = None,
        target_message_id: str | None = None,
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
            llm=self._llm,
            embeddings=self._embeddings,
            events=event_bus,
            caches=self._shared_caches,
            state=self._state_manager,
            executor=self._executor,
            mode=ResearchMode.FRESH,
            started_at=time.time(),
            research_date=datetime.now().strftime("%Y-%m-%d"),
            prompt=prompt,
            history=history,
            embeddings_diagnostics=(
                self._embeddings_throttle.diagnostics()
                if self._embeddings_throttle is not None
                else None
            ),
            llm_diagnostics=(
                self._llm_throttle.diagnostics()
                if self._llm_throttle is not None
                else None
            ),
            cancellation_token=cancellation_token,
            target_message_id=target_message_id,
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

        # Skip initial_queries on a feedback resume: outline_feedback has
        # already produced the finalized outline this turn. Re-running
        # initial_queries would regenerate the outline from scratch and
        # re-arm the gate, looping the user back to the topic-selection
        # prompt instead of starting the main research cycles.
        if not phase_state.get("outline_finalized"):
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
