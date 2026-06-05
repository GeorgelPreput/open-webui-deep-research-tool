from pydantic import BaseModel, Field

# Engineering knobs that aren't user-facing live in config.constants.
# Only fields that an OWUI admin or end user might reasonably want to tune
# appear on the Valves model below — see REFACTOR_PLAN.md A.3.


class ModelsValves(BaseModel):
    research_model: str = Field("gemma3:12b", description="Primary research LLM")
    synthesis_model: str = Field("gemma3:27b", description="Synthesis LLM (optional override)")
    quality_filter_model: str = Field("gemma3:4b", description="Relevance filter LLM")
    embedding_model: str = Field(
        "nomic-embed-text",
        description="Embedding model ID; must match an OWUI-registered embedding model",
    )

    research_context_window: int | None = Field(
        None, description="Override; None = auto-detect from /api/v1/models/list"
    )
    synthesis_context_window: int | None = Field(
        None, description="Override; None = auto-detect from /api/v1/models/list"
    )

    temperature: float = 0.7
    synthesis_temperature: float = 0.6


class CyclesValves(BaseModel):
    min_cycles: int = 10
    max_cycles: int = 15
    gap_exploration_weight: float = 0.4
    trajectory_momentum: float = 0.6
    followup_weight: float = 0.5


class WebValves(BaseModel):
    search_results_per_query: int = 3
    successful_results_per_query: int = 1
    extra_results_per_query: int = 3
    repeats_before_expansion: int = 3
    max_result_tokens: int = 4000
    domain_priority: str = ""
    content_priority: str = ""
    quality_filter_enabled: bool = True
    quality_similarity_threshold: float = 0.60
    fetch_concurrency: int = 4
    search_concurrency: int = 2


class CompressionValves(BaseModel):
    chunk_level: int = 2
    compression_level: int = 4
    stepped_synthesis_compression: bool = True


class PersistenceValves(BaseModel):
    export_research_data: bool = True
    interactive_research: bool = True
    user_preference_throughout: bool = True
    # Cap on KB source uploads per research cycle. 0 = unlimited (current behaviour).
    max_kb_uploads_per_cycle: int = 0
    # Min gap between consecutive KB uploads, in milliseconds. 0 = no delay.
    kb_upload_delay_ms: int = 0
    # When the embedding throttle has tripped degraded mode, skip KB ingestion
    # entirely until it recovers. Off by default so existing deployments keep
    # persisting; recommended for low-TPM dev keys where OWUI's own ingestion
    # competes for the same embedding quota.
    disable_during_degraded: bool = False
    disable_kb_persistence: bool = Field(
        default=False,
        description=(
            "Set true to skip uploading research sources and the final "
            "report to Open WebUI's knowledge base. Saves embedding "
            "tokens significantly. Trade-off: the research becomes "
            "ephemeral — rehydrating state on a follow-up turn won't "
            "work, and the engine can't answer post-report questions "
            "against the KB. The in-chat report still appears as "
            "Markdown content and remains there."
        ),
    )


class EventsValves(BaseModel):
    enable_progress_embed: bool = True
    flush_interval_ms: int = 400
    quiet_chat_mode: bool = True


class LoggingValves(BaseModel):
    level: str = "INFO"
    format: str = "text"
    include_tracebacks: bool = True


class LLMValves(BaseModel):
    base_url: str = Field("", description="OpenAI-compatible LLM base URL (required)")
    api_key: str = Field("", description="Bearer token for the LLM provider (required)")
    chat_path: str = Field(
        "/chat/completions",
        description="Chat completions path appended to base_url",
    )


class EmbeddingValves(BaseModel):
    base_url: str = Field(
        "",
        description="OpenAI-compatible base URL for the embedding model (required)",
    )
    api_key: str = Field(
        "",
        description="Bearer token for the embedding provider (required)",
    )
    embeddings_path: str = Field(
        "/embeddings",
        description="Embeddings path appended to base_url",
    )


class _ThrottleFieldsMixin(BaseModel):
    """Shared per-client throttle knobs.

    Both LLM and embedding clients share the same shape because the two
    providers exhibit the same constraints in practice (separate quotas,
    same 429 shape). Embedding adds ``batch_max_inputs`` on top.
    """
    # Token-bucket cap on dispatched HTTP calls. 0 disables the throttle.
    max_requests_per_second: float = 0.0
    # Minimum gap between dispatched calls, in milliseconds. 0 disables.
    min_interval_ms: int = 0
    # Retries on transient (incl. 429) errors. Overrides advanced.http_max_retries.
    max_retries: int = 5
    # Exponential-backoff base delay, in seconds. Honoured unless Retry-After
    # is present on the response.
    base_delay_seconds: float = 1.0
    # Backoff ceiling. Also feeds the degraded-mode cooldown derivation.
    max_delay_seconds: float = 60.0


class LLMThrottleValves(_ThrottleFieldsMixin):
    pass


class EmbeddingsThrottleValves(_ThrottleFieldsMixin):
    # Cap on the ``inputs`` array size in a single /embeddings POST. The
    # vocabulary loader is the main reason this exists (a 10k-word batch
    # explodes TPM); keep ≤ provider tolerance.
    batch_max_inputs: int = 64


class WritebackThrottleValves(_ThrottleFieldsMixin):
    """Throttle knobs for the OpenAPI Tool Server's writeback ``OWUIClient``.

    The writeback channel posts to OWUI's per-message ``/event`` endpoint with
    a static admin token and also calls ``upload_file`` for KB ingestion, which
    triggers OWUI's own embedding pipeline downstream. A research run emits
    dozens of events (status pills, citations, final report, iframe replaces)
    plus 1+ uploads per cycle; without a throttle, an outbox drain burst goes
    straight at OWUI and into the same embedding provider quota the engine is
    already pulling on. Defaults to ``max_rps=0`` (gate disabled) so existing
    deployments see no behaviour change unless operators tune it.
    """
    pass


class JobsValves(BaseModel):
    """OpenAPI Tool Server job-store and writeback knobs."""
    completed_retention_s: int = 30 * 24 * 3600
    failed_retention_s: int = 24 * 3600
    cleanup_interval_s: int = 3600
    sqlite_busy_timeout_ms: int = 5000
    # Phase 2 fields (declared now, used when the writeback outbox lands):
    writeback_enabled: bool = True
    outbox_poll_interval_ms: int = 250
    outbox_max_attempts: int = 10
    outbox_max_backoff_s: int = 60
    # Hard ceiling on server-supplied ``Retry-After`` for outbox writebacks.
    # Defaults to 10 minutes — generous because the server told us this value
    # and clamping it causes "retried too soon, throttled harder". Production
    # runs last 40–90 min so a 10-min deferral fits inside the run window.
    outbox_max_retry_after_s: int = 600


class AdvancedValves(BaseModel):
    query_weight: float = 0.5
    llm_concurrency: int = 4
    embedding_concurrency: int = 8
    executor_workers: int = 2
    http_timeout_seconds: int = 600
    http_max_retries: int = 3
    # Verify TLS certs on legacy PDF downloads. Disable only for trusted hosts
    # with self-signed certs; leaving it on is the secure default.
    pdf_legacy_tls_verify: bool = True


class Valves(BaseModel):
    enabled: bool = True
    # mypy treats `ModelsValves` as having required positional args because
    # the inner Field(default, description=…) form confuses the stubs; the
    # other group models have plain `= default` and accept default_factory
    # without complaint.
    models: ModelsValves = Field(default_factory=ModelsValves)  # type: ignore[arg-type]
    cycles: CyclesValves = Field(default_factory=CyclesValves)
    web: WebValves = Field(default_factory=WebValves)
    compression: CompressionValves = Field(default_factory=CompressionValves)
    persistence: PersistenceValves = Field(default_factory=PersistenceValves)
    events: EventsValves = Field(default_factory=EventsValves)
    jobs: JobsValves = Field(default_factory=JobsValves)
    advanced: AdvancedValves = Field(default_factory=AdvancedValves)
    logging: LoggingValves = Field(default_factory=LoggingValves)
    llm: LLMValves = Field(default_factory=LLMValves)  # type: ignore[arg-type]
    embeddings: EmbeddingValves = Field(default_factory=EmbeddingValves)  # type: ignore[arg-type]
    llm_throttle: LLMThrottleValves = Field(default_factory=LLMThrottleValves)
    embeddings_throttle: EmbeddingsThrottleValves = Field(
        default_factory=EmbeddingsThrottleValves
    )
    writeback_throttle: WritebackThrottleValves = Field(
        default_factory=WritebackThrottleValves
    )
