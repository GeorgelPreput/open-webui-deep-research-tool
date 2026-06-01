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
    advanced: AdvancedValves = Field(default_factory=AdvancedValves)
    logging: LoggingValves = Field(default_factory=LoggingValves)
    llm: LLMValves = Field(default_factory=LLMValves)  # type: ignore[arg-type]
    embeddings: EmbeddingValves = Field(default_factory=EmbeddingValves)  # type: ignore[arg-type]
