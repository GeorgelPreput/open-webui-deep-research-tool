import enum
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict


class BibliographyEntry(TypedDict):
    id: int
    title: str
    url: str


class BibliographyData(TypedDict):
    bibliography: list[BibliographyEntry]
    title_to_global_id: dict[str, int]
    url_to_global_id: dict[str, int]


class ResearchMode(enum.StrEnum):
    FRESH = "fresh"
    FOLLOW_UP = "follow_up"
    OUTLINE_FEEDBACK = "outline_feedback"
    POST_REPORT_QA = "post_report_qa"


@dataclass(slots=True, frozen=True)
class RunUser:
    id: str
    name: str
    email: str | None = None
    role: str = "user"


@dataclass(slots=True, frozen=True)
class ChatMessage:
    role: str
    content: str
    name: str | None = None


@dataclass(slots=True)
class Report:
    """Structured return type of Coordinator.run()."""

    content: str
    title: str = ""
    sources: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] = field(default_factory=dict)
    report_file_id: str | None = None
    conversation_id: str = ""

    def __str__(self) -> str:
        return self.content


@dataclass(slots=True)
class PersistenceGate:
    """Per-run KB-upload counter and timestamp.

    Tracks ``persist_selected_source`` activity within the current cycle so
    ``PersistenceValves.max_kb_uploads_per_cycle`` and
    ``PersistenceValves.kb_upload_delay_ms`` can throttle OWUI ingestion under
    embedding quota pressure. Reset by ``cycles.run_cycles`` at the start of
    each iteration.
    """

    uploads_this_cycle: int = 0
    last_upload_monotonic: float = 0.0

    def reset_cycle(self) -> None:
        self.uploads_this_cycle = 0


@dataclass(slots=True)
class RunContext:
    user: RunUser
    conversation_id: str
    chat_id: str | None
    request_id: str
    run_id: str
    valves: Any
    config: Any
    client: Any
    llm: Any
    embeddings: Any
    events: Any
    caches: Any
    state: Any
    executor: Any
    mode: ResearchMode
    started_at: float
    trajectory_accumulator: Any = None
    research_date: str = ""
    prompt: str = ""
    history: list = field(default_factory=list)
    # Per-call ephemeral dedupe sets for one-shot status emission during
    # synthesis (ported from pipe.py's _seen_subtopics / _seen_sections
    # ContextVars). Not persisted.
    seen_subtopics: set = field(default_factory=set)
    seen_sections: set = field(default_factory=set)
    # Per-client throttle diagnostics. Consumers read ``.degraded`` to
    # opportunistically skip embedding-heavy work; they read ``.snapshot()``
    # for end-of-run reporting. Filled in by Coordinator._build_context.
    embeddings_diagnostics: Any = None
    llm_diagnostics: Any = None
    persistence_gate: PersistenceGate = field(default_factory=PersistenceGate)
    # One-shot flags so a degraded-mode warning is emitted at most once per
    # run and per side (embeddings vs LLM).
    embeddings_degraded_warned: bool = False
    llm_degraded_warned: bool = False
