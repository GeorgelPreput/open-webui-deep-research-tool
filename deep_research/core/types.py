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
class RunContext:
    user: RunUser
    conversation_id: str
    chat_id: str | None
    request_id: str
    run_id: str
    valves: Any
    config: Any
    client: Any
    events: Any
    caches: Any
    state: Any
    executor: Any
    mode: ResearchMode
    started_at: float
    trajectory_accumulator: Any = None
