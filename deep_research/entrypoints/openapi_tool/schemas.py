"""Request/response schemas for the OpenAPI Tool Server.

Phase 1 of the rewrite. The old `POST /research`, the `_jobs` dict,
and the corresponding models (`ResearchRequest`, `ResearchResponse`,
`Citation`, `ResearchMetadata`, `ResearchJobAccepted`, `JobProgress`,
`ResearchJobStatus`) were deleted in favour of the four
job-lifecycle endpoints plus two live-view endpoints documented in
the plan.

`StartResearchResponse.user_facing_instruction` is the Phase 1
mitigation for the LLM-skips-the-topic-list bug. Its description is
the OpenAPI prompt the LLM sees; the default value is the literal
text the LLM is told to emit. Phase 2 makes the field a no-op
(the topic list lands directly in chat content via `replace`).
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryMessage(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "I'm looking into transformer architectures.",
            }
        }
    )

    role: Literal["user", "assistant", "system"] = Field(
        description="Conversation role of this prior turn.",
    )
    content: str = Field(
        description="The verbatim text content of this prior turn.",
    )


class StartResearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Compare Mamba and Transformer architectures for long-context reasoning.",
                "user_id": "alice",
                "user_name": "Alice",
                "history": [],
            }
        }
    )

    prompt: str = Field(
        description=(
            "The research question or topic to investigate. Should be a"
            " self-contained natural-language description of what the user"
            " wants to learn; the engine plans its own sub-queries."
        ),
        min_length=1,
    )
    user_id: str = Field(
        default="api_user",
        description="Stable identifier for the requesting user.",
    )
    user_name: str = Field(
        default="API User",
        description="Human-readable user name.",
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description=(
            "Prior conversation turns. The last user turn should NOT be"
            " repeated here; pass that as `prompt`."
        ),
    )


class StartResearchResponse(BaseModel):
    """Tool-call response handed back to the LLM after a successful start.

    The fields are designed to be self-explanatory so the LLM follows
    the next step without paraphrasing or skipping the topic list.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "status": "running",
                "next_action": "await_user_selection",
                "user_facing_instruction": (
                    "I've started the preliminary research. When the topic "
                    "list is ready you'll see it appear right here — reply "
                    "with `/k <numbers>` to keep specific topics, "
                    "`/r <numbers>` to remove specific topics, or "
                    "`/continue` to research all topics as-is. A live "
                    "progress view will appear once you've chosen which "
                    "topics to research. You can cancel at any time by "
                    "replying `/q` or `/quit`."
                ),
            }
        }
    )

    job_id: str = Field(
        description="Identifier for the freshly-started job; pass it on follow-up calls.",
    )
    status: Literal["running"] = Field(
        default="running",
        description="Sentinel — the job has been accepted and is in flight.",
    )
    next_action: Literal["await_user_selection"] = Field(
        default="await_user_selection",
        description=(
            "What the LLM should expect from the user next. Currently the"
            " only path: wait for the user's topic-selection reply."
        ),
    )
    user_facing_instruction: str = Field(
        description=(
            "Verbatim instruction the LLM MUST emit to the user. "
            "Do not paraphrase. Do not omit. The tool has already "
            "started running and a live progress iframe is being "
            "attached to your assistant message; this string tells "
            "the user what to do next."
        ),
        default=(
            "I've started the preliminary research. When the topic list "
            "is ready you'll see it appear right here — reply with "
            "`/k <numbers>` to keep specific topics, `/r <numbers>` to "
            "remove specific topics, or `/continue` to research all "
            "topics as-is. A live progress view will appear once you've "
            "chosen which topics to research. You can cancel at any time "
            "by replying `/q` or `/quit`."
        ),
    )


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"selection": "/k 1,3,5"}
        }
    )

    selection: str = Field(
        description=(
            "User's reply, forwarded verbatim. The server parses"
            " /k|/keep <list>, /r|/remove <list>, /continue (or /c),"
            " or freeform natural-language feedback."
        ),
    )


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "status": "running",
                "next_phase": "researching",
            }
        }
    )

    job_id: str = Field(description="Job identifier echoed from the feedback call.")
    status: Literal["running"] = Field(
        default="running",
        description="Sentinel — the engine has resumed after the outline-feedback gate.",
    )
    next_phase: Literal["researching", "drafting", "finalizing"] = Field(
        description="Which phase the engine is moving into next.",
    )


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "phase": "researching",
                "revision": 12,
                "progress": {"cycle": 2, "max_cycles": 5},
                "report_markdown": None,
                "error": None,
            }
        }
    )

    job_id: str = Field(description="Job identifier echoed from the status call.")
    phase: str = Field(
        description="Current job phase (queued|bootstrapping|outlining|awaiting_outline_feedback|researching|drafting|finalizing|completed|failed|cancelled).",
    )
    revision: int = Field(
        description="Monotonically-incrementing revision for cache-busting and polling.",
    )
    progress: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of in-flight progress info (categories, token counts).",
    )
    report_markdown: str | None = Field(
        default=None,
        description="The full Markdown report when phase == 'completed'. Null otherwise.",
    )
    error: str | None = Field(
        default=None,
        description="Error description when phase == 'failed'. Null otherwise.",
    )


class CancelResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "status": "cancel_requested",
            }
        }
    )

    job_id: str = Field(description="Job identifier echoed from the cancel call.")
    status: Literal["cancel_requested", "already_terminal"] = Field(
        description=(
            "`cancel_requested` — the engine will bail at the next phase boundary."
            " `already_terminal` — the job is already finished and cannot be cancelled."
        ),
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "code": "already_running",
                "message": "Active job exists for this chat: 9c8b7a6f-...",
            }
        }
    )

    status: Literal["error"] = "error"
    code: str = Field(
        description=(
            "Machine-readable error code: `unknown_job`, `already_running`,"
            " `not_awaiting_feedback`, `forbidden`, `internal_error`."
        ),
    )
    message: str = Field(description="Human-readable error description.")


class LiveViewSnapshot(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "phase": "researching",
                "revision": 12,
                "progress": {"cycle": 2, "max_cycles": 5},
                "completed": False,
            }
        }
    )

    job_id: str = Field(description="Job identifier.")
    phase: str = Field(description="Current job phase.")
    revision: int = Field(
        description="Monotonically-incrementing revision. The iframe polls with `since_version`.",
    )
    progress: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of in-flight progress info.",
    )
    completed: bool = Field(
        description="True when the job has reached a terminal phase.",
    )
