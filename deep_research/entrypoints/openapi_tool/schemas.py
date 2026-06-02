from typing import Literal

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


class ResearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Compare Mamba and Transformer architectures for long-context reasoning.",
                "user_id": "alice",
                "user_name": "Alice",
            }
        }
    )

    prompt: str = Field(
        description=(
            "The research question or topic to investigate. Should be a"
            " self-contained natural-language description of what the user"
            " wants to learn; the engine will plan its own sub-queries."
        ),
        min_length=1,
    )
    user_id: str = Field(
        default="api_user",
        description="Stable identifier for the requesting user (used for inflight"
                    " deduplication and quota accounting).",
    )
    user_name: str = Field(
        default="API User",
        description="Human-readable user name (used in trajectory traces and logs).",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation identifier. When supplied, follow-up"
                    " calls on the same id reuse cached state (outline, KB).",
    )
    chat_id: str | None = Field(
        default=None,
        description="Optional Open WebUI chat id for cross-system correlation.",
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="Prior conversation turns. The last user turn should NOT be"
                    " repeated here; pass that as `prompt`.",
    )


class Citation(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "url": "https://arxiv.org/abs/2312.00752",
                "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
                "snippet": None,
            }
        }
    )

    id: int = Field(
        description="Sequential citation number that matches `[N]` markers in `report`.",
    )
    url: str = Field(description="Source URL.")
    title: str = Field(description="Source title (may be empty if unknown).")
    snippet: str | None = Field(
        default=None,
        description="Short excerpt from the source, when available.",
    )


class ResearchMetadata(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token_usage": {"prompt": 12345, "completion": 6789, "total": 19134},
                "elapsed_s": 124.7,
                "report_file_id": None,
            }
        }
    )

    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token counts aggregated across all LLM calls in the run.",
    )
    elapsed_s: float = Field(
        default=0.0,
        description="Wall-clock duration of the run, in seconds.",
    )
    report_file_id: str | None = Field(
        default=None,
        description="Identifier of the persisted report file, if persistence was enabled.",
    )


class ResearchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "report": "# Mamba vs Transformer\n\nMamba shows linear-time scaling [1]...",
                "title": "Mamba vs Transformer for Long-Context Reasoning",
                "citations": [
                    {
                        "id": 1,
                        "url": "https://arxiv.org/abs/2312.00752",
                        "title": "Mamba: Linear-Time Sequence Modeling",
                        "snippet": None,
                    }
                ],
                "conversation_id": "conv_abc",
                "metadata": {
                    "token_usage": {"prompt": 12345, "completion": 6789, "total": 19134},
                    "elapsed_s": 124.7,
                    "report_file_id": None,
                },
            }
        }
    )

    status: Literal["ok"] = Field(
        default="ok",
        description="Sentinel indicating a successful research run.",
    )
    report: str = Field(
        description="Final markdown report. Contains inline `[N]` citation markers"
                    " that index into `citations`. Surface this verbatim to the user;"
                    " do not paraphrase the citation numbers.",
    )
    title: str = Field(
        default="",
        description="Short human-readable title for the report.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ordered list of citations referenced by `report` via `[N]` markers.",
    )
    conversation_id: str = Field(
        description="Conversation identifier used for this run. Pass it back on"
                    " follow-up calls to reuse the same research state.",
    )
    metadata: ResearchMetadata = Field(
        default_factory=ResearchMetadata,
        description="Run-level metadata (token usage, duration, persistence info).",
    )


class ResearchErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "error",
                "code": "already_running",
                "message": "Research already running for conversation conv_abc",
            }
        }
    )

    status: Literal["error"] = "error"
    code: str = Field(
        description="Machine-readable error code (e.g. `already_running`,"
                    " `coordinator_unavailable`, `internal_error`).",
    )
    message: str = Field(description="Human-readable error description.")


class ResearchJobAccepted(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "status": "pending",
                "poll_url": "/research_jobs/9c8b7a6f-1234-4567-89ab-cdef01234567",
            }
        }
    )

    job_id: str = Field(description="Identifier to use when polling for completion.")
    status: Literal["pending"] = "pending"
    poll_url: str = Field(
        description="Relative URL to poll for job status and the eventual result.",
    )


class JobProgress(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "phase": "cycles",
                "message": "Cycle 2/5: refining queries",
            }
        }
    )

    phase: str = Field(
        default="",
        description="Current pipeline phase name (e.g. `planning`, `cycles`, `synthesize`).",
    )
    message: str = Field(
        default="",
        description="Latest human-readable status line from the engine.",
    )


class ResearchJobStatus(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "9c8b7a6f-1234-4567-89ab-cdef01234567",
                "status": "running",
                "result": None,
                "error": None,
                "progress": {"phase": "cycles", "message": "Cycle 2/5"},
            }
        }
    )

    job_id: str = Field(description="Job identifier echoed from the start call.")
    status: Literal["pending", "running", "completed", "failed"] = Field(
        description="Lifecycle state. Poll until `completed` or `failed`.",
    )
    result: ResearchResponse | None = Field(
        default=None,
        description="The full research response when `status == 'completed'`."
                    " Identical shape to `POST /research`. Null otherwise.",
    )
    error: ResearchErrorResponse | None = Field(
        default=None,
        description="Error details when `status == 'failed'`. Null otherwise.",
    )
    progress: JobProgress | None = Field(
        default=None,
        description="Latest progress snapshot while running.",
    )
