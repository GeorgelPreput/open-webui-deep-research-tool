import httpx

_TRANSIENT_HTTPX_TYPES: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.NetworkError,
)

_TRANSIENT_COMPLETION_CODES = frozenset({429, 500, 502, 503, 504})
_OWUI_TRANSIENT_DETAIL = "open webui: server connection error"
_TRANSIENT_FALLBACK_PHRASES = (
    "server disconnected",
    "server connection error",
    "connection reset",
    "connection refused",
)


def classify_transient_completion_error(e: BaseException) -> str | None:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status in _TRANSIENT_COMPLETION_CODES:
            return f"http_status={status}"
        return None
    if isinstance(e, _TRANSIENT_HTTPX_TYPES):
        return f"httpx_type={type(e).__name__}"
    # Duck-typed: any exception carrying a `status` attribute (e.g. the
    # adapter layer's AdapterError) is inspected for a transient code.
    duck_status = getattr(e, "status", None)
    if isinstance(duck_status, int) and duck_status in _TRANSIENT_COMPLETION_CODES:
        return f"http_status={duck_status}"
    err_str = str(e).lower()
    for phrase in _TRANSIENT_FALLBACK_PHRASES:
        if phrase in err_str:
            return f"fallback_phrase={phrase!r}"
    return None


class CompletionError(RuntimeError):
    def __init__(
        self,
        model: str,
        original: BaseException,
        attempts: int,
        transient_reason: str | None,
    ) -> None:
        self.model = model
        self.original = original
        self.attempts = attempts
        self.transient_reason = transient_reason
        kind = (
            f"transient (reason={transient_reason}) retries exhausted"
            if transient_reason
            else "non-transient"
        )
        super().__init__(
            f"Completion failed for model {model!r}: {kind} after "
            f"{attempts} attempt(s): {type(original).__name__}: {original}"
        )
