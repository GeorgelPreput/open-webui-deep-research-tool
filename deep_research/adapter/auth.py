import contextvars
from typing import Protocol


class BearerTokenProvider(Protocol):
    async def get_token(self) -> str: ...


class StaticToken:
    """Bearer token captured once at construction time.

    Covers every current call-site: server-side static API keys
    (OpenAPI/MCP/Pipeline runtimes) and per-request tokens lifted from
    the Authorization header in the OWUI Function shim. A dynamic
    rotating-token provider would implement BearerTokenProvider directly.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def get_token(self) -> str:
        return self._token


_current_token: contextvars.ContextVar[str] = contextvars.ContextVar(
    "deep_research_owui_token", default=""
)


class ContextTokenProvider:
    """Per-request bearer token read from a contextvar.

    Set the token at the start of a request via set_current_token(); the
    OWUIClient reads it on every outbound call. This lets us share one
    OWUIClient (and its connection pool) across many concurrent requests
    while still scoping the bearer per-call.
    """

    async def get_token(self) -> str:
        return _current_token.get()


def set_current_token(token: str) -> contextvars.Token:
    return _current_token.set(token)


def reset_current_token(reset_token: contextvars.Token) -> None:
    _current_token.reset(reset_token)
