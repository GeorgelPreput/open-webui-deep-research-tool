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
