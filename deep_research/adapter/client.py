import asyncio
import json
import logging
import time
from typing import Any

import httpx

from deep_research.adapter.auth import BearerTokenProvider
from deep_research.adapter.models import (
    FileUploadResponse,
    KBResponse,
    ProcessFileResponse,
    ProcessWebResponse,
    QueryCollectionResponse,
)
from deep_research.adapter.retry import with_retry
from deep_research.adapter.throttle import HttpThrottle

logger = logging.getLogger("deep_research.adapter.client")


PERSISTED_EVENT_TYPES: frozenset[str] = frozenset(
    {"status", "message", "replace", "embeds", "files", "source", "citation"}
)


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _body_model(json_body: Any) -> str:
    if isinstance(json_body, dict):
        m = json_body.get("model")
        if isinstance(m, str) and m:
            return m
    return "-"


def _body_keys(json_body: Any) -> list[str]:
    if isinstance(json_body, dict):
        return list(json_body.keys())
    return []


class AdapterError(Exception):
    def __init__(
        self,
        message: str,
        status: int = 0,
        headers: "dict[str, str] | None" = None,
    ) -> None:
        self.status = status
        # Carries upstream response headers so retry/backoff logic can read
        # ``Retry-After``. Mapping-shaped so ``extract_retry_after_seconds``
        # picks it up via duck-typing.
        self.headers = headers or {}
        super().__init__(message)


class OWUIClient:
    def __init__(
        self,
        *,
        base_url: str,
        token_provider: BearerTokenProvider,
        timeout_seconds: int = 600,
        max_retries: int = 3,
        search_semaphore: asyncio.Semaphore | None = None,
        fetch_semaphore: asyncio.Semaphore | None = None,
        throttle: HttpThrottle | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._search_sem = search_semaphore or asyncio.Semaphore(2)
        self._fetch_sem = fetch_semaphore or asyncio.Semaphore(4)
        self._throttle = throttle
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is not None:
            return
        limits = httpx.Limits(
            max_connections=64,
            max_keepalive_connections=32,
        )
        timeout = httpx.Timeout(
            connect=30,
            read=self._timeout,
            write=30,
            pool=60,
        )
        self._client = httpx.AsyncClient(
            http2=True,
            limits=limits,
            timeout=timeout,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise AdapterError("OWUIClient not started; call start() first")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict | None = None,
        headers: dict | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        token = await self._token_provider.get_token()

        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if headers:
            request_headers.update(headers)

        model = _body_model(json_body)
        body_keys = _body_keys(json_body)

        async def _do() -> Any:
            sem = semaphore
            if sem is None:
                sem = self._fetch_sem

            if self._throttle is not None:
                await self._throttle.acquire()
                self._throttle.record_attempt()

            async with sem:
                logger.debug(
                    "HTTP %s %s body_keys=%s model=%s",
                    method,
                    path,
                    body_keys,
                    model,
                )
                t0 = time.monotonic()
                resp = await self.client.request(
                    method,
                    url,
                    json=json_body,
                    params=params or {},
                    headers=request_headers,
                )
                elapsed = time.monotonic() - t0
                if resp.status_code >= 400:
                    try:
                        text = resp.text
                    except Exception:
                        text = ""
                    log_fn = logger.error if resp.status_code >= 500 else logger.warning
                    log_fn(
                        "HTTP %s %s -> %d model=%s elapsed_s=%.2f body=%s",
                        method,
                        path,
                        resp.status_code,
                        model,
                        elapsed,
                        _truncate(text, 500),
                    )
                    raise AdapterError(
                        f"{method} {path} -> {resp.status_code}: {text[:500]}",
                        status=resp.status_code,
                        headers=dict(resp.headers),
                    )
                logger.debug(
                    "HTTP %s %s -> %d in %.2fs",
                    method,
                    path,
                    resp.status_code,
                    elapsed,
                )
                if not resp.content:
                    return None
                try:
                    return resp.json()
                except json.JSONDecodeError as e:
                    logger.error(
                        "HTTP %s %s -> %d non-JSON body: %s",
                        method,
                        path,
                        resp.status_code,
                        _truncate(resp.text, 200),
                    )
                    raise AdapterError(
                        f"{method} {path} -> 200 but body is not JSON: "
                        f"{resp.text[:200]}",
                        status=resp.status_code,
                    ) from e

        def _on_transient(exc: BaseException, _attempt: int, reason: str) -> None:
            if self._throttle is None:
                return
            if "http_status=429" in reason:
                self._throttle.record_429(exhausted=False)
            self._throttle.record_retry()

        def _on_exhausted(exc: BaseException, _attempt: int, reason: str) -> None:
            if self._throttle is None:
                return
            if "http_status=429" in reason:
                self._throttle.record_429(exhausted=True)

        result = await with_retry(
            _do,
            max_retries=self._max_retries,
            label=f"{method} {path}",
            on_transient=_on_transient,
            on_exhausted=_on_exhausted,
        )
        if self._throttle is not None:
            self._throttle.record_success()
        return result

    # ---- Models (OWUI connectivity probe) ----

    async def list_models(self) -> list:
        """Lightweight OWUI connectivity probe used by owui_extraction_available.

        Returns the raw list of model dicts from OWUI's /api/v1/models/list.
        Use ctx.llm.list_models() (LLMProviderClient) for actual model metadata
        such as context-window sizes; this method is only for liveness checks.
        """
        resp = await self._request("GET", "/api/v1/models/list")
        if isinstance(resp, dict):
            return resp.get("data") or []
        if isinstance(resp, list):
            return resp
        return []

    # ---- Session user (admin probe) ----

    async def get_session_user(self) -> dict:
        """Return the OWUI session-user object for the API token.

        Calls ``GET /api/v1/auths/``, which uses ``Depends(get_current_user)``
        and admits any valid token. The response JSON includes ``role``, so
        callers can probe admin status by reading ``role == 'admin'``
        directly instead of inferring it from a 401/403 on an admin-only
        endpoint. Used by the OpenAPI Tool Server's startup config audit.
        """
        resp = await self._request("GET", "/api/v1/auths/")
        if not isinstance(resp, dict):
            raise AdapterError(
                f"GET /api/v1/auths/ returned non-dict body "
                f"({type(resp).__name__}); cannot determine role",
                status=200,
            )
        return resp

    # ---- Web ----

    async def web_search(self, queries: list[str]) -> dict:
        return await self._request(
            "POST",
            "/api/v1/retrieval/process/web/search",
            json_body={"queries": queries},
            semaphore=self._search_sem,
        )

    async def process_web_url(self, url: str, process: bool = False) -> ProcessWebResponse:
        data = await self._request(
            "POST",
            "/api/v1/retrieval/process/web",
            json_body={"url": url},
            params={"process": "true" if process else "false"},
            semaphore=self._fetch_sem,
        )
        return ProcessWebResponse.model_validate(data)

    async def process_file(
        self, file_id: str, collection_name: str | None = None
    ) -> ProcessFileResponse:
        body = {"file_id": file_id}
        if collection_name is not None:
            body["collection_name"] = collection_name
        data = await self._request(
            "POST",
            "/api/v1/retrieval/process/file",
            json_body=body,
            semaphore=self._fetch_sem,
        )
        return ProcessFileResponse.model_validate(data)

    # ---- Files ----

    async def upload_file(
        self, content: bytes, filename: str, process: bool = True
    ) -> FileUploadResponse:
        url = f"{self._base_url}/api/v1/files/"
        token = await self._token_provider.get_token()
        params = {"process": "true" if process else "false"}

        async def _do() -> Any:
            if self._throttle is not None:
                await self._throttle.acquire()
                self._throttle.record_attempt()
            async with self._fetch_sem:
                resp = await self.client.post(
                    url,
                    params=params,
                    headers={
                        "Authorization": f"Bearer {token}",
                    },
                    files={"file": (filename, content, "application/octet-stream")},
                )
                if resp.status_code >= 400:
                    raise AdapterError(
                        f"POST /api/v1/files/ -> {resp.status_code}",
                        status=resp.status_code,
                        headers=dict(resp.headers),
                    )
                return resp.json()

        def _on_transient(exc: BaseException, _attempt: int, reason: str) -> None:
            if self._throttle is None:
                return
            if "http_status=429" in reason:
                self._throttle.record_429(exhausted=False)
            self._throttle.record_retry()

        def _on_exhausted(exc: BaseException, _attempt: int, reason: str) -> None:
            if self._throttle is None:
                return
            if "http_status=429" in reason:
                self._throttle.record_429(exhausted=True)

        data = await with_retry(
            _do,
            max_retries=self._max_retries,
            label="upload_file",
            on_transient=_on_transient,
            on_exhausted=_on_exhausted,
        )
        if self._throttle is not None:
            self._throttle.record_success()
        return FileUploadResponse.model_validate(data)

    async def get_file(self, file_id: str) -> FileUploadResponse:
        data = await self._request("GET", f"/api/v1/files/{file_id}")
        return FileUploadResponse.model_validate(data)

    # ---- KB ----

    async def create_kb(self, name: str, description: str | None = None) -> KBResponse:
        body = {"name": name}
        if description is not None:
            body["description"] = description
        data = await self._request("POST", "/api/v1/knowledge/create", json_body=body)
        return KBResponse.model_validate(data)

    async def get_kb(self, kb_id: str) -> KBResponse:
        data = await self._request("GET", f"/api/v1/knowledge/{kb_id}")
        return KBResponse.model_validate(data)

    async def add_file_to_kb(
        self, kb_id: str, file_id: str, collection_name: str | None = None
    ) -> bool:
        body = {"file_id": file_id}
        if collection_name is not None:
            body["collection_name"] = collection_name
        data = await self._request(
            "POST", f"/api/v1/knowledge/{kb_id}/file/add", json_body=body
        )
        return isinstance(data, dict) and data.get("status", True) is not False

    async def query_collection(
        self,
        collection_names: list[str],
        query: str,
        k: int = 10,
        hybrid: bool = False,
    ) -> QueryCollectionResponse:
        body = {
            "collection_names": collection_names,
            "query": query,
            "k": k,
            "hybrid": hybrid,
        }
        data = await self._request(
            "POST", "/api/v1/retrieval/query/collection", json_body=body
        )
        return QueryCollectionResponse.model_validate(data)

    # ---- Chat persistence ----

    async def get_chat(self, chat_id: str) -> dict | None:
        try:
            return await self._request("GET", f"/api/v1/chats/{chat_id}")
        except AdapterError as e:
            if e.status == 404:
                return None
            raise

    async def update_chat(self, chat_id: str, chat: dict) -> bool:
        data = await self._request(
            "POST", f"/api/v1/chats/{chat_id}", json_body={"chat": chat}
        )
        return isinstance(data, dict) and data.get("status", True) is not False

    # ---- Per-message persisted events ----

    async def post_message_event(
        self,
        chat_id: str,
        message_id: str,
        event_type: str,
        data: dict,
    ) -> None:
        """POST a persisted event to a specific chat message.

        OWUI's per-message endpoint
        ``POST /api/v1/chats/{chat_id}/messages/{message_id}/event``
        broadcasts to live WebSocket clients AND persists the event to
        the message row when ``event_type`` is one of the documented
        short names. Long-name aliases (``chat:message:embeds`` etc.)
        broadcast but do NOT persist. Unlike ``GET /api/v1/chats/{id}``,
        this endpoint admin-bypasses chat ownership so a server-side
        admin token can write to chats owned by other users — the
        mechanism the OpenAPI Tool Server uses to land content directly
        in the assistant message.

        Args:
            chat_id, message_id: Forwarded by OWUI in
                ``X-OpenWebUI-Chat-Id`` / ``X-OpenWebUI-Message-Id``
                headers when ``ENABLE_FORWARD_USER_INFO_HEADERS=true``.
            event_type: Must be one of ``PERSISTED_EVENT_TYPES``.
            data: Payload per OWUI's event schema; varies by event type.

        Raises:
            ValueError: ``event_type`` not in ``PERSISTED_EVENT_TYPES``.
            AdapterError: HTTP failure after retry exhaustion.
        """
        if event_type not in PERSISTED_EVENT_TYPES:
            raise ValueError(
                f"event_type {event_type!r} is not persisted by OWUI; "
                f"allowed: {sorted(PERSISTED_EVENT_TYPES)}"
            )
        path = f"/api/v1/chats/{chat_id}/messages/{message_id}/event"
        await self._request(
            "POST",
            path,
            json_body={"type": event_type, "data": data},
        )
