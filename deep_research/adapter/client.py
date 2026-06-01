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

logger = logging.getLogger("deep_research.adapter.client")


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
    def __init__(self, message: str, status: int = 0) -> None:
        self.status = status
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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._search_sem = search_semaphore or asyncio.Semaphore(2)
        self._fetch_sem = fetch_semaphore or asyncio.Semaphore(4)
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

        return await with_retry(
            _do,
            max_retries=self._max_retries,
            label=f"{method} {path}",
        )

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
                    )
                return resp.json()

        data = await with_retry(_do, max_retries=self._max_retries, label="upload_file")
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
