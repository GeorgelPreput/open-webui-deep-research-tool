import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from deep_research.adapter.auth import BearerTokenProvider
from deep_research.adapter.models import (
    FileUploadResponse,
    KBResponse,
    ModelInfo,
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


class OWUIClientError(Exception):
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
        llm_semaphore: asyncio.Semaphore | None = None,
        embedding_semaphore: asyncio.Semaphore | None = None,
        search_semaphore: asyncio.Semaphore | None = None,
        fetch_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_provider = token_provider
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._llm_sem = llm_semaphore or asyncio.Semaphore(4)
        self._embedding_sem = embedding_semaphore or asyncio.Semaphore(8)
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
            raise OWUIClientError("OWUIClient not started; call start() first")
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
                sem = self._llm_sem

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
                    raise OWUIClientError(
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
                    raise OWUIClientError(
                        f"{method} {path} -> 200 but body is not JSON: "
                        f"{resp.text[:200]}",
                        status=resp.status_code,
                    ) from e

        return await with_retry(
            _do,
            max_retries=self._max_retries,
            label=f"{method} {path}",
        )

    async def _request_stream(
        self,
        path: str,
        json_body: dict,
    ) -> AsyncIterator[str]:
        url = f"{self._base_url}{path}"
        token = await self._token_provider.get_token()
        model = _body_model(json_body)

        async def _stream() -> AsyncIterator[str]:
            logger.debug(
                "HTTP POST %s (stream) body_keys=%s model=%s",
                path,
                _body_keys(json_body),
                model,
            )
            async with self._llm_sem, self.client.stream(
                "POST",
                url,
                json=json_body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "text/event-stream",
                },
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    log_fn = logger.error if resp.status_code >= 500 else logger.warning
                    log_fn(
                        "HTTP POST %s (stream) -> %d model=%s body=%s",
                        path,
                        resp.status_code,
                        model,
                        _truncate(body, 500),
                    )
                    raise OWUIClientError(
                        f"POST {path} -> {resp.status_code}: {body[:500]}",
                        status=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

        attempt = 0
        last_exc: Exception | None = None
        yielded_any = False
        while attempt <= self._max_retries:
            try:
                async for chunk in _stream():
                    yielded_any = True
                    yield chunk
                return
            except OWUIClientError as e:
                # Once any chunk has reached the caller, retrying would re-yield
                # from the start and duplicate already-emitted content.
                if (
                    not yielded_any
                    and e.status in {429, 502, 503, 504}
                    and attempt < self._max_retries
                ):
                    last_exc = e
                    logger.warning(
                        "Stream retry POST %s: attempt=%d/%d status=%d yielded_any=False",
                        path,
                        attempt + 1,
                        self._max_retries,
                        e.status,
                    )
                    await asyncio.sleep(min(1.5 * (2**attempt), 30.0))
                    attempt += 1
                    continue
                raise
            except httpx.HTTPError as e:
                if not yielded_any and attempt < self._max_retries:
                    last_exc = e
                    logger.warning(
                        "Stream retry POST %s: attempt=%d/%d exc=%s yielded_any=False",
                        path,
                        attempt + 1,
                        self._max_retries,
                        type(e).__name__,
                    )
                    await asyncio.sleep(min(1.5 * (2**attempt), 30.0))
                    attempt += 1
                    continue
                raise OWUIClientError(f"stream failed: {e}") from e
        if last_exc:
            raise OWUIClientError(f"stream exhausted retries: {last_exc}") from last_exc

    # ---- LLM ----

    async def chat_completions(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        chat_id: str | None = None,
    ) -> dict:
        body = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        if chat_id is not None:
            body["chat_id"] = chat_id

        if not stream:
            return await self._request("POST", "/api/chat/completions", json_body=body)

        chunks: list[str] = []
        async for delta in self._request_stream("/api/chat/completions", body):
            chunks.append(delta)
        return {"choices": [{"message": {"content": "".join(chunks)}}]}

    async def stream_chat_completions(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        body = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            body["temperature"] = temperature
        async for delta in self._request_stream("/api/chat/completions", body):
            yield delta

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        resp = await self._request(
            "POST",
            "/api/embeddings",
            json_body={"model": model, "input": inputs},
            semaphore=self._embedding_sem,
        )
        data = (resp or {}).get("data") or []
        return [d.get("embedding") or [] for d in data]

    async def list_models(self, refresh: bool = False) -> list[ModelInfo]:
        resp = await self._request("GET", "/api/v1/models/list")
        items: list[dict] = []
        if isinstance(resp, dict):
            items = resp.get("data") or []
        elif isinstance(resp, list):
            items = resp
        models: list[ModelInfo] = []
        for item in items:
            meta = item.get("meta") or item.get("details") or {}
            ctx_window = meta.get("context_length") or meta.get("num_ctx") or None
            models.append(
                ModelInfo(
                    id=item.get("id") or item.get("name", ""),
                    name=item.get("name", ""),
                    context_window=ctx_window,
                    meta=meta,
                )
            )
        return models

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
                    raise OWUIClientError(
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
        except OWUIClientError as e:
            if e.status == 404:
                return None
            raise

    async def update_chat(self, chat_id: str, chat: dict) -> bool:
        data = await self._request(
            "POST", f"/api/v1/chats/{chat_id}", json_body={"chat": chat}
        )
        return isinstance(data, dict) and data.get("status", True) is not False
