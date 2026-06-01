import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from deep_research.adapter.client import AdapterError, _body_keys, _body_model, _truncate
from deep_research.adapter.models import ModelInfo
from deep_research.adapter.retry import with_retry
from deep_research.adapter.throttle import JITTER_RATIO, RESPECT_RETRY_AFTER, HttpThrottle
from deep_research.core.errors import extract_retry_after_seconds  # used by _request_stream

logger = logging.getLogger("deep_research.adapter.llm_provider")


class _OpenAIHTTPBase:
    """Shared httpx wiring for OpenAI-compatible providers.

    Subclasses configure base_url + api_key + the single semaphore that gates
    their request profile (chat vs embeddings), and use ``_request`` /
    ``_request_stream`` to talk to the provider. Subclasses are responsible for
    their own public API surface (chat_completions, embeddings, ...).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        max_retries: int,
        semaphore: asyncio.Semaphore,
        throttle: HttpThrottle,
        base_delay_seconds: float,
        max_delay_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._sem = semaphore
        self._throttle = throttle
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
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
            raise AdapterError(
                f"{type(self).__name__} not started; call start() first"
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"

        request_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        model = _body_model(json_body)
        body_keys = _body_keys(json_body)

        async def _do() -> Any:
            await self._throttle.acquire()
            self._throttle.record_attempt()
            async with self._sem:
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
            if "http_status=429" in reason:
                self._throttle.record_429(exhausted=False)
            self._throttle.record_retry()

        def _on_exhausted(exc: BaseException, _attempt: int, reason: str) -> None:
            if "http_status=429" in reason:
                self._throttle.record_429(exhausted=True)

        result = await with_retry(
            _do,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            jitter=JITTER_RATIO,
            respect_retry_after=RESPECT_RETRY_AFTER,
            label=f"{method} {path}",
            on_transient=_on_transient,
            on_exhausted=_on_exhausted,
        )
        self._throttle.record_success()
        return result

    async def _request_stream(
        self,
        path: str,
        json_body: dict,
    ) -> AsyncIterator[str]:
        url = f"{self._base_url}{path}"
        model = _body_model(json_body)

        async def _stream() -> AsyncIterator[str]:
            await self._throttle.acquire()
            self._throttle.record_attempt()
            logger.debug(
                "HTTP POST %s (stream) body_keys=%s model=%s",
                path,
                _body_keys(json_body),
                model,
            )
            async with self._sem, self.client.stream(
                "POST",
                url,
                json=json_body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
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
                    raise AdapterError(
                        f"POST {path} -> {resp.status_code}: {body[:500]}",
                        status=resp.status_code,
                        headers=dict(resp.headers),
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
                self._throttle.record_success()
                return
            except AdapterError as e:
                if (
                    not yielded_any
                    and e.status in {429, 502, 503, 504}
                    and attempt < self._max_retries
                ):
                    last_exc = e
                    is_429 = e.status == 429
                    if is_429:
                        self._throttle.record_429(exhausted=False)
                    self._throttle.record_retry()
                    delay = self._compute_stream_retry_delay(attempt, e)
                    logger.warning(
                        "Stream retry POST %s: attempt=%d/%d status=%d delay=%.2fs yielded_any=False",
                        path,
                        attempt + 1,
                        self._max_retries,
                        e.status,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                if e.status == 429:
                    self._throttle.record_429(exhausted=True)
                raise
            except httpx.HTTPError as e:
                if not yielded_any and attempt < self._max_retries:
                    last_exc = e
                    self._throttle.record_retry()
                    delay = self._compute_stream_retry_delay(attempt, e)
                    logger.warning(
                        "Stream retry POST %s: attempt=%d/%d exc=%s delay=%.2fs yielded_any=False",
                        path,
                        attempt + 1,
                        self._max_retries,
                        type(e).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise AdapterError(f"stream failed: {e}") from e
        if last_exc:
            raise AdapterError(f"stream exhausted retries: {last_exc}") from last_exc

    def _compute_stream_retry_delay(
        self, attempt: int, exc: BaseException
    ) -> float:
        """Backoff for the streaming retry loop.

        Mirrors ``with_retry`` semantics for the non-streaming path: Retry-After
        wins when present, otherwise exponential with jitter, clamped to
        ``max_delay``.
        """
        delay = min(self._base_delay * (2**attempt), self._max_delay)
        if RESPECT_RETRY_AFTER:
            hint = extract_retry_after_seconds(exc)
            if hint is not None and hint > 0:
                delay = min(hint, self._max_delay)
        if JITTER_RATIO > 0:
            spread = max(0.0, min(1.0, JITTER_RATIO))
            delay = max(0.0, delay * random.uniform(1.0 - spread, 1.0 + spread))
        return delay


class LLMProviderClient(_OpenAIHTTPBase):
    """OpenAI-compatible HTTP client for chat completions and model listing.

    Embeddings live on a separate client (``EmbeddingProviderClient``) so that
    chat and embedding providers can be configured independently — different
    base URLs, different API keys.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_path: str = "/chat/completions",
        timeout_seconds: int = 600,
        max_retries: int = 3,
        llm_semaphore: asyncio.Semaphore | None = None,
        throttle: HttpThrottle | None = None,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 30.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            semaphore=llm_semaphore or asyncio.Semaphore(4),
            throttle=throttle
            or HttpThrottle(
                label="llm",
                max_rps=0.0,
                min_interval_ms=0,
                max_delay_seconds=max_delay_seconds,
            ),
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
        )
        self._chat_path = chat_path

    async def chat_completions(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature

        if not stream:
            return await self._request("POST", self._chat_path, json_body=body)

        chunks: list[str] = []
        async for delta in self._request_stream(self._chat_path, body):
            chunks.append(delta)
        return {"choices": [{"message": {"content": "".join(chunks)}}]}

    async def stream_chat_completions(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if temperature is not None:
            body["temperature"] = temperature
        async for delta in self._request_stream(self._chat_path, body):
            yield delta

    async def list_models(self, refresh: bool = False) -> list[ModelInfo]:
        resp = await self._request("GET", "/models")
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


class EmbeddingProviderClient(_OpenAIHTTPBase):
    """OpenAI-compatible HTTP client for embeddings only.

    Constructed from a base URL + API key distinct from the chat provider's,
    so operators can mix backends (e.g. chat at OpenAI, embeddings at a local
    Ollama).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        embeddings_path: str = "/embeddings",
        timeout_seconds: int = 600,
        max_retries: int = 3,
        embedding_semaphore: asyncio.Semaphore | None = None,
        throttle: HttpThrottle | None = None,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 30.0,
        batch_max_inputs: int = 0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            semaphore=embedding_semaphore or asyncio.Semaphore(8),
            throttle=throttle
            or HttpThrottle(
                label="embeddings",
                max_rps=0.0,
                min_interval_ms=0,
                max_delay_seconds=max_delay_seconds,
            ),
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
        )
        self._embeddings_path = embeddings_path
        # 0 disables chunking; preserves "single POST per call" for callers
        # that already pre-batched inputs.
        self._batch_max_inputs = max(0, int(batch_max_inputs))

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        cap = self._batch_max_inputs
        if cap <= 0 or len(inputs) <= cap:
            return await self._embeddings_once(model, inputs)
        # Chunk to protect upstream TPM. Each chunk goes through the throttle
        # gate independently so RPS / min-interval limits apply.
        out: list[list[float]] = []
        for start in range(0, len(inputs), cap):
            chunk = inputs[start : start + cap]
            out.extend(await self._embeddings_once(model, chunk))
        return out

    async def _embeddings_once(
        self, model: str, inputs: list[str]
    ) -> list[list[float]]:
        resp = await self._request(
            "POST",
            self._embeddings_path,
            json_body={"model": model, "input": inputs},
        )
        data = (resp or {}).get("data") or []
        return [d.get("embedding") or [] for d in data]
