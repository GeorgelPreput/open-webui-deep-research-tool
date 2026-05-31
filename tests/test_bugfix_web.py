"""Regression tests for web-layer bug fixes.

Covers BUG 18 + 19 - owui_extraction_available uses a TTL cache keyed by
(base_url, token): a transient failure no longer latches "off" for the process
lifetime, and distinct instances/tokens get independent verdicts.
"""
import types

import pytest

from deep_research.adapter.auth import StaticToken
from deep_research.web import classify


class _FakeClient:
    def __init__(self, base_url: str, token: str, models):
        self._base_url = base_url
        self._token_provider = StaticToken(token)
        self._models = models  # list -> returned; Exception -> raised
        self.calls = 0

    async def list_models(self):
        self.calls += 1
        if isinstance(self._models, Exception):
            raise self._models
        return self._models


def _ctx(client):
    return types.SimpleNamespace(client=client)


@pytest.fixture(autouse=True)
def _clear_cap():
    classify._owui_ext_cap.clear()
    yield
    classify._owui_ext_cap.clear()


@pytest.mark.asyncio
async def test_available_true_is_cached():
    fc = _FakeClient("http://a", "tok", [{"id": "m"}])
    ctx = _ctx(fc)
    assert await classify.owui_extraction_available(ctx, "web") is True
    assert await classify.owui_extraction_available(ctx, "web") is True
    assert fc.calls == 1  # second call served from cache


@pytest.mark.asyncio
async def test_transient_failure_does_not_latch_forever():
    fc = _FakeClient("http://a", "tok", RuntimeError("temporarily down"))
    ctx = _ctx(fc)
    assert await classify.owui_extraction_available(ctx, "web") is False

    # Expire the cached verdict, then let the instance recover.
    key = ("http://a", "tok")
    assert key in classify._owui_ext_cap
    classify._owui_ext_cap[key] = (False, 0.0)  # expiry in the past
    fc._models = [{"id": "m"}]

    assert await classify.owui_extraction_available(ctx, "web") is True
    assert fc.calls == 2  # re-probed after expiry


@pytest.mark.asyncio
async def test_verdict_keyed_per_base_url():
    up = _FakeClient("http://a", "tok", [{"id": "m"}])
    down = _FakeClient("http://b", "tok", RuntimeError("down"))
    assert await classify.owui_extraction_available(_ctx(up), "web") is True
    assert await classify.owui_extraction_available(_ctx(down), "web") is False


@pytest.mark.asyncio
async def test_verdict_keyed_per_token():
    # Same base_url, different tokens -> independent probes/verdicts.
    good = _FakeClient("http://a", "good-token", [{"id": "m"}])
    bad = _FakeClient("http://a", "bad-token", RuntimeError("401"))
    assert await classify.owui_extraction_available(_ctx(good), "web") is True
    assert await classify.owui_extraction_available(_ctx(bad), "web") is False
