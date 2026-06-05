"""Regression tests for web-layer bug fixes.

Covers BUG 18 + 19 - owui_extraction_available uses a TTL cache keyed by
(base_url, token): a transient failure no longer latches "off" for the process
lifetime, and distinct instances/tokens get independent verdicts.

The probe targets ``GET /api/v1/auths/`` (via ``OWUIClient.get_session_user``)
because that endpoint's gate accepts any valid token regardless of the user's
model access grants — earlier the probe used ``/api/v1/models/list`` and would
silently latch False when the token had no granted models, even though the
retrieval endpoints we actually rely on were reachable.
"""
import types

import pytest

from deep_research.adapter.auth import StaticToken
from deep_research.web import classify


class _FakeClient:
    """Stub matching the slice of ``OWUIClient`` owui_extraction_available reads.

    ``session_user`` is the value to return from ``get_session_user``; an
    Exception instance is raised instead.
    """

    def __init__(self, base_url: str, token: str, session_user):
        self._base_url = base_url
        self._token_provider = StaticToken(token)
        self._session_user = session_user
        self.calls = 0

    async def get_session_user(self):
        self.calls += 1
        if isinstance(self._session_user, Exception):
            raise self._session_user
        return self._session_user


def _ctx(client):
    return types.SimpleNamespace(client=client)


@pytest.fixture(autouse=True)
def _clear_cap():
    classify._owui_ext_cap.clear()
    yield
    classify._owui_ext_cap.clear()


@pytest.mark.asyncio
async def test_available_true_is_cached():
    fc = _FakeClient("http://a", "tok", {"id": "u1", "role": "user"})
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
    fc._session_user = {"id": "u1", "role": "user"}

    assert await classify.owui_extraction_available(ctx, "web") is True
    assert fc.calls == 2  # re-probed after expiry


@pytest.mark.asyncio
async def test_verdict_keyed_per_base_url():
    up = _FakeClient("http://a", "tok", {"id": "u1", "role": "user"})
    down = _FakeClient("http://b", "tok", RuntimeError("down"))
    assert await classify.owui_extraction_available(_ctx(up), "web") is True
    assert await classify.owui_extraction_available(_ctx(down), "web") is False


@pytest.mark.asyncio
async def test_verdict_keyed_per_token():
    # Same base_url, different tokens -> independent probes/verdicts.
    good = _FakeClient("http://a", "good-token", {"id": "u1", "role": "user"})
    bad = _FakeClient("http://a", "bad-token", RuntimeError("401"))
    assert await classify.owui_extraction_available(_ctx(good), "web") is True
    assert await classify.owui_extraction_available(_ctx(bad), "web") is False


@pytest.mark.asyncio
async def test_empty_or_non_dict_session_user_is_unavailable():
    """A probe that succeeds but returns an empty/non-dict body is a malformed
    response and should be treated as unavailable — same defensive shape the
    config audit relies on.
    """
    for bad in ({}, [], None, "not a dict"):
        classify._owui_ext_cap.clear()
        fc = _FakeClient("http://a", "tok", bad)
        assert await classify.owui_extraction_available(_ctx(fc), "web") is False
