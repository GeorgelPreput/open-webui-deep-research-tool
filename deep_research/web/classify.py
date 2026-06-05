import logging
import time
from urllib.parse import urlparse

from deep_research.config.constants import MIN_EXTRACTION_LENGTH, REQUIRE_EXTRACTION_QUALITY
from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.web.classify")

# (base_url, token) -> (verdict, monotonic_expiry). A short TTL so a transient
# failure expires instead of latching off for the process lifetime, and distinct
# OWUI instances / tokens get independent verdicts.
_owui_ext_cap: dict[tuple[str, str], tuple[bool, float]] = {}
_OWUI_EXT_CAP_TTL_SECONDS = 300.0


def classify_url(ctx: RunContext, url: str) -> str:
    """Classify a URL for extraction routing.

    Returns one of "pdf", "html", "unknown". Suffix-based; runtime
    content-type-based PDF detection happens later in fetch_content.
    """
    if not url or not isinstance(url, str):
        return "unknown"
    base = url.lower().split("?", 1)[0].split("#", 1)[0]
    if base.endswith(".pdf"):
        return "pdf"
    return "html"


def url_fallback_title(ctx: RunContext, url: str) -> str:
    """Derive a title from a URL path when none was provided."""
    try:
        parsed = urlparse(url)
        last = parsed.path.rsplit("/", 1)[-1]
        if last:
            return (
                last.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()
                or url
            )
        return parsed.netloc or url
    except Exception:
        return url


def check_extraction_quality(ctx: RunContext, text: str) -> bool:
    """Minimum-quality gate for accepting a primary-path extraction."""
    if not REQUIRE_EXTRACTION_QUALITY:
        return bool(text and isinstance(text, str) and text.strip())
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < int(MIN_EXTRACTION_LENGTH):
        return False
    lowered = stripped.lower()
    if lowered.startswith("error") or lowered.startswith("could not extract"):
        return False
    return not (stripped.startswith("<") and stripped.count("<") > max(20, len(stripped) // 20))


async def owui_extraction_available(ctx: RunContext, kind: str) -> bool:
    """Check whether OWUI extraction endpoints are reachable.

    Both kinds share the same liveness signal (ctx.client connectivity), so
    ``kind`` is accepted for call-site compatibility but does not key the cache.
    Cached briefly per (base_url, token): a single ``GET /api/v1/auths/`` ping
    tells us the OWUI URL is reachable and the token is valid; if it is, both
    process_web and process_file/upload_file will be available too. The TTL
    means a transient failure re-probes instead of latching off, and distinct
    instances/tokens get independent verdicts.

    Probes ``get_session_user`` (``GET /api/v1/auths/``) rather than
    ``/api/v1/models/list`` because the latter returns ``{"items": [...]}``
    that depends on the API key's model access grants — a token with no
    granted models would silently latch the probe to False even though every
    retrieval endpoint we care about is reachable. ``/api/v1/auths/`` is
    gated by ``get_current_user`` on the OWUI side, so any valid token
    (any role) succeeds; the probe is a pure reachability + auth check.
    """
    base_url = getattr(ctx.client, "_base_url", "")
    try:
        token = await ctx.client._token_provider.get_token()
    except Exception:
        token = ""
    cache_key = (base_url, token)

    cached = _owui_ext_cap.get(cache_key)
    if cached is not None and cached[1] > time.monotonic():
        return cached[0]

    available = False
    try:
        user = await ctx.client.get_session_user()
        available = isinstance(user, dict) and bool(user)
    except Exception as e:
        logger.info(
            f"OWUI extraction unavailable ({type(e).__name__}: {e}); "
            f"primary-path extraction disabled (will re-probe after TTL)"
        )
        available = False

    _owui_ext_cap[cache_key] = (available, time.monotonic() + _OWUI_EXT_CAP_TTL_SECONDS)
    return available
