import logging
from urllib.parse import urlparse

from deep_research.config.constants import MIN_EXTRACTION_LENGTH, REQUIRE_EXTRACTION_QUALITY
from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.web.classify")

_owui_ext_cap: dict[str, bool | None] = {"web": None, "doc": None}


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

    Both kinds share the same liveness signal (ctx.client connectivity).
    Cached per process: a single list_models() ping is enough to tell us
    the OWUI URL is reachable and the token is valid; if it is, both
    process_web and process_file/upload_file will be available too.
    """
    cached = _owui_ext_cap.get(kind)
    if cached is not None:
        return cached

    available = False
    try:
        models = await ctx.client.list_models()
        available = bool(models)
    except Exception as e:
        logger.info(
            f"OWUI extraction unavailable ({type(e).__name__}: {e}); "
            f"primary-path extraction disabled"
        )
        available = False

    _owui_ext_cap["web"] = available
    _owui_ext_cap["doc"] = available
    return available
