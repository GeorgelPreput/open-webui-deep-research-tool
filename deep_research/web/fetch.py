import logging
import re
from datetime import datetime
from typing import Any

import httpx

from deep_research.config.constants import (
    EXTRACTION_CONTENT_ONLY,
    HANDLE_PDFS,
    POST_CLEAN_PRIMARY_OUTPUT,
    PRIMARY_DOCUMENT_EXTRACTION,
    PRIMARY_WEB_EXTRACTION,
    USE_OPENWEBUI_EXTRACTION,
)
from deep_research.core.types import RunContext
from deep_research.persistence.kb import persist_selected_source
from deep_research.web.classify import (
    check_extraction_quality,
    classify_url,
    owui_extraction_available,
    url_fallback_title,
)
from deep_research.web.html_extract import extract_text_from_html
from deep_research.web.pdf_extract import extract_text_from_pdf

logger = logging.getLogger("deep_research.web.fetch")


async def try_primary_web_flow(ctx: RunContext, url: str) -> str | None:
    """Route an HTML URL through OWUI's REST extraction.

    Returns extracted text on success (and registers the source), None
    if extraction is disabled by valves, the URL is not HTML-classified,
    OWUI is unreachable, or the result fails the quality gate.
    fetch_content treats a None return as a hard failure for HTML URLs.
    """
    if not (
        USE_OPENWEBUI_EXTRACTION
        and PRIMARY_WEB_EXTRACTION
    ):
        return None
    if classify_url(ctx, url) != "html":
        return None
    if not await owui_extraction_available(ctx, "web"):
        return None

    try:
        result = await primary_web_extract(ctx, url)
    except Exception as e:
        logger.warning(f"Primary web extraction failed for {url}: {e}")
        return None
    if not result:
        return None
    text = result.get("text", "")
    if not check_extraction_quality(ctx, text):
        logger.info(
            f"Primary web extraction rejected by quality gate for {url}"
        )
        return None

    if POST_CLEAN_PRIMARY_OUTPUT:
        try:
            cleaned = await extract_text_from_html(ctx, text)
            if cleaned and cleaned.strip():
                result["text"] = cleaned
                text = cleaned
        except Exception:
            pass

    if not result.get("title"):
        result["title"] = url_fallback_title(ctx, url)

    await register_primary_source(ctx, url, result)
    logger.info(f"Primary web extraction succeeded for {url} ({len(text)} chars)")
    return text


async def primary_web_extract(ctx: RunContext, url: str) -> dict[str, Any] | None:
    """REST-based primary web extraction via OWUI's process_web endpoint."""
    try:
        resp = await ctx.client.process_web_url(url, process=False)
    except Exception as e:
        logger.info(f"OWUI process_web_url failed for {url}: {e}")
        return None
    text = (resp.content or "").strip()
    if not text:
        return None
    return {
        "text": text,
        "title": None,
        "source_type": "web",
        "archived": False,
    }


async def primary_document_extract(
    ctx: RunContext,
    content_bytes: bytes,
    url: str,
    content_type: str = "application/pdf",
) -> str | None:
    """REST-based primary document extraction via OWUI upload + process_file."""
    if not await owui_extraction_available(ctx, "doc"):
        return None

    parsed_path = ""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url) if url else None
        if parsed and parsed.path:
            parsed_path = parsed.path.rsplit("/", 1)[-1]
    except Exception:
        parsed_path = ""
    basename = parsed_path or "document"
    if "." not in basename:
        if "pdf" in (content_type or "").lower():
            basename += ".pdf"
        elif "html" in (content_type or "").lower():
            basename += ".html"
        else:
            basename += ".bin"

    payload = content_bytes if isinstance(content_bytes, bytes) else (
        content_bytes.encode("utf-8", errors="ignore")
    )

    try:
        upload = await ctx.client.upload_file(payload, basename, process=True)
    except Exception as e:
        logger.info(f"OWUI upload_file failed for {url}: {e}")
        return None

    data = getattr(upload, "data", None) or {}
    content = data.get("content") if isinstance(data, dict) else None
    if isinstance(content, str) and content.strip():
        return content
    try:
        proc = await ctx.client.process_file(upload.id)
    except Exception as e:
        logger.info(f"OWUI process_file failed for {url} (file={upload.id}): {e}")
        return None
    text = (proc.content or "").strip()
    return text or None


async def extract_pdf_with_primary_fallback(
    ctx: RunContext,
    pdf_content: bytes,
    url: str,
    content_type: str = "application/pdf",
) -> str:
    """PDF extraction: REST-first, pypdf/pdfplumber fallback."""
    if (
        USE_OPENWEBUI_EXTRACTION
        and PRIMARY_DOCUMENT_EXTRACTION
        and HANDLE_PDFS
    ):
        try:
            primary_text = await primary_document_extract(
                ctx, pdf_content, url, content_type
            )
            if primary_text and check_extraction_quality(ctx, primary_text):
                logger.info(
                    f"Primary document extraction succeeded for {url} "
                    f"({len(primary_text)} chars)"
                )
                return primary_text
            if primary_text is not None:
                logger.info(
                    f"Primary document extraction rejected by quality gate for {url}; "
                    f"using legacy PDF extractor"
                )
            else:
                logger.info(
                    f"Primary document extraction returned no result for {url}; "
                    f"using legacy PDF extractor"
                )
        except Exception as e:
            logger.warning(
                f"Primary document extraction error for {url}: {e}; "
                f"using legacy PDF extractor"
            )
    logger.info(f"Using legacy PDF extractor for {url}")
    return await extract_text_from_pdf(ctx, pdf_content)


async def register_primary_source(
    ctx: RunContext, url: str, result: dict[str, Any]
) -> None:
    """Cache + master_source_table + KB write-through for primary extractions."""
    from deep_research.budget.tokens import count_tokens

    state = ctx.state.get_state(ctx.conversation_id)
    url_results_cache = state.get("url_results_cache", {})
    master_source_table = state.get("master_source_table", {})

    text = result.get("text") or ""
    title = result.get("title") or url_fallback_title(ctx, url)
    source_type = result.get("source_type") or "web"
    archived = bool(result.get("archived", False))

    if isinstance(text, str) and text:
        tokens = await count_tokens(ctx, text)
        token_limit = ctx.valves.web.max_result_tokens * 3
        if tokens > token_limit:
            char_limit = int(len(text) * (token_limit / tokens))
            to_cache = text[:char_limit]
            logger.info(
                f"Limiting cached primary content for URL {url} "
                f"from {tokens} to {token_limit} tokens"
            )
        else:
            to_cache = text
    else:
        to_cache = text

    url_results_cache[url] = to_cache
    ctx.state.update_state(ctx.conversation_id, "url_results_cache", url_results_cache)

    if url not in master_source_table:
        source_id = f"S{len(master_source_table) + 1}"
        entry: dict[str, Any] = {
            "id": source_id,
            "title": title,
            "content_preview": text[:500] if isinstance(text, str) else "",
            "source_type": source_type,
            "accessed_date": datetime.now().strftime("%Y-%m-%d"),
            "cited_in_sections": set(),
        }
        if archived:
            entry["archived"] = True
        master_source_table[url] = entry
        ctx.state.update_state(
            ctx.conversation_id, "master_source_table", master_source_table
        )

    if isinstance(text, str) and text.strip():
        try:
            await persist_selected_source(
                ctx=ctx,
                url=url,
                full_text=text,
                title=title,
                source_type=source_type,
                archived=archived,
            )
        except Exception as e:
            logger.warning(f"KB persistence failed for {url}: {e}")


async def fetch_content(ctx: RunContext, url: str) -> str:
    """Fetch and extract content for a URL.

    Routing:
      1. Cache short-circuit on url_results_cache.
      2. HTML URLs go through OWUI's REST extraction (try_primary_web_flow).
         No legacy HTML fetch fallback — if OWUI extraction is unavailable
         or returns nothing usable, we return an error string.
      3. PDF URLs go through fetch_pdf_via_legacy_download (paywall.py),
         which retains the per-domain anti-blocking machinery.
    """
    try:
        state = ctx.state.get_state(ctx.conversation_id)
        url_considered_count = state.get("url_considered_count", {})
        url_results_cache = state.get("url_results_cache", {})

        url_considered_count[url] = url_considered_count.get(url, 0) + 1
        ctx.state.update_state(
            ctx.conversation_id, "url_considered_count", url_considered_count
        )

        if url in url_results_cache:
            logger.info(f"Using cached content for URL: {url}")
            return url_results_cache[url]

        url_kind = classify_url(ctx, url)
        if url_kind == "html":
            primary_text = await try_primary_web_flow(ctx, url)
            if primary_text is not None:
                return primary_text
            logger.warning(
                f"OWUI extraction returned no usable content for {url}; "
                f"no HTML fallback configured"
            )
            return f"Error fetching content: OWUI extraction failed for {url}"

        from deep_research.web.paywall import fetch_pdf_via_legacy_download
        return await fetch_pdf_via_legacy_download(ctx, url)

    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return f"Error fetching content: {e}"


async def fetch_from_archive(
    ctx: RunContext, url: str, session: httpx.AsyncClient | None = None
) -> str:
    """Wayback Machine rescue for non-HTML URL fetches.

    Reachable from fetch_pdf_via_legacy_download when the live host
    returns HTTP 403/271. Archived PDFs are routed through
    extract_pdf_with_primary_fallback; archived HTML pages still use
    extract_text_from_html locally (since OWUI process_web won't reach
    archive.org-hosted snapshots reliably).
    """
    close_session = False
    if session is None:
        close_session = True
        session = httpx.AsyncClient(timeout=httpx.Timeout(20.0), verify=False)

    try:
        wayback_api_url = f"https://archive.org/wayback/available?url={url}"
        response = await session.get(wayback_api_url)
        if response.status_code != 200:
            logger.warning(
                f"Error accessing archive.org API: {response.status_code}"
            )
            return ""

        data = response.json()
        snapshots = data.get("archived_snapshots", {})
        closest = snapshots.get("closest", {})
        archived_url = closest.get("url")
        if not archived_url:
            logger.warning(f"No archived version found for {url}")
            return ""

        logger.info(f"Found archive for {url}: {archived_url}")
        archive_response = await session.get(archived_url)
        if archive_response.status_code != 200:
            return ""

        content_type = archive_response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            pdf_content = archive_response.content
            extracted_content = await extract_pdf_with_primary_fallback(
                ctx, pdf_content, url, "application/pdf"
            )
            await _persist_archive_result(
                ctx, url, extracted_content, source_type="pdf"
            )
            return extracted_content

        content = archive_response.text
        if EXTRACTION_CONTENT_ONLY and content.strip().startswith("<"):
            extracted = await extract_text_from_html(ctx, content)
            title = _extract_html_title(content) or f"Archived: {url}"
            await _persist_archive_result(
                ctx, url, extracted, source_type="web", title=title
            )
            return extracted
        return content

    except Exception as e:
        logger.error(f"Error fetching from archive.org: {e}")
        return ""
    finally:
        if close_session:
            await session.aclose()


def _extract_html_title(content: str) -> str | None:
    match = re.search(
        r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL
    )
    if match:
        return f"Archived: {match.group(1).strip()}"
    return None


async def _persist_archive_result(
    ctx: RunContext,
    url: str,
    text: str,
    *,
    source_type: str,
    title: str | None = None,
) -> None:
    if not text:
        return
    state = ctx.state.get_state(ctx.conversation_id)
    url_results_cache = state.get("url_results_cache", {})
    url_results_cache[url] = text
    ctx.state.update_state(ctx.conversation_id, "url_results_cache", url_results_cache)

    master_source_table = state.get("master_source_table", {})
    if url not in master_source_table:
        default_title = (
            title
            or f"Archived {source_type.upper()}: "
            + url.rsplit("/", 1)[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")
        )
        source_id = f"S{len(master_source_table) + 1}"
        master_source_table[url] = {
            "id": source_id,
            "title": default_title,
            "content_preview": text[:500],
            "source_type": source_type,
            "accessed_date": datetime.now().strftime("%Y-%m-%d"),
            "cited_in_sections": set(),
            "archived": True,
        }
        ctx.state.update_state(
            ctx.conversation_id, "master_source_table", master_source_table
        )

    try:
        await persist_selected_source(
            ctx=ctx,
            url=url,
            full_text=text,
            title=master_source_table.get(url, {}).get("title", url),
            source_type=source_type,
            archived=True,
        )
    except Exception as e:
        logger.warning(f"KB persistence failed for {url}: {e}")
