import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime
from typing import Any

from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import (
    checkpoint,
    get_dr_state,
    new_dr_state,
    set_dr_state,
)

logger = logging.getLogger("Deep Research")


def build_source_markdown(
    url: str,
    title: str,
    exact_text: str,
    meta: dict[str, Any],
) -> str:
    """Wrap exact extractor output with a deterministic metadata preamble.

    The metadata preamble is programmatically generated; the extracted
    body below the '## Extracted Content' header is preserved verbatim
    (no summarization, no reflow, no truncation).
    """
    accessed = meta.get("accessed") or datetime.now().isoformat()
    source_id = meta.get("source_id") or ""
    search_query = meta.get("search_query") or ""
    archived = "true" if meta.get("archived") else "false"
    head_lines = [
        f"# {title or url}",
        "",
        "---",
        f"- **Source URL**: {url}",
        f"- **Accessed**: {accessed}",
        f"- **Archived**: {archived}",
    ]
    if source_id:
        head_lines.insert(3, f"- **Source ID**: {source_id}")
    if search_query:
        head_lines.insert(4, f"- **Search Query**: {search_query}")
    head_lines.append("- **Extracted by**: Deep Research (exact copy)")
    head_lines.append("")
    head_lines.append("## Extracted Content")
    head_lines.append("")
    return "\n".join(head_lines) + "\n" + (exact_text or "")


def safe_filename_from_url(url: str, suffix: str = ".md") -> str:
    """Derive a safe filename from a URL for KB file naming."""
    import re
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        base = (parsed.netloc + parsed.path).rstrip("/")
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", base)
        safe = re.sub(r"_+", "_", safe).strip("_")
        if len(safe) > 100:
            h = hashlib.sha256(url.encode()).hexdigest()[:12]
            safe = safe[:85] + "_" + h
        return safe + suffix
    except Exception:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"source_{h}{suffix}"


async def attach_collection_to_chat(
    ctx: RunContext, chat_id: str | None, kb_id: str, kb_name: str
) -> None:
    """Append the research KB as a collection entry on the chat's files list.

    This makes future user messages re-send the collection automatically,
    so OWUI middleware retrieves from it before the pipe even runs.
    """
    if not chat_id or not kb_id:
        return
    try:
        chat = await ctx.client.get_chat(chat_id)
        if not chat:
            return
        chat_data = chat.get("chat") if isinstance(chat, dict) and "chat" in chat else chat
        merged = dict(chat_data or {})
        files = list(merged.get("files") or [])
        already = any(
            isinstance(f, dict)
            and f.get("type") == "collection"
            and f.get("id") == kb_id
            for f in files
        )
        if not already:
            files.append(
                {
                    "type": "collection",
                    "id": kb_id,
                    "name": kb_name,
                    "status": "processed",
                }
            )
        merged["files"] = files
        await ctx.client.update_chat(chat_id, merged)
    except Exception as e:
        logger.warning(
            f"Failed to attach collection {kb_id} to chat {chat_id}: {e}"
        )

def slugify_research_title(title: str) -> str:
    """Collapse a user title to lowercase alphanumeric for KB naming."""
    if not title:
        return "research"
    s = re.sub(r"[^a-z0-9]+", "", title.lower())
    return s[:48] or "research"

def build_kb_name(title: str, now_dt: datetime) -> str:
    timestamp = now_dt.strftime("%Y%m%d%H%M%S")
    slug = slugify_research_title(title or "")
    return f"dr-{timestamp}-{slug}"

async def ensure_research_kb(ctx: RunContext, title_hint: str) -> tuple[str, str] | None:
    """Create (or return existing) private research KB. Returns (id, name)."""
    dr = get_dr_state(ctx)
    if dr and dr.get("kb_id") and dr.get("kb_name"):
        return dr["kb_id"], dr["kb_name"]

    try:
        now = datetime.now()
        kb_name = build_kb_name(title_hint, now)
        description = (
            f"Deep Research run for: {(title_hint or '')[:200]}"
            if title_hint
            else "Deep Research run"
        )
        knowledge = await ctx.client.create_kb(kb_name, description)
        kb_id = knowledge.id
        if dr is None:
            dr = new_dr_state(ctx)
        dr["kb_id"] = kb_id
        dr["kb_name"] = kb_name
        dr["status"] = "discovering"
        set_dr_state(ctx, dr)
        await checkpoint(ctx)
        logger.info(f"Created research KB id={kb_id} name={kb_name}")
        return kb_id, kb_name
    except Exception as e:
        logger.error(f"Error creating research KB: {e}")
        return None

async def upload_markdown_to_kb(
    ctx: RunContext,
    kb_id: str,
    filename: str,
    markdown_text: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Upload synthetic markdown, process inline, add to the KB.

    Returns the OWUI file_id on success, or None on failure. The upload
    is forced synchronous so embedding completes before we add to the collection.
    The `metadata` parameter is accepted for caller compatibility but not
    forwarded — OWUI's /api/v1/files/ endpoint derives file metadata server-side.
    """
    del metadata  # accepted for caller compatibility; OWUI derives meta server-side
    payload = markdown_text.encode("utf-8", errors="replace")

    try:
        upload = await ctx.client.upload_file(payload, filename, process=True)
    except Exception as e:
        logger.warning(f"upload_file failed for {filename}: {e}")
        return None

    file_id = getattr(upload, "id", None)
    if not file_id:
        logger.warning(f"upload returned no id for {filename}")
        return None

    try:
        await ctx.client.add_file_to_kb(kb_id, file_id)
    except Exception as e:
        logger.error(
            f"add_file_to_kb failed (kb={kb_id}, file={file_id}): {e}"
        )
        return None

    return file_id

async def persist_selected_source(ctx: RunContext,
    url: str,
    full_text: str,
    title: str,
    source_type: str = "web",
    archived: bool = False,
    search_query: str = "",
) -> str | None:
    """Write a selected source through to the research KB.

    Idempotent per URL: if the URL is already in the source_manifest with
    the same content hash, returns the existing file_id without
    re-uploading. If the content changed, uploads a new version and
    retains the older file_id under a 'versions' list.
    """
    if not url or not isinstance(full_text, str) or not full_text.strip():
        return None

    # Throttle KB ingestion under embedding-quota pressure.
    pv = getattr(ctx.valves, "persistence", None)
    if pv is not None:
        diag = getattr(ctx, "embeddings_diagnostics", None)
        if (
            getattr(pv, "disable_during_degraded", False)
            and diag is not None
            and diag.degraded
        ):
            logger.info(
                "Skipping KB persistence for %s: embedding throttle in "
                "degraded mode (disable_during_degraded=True)",
                url,
            )
            return None
        cap = int(getattr(pv, "max_kb_uploads_per_cycle", 0) or 0)
        gate = getattr(ctx, "persistence_gate", None)
        if cap > 0 and gate is not None and gate.uploads_this_cycle >= cap:
            logger.info(
                "Skipping KB persistence for %s: max_kb_uploads_per_cycle=%d reached",
                url,
                cap,
            )
            return None
        delay_ms = int(getattr(pv, "kb_upload_delay_ms", 0) or 0)
        if delay_ms > 0 and gate is not None and gate.last_upload_monotonic > 0:
            gap_s = time.monotonic() - gate.last_upload_monotonic
            min_gap_s = delay_ms / 1000.0
            if gap_s < min_gap_s:
                await asyncio.sleep(min_gap_s - gap_s)

    dr = get_dr_state(ctx)
    if dr is None:
        dr = new_dr_state(ctx)
        set_dr_state(ctx, dr)

    if not dr.get("kb_id"):
        ensured = await ensure_research_kb(ctx,
            dr.get("user_request_summary") or title or url
        )
        if not ensured:
            return None
        kb_id, _kb_name = ensured
    else:
        kb_id = dr["kb_id"]

    content_hash = hashlib.sha256(
        full_text.encode("utf-8", errors="replace")
    ).hexdigest()
    manifest = dr.setdefault("source_manifest", {})
    existing = manifest.get(url)
    if (
        existing
        and existing.get("hash") == content_hash
        and existing.get("file_id")
    ):
        return existing.get("file_id")

    master = ctx.state.get_state(ctx.conversation_id).get("master_source_table", {})
    source_id = ""
    if url in master:
        source_id = master[url].get("id", "")
    meta_for_md = {
        "accessed": datetime.now().isoformat(),
        "source_id": source_id,
        "search_query": search_query,
        "archived": archived,
    }
    body_md = build_source_markdown(url, title, full_text, meta_for_md)
    filename = safe_filename_from_url(url, ".md")

    file_id = await upload_markdown_to_kb(ctx,
        kb_id=kb_id,
        filename=filename,
        markdown_text=body_md,
        metadata={
            "name": filename,
            "content_type": "text/markdown",
            "source_url": url,
            "source_title": title,
            "source_type": source_type,
            "archived": archived,
            "research_kb_id": kb_id,
        },
    )
    if not file_id:
        return None

    entry = {
        "title": title or url,
        "file_id": file_id,
        "hash": content_hash,
        "source_type": source_type,
        "archived": archived,
        "citation_id": source_id,
        "persisted_at": datetime.now().isoformat(),
        "search_query": search_query,
    }
    if existing and existing.get("file_id") and existing.get("file_id") != file_id:
        versions = list(existing.get("versions") or [])
        versions.append(
            {
                "file_id": existing.get("file_id"),
                "hash": existing.get("hash"),
                "persisted_at": existing.get("persisted_at"),
            }
        )
        entry["versions"] = versions
    manifest[url] = entry
    set_dr_state(ctx, dr)
    await checkpoint(ctx)
    gate = getattr(ctx, "persistence_gate", None)
    if gate is not None:
        gate.uploads_this_cycle += 1
        gate.last_upload_monotonic = time.monotonic()
    logger.info(f"Persisted source url={url} file_id={file_id}")
    return file_id

async def persist_final_report(ctx: RunContext, report_md: str, report_title: str
) -> str | None:
    """Persist the finalized report markdown into the research KB."""
    dr = get_dr_state(ctx)
    if not dr or not dr.get("kb_id"):
        return None
    kb_id = dr["kb_id"]
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"deep-research-report-{ts}.md"
    meta = {
        "name": filename,
        "content_type": "text/markdown",
        "report_title": report_title,
        "research_kb_id": kb_id,
        "is_final_report": True,
    }
    file_id = await upload_markdown_to_kb(ctx,
        kb_id=kb_id,
        filename=filename,
        markdown_text=report_md,
        metadata=meta,
    )
    if file_id:
        dr["report_file_id"] = file_id
        dr["report_completed"] = True
        dr["status"] = "completed"
        dr["mode"] = "post_report_user_qa"
        set_dr_state(ctx, dr)
        await checkpoint(ctx)
        logger.info(f"Persisted final report file_id={file_id}")
    return file_id

async def rehydrate_working_corpus_from_kb(ctx: RunContext) -> dict[str, str]:
    """Rebuild url_results_cache from persisted KB file content.

    Returns the rehydrated cache (and also installs it into in-memory
    state). Used on resume after process restart, where the in-memory
    ephemeral corpus has been lost but the KB still has the sources.
    """
    dr = get_dr_state(ctx)
    if not dr:
        return {}
    manifest = dr.get("source_manifest") or {}
    if not manifest:
        return {}


    rehydrated: dict[str, str] = {}
    for url, entry in manifest.items():
        file_id = entry.get("file_id")
        if not file_id:
            continue
        try:
            f = await ctx.client.get_file(file_id)
            if not f or not f.data:
                continue
            content = f.data.get("content")
            if isinstance(content, str) and content:
                rehydrated[url] = content
        except Exception as e:
            logger.debug(f"Rehydrate skipped for {url} (file {file_id}): {e}")
            continue
    if rehydrated:
        state = ctx.state.get_state(ctx.conversation_id)
        cache = dict(state.get("url_results_cache") or {})
        for k, v in rehydrated.items():
            cache.setdefault(k, v)
        ctx.state.update_state(ctx.conversation_id, "url_results_cache", cache)
        logger.info(
            f"Rehydrated {len(rehydrated)} sources from KB into working corpus"
        )
    return rehydrated

async def kb_search(
    ctx: RunContext, kb_id: str, query: str, k: int = 6
) -> list[dict[str, Any]]:
    """Run a vector search against the research KB collection via OWUI REST."""
    if not kb_id or not query:
        return []
    try:
        resp = await ctx.client.query_collection(
            collection_names=[kb_id],
            query=query,
            k=int(k),
            hybrid=False,
        )
    except Exception as e:
        logger.warning(
            f"KB vector search failed (kb={kb_id}, q={query[:60]!r}): {e}"
        )
        return []

    documents = getattr(resp, "documents", None) or [[]]
    metadatas = getattr(resp, "metadatas", None) or [[]]
    distances = getattr(resp, "distances", None) or [[]]
    docs0 = documents[0] if documents else []
    metas0 = metadatas[0] if metadatas else []
    dists0 = distances[0] if distances else []

    out: list[dict[str, Any]] = []
    for i, txt in enumerate(docs0):
        meta = metas0[i] if i < len(metas0) else {}
        dist = dists0[i] if i < len(dists0) else None
        out.append({"text": txt or "", "source": meta, "distance": dist})
    return out

