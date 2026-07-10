import hashlib
import html
import json
import secrets
from typing import Any

from deep_research.core.text import escape_html
from deep_research.core.types import RunContext
from deep_research.progress.events import EmbedEvent
from deep_research.progress.snapshot import build_progress_snapshot, normalize_progress_categories


def render_progress_embed_html(
    snapshot: dict[str, Any],
    *,
    poll_url: str | None = None,
    view_token: str | None = None,
) -> str:
    """Render the iframe HTML for the progress embed.

    Push-only mode (default, ``poll_url=None``) emits the same DOM the
    Function runtime has always emitted: header, token counts, the five
    topic-category sections, and the parent-height reporting script.

    Self-polling mode (``poll_url`` set) adds a second inline script that
    polls the status endpoint every 2s and reloads the iframe on a
    revision bump. Both modes share the same CSP meta. ``view_token`` is
    required when ``poll_url`` is set so the polling script can include
    it in requests.
    """
    if poll_url is not None and not view_token:
        raise ValueError("view_token is required when poll_url is provided")

    categories = normalize_progress_categories(snapshot)
    e = escape_html

    query_text = snapshot.get("query", "") or ""
    if len(query_text) > 240:
        query_text = query_text[:237] + "..."

    header = (
        f'<div class="hdr">'
        f'<div class="q"><span class="lbl">Query:</span> {e(query_text)}</div>'
        f'<div class="meta">'
        f"<span>Cycle {e(snapshot.get('cycle', 0))}/{e(snapshot.get('max_cycles', 0))}</span>"
        f"<span>Revision {e(snapshot.get('revision', 0))}</span>"
        f"<span>{e(snapshot.get('updated_at', ''))}</span>"
        f"</div></div>"
    )

    tokens = (
        f'<div class="tokens">'
        f'<div><span class="lbl">Research Results</span><span class="num">{e(snapshot.get("results_tokens", 0))}</span></div>'
        f'<div><span class="lbl">Synthesis</span><span class="num">{e(snapshot.get("synthesis_tokens", 0))}</span></div>'
        f'<div><span class="lbl">Total</span><span class="num">{e(snapshot.get("total_tokens", 0))}</span></div>'
        f"</div>"
    )

    def checklist(items: list[str], checked: bool) -> str:
        if not items:
            return '<div class="empty">—</div>'
        attrs = " checked" if checked else ""
        rows = [
            f'<label class="row"><input type="checkbox" disabled{attrs}><span>{e(t)}</span></label>'
            for t in items
        ]
        return "".join(rows)

    def emoji_list(items: list[str], emoji: str) -> str:
        if not items:
            return '<div class="empty">—</div>'
        rows = [
            f'<div class="row"><span class="emo">{emoji}</span><span>{e(t)}</span></div>'
            for t in items
        ]
        return "".join(rows)

    green_circle = "✅"
    yellow_circle = "\U0001f7e1"
    orange_circle = "\U0001f7e3"
    red_circle = "\U0001f534"
    white_circle = "⚪"
    sections = (
        f'<section><h2><span class="emo">{green_circle}</span>Topics Completed <span class="count">({len(categories["completed"])})</span></h2>{checklist(categories["completed"], True)}</section>'
        f'<section><h2><span class="emo">{yellow_circle}</span>Topics Partially Addressed <span class="count">({len(categories["partial"])})</span></h2>{emoji_list(categories["partial"], yellow_circle)}</section>'
        f'<section><h2><span class="emo">{orange_circle}</span>New Topics Discovered <span class="count">({len(categories["new"])})</span></h2>{emoji_list(categories["new"], orange_circle)}</section>'
        f'<section><h2><span class="emo">{red_circle}</span>Irrelevant / Distraction Topics <span class="count">({len(categories["irrelevant"])})</span></h2>{emoji_list(categories["irrelevant"], red_circle)}</section>'
        f'<section><h2><span class="emo">{white_circle}</span>Remaining Topics <span class="count">({len(categories["remaining"])})</span></h2>{checklist(categories["remaining"], False)}</section>'
    )

    nonce = secrets.token_urlsafe(16)
    csp = (
        f'<meta http-equiv="Content-Security-Policy" '
        f'content="default-src \'none\'; '
        f"connect-src 'self' https:; "
        f"img-src data: https:; "
        f"style-src 'unsafe-inline'; "
        f"script-src 'nonce-{nonce}'; "
        f"object-src 'none'; "
        f"base-uri 'none'; "
        f'form-action \'none\';">'
    )

    height_script = (
        f'<script nonce="{nonce}">'
        "function reportHeight() {"
        "const h = document.documentElement.scrollHeight;"
        "parent.postMessage({ type: 'iframe:height', height: h }, '*');"
        "}"
        "window.addEventListener('load', reportHeight);"
        "new ResizeObserver(reportHeight).observe(document.body);"
        "</script>"
    )

    poll_script = ""
    if poll_url is not None:
        bootstrap = {
            "poll_url": poll_url,
            "view_token": view_token,
            "since_version": int(snapshot.get("revision", 0) or 0),
        }
        # Carry the bootstrap config through an HTML-escaped data attribute
        # rather than interpolating JSON into the <script> body. The three
        # current fields are all server-side: ``poll_url`` is built from
        # ``DR_OPENAPI_PUBLIC_BASE_URL`` (or the request host) and the
        # UUID-shaped ``job_id``; ``view_token`` is
        # ``secrets.token_urlsafe(32)`` minted at job creation;
        # ``since_version`` is an int coerced from ``snapshot["revision"]``.
        # None of these are user-derived today. The escape is retained as
        # defence-in-depth: a future addition to the bootstrap dict carrying
        # user-derived data (e.g. an echoed query string) must not be able
        # to break out of the data attribute or terminate the surrounding
        # ``<script>`` block with a stray ``</script>`` / ``<!--`` sequence.
        bootstrap_attr = html.escape(json.dumps(bootstrap), quote=True)
        poll_script = (
            f'<div id="dr-bootstrap" hidden data-bootstrap="{bootstrap_attr}"></div>'
            f'<script nonce="{nonce}">'
            "(function(){"
            "var bs = JSON.parse(document.getElementById('dr-bootstrap').dataset.bootstrap);"
            "var sinceVersion = bs.since_version;"
            "var stopped = false;"
            "async function poll() {"
            "if (stopped) return;"
            "try {"
            "var url = bs.poll_url + '?token=' + encodeURIComponent(bs.view_token) + '&since_version=' + sinceVersion;"
            "var resp = await fetch(url, { credentials: 'omit' });"
            "if (resp.status === 204) return;"
            "if (resp.status >= 400 && resp.status < 500) {"
            "var panel = document.querySelector('.panel');"
            "if (panel) panel.innerHTML = '<p style=\"padding:16px;color:#888;\">Session expired or job not found.</p>';"
            "stopped = true;"
            "return;"
            "}"
            "if (resp.status >= 500) { console.warn('[deep-research] poll', resp.status); return; }"
            "var data = await resp.json();"
            "if (typeof data.revision === 'number' && data.revision > sinceVersion) {"
            "sinceVersion = data.revision;"
            "window.location.reload();"
            "}"
            "} catch (err) { console.warn('[deep-research] poll error', err); }"
            "}"
            "setInterval(poll, 2000);"
            "})();"
            "</script>"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">{csp}<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 16px; color: #222; background: #fafafa; }}
.panel {{ background: #fff; border: 1px solid #e3e3e3; border-radius: 12px; padding: 18px; }}
.hdr {{ margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid #eee; }}
.hdr .q {{ font-size: 0.95em; line-height: 1.4; margin-bottom: 6px; word-break: break-word; }}
.hdr .q .lbl {{ color: #888; font-weight: 600; margin-right: 6px; }}
.hdr .meta {{ display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.8em; color: #777; }}
.tokens {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.tokens > div {{ flex: 1; min-width: 140px; background: #f5f7fa; border-radius: 8px; padding: 10px 12px; display: flex; flex-direction: column; gap: 2px; }}
.tokens .lbl {{ font-size: 0.75em; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }}
.tokens .num {{ font-size: 1.2em; font-weight: 600; color: #222; }}
section {{ margin-top: 14px; }}
section h2 {{ font-size: 0.95em; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }}
section h2 .count {{ color: #888; font-weight: 400; font-size: 0.85em; }}
.row {{ display: flex; align-items: flex-start; gap: 8px; padding: 4px 0; font-size: 0.9em; line-height: 1.35; }}
.row input[type=checkbox] {{ margin-top: 3px; }}
.emo {{ display: inline-block; min-width: 1.2em; }}
.empty {{ color: #bbb; font-size: 0.85em; padding: 4px 0; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #1f2023; color: #e6e6e6; }}
  .panel {{ background: #2a2c30; border-color: #3a3d42; }}
  .hdr {{ border-bottom-color: #3a3d42; }}
  .hdr .q .lbl, .hdr .meta, section h2 .count, .tokens .lbl {{ color: #9aa0a6; }}
  .tokens > div {{ background: #35383d; }}
  .tokens .num {{ color: #f0f0f0; }}
}}
</style></head><body><div class="panel">{header}{tokens}{sections}</div>
{height_script}{poll_script}
</body></html>"""


async def refresh_progress_embed(
    ctx: RunContext, cycle: int | None = None, force: bool = False
) -> None:
    if not ctx.valves.events.enable_progress_embed:
        return
    snapshot = build_progress_snapshot(ctx, cycle=cycle)
    html_content = render_progress_embed_html(snapshot)
    # Dedup on the meaningful content only: updated_at (a fresh timestamp) and
    # revision (incremented every call) would otherwise change the hash every
    # time and defeat the de-duplication entirely.
    stable = {k: v for k, v in snapshot.items() if k not in ("updated_at", "revision")}
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    state = ctx.state.get_state(ctx.conversation_id)
    if not force and state.get("progress_embed_last_hash") == digest:
        return
    ctx.state.update_state(ctx.conversation_id, "progress_embed_last_hash", digest)
    ctx.state.update_state(ctx.conversation_id, "progress_embed_revision", snapshot.get("revision", 0))
    ctx.state.update_state(ctx.conversation_id, "progress_last_updated_at", snapshot.get("updated_at", ""))
    await ctx.events.emit(EmbedEvent(html=html_content, snapshot=snapshot))
