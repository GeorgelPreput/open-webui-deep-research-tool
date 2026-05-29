import hashlib
from typing import Any

from deep_research.core.text import escape_html
from deep_research.core.types import RunContext
from deep_research.progress.events import EmbedEvent
from deep_research.progress.snapshot import build_progress_snapshot, normalize_progress_categories


def render_progress_embed_html(snapshot: dict[str, Any]) -> str:
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
            return '<div class="empty">\u2014</div>'
        attrs = " checked" if checked else ""
        rows = [
            f'<label class="row"><input type="checkbox" disabled{attrs}><span>{e(t)}</span></label>'
            for t in items
        ]
        return "".join(rows)

    def emoji_list(items: list[str], emoji: str) -> str:
        if not items:
            return '<div class="empty">\u2014</div>'
        rows = [
            f'<div class="row"><span class="emo">{emoji}</span><span>{e(t)}</span></div>'
            for t in items
        ]
        return "".join(rows)

    green_circle = "\u2705"
    yellow_circle = "\U0001f7e1"
    orange_circle = "\U0001f7e3"
    red_circle = "\U0001f534"
    white_circle = "\u26aa"
    sections = (
        f'<section><h2><span class="emo">{green_circle}</span>Topics Completed <span class="count">({len(categories["completed"])})</span></h2>{checklist(categories["completed"], True)}</section>'
        f'<section><h2><span class="emo">{yellow_circle}</span>Topics Partially Addressed <span class="count">({len(categories["partial"])})</span></h2>{emoji_list(categories["partial"], yellow_circle)}</section>'
        f'<section><h2><span class="emo">{orange_circle}</span>New Topics Discovered <span class="count">({len(categories["new"])})</span></h2>{emoji_list(categories["new"], orange_circle)}</section>'
        f'<section><h2><span class="emo">{red_circle}</span>Irrelevant / Distraction Topics <span class="count">({len(categories["irrelevant"])})</span></h2>{emoji_list(categories["irrelevant"], red_circle)}</section>'
        f'<section><h2><span class="emo">{white_circle}</span>Remaining Topics <span class="count">({len(categories["remaining"])})</span></h2>{checklist(categories["remaining"], False)}</section>'
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
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
<script>
function reportHeight() {{
    const h = document.documentElement.scrollHeight;
    parent.postMessage({{ type: 'iframe:height', height: h }}, '*');
}}
window.addEventListener('load', reportHeight);
new ResizeObserver(reportHeight).observe(document.body);
</script>
</body></html>"""


async def refresh_progress_embed(
    ctx: RunContext, cycle: int | None = None, force: bool = False
) -> None:
    if not ctx.valves.events.enable_progress_embed:
        return
    snapshot = build_progress_snapshot(ctx, cycle=cycle)
    html_content = render_progress_embed_html(snapshot)
    digest = hashlib.sha256(html_content.encode("utf-8")).hexdigest()
    state = ctx.state.get_state(ctx.conversation_id)
    if not force and state.get("progress_embed_last_hash") == digest:
        return
    ctx.state.update_state(ctx.conversation_id, "progress_embed_last_hash", digest)
    ctx.state.update_state(ctx.conversation_id, "progress_embed_revision", snapshot.get("revision", 0))
    ctx.state.update_state(ctx.conversation_id, "progress_last_updated_at", snapshot.get("updated_at", ""))
    await ctx.events.emit(EmbedEvent(html=html_content))
