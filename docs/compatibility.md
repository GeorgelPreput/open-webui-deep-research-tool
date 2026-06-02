# Compatibility

## OWUI version matrix

| OWUI version | Status |
|---|---|
| **≥ 0.9.5** | Fully supported. The progress embed uses the [`replace` flag for `embeds` events](https://github.com/open-webui/open-webui/commit/aa51ce482c161fb423767c41e8166197dce2d11b), so each conversation has exactly one live progress iframe instead of N stacked snapshots on reload. |
| **0.9.2 – 0.9.4** | Functional. The `replace` flag is silently ignored, so reloading a finished run will show stacked progress embeds. Live progress during a run is unaffected. |
| **< 0.9.2** | Untested. Async-ORM-aware endpoints in this version range should work via REST, but the progress embed format may render differently. |

## Chat persistence caveat

Open WebUI's `GET /api/v1/chats/{id}` filters by
`chat.user_id == authenticated_user_id`. **Admin role does not bypass this
filter.** If the `DR_OWUI_API_KEY` belongs to user A and the user invoking
Deep Research is user B, persistence to `chat.deepResearch` silently
fails. The research itself still runs.

**Mitigation:**

- **OWUI Function runtime** — the caller's `Authorization` header is
  passed through, so chat persistence works correctly without any extra
  configuration.
- **Pipelines / OpenAPI Tool / MCP runtimes** — the API key is fixed at
  the container level, so persistence only works for chats owned by that
  key's user. Either use a per-user key, or accept that the report still
  reaches the user (it is appended to the assistant message) even though
  the structured `chat.deepResearch` checkpoint is not written.

See also [troubleshooting › OWUI KB persistence "silently does nothing"](./troubleshooting.md#owui-kb-persistence-silently-does-nothing-but-chat-succeeds).
