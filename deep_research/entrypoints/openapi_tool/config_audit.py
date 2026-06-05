"""Startup configuration audit for the OpenAPI Tool Server.

Detects misconfiguration that silently degrades UX — missing public
base URL, non-admin writeback token, OWUI-side header forwarding — and
returns a structured warning list. Consumers:

  - ``server.py`` lifespan logs each warning with its stable ``code``
    and remediation text at startup.
  - ``GET /health`` returns the cached list as JSON for ``curl``-able
    operator inspection.
  - ``start_research_job`` appends ``OWUI_HEADERS_NOT_FORWARDED`` at
    runtime (one-shot per process) when an authenticated request
    arrives without ``X-OpenWebUI-Chat-Id``.

Fail-fast (server refuses to start) lives in ``server.py:lifespan``,
not here — this module produces warnings only.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from deep_research.adapter.client import OWUIClient
    from deep_research.config.valves import Valves

logger = logging.getLogger("deep_research.entrypoints.openapi.config_audit")


@dataclass(frozen=True)
class ConfigWarning:
    code: str
    severity: Literal["warning", "info"]
    message: str
    remediation: str


async def audit_writeback_configuration(
    valves: Valves,
    env: Mapping[str, str],
    writeback_client: OWUIClient | None,
) -> list[ConfigWarning]:
    """Audit the runtime config that controls writeback + iframe behaviour.

    Returns a list of :class:`ConfigWarning` covering: non-admin token,
    probe network/schema failure, and missing public base URL. The list
    is empty when everything is set correctly.

    The admin probe calls ``writeback_client.get_session_user()``; if
    ``writeback_client`` is None (writeback disabled or token unset) the
    probe checks are skipped.
    """
    warnings: list[ConfigWarning] = []

    if valves.jobs.writeback_enabled and writeback_client is not None:
        try:
            session_user = await writeback_client.get_session_user()
        except Exception as exc:
            warnings.append(
                ConfigWarning(
                    code="OWUI_API_KEY_PROBE_FAILED",
                    severity="warning",
                    message=(
                        "Could not verify the admin role of DR_OWUI_API_KEY: "
                        f"probe to GET /api/v1/auths/ failed ({exc!r}). "
                        "Writeback POSTs may fail at runtime."
                    ),
                    remediation=(
                        "Verify the OWUI base URL is reachable and the token "
                        "is valid. Once OWUI is reachable, restart the tool "
                        "server to re-run the probe."
                    ),
                )
            )
        else:
            role = session_user.get("role")
            if role != "admin":
                warnings.append(
                    ConfigWarning(
                        code="OWUI_API_KEY_NOT_ADMIN",
                        severity="warning",
                        message=(
                            "DR_OWUI_API_KEY is set but its OWUI role is "
                            f"{role!r}, not 'admin'. Writeback POSTs to "
                            "OWUI's per-message /event endpoint will return "
                            "401 because that endpoint only admin-bypasses "
                            "chat ownership for admin tokens."
                        ),
                        remediation=(
                            "Issue a fresh admin API key in OWUI's Admin "
                            "Settings → Users and update DR_OWUI_API_KEY."
                        ),
                    )
                )

    if not env.get("DR_OPENAPI_PUBLIC_BASE_URL"):
        warnings.append(
            ConfigWarning(
                code="MISSING_PUBLIC_BASE_URL",
                severity="info",
                message=(
                    "DR_OPENAPI_PUBLIC_BASE_URL is unset. The iframe's "
                    "polling URL will be derived from the inbound request's "
                    "host header, which fails when users access this tool "
                    "via a different hostname than OWUI uses internally "
                    "(reverse proxy, K8s internal service name)."
                ),
                remediation=(
                    "Set DR_OPENAPI_PUBLIC_BASE_URL to the URL the user's "
                    "browser uses to reach this tool server."
                ),
            )
        )

    return warnings
