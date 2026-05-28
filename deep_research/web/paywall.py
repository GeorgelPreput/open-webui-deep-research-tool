import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Any

import httpx

from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.web.paywall")

try:
    from fake_useragent import UserAgent as _UserAgent
    _ua_provider: Any | None = _UserAgent()
except Exception:
    _ua_provider = None

_FALLBACK_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/123.0.0.0 Safari/537.36",
]


async def fetch_pdf_via_legacy_download(ctx: RunContext, url: str) -> str:
    """Download and extract PDF content using the retained legacy path.

    This helper is used only for PDF-classified URLs. It keeps the
    anti-blocking machinery (rate limiting, randomized user-agent,
    spoofed headers/cookies/referrers), supports archive rescue on
    HTTP 403/271, and routes bytes through
    _extract_pdf_with_primary_fallback.
    """
    from deep_research.budget.tokens import count_tokens
    from deep_research.persistence.kb import persist_selected_source
    from deep_research.web.fetch import extract_pdf_with_primary_fallback, fetch_from_archive
    try:
        state = ctx.state.get_state(ctx.conversation_id)
        url_results_cache = state.get("url_results_cache", {})
        master_source_table = state.get("master_source_table", {})
        domain_session_map = state.get("domain_session_map", {})

        # Extract domain for session management and tracking
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Domain-specific rate limiting
        # Check if we've recently accessed this domain
        if domain in domain_session_map:
            domain_info = domain_session_map[domain]
            last_access_time = domain_info.get("last_visit", 0)
            current_time = time.time()
            time_since_last_access = current_time - last_access_time

            # If we accessed this domain recently, delay to avoid rate limiting
            # Only delay if less than 2-3 seconds have passed since last access
            if time_since_last_access < 3.0:
                # Add randomness to the delay (between 2-3 seconds total between requests)
                base_delay = 2.0
                jitter = random.uniform(0.1, 1.0)
                delay_time = max(0, base_delay - time_since_last_access + jitter)

                if delay_time > 0.1:  # Only log/delay if significant
                    logger.info(
                        f"Rate limiting for domain {domain}: Delaying for {delay_time:.2f} seconds"
                    )
                    await asyncio.sleep(delay_time)

        if _ua_provider is not None:
            try:
                random_user_agent = _ua_provider.random
            except Exception:
                random_user_agent = random.choice(_FALLBACK_USER_AGENTS)
        else:
            random_user_agent = random.choice(_FALLBACK_USER_AGENTS)

        # Create comprehensive browser fingerprint headers
        headers = {
            "User-Agent": random_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Chromium";v="116", "Google Chrome";v="116", "Not=A?Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }

        # Add EZproxy-like headers
        university_ips = {
            "Harvard": "128.103.192." + str(random.randint(1, 254)),
            "Princeton": "128.112.203." + str(random.randint(1, 254)),
            "MIT": "18.7."
            + str(random.randint(1, 254))
            + "."
            + str(random.randint(1, 254)),
            "Stanford": "171.64."
            + str(random.randint(1, 254))
            + "."
            + str(random.randint(1, 254)),
        }

        chosen_university = random.choice(list(university_ips.keys()))
        headers["X-Forwarded-For"] = university_ips[chosen_university]
        headers["X-Requested-With"] = "XMLHttpRequest"

        # Add institutional cookies
        if domain not in domain_session_map:
            domain_session_map[domain] = {
                "cookies": {},
                "last_visit": 0,
                "visit_count": 0,
            }

        domain_session_map[domain]["cookies"] = {
            "ezproxy_authenticated": "true",
            "institution": chosen_university,
            "access_token": "academic_access_" + str(int(time.time())),
        }

        # Use a mix of academic and standard referrers
        referrers = [
            f"https://library.{chosen_university.lower()}.edu/find/",
            "https://scholar.google.com/scholar?q=",
            "https://www.google.com/search?q=",
            "https://www.bing.com/search?q=",
            "https://search.yahoo.com/search?p=",
            "https://www.scopus.com/record/display.uri",
            "https://www.webofscience.com/wos/woscc/full-record/",
            "https://www.sciencedirect.com/search?",
            "https://www.base-search.net/Search/Results?",
        ]

        # Create rich search terms
        search_terms = [
            parsed_url.path.split("/")[-1].replace(".pdf", "").replace("-", " "),
            (
                "doi " + parsed_url.path.split("/")[-1]
                if "/" in parsed_url.path
                else domain
            ),
            domain + " research",
            domain,
            domain + " publication",
        ]

        # Filter out empty or very short ones
        search_terms = [term for term in search_terms if len(term.strip()) > 3]

        # Choose a referrer and term - use hash of domain for consistency while still appearing varied
        domain_hash = hash(domain)
        chosen_referrer = referrers[domain_hash % len(referrers)]
        search_term = search_terms[0] if search_terms else domain
        if len(search_terms) > 1:
            search_term = search_terms[domain_hash % len(search_terms)]

        # Apply the search term
        search_term = search_term.replace(" ", "+")
        headers["Referer"] = chosen_referrer + search_term

        domain_session = domain_session_map[domain]
        domain_session["visit_count"] += 1

        domain_session["last_visit"] = time.time()
        ctx.state.update_state(ctx.conversation_id, "domain_session_map", domain_session_map)

        # Stored cookies are always a plain dict (either the spoofed-cookie
        # literal set above or the snapshot we write back below from
        # session.cookies after each successful request).
        cookie_dict = domain_session_map[domain].get("cookies", {})

        async with httpx.AsyncClient(
            verify=False,
            cookies=cookie_dict,
            timeout=httpx.Timeout(20.0),
            follow_redirects=True,
        ) as session:
            try:
                response = await session.get(url, headers=headers)
            except httpx.TimeoutException:
                logger.error(f"Timeout fetching content from {url}")
                return f"Timeout while fetching content from {url}"

            # Persist refreshed cookies for next visit to this domain
            if domain in domain_session_map:
                domain_session_map[domain]["cookies"] = dict(session.cookies)
                ctx.state.update_state(
                    ctx.conversation_id, "domain_session_map", domain_session_map
                )

            status = response.status_code
            if status == 200:
                pdf_content = response.content
                extracted_content = await extract_pdf_with_primary_fallback(
                    ctx, pdf_content, url, "application/pdf"
                )

                # Cap cached content to 3x MAX_RESULT_TOKENS (full text still
                # returned to the caller and persisted to the KB).
                if extracted_content:
                    tokens = await count_tokens(ctx, extracted_content)
                    token_limit = ctx.valves.web.max_result_tokens * 3
                    if tokens > token_limit:
                        char_limit = int(
                            len(extracted_content) * (token_limit / tokens)
                        )
                        extracted_content_to_cache = extracted_content[:char_limit]
                        logger.info(
                            f"Limiting cached PDF content for URL {url} "
                            f"from {tokens} to {token_limit} tokens"
                        )
                    else:
                        extracted_content_to_cache = extracted_content
                    url_results_cache[url] = extracted_content_to_cache
                else:
                    url_results_cache[url] = extracted_content

                ctx.state.update_state(
                    ctx.conversation_id, "url_results_cache", url_results_cache
                )

                if url not in master_source_table:
                    title = (
                        url.split("/")[-1]
                        .replace(".pdf", "")
                        .replace("-", " ")
                        .replace("_", " ")
                    )
                    source_id = f"S{len(master_source_table) + 1}"
                    master_source_table[url] = {
                        "id": source_id,
                        "title": title,
                        "content_preview": extracted_content[:500]
                        if isinstance(extracted_content, str)
                        else "",
                        "source_type": "pdf",
                        "accessed_date": datetime.now().strftime("%Y-%m-%d"),
                        "cited_in_sections": set(),
                    }
                    ctx.state.update_state(
                        ctx.conversation_id,
                        "master_source_table",
                        master_source_table,
                    )

                if (
                    isinstance(extracted_content, str)
                    and extracted_content.strip()
                ):
                    try:
                        await persist_selected_source(
                            ctx=ctx,
                            url=url,
                            full_text=extracted_content,
                            title=master_source_table.get(url, {}).get(
                                "title", url
                            ),
                            source_type="pdf",
                            archived=False,
                        )
                    except Exception as e:
                        logger.warning(f"KB persistence failed for {url}: {e}")

                return extracted_content

            if status in (403, 271):
                logger.info(
                    f"Received {status} for PDF {url}, trying archive.org"
                )
                archive_content = await fetch_from_archive(ctx, url, session)
                if archive_content:
                    return archive_content
                logger.error(
                    f"Error fetching URL {url}: HTTP {status} (archive fallback failed)"
                )
                return (
                    f"Error fetching content: HTTP status {status} "
                    f"(archive fallback failed)"
                )

            logger.error(f"Error fetching URL {url}: HTTP {status}")
            return f"Error fetching content: HTTP status {status}"

    except httpx.ConnectError as e:
        logger.error(f"Connection error for {url}: {e}")
        return f"Connection error: {e}"
    except httpx.NetworkError as e:
        logger.error(f"Network error for {url}: {e}")
        return f"Network error: {e}"
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return f"Error fetching content: {e}"
