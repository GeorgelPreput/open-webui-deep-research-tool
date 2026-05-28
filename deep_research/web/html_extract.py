import asyncio
import html
import logging
import re

from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.web.html_extract")


async def extract_text_from_html(ctx: RunContext, html_content: str) -> str:
    """HTML->text extractor used by retained code paths.

    After the legacy HTML fetch was removed from fetch_content, this
    method is only invoked from:
      - _try_primary_web_flow when POST_CLEAN_PRIMARY_OUTPUT is enabled,
      - fetch_from_archive for archived HTML pages.
    It remains a BeautifulSoup-then-regex extractor; it is not a fallback
    for live HTML fetching.
    """
    try:
        # Try BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup

            # Create a task for BS4 extraction
            def extract_with_bs4():
                # First unescape HTML entities properly
                unescaped_content = html.unescape(html_content)

                soup = BeautifulSoup(unescaped_content, "html.parser")

                # Remove common navigation elements by tag
                for element in soup(
                    [
                        "script",
                        "style",
                        "head",
                        "iframe",
                        "noscript",
                        "nav",
                        "header",
                        "footer",
                        "aside",
                        "form",
                    ]
                ):
                    element.decompose()

                # Remove common menu and navigation classes - expanded list
                nav_patterns = [
                    "menu",
                    "nav",
                    "header",
                    "footer",
                    "sidebar",
                    "dropdown",
                    "ibar",
                    "navigation",
                    "navbar",
                    "topbar",
                    "tab",
                    "toolbar",
                    "section",
                    "submenu",
                    "subnav",
                    "panel",
                    "drawer",
                    "accordion",
                    "toc",
                    "login",
                    "signin",
                    "auth",
                    "user-login",
                    "authType",
                ]

                # Case-insensitive class matching with partial matches
                for element in soup.find_all(
                    class_=lambda c: bool(
                        c and any(x.lower() in c.lower() for x in nav_patterns)
                    )
                ):
                    element.decompose()

                # Remove all unordered lists that contain mostly links (likely menus)
                for ul in soup.find_all("ul"):
                    links = ul.find_all("a")
                    list_items = ul.find_all("li")

                    # If it contains links and either:
                    # 1. Most children are links, or
                    # 2. There are many list items (10+)
                    # Then it's likely a navigation menu
                    if links and (
                        (list_items and len(links) / len(list_items) > 0.7)
                        or len(links) >= 10
                        or len(list_items) >= 10
                    ):
                        ul.decompose()

                # Extract text with proper whitespace handling
                text = soup.get_text(" ", strip=True)

                # Normalize whitespace while preserving intended breaks
                # Replace multiple spaces with a single space
                text = re.sub(r" {2,}", " ", text)

                # Fix common issues with periods and spaces
                text = re.sub(
                    r"\.([A-Z])", ". \\1", text
                )  # Fix "years.Today's" -> "years. Today's"

                # Process text line by line to better handle paragraph breaks
                lines = text.split("\n")
                processed_lines = []

                for line in lines:
                    # Remove excess whitespace within each line
                    line = re.sub(r"\s+", " ", line).strip()
                    if line:
                        processed_lines.append(line)

                # Join with proper paragraph breaks
                return "\n\n".join(processed_lines)

            # Run in executor to avoid blocking
            loop = asyncio.get_running_loop()
            bs4_extraction_task = loop.run_in_executor(None, extract_with_bs4)
            bs4_result = await asyncio.wait_for(bs4_extraction_task, timeout=5.0)

            # If BS4 extraction gave substantial content, use it
            if bs4_result and len(bs4_result) > len(html_content) * 0.1:
                return bs4_result

            # Otherwise fall back to the regex version
            # Quick regex extraction first

            # First unescape HTML entities properly
            unescaped_content = html.unescape(html_content)

            # Remove script and style tags
            content = re.sub(
                r"(?i)<script[^>]*>.*?</script>",
                " ",
                unescaped_content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            content = re.sub(
                r"(?i)<style[^>]*>.*?</style>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(
                r"(?i)<head[^>]*>.*?</head>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(r"(?i)<nav[^>]*>.*?</nav>", " ", content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(
                r"(?i)<header[^>]*>.*?</header>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(
                r"(?i)<footer[^>]*>.*?</footer>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )

            # Remove HTML tags
            content = re.sub(r"<[^>]*>", " ", content)

            # Fix common issues with periods and spaces
            content = re.sub(
                r"\.([A-Z])", ". \\1", content
            )  # Fix "years.Today's" -> "years. Today's"

            # Cleanup whitespace
            content = re.sub(r"\s+", " ", content).strip()

            return content

        except Exception as e:
            logger.warning(
                f"BeautifulSoup extraction failed: {e}, using regex fallback"
            )
            # Use regex version if BS4 fails
            # First unescape HTML entities properly
            unescaped_content = (
                html.unescape(html_content)
                if isinstance(html_content, str)
                else html_content
            )

            # Remove script and style tags
            content = re.sub(
                r"(?i)<script[^>]*>.*?</script>",
                " ",
                unescaped_content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            content = re.sub(
                r"(?i)<style[^>]*>.*?</style>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(
                r"(?i)<head[^>]*>.*?</head>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(r"(?i)<nav[^>]*>.*?</nav>", " ", content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(
                r"(?i)<header[^>]*>.*?</header>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(
                r"(?i)<footer[^>]*>.*?</footer>", " ", content, flags=re.DOTALL | re.IGNORECASE
            )

            # Remove HTML tags
            content = re.sub(r"<[^>]*>", " ", content)

            # Fix common issues with periods and spaces
            content = re.sub(
                r"\.([A-Z])", ". \\1", content
            )  # Fix "years.Today's" -> "years. Today's"

            # Cleanup whitespace
            content = re.sub(r"\s+", " ", content).strip()

            return content

    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        # Simple fallback - remove all HTML tags and unescape HTML entities
        try:
            # Unescape HTML entities
            if isinstance(html_content, str):
                unescaped = html.unescape(html_content)
            else:
                unescaped = html_content

            # Remove HTML tags
            text = re.sub(r"<[^>]*>", " ", unescaped)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            return text
        except Exception:
            return html_content
