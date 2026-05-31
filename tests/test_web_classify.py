"""Coverage tests for the pure URL/quality helpers in web/classify.py."""
import pytest

from deep_research.web.classify import (
    check_extraction_quality,
    classify_url,
    url_fallback_title,
)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com/report.pdf", "pdf"),
        ("https://example.com/report.PDF?download=1", "pdf"),
        ("https://example.com/page.html", "html"),
        ("https://example.com/", "html"),
        ("https://example.com/doc.pdf#section", "pdf"),
        ("", "unknown"),
        (None, "unknown"),
        (1234, "unknown"),
    ],
)
def test_classify_url(url, expected):
    assert classify_url(None, url) == expected


def test_url_fallback_title_from_filename():
    title = url_fallback_title(None, "https://example.com/papers/My_Cool-Doc.pdf")
    assert title == "My Cool Doc"


def test_url_fallback_title_uses_netloc_when_no_path():
    assert url_fallback_title(None, "https://example.com") == "example.com"


def test_url_fallback_title_handles_garbage():
    # Must not raise on a non-URL string.
    assert isinstance(url_fallback_title(None, "not a url"), str)


def test_extraction_quality_accepts_substantial_text():
    assert check_extraction_quality(None, "word " * 40) is True  # > 80 chars


def test_extraction_quality_rejects_short_text():
    assert check_extraction_quality(None, "too short") is False


def test_extraction_quality_rejects_empty_and_non_str():
    assert check_extraction_quality(None, "") is False
    assert check_extraction_quality(None, None) is False


def test_extraction_quality_rejects_error_prefixes():
    assert check_extraction_quality(None, "Error: " + "x" * 100) is False
    assert check_extraction_quality(None, "Could not extract " + "y" * 100) is False


def test_extraction_quality_rejects_html_heavy_content():
    html = "<div>" * 40  # starts with "<" and many tags
    assert check_extraction_quality(None, html) is False
