"""Regression: the regex-fallback script-tag stripper must also catch
``</script >`` (HTML5 permits whitespace before ``>`` in end tags).

CodeQL flagged the original ``</script>``-only regex as a bad tag filter
(py/bad-tag-filter); this test pins the fix.
"""

from __future__ import annotations

import re

# Mirrors the production regex in deep_research/web/html_extract.py.
SCRIPT_STRIP = re.compile(
    r"(?i)<script[^>]*>.*?</script\s*>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip(html_text: str) -> str:
    return SCRIPT_STRIP.sub(" ", html_text)


def test_script_close_with_no_whitespace():
    out = _strip("<p>before</p><script>alert(1)</script><p>after</p>")
    assert "alert" not in out
    assert "before" in out and "after" in out


def test_script_close_with_trailing_space():
    out = _strip("<p>before</p><script>alert(1)</script ><p>after</p>")
    assert "alert" not in out, out
    assert "before" in out and "after" in out


def test_script_close_with_newline_inside_tag():
    out = _strip("<p>before</p><script>alert(1)</script\n><p>after</p>")
    assert "alert" not in out, out


def test_script_close_with_tab():
    out = _strip("<p>before</p><script>alert(1)</script\t><p>after</p>")
    assert "alert" not in out, out


def test_uppercase_script_tag():
    out = _strip("<p>before</p><SCRIPT>alert(1)</SCRIPT ><p>after</p>")
    assert "alert" not in out, out
