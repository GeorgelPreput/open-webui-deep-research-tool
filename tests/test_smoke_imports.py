"""Smoke test: every module in the package imports cleanly.

Catches regressions like the `import aiohttp` slip-up where pyproject
removed the dep but a module still tried to import it.
"""
import importlib
import pkgutil

import deep_research


def _walk_modules(pkg) -> list[str]:
    out = []
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        out.append(info.name)
    return out


def test_all_modules_import():
    failures = []
    for mod in _walk_modules(deep_research):
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures.append(f"{mod}: {type(e).__name__}: {e}")
    assert not failures, "Import failures:\n" + "\n".join(failures)
