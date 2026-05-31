"""Regression tests for core fixes and dead-code removal.

Covers:
  BUG 37-40 - stable_text_key produces process-stable, collision-resistant keys.
  BUG 2     - RunContext carries a research_date field (default "").
  BUG 43-50 - removed dead code stays removed; the live symbols stay present.
"""
import dataclasses
import importlib

import pytest

from deep_research.core.text import stable_text_key
from deep_research.core.types import RunContext

# --- BUG 37-40: stable_text_key ---------------------------------------------

def test_stable_text_key_is_deterministic():
    assert stable_text_key("the quick brown fox") == stable_text_key("the quick brown fox")


def test_stable_text_key_is_hex_sha256():
    key = stable_text_key("anything")
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_stable_text_key_distinguishes_inputs():
    assert stable_text_key("abc") != stable_text_key("abd")


def test_stable_text_key_handles_none_and_non_str():
    assert stable_text_key(None) == stable_text_key("")
    assert stable_text_key(123) == stable_text_key("123")
    assert stable_text_key(["a", "b"]) == stable_text_key(str(["a", "b"]))


# --- BUG 2: research_date on RunContext -------------------------------------

def test_run_context_has_research_date_field():
    fields = {f.name: f for f in dataclasses.fields(RunContext)}
    assert "research_date" in fields
    assert fields["research_date"].default == ""


# --- BUG 43-50: dead code removed, live code intact -------------------------

def test_verify_citations_removed_survivors_kept():
    from deep_research.synthesis import verify

    assert not hasattr(verify, "verify_citations")
    assert hasattr(verify, "verify_citation_batch")
    assert hasattr(verify, "add_verification_note")


def test_unused_client_methods_removed():
    from deep_research.adapter.client import OWUIClient

    assert not hasattr(OWUIClient, "get_file_content")
    assert not hasattr(OWUIClient, "process_text")
    # Live file/KB methods stay.
    for keep in ("process_file", "process_web_url", "get_file", "upload_file", "create_kb"):
        assert hasattr(OWUIClient, keep)


def test_unused_models_removed_used_models_kept():
    from deep_research.adapter import models

    for gone in (
        "FileContentResponse",
        "ChatCompletionResponse",
        "ChatCompletionChoice",
        "ChatCompletionMessage",
        "WebSearchResponse",
    ):
        assert not hasattr(models, gone), gone
    for keep in (
        "QueryCollectionResponse",
        "ProcessFileResponse",
        "ProcessWebResponse",
        "FileUploadResponse",
        "KBResponse",
        "ModelInfo",
    ):
        assert hasattr(models, keep), keep


def test_extraction_quality_constant_removed():
    from deep_research.config import constants

    assert not hasattr(constants, "EXTRACTION_QUALITY")
    assert hasattr(constants, "REQUIRE_EXTRACTION_QUALITY")


def test_resolve_conversation_id_helper_removed():
    from deep_research.entrypoints.owui_function import pipe

    assert not hasattr(pipe, "_resolve_conversation_id")


def test_core_logging_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("deep_research.core.logging")
