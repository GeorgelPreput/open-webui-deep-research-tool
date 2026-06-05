import asyncio
import logging
import re

import httpx
import numpy as np

from deep_research.core.types import RunContext
from deep_research.progress.events import StatusEvent

logger = logging.getLogger("deep_research.semantics.vocabulary")

# Emit a StatusEvent every N batches during vocab embedding generation so the
# iframe revision advances even when the embedding throttle paces requests
# slowly. ~25 batches at the documented 1 req/s pacing = ~25s between pulses,
# enough to keep the UI alive without flooding the writeback channel.
_VOCAB_PROGRESS_EVERY_N_BATCHES = 25

_vocabulary_cache: list[str] | None = None
_vocabulary_embeddings: dict[str, list[float]] | None = None
# Two distinct locks: the embeddings loader calls the text-vocabulary loader
# while holding its own lock, so they MUST NOT share one — asyncio.Lock is not
# reentrant and a single shared lock deadlocks (nested acquire).
_vocab_load_lock = asyncio.Lock()
_vocab_emb_load_lock = asyncio.Lock()


async def _emit_vocab_progress(ctx: RunContext, done: int, total: int) -> None:
    """Best-effort StatusEvent emit so a long vocab-embedding load keeps
    the iframe revision moving.

    Why: the load can take many minutes on a cold cache (10k words at a
    throttled embeddings rate). Without these pulses, the engine sits in
    a tight loop with the sink silent, the iframe polls return the same
    snapshot, and the user-visible chat stays pinned at the last
    ``replace`` event. The pulses don't carry payload — they only bump
    the revision counter so the iframe shows "alive".
    """
    events = getattr(ctx, "events", None)
    if events is None:
        return
    try:
        await events.emit(
            StatusEvent(
                description=f"Loading vocabulary embeddings ({done}/{total})",
                level="info",
                done=False,
            )
        )
    except Exception:  # noqa: BLE001 — best-effort: never let progress break the load
        logger.debug("Vocab progress emit failed", exc_info=True)


async def create_context_vocabulary(
    ctx: RunContext, context_text: str, min_size: int = 1000
) -> list[str]:
    logger.info("Creating vocabulary from context as fallback")

    words = re.findall(r"\b[a-zA-Z]{4,}\b", context_text.lower())

    unique_words = list(set(words))
    logger.info(f"Created context vocabulary with {len(unique_words)} words")

    return unique_words


async def load_vocabulary(ctx: RunContext) -> list[str] | None:
    global _vocabulary_cache
    if _vocabulary_cache is not None:
        return _vocabulary_cache

    async with _vocab_load_lock:
        if _vocabulary_cache is not None:
            return _vocabulary_cache

        cache_dir = ctx.config.data_dir / "deep_research"
        cache_dir.mkdir(parents=True, exist_ok=True)
        disk_path = str(cache_dir / "vocabulary.txt")
        try:
            with open(disk_path) as f:
                words = [w.strip() for w in f.readlines() if w.strip()]
            if words:
                _vocabulary_cache = words
                logger.info(
                    f"Loaded {len(_vocabulary_cache)} words vocabulary from disk cache"
                )
                return _vocabulary_cache
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not read vocabulary from disk cache: {e}")

        try:
            url = "https://www.mit.edu/~ecprice/wordlist.10000"
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    text = response.text
                    _vocabulary_cache = [
                        word.strip()
                        for word in text.splitlines()
                        if word.strip()
                    ]
                    logger.info(
                        f"Loaded {len(_vocabulary_cache)} words vocabulary"
                    )
                    try:
                        with open(disk_path, "w") as f:
                            f.write(text)
                        logger.info(
                            f"Saved vocabulary to disk cache: {disk_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not save vocabulary to disk cache: {e}"
                        )
                    return _vocabulary_cache
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")

            context_text = ""
            conv_state = ctx.state.get_state(ctx.conversation_id)
            results_history = conv_state.get("results_history", [])
            search_history = conv_state.get("search_history", [])
            section_synthesized_content = conv_state.get(
                "section_synthesized_content", {}
            )

            if results_history:
                for result in results_history[-5:]:
                    context_text += result.get("content", "") + " "

            if search_history:
                context_text += " ".join(search_history) + " "

            if section_synthesized_content:
                for content in list(section_synthesized_content.values())[:3]:
                    context_text += content + " "

            if len(context_text) < 5000:
                logger.error("Insufficient context for vocabulary creation")
                return None

            _vocabulary_cache = await create_context_vocabulary(
                ctx, context_text
            )
            return _vocabulary_cache

    # Reached only if the network fetch returned a non-200 without raising;
    # caller treats None as "no vocabulary available".
    return None


def vocab_embeddings_disk_path(ctx: RunContext) -> str:
    model_name = ctx.valves.models.embedding_model or "default"
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:64]
    cache_dir = ctx.config.data_dir / "deep_research"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"vocab_emb_{safe_model}.npz")


async def load_vocabulary_embeddings(ctx: RunContext) -> dict[str, list[float]]:
    global _vocabulary_embeddings
    if _vocabulary_embeddings is not None:
        return _vocabulary_embeddings

    async with _vocab_emb_load_lock:
        if _vocabulary_embeddings is not None:
            return _vocabulary_embeddings

        disk_path = vocab_embeddings_disk_path(ctx)
        try:
            data = np.load(disk_path)
            words = data["words"].tolist()
            embeddings = data["embeddings"].tolist()
            _vocabulary_embeddings = {w: e for w, e in zip(words, embeddings, strict=False)}
            logger.info(
                f"Loaded {len(_vocabulary_embeddings)} vocabulary embeddings from disk cache"
            )
            return _vocabulary_embeddings
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(
                f"Could not load vocabulary embeddings from disk cache: {e}"
            )

        conv_state = ctx.state.get_state(ctx.conversation_id)
        cached_embeddings = conv_state.get("vocabulary_embeddings")
        if cached_embeddings:
            _vocabulary_embeddings = cached_embeddings
            logger.info(
                f"Loaded {len(_vocabulary_embeddings)} vocabulary embeddings from state"
            )
            return _vocabulary_embeddings

        # Under embedding-throttle pressure with no disk cache, skip vocabulary
        # generation entirely. The 10k-word load is the biggest single embedding
        # burst in a run; downstream dimension translation falls through to
        # "Dimension N" labels which are still informative.
        diag = getattr(ctx, "embeddings_diagnostics", None)
        if diag is not None and diag.degraded:
            logger.warning(
                "Skipping vocabulary embedding generation: embedding throttle "
                "is in degraded mode and no disk cache is present"
            )
            diag.record_skipped()
            return {}

        vocab = await load_vocabulary(ctx)
        if not vocab:
            logger.error("Failed to load vocabulary for embeddings")
            return {}

        try:
            # batch_size is capped by the embedding throttle's batch_max_inputs;
            # the embedding client also re-chunks internally, but capping here
            # keeps the per-batch progress log honest.
            batch_size = max(
                1, int(ctx.valves.embeddings_throttle.batch_max_inputs or 512)
            )
            logger.info(
                f"Generating embeddings for {len(vocab)} vocabulary words (batch_size={batch_size})"
            )
            await _emit_vocab_progress(ctx, 0, len(vocab))
            all_embeddings = []
            embedding_model = ctx.valves.models.embedding_model
            batch_index = 0
            for i in range(0, len(vocab), batch_size):
                batch = vocab[i : i + batch_size]
                batch_result = await ctx.embeddings.embeddings(embedding_model, batch)
                all_embeddings.extend(batch_result)
                batch_index += 1
                if batch_index % _VOCAB_PROGRESS_EVERY_N_BATCHES == 0:
                    await _emit_vocab_progress(ctx, len(all_embeddings), len(vocab))

            _vocabulary_embeddings = {
                word: emb
                for word, emb in zip(vocab, all_embeddings, strict=False)
                if emb
            }
        except Exception as e:
            logger.error(f"Failed to generate vocabulary embeddings: {e}")
            return {}

        if not _vocabulary_embeddings:
            return {}

        logger.info(
            f"Generated embeddings for {len(_vocabulary_embeddings)} vocabulary words"
        )
        await _emit_vocab_progress(ctx, len(_vocabulary_embeddings), len(vocab))
        try:
            valid_words = list(_vocabulary_embeddings.keys())
            valid_embs = list(_vocabulary_embeddings.values())
            np.savez_compressed(
                disk_path,
                words=np.array(valid_words, dtype="U64"),
                embeddings=np.array(valid_embs, dtype=np.float32),
            )
            logger.info(f"Saved vocabulary embeddings to disk cache: {disk_path}")
        except Exception as e:
            logger.warning(
                f"Could not save vocabulary embeddings to disk cache: {e}"
            )
        return _vocabulary_embeddings
