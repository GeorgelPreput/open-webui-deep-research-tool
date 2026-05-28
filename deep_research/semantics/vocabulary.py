import asyncio
import logging
import re

import httpx
import numpy as np

from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.semantics.vocabulary")

_vocabulary_cache: list[str] | None = None
_vocabulary_embeddings: dict[str, list[float]] | None = None
_vocab_load_lock = asyncio.Lock()


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
    try:
        model_name = getattr(ctx.config, "embedding_model", "") or "default"
    except Exception:
        model_name = "default"
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:64]
    cache_dir = ctx.config.data_dir / "deep_research"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"vocab_emb_{safe_model}.npz")


async def load_vocabulary_embeddings(ctx: RunContext) -> dict[str, list[float]]:
    global _vocabulary_embeddings
    if _vocabulary_embeddings is not None:
        return _vocabulary_embeddings

    async with _vocab_load_lock:
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

        vocab = await load_vocabulary(ctx)
        if not vocab:
            logger.error("Failed to load vocabulary for embeddings")
            return {}

        try:
            batch_size = 512
            logger.info(
                f"Generating embeddings for {len(vocab)} vocabulary words (batch_size={batch_size})"
            )
            all_embeddings = []
            embedding_model = ctx.valves.models.research_model
            for i in range(0, len(vocab), batch_size):
                batch = vocab[i : i + batch_size]
                batch_result = await ctx.client.embeddings(embedding_model, batch)
                all_embeddings.extend(batch_result)

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
