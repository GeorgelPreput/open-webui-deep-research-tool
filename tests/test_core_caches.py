import pytest

from deep_research.core.caches import (
    EmbeddingCache,
    LRUBytesBoundedCache,
    TransformationCache,
)


@pytest.mark.asyncio
async def test_lru_basic_get_miss_then_hit():
    cache = LRUBytesBoundedCache(max_bytes=1_000_000)
    assert await cache._lru_get("k") is None
    await cache._lru_set("k", [1.0, 2.0, 3.0])
    val = await cache._lru_get("k")
    assert val == pytest.approx([1.0, 2.0, 3.0])
    assert cache.hit_count == 1
    assert cache.miss_count == 1


@pytest.mark.asyncio
async def test_lru_byte_bounded_eviction():
    # 4 bytes per float32 + 512 overhead per entry; cap to fit ~2 entries
    cache = LRUBytesBoundedCache(max_bytes=1100, max_entries=10_000)
    await cache._lru_set("a", [1.0])
    await cache._lru_set("b", [2.0])
    await cache._lru_set("c", [3.0])
    # "a" should have been evicted (FIFO order at insertion time)
    assert await cache._lru_get("a") is None
    assert cache.eviction_count >= 1


@pytest.mark.asyncio
async def test_lru_recency_protects_against_eviction():
    cache = LRUBytesBoundedCache(max_bytes=1100, max_entries=10_000)
    await cache._lru_set("a", [1.0])
    await cache._lru_set("b", [2.0])
    # Touch "a" so it becomes most recent
    await cache._lru_get("a")
    await cache._lru_set("c", [3.0])
    # "b" should be evicted, not "a"
    assert await cache._lru_get("a") is not None
    assert await cache._lru_get("b") is None


@pytest.mark.asyncio
async def test_embedding_cache_stable_key():
    cache = EmbeddingCache(max_bytes=1_000_000)
    await cache.set("hello world", [0.1, 0.2, 0.3])
    assert await cache.get("hello world") == pytest.approx([0.1, 0.2, 0.3])
    # Different text → cache miss
    assert await cache.get("goodbye") is None


@pytest.mark.asyncio
async def test_transformation_cache_composite_key():
    cache = TransformationCache(max_bytes=1_000_000)
    await cache.set("text", "transform-1", [1.0, 2.0])
    assert await cache.get("text", "transform-1") == pytest.approx([1.0, 2.0])
    # Same text, different transform → miss
    assert await cache.get("text", "transform-2") is None


@pytest.mark.asyncio
async def test_stats_reflect_activity():
    cache = LRUBytesBoundedCache(max_bytes=1_000_000)
    await cache._lru_set("k", [1.0, 2.0])
    await cache._lru_get("k")  # hit
    await cache._lru_get("missing")  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1
    assert stats["bytes"] > 0
