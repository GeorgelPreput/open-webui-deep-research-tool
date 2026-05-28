import asyncio
import collections

import numpy as np

from deep_research.core.text import materialize_embedding, snapshot_embedding, stable_text_key


class LRUBytesBoundedCache:
    _PER_ENTRY_OVERHEAD = 512

    def __init__(self, max_bytes: int, max_entries: int = 50_000):
        self._od: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
        self._max_bytes = int(max_bytes)
        self._max_entries = int(max_entries)
        self._bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self._lock = asyncio.Lock()

    def _entry_bytes(self, arr: np.ndarray) -> int:
        return int(arr.nbytes) + self._PER_ENTRY_OVERHEAD

    async def _lru_get(self, key: str):
        async with self._lock:
            arr = self._od.get(key)
            if arr is None:
                self.miss_count += 1
                return None
            self._od.move_to_end(key)
            self.hit_count += 1
        return materialize_embedding(arr)

    async def _lru_set(self, key: str, embedding):
        snapshot = snapshot_embedding(embedding)
        if snapshot is None:
            return
        entry_bytes = self._entry_bytes(snapshot)
        async with self._lock:
            old = self._od.pop(key, None)
            if old is not None:
                self._bytes -= self._entry_bytes(old)
            self._od[key] = snapshot
            self._bytes += entry_bytes
            while (
                self._bytes > self._max_bytes or len(self._od) > self._max_entries
            ) and self._od:
                _, evicted = self._od.popitem(last=False)
                self._bytes -= self._entry_bytes(evicted)
                self.eviction_count += 1

    def stats(self) -> dict[str, float | int]:
        total = self.hit_count + self.miss_count
        return {
            "entries": len(self._od),
            "bytes": self._bytes,
            "max_bytes": self._max_bytes,
            "max_entries": self._max_entries,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "evictions": self.eviction_count,
            "hit_rate": (self.hit_count / total) if total else 0.0,
        }


class EmbeddingCache(LRUBytesBoundedCache):
    def __init__(self, max_bytes: int, max_entries: int = 50_000):
        super().__init__(max_bytes=max_bytes, max_entries=max_entries)

    async def get(self, text_key):
        return await self._lru_get(stable_text_key(text_key))

    async def set(self, text_key, embedding):
        await self._lru_set(stable_text_key(text_key), embedding)


class TransformationCache(LRUBytesBoundedCache):
    def __init__(self, max_bytes: int, max_entries: int = 25_000):
        super().__init__(max_bytes=max_bytes, max_entries=max_entries)

    @staticmethod
    def _make_key(text, transform_id) -> str:
        return f"{stable_text_key(text)}_{stable_text_key(transform_id)}"

    async def get(self, text, transform_id):
        return await self._lru_get(self._make_key(text, transform_id))

    async def set(self, text, transform_id, transformed_embedding):
        await self._lru_set(self._make_key(text, transform_id), transformed_embedding)
