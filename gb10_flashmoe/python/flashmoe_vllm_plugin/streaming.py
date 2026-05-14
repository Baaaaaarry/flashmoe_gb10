from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Lock
from typing import Iterable

from .config import FlashMoEConfig
from .policy import CachePolicyFeatures, CachePolicyModel
from .q3like import unpack_q3_tensor


ExpertKey = tuple[int, int]


@dataclass(slots=True)
class ExpertManifestEntry:
    layer_id: int
    expert_id: int
    path: str
    offset: int
    size_bytes: int
    format: str = "dense"

    @property
    def key(self) -> ExpertKey:
        return (self.layer_id, self.expert_id)


@dataclass(slots=True)
class StreamingStats:
    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prefetch_requests: int = 0
    bytes_read: int = 0
    read_ops: int = 0
    evictions: int = 0
    current_resident_bytes: int = 0
    peak_resident_bytes: int = 0

    def hit_rate(self) -> float:
        return self.cache_hits / self.requests if self.requests else 0.0

    def miss_rate(self) -> float:
        return self.cache_misses / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prefetch_requests": self.prefetch_requests,
            "bytes_read_gb": self.bytes_read / 1e9,
            "read_ops": self.read_ops,
            "evictions": self.evictions,
            "current_resident_gb": self.current_resident_bytes / 1e9,
            "peak_resident_gb": self.peak_resident_bytes / 1e9,
            "hit_rate": self.hit_rate(),
            "miss_rate": self.miss_rate(),
        }


@dataclass(slots=True)
class ResidentExpert:
    entry: ExpertManifestEntry
    payload: bytes
    last_step: int
    access_count: int = 0
    prefetched: bool = False
    tensors: dict[str, object] | None = None

    @property
    def size_bytes(self) -> int:
        return self.entry.size_bytes


class ExpertManifest:
    def __init__(self, entries: dict[ExpertKey, ExpertManifestEntry]):
        self._entries = entries

    @classmethod
    def from_file(cls, path: str | Path) -> "ExpertManifest":
        raw = json.loads(Path(path).read_text())
        entries = {
            (int(item["layer_id"]), int(item["expert_id"])): ExpertManifestEntry(
                layer_id=int(item["layer_id"]),
                expert_id=int(item["expert_id"]),
                path=str(item["path"]),
                offset=int(item.get("offset", 0)),
                size_bytes=int(item["size_bytes"]),
                format=str(item.get("format", "dense")),
            )
            for item in raw["entries"]
        }
        return cls(entries)

    def get(self, key: ExpertKey) -> ExpertManifestEntry:
        return self._entries[key]

    def __contains__(self, key: ExpertKey) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class SSDExpertStore:
    def __init__(self, manifest: ExpertManifest, pread_chunks: int):
        self.manifest = manifest
        self.pread_chunks = max(1, pread_chunks)

    def read_entry(self, entry: ExpertManifestEntry, stats: StreamingStats) -> bytes:
        fd = os.open(entry.path, os.O_RDONLY)
        try:
            if self.pread_chunks == 1 or entry.size_bytes < self.pread_chunks:
                payload = os.pread(fd, entry.size_bytes, entry.offset)
                stats.bytes_read += len(payload)
                stats.read_ops += 1
                return payload

            chunk_size = (entry.size_bytes + self.pread_chunks - 1) // self.pread_chunks
            parts: list[bytes] = []
            remaining = entry.size_bytes
            cursor = entry.offset
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = os.pread(fd, to_read, cursor)
                parts.append(chunk)
                stats.bytes_read += len(chunk)
                stats.read_ops += 1
                remaining -= to_read
                cursor += to_read
            return b"".join(parts)
        finally:
            os.close(fd)


class StreamingExpertCache:
    def __init__(self, config: FlashMoEConfig, manifest: ExpertManifest, policy: CachePolicyModel | None):
        self.config = config
        self.manifest = manifest
        self.policy = policy
        self.store = SSDExpertStore(manifest, config.pread_chunks)
        self.capacity_bytes = int(config.expert_cache_gb * (1024 ** 3))
        self.stats = StreamingStats()
        self._resident: OrderedDict[ExpertKey, ResidentExpert] = OrderedDict()
        self._lock = Lock()

    def contains(self, key: ExpertKey) -> bool:
        return key in self._resident

    def ensure(self, layer_id: int, expert_id: int, step: int, prefetched: bool = False) -> ResidentExpert:
        key = (layer_id, expert_id)
        entry = self.manifest.get(key)
        with self._lock:
            self.stats.requests += 0 if prefetched else 1
            if key in self._resident:
                resident = self._resident.pop(key)
                resident.last_step = step
                resident.access_count += 1
                resident.prefetched = resident.prefetched and prefetched
                self._resident[key] = resident
                if not prefetched:
                    self.stats.cache_hits += 1
                else:
                    self.stats.prefetch_requests += 1
                return resident

            if prefetched:
                self.stats.prefetch_requests += 1
            else:
                self.stats.cache_misses += 1

        payload = self.store.read_entry(entry, self.stats)
        resident = ResidentExpert(entry=entry, payload=payload, last_step=step, access_count=1, prefetched=prefetched)

        with self._lock:
            self._ensure_capacity(entry.size_bytes, step)
            self._resident[key] = resident
            self.stats.current_resident_bytes += entry.size_bytes
            self.stats.peak_resident_bytes = max(
                self.stats.peak_resident_bytes, self.stats.current_resident_bytes)
            return resident

    def materialize_tensors(self, resident: ResidentExpert):
        if resident.tensors is not None:
            return resident.tensors

        try:
            from safetensors.torch import load as load_safetensors_bytes
        except ModuleNotFoundError as exc:
            raise RuntimeError("safetensors is required to materialize streamed experts") from exc

        tensors = load_safetensors_bytes(resident.payload)
        if resident.entry.format == "q3like":
            decoded = {
                "gate_proj.weight": unpack_q3_tensor({
                    "qweight": tensors["gate_proj.qweight"],
                    "scale": tensors["gate_proj.scale"],
                    "shape": tensors["gate_proj.shape"],
                }),
                "up_proj.weight": unpack_q3_tensor({
                    "qweight": tensors["up_proj.qweight"],
                    "scale": tensors["up_proj.scale"],
                    "shape": tensors["up_proj.shape"],
                }),
                "down_proj.weight": unpack_q3_tensor({
                    "qweight": tensors["down_proj.qweight"],
                    "scale": tensors["down_proj.scale"],
                    "shape": tensors["down_proj.shape"],
                }),
            }
        else:
            decoded = tensors
        resident.tensors = decoded
        return decoded

    def prefetch(self, keys: Iterable[ExpertKey], step: int) -> None:
        for layer_id, expert_id in keys:
            if (layer_id, expert_id) in self.manifest:
                self.ensure(layer_id, expert_id, step=step, prefetched=True)

    def _ensure_capacity(self, incoming_bytes: int, step: int) -> None:
        while self.stats.current_resident_bytes + incoming_bytes > self.capacity_bytes and self._resident:
            victim_key = self._select_victim(step)
            victim = self._resident.pop(victim_key)
            self.stats.current_resident_bytes -= victim.size_bytes
            self.stats.evictions += 1

    def _select_victim(self, step: int) -> ExpertKey:
        if not self.policy:
            return next(iter(self._resident))

        best_key: ExpertKey | None = None
        best_score: float | None = None
        busiest_layer_entries = self._layer_pressure_max()
        for key, resident in self._resident.items():
            recency = float(step - resident.last_step)
            frequency = float(resident.access_count)
            features = CachePolicyFeatures(
                recency=recency,
                frequency=frequency,
                reuse_distance=recency,
                size_ratio=resident.size_bytes / max(self.capacity_bytes, 1),
                layer_pressure=busiest_layer_entries,
                is_prefetched=1.0 if resident.prefetched else 0.0,
            )
            score = self.policy.score(features)
            if best_score is None or score > best_score:
                best_score = score
                best_key = key
        assert best_key is not None
        return best_key

    def _layer_pressure_max(self) -> float:
        counts: dict[int, int] = {}
        for key in self._resident:
            counts[key[0]] = counts.get(key[0], 0) + 1
        return float(max(counts.values(), default=0))


def format_stats(stats: StreamingStats, elapsed_s: float) -> str:
    effective_gbps = stats.bytes_read / max(elapsed_s, 1e-9) / 1e9
    values = stats.to_dict()
    return (
        f"requests={values['requests']} "
        f"hit_rate={values['hit_rate']:.3f} "
        f"miss_rate={values['miss_rate']:.3f} "
        f"bytes_read_gb={values['bytes_read_gb']:.3f} "
        f"effective_read_gbps={effective_gbps:.3f} "
        f"read_ops={values['read_ops']} "
        f"evictions={values['evictions']} "
        f"peak_resident_gb={values['peak_resident_gb']:.3f}"
    )
