"""Bounded, population-local source cloud cache; never modifies dataset storage."""

from collections import OrderedDict
import numpy as np


class SourcePointCache:
    def __init__(self, loader, max_bytes):
        self.loader = loader
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError('Source cache byte budget must be non-negative.')
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.loads = self.hits = 0

    def get(self, key, default=None):
        if key in self.entries:
            self.entries.move_to_end(key)
            self.hits += 1
            return self.entries[key][0]
        value = self.loader(key)
        self.loads += 1
        if value is None:
            return default
        if 0 < value.nbytes <= self.max_bytes:
            while self.entries and self.bytes + value.nbytes > self.max_bytes:
                _, (old, _) = self.entries.popitem(last=False)
                self.bytes -= old.nbytes
            self.entries[key] = (value, {})
            self.bytes += value.nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value

    def floor(self, key, points, percentile):
        floors = self.entries[key][1] if key in self.entries else {}
        if percentile not in floors:
            floors[percentile] = float(np.percentile(points[:, 2], percentile))
        return floors[percentile]

    def statistics(self):
        return dict(source_loads=self.loads, cache_hits=self.hits,
                    peak_cached_point_bytes=self.peak_bytes,
                    max_cached_point_bytes=self.max_bytes)
