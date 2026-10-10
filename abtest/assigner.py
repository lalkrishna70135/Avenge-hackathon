"""Deterministic, sticky, per-caller assignment.

bucket = sha256(salt + ":" + caller_id) mod 10000 ;  B if bucket < share * 10000

No lookup is needed to reproduce an assignment; assignments are still written to the store
for audit. A per-experiment salt decorrelates buckets across experiments.
"""
from __future__ import annotations

import hashlib

import numpy as np

BUCKETS = 10_000


def bucket(salt: str, caller_id: str) -> int:
    h = hashlib.sha256(f"{salt}:{caller_id}".encode()).hexdigest()
    return int(h[:8], 16) % BUCKETS


def assign(salt: str, caller_id: str, share_b: float) -> str:
    return "B" if bucket(salt, caller_id) < round(share_b * BUCKETS) else "A"


def buckets(salt: str, caller_ids) -> np.ndarray:
    return np.fromiter((bucket(salt, c) for c in caller_ids), dtype=np.int32, count=len(caller_ids))


def assign_many(salt: str, caller_ids, share_b: float) -> np.ndarray:
    """Vectorised over unique callers; returns array of 'A'/'B'."""
    return np.where(buckets(salt, caller_ids) < round(share_b * BUCKETS), "B", "A")


class Router:
    """Live routing: during a running experiment use the hash split; once the rollout
    controller sets a terminal routing (promote -> all B, stop -> all A) every caller follows it."""

    def __init__(self, salt: str, share_b: float):
        self.salt = salt
        self.share_b = share_b

    def route(self, caller_id: str) -> str:
        if self.share_b >= 1.0:
            return "B"
        if self.share_b <= 0.0:
            return "A"
        return assign(self.salt, caller_id, self.share_b)
