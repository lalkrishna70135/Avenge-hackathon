"""Call simulator on a virtual clock.

HOW OUTCOMES ARE GENERATED (shown on the dashboard):
  * Arrivals: Poisson process at `calls_per_hour`. Each arrival is a returning caller with
    probability `repeat_call_prob` (uniform over callers seen so far), else a new caller.
  * Control (A): the call outcome (disposition + duration) is a bootstrap draw of a real
    historical call.
  * Variant (B): the same bootstrap draw, then the configured TRUE effect is injected, only
    into CONNECTED calls (a call that failed before any conversation cannot be changed by a prompt):
      - +pp: a connected non-success draw is replaced, with prob pp/(1-p_c), by a random
             connected historical success call;  -pp: a success draw is replaced, with prob |pp|/p_c, by a
             random connected non-success call. p_c = success rate among connected calls,
             so E[P(success | B, connected)] = p_c + pp.
      - The replacement brings its own real duration (success calls are longer than e.g.
        abrupt disconnects), then duration is multiplied by `duration_mult`.
  Nothing about the prompt text changes these numbers: the effect is injected, by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assigner import assign_many
from .config import ExperimentConfig
from .loader import Dataset


@dataclass
class Arrivals:
    t_hours: np.ndarray        # arrival time (virtual hours since start)
    caller_idx: np.ndarray     # integer caller index
    caller_ids: np.ndarray     # caller id per caller index
    hist_idx: np.ndarray       # bootstrap row into the historical calls
    u_effect: np.ndarray       # uniform draw used for effect injection (common random numbers)
    alt_mf_idx: np.ndarray     # replacement row if flipped to success
    alt_non_idx: np.ndarray    # replacement row if flipped to (connected) non-success


def generate_arrivals(ds: Dataset, cfg: ExperimentConfig, horizon_hours: float, seed: int | None = None) -> Arrivals:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n = rng.poisson(cfg.calls_per_hour * horizon_hours)
    t = np.sort(rng.uniform(0, horizon_hours, n))
    is_new = rng.random(n) >= cfg.repeat_call_prob
    if n:
        is_new[0] = True
    n_seen = np.cumsum(is_new)                       # callers seen up to and including i
    new_idx = n_seen - 1
    ret_idx = np.floor(rng.random(n) * np.maximum(n_seen, 1)).astype(np.int64)
    caller_idx = np.where(is_new, new_idx, ret_idx)
    n_callers = int(n_seen[-1]) if n else 0

    # caller ids: reuse real historical ids first (keeps the demo grounded), then synthetic
    hist_ids = pd.unique(ds.calls["caller_id"].to_numpy())
    rng.shuffle(hist_ids)
    ids = list(hist_ids[:n_callers])
    ids += [f"SIM{cfg.seed:03d}-{i:07d}" for i in range(len(ids), n_callers)]

    mf = ds.calls["success"].to_numpy()
    conn = ds.calls["connected"].to_numpy()
    mf_rows, non_rows = np.flatnonzero(mf & conn), np.flatnonzero(conn & ~mf)   # replacements are connected calls
    return Arrivals(
        t_hours=t,
        caller_idx=caller_idx,
        caller_ids=np.array(ids, dtype=object),
        hist_idx=rng.integers(0, len(ds.calls), n),
        u_effect=rng.random(n),
        alt_mf_idx=mf_rows[rng.integers(0, len(mf_rows), n)],
        alt_non_idx=non_rows[rng.integers(0, len(non_rows), n)],
    )


def realise_outcomes(ds: Dataset, arr: Arrivals, variant: np.ndarray, cfg: ExperimentConfig) -> pd.DataFrame:
    """Turn arrivals + assigned variant into call records."""
    calls = ds.calls
    mf_hist = calls["success"].to_numpy()
    conn = calls["connected"].to_numpy()
    p_c = float(mf_hist[conn].mean())
    row = arr.hist_idx.copy()
    is_b = (variant == "B") & conn[row]
    pp = cfg.true_effect.success_pp
    flip = np.zeros(len(row), dtype=bool)
    if pp > 0:
        flip = is_b & ~mf_hist[row] & (arr.u_effect < pp / (1 - p_c))
        row[flip] = arr.alt_mf_idx[flip]
    elif pp < 0:
        flip = is_b & mf_hist[row] & (arr.u_effect < -pp / p_c)
        row[flip] = arr.alt_non_idx[flip]
    dur = calls["duration_s"].to_numpy()[row].astype(float)
    dur = np.where(is_b, dur * cfg.true_effect.duration_mult, dur)
    return pd.DataFrame({
        "call_id": np.arange(len(row)),
        "t_hours": arr.t_hours,
        "caller_idx": arr.caller_idx,
        "caller_id": arr.caller_ids[arr.caller_idx],
        "variant": variant,
        "disposition": calls["disposition"].to_numpy()[row],
        "success": mf_hist[row],
        "connected": conn[row],
        "duration_s": np.round(dur, 1),
        # provenance: the bootstrap draw, and the replacement row when B's injected effect flipped it
        "drawn_src_row": calls["src_row"].to_numpy()[arr.hist_idx],
        "used_src_row": calls["src_row"].to_numpy()[row],
        "effect_applied": flip,
    })


def experiment_variants(cfg: ExperimentConfig, arr: Arrivals) -> np.ndarray:
    """Hash split per caller (computed once per unique caller), broadcast to calls."""
    per_caller = assign_many(cfg.salt, arr.caller_ids, cfg.share_b)
    return per_caller[arr.caller_idx]
