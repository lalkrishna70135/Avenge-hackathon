"""Split-accuracy evidence for the hash assigner: achieved vs configured share, stickiness on
replay, independence across experiment salts, and an SRM check that fires on a broken split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as st
from .assigner import BUCKETS, assign_many, buckets


def caller_population(real_ids, n_total: int = 50_000) -> list[str]:
    """The real caller IDs from the call log, topped up with synthetic IDs to n_total."""
    ids = [str(c) for c in pd.unique(pd.Series(real_ids, dtype=str))]
    return ids + [f"SIM-{i:07d}" for i in range(n_total - len(ids))]


def split_accuracy(caller_ids, salt: str, shares=(0.01, 0.05, 0.10, 0.25, 0.50)) -> pd.DataFrame:
    rows = []
    for share in shares:
        v = assign_many(salt, caller_ids, share)
        n, n_b = len(v), int((v == "B").sum())
        lo, hi = st.wilson_ci(n_b, n)
        srm = st.srm_check(n - n_b, n_b, share)
        rows.append({"configured_share_b": share, "n_callers": n, "n_b": n_b, "achieved_share_b": n_b / n,
                     "ci95_low": lo, "ci95_high": hi, "inside_ci": lo <= share <= hi,
                     "srm_p": srm["p"], "srm_flag": srm["srm"]})
    return pd.DataFrame(rows)


def stickiness(caller_ids, salt: str, share: float, repeats: int = 2) -> dict:
    """Assign the same callers `repeats` times, in shuffled order, and count disagreements."""
    rng = np.random.default_rng(0)
    ids = np.array(caller_ids, dtype=object)
    first = pd.Series(assign_many(salt, ids, share), index=ids)
    mismatches = 0
    for _ in range(repeats - 1):
        order = rng.permutation(len(ids))
        again = pd.Series(assign_many(salt, ids[order], share), index=ids[order])
        mismatches += int((again.reindex(first.index) != first).sum())
    return {"callers": len(ids), "replays": repeats, "mismatches": mismatches,
            "consistency": 1 - mismatches / (len(ids) * (repeats - 1))}


def salt_independence(caller_ids, salt_a: str, salt_b: str, share: float = 0.10) -> dict:
    """Correlation of buckets and 2x2 overlap of B assignments across two experiments."""
    ba, bb = buckets(salt_a, caller_ids), buckets(salt_b, caller_ids)
    in_a, in_b = ba < round(share * BUCKETS), bb < round(share * BUCKETS)
    table = np.array([[np.sum(~in_a & ~in_b), np.sum(~in_a & in_b)], [np.sum(in_a & ~in_b), np.sum(in_a & in_b)]])
    chi2, p, _, _ = sps.chi2_contingency(table)
    return {"salt_a": salt_a, "salt_b": salt_b, "bucket_corr": float(np.corrcoef(ba, bb)[0, 1]),
            "b_in_both": int(table[1, 1]), "expected_b_in_both": len(caller_ids) * share * share,
            "independence_p": float(p)}


def broken_split_demo(caller_ids, salt: str, share: float = 0.10, drop_b: float = 0.15) -> dict:
    """Simulate a logging bug that loses `drop_b` of B's callers, and show SRM catching it."""
    v = assign_many(salt, caller_ids, share)
    n_b = int((v == "B").sum())
    n_a = len(v) - n_b
    lost = int(n_b * drop_b)
    srm = st.srm_check(n_a, n_b - lost, share)
    return {"configured_share_b": share, "callers_lost_from_b": lost, "observed_n_a": n_a,
            "observed_n_b": n_b - lost, "achieved_share_b": (n_b - lost) / (len(v) - lost),
            "srm_chi2": srm["chi2"], "srm_p": srm["p"], "srm_flag": srm["srm"]}


def ci_coverage(caller_ids, share: float, n_salts: int = 200) -> dict:
    """Across many experiment salts, how often the configured share falls inside the 95% CI of
    the achieved share. A correct, unbiased assigner gives ~95%; a single miss is not a defect."""
    hits, srm_flags = 0, 0
    for i in range(n_salts):
        v = assign_many(f"coverage-{i}", caller_ids, share)
        n, n_b = len(v), int((v == "B").sum())
        lo, hi = st.wilson_ci(n_b, n)
        hits += lo <= share <= hi
        srm_flags += st.srm_check(n - n_b, n_b, share)["srm"]
    return {"share": share, "salts": n_salts, "coverage": hits / n_salts, "srm_false_alarms": srm_flags}
