"""Paired A-vs-B estimate from simulated conversations, and its translation into an engine scenario.

Pairs share a persona (the same real buyer profile); once the bots diverge the buyer's replies diverge
too, so pairing is by persona, not a replay. Control A is the same prompt for every experiment.

The A/A arm (AA_ID) is a second, independent call on prompt A for each persona. It does two jobs: compared with
the first A call it measures pure noise (the yardstick for every lift), and for the real changes it is pooled
with the first A call into a two-call control per persona, which halves the control's own noise.
"""
from __future__ import annotations

import numpy as np

PARAM_FATAL = "Fatal"
AA_ID = "exp-aa"


def _lead(c: dict) -> int:
    return int(bool((c.get("judge") or {}).get("lead_created")))


def _asks(c: dict) -> float:
    j = c.get("judge") or {}
    return float((j.get("qty_asks") or 0) + (j.get("spec_asks") or 0))


def _reasks(c: dict) -> int:
    """Asks beyond one per field: quantity asked more than once, or specs asked more often than there are fields."""
    j = c.get("judge") or {}
    n_spec = len((c.get("fields") or {}).get("specification_options") or {})
    return max((j.get("qty_asks") or 0) - 1, 0) + max((j.get("spec_asks") or 0) - n_spec, 0)


def paired(convs: list[dict], exp_id: str, n_boot: int = 4000, n_perm: int = 10000, seed: int = 0) -> dict | None:
    """B's call against the control calls on prompt A, paired by persona.

    Control per persona: the A call, plus the A/A replicate when there is one, averaged. The A/A estimate itself
    compares the replicate with the first A call only. CI: bootstrap over personas. p: paired sign-flip
    permutation test on the per-persona differences (exact for any outcome scale, unlike McNemar)."""
    first = {c["persona_id"]: c for c in convs if c["arm"] == "A"}
    rep = {c["persona_id"]: c for c in convs if c["arm"] == "B" and c["experiment_id"] == AA_ID}
    b = rep if exp_id == AA_ID else {c["persona_id"]: c for c in convs
                                     if c["arm"] == "B" and c["experiment_id"] == exp_id}
    ids = sorted(set(first) & set(b))
    if not ids:
        return None
    control = {i: [first[i]] + ([rep[i]] if exp_id != AA_ID and i in rep else []) for i in ids}
    la = np.array([np.mean([_lead(c) for c in control[i]]) for i in ids])
    lb = np.array([_lead(b[i]) for i in ids], dtype=float)
    da = np.array([np.mean([c["est_duration_s"] for c in control[i]]) for i in ids])
    db = np.array([b[i]["est_duration_s"] for i in ids])
    diff = lb - la
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), (n_boot, len(ids)))
    lifts = diff[idx].mean(axis=1)
    ratios = db[idx].mean(axis=1) / da[idx].mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(ids)))
    p_value = float((np.abs((flips * diff).mean(axis=1)) >= abs(diff.mean()) - 1e-12).mean())

    def arm_stats(cs: list[dict]) -> dict:
        js = [c.get("judge") or {} for c in cs]
        return {"calls": len(cs),
                "lead_rate": float(np.mean([_lead(c) for c in cs])),
                "fatal_rate": float(np.mean([j.get("overall") == PARAM_FATAL for j in js])),
                "hangup_rate": float(np.mean([c["ended"] == "buyer_hung_up" for c in cs])),
                "mean_asks": float(np.mean([_asks(c) for c in cs])),
                "reask_rate": float(np.mean([_reasks(c) > 0 for c in cs])),
                "mean_turns": float(np.mean([len(c["turns"]) for c in cs])),
                "mean_duration_s": float(np.mean([c["est_duration_s"] for c in cs])),
                "bot_judge_agree": float(np.mean([_bot_says_lead(c) == bool(_lead(c)) for c in cs]))}

    return {"experiment_id": exp_id, "pairs": len(ids),
            "A": arm_stats([c for i in ids for c in control[i]]), "B": arm_stats([b[i] for i in ids]),
            "lift_pp": float(diff.mean()),
            "lift_ci": [float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975))],
            "discordant": {"B_better": int((diff > 0).sum()), "A_better": int((diff < 0).sum())},
            "p_value": p_value,
            "duration_ratio": float(db.mean() / da.mean()),
            "duration_ratio_ci": [float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))]}


def _bot_says_lead(c: dict) -> bool:
    d = (c.get("bot_disposition") or "").lower()
    return "bl generated" in d or "bl enriched" in d or "bl approved" in d


def to_true_effect(summary: dict, p_connected: float) -> dict:
    """Translate the pilot into the engine's true effect on P(lead | connected call).

    The simulated world's baseline (A's lead rate, ~28%) is far from the real connected rate (~71%), so the effect
    is carried over as an ODDS RATIO, the usual way to move a binary effect to a new baseline: odds_B = odds_real x OR.
    A relative risk would not transport: +43% relative on a 71% base is over 100%. Each arm's odds get a 0.5
    continuity correction so a 0% or 100% arm still gives a finite ratio. The simulator can only realise success_pp
    in (-p_c, 1 - p_c), so the value is still clamped as a safeguard, and the clamp recorded."""
    n = summary["pairs"]
    xa, xb = summary["A"]["lead_rate"] * n, summary["B"]["lead_rate"] * n
    odds_ratio = ((xb + 0.5) / (n - xb + 0.5)) / ((xa + 0.5) / (n - xa + 0.5))
    odds_real = p_connected / (1 - p_connected)
    pp = odds_real * odds_ratio / (1 + odds_real * odds_ratio) - p_connected
    lo, hi = -0.95 * p_connected, 0.95 * (1 - p_connected)
    clamped = min(max(pp, lo), hi)
    return {"success_pp": round(clamped, 4), "odds_ratio": round(odds_ratio, 4), "raw_pp": round(pp, 4),
            "clamped": clamped != pp, "p_connected_real": round(p_connected, 4),
            "duration_mult": round(summary["duration_ratio"], 3)}
