"""Core loop on the virtual clock:

call arrives -> assign variant (sticky hash) -> outcome recorded -> stats at scheduled looks
-> decision engine -> rollout controller updates routing -> decision log written.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import stats as st
from . import store
from .assigner import buckets
from .config import ExperimentConfig
from .decision import RULES, TERMINAL, Decision, decide, routing_after
from .loader import Dataset
from .simulator import experiment_variants, generate_arrivals, realise_outcomes

INFO_ROUND = 0.01   # information fractions rounded to 0.01 so boundary sets are cached across runs


@dataclass
class RunResult:
    cfg: ExperimentConfig
    design: dict
    looks: list[dict] = field(default_factory=list)
    decision: str = Decision.CONTINUE.value
    rule: str = ""
    decision_hour: float = float("nan")
    decision_look: int = 0
    final_share_b: float = 0.0
    calls: pd.DataFrame | None = None


def baseline_per_caller(ds: Dataset, cfg: ExperimentConfig, hours: float = 200.0) -> float:
    """Per-caller primary-goal rate under the simulator's arrival process (no effect), scored the way the
    looks score callers. The unit of analysis is the caller, so the design must use this rate, not the per-call one."""
    ds = ds.with_goal(cfg.primary_goal)
    pilot = cfg.model_copy(update={"seed": 0, "true_effect": cfg.true_effect.model_copy(update={"success_pp": 0.0})})
    horizon = min(hours, 60000 / cfg.calls_per_hour)
    arr = generate_arrivals(ds, pilot, horizon, seed=0)
    calls = realise_outcomes(ds, arr, np.full(len(arr.t_hours), "A"), pilot)
    return float(aggregate_callers(calls, cfg.score_window_hours, horizon)["success"].mean())


def plan_design(cfg: ExperimentConfig, ds: Dataset) -> dict:
    ds = ds.with_goal(cfg.primary_goal)
    p0 = cfg.baseline_rate if cfg.baseline_rate is not None else baseline_per_caller(ds, cfg)
    infl = st.gs_inflation_factor(cfg.n_looks, cfg.alpha / 2, cfg.power, cfg.efficacy_spending)
    ss = st.sample_size_two_prop(p0, p0 + cfg.mde_pp, cfg.alpha, cfg.power, cfg.share_b, infl)
    callers_per_hour = cfg.calls_per_hour * (1 - cfg.repeat_call_prob)
    window = cfg.window_hours or math.ceil(ss["n_total"] / callers_per_hour)
    looks = [round(window * (k + 1) / cfg.n_looks, 3) for k in range(cfg.n_looks)]
    planned_t = [(k + 1) / cfg.n_looks for k in range(cfg.n_looks)]
    per_caller_dur = ds.calls.groupby("caller_id")["duration_s"].mean().dropna()
    cv = float(per_caller_dur.std() / per_caller_dur.mean())
    with_dur = float(len(per_caller_dur) / ds.calls["caller_id"].nunique())   # callers with a connected call
    g_na, g_nb = int(ss["n_a"] * with_dur), int(ss["n_b"] * with_dur)
    return {
        "baseline_rate_per_caller": p0,
        "baseline_rate_per_call": ds.p0,
        "mde_pp": cfg.mde_pp,
        "sample_size": ss,
        "callers_per_hour": callers_per_hour,
        "window_hours": window,
        "look_hours": looks,
        "planned_eff_bounds": st.gs_upper_boundaries(planned_t, cfg.alpha / 2, cfg.efficacy_spending),
        "planned_harm_bounds": [-b for b in st.gs_upper_boundaries(planned_t, cfg.harm_alpha, cfg.harm_spending)],
        "guardrail": {"cv_per_caller": cv, "callers_with_duration_share": with_dur, "ni_margin": cfg.ni_margin,
                      "n_a": g_na, "n_b": g_nb,
                      "power_at_final_look": st.ni_power(cv, g_na, g_nb, cfg.ni_margin),
                      "power_if_margin_10pct": st.ni_power(cv, g_na, g_nb, 0.10)},
        "config_hash": cfg.config_hash,
    }


def aggregate_callers(calls: pd.DataFrame, window: float | None = None, as_of: float | None = None) -> pd.DataFrame:
    """Unit of analysis = caller: success on ANY of their scored calls; mean duration of their scored
    connected calls (NaN if none connected; dropped by the guardrail).

    With `window`, a caller is scored on the calls within `window` hours of their first call and is included
    only once that window has closed by `as_of`. Without it, a repeat caller's outcome could change between
    looks, which breaks the independent increments the group-sequential boundaries assume (in 1,000 null
    reruns that pushed the false harm-stop rate from about 5% to 7.1%)."""
    if window is not None:
        first = calls.groupby("caller_idx")["t_hours"].transform("min")
        keep = calls["t_hours"] <= first + window
        if as_of is not None:
            keep &= first <= as_of - window
        calls = calls[keep]
    g = calls.groupby("caller_idx", sort=False)
    return pd.DataFrame({
        "variant": g["variant"].first(),
        "success": g["success"].max().astype(int),
        "duration_s": g["duration_s"].mean(),
        "n_calls": g.size(),
    })


def evaluate_look(cfg: ExperimentConfig, design: dict, calls_so_far: pd.DataFrame, look: int, hour: float,
                  t_hist: list[float], is_final: bool) -> dict:
    agg = aggregate_callers(calls_so_far, cfg.score_window_hours, hour)
    a, b = agg[agg["variant"] == "A"], agg[agg["variant"] == "B"]
    n_target = design["sample_size"]["n_total"]
    t_raw = len(agg) / n_target
    if is_final:
        t = 1.0
    else:
        t = min(round(t_raw / INFO_ROUND) * INFO_ROUND, 0.99)
        if t_hist:
            t = max(t, round(t_hist[-1] + INFO_ROUND, 2))
        t = round(t, 2)
    ts = t_hist + [t]
    eff = st.gs_upper_boundaries(ts, cfg.alpha / 2, cfg.efficacy_spending)[-1]
    harm = -st.gs_upper_boundaries(ts, cfg.harm_alpha, cfg.harm_spending)[-1]
    pt = st.two_prop_ztest(int(a["success"].sum()), len(a), int(b["success"].sum()), len(b))
    dt = st.duration_guardrail(a["duration_s"], b["duration_s"], cfg.ni_margin, cfg.hard_margin)
    srm = st.srm_check(len(a), len(b), cfg.share_b, cfg.srm_threshold)
    wl, wh = st.wilson_ci(len(b), len(agg))
    rci_c = eff if math.isfinite(eff) else 10.0
    return {
        "look": look, "virtual_hour": hour, "is_final": is_final,
        "info_fraction": t, "info_fraction_raw": round(t_raw, 4),
        "n_a": len(a), "n_b": len(b), "n_target": n_target,
        "calls_a": int((calls_so_far["variant"] == "A").sum()), "calls_b": int((calls_so_far["variant"] == "B").sum()),
        "x_a": pt.x_a, "x_b": pt.x_b, "p_a": pt.p_a, "p_b": pt.p_b,
        "lift_abs": pt.lift_abs, "lift_rel": pt.lift_rel, "se": pt.se,
        "ci95_low": pt.ci_low, "ci95_high": pt.ci_high,
        # repeated confidence interval: valid under repeated looks (Jennison & Turnbull)
        "rci_low": pt.lift_abs - rci_c * pt.se, "rci_high": pt.lift_abs + rci_c * pt.se,
        "z": pt.z, "p_value_nominal": pt.p_two_sided,
        "eff_bound": eff, "harm_bound": harm,
        "achieved_share_b": len(b) / max(len(agg), 1), "share_ci": [wl, wh], "configured_share_b": cfg.share_b,
        "srm": srm["srm"], "srm_p": srm["p"], "srm_chi2": srm["chi2"],
        "duration": dt.to_dict(), "dur_ni_ok": dt.ni_ok, "dur_ni_violated": dt.ni_violated,
        "dur_hard_breach": dt.hard_breach,
        "detectable_effect_pp": st.detectable_effect(pt.p_a, len(a), len(b), cfg.alpha, cfg.power),
    }


def run_experiment(cfg: ExperimentConfig, ds: Dataset, con=None, design: dict | None = None,
                   narrate: Callable[[dict], str | None] | None = None, keep_calls: bool = True) -> RunResult:
    ds = ds.with_goal(cfg.primary_goal)
    design = design or plan_design(cfg, ds)
    window, post = design["window_hours"], cfg.post_rollout_hours
    arr = generate_arrivals(ds, cfg, window + post)
    variants = experiment_variants(cfg, arr)
    calls = realise_outcomes(ds, arr, variants, cfg)

    res = RunResult(cfg, design)
    if con is not None:
        store.reset_experiment(con, cfg.experiment_id)
        store.save_experiment(con, cfg, "running", design, ds.card())
        store.add_routing(con, cfg.experiment_id, cfg.share_b, "experiment start", None, 0.0)

    t_hist: list[float] = []
    decision, rule, dec_hour, dec_look = Decision.CONTINUE, "", window, cfg.n_looks
    for k, hour in enumerate(design["look_hours"], start=1):
        is_final = k == cfg.n_looks
        so_far = calls[calls["t_hours"] <= hour]
        ev = evaluate_look(cfg, design, so_far, k, hour, t_hist, is_final)
        t_hist.append(ev["info_fraction"])
        decision, rule = decide(ev, cfg, is_final)
        ev["decision"], ev["rule"] = decision.value, rule
        res.looks.append(ev)
        if con is not None:
            store.add_look(con, cfg.experiment_id, k, hour, ev)
            text = narrate(ev) if (narrate and decision in TERMINAL) else None
            did = store.add_decision(con, cfg.experiment_id, cfg.config_hash, hour, k, decision.value, rule,
                                     RULES[rule], ev, "auto", text)
        if decision in TERMINAL:
            dec_hour, dec_look = hour, k
            break

    share_after = routing_after(decision, cfg.share_b)
    res.decision, res.rule, res.decision_hour, res.decision_look = decision.value, rule, dec_hour, dec_look
    res.final_share_b = share_after

    # rollout phase: every caller now follows the terminal routing
    exp_calls = calls[calls["t_hours"] <= dec_hour].assign(phase="experiment")
    post_mask = (arr.t_hours > dec_hour) & (arr.t_hours <= dec_hour + post)
    if post_mask.any():
        post_variant = np.full(len(arr.t_hours), "B" if share_after >= 1.0 else "A")
        post_calls = realise_outcomes(ds, arr, post_variant, cfg)[post_mask].assign(phase="rollout")
        all_calls = pd.concat([exp_calls, post_calls], ignore_index=True)
    else:
        all_calls = exp_calls
    if keep_calls:
        res.calls = all_calls

    if con is not None:
        if decision in TERMINAL:
            store.add_routing(con, cfg.experiment_id, share_after, f"{decision.value} via {rule}", did, dec_hour)
        status = {Decision.PROMOTE: "promoted", Decision.STOP_HARM: "stopped", Decision.INCONCLUSIVE: "inconclusive",
                  Decision.INVALID: "invalid", Decision.HOLD: "awaiting_approval"}.get(decision, "running")
        store.set_status(con, cfg.experiment_id, status)
        first = exp_calls.sort_values("t_hours").drop_duplicates("caller_idx")
        store.write_frame(con, "assignments", pd.DataFrame({
            "experiment_id": cfg.experiment_id, "caller_id": first["caller_id"].astype(str),
            "variant": first["variant"], "bucket": buckets(cfg.salt, first["caller_id"].astype(str).tolist()),
            "assigned_at_hour": first["t_hours"].round(4)}))
        store.write_frame(con, "calls", pd.DataFrame({
            "experiment_id": cfg.experiment_id, "config_hash": cfg.config_hash, "call_id": all_calls["call_id"],
            "caller_id": all_calls["caller_id"].astype(str), "variant": all_calls["variant"],
            "phase": all_calls["phase"], "disposition": all_calls["disposition"],
            "success": all_calls["success"].astype(int), "duration_s": all_calls["duration_s"],
            "t_hours": all_calls["t_hours"].round(4), "drawn_src_row": all_calls["drawn_src_row"],
            "used_src_row": all_calls["used_src_row"], "effect_applied": all_calls["effect_applied"].astype(int)}))
    return res


def resolve_hold(con, experiment_id: str, approve: bool, approver: str, note: str = "") -> int:
    """The logged approval step for a HOLD_FOR_APPROVAL: a named approver promotes B or keeps A.
    Appends a decision (actor = the approver, evidence = the held look's), the routing change and the status."""
    exp = store.read(con, "SELECT config_hash, status FROM experiments WHERE experiment_id = ?", (experiment_id,))
    if exp.empty or exp["status"].iloc[0] != "awaiting_approval":
        raise ValueError(f"{experiment_id} is not awaiting approval")
    if not approver.strip():
        raise ValueError("an approver name is required")
    held = store.read(con, "SELECT virtual_hour, look, evidence_json FROM decisions WHERE experiment_id = ? "
                           "ORDER BY decision_id DESC LIMIT 1", (experiment_id,)).iloc[0]
    decision, rule = (Decision.PROMOTE, "R7_APPROVED") if approve else (Decision.REJECT, "R7_REJECTED")
    hour = float(held["virtual_hour"])
    did = store.add_decision(con, experiment_id, exp["config_hash"].iloc[0], hour, int(held["look"]), decision.value,
                             rule, RULES[rule], json.loads(held["evidence_json"]), f"approver: {approver.strip()}",
                             note.strip() or None)
    store.add_routing(con, experiment_id, routing_after(decision, 0.0), f"{decision.value} by {approver.strip()}",
                      did, hour)
    store.set_status(con, experiment_id, "promoted" if approve else "rejected")
    return did
