"""Decision engine: a pure function from look evidence + config to (decision, rule).

Primary metric decides; the duration guardrail can only veto (or hard-stop on harm).
"""
from __future__ import annotations

from enum import Enum

from .config import ExperimentConfig


class Decision(str, Enum):
    CONTINUE = "CONTINUE"
    PROMOTE = "PROMOTE"
    STOP_HARM = "STOP_HARM"
    HOLD = "HOLD_FOR_APPROVAL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID_SRM"
    REJECT = "REJECTED_BY_APPROVER"


TERMINAL = {Decision.PROMOTE, Decision.STOP_HARM, Decision.HOLD, Decision.INCONCLUSIVE, Decision.INVALID,
            Decision.REJECT}

RULES = {
    "R0_SRM": "Sample ratio mismatch (chi-square p < threshold): experiment invalid, revert to A, no auto-action",
    "R1_MIN_SAMPLE": "Fewer than min_sample_per_arm callers in an arm: no stopping allowed, continue",
    "R2_HARM_PRIMARY": "Z <= harm boundary (one-sided Pocock-type spending): stop early, revert to A",
    "R2_HARM_GUARDRAIL": "Duration hard guardrail breached (99% CI lower bound of B/A ratio > 1 + hard_margin): stop early, revert to A",
    "R3_WIN": "Z >= efficacy boundary (O'Brien-Fleming spending) and duration non-inferior: promote B to 100%",
    "R4_HOLD": "Z >= efficacy boundary but duration guardrail violated (95% CI lower bound of B/A > 1 + ni_margin), "
               "or non-inferiority still not shown at the final look: hold, needs human approval",
    "R4b_WAIT_GUARDRAIL": "Z >= efficacy boundary but duration non-inferiority not yet shown (CI still straddles the "
                          "margin): continue to the next look to collect more evidence",
    "R5_FINAL_INCONCLUSIVE": "Final look, no boundary crossed: inconclusive, stay on A",
    "R6_CONTINUE": "No boundary crossed: continue to next scheduled look",
    "R7_APPROVED": "Held win approved by a named approver: promote B to 100% (logged with their name and note)",
    "R7_REJECTED": "Held win rejected by a named approver: keep A, B to 0% (logged with their name and note)",
}


def decide(ev: dict, cfg: ExperimentConfig, is_final: bool) -> tuple[Decision, str]:
    """ev keys: n_a, n_b, z, eff_bound, harm_bound, srm (bool), dur_ni_ok, dur_ni_violated, dur_hard_breach."""
    if ev["srm"]:
        return Decision.INVALID, "R0_SRM"
    if min(ev["n_a"], ev["n_b"]) < cfg.min_sample_per_arm:
        if is_final:
            return Decision.INCONCLUSIVE, "R5_FINAL_INCONCLUSIVE"
        return Decision.CONTINUE, "R1_MIN_SAMPLE"
    if ev["z"] <= ev["harm_bound"]:
        return Decision.STOP_HARM, "R2_HARM_PRIMARY"
    if ev["dur_hard_breach"]:
        return Decision.STOP_HARM, "R2_HARM_GUARDRAIL"
    if ev["z"] >= ev["eff_bound"]:
        if ev["dur_ni_ok"]:
            return (Decision.PROMOTE if cfg.auto_promote else Decision.HOLD), ("R3_WIN" if cfg.auto_promote else "R4_HOLD")
        if ev["dur_ni_violated"] or is_final:
            return Decision.HOLD, "R4_HOLD"
        return Decision.CONTINUE, "R4b_WAIT_GUARDRAIL"
    if is_final:
        return Decision.INCONCLUSIVE, "R5_FINAL_INCONCLUSIVE"
    return Decision.CONTINUE, "R6_CONTINUE"


def routing_after(decision: Decision, current_share_b: float) -> float:
    """Rollout controller policy: the B share after a decision."""
    return {
        Decision.PROMOTE: 1.0,
        Decision.STOP_HARM: 0.0,
        Decision.INCONCLUSIVE: 0.0,
        Decision.INVALID: 0.0,
        Decision.HOLD: 0.0,          # safe default while waiting for an approver
        Decision.REJECT: 0.0,
    }.get(decision, current_share_b)
