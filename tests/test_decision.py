import pytest

from abtest.config import ExperimentConfig
from abtest.decision import RULES, Decision, decide

CFG = ExperimentConfig(experiment_id="t", salt="t", min_sample_per_arm=300)


def ev(**kw):
    base = dict(n_a=5000, n_b=500, z=0.0, eff_bound=2.5, harm_bound=-2.2, srm=False,
                dur_ni_ok=True, dur_ni_violated=False, dur_hard_breach=False)
    return {**base, **kw}


@pytest.mark.parametrize("evidence,is_final,expected,rule", [
    (ev(srm=True, z=5), False, Decision.INVALID, "R0_SRM"),
    (ev(n_b=100, z=5), False, Decision.CONTINUE, "R1_MIN_SAMPLE"),
    (ev(n_b=100, z=5), True, Decision.INCONCLUSIVE, "R5_FINAL_INCONCLUSIVE"),
    (ev(z=-3), False, Decision.STOP_HARM, "R2_HARM_PRIMARY"),
    (ev(z=1, dur_hard_breach=True), False, Decision.STOP_HARM, "R2_HARM_GUARDRAIL"),
    (ev(z=3), False, Decision.PROMOTE, "R3_WIN"),
    (ev(z=3, dur_ni_ok=False, dur_ni_violated=True), False, Decision.HOLD, "R4_HOLD"),
    (ev(z=3, dur_ni_ok=False), False, Decision.CONTINUE, "R4b_WAIT_GUARDRAIL"),
    (ev(z=3, dur_ni_ok=False), True, Decision.HOLD, "R4_HOLD"),
    (ev(z=1), True, Decision.INCONCLUSIVE, "R5_FINAL_INCONCLUSIVE"),
    (ev(z=1), False, Decision.CONTINUE, "R6_CONTINUE"),
])
def test_decision_table(evidence, is_final, expected, rule):
    assert decide(evidence, CFG, is_final) == (expected, rule)
    assert rule in RULES


def test_primary_harm_beats_guardrail_and_win():
    # a harmful primary stops even if the guardrail is fine
    assert decide(ev(z=-3, dur_ni_ok=True), CFG, False)[0] is Decision.STOP_HARM


def test_manual_approval_mode():
    cfg = CFG.model_copy(update={"auto_promote": False})
    assert decide(ev(z=3), cfg, False) == (Decision.HOLD, "R4_HOLD")


def test_approver_rules_and_routing():
    from abtest.decision import routing_after
    assert {"R7_APPROVED", "R7_REJECTED"} <= set(RULES)
    assert routing_after(Decision.REJECT, 0.1) == 0.0 and routing_after(Decision.PROMOTE, 0.1) == 1.0
