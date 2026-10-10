import pytest

from abtest.config import load_scenario
from abtest.loader import ROOT, load
from abtest.runner import run_experiment


@pytest.fixture(scope="module")
def ds():
    return load()


@pytest.mark.parametrize("name,decision,share_after", [
    ("win", "PROMOTE", 1.0),
    ("harm", "STOP_HARM", 0.0),
    ("null", "INCONCLUSIVE", 0.0),
    ("hold", "HOLD_FOR_APPROVAL", 0.0),
])
def test_demo_scenarios(ds, name, decision, share_after):
    r = run_experiment(load_scenario(ROOT / "scenarios" / f"{name}.yaml"), ds, keep_calls=False)
    assert r.decision == decision
    assert r.final_share_b == share_after


def test_config_hash_pins_prompt_text():
    cfg = load_scenario(ROOT / "scenarios" / "win.yaml")
    assert cfg.base_prompt_sha256 and cfg.variant_b.prompt_sha256
    assert cfg.bump(share_b=0.2).config_hash != cfg.config_hash


def test_scoring_window_freezes_each_callers_outcome():
    """A caller enters a look only once their window has closed, scored on calls inside it, so a later
    call can never change an outcome already analysed (independent increments)."""
    import pandas as pd
    from abtest.runner import aggregate_callers
    calls = pd.DataFrame({"caller_idx": [0, 0, 1, 2, 2], "variant": ["A", "A", "B", "A", "A"],
                          "t_hours": [0.0, 5.0, 9.5, 1.0, 1.5], "success": [False, True, True, False, True],
                          "duration_s": [60.0, 90.0, 30.0, 40.0, 50.0]})
    agg = aggregate_callers(calls, window=2.0, as_of=10.0)
    assert set(agg.index) == {0, 2}                    # caller 1's window (9.5 h -> 11.5 h) is still open
    assert agg.loc[0, "success"] == 0                  # the lead at 5 h is outside caller 0's 2 h window
    assert agg.loc[2, "success"] == 1 and agg.loc[2, "duration_s"] == 45.0
    assert aggregate_callers(calls).loc[0, "success"] == 1     # no window: any call counts


def test_selectable_primary_goal(ds):
    goal = ("BL Approved",)                     # approved leads only, instead of any lead created
    g = ds.with_goal(goal)
    assert g.success_labels == list(goal) and g.calls["success"].sum() == ds.calls["disposition"].isin(goal).sum()
    assert ds.with_goal(None) is ds and ds.with_goal(list(ds.success_labels)) is ds
    cfg = load_scenario(ROOT / "scenarios" / "null.yaml", primary_goal=goal)
    r = run_experiment(cfg, ds, keep_calls=False)
    assert r.decision in {"INCONCLUSIVE", "PROMOTE", "STOP_HARM", "HOLD_FOR_APPROVAL", "INVALID_SRM"}
    assert abs(r.design["baseline_rate_per_caller"] - run_experiment(
        load_scenario(ROOT / "scenarios" / "null.yaml"), ds, keep_calls=False).design["baseline_rate_per_caller"]) > 0.01


@pytest.mark.parametrize("approve,status,share,decision", [
    (True, "promoted", 1.0, "PROMOTE"), (False, "rejected", 0.0, "REJECTED_BY_APPROVER")])
def test_hold_is_resolved_by_a_logged_approver(ds, tmp_path, approve, status, share, decision):
    from abtest import store
    from abtest.runner import resolve_hold
    con = store.connect(tmp_path / "ab.db")
    run_experiment(load_scenario(ROOT / "scenarios" / "hold.yaml"), ds, con=con, keep_calls=False)
    with pytest.raises(ValueError):
        resolve_hold(con, "exp-hold", approve, "  ")           # an approver name is required
    resolve_hold(con, "exp-hold", approve, "Priya (category lead)", "longer calls are worth it")
    last = store.read(con, "SELECT * FROM decisions WHERE experiment_id = 'exp-hold' ORDER BY decision_id").iloc[-1]
    assert (last["decision"], last["actor"]) == (decision, "approver: Priya (category lead)")
    assert last["rule_fired"] == ("R7_APPROVED" if approve else "R7_REJECTED")
    assert last["narration"] == "longer calls are worth it"
    routing = store.read(con, "SELECT share_b FROM routing_config WHERE experiment_id = 'exp-hold' "
                              "ORDER BY routing_version")["share_b"].tolist()
    assert routing == [0.1, 0.0, share]                       # start, held (safe default), approver's call
    assert store.read(con, "SELECT status FROM experiments WHERE experiment_id = 'exp-hold'")["status"][0] == status
    with pytest.raises(ValueError):
        resolve_hold(con, "exp-hold", approve, "someone")      # already resolved
