import numpy as np

from abtest.assigner import Router, assign, assign_many, bucket
from abtest.split import salt_independence, split_accuracy, stickiness

IDS = [f"C{i:06d}" for i in range(20_000)]


def test_assignment_is_deterministic_and_sticky():
    assert all(assign("s1", c, 0.1) == assign("s1", c, 0.1) for c in IDS[:500])
    assert stickiness(IDS, "s1", 0.1)["mismatches"] == 0


def test_vectorised_matches_scalar():
    v = assign_many("s1", IDS[:2000], 0.25)
    assert list(v) == [assign("s1", c, 0.25) for c in IDS[:2000]]


def test_bucket_range_and_uniformity():
    b = np.array([bucket("s1", c) for c in IDS])
    assert b.min() >= 0 and b.max() < 10_000
    hist, _ = np.histogram(b, bins=10, range=(0, 10_000))
    assert hist.min() > 1800 and hist.max() < 2200       # 2000 expected per decile


def test_achieved_share_close_to_configured():
    acc = split_accuracy(IDS, "s1", shares=(0.1, 0.5))
    assert (abs(acc["achieved_share_b"] - acc["configured_share_b"]) < 0.01).all()
    assert not acc["srm_flag"].any()


def test_salts_are_independent():
    r = salt_independence(IDS, "exp-a", "exp-b")
    assert abs(r["bucket_corr"]) < 0.03
    assert r["independence_p"] > 0.001


def test_router_terminal_routing():
    assert Router("s", 1.0).route("x") == "B"
    assert Router("s", 0.0).route("x") == "A"
