import numpy as np
import pytest

from abtest import stats as st

T5 = [0.2, 0.4, 0.6, 0.8, 1.0]


def test_obf_boundaries_match_gsdesign():
    # gsDesign sfLDOF, 5 equally spaced looks, one-sided alpha 0.025
    expected = [4.877, 3.357, 2.680, 2.290, 2.031]
    assert st.gs_upper_boundaries(T5, 0.025, "obf") == pytest.approx(expected, abs=0.01)


def test_obf_boundaries_control_type_one_error():
    """Monte Carlo: under H0, P(any Z_k >= b_k) should equal the one-sided alpha."""
    rng = np.random.default_rng(0)
    b = np.array(st.gs_upper_boundaries(T5, 0.025, "obf"))
    s = np.cumsum(rng.standard_normal((200_000, 5)) * np.sqrt(0.2), axis=1)   # score process
    z = s / np.sqrt(np.array(T5))
    assert (z >= b).any(axis=1).mean() == pytest.approx(0.025, abs=0.002)


def test_pocock_harm_boundary_controls_error():
    rng = np.random.default_rng(1)
    b = np.array(st.gs_upper_boundaries(T5, 0.05, "pocock"))
    z = np.cumsum(rng.standard_normal((200_000, 5)) * np.sqrt(0.2), axis=1) / np.sqrt(np.array(T5))
    assert (z >= b).any(axis=1).mean() == pytest.approx(0.05, abs=0.003)


def test_sample_size_textbook_example():
    # 10% -> 12%, alpha 0.05, power 0.8: (1.96 + 0.8416)^2 * 0.1956 / 0.02^2 = 3839 per arm
    ss = st.sample_size_two_prop(0.10, 0.12, 0.05, 0.80, share_b=0.5, gs_inflation=1.0)
    assert ss["per_arm_equal_split"] == pytest.approx(3839, abs=2)
    assert ss["n_a"] == pytest.approx(3839, abs=2)


def test_unequal_split_needs_more_callers():
    eq = st.sample_size_two_prop(0.5, 0.53, share_b=0.5)["n_total"]
    tenth = st.sample_size_two_prop(0.5, 0.53, share_b=0.1)["n_total"]
    assert tenth > 2.5 * eq


def test_two_prop_ztest_sign_and_ci():
    r = st.two_prop_ztest(500, 1000, 580, 1000)
    assert r.z > 0 and r.lift_abs == pytest.approx(0.08)
    assert r.ci_low < 0.08 < r.ci_high


def test_srm_fires_on_broken_split():
    assert st.srm_check(9000, 1000, 0.10)["srm"] is False
    assert st.srm_check(9000, 800, 0.10)["srm"] is True


def test_duration_guardrail_ratio_of_means():
    rng = np.random.default_rng(2)
    a = rng.choice([0, 60, 120, 180], size=5000, p=[0.2, 0.3, 0.3, 0.2]).astype(float)
    same = st.duration_guardrail(a, rng.permutation(a), ni_margin=0.15)
    assert same.ratio == pytest.approx(1.0, abs=1e-9)
    assert same.ni_ok and not same.ni_violated and not same.hard_breach
    longer = st.duration_guardrail(a, a * 1.5, ni_margin=0.15, hard_margin=0.30)
    assert longer.ni_violated and longer.hard_breach and not longer.ni_ok


def test_duration_guardrail_drops_nan():
    r = st.duration_guardrail([60, 120, np.nan, 180], [60, np.nan, 120, 180])
    assert r.n_a == 3 and r.n_b == 3


def test_ni_power_increases_with_margin():
    assert st.ni_power(1.5, 16000, 1800, 0.15) > st.ni_power(1.5, 16000, 1800, 0.10)
