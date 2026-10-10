"""Stats engine: sample size, proportion tests, group-sequential boundaries,
duration guardrail (Welch on log-duration), split checks (Wilson CI, SRM).

Everything here is a pure function of its inputs so it can be unit-tested and
re-run from the decision log evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import optimize, stats

# ----------------------------------------------------------------------------
# Sample size
# ----------------------------------------------------------------------------


def sample_size_two_prop(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8,
                         share_b: float = 0.5, gs_inflation: float = 1.0) -> dict:
    """Total sample (callers) for a two-sided two-proportion test with allocation share_b to B.

    n_total = (z_{1-a/2} + z_{1-b})^2 * [p1 q1/(1-r) + p2 q2/r] / (p2-p1)^2
    With r = 0.5 this reduces to the textbook per-arm formula.
    """
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    r = share_b
    var = p1 * (1 - p1) / (1 - r) + p2 * (1 - p2) / r
    n_total = (za + zb) ** 2 * var / (p2 - p1) ** 2 * gs_inflation
    n_total = math.ceil(n_total)
    return {
        "n_total": n_total,
        "n_a": math.ceil(n_total * (1 - r)),
        "n_b": math.ceil(n_total * r),
        "per_arm_equal_split": math.ceil((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2),
        "gs_inflation": gs_inflation,
    }


# ----------------------------------------------------------------------------
# Primary metric: two-proportion z-test
# ----------------------------------------------------------------------------


@dataclass
class PropTest:
    n_a: int
    n_b: int
    x_a: int
    x_b: int
    p_a: float
    p_b: float
    lift_abs: float
    lift_rel: float
    se: float
    z: float
    p_two_sided: float
    ci_low: float
    ci_high: float

    def to_dict(self):
        return asdict(self)


def two_prop_ztest(x_a: int, n_a: int, x_b: int, n_b: int, conf: float = 0.95) -> PropTest:
    """Unpooled two-proportion z (B - A). Unpooled SE is what the sequential Z uses."""
    p_a = x_a / n_a if n_a else 0.0
    p_b = x_b / n_b if n_b else 0.0
    se = math.sqrt(p_a * (1 - p_a) / max(n_a, 1) + p_b * (1 - p_b) / max(n_b, 1))
    if se == 0:
        # fall back to pooled variance so all-zero/all-one samples don't divide by zero
        pp = (x_a + x_b) / max(n_a + n_b, 1)
        se = math.sqrt(max(pp * (1 - pp), 1e-12) * (1 / max(n_a, 1) + 1 / max(n_b, 1)))
    d = p_b - p_a
    z = d / se
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    return PropTest(n_a, n_b, x_a, x_b, p_a, p_b, d, (d / p_a if p_a else float("nan")), se, z,
                    float(2 * stats.norm.sf(abs(z))), d - zc * se, d + zc * se)


def two_prop_z_vec(x_a, n_a, x_b, n_b):
    """Vectorised unpooled z for Monte Carlo."""
    x_a, n_a, x_b, n_b = (np.asarray(v, dtype=float) for v in (x_a, n_a, x_b, n_b))
    p_a = x_a / np.maximum(n_a, 1)
    p_b = x_b / np.maximum(n_b, 1)
    se = np.sqrt(p_a * (1 - p_a) / np.maximum(n_a, 1) + p_b * (1 - p_b) / np.maximum(n_b, 1))
    pp = (x_a + x_b) / np.maximum(n_a + n_b, 1)
    se_pool = np.sqrt(np.maximum(pp * (1 - pp), 1e-12) * (1 / np.maximum(n_a, 1) + 1 / np.maximum(n_b, 1)))
    se = np.where(se > 0, se, se_pool)
    return (p_b - p_a) / se


# ----------------------------------------------------------------------------
# Group sequential design (Lan-DeMets alpha spending)
# ----------------------------------------------------------------------------


def spend_obf(t: float, alpha: float) -> float:
    """Lan-DeMets O'Brien-Fleming-type spending for a ONE-sided level alpha."""
    if t <= 0:
        return 0.0
    t = min(t, 1.0)
    return float(2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / math.sqrt(t))))


def spend_pocock(t: float, alpha: float) -> float:
    """Lan-DeMets Pocock-type spending: alpha * ln(1 + (e-1) t)."""
    if t <= 0:
        return 0.0
    return float(alpha * math.log(1 + (math.e - 1) * min(t, 1.0)))


SPENDING = {"obf": spend_obf, "pocock": spend_pocock}


# Boundaries are computed with the Armitage-McPherson-Rowe recursive integration (the
# algorithm behind gsDesign / ldbounds). Work on the score scale S_k = Z_k * sqrt(t_k):
# S has independent N(drift*dt, dt) increments, so the sub-density of "not yet stopped"
# paths is carried look to look by a 1-D convolution on a grid.

_GRID_H = 0.01   # grid step on the score scale
_TAIL_SD = 10.0  # grid half-width in standard deviations


def _simpson_grid(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    n = max(2 * math.ceil((hi - lo) / (2 * _GRID_H)), 2) + 1
    s = np.linspace(lo, hi, n)
    h = (hi - lo) / (n - 1)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return s, w * h / 3


class _AMR:
    """Carries the continuation sub-density across looks for a single upper boundary."""

    def __init__(self, drift: float = 0.0):
        self.drift = drift
        self.t_prev = 0.0
        self.s = None   # grid of the continuation region at the previous look
        self.g = None   # sub-density * quadrature weight on that grid

    def cross_prob(self, t: float, b: float) -> float:
        """P(not stopped before, S_t >= b)."""
        dt = t - self.t_prev
        if self.s is None:
            return float(stats.norm.sf((b - self.drift * t) / math.sqrt(t)))
        return float(np.sum(self.g * stats.norm.sf((b - self.s - self.drift * dt) / math.sqrt(dt))))

    def advance(self, t: float, b: float) -> None:
        """Move to look t, keeping only paths with S_t < b (b may be +inf)."""
        dt = t - self.t_prev
        sd = math.sqrt(t)
        lo = self.drift * t - _TAIL_SD * sd
        hi = b if math.isfinite(b) else self.drift * t + _TAIL_SD * sd
        hi = max(hi, lo + 4 * _GRID_H)
        s_new, w_new = _simpson_grid(lo, hi)
        if self.s is None:
            f = stats.norm.pdf((s_new - self.drift * t) / sd) / sd
        else:
            sdd = math.sqrt(dt)
            kern = stats.norm.pdf((s_new[:, None] - self.s[None, :] - self.drift * dt) / sdd) / sdd
            f = kern @ self.g
        self.s, self.g, self.t_prev = s_new, f * w_new, t


def gs_upper_boundaries(ts, alpha: float, spending: str = "obf") -> list[float]:
    """Upper one-sided Z boundaries c_1..c_K for information fractions ts (increasing; pass
    1.0 at the final look to spend all remaining alpha).

    c_k solves P(no crossing before k, Z_k >= c_k) = alpha(t_k) - alpha(t_{k-1}).
    Non-binding with respect to the opposite (harm) boundary: standard and conservative.
    """
    return list(_gs_upper_cached(tuple(round(float(t), 6) for t in ts), float(alpha), spending))


@lru_cache(maxsize=4096)
def _gs_upper_cached(ts: tuple, alpha: float, spending: str) -> tuple:
    f = SPENDING[spending]
    amr = _AMR(0.0)
    bounds: list[float] = []
    prev_spent = 0.0
    for t in ts:
        spent = f(t, alpha)
        inc = spent - prev_spent
        sq = math.sqrt(t)
        if inc <= 1e-12:
            c = float("inf")
        else:
            g = lambda c: amr.cross_prob(t, c * sq) - inc  # noqa: E731
            c = float(optimize.brentq(g, -3.0, 12.0, xtol=1e-7)) if g(-3.0) > 0 else -3.0
        bounds.append(c)
        amr.advance(t, c * sq)
        prev_spent = spent
    return tuple(bounds)


def gs_power(ts, bounds, drift: float) -> float:
    """P(cross upper boundary at some look) when E[Z_k] = drift*sqrt(t_k)."""
    amr = _AMR(drift)
    total = 0.0
    for t, c in zip(ts, bounds):
        b = c * math.sqrt(t)
        if math.isfinite(b):
            total += amr.cross_prob(t, b)
        amr.advance(t, b)
    return total


def gs_inflation_factor(n_looks: int, alpha_one_sided: float, power: float, spending: str = "obf") -> float:
    """Max-sample inflation vs a fixed design so the GS design keeps the target power."""
    ts = [(k + 1) / n_looks for k in range(n_looks)]
    bounds = gs_upper_boundaries(ts, alpha_one_sided, spending)
    target = stats.norm.ppf(1 - alpha_one_sided) + stats.norm.ppf(power)
    drift = optimize.brentq(lambda d: gs_power(ts, bounds, d) - power, 0.5, 10)
    return float((drift / target) ** 2)


# ----------------------------------------------------------------------------
# Secondary metric: duration guardrail (non-inferiority on the ratio of geometric means)
# ----------------------------------------------------------------------------


@dataclass
class DurationTest:
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    ratio: float          # ratio of means B/A
    ci_low: float
    ci_high: float
    ci_low_hard: float    # lower bound of the stricter (hard-stop) CI
    t: float
    df: float
    p_two_sided: float
    ni_margin: float
    ni_ok: bool           # upper CI < 1 + ni_margin: non-inferiority shown
    ni_violated: bool     # lower CI > 1 + ni_margin: B is clearly longer than the margin allows
    hard_margin: float
    hard_breach: bool     # strict lower CI > 1 + hard_margin
    mw_p: float           # Mann-Whitney robustness check

    def to_dict(self):
        return asdict(self)


def duration_guardrail(dur_a, dur_b, ni_margin: float = 0.10, hard_margin: float = 0.20,
                       conf: float = 0.95, hard_conf: float = 0.99) -> DurationTest:
    """Non-inferiority on the ratio of MEAN durations B/A (raw seconds, NaN dropped).

    The duration proxy has minute resolution and many zeros, so a log transform is unsafe. The CI
    for log(mean_b/mean_a) uses the delta method, se^2 = var_a/(n_a mean_a^2) + var_b/(n_b mean_b^2),
    with Welch-Satterthwaite degrees of freedom on the same terms."""
    a = np.asarray(pd.Series(dur_a, dtype=float).dropna())
    b = np.asarray(pd.Series(dur_b, dtype=float).dropna())
    na, nb = len(a), len(b)
    ma, mb = (a.mean() if na else float("nan")), (b.mean() if nb else float("nan"))
    if na < 2 or nb < 2 or ma <= 0 or mb <= 0:
        nan = float("nan")
        return DurationTest(na, nb, float(ma), float(mb), nan, nan, nan, nan, nan, nan, nan,
                            ni_margin, False, False, hard_margin, False, nan)
    ta, tb = a.var(ddof=1) / (na * ma ** 2), b.var(ddof=1) / (nb * mb ** 2)
    se = math.sqrt(ta + tb) or 1e-9
    d = math.log(mb / ma)
    den = ta ** 2 / (na - 1) + tb ** 2 / (nb - 1)
    df = (ta + tb) ** 2 / den if den > 0 else float(na + nb - 2)
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df)
    tcrit_hard = stats.t.ppf(1 - (1 - hard_conf) / 2, df)
    lo, hi = math.exp(d - tcrit * se), math.exp(d + tcrit * se)
    lo_hard = math.exp(d - tcrit_hard * se)
    t = d / se
    mw_p = float(stats.mannwhitneyu(a, b).pvalue)
    return DurationTest(na, nb, float(ma), float(mb), math.exp(d), lo, hi, lo_hard, t, df,
                        float(2 * stats.t.sf(abs(t), df)), ni_margin, hi < 1 + ni_margin,
                        lo > 1 + ni_margin, hard_margin, lo_hard > 1 + hard_margin, mw_p)


def ni_power(cv: float, n_a: int, n_b: int, margin: float, conf: float = 0.95, true_ratio: float = 1.0) -> float:
    """Power of the ratio-of-means non-inferiority test (upper CI < 1 + margin) at the given
    per-arm sizes, with coefficient of variation `cv` in both arms (delta method, normal approx)."""
    se = cv * math.sqrt(1 / max(n_a, 1) + 1 / max(n_b, 1))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    return float(stats.norm.cdf((math.log(1 + margin) - math.log(true_ratio)) / se - z))


# ----------------------------------------------------------------------------
# Split checks
# ----------------------------------------------------------------------------


def wilson_ci(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def srm_check(n_a: int, n_b: int, share_b: float, threshold: float = 0.001) -> dict:
    """Chi-square goodness-of-fit of observed (A, B) caller counts vs the configured split."""
    n = n_a + n_b
    if n == 0 or share_b in (0.0, 1.0):
        return {"chi2": 0.0, "p": 1.0, "srm": False, "threshold": threshold}
    exp = np.array([n * (1 - share_b), n * share_b])
    chi2, p = stats.chisquare([n_a, n_b], exp)
    return {"chi2": float(chi2), "p": float(p), "srm": bool(p < threshold), "threshold": threshold}


def detectable_effect(p_a: float, n_a: int, n_b: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift detectable with the sample actually collected (for 'inconclusive')."""
    if n_a == 0 or n_b == 0:
        return float("nan")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    se = math.sqrt(p_a * (1 - p_a) * (1 / n_a + 1 / n_b))
    return float((za + zb) * se)
