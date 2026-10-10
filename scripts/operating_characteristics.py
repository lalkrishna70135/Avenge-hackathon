"""Repeat each scenario over many seeds and report how often each decision is reached: the
engine's empirical false-positive rate (null), power (win) and harm-detection rate (harm).

Also re-runs null without the per-caller scoring window (`_diagnostic_null_without_scoring_window`): scored on
all their calls so far, a repeat caller's outcome can change between looks, which pushed the false harm-stop rate
above its 5% target. The comparison shows what the fixed window from each caller's first call fixes.

Usage: .venv/bin/python scripts/operating_characteristics.py [n_seeds]   (default 100; the slide uses 1000)
Writes runs/operating_characteristics.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abtest.config import load_scenario  # noqa: E402
from abtest.loader import load  # noqa: E402
from abtest.runner import plan_design, run_experiment  # noqa: E402

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ds = load()
out = {}
names = ("win", "harm", "null", "hold") + (("pilot",) if (ROOT / "scenarios" / "pilot.yaml").exists() else ())
for name in names:
    base = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
    design = plan_design(base, ds)        # the design is fixed up front; only the data varies
    decisions, looks, by_look = Counter(), [], Counter()
    for seed in range(1000, 1000 + n_seeds):
        r = run_experiment(base.model_copy(update={"seed": seed}), ds, design=design, keep_calls=False)
        decisions[r.decision] += 1
        by_look[f"{r.decision}@{r.decision_look}"] += 1
        if r.decision in ("PROMOTE", "STOP_HARM"):
            looks.append(r.decision_look)
    rates = {k: v / n_seeds for k, v in decisions.most_common()}
    out[name] = {"seeds": n_seeds, "rates": rates, "counts": dict(decisions), "decision_at_look": dict(sorted(by_look.items())),
                 "mean_stop_look": sum(looks) / len(looks) if looks else None}
    print(f"{name:5s} ({n_seeds} seeds): " + ", ".join(f"{k} {v:.0%}" for k, v in rates.items())
          + (f"; mean stopping look {out[name]['mean_stop_look']:.1f}" if looks else ""))

fixed = load_scenario(ROOT / "scenarios" / "null.yaml").score_window_hours
base = load_scenario(ROOT / "scenarios" / "null.yaml").model_copy(update={"score_window_hours": None})
design = plan_design(base, ds)
counts, harm_by_look = Counter(), Counter()
for seed in range(1000, 1000 + n_seeds):          # the same seeds as null above: only the scoring differs
    r = run_experiment(base.model_copy(update={"seed": seed}), ds, design=design, keep_calls=False)
    counts[r.decision] += 1
    if r.decision == "STOP_HARM":
        harm_by_look[str(r.decision_look)] += 1
out["_diagnostic_null_without_scoring_window"] = {
    "seeds": n_seeds, "seed_range": [1000, 1000 + n_seeds - 1], "score_window_hours": None,
    "score_window_hours_fixed": fixed, "counts": dict(counts.most_common()),
    "harm_stops_by_look": dict(sorted(harm_by_look.items()))}
print(f"null without the scoring window ({n_seeds} seeds): false harm-stop {counts['STOP_HARM'] / n_seeds:.1%} "
      f"(with it: {out['null']['rates'].get('STOP_HARM', 0):.1%})")

(ROOT / "runs").mkdir(exist_ok=True)
(ROOT / "runs" / "operating_characteristics.json").write_text(json.dumps(out, indent=2))
