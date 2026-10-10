"""Run the demo scenarios end to end into runs/ab.db and print a summary.

Usage: .venv/bin/python scripts/run_scenarios.py [scenarios/win.yaml ...]   (default: all scenarios)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from abtest import store  # noqa: E402
from abtest.config import load_scenario  # noqa: E402
from abtest.loader import load  # noqa: E402
from abtest.runner import run_experiment  # noqa: E402

paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "scenarios").glob("*.yaml"))
ds = load()
card = ds.card()
print(f"Data: {card['source']} ({'SYNTHETIC' if ds.synthetic else 'real'}), {card['n_calls']} calls, "
      f"{card['n_callers']} callers, lead-created rate {card['success_rate_per_call']:.1%} per call\n")

con = store.connect()
rows = []
for path in paths:
    cfg = load_scenario(path)
    r = run_experiment(cfg, ds, con=con)
    last = r.looks[-1]
    d = r.design
    rows.append({
        "experiment": cfg.experiment_id,
        "injected_pp": cfg.true_effect.success_pp,
        "dur_mult": cfg.true_effect.duration_mult,
        "decision": r.decision,
        "rule": r.rule,
        "look": f"{r.decision_look}/{cfg.n_looks}",
        "hour": r.decision_hour,
        "n_a": last["n_a"],
        "n_b": last["n_b"],
        "lift_pp": round(100 * last["lift_abs"], 2),
        "z": round(last["z"], 2),
        "eff_bound": round(last["eff_bound"], 2),
        "harm_bound": round(last["harm_bound"], 2),
        "dur_ratio": round(last["duration"]["ratio"], 3),
        "share_b_after": r.final_share_b,
    })
    print(f"{cfg.experiment_id}: planned {d['sample_size']['n_total']:,} callers "
          f"({d['sample_size']['n_b']:,} in B) over {d['window_hours']} virtual hours, "
          f"baseline {d['baseline_rate_per_caller']:.1%} per caller, MDE {cfg.mde_pp:+.0%}; "
          f"guardrail power {d['guardrail']['power_at_final_look']:.0%} at a {cfg.ni_margin:.0%} margin")

pd.set_option("display.width", 200)
print()
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nWrote {store.DB_PATH.relative_to(ROOT)}")
