"""Split-accuracy evidence: configured vs achieved share, stickiness, salt independence, SRM.

Usage: .venv/bin/python scripts/split_report.py     (also writes runs/split_report.json)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from abtest import split  # noqa: E402
from abtest.loader import load  # noqa: E402

ds = load()
ids = split.caller_population(ds.calls["caller_id"])
salt = "exp-win-7f3a"

acc = split.split_accuracy(ids, salt)
sticky = split.stickiness(ids, salt, 0.10)
indep = split.salt_independence(ids, "exp-win-7f3a", "exp-harm-91c2")
broken = split.broken_split_demo(ids, salt)
coverage = [split.ci_coverage(ids, share, 200) for share in (0.01, 0.10)]

pd.set_option("display.width", 160)
print(f"Callers: {len(ids):,} ({ds.calls['caller_id'].nunique()} real from the call log, rest synthetic)\n")
print("Configured vs achieved share (95% Wilson CI, SRM chi-square)")
print(acc.round(4).to_string(index=False), "\n")
print("With five 95% intervals, an occasional row outside its CI is expected (1 - 0.95^5 = 23% chance of at least one).")
for c in coverage:
    print(f"Calibration over {c['salts']} salts at {c['share']:.0%}: configured share inside the 95% CI "
          f"{c['coverage']:.1%} of the time; SRM false alarms {c['srm_false_alarms']}")
print()
print(f"Stickiness: {sticky['callers']:,} callers assigned twice in shuffled order -> "
      f"{sticky['mismatches']} mismatches, {sticky['consistency']:.0%} consistent")
print(f"Salt independence (10% B): bucket correlation {indep['bucket_corr']:+.4f}, B in both "
      f"{indep['b_in_both']} vs {indep['expected_b_in_both']:.0f} expected, p = {indep['independence_p']:.3f}")
print(f"Broken split (lose {broken['callers_lost_from_b']} B callers): achieved {broken['achieved_share_b']:.4f}, "
      f"SRM p = {broken['srm_p']:.2e} -> {'FLAGGED' if broken['srm_flag'] else 'not flagged'}")

out = ROOT / "runs"
out.mkdir(exist_ok=True)
(out / "split_report.json").write_text(json.dumps(
    {"accuracy": acc.to_dict("records"), "stickiness": sticky, "independence": indep, "broken": broken, "coverage": coverage},
    indent=2, default=float))
