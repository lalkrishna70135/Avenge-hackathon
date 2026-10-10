"""Stage 4: simulated pilot. Personas from real calls talk to bot A and bot B for each experiment.

Usage: .venv/bin/python scripts/simulate_pilot.py [--personas 60] [--experiments exp-win,exp-harm] [--workers 8] [--no-aa]
Appends to runs/pilot/conversations.jsonl (finished conversations are skipped on re-run) and writes
runs/pilot/summary.json. The personas used are frozen in runs/pilot/personas.json. Control A conversations are shared by every experiment (identical prompt).
"""
import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from abtest import catalog, convo, pilot  # noqa: E402
from abtest.realcalls import load_real_calls  # noqa: E402
from abtest.sarvam import run_pool, usage_totals, write_atomic  # noqa: E402

PERSONAS = convo.PILOT_DIR / "personas.json"
FIELDS = convo.PILOT_DIR / "fields.json"

ap = argparse.ArgumentParser()
ap.add_argument("--personas", type=int, default=60)
ap.add_argument("--experiments", default=None, help="comma-separated; default all with a B prompt")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no-aa", action="store_true", help="skip the A/A check (prompt A against an independent replicate)")
args = ap.parse_args()

exps = args.experiments.split(",") if args.experiments else list(convo.experiments())
if not args.no_aa and convo.AA_ID not in exps:
    exps.append(convo.AA_ID)          # noise yardstick: B is an independent second call on prompt A
calls = [r for r in load_real_calls() if r.get("usable") and r.get("persona", {}).get("product")]
by_id = {c["recording"]: c for c in calls}
# The persona list is frozen in runs/pilot/personas.json on the first run, so auditing more real calls later
# never changes which buyers a resumed or extended pilot uses; new personas are only appended.
frozen = json.loads(PERSONAS.read_text()) if PERSONAS.exists() else []
picked = [by_id[i] for i in frozen if i in by_id][:args.personas]
if len(picked) < args.personas:
    # spread personas across real outcomes: alternate lead / no-lead calls, in a fixed order
    rng = np.random.default_rng(5)
    have = {c["recording"] for c in picked}
    lead = [c for c in calls if (c.get("qa") or {}).get("lead_created")]
    nolead = [c for c in calls if not (c.get("qa") or {}).get("lead_created")]
    rng.shuffle(lead)
    rng.shuffle(nolead)
    lead, nolead = [c for c in lead if c["recording"] not in have], [c for c in nolead if c["recording"] not in have]
    while len(picked) < args.personas and (lead or nolead):
        for pool in (lead, nolead):
            if pool and len(picked) < args.personas:
                picked.append(pool.pop())
    PERSONAS.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(PERSONAS, json.dumps(list(dict.fromkeys(frozen + [c["recording"] for c in picked])), indent=1))
print(f"{len(picked)} personas from {len(calls)} usable real calls; experiments {exps}")

core = convo.CORE.read_text()
convs = convo.load_conversations()
# Collection fields are pinned per persona in runs/pilot/fields.json (the A call's, when one exists), so every
# arm of a persona gets the same ones. A conversation run with other fields is set aside and re-run.
pinned = json.loads(FIELDS.read_text()) if FIELDS.exists() else {}
a_fields = {c["persona_id"]: c.get("fields") for c in convs if c["arm"] == "A" and c.get("fields")}
for p in picked:
    pid = p["recording"]
    pinned.setdefault(pid, a_fields.get(pid) or catalog.collection_fields(p["persona"].get("product") or "a product"))
write_atomic(FIELDS, json.dumps(pinned, ensure_ascii=False, indent=1))
stale = [c for c in convs if c["persona_id"] in pinned and c.get("fields") != pinned[c["persona_id"]]]
if stale:
    with (convo.PILOT_DIR / "conversations_set_aside.jsonl").open("a") as fh:
        fh.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in stale)
    keep = [c for c in convs if not (c["persona_id"] in pinned and c.get("fields") != pinned[c["persona_id"]])]
    write_atomic(convo.CONVERSATIONS, "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in keep))
    print(f"{len(stale)} conversations ran with other collection fields than their persona's pinned ones: "
          "set aside in conversations_set_aside.jsonl and re-run")
done = {(c["persona_id"], c["arm"], c["experiment_id"]) for c in convo.load_conversations()}
jobs = []
for p in picked:
    pid = p["recording"]
    if (pid, "A", "control") not in done:
        jobs.append((p, "A", "control"))
    jobs += [(p, "B", e) for e in exps if (pid, "B", e) not in done]
print(f"{len(done)} conversations already done, {len(jobs)} to run")
convo.PILOT_DIR.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()


def one(job):
    p, arm, eid = job
    if eid == convo.AA_ID:            # the A/A arm: prompt A again, sampled independently, scored as "B"
        c = convo.simulate(p["persona"], "A", None, p["recording"], core, pinned[p["recording"]], replicate=1)
        c["arm"], c["replicate"] = "B", 1
    else:
        c = convo.simulate(p["persona"], arm, None if arm == "A" else eid, p["recording"], core, pinned[p["recording"]])
    c["experiment_id"] = eid
    with lock, convo.CONVERSATIONS.open("a") as fh:
        fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return c


_, failed = run_pool(one, jobs, workers=args.workers, label="conversations", every=5)
for (p, arm, eid), err in failed:
    print(f"  failed: {p['recording']} {arm} {eid}: {err}")

convs = convo.load_conversations()
summary = {e: pilot.paired(convs, e) for e in exps}
write_atomic(convo.PILOT_DIR / "summary.json", json.dumps(summary, indent=2))
for e, s in summary.items():
    if s:
        print(f"{e}: {s['pairs']} pairs, lead A {s['A']['lead_rate']:.0%} -> B {s['B']['lead_rate']:.0%} "
              f"(lift {100 * s['lift_pp']:+.1f} pp, 95% CI [{100 * s['lift_ci'][0]:+.1f}, {100 * s['lift_ci'][1]:+.1f}], "
              f"paired p {s['p_value']:.3f}); asks/call {s['A']['mean_asks']:.1f} -> {s['B']['mean_asks']:.1f}; "
              f"duration x{s['duration_ratio']:.2f}; fatal {s['A']['fatal_rate']:.0%} -> {s['B']['fatal_rate']:.0%}")
print("Sarvam usage so far:", usage_totals())
