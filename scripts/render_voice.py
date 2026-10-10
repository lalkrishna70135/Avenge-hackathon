"""Stage 6: voice selected simulated conversations with Sarvam TTS (bulbul:v3), as A/B pairs per persona.

Usage: .venv/bin/python scripts/render_voice.py [--pairs 10]
Writes runs/audio/<persona>__<arm>__<experiment>.wav. Picks, per experiment, the personas whose A and B
outcomes differ first (the most instructive), then the rest. Existing files are skipped.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abtest import convo, voice  # noqa: E402
from abtest.realcalls import load_real_calls  # noqa: E402
from abtest.sarvam import run_pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", type=int, default=10, help="A/B pairs in total, spread across experiments")
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

convs = convo.load_conversations()
personas = {r["recording"]: r.get("persona") or {} for r in load_real_calls()}
a = {c["persona_id"]: c for c in convs if c["arm"] == "A"}
exps = sorted({c["experiment_id"] for c in convs if c["arm"] == "B"} - {convo.AA_ID})   # A/A: nothing to hear
per_exp = max(1, args.pairs // max(len(exps), 1))
jobs = []
for e in exps:
    b = {c["persona_id"]: c for c in convs if c["arm"] == "B" and c["experiment_id"] == e}
    ids = sorted(set(a) & set(b), key=lambda i: (bool((a[i]["judge"] or {}).get("lead_created"))
                                                 == bool((b[i]["judge"] or {}).get("lead_created")), i))
    for i in ids[:per_exp]:
        jobs += [(a[i], e), (b[i], e)]


def path_for(c, e):
    return voice.AUDIO_DIR / f"{c['persona_id']}__{c['arm']}__{e}.wav"


jobs = [(c, e) for c, e in jobs if not path_for(c, e).exists()]
voice.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
print(f"{len(jobs)} conversations to voice")


def one(job):
    c, e = job
    p = personas.get(c["persona_id"], {})
    data = voice.render_conversation(c["turns"], p.get("gender") or "unknown", p.get("language"))
    path_for(c, e).write_bytes(data)
    return voice.wav_seconds(data)


secs, failed = run_pool(one, jobs, workers=args.workers, label="voiced", every=2)
for (c, e), err in failed:
    print(f"  failed: {c['persona_id']} {c['arm']} {e}: {err}")
if secs:
    print(f"voiced {len(secs)} conversations, {sum(secs) / 60:.1f} min of audio")
