"""Stage 2: transcripts -> VANI Call Quality Matrix verdicts + a buyer persona per call (sarvam-105b).

Usage: .venv/bin/python scripts/analyse_calls.py [--limit N] [--workers 6]
Writes data/sarvam/real_calls.jsonl. Re-running only analyses transcripts not yet in the file.
"""
import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abtest.realcalls import REAL_CALLS, TRANSCRIPT_DIR, analyse, load_real_calls  # noqa: E402
from abtest.sarvam import run_pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()

done = {r["recording"] for r in load_real_calls()}
todo = [p for p in sorted(TRANSCRIPT_DIR.glob("*.json")) if p.name.removesuffix(".json") not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"{len(done)} already analysed, {len(todo)} to go")
lock = threading.Lock()


def one(path):
    rec = analyse(path)
    with lock, REAL_CALLS.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


results, failed = run_pool(one, todo, workers=args.workers, label="calls analysed")
for item, err in failed:
    print("  failed:", item.name, err)
usable = [r for r in load_real_calls() if r.get("usable")]
print(f"{len(usable)} usable calls with QA + persona in {REAL_CALLS.relative_to(ROOT)}")
