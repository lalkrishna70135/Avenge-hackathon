"""Run the whole Sarvam prompt-lab pipeline in order. Every stage resumes, so re-running is cheap.

Usage: .venv/bin/python scripts/pipeline.py [--recordings <call_recordings.zip>] [--personas 60] [--pairs 10]

  1 transcribe_calls     real recordings -> diarized transcripts (Sarvam batch STT)
  2 analyse_calls        transcripts -> Call Quality Matrix audit + buyer personas (sarvam-105b)
  3 build_core_prompt    163 KB VANI prompt -> condensed bot prompt (skipped if it exists; --rebuild-core)
  4 simulate_pilot       persona buyers x bot A / bot B per prompt change -> paired pilot estimate
  5 pilot_to_scenario    pilot estimate -> scenarios/pilot.yaml
  6 run_scenarios        A/B engine on the real call log for win / harm / null / pilot
  7 render_voice         Sarvam TTS for selected A/B conversation pairs
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

ap = argparse.ArgumentParser()
ap.add_argument("--recordings", help="call_recordings.zip or folder (only needed the first time)")
ap.add_argument("--personas", type=int, default=60)
ap.add_argument("--pairs", type=int, default=10)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--rebuild-core", action="store_true")
args = ap.parse_args()

stages = [
    ["transcribe_calls.py"] + (["--from", args.recordings] if args.recordings else []),
    ["analyse_calls.py", "--workers", str(args.workers)],
]
if args.rebuild_core or not (ROOT / "data" / "sarvam" / "vani_core.txt").exists():
    stages.append(["build_core_prompt.py"])
stages += [
    ["simulate_pilot.py", "--personas", str(args.personas), "--workers", str(args.workers)],
    ["pilot_to_scenario.py"],
    ["run_scenarios.py"],
    ["render_voice.py", "--pairs", str(args.pairs)],
]
for cmd in stages:
    print(f"\n=== {cmd[0]} ===", flush=True)
    rc = subprocess.call([PY, str(ROOT / "scripts" / cmd[0]), *cmd[1:]], cwd=ROOT)
    if rc:
        sys.exit(f"{cmd[0]} failed with exit code {rc}; fix and re-run (finished work is kept)")
print("\nDone. Start the dashboard: .venv/bin/streamlit run app/dashboard.py")
