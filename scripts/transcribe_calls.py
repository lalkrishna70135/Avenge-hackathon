"""Stage 1: real VANI recordings -> diarized transcripts with Sarvam batch STT (saaras:v3, codemix).

Usage: .venv/bin/python scripts/transcribe_calls.py [--from <call_recordings.zip or folder>] [--limit N]
Copies recordings into data/recordings/ (once), skips files under 20 KB, and writes
data/sarvam/transcripts/<file>.mp3.json. Already-transcribed files are skipped, so it can be re-run.
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abtest.realcalls import MIN_BYTES, REC_DIR, transcribe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="src", help="call_recordings.zip or a folder of .mp3 files")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

REC_DIR.mkdir(parents=True, exist_ok=True)
if args.src:
    src = Path(args.src)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for name in z.namelist():
                if name.lower().endswith(".mp3"):
                    target = REC_DIR / Path(name).name          # flatten; never trust paths inside a zip
                    if not target.exists():
                        target.write_bytes(z.read(name))
    else:
        for p in src.glob("*.mp3"):
            if not (REC_DIR / p.name).exists():
                shutil.copy2(p, REC_DIR / p.name)

files = sorted(p for p in REC_DIR.glob("*.mp3") if p.stat().st_size >= MIN_BYTES)
if args.limit:
    files = files[:args.limit]
print(f"{len(files)} recordings of at least {MIN_BYTES // 1000} KB")
errors = transcribe(files)
for e in errors:
    print("  problem:", e)
