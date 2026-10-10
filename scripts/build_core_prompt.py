"""Stage 3: condense the 163 KB VANI prompt into an operating guide for simulated calls (sarvam-105b).

Usage: .venv/bin/python scripts/build_core_prompt.py
Writes data/sarvam/vani_core.txt. Variant-controlled sections are excluded here and appended verbatim
to each bot's prompt (abtest/convo.py::bot_system).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abtest.convo import CORE, bot_system, build_core, experiments, prompt_sha  # noqa: E402

core = build_core()
print(f"{CORE.relative_to(ROOT)}: {len(core):,} chars (base prompt {(ROOT / 'data/base_prompt.txt').stat().st_size:,})")
a = bot_system("A", None, "TMT bars")
print(f"bot A prompt: {len(a):,} chars, sha {prompt_sha(a)}")
for eid in experiments():
    b = bot_system("B", eid, "TMT bars")
    print(f"bot B for {eid}: {len(b):,} chars, sha {prompt_sha(b)}")
