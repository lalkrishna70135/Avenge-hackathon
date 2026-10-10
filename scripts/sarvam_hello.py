"""One Sarvam chat completion. Reads SARVAM_API_KEY from the environment or ./.env."""
import os
from pathlib import Path

from sarvamai import SarvamAI

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() and not line.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

key = os.environ.get("SARVAM_API_KEY")
if not key:
    raise SystemExit("SARVAM_API_KEY is not set. Paste it into .env first.")

client = SarvamAI(api_subscription_key=key)
resp = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[{"role": "user", "content": "Say hello in Hindi and English in one short line."}],
)
print(resp.choices[0].message.content)
