"""Versioned, immutable experiment config. The sha256 of its canonical JSON is the
config_hash stamped on every call and decision, so any result can be reproduced."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent


class Variant(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    prompt: str | None = None        # full prompt, or
    diff: str | None = None          # instruction diff applied to the base prompt
    anchor: str | None = None        # base-prompt line where the section the diff edits starts
    window: int | None = None        # number of lines in that section (default 30)
    prompt_file: str | None = None   # rewritten prompt section generated from `diff` (scripts/gen_variant.py)
    prompt_sha256: str | None = None # filled by load_scenario so the config hash pins the exact text


class TrueEffect(BaseModel):
    """Simulation-only: the injected truth for variant B. Never seen by the stats engine."""
    model_config = ConfigDict(frozen=True)
    success_pp: float = 0.0          # absolute change in P(lead created | connected call)
    duration_mult: float = 1.0


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_id: str
    version: int = 1
    title: str = ""
    salt: str
    base_prompt: str = ""
    base_prompt_file: str | None = None
    base_prompt_sha256: str | None = None   # filled by load_scenario
    variant_a: Variant = Variant(name="A (control)")
    variant_b: Variant = Variant(name="B")
    share_b: float = Field(0.10, gt=0, lt=1)

    primary_metric: Literal["success"] = "success"   # lead created (BL Enriched/Approved/Generated)
    # dispositions that count as success for the primary goal; None = lead created (the loader's labels)
    primary_goal: tuple[str, ...] | None = None
    primary_direction: Literal["higher_is_better"] = "higher_is_better"
    # Each caller is scored on their calls within this many hours of their first call, and enters the analysis
    # only once that window has closed, so a caller's outcome never changes between looks (the boundaries
    # assume independent increments). 81% of real repeat calls come within 1 h of the first call.
    score_window_hours: float | None = 2.0
    secondary_metric: Literal["duration_s"] = "duration_s"
    secondary_direction: Literal["lower_is_better"] = "lower_is_better"
    # Duration is a minute-resolution proxy with a ~2 min median, so 10% (~12 s) is below what it
    # can resolve. plan_design reports the guardrail's power at the planned sample for this margin.
    ni_margin: float = 0.15          # B mean duration may not exceed A by more than 15% (95% CI upper)
    hard_margin: float = 0.30        # hard stop if 99% CI lower bound of ratio > 1 + hard_margin

    alpha: float = 0.05              # two-sided; efficacy boundary uses alpha/2 one-sided
    power: float = 0.80
    mde_pp: float = 0.02             # minimum detectable absolute lift in lead-created rate (per caller)
    baseline_rate: float | None = None   # filled from data if None
    n_looks: int = 5
    efficacy_spending: Literal["obf", "pocock"] = "obf"
    harm_alpha: float = 0.05         # one-sided
    harm_spending: Literal["obf", "pocock"] = "pocock"
    min_sample_per_arm: int = 300
    srm_threshold: float = 0.001
    auto_promote: bool = True

    # simulator
    calls_per_hour: float = 250.0
    repeat_call_prob: float = 0.25   # chance an arrival is a returning caller
    window_hours: float | None = None    # derived from sample size if None
    post_rollout_hours: float = 12.0
    seed: int = 1
    true_effect: TrueEffect = TrueEffect()
    # set when true_effect came from a simulated-conversation pilot (scripts/pilot_to_scenario.py)
    pilot_of: str | None = None
    pilot_evidence: dict | None = None

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def bump(self, **changes) -> "ExperimentConfig":
        """Configs are immutable: any change produces a new version."""
        return self.model_copy(update={**changes, "version": self.version + 1})


def _sha(path: str) -> str | None:
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def load_scenario(path: str | Path, **overrides) -> ExperimentConfig:
    """Prompt texts live in files; their content hashes go into the config so that editing a
    prompt changes the config hash."""
    data = yaml.safe_load(Path(path).read_text())
    data.update(overrides)
    if data.get("base_prompt_file"):
        data["base_prompt_sha256"] = _sha(data["base_prompt_file"])
    vb = data.get("variant_b") or {}
    if vb.get("prompt_file"):
        vb["prompt_sha256"] = _sha(vb["prompt_file"])
    return ExperimentConfig(**data)
