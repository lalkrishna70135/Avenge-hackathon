"""Data loader adapter: raw historical call data -> canonical frame.

Canonical columns: caller_id, disposition, success (bool), connected (bool), duration_s (float|NaN),
ts (datetime|NaT), src_row (row in the source file)

* success   = the primary metric. For the VANI buyer-side bot: a buy lead was created
              (BL Enriched / BL Approved / BL Generated).
* connected = a conversation actually took place, so the prompt could influence the outcome.
              Calls that fail before any conversation (technical failure, queue drop, filtered
              users) are not connected; the simulator never injects an effect into them.
* duration_s is only defined for connected calls (NaN otherwise).

The VANI call log (Call_Logs_Hackathon.csv) is recognised by its columns and handled by
`_load_vani`. Other files go through the generic adapter (columns + success labels in
data/mapping.yaml, auto-detected when absent). If no data file is present, a clearly-labelled
SYNTHETIC dataset is generated; the data card says which one was used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

_CANDIDATES = {
    "caller_id": ["caller_id", "lead_id", "leadid", "customer_id", "phone", "mobile", "caller", "lead", "contact"],
    "disposition": ["disposition", "call_disposition", "outcome", "status", "call_status", "result"],
    "duration_s": ["duration", "call_duration", "duration_s", "duration_sec", "talk_time", "call_length"],
    "ts": ["timestamp", "call_time", "start_time", "created_at", "date", "call_date", "time"],
}
_SUCCESS_PATTERNS = [r"meeting\s*fixed", r"meeting\s*scheduled", r"meeting\s*booked", r"appointment"]

# VANI buyer-side call log
VANI_SUCCESS = ["BL Enriched", "BL Approved", "BL Generated"]
# no conversation happened, so no prompt can change the outcome. `is_call_answered` is NOT used:
# it is 0 on 38 "BL Enriched" calls, so it does not mean what its name says.
VANI_NOT_CONNECTED = ["Technical Issue - Call Not Connected", "Technical Issue", "Queue Drop",
                      "No Enrichment Scope", "Foreign or Disabled Users", "Unknown"]
_VANI_COLUMNS = {"disposition_remarks", "fk_glusr_usr_id", "entered_on", "last_modified_date"}


@dataclass
class Dataset:
    calls: pd.DataFrame
    source: str                      # file path or "SYNTHETIC"
    synthetic: bool
    mapping: dict
    success_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def p0(self) -> float:
        return float(self.calls["success"].mean())

    def with_goal(self, labels) -> "Dataset":
        """The same calls with `success` redefined as 'disposition is one of labels' (None: unchanged)."""
        if not labels or sorted(labels) == sorted(self.success_labels):
            return self
        calls = self.calls.assign(success=self.calls["disposition"].isin(list(labels)))
        return Dataset(calls, self.source, self.synthetic, self.mapping, list(labels),
                       self.notes + [f"Primary goal: {' / '.join(labels)}"])

    def card(self) -> dict:
        c = self.calls
        per_caller = c.groupby("caller_id").size()
        conn = c[c["connected"]]
        q = conn["duration_s"].quantile([0.1, 0.5, 0.9]).round(1).tolist()
        return {
            "source": self.source,
            "synthetic": self.synthetic,
            "n_calls": int(len(c)),
            "n_callers": int(per_caller.size),
            "repeat_caller_share": round(float((per_caller > 1).mean()), 4),
            "calls_per_caller_mean": round(float(per_caller.mean()), 3),
            "success_rate_per_call": round(self.p0, 4),
            "success_rate_per_caller": round(float(c.groupby("caller_id")["success"].max().mean()), 4),
            "connected_share": round(float(c["connected"].mean()), 4),
            "success_rate_connected": round(float(conn["success"].mean()), 4) if len(conn) else None,
            "success_labels": self.success_labels,
            "duration_p10_p50_p90_s_connected": q,
            "date_range": [str(c["ts"].min()), str(c["ts"].max())],
            "notes": self.notes,
            "dispositions": c["disposition"].value_counts().head(15).to_dict(),
        }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", str(s).strip().lower())


def _detect(columns, role):
    normed = {_norm(c): c for c in columns}
    for cand in _CANDIDATES[role]:
        if cand in normed:
            return normed[cand]
    for cand in _CANDIDATES[role]:
        for n, orig in normed.items():
            if cand in n:
                return orig
    return None


def parse_duration(v) -> float:
    """Accepts seconds, 'mm:ss', 'hh:mm:ss', '1m 20s'."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    m = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", s.lower())
    if m:
        mult = {"h": 3600, "m": 60, "s": 1}
        return float(sum(float(n) * mult[u] for n, u in m))
    return np.nan


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".csv", ".txt"):
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported file {path}")


def find_data_file(data_dir: Path = DATA_DIR) -> Path | None:
    files = [p for p in data_dir.glob("*") if p.is_file()   # top level only: data/sarvam holds JSON
             and p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet")
             and p.name != "mapping.yaml"]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def load(data_dir: Path = DATA_DIR, seed: int = 7) -> Dataset:
    mapping_path = data_dir / "mapping.yaml"
    mapping = yaml.safe_load(mapping_path.read_text()) if mapping_path.exists() else {}
    path = Path(mapping["file"]) if mapping.get("file") else find_data_file(data_dir)
    if path is None:
        return synthetic(seed=seed)
    if not path.is_absolute():
        path = data_dir / path
    raw = _read_any(path)
    if _VANI_COLUMNS <= set(raw.columns):
        return _load_vani(raw, path)
    cols = {role: mapping.get(role) or _detect(raw.columns, role) for role in _CANDIDATES}
    if cols["disposition"] is None or cols["duration_s"] is None:
        raise ValueError(f"Could not detect disposition/duration columns in {path.name}: {list(raw.columns)}. "
                         f"Write data/mapping.yaml with keys disposition, duration_s, caller_id, ts.")
    df = pd.DataFrame({
        "disposition": raw[cols["disposition"]].astype(str).str.strip(),
        "duration_s": raw[cols["duration_s"]].map(parse_duration),
    })
    if cols["caller_id"] is not None:
        df["caller_id"] = raw[cols["caller_id"]].astype(str)
    else:
        df["caller_id"] = _synth_caller_ids(len(df), np.random.default_rng(seed))
        cols["caller_id"] = "<synthesized>"
    df["ts"] = pd.to_datetime(raw[cols["ts"]], errors="coerce") if cols["ts"] is not None else pd.NaT
    df["src_row"] = raw.index

    labels = mapping.get("success_labels")
    if not labels:
        uniq = df["disposition"].unique()
        labels = [u for u in uniq if any(re.search(p, u.lower()) for p in _SUCCESS_PATTERNS)]
    if not labels:
        raise ValueError(f"No success disposition found. Dispositions: {sorted(df['disposition'].unique())[:30]}. "
                         f"Set success_labels in data/mapping.yaml.")
    df["success"] = df["disposition"].isin(labels)
    df["connected"] = ~df["disposition"].isin(mapping.get("not_connected_labels", []))
    df = df.dropna(subset=["duration_s"])
    df = df[df["duration_s"] > 0].reset_index(drop=True)
    return Dataset(df[["caller_id", "disposition", "success", "connected", "duration_s", "ts", "src_row"]], str(path), False,
                   cols, list(labels))


def _load_vani(raw: pd.DataFrame, path: Path) -> Dataset:
    """Buyer-side VANI call log. There is no talk-time column: duration is the gap between
    entered_on and last_modified_date, which has minute resolution (0 = under a minute). On
    calls that never connected the gap is record-update lag (median ~35 min), not talk time,
    so duration is only kept for connected calls."""
    fmt = "%d/%m/%y %H:%M"
    start = pd.to_datetime(raw["entered_on"], format=fmt, errors="coerce")
    end = pd.to_datetime(raw["last_modified_date"], format=fmt, errors="coerce")
    disp = raw["disposition_remarks"].fillna("Unknown").astype(str).str.strip()
    connected = ~disp.isin(VANI_NOT_CONNECTED)
    dur = (end - start).dt.total_seconds()
    df = pd.DataFrame({
        "caller_id": raw["fk_glusr_usr_id"].map(lambda v: "" if pd.isna(v) else str(int(v))),
        "disposition": disp,
        "success": disp.isin(VANI_SUCCESS),
        "connected": connected,
        "duration_s": dur.where(connected),
        "ts": start,
        "src_row": raw.index,           # row in the source file (0-based, header excluded)
    })
    df = df[df["caller_id"] != ""].reset_index(drop=True)
    cols = {"caller_id": "fk_glusr_usr_id", "disposition": "disposition_remarks",
            "duration_s": "last_modified_date - entered_on (connected calls only)", "ts": "entered_on"}
    notes = [f"Small bootstrap pool: {len(df)} historical calls, {df['caller_id'].nunique()} callers.",
             "Duration is a minute-resolution timestamp proxy, not measured talk time."]
    return Dataset(df, str(path), False, cols, list(VANI_SUCCESS), notes)


def _synth_caller_ids(n: int, rng: np.random.Generator, repeat_share: float = 0.25) -> np.ndarray:
    """Caller IDs where ~repeat_share of callers make 2+ calls."""
    ids = []
    k = 0
    while len(ids) < n:
        k += 1
        calls = 1 if rng.random() > repeat_share else 1 + rng.geometric(0.55)
        ids.extend([f"C{k:06d}"] * calls)
    return np.array(ids[:n])


def synthetic(n: int = 20000, seed: int = 7) -> Dataset:
    """Stand-in dataset used only until the real data is dropped into data/.
    Shape loosely mimics an outbound sales voice bot."""
    rng = np.random.default_rng(seed)
    dispositions = {
        # name: (probability, lognormal median seconds, sigma)
        "Meeting Fixed": (0.10, 210, 0.45),
        "Callback Requested": (0.16, 120, 0.55),
        "Not Interested": (0.30, 55, 0.6),
        "No Answer": (0.22, 8, 0.5),
        "Wrong Number": (0.07, 20, 0.5),
        "Already Purchased": (0.08, 70, 0.55),
        "Language Barrier": (0.07, 40, 0.6),
    }
    names = list(dispositions)
    probs = np.array([dispositions[d][0] for d in names])
    disp = rng.choice(names, size=n, p=probs / probs.sum())
    med = np.array([dispositions[d][1] for d in disp])
    sig = np.array([dispositions[d][2] for d in disp])
    dur = np.round(np.exp(np.log(med) + sig * rng.standard_normal(n)), 1)
    ts = pd.Timestamp("2026-08-01 09:00") + pd.to_timedelta(np.sort(rng.uniform(0, 30 * 24 * 3600, n)), unit="s")
    df = pd.DataFrame({"caller_id": _synth_caller_ids(n, rng), "disposition": disp,
                       "success": disp == "Meeting Fixed", "connected": disp != "No Answer",
                       "duration_s": dur, "ts": ts, "src_row": np.arange(n)})
    return Dataset(df, "SYNTHETIC", True, {"note": "generated"}, ["Meeting Fixed"], ["SYNTHETIC data"])


if __name__ == "__main__":
    import json
    print(json.dumps(load().card(), indent=2, default=str))
