"""SQLite store. Decisions and routing changes are append-only."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "runs" / "ab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  experiment_id TEXT, version INTEGER, config_hash TEXT, title TEXT, status TEXT,
  share_b REAL, config_json TEXT, design_json TEXT, data_card_json TEXT, created_at TEXT,
  PRIMARY KEY (experiment_id, version)
);
CREATE TABLE IF NOT EXISTS routing_config (
  experiment_id TEXT, routing_version INTEGER, share_b REAL, reason TEXT, decision_id INTEGER,
  virtual_hour REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS assignments (
  experiment_id TEXT, caller_id TEXT, variant TEXT, bucket INTEGER, assigned_at_hour REAL,
  PRIMARY KEY (experiment_id, caller_id)
);
CREATE TABLE IF NOT EXISTS calls (
  experiment_id TEXT, config_hash TEXT, call_id INTEGER, caller_id TEXT, variant TEXT, phase TEXT,
  disposition TEXT, success INTEGER, duration_s REAL, t_hours REAL,
  drawn_src_row INTEGER, used_src_row INTEGER, effect_applied INTEGER
);
CREATE TABLE IF NOT EXISTS looks (
  experiment_id TEXT, look INTEGER, virtual_hour REAL, evidence_json TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, config_hash TEXT,
  virtual_hour REAL, look INTEGER, decision TEXT, rule_fired TEXT, rule_text TEXT,
  evidence_json TEXT, actor TEXT, narration TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_calls_exp ON calls(experiment_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or DB_PATH)          # resolved at call time so tests can point it elsewhere
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.executescript(SCHEMA)
    return con


def reset_experiment(con, experiment_id: str) -> None:
    for t in ("experiments", "routing_config", "assignments", "calls", "looks", "decisions"):
        con.execute(f"DELETE FROM {t} WHERE experiment_id = ?", (experiment_id,))
    con.commit()


def save_experiment(con, cfg, status: str, design: dict, data_card: dict) -> None:
    con.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cfg.experiment_id, cfg.version, cfg.config_hash, cfg.title, status, cfg.share_b,
                 cfg.canonical_json(), json.dumps(design, default=float), json.dumps(data_card, default=str), now()))
    con.commit()


def set_status(con, experiment_id: str, status: str) -> None:
    con.execute("UPDATE experiments SET status = ? WHERE experiment_id = ?", (status, experiment_id))
    con.commit()


def add_routing(con, experiment_id: str, share_b: float, reason: str, decision_id, hour: float) -> None:
    v = con.execute("SELECT COALESCE(MAX(routing_version), 0) FROM routing_config WHERE experiment_id = ?",
                    (experiment_id,)).fetchone()[0]
    con.execute("INSERT INTO routing_config VALUES (?,?,?,?,?,?,?)",
                (experiment_id, v + 1, share_b, reason, decision_id, hour, now()))
    con.commit()


def add_decision(con, experiment_id, config_hash, hour, look, decision, rule, rule_text, evidence,
                 actor="auto", narration=None) -> int:
    cur = con.execute(
        "INSERT INTO decisions (experiment_id, config_hash, virtual_hour, look, decision, rule_fired, rule_text,"
        " evidence_json, actor, narration, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (experiment_id, config_hash, hour, look, decision, rule, rule_text, json.dumps(evidence, default=float),
         actor, narration, now()))
    con.commit()
    return cur.lastrowid


def add_look(con, experiment_id, look, hour, evidence) -> None:
    con.execute("INSERT INTO looks VALUES (?,?,?,?)", (experiment_id, look, hour, json.dumps(evidence, default=float)))
    con.commit()


def write_frame(con, table: str, df: pd.DataFrame) -> None:
    df.to_sql(table, con, if_exists="append", index=False, chunksize=5000)


def read(con, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, con, params=params)
