"""Every stored simulated call must trace back correctly to the real call-log row it was drawn from."""
import json

import numpy as np
import pandas as pd
import pytest

from abtest import store
from abtest.assigner import BUCKETS, buckets
from abtest.config import load_scenario
from abtest.loader import ROOT
from abtest.runner import run_experiment

FMT = "%d/%m/%y %H:%M"


@pytest.fixture(scope="module")
def db(ds, tmp_path_factory):
    con = store.connect(tmp_path_factory.mktemp("db") / "ab.db")
    for name in ("win", "harm"):
        run_experiment(load_scenario(ROOT / "scenarios" / f"{name}.yaml"), ds, con=con)
    return con


@pytest.mark.parametrize("exp_id", ["exp-win", "exp-harm"])
def test_calls_trace_back_to_source_rows(ds, db, exp_id):
    raw = pd.read_csv(ds.source)
    gap = (pd.to_datetime(raw["last_modified_date"], format=FMT)
           - pd.to_datetime(raw["entered_on"], format=FMT)).dt.total_seconds()
    disp = raw["disposition_remarks"].fillna("Unknown").astype(str).str.strip()
    cfg = json.loads(db.execute("SELECT config_json FROM experiments WHERE experiment_id = ?", (exp_id,)).fetchone()[0])
    c = store.read(db, "SELECT * FROM calls WHERE experiment_id = ?", (exp_id,))

    assert (disp.loc[c["used_src_row"]].to_numpy() == c["disposition"].to_numpy()).all()
    fx = c["effect_applied"] == 1
    assert fx.any() and not ((c["used_src_row"] != c["drawn_src_row"]) & ~fx).any()
    assert (c.loc[c["variant"] == "A", "effect_applied"] == 0).all()

    mult = np.where(c["variant"] == "B", cfg["true_effect"]["duration_mult"], 1.0)
    has = c["duration_s"].notna().to_numpy()
    assert np.allclose(c["duration_s"].to_numpy()[has], np.round(gap.loc[c["used_src_row"]].to_numpy() * mult, 1)[has])

    e = c[c["phase"] == "experiment"]
    hashed = np.where(buckets(cfg["salt"], e["caller_id"].tolist()) < round(cfg["share_b"] * BUCKETS), "B", "A")
    assert (hashed == e["variant"].to_numpy()).all()
    final = db.execute("SELECT share_b FROM routing_config WHERE experiment_id = ? "
                       "ORDER BY routing_version DESC LIMIT 1", (exp_id,)).fetchone()[0]
    assert (c.loc[c["phase"] == "rollout", "variant"] == ("B" if final >= 1 else "A")).all()
