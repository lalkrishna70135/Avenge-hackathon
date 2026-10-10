"""Headless dashboard checks: every tab renders, row drill-downs work (including stale selections),
and a playground run completes. Uses a temporary database, never runs/ab.db."""
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from abtest import convo, realcalls, store
from abtest.loader import ROOT

APP = str(ROOT / "app" / "dashboard.py")


@pytest.fixture
def app(db_path, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", db_path)
    for mod, attr in ((realcalls, "REAL_CALLS"), (realcalls, "TRANSCRIPT_DIR"), (convo, "CONVERSATIONS"),
                      (convo, "PILOT_DIR"), (convo, "CORE")):
        monkeypatch.setattr(mod, attr, tmp_path / "absent" / attr)     # prompt-lab tabs: no pipeline outputs
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    return at


def select_rows(monkeypatch, rows):
    """AppTest cannot click dataframe rows, so make selectable tables report a selection."""
    real = st.dataframe

    def fake(data=None, *a, key=None, on_select="ignore", **kw):
        real(data, *a, key=key, **kw)
        base = (key or "").split(":")[0]
        return SimpleNamespace(selection=SimpleNamespace(rows=[rows[base]] if base in rows else []))
    monkeypatch.setattr(st, "dataframe", fake)


def test_all_tabs_render(app):
    assert [t.label for t in app.tabs] == ["Call log", "Real calls", "Prompt lab", "Live call", "Experiments",
                                           "Simulated calls", "Playground", "Split evidence", "Error rates", "Method"]
    assert app.metric[0].value == "1,000" and app.metric[4].value == "517"   # rows in file, leads


@pytest.mark.parametrize("exp_id,decision", [("exp-win", "PROMOTE"), ("exp-harm", "STOP_HARM"),
                                             ("exp-null", "INCONCLUSIVE")])
def test_experiment_and_replay(app, exp_id, decision):
    app.selectbox[0].set_value(exp_id)
    app.selectbox(key="sim_exp").set_value(exp_id)
    for hour in (0.0, 30.0):
        app.slider(key="sim_hour").set_value(hour).run()
        assert not app.exception
    assert next(m for m in app.metric if m.label == "Decision").value == decision   # Experiments headline


def test_drill_downs_including_stale_selection(app, monkeypatch):
    select_rows(monkeypatch, {"log_table": 1, "sim_table": 0})
    app.toggle(key="sim_fx").set_value(True).run()
    assert not app.exception
    text = [m.value for m in app.markdown]
    assert any(t.startswith("**Source row 1**") for t in text)
    assert any(t.startswith("**B's injected effect replaced it with row") for t in text)

    select_rows(monkeypatch, {"log_table": 900, "sim_table": 20000})   # beyond the filtered view
    app.multiselect(key="log_cat").set_value(["Dropped: no caller id"])
    app.selectbox(key="sim_exp").set_value("exp-harm").run()
    assert not app.exception


def test_route_a_caller(app):
    app.text_input(key="route_caller").set_value("52549412").run()
    assert not app.exception


def test_playground_run(app):
    play = [s for s in app.slider if s.key is None or not s.key.startswith("sim")]
    play[2].set_value(-10.0)                                   # true effect on B (pp)
    app.button[0].click().run()
    assert not app.exception
    assert any("engine says **STOP_HARM**" in m.value for m in app.markdown)


def test_hold_approval_is_logged(db_path, monkeypatch, tmp_path):
    """A held win shows the approval panel; approving needs a name and appends a decision by that approver."""
    import shutil
    db = tmp_path / "ab.db"
    shutil.copy(db_path, db)                       # this test writes a decision: keep the shared database clean
    monkeypatch.setattr(store, "DB_PATH", db)
    for mod, attr in ((realcalls, "REAL_CALLS"), (realcalls, "TRANSCRIPT_DIR"), (convo, "CONVERSATIONS"),
                      (convo, "PILOT_DIR"), (convo, "CORE")):
        monkeypatch.setattr(mod, attr, tmp_path / "absent" / attr)
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.selectbox[0].set_value("exp-hold").run()
    assert next(m for m in at.metric if m.label == "Decision").value == "HOLD_FOR_APPROVAL"
    assert at.button(key="approve:exp-hold").disabled                 # no approver name yet
    at.text_input(key="approver:exp-hold").set_value("Priya, category lead")
    at.text_input(key="approve_note:exp-hold").set_value("worth the longer calls").run()
    at.button(key="approve:exp-hold").click().run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Decision").value == "PROMOTE"
    log = store.read(store.connect(db), "SELECT actor, rule_fired FROM decisions WHERE experiment_id = 'exp-hold'")
    assert log.iloc[-1].tolist() == ["approver: Priya, category lead", "R7_APPROVED"]
