"""Experiment dashboard. Reads runs/ab.db (written by scripts/run_scenarios.py) and the raw call log.

Run: .venv/bin/streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from abtest import store  # noqa: E402
from app import lab_tabs  # noqa: E402
from abtest.assigner import BUCKETS, bucket  # noqa: E402
from abtest.config import load_scenario  # noqa: E402
from abtest.loader import VANI_NOT_CONNECTED, VANI_SUCCESS, load  # noqa: E402
from abtest.runner import baseline_per_caller, plan_design, resolve_hold, run_experiment  # noqa: E402

st.set_page_config(page_title="VANI prompt experiments", layout="wide")

STATUS_COLOUR = {"promoted": "green", "stopped": "red", "inconclusive": "gray", "running": "blue",
                 "awaiting_approval": "orange", "invalid": "red", "rejected": "gray"}
CATEGORY_COLOUR = {"Lead created": "#2E8B57", "Connected, no lead": "#E07A1F", "Not connected": "#8A94A6",
                   "Dropped: no caller id": "#C0392B"}
SCENARIOS = {p.stem: p for p in sorted((ROOT / "scenarios").glob("*.yaml"))}


@st.cache_resource
def con():
    return store.connect()


@st.cache_resource
def dataset():
    return load()


@st.cache_data
def raw_log() -> pd.DataFrame:
    """The source CSV with the loader's processing shown next to it, one row per source row."""
    ds = dataset()
    raw = pd.read_csv(ds.source)
    proc = ds.calls.set_index("src_row")
    out = pd.DataFrame({"src_row": raw.index})
    out["kept"] = out["src_row"].isin(proc.index)
    out["caller_id"] = proc["caller_id"].reindex(raw.index).fillna("").to_numpy()
    out["disposition"] = raw["disposition_remarks"].fillna("Unknown").astype(str).str.strip()
    out["connected"] = proc["connected"].reindex(raw.index).to_numpy()
    out["lead_created"] = proc["success"].reindex(raw.index).to_numpy()
    out["duration_s"] = proc["duration_s"].reindex(raw.index).to_numpy()
    out["category"] = np.select(
        [~out["kept"], out["lead_created"] == True, out["connected"] == True],  # noqa: E712 (object dtype)
        ["Dropped: no caller id", "Lead created", "Connected, no lead"], "Not connected")
    out["entered_on"] = raw["entered_on"]
    out["last_modified_date"] = raw["last_modified_date"]
    out["ts"] = pd.to_datetime(raw["entered_on"], format="%d/%m/%y %H:%M", errors="coerce")
    return out, raw


@st.cache_resource
def baseline(scenario: str, goal: tuple | None = None) -> float:
    """Per-caller baseline goal rate for a scenario's arrival process (the slow part of plan_design)."""
    return baseline_per_caller(dataset(), load_scenario(SCENARIOS[scenario], primary_goal=goal))


def goal_name(cfg: dict) -> str:
    return " / ".join(cfg["primary_goal"]) if cfg.get("primary_goal") else "Lead created"


def q(sql, params=()):
    return store.read(con(), sql, params)


def read_json(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def why(r) -> str:
    """Plain-language explanation of how the loader treated one source row."""
    if not r["kept"]:
        return "Dropped: `fk_glusr_usr_id` is empty, so the call can't be tied to a caller."
    parts = []
    if r["lead_created"]:
        parts.append(f"**Lead created**: `{r['disposition']}` is one of {', '.join(VANI_SUCCESS)}.")
    else:
        parts.append(f"**No lead**: `{r['disposition']}` is not a lead-created disposition.")
    if r["connected"]:
        parts.append(f"**Connected**, so B's injected effect can apply to it. Duration = last_modified − entered_on "
                     f"= {r['last_modified_date']} − {r['entered_on']} = **{r['duration_s']:.0f} s**.")
    else:
        parts.append(f"**Not connected** (`{r['disposition']}` is in the no-conversation list), so no prompt can "
                     "change it and it has no duration.")
    return "  \n".join(parts)


# ---------------------------------------------------------------------------- charts (shared)
def z_chart(design, lk, dec_hour, decision):
    fig = go.Figure()
    planned_h = design["look_hours"]
    fig.add_scatter(x=planned_h, y=design["planned_eff_bounds"], name="efficacy boundary (OBF, planned)",
                    line=dict(color="green", dash="dot"))
    fig.add_scatter(x=planned_h, y=design["planned_harm_bounds"], name="harm boundary (Pocock, planned)",
                    line=dict(color="red", dash="dot"))
    fig.add_scatter(x=lk["virtual_hour"], y=lk["eff_bound"], mode="markers", name="efficacy boundary at look",
                    marker=dict(color="green", symbol="line-ew-open", size=14))
    fig.add_scatter(x=lk["virtual_hour"], y=lk["harm_bound"], mode="markers", name="harm boundary at look",
                    marker=dict(color="red", symbol="line-ew-open", size=14))
    fig.add_scatter(x=lk["virtual_hour"], y=lk["z"], mode="lines+markers", name="observed Z (B − A)",
                    line=dict(color="black", width=3))
    fig.add_vline(x=dec_hour, line_dash="dash", annotation_text=decision)
    fig.update_layout(title="Test statistic vs group-sequential boundaries", xaxis_title="virtual hour",
                      yaxis_title="Z", height=420, legend=dict(orientation="h", y=-0.25))
    return fig


def lift_chart(cfg, lk, dec_hour):
    hours = lk["virtual_hour"]
    fig = go.Figure()
    fig.add_scatter(x=hours, y=100 * lk["rci_high"], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=hours, y=100 * lk["rci_low"], fill="tonexty", line=dict(width=0),
                    name="repeated CI (valid under peeking)", fillcolor="rgba(100,100,255,0.15)")
    fig.add_scatter(x=hours, y=100 * lk["lift_abs"], mode="lines+markers", name="lift (pp)",
                    error_y=dict(type="data", symmetric=False, array=100 * (lk["ci95_high"] - lk["lift_abs"]),
                                 arrayminus=100 * (lk["lift_abs"] - lk["ci95_low"])),
                    line=dict(color="black"))
    fig.add_hline(y=0, line_color="gray")
    fig.add_hline(y=100 * cfg["mde_pp"], line_dash="dot", annotation_text="MDE")
    fig.add_vline(x=dec_hour, line_dash="dash")
    fig.update_layout(title=f"Lift in {goal_name(cfg)} rate, ± 95% CI", xaxis_title="virtual hour",
                      yaxis_title="B − A (pp)", height=420, legend=dict(orientation="h", y=-0.25))
    return fig


def duration_chart(cfg, looks):
    dur = pd.DataFrame([e["duration"] for e in looks])
    hours = [e["virtual_hour"] for e in looks]
    fig = go.Figure()
    fig.add_scatter(x=hours, y=dur["ratio"], mode="lines+markers", name="mean duration B/A",
                    error_y=dict(type="data", symmetric=False, array=dur["ci_high"] - dur["ratio"],
                                 arrayminus=dur["ratio"] - dur["ci_low"]))
    fig.add_hline(y=1 + cfg["ni_margin"], line_dash="dot", line_color="orange",
                  annotation_text=f"non-inferiority margin +{cfg['ni_margin']:.0%}")
    fig.add_hline(y=1 + cfg["hard_margin"], line_dash="dot", line_color="red",
                  annotation_text=f"hard stop +{cfg['hard_margin']:.0%}")
    fig.add_hline(y=1, line_color="gray")
    fig.update_layout(title="Secondary guardrail: call duration (connected calls)", xaxis_title="virtual hour",
                      yaxis_title="ratio", height=360)
    return fig


def routing_chart(steps, horizon):
    xs = list(steps["virtual_hour"]) + [horizon]
    ys = list(steps["share_b"])
    ys.append(ys[-1])
    fig = go.Figure()
    fig.add_scatter(x=xs, y=[100 * y for y in ys], line_shape="hv", name="B share", line=dict(width=3))
    fig.update_layout(title="Rollout controller: live share of traffic on B", xaxis_title="virtual hour",
                      yaxis_title="% of callers on B", yaxis_range=[-5, 105], height=360)
    return fig


def look_frame(looks):
    return pd.DataFrame([{k: v for k, v in e.items() if not isinstance(v, (dict, list))} for e in looks])


def headline(cfg, design, last, decision=None, rule=None):
    c = st.columns(5)
    c[0].metric("Decision", decision or last["decision"], rule or last["rule"], delta_color="off")
    c[1].metric("B share: configured → achieved", f"{cfg['share_b']:.0%} → {last['achieved_share_b']:.2%}",
                f"SRM p = {last['srm_p']:.3f}" + (" ⚠ SRM" if last["srm"] else " · OK"), delta_color="off")
    c[2].metric("Callers A / B", f"{last['n_a']:,} / {last['n_b']:,}",
                f"{(last['n_a'] + last['n_b']) / design['sample_size']['n_total']:.0%} of planned "
                f"{design['sample_size']['n_total']:,}", delta_color="off")
    c[3].metric(f"{goal_name(cfg)} rate A → B", f"{last['p_a']:.1%} → {last['p_b']:.1%}",
                f"{100 * last['lift_abs']:+.2f} pp [{100 * last['ci95_low']:+.2f}, {100 * last['ci95_high']:+.2f}]",
                delta_color="off")
    d = last["duration"]
    c[4].metric("Duration B/A (guardrail)", f"{d['ratio']:.3f}",
                f"95% CI [{d['ci_low']:.3f}, {d['ci_high']:.3f}] vs ≤ {1 + cfg['ni_margin']:.2f}", delta_color="off")


exps = q("SELECT * FROM experiments ORDER BY experiment_id")
if exps.empty:
    st.error("No experiments in runs/ab.db. Run `.venv/bin/python scripts/run_scenarios.py` first.")
    st.stop()
exps_by_id = exps.set_index("experiment_id")

st.title("VANI prompt A/B testing and auto-rollout")
(tab_log, tab_real, tab_lab, tab_live, tab_exp, tab_sim, tab_play, tab_split, tab_oc, tab_method) = st.tabs(
    ["Call log", "Real calls", "Prompt lab", "Live call", "Experiments", "Simulated calls", "Playground",
     "Split evidence", "Error rates", "Method"])

# ---------------------------------------------------------------------------- call log (raw -> processed)
with tab_log:
    log, raw = raw_log()
    kept = log[log["kept"]]
    st.caption(f"Source: `{Path(dataset().source).relative_to(ROOT)}`, the real VANI buyer-side call log. "
               "This is the pool every simulated call is drawn from.")
    c = st.columns(5)
    c[0].metric("Rows in file", f"{len(log):,}")
    c[1].metric("Kept (have a caller id)", f"{len(kept):,}", f"{len(log) - len(kept)} dropped", delta_color="off")
    c[2].metric("Distinct callers", f"{kept['caller_id'].nunique():,}",
                f"{(kept.groupby('caller_id').size() > 1).sum()} called more than once", delta_color="off")
    c[3].metric("Connected", f"{int((kept['connected'] == True).sum()):,}",  # noqa: E712
                f"{(kept['connected'] == True).mean():.0%} of kept", delta_color="off")  # noqa: E712
    c[4].metric("Lead created", f"{int((kept['lead_created'] == True).sum()):,}",  # noqa: E712
                f"{(kept['lead_created'] == True).mean():.1%} per call", delta_color="off")  # noqa: E712

    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Funnel(
            y=["Rows in file", "Kept", "Connected", "Lead created"],
            x=[len(log), len(kept), int((kept["connected"] == True).sum()),  # noqa: E712
               int((kept["lead_created"] == True).sum())],  # noqa: E712
            textinfo="value+percent initial", marker=dict(color=["#14213D", "#4A5568", "#E07A1F", "#2E8B57"])))
        fig.update_layout(title="What the loader keeps", height=380, margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        counts = log.groupby(["disposition", "category"]).size().reset_index(name="calls")
        order = log["disposition"].value_counts().index.tolist()
        fig = go.Figure()
        for cat, colour in CATEGORY_COLOUR.items():
            part = counts[counts["category"] == cat]
            if len(part):
                fig.add_bar(y=part["disposition"], x=part["calls"], orientation="h", name=cat, marker_color=colour)
        fig.update_layout(title="Dispositions and how each is classified", barmode="stack", height=380,
                          yaxis=dict(categoryorder="array", categoryarray=order[::-1]), xaxis_title="calls",
                          legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        by_time = log.dropna(subset=["ts"]).assign(hour=lambda d: d["ts"].dt.floor("h"))
        by_time = by_time.groupby(["hour", "category"]).size().reset_index(name="calls")
        fig = go.Figure()
        for cat, colour in CATEGORY_COLOUR.items():
            part = by_time[by_time["category"] == cat]
            if len(part):
                fig.add_bar(x=part["hour"], y=part["calls"], name=cat, marker_color=colour)
        fig.update_layout(title="Calls over time (entered_on, hourly)", barmode="stack", height=340,
                          legend=dict(orientation="h", y=-0.3), margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        conn = kept[kept["connected"] == True]  # noqa: E712
        fig = go.Figure()
        for lead, name, colour in ((True, "Lead created", "#2E8B57"), (False, "Connected, no lead", "#E07A1F")):
            fig.add_histogram(x=conn.loc[conn["lead_created"] == lead, "duration_s"] / 60, name=name,
                              marker_color=colour, xbins=dict(size=1), opacity=0.8)
        fig.update_layout(title="Duration of connected calls (minutes, timestamp proxy)", barmode="overlay",
                          height=340, xaxis_title="minutes", yaxis_title="calls", xaxis_range=[-0.5, 15],
                          legend=dict(orientation="h", y=-0.3), margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Browse the log")
    f = st.columns([2, 2, 1, 1])
    cats = f[0].multiselect("Category", list(CATEGORY_COLOUR), default=list(CATEGORY_COLOUR), key="log_cat")
    disps = f[1].multiselect("Disposition", order, key="log_disp", placeholder="All dispositions")
    caller = f[2].text_input("Caller id contains", key="log_caller")
    show_raw = f[3].toggle("All raw columns", key="log_raw")
    view = log[log["category"].isin(cats)]
    if disps:
        view = view[view["disposition"].isin(disps)]
    if caller:
        view = view[view["caller_id"].str.contains(caller.strip(), regex=False)]
    cols = ["src_row", "entered_on", "last_modified_date", "caller_id", "disposition", "category", "connected",
            "lead_created", "duration_s"]
    shown = view[cols].join(raw.drop(columns=["disposition_remarks", "entered_on", "last_modified_date"])) \
        if show_raw else view[cols]
    st.caption(f"{len(view):,} of {len(log):,} rows. Click a row to see how it was processed.")
    ev = st.dataframe(shown, hide_index=True, width="stretch", height=320, on_select="rerun",
                      selection_mode="single-row",
                      key=f"log_table:{cats}:{disps}:{caller}:{show_raw}")   # new view -> fresh selection
    if ev.selection.rows and ev.selection.rows[0] < len(view):
        r = view.iloc[ev.selection.rows[0]]
        a, b = st.columns([1, 1])
        a.markdown(f"**Source row {r['src_row']}** (CSV line {r['src_row'] + 2})")
        a.json({k: (None if pd.isna(v) else v) for k, v in raw.loc[r["src_row"]].items()}, expanded=True)
        b.markdown("**How it was processed**")
        b.markdown(why(r))
        if r["kept"]:
            used = q("SELECT experiment_id, SUM(drawn_src_row = :r) AS drawn, "
                     "SUM(drawn_src_row = :r AND effect_applied = 0) AS kept_as_drawn, "
                     "SUM(used_src_row = :r AND effect_applied = 1) AS used_as_b_replacement "
                     "FROM calls WHERE drawn_src_row = :r OR used_src_row = :r GROUP BY experiment_id",
                     {"r": int(r["src_row"])})
            b.markdown("**Times the simulator drew this row as a call outcome**")
            b.dataframe(used, hide_index=True)

    with st.expander("Classification rules"):
        rules = log[log["kept"]].groupby("disposition").agg(calls=("src_row", "size"),
                                                           connected=("connected", "first"),
                                                           lead_created=("lead_created", "first"))
        st.dataframe(rules.sort_values("calls", ascending=False), width="stretch")
        st.markdown(f"- Lead created = {', '.join(VANI_SUCCESS)}\n"
                    f"- Not connected = {', '.join(VANI_NOT_CONNECTED)} (blank disposition becomes Unknown)\n"
                    "- `is_call_answered` is ignored: it is 0 on 38 calls that created a lead.\n"
                    "- Duration is kept only for connected calls; on failed calls the timestamp gap is "
                    "record-update lag, not talk time.")
    with st.expander("Data card (JSON)"):
        st.json(json.loads(exps["data_card_json"].iloc[0]))

# ---------------------------------------------------------------------------- Sarvam prompt lab
with tab_real:
    lab_tabs.real_calls_tab(log.loc[log["kept"], "disposition"])
with tab_lab:
    lab_tabs.prompt_lab_tab()
with tab_live:
    lab_tabs.live_call_tab(exps_by_id)

# ---------------------------------------------------------------------------- experiments
with tab_exp:
    exp_id = st.selectbox("Experiment", exps["experiment_id"],
                          format_func=lambda e: f"{e}: {exps_by_id.loc[e, 'title']}")
    row = exps_by_id.loc[exp_id]
    cfg, design = json.loads(row["config_json"]), json.loads(row["design_json"])
    looks = [json.loads(e) for e in q("SELECT evidence_json FROM looks WHERE experiment_id = ? ORDER BY look",
                                      (exp_id,))["evidence_json"]]
    decisions = q("SELECT * FROM decisions WHERE experiment_id = ? ORDER BY decision_id", (exp_id,))
    routing = q("SELECT * FROM routing_config WHERE experiment_id = ? ORDER BY routing_version", (exp_id,))
    last = looks[-1]
    status = row["status"]

    st.markdown(f"### :{STATUS_COLOUR.get(status, 'blue')}[{status.upper()}] {row['title']}")
    st.caption(f"config v{row['version']} · hash `{row['config_hash']}` · salt `{cfg['salt']}` · "
               f"base prompt sha `{cfg.get('base_prompt_sha256')}` · variant B sha `{cfg['variant_b'].get('prompt_sha256')}`")
    headline(cfg, design, last, decisions["decision"].iloc[-1], decisions["rule_fired"].iloc[-1])
    if cfg.get("pilot_of") and cfg.get("pilot_evidence"):
        pe = cfg["pilot_evidence"]
        st.info(f"**True effect from the simulated pilot of {cfg['pilot_of']}**: {pe['pairs']} persona pairs, simulated "
                f"lead rate {pe['sim_lead_rate_a']:.0%} → {pe['sim_lead_rate_b']:.0%} (lift {100 * pe['lift_pp']:+.1f} pp, "
                f"95% CI [{100 * pe['lift_ci'][0]:+.1f}, {100 * pe['lift_ci'][1]:+.1f}]). Carried over as an odds ratio "
                f"({pe['odds_ratio']:.2f}) to the real {pe['p_connected_real']:.0%} connected lead rate: "
                f"{100 * pe['success_pp']:+.1f} pp, duration ×{pe['duration_mult']}. This asks: if the pilot is right, "
                "does the engine find it at scale?")
    if status == "awaiting_approval":
        d = last["duration"]
        with st.container(border=True):
            st.markdown(
                f"**Approval needed.** {goal_name(cfg)} crossed the win line (Z = {last['z']:+.2f} ≥ "
                f"{last['eff_bound']:.2f}), but the duration guardrail is "
                f"{'violated' if last['dur_ni_violated'] else 'not shown'}: B/A = {d['ratio']:.3f}, 95% CI "
                f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}] against ≤ {1 + cfg['ni_margin']:.2f}. The primary decides "
                "and the guardrail can only veto, so B is held at 0% until a named approver decides. "
                "Their decision is appended to the log with their name and reason.")
            a = st.columns([2, 3, 1, 1], vertical_alignment="bottom")
            who = a[0].text_input("Approver", key=f"approver:{exp_id}", placeholder="name and role")
            note = a[1].text_input("Reason", key=f"approve_note:{exp_id}")
            for col, approve, label in ((a[2], True, "Approve: promote B"), (a[3], False, "Reject: keep A")):
                if col.button(label, key=f"{'approve' if approve else 'reject'}:{exp_id}",
                              type="primary" if approve else "secondary", disabled=not who.strip()):
                    resolve_hold(con(), exp_id, approve, who, note)
                    st.rerun()

    lk = look_frame(looks)
    dec_hour = float(decisions["virtual_hour"].iloc[-1])
    left, right = st.columns(2)
    left.plotly_chart(z_chart(design, lk, dec_hour, last["decision"]), width="stretch")
    right.plotly_chart(lift_chart(cfg, lk, dec_hour), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(duration_chart(cfg, looks), width="stretch")
    right.plotly_chart(routing_chart(routing, dec_hour + cfg["post_rollout_hours"]), width="stretch")

    st.subheader("Decision log (append-only)")
    dlog = decisions[["decision_id", "virtual_hour", "look", "decision", "rule_fired", "rule_text", "actor",
                      "narration", "config_hash", "created_at"]].rename(columns={"narration": "note"})
    evs = [json.loads(e) for e in decisions["evidence_json"]]
    dlog.insert(5, "evidence", [f"n={e['n_a']}/{e['n_b']}  p={e['p_a']:.3f}/{e['p_b']:.3f}  "
                                f"Z={e['z']:+.2f} (win ≥ {e['eff_bound']:.2f}, harm ≤ {e['harm_bound']:.2f})  "
                                f"dur {e['duration']['ratio']:.3f} [{e['duration']['ci_low']:.3f}, "
                                f"{e['duration']['ci_high']:.3f}]" for e in evs])
    st.dataframe(dlog, hide_index=True, width="stretch")
    with st.expander("Routing config history"):
        st.dataframe(routing, hide_index=True, width="stretch")
    with st.expander("Design (fixed before the experiment started)"):
        ss, g = design["sample_size"], design["guardrail"]
        st.markdown(
            f"- Baseline lead-created rate: **{design['baseline_rate_per_caller']:.1%} per caller** "
            f"({design['baseline_rate_per_call']:.1%} per call)\n"
            f"- MDE {cfg['mde_pp']:+.0%} absolute, α = {cfg['alpha']} two-sided, power {cfg['power']:.0%}, "
            f"{cfg['n_looks']} looks, group-sequential inflation ×{ss['gs_inflation']:.3f}\n"
            f"- Planned sample: **{ss['n_total']:,} callers** ({ss['n_b']:,} in B at {cfg['share_b']:.0%}); "
            f"a 50/50 split would need {ss['per_arm_equal_split']:,} per arm\n"
            f"- Window: {design['window_hours']} virtual hours at {design['callers_per_hour']:.0f} new callers/hour; "
            f"looks at {', '.join(f'{h:g}' for h in design['look_hours'])} h\n"
            + (f"- Each caller is scored on their calls within {cfg['score_window_hours']:g} h of their first call and "
               "enters a look once that window has closed, so an analysed outcome never changes\n"
               if cfg.get("score_window_hours") else "")
            + f"- Primary goal: **{goal_name(cfg)}** per caller; secondary: call duration (guardrail)\n"
            f"- Duration guardrail: power {g['power_at_final_look']:.0%} at a {cfg['ni_margin']:.0%} margin "
            f"({g['power_if_margin_10pct']:.0%} at 10%); per-caller CV {g['cv_per_caller']:.2f}\n"
            f"- Minimum {cfg['min_sample_per_arm']} callers per arm before any stop")
        if last["decision"] == "INCONCLUSIVE":
            st.info(f"Inconclusive: with the sample collected, the smallest lift detectable at "
                    f"{cfg['power']:.0%} power was {100 * last['detectable_effect_pp']:.1f} pp.")
    with st.expander("Prompt change: A vs B"):
        vb = cfg["variant_b"]
        st.markdown(f"**{vb['name']}**  \nRequested change: _{vb.get('diff')}_")
        if vb.get("prompt_file") and (ROOT / vb["prompt_file"]).exists():
            text = (ROOT / vb["prompt_file"]).read_text()
            header = [ln for ln in text.splitlines() if ln.startswith("# Replaces lines")]
            body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("# ")).strip()
            a_col, b_col = st.columns(2)
            if header and cfg.get("base_prompt_file"):
                lo, hi = (int(x) for x in header[0].split("lines ")[1].split(" of")[0].split("-"))
                base = (ROOT / cfg["base_prompt_file"]).read_text().splitlines()
                a_col.markdown("**A: current prompt section**")
                a_col.code("\n".join(base[lo - 1:hi]), language=None, wrap_lines=True)
            b_col.markdown("**B: rewritten by Sarvam (sarvam-105b)**")
            b_col.code(body, language=None, wrap_lines=True)
        else:
            st.caption("This variant is an instruction diff on the base prompt (no Sarvam rewrite generated). "
                       "Its outcomes are simulated from the injected true effect like every scenario.")

# ---------------------------------------------------------------------------- simulated calls (replay)
with tab_sim:
    sim_id = st.selectbox("Experiment", exps["experiment_id"], key="sim_exp",
                          format_func=lambda e: f"{e}: {exps_by_id.loc[e, 'title']}")
    s_cfg = json.loads(exps_by_id.loc[sim_id, "config_json"])
    s_looks = [json.loads(e) for e in q("SELECT evidence_json FROM looks WHERE experiment_id = ? ORDER BY look",
                                        (sim_id,))["evidence_json"]]
    s_routing = q("SELECT * FROM routing_config WHERE experiment_id = ? ORDER BY routing_version", (sim_id,))
    calls = q("SELECT call_id, t_hours, caller_id, variant, phase, disposition, success, duration_s, "
              "drawn_src_row, used_src_row, effect_applied FROM calls WHERE experiment_id = ? ORDER BY t_hours",
              (sim_id,))
    t_max = float(calls["t_hours"].max())
    hour = st.slider("Replay to virtual hour", 0.0, round(t_max, 1), round(t_max, 1), 0.5, key="sim_hour",
                     help="Everything below shows the experiment as it stood at this hour.")
    now_calls = calls[calls["t_hours"] <= hour]
    exp_now = now_calls[now_calls["phase"] == "experiment"]
    per_caller = exp_now.groupby("caller_id").agg(variant=("variant", "first"), lead=("success", "max"))
    done = [e for e in s_looks if e["virtual_hour"] <= hour]
    share_now = s_routing[s_routing["virtual_hour"] <= hour]["share_b"].iloc[-1]

    c = st.columns(5)
    c[0].metric("Calls so far", f"{len(now_calls):,}",
                f"{int((now_calls['phase'] == 'rollout').sum()):,} after rollout", delta_color="off")
    for i, v in ((1, "A"), (2, "B")):
        pc = per_caller[per_caller["variant"] == v]
        c[i].metric(f"Callers on {v} (experiment)", f"{len(pc):,}",
                    f"lead rate {pc['lead'].mean():.1%}" if len(pc) else "–", delta_color="off")
    c[3].metric("Live B share", f"{share_now:.0%}", "set by the rollout controller", delta_color="off")
    c[4].metric("Last look", f"{done[-1]['look']} → {done[-1]['decision']}" if done else "none yet",
                f"Z = {done[-1]['z']:+.2f}" if done else f"first look at {s_looks[0]['virtual_hour']:g} h",
                delta_color="off")

    grid = np.linspace(0, t_max, 60)
    t_sorted = calls[calls["phase"] == "experiment"]
    first_t = t_sorted.groupby("caller_id").agg(t=("t_hours", "min"), variant=("variant", "first"))
    rates = []
    for g in grid[1:]:
        sub = t_sorted[t_sorted["t_hours"] <= g].groupby("caller_id").agg(v=("variant", "first"), s=("success", "max"))
        rates.append({"t": g, **{v: sub.loc[sub["v"] == v, "s"].mean() for v in ("A", "B")}})
    rates = pd.DataFrame(rates)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for v, colour in (("A", "#14213D"), ("B", "#E07A1F")):
            ft = np.sort(first_t.loc[first_t["variant"] == v, "t"].to_numpy())
            fig.add_scatter(x=ft, y=np.arange(1, len(ft) + 1), name=f"callers on {v}", line=dict(color=colour))
        for e in s_looks:
            fig.add_vline(x=e["virtual_hour"], line_dash="dot", line_color="gray",
                          annotation_text=f"look {e['look']}", annotation_position="top left")
        fig.add_vline(x=hour, line_color="red")
        fig.update_layout(title="Callers enrolled per arm", xaxis_title="virtual hour", yaxis_title="callers",
                          height=340, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        for v, colour in (("A", "#14213D"), ("B", "#E07A1F")):
            fig.add_scatter(x=rates["t"], y=100 * rates[v], name=f"{v}", line=dict(color=colour))
        fig.add_vline(x=hour, line_color="red")
        fig.update_layout(title="Running lead-created rate per caller", xaxis_title="virtual hour",
                          yaxis_title="% of callers with a lead", height=340, legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Call stream")
    f = st.columns([1, 1, 1, 2])
    v_sel = f[0].multiselect("Variant", ["A", "B"], default=["A", "B"], key="sim_var")
    ph_sel = f[1].multiselect("Phase", ["experiment", "rollout"], default=["experiment", "rollout"], key="sim_phase")
    only_fx = f[2].toggle("Only calls B's effect changed", key="sim_fx")
    c_q = f[3].text_input("Caller id contains", key="sim_caller")
    view = now_calls[now_calls["variant"].isin(v_sel) & now_calls["phase"].isin(ph_sel)]
    if only_fx:
        view = view[view["effect_applied"] == 1]
    if c_q:
        view = view[view["caller_id"].str.contains(c_q.strip(), regex=False)]
    view = view.iloc[::-1]   # newest first
    st.caption(f"{len(view):,} calls (newest first). Click one to trace it back to the real log.")
    ev = st.dataframe(view.assign(success=view["success"].astype(bool), effect_applied=view["effect_applied"].astype(bool))
                      .rename(columns={"success": "lead_created", "drawn_src_row": "drawn_row",
                                       "used_src_row": "used_row"}),
                      hide_index=True, width="stretch", height=320, on_select="rerun", selection_mode="single-row",
                      key=f"sim_table:{sim_id}:{hour}:{v_sel}:{ph_sel}:{only_fx}:{c_q}")
    if ev.selection.rows and ev.selection.rows[0] < len(view):
        r = view.iloc[ev.selection.rows[0]]
        _, raw_df = raw_log()
        b_ = bucket(s_cfg["salt"], r["caller_id"])
        threshold = round(s_cfg["share_b"] * BUCKETS)
        st.markdown(
            f"**Call {r['call_id']}** at hour {r['t_hours']:.2f} from caller `{r['caller_id']}`  \n"
            f"Routing: sha256(`{s_cfg['salt']}:{r['caller_id']}`) mod {BUCKETS} = **{b_}**, "
            f"{'<' if b_ < threshold else '≥'} {threshold} → hash says **{'B' if b_ < threshold else 'A'}**"
            + (f"; this call was in the **{r['phase']}** phase, so it got **{r['variant']}**"
               if r["phase"] == "rollout" else f"; it got **{r['variant']}**"))
        a, b = st.columns(2)
        a.markdown(f"**Outcome drawn from real log row {r['drawn_src_row']}**")
        a.json({k: (None if pd.isna(v) else v) for k, v in raw_df.loc[int(r["drawn_src_row"])].items()})
        if r["effect_applied"]:
            b.markdown(f"**B's injected effect replaced it with row {r['used_src_row']}**")
            b.json({k: (None if pd.isna(v) else v) for k, v in raw_df.loc[int(r["used_src_row"])].items()})
        else:
            b.markdown("**No injected effect on this call**: the outcome is the drawn row as-is"
                       + (f", with duration × {s_cfg['true_effect']['duration_mult']}"
                          if r["variant"] == "B" and s_cfg["true_effect"]["duration_mult"] != 1 else "") + ".")
        st.info("Simulated callers reuse real caller ids at random to keep the demo grounded, but each call's "
                "outcome is a bootstrap draw from *any* row of the log. So the source row usually belongs to a "
                "different real caller; that's expected.")
        others = calls[calls["caller_id"] == r["caller_id"]]
        if len(others) > 1:
            st.markdown(f"**All {len(others)} calls from this caller** (always the same variant during the "
                        "experiment: assignment is sticky)")
            st.dataframe(others, hide_index=True, width="stretch")

    st.subheader("Route a caller")
    cid = st.text_input("Caller id", value=str(calls["caller_id"].iloc[0]), key="route_caller")
    if cid:
        rows = []
        for e_id, e in exps_by_id.iterrows():
            ec = json.loads(e["config_json"])
            bk = bucket(ec["salt"], cid.strip())
            rt = q("SELECT share_b FROM routing_config WHERE experiment_id = ? ORDER BY routing_version DESC LIMIT 1",
                   (e_id,))["share_b"].iloc[0]
            rows.append({"experiment": e_id, "salt": ec["salt"], "bucket": bk,
                         "B if bucket <": round(ec["share_b"] * BUCKETS),
                         "during experiment": "B" if bk < round(ec["share_b"] * BUCKETS) else "A",
                         "after decision": "B" if rt >= 1 else ("A" if rt <= 0 else "hash split")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Each experiment has its own salt, so a caller's buckets are independent across experiments.")

# ---------------------------------------------------------------------------- playground
with tab_play:
    st.markdown("Set a true effect for variant B, run a fresh experiment on the real call log, and see whether "
                "the engine finds it. Runs are in memory only; the stored demo runs are not touched.")
    with st.form("play"):
        f = st.columns(3)
        base_name = f[0].selectbox("Start from scenario", list(SCENARIOS), index=list(SCENARIOS).index("win")
                                   if "win" in SCENARIOS else 0)
        share = f[1].slider("Share of callers on B", 0.05, 0.50, 0.10, 0.05)
        mde = f[2].slider("Minimum detectable effect (pp)", 2.0, 8.0, 3.0, 0.5)
        goal_opts = list(dataset().calls["disposition"].value_counts().index)
        goal = st.multiselect("Primary goal: dispositions that count as success (other dispositions can be goals too)",
                              goal_opts, default=list(VANI_SUCCESS), key="play_goal")
        f = st.columns(3)
        eff = f[0].slider("True effect of B on the goal rate, connected calls (pp)", -10.0, 10.0, 4.0, 0.5)
        dmult = f[1].slider("True duration multiplier for B", 0.8, 1.6, 1.0, 0.05)
        seed = f[2].number_input("Random seed", 0, 99999, 42)
        go_run = st.form_submit_button("Run experiment", type="primary")

    if go_run:
        base = load_scenario(SCENARIOS[base_name])
        goal_t = tuple(goal) if goal and sorted(goal) != sorted(VANI_SUCCESS) else None
        if not goal:
            st.warning("No disposition selected, so the goal is lead created.")
        cfg_p = base.model_copy(update={
            "experiment_id": "playground", "share_b": share, "mde_pp": mde / 100, "seed": int(seed),
            "primary_goal": goal_t, "baseline_rate": baseline(base_name, goal_t),
            "true_effect": base.true_effect.model_copy(update={"success_pp": eff / 100, "duration_mult": dmult})})
        with st.spinner("Simulating calls and running the looks…"):
            res = run_experiment(cfg_p, dataset(), design=plan_design(cfg_p, dataset()))
        st.session_state["play_result"] = res

    res = st.session_state.get("play_result")
    if res is not None:
        p_cfg = json.loads(res.cfg.canonical_json())
        p_last = res.looks[-1]
        truth = res.cfg.true_effect
        st.markdown(f"#### Goal: {goal_name(p_cfg)} · true effect {100 * truth.success_pp:+.1f} pp, "
                    f"duration ×{truth.duration_mult:.2f} → "
                    f"engine says **{res.decision}** at look {res.decision_look} (hour {res.decision_hour:g})")
        headline(p_cfg, res.design, p_last)
        p_lk = look_frame(res.looks)
        left, right = st.columns(2)
        left.plotly_chart(z_chart(res.design, p_lk, res.decision_hour, res.decision), width="stretch", key="p_z")
        right.plotly_chart(lift_chart(p_cfg, p_lk, res.decision_hour), width="stretch", key="p_lift")
        left, right = st.columns(2)
        left.plotly_chart(duration_chart(p_cfg, res.looks), width="stretch", key="p_dur")
        steps = pd.DataFrame({"virtual_hour": [0.0, res.decision_hour],
                              "share_b": [res.cfg.share_b, res.final_share_b]})
        right.plotly_chart(routing_chart(steps, res.decision_hour + res.cfg.post_rollout_hours), width="stretch",
                           key="p_route")
        st.dataframe(p_lk[["look", "virtual_hour", "n_a", "n_b", "p_a", "p_b", "lift_abs", "z", "eff_bound",
                           "harm_bound", "decision", "rule"]].round(4), hide_index=True, width="stretch")
        st.caption(f"Planned {res.design['sample_size']['n_total']:,} callers over {res.design['window_hours']} "
                   f"virtual hours; {len(res.calls):,} calls simulated. Re-run with other seeds to see how often "
                   "the same true effect gets the same decision (the Error rates tab does this 1,000 times).")

# ---------------------------------------------------------------------------- split evidence
with tab_split:
    rep = read_json("runs/split_report.json")
    if not rep:
        st.warning("Run `.venv/bin/python scripts/split_report.py` to generate split evidence.")
    else:
        acc = pd.DataFrame(rep["accuracy"])
        fig = go.Figure()
        fig.add_scatter(x=100 * acc["configured_share_b"], y=100 * acc["achieved_share_b"], mode="markers",
                        name="achieved (95% CI)", error_y=dict(
                            type="data", symmetric=False, array=100 * (acc["ci95_high"] - acc["achieved_share_b"]),
                            arrayminus=100 * (acc["achieved_share_b"] - acc["ci95_low"])))
        fig.add_scatter(x=[0, 55], y=[0, 55], mode="lines", line=dict(dash="dot", color="gray"), name="y = x")
        fig.update_layout(title="Configured vs achieved B share", xaxis_title="configured %",
                          yaxis_title="achieved %", height=380)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(acc.round(4), hide_index=True, width="stretch")
        s, ind, br = rep["stickiness"], rep["independence"], rep["broken"]
        c = st.columns(4)
        c[0].metric("Stickiness on replay", f"{s['consistency']:.0%}", f"{s['mismatches']} mismatches / "
                    f"{s['callers']:,} callers", delta_color="off")
        c[1].metric("Bucket correlation across salts", f"{ind['bucket_corr']:+.4f}",
                    f"B in both: {ind['b_in_both']} vs {ind['expected_b_in_both']:.0f} expected", delta_color="off")
        if rep.get("coverage"):
            cov = rep["coverage"]
            c[2].metric("95% CI coverage over salts", " / ".join(f"{x['coverage']:.0%}" for x in cov),
                        " / ".join(f"{x['share']:.0%} split" for x in cov), delta_color="off")
        c[3].metric("Broken split → SRM", "FLAGGED" if br["srm_flag"] else "missed",
                    f"lost {br['callers_lost_from_b']} B callers, p = {br['srm_p']:.1e}", delta_color="off")
        st.caption("Assignment: bucket = sha256(salt + ':' + caller_id) mod 10000; B if bucket < share × 10000. "
                   "Split is per caller (fk_glusr_usr_id), not per call.")

# ---------------------------------------------------------------------------- operating characteristics
with tab_oc:
    oc = read_json("runs/operating_characteristics.json")
    if not oc:
        st.warning("Run `.venv/bin/python scripts/operating_characteristics.py` to estimate error rates.")
    else:
        diag = oc.pop("_diagnostic_null_without_scoring_window", None)
        oc.pop("_diagnostic_null_no_repeat_callers", None)
        rows = [{"scenario": k, "seeds": v["seeds"], **{d: f"{r:.1%}" for d, r in v["rates"].items()},
                 "mean stopping look": f"{v['mean_stop_look']:.1f}" if v["mean_stop_look"] else "–"}
                for k, v in oc.items()]
        st.dataframe(pd.DataFrame(rows).fillna("–"), hide_index=True, width="stretch")
        st.caption("Each scenario re-run with fresh simulated data per seed and a fixed design. "
                   "null → PROMOTE is the false-winner rate (target ≤ 2.5%); null → STOP_HARM is the false-harm rate "
                   "(target ≤ 5%); win → PROMOTE is power; harm → STOP_HARM is harm-detection rate.")
        if diag and "null" in oc:
            n = diag["seeds"]
            st.info(
                f"**Repeat callers, found and fixed.** Scoring a caller on all their calls so far lets a repeat "
                f"caller's outcome change between looks, which breaks the independent increments the boundaries "
                f"assume: scored that way, the null scenario is wrongly stopped for harm "
                f"{diag['counts'].get('STOP_HARM', 0) / n:.1%} of the time over {n} seeds (target ≤ 5%). Each caller is "
                f"now scored on a fixed {diag.get('score_window_hours_fixed', 2):g} h window from their first call and "
                f"enters a look only once it has closed: {oc['null']['rates'].get('STOP_HARM', 0):.1%}.")

# ---------------------------------------------------------------------------- method
with tab_method:
    st.markdown("""
**Loop:** call arrives → hash-assign variant (sticky per caller) → outcome recorded → stats at scheduled looks →
decision engine → rollout controller updates routing → decision log written.

**Simulator (transparent by design):** control outcomes are bootstrap draws of real VANI calls. Variant B uses the
same draws with an injected true effect, applied only to calls that connected: a lead is added (or removed) with the
configured probability, bringing that real call's duration with it, and B's duration is scaled by `duration_mult`.
The prompt text does not change these numbers.

**Primary metric:** lead created (BL Enriched / BL Approved / BL Generated), per caller; any other set of
dispositions can be chosen as the goal (`primary_goal` in the config, or the Playground picker). Each caller is scored
on their calls within 2 h of their first call (81% of real repeat calls come within an hour) and enters a look once
that window has closed, so an analysed outcome never changes between looks. Two-proportion z-test with a
group-sequential design: O'Brien-Fleming-type alpha spending for efficacy (one-sided 2.5%), Pocock-type spending
for harm (one-sided 5%), 5 equally spaced looks, ≥ 300 callers per arm before any stop.

**Secondary guardrail:** mean duration of connected calls, ratio B/A, delta-method CI. Non-inferiority margin +15%;
hard stop if the 99% CI lower bound exceeds +30%.

| Primary | Guardrail | Decision |
|---|---|---|
| Crosses harm boundary | any | **Stop, revert to A** |
| any | Hard breach | **Stop, revert to A** |
| Crosses efficacy boundary | Non-inferior | **Promote B to 100%** |
| Crosses efficacy boundary | Not yet shown | **Continue** to next look |
| Crosses efficacy boundary | Violated, or not shown at final look | **Hold for approval**: B at 0% until a named approver promotes or rejects it (logged with name and reason) |
| No boundary by final look | | **Inconclusive, stay on A** |
| SRM detected | | **Invalid, revert to A** |
""")
