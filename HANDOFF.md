# Handoff: VANI Prompt Lab (Hackathon PS05)

Last updated 2026-10-10. Read this first when picking the project back up.

## Where things stand

- **Done and working:**
  - A/B engine on the real call log: 4 demo scenarios (win, harm, null, hold) plus the pilot-driven one.
  - Selectable primary goal (any set of dispositions), per-caller scoring window, logged approval step for held wins.
  - Dashboard with 10 tabs, including Approve/Reject for a held win (Experiments tab).
  - Sarvam pipeline, all stages run: 594 recordings transcribed, 591 audited, 60-persona pilot × 3 changes + an A/A check, 12 A/B pairs voiced.
  - Test suite: 66 tests, all passing offline.
  - 12-slide deck, one-slide summary, demo video script (`DEMO_VIDEO_SCRIPT.md`).
- **Sarvam keys:** `.env` holds `SARVAM_API_KEY` (the 1,100-credit key used for the 2026-10-10 runs) and `SARVAM_API_KEY_DEMO` (the older key, ~270 credits, kept for the live call). Swap the two lines to switch. Both keys were pasted in chat, so rotate them after the hackathon.

## Links
- **Deck (12 slides, downloads as .pptx/PDF):** https://claude.ai/artifact/AYTNmqvrbbRxgZPBwsiEzs. Private until shared from the Share menu.
- **One-slide summary (the brief's one-slide deliverable):** https://claude.ai/artifact/WvB79ZVPyqdnof7ijZp5bh. Download it as .pptx to replace the stale `ps05_onepager.pptx` in Downloads (that file has old numbers: Meeting Fixed 19.1→21.9%, +10% guardrail, 6,275 per arm).
- **Plan file:** `~/.claude/plans/zippy-conjuring-giraffe.md`.

## Run it
```bash
cd ~/ab-hackathon
.venv/bin/python scripts/run_scenarios.py        # win / harm / null / hold / pilot → runs/ab.db   (no API cost)
.venv/bin/python scripts/split_report.py         # split evidence → runs/split_report.json
.venv/bin/streamlit run app/dashboard.py         # http://localhost:8501 (localhost, for the mic)
.venv/bin/pytest -q                              # all offline
```
- **Optional, no API cost, ~9 min:** `.venv/bin/python scripts/operating_characteristics.py 1000` regenerates the error rates, including the run without the scoring window.
- **Costs Sarvam credits:** `scripts/pipeline.py`, or any single Sarvam stage script. Each resumes where it stopped and never re-pays for cached work.
- `run_scenarios.py` resets exp-hold to "awaiting approval", so the approval can be shown again.

## Key results (all from real runs; these are the deck numbers)
- **Data:** 1,000 rows, 997 with a caller ID, 825 buyers, 517 leads, 728 connected calls; lead rate 52% per call, 71% among connected calls.
- **Design (all scenarios):** 24,581 buyers planned, 2,459 on B, over 132 h; baseline 52.2% per buyer; MDE +3 pp; guardrail power 97% at the 15% margin (74% at 10%).
- **Demo experiments**, 10% of buyers on B:
  - win (+5 pp): **PROMOTE** at look 3 (79 h), +4.2 pp.
  - harm (−6 pp, calls ×1.3): **STOP_HARM** at look 1 (26 h), −6.5 pp.
  - null (no effect): **INCONCLUSIVE** at look 5.
  - hold (+5 pp, calls ×1.25): crosses the win line at look 4, keeps running until the guardrail is clear (R4b), **HOLD_FOR_APPROVAL** at look 5 (+3.2 pp, duration ×1.24, CI [1.156, 1.333] > 1.15). An approver then promotes or rejects it (R7, logged with name and reason).
  - pilot: true effect +4.8 pp, duration ×1.02, from exp-win's pilot estimate carried over as an odds ratio (1.28). Crosses the win line at look 2, waits for the guardrail (R4b) at looks 2–3, **PROMOTE** at look 4 (106 h), +5.4 pp. Seed 22 is an illustrative run: seed 21 was the 9% inconclusive case. Over 1,000 reruns: 68.1% promoted, 23.0% held, 8.7% inconclusive.
- **1,000 reruns:**
  - win: 77.4% promoted, 15.4% held, 7.0% inconclusive (caught 92.8%).
  - harm: stopped 98.8%.
  - null: false win 1.7% (target 2.5%); false harm-stop 5.0% (target 5%).
  - Without the scoring window (same seeds): false harm-stop 7.1%, false win 2.7%. That was the old engine.
  - hold: held 92.7%.
- **Split:** 10% configured gives 10.04% achieved; stickiness 100% (50,000 buyers, 0 mismatches); CI coverage over 200 salts 96% / 95%; a broken split is flagged with SRM p = 5e-32.
- **Real-call audit (591 calls, model judgements, not human QA):** 
  - 591 of 594 recordings audited (3 have no buyer speech). 63% created a lead, 35% Fatal overall, 112 calls with dead air over 5 s.
  - Top failures: Spec Missed Opportunity 15% fatal, 21% non-fatal; Qty Probing Error 3% / 16%; Looping Behavior 2% / 13%; Outcome Accuracy 11% / 3%; Qty Missed Opportunity 8% / 5%.
  - Languages: 554 hi-IN, 36 en-IN, 1 ml-IN; median call 72 s.
  - 7 persona replies had `null"` (a stray quote); `sarvam.parse_json` repairs it.
- **Simulated pilot (60 personas, judged blind):** 
  - Control = two calls on prompt A per persona (the A call and the A/A replicate), 120 calls, 34.2% leads.
  - exp-win: leads 34% → 40%, +5.8 pp [−5.0, +16.7], p 0.38; asks 0.81 → 1.00; re-asking calls 5% → 5%; duration ×1.02.
  - exp-harm: leads 34% → 28%, −5.8 pp [−16.7, +5.0], p 0.38; asks 0.81 → 1.72; **re-asking calls 5% → 22%**; **duration ×1.17 [1.08, 1.26]**; hang-ups 52% → 63%; Fatal 43% → 53%.
  - exp-null: leads 34% → 40%, +5.8 pp [−4.2, +16.7], p 0.36; duration ×1.09 [1.01, 1.17].
  - **A/A check** (prompt A vs an independent second call on prompt A): +11.7 pp [−3.3, +26.7]. With 60 buyers, lead effects are within noise; the pilot is for behaviour, the A/B engine for leads.
  - 50 of 60 personas matched an IndiaMART MCAT with spec fields; 10 got quantity units only.
  - 300 simulated calls; Sarvam usage for the whole project so far: ~4,450 cached chat calls, 10.1 M prompt and 7.4 M completion tokens.

## Decisions already made with the user (don't re-litigate)
- **Primary metric:** "Lead created" = BL Enriched / BL Approved / BL Generated, counted per buyer as a lead on any of their calls in the scoring window. Other goals can be picked (`primary_goal`).
- **Duration:** a timestamp proxy (`last_modified_date − entered_on`, minute resolution), used for connected calls only.
- **Guardrail, "continue until shown":** a win whose duration non-inferiority isn't shown yet keeps running. It holds only if the guardrail is violated, or still unshown at the final look. A hold goes to a named approver (`runner.resolve_hold`).
- **Guardrail margin:** 15%, not the brief's 10%. The minute resolution makes 10% underpowered.
- **`is_call_answered` is ignored:** it is 0 on 38 lead calls.
- **Never commit:** this is not a git repo, and the user's rule is never to commit. Never print the Sarvam keys, which live in `.env`.

## Choices made on 2026-10-10 (by Claude while the user was away; the user can revisit them)
- **Scoring window:** 2 h from each buyer's first call; a buyer enters a look only once their window has closed. Fixes the 7.1% false harm-stop (the fix was next step #3 of the previous handoff). 2 h comes from the real log: 81% of repeat calls come within an hour (median gap 5 min).
- **Hold scenario** (`scenarios/hold.yaml`, seed 14) to show the duration-veto rule and the approval step; its B is a diff only (no Sarvam rewrite), so it stays out of the pilot.
- **Pilot → engine:** effects carried over as an **odds ratio** (relative risk overshot 100% at the real 71% base); control pooled from the A call and the A/A replicate; exp-win chosen as the source (exp-win and exp-null tied); `exp-pilot` seed 22 (seed 21 was the 9% inconclusive tail), always quoted with the 1,000-rerun rates.
- **Simulated bot gets collection fields** from IndiaMART category data (product → MCAT match), pinned per persona.

## Sarvam: hard-won facts
- `sarvam-105b` reasons first, using 5–15k tokens and ~40 s per call. Give it `max_tokens` ≥ 16000, or it returns empty or truncated replies. It is used for the judge, prompt rewrites and prompt condensing.
- `sarvam-105b-conversations` has no reasoning and takes ~1 s. It is used for bot and buyer turns, personas and quantity units.
  - It sometimes returns **empty content** where the prompt says to call a tool; `convo.action_line` writes the spoken line instead.
  - It sometimes returns **malformed JSON** (personas): `realcalls._json_chat` retries once at temperature 0.3, a different request with its own cache entry.
  - It is **not deterministic at temperature 0**: the same units question gave "Piece, Set" and "Piece, Carton". Parallel threads that missed the cache together gave one persona different fields in A and B. Fixed with a lock in `catalog.collection_fields` and per-persona pinned fields (`runs/pilot/fields.json`).
- **The VANI prompt is a Jinja template.** Quantity and spec collection only appear when `quantity_unit_options` / `specification_options` are filled. Without them the simulated bot never asked for either (all of the first 8 pilot calls: 0 asks). The bot now gets the product's MCAT fields from `abtest/catalog.py` (IndiaMART category data in `data/catalog/`, from the Seller Enrichment datasets in the Drive zip).
- **Condensed bot prompt length:** at 60k characters the fast model went empty mid-call; the 2-pass result is ~16k characters, in `data/sarvam/vani_core.txt`.
- **STT:** real-time STT only accepts audio up to 30 s, so real calls go through the **batch job** API (`speech_to_text_job`, `saaras:v3`, `codemix`, diarization). Speaker IDs vary per call; `realcalls.bot_speaker` picks the bot by IndiaMART / "requirement" mentions.
- **TTS:** `bulbul:v3` speakers include priya, rahul and kavya; `anushka` is not supported. Use `language_code=`, not `target_language_code`.
- **Rate limits:** two chat stages at 6 workers each hit 429s. The pilot (6 workers) and the audit (3 workers) ran together fine.
- **Token use per simulated call:** ~32k prompt tokens on the fast model (about 12 turns × ~2.7k) plus one judge call (~1.5k in, ~10k reasoning out).

## File map
| Path | What |
|---|---|
| `abtest/` | Engine: `config`, `assigner` (hash router), `loader` (VANI CSV, `with_goal`), `simulator`, `stats` (group-sequential), `decision`, `runner` (scoring window, `resolve_hold`), `store` (SQLite), `split` |
| `abtest/sarvam.py` | Sarvam client with a disk cache (`data/sarvam/cache/`), retries, JSON parsing, thread pool, `replicate` cache tag |
| `abtest/catalog.py` | Product → IndiaMART MCAT → quantity units + two main spec fields for the simulated bot |
| `abtest/realcalls.py` | Batch STT, turn split, dead air, judge (`JUDGE_SYSTEM` + `data/call_quality_matrix.md`), personas |
| `abtest/convo.py` | Bot prompt assembly (core + collection fields + variant sections verbatim), buyer agent, `simulate`, `action_line`, the A/A arm |
| `abtest/pilot.py` | Paired A/B estimate (bootstrap CI, McNemar) and `to_true_effect` (relative scale, clamped) |
| `abtest/voice.py` | TTS per turn and WAV joining |
| `abtest/variants.py` | Sarvam rewrites the prompt section for a variant diff |
| `scripts/pipeline.py` | Runs stages 1–7 in order (transcribe → analyse → core → simulate → pilot.yaml → run_scenarios → voice) |
| `app/dashboard.py`, `app/lab_tabs.py` | Streamlit dashboard; the Real calls, Prompt lab and Live call tabs live in `lab_tabs.py` |
| `scenarios/*.yaml`, `scenarios/variants/*.txt` | win / harm / null / hold / pilot configs and the Sarvam-rewritten B sections |
| `data/` | Call log CSV, `base_prompt.txt`, `call_quality_matrix.md`, `recordings/` (713 mp3), `catalog/` (category data), `sarvam/` (transcripts, real_calls.jsonl, vani_core.txt, cache) |
| `runs/` | ab.db, split_report.json, operating_characteristics.json (and `_before_window`), pilot/ (conversations, personas, fields, summary), audio/ |
| `tests/` | Engine, split, scenarios (incl. window, goal, approval), dashboard (AppTest), prompt lab (Sarvam mocked) |
| `SUBMISSION.md` | Hackathon submission form copy (team Ai Avengers 2: lal krishna gupta, kirti jain, kumari aayushi singh) |
| `DEMO_VIDEO_SCRIPT.md` | Scene-by-scene video script with narration |
| `deck/project/` | Source of the 12-slide deck (deck.json + slides/*.html), a copy of the published artifact |

## Next steps
0. **Submission form:** paste-ready copy for every field is in `SUBMISSION.md` (pitch, PS05, approach note, impact). Still missing: the Sarvam agent ID / live link (we have a pipeline, not a dashboard.sarvam.ai agent; build one or ask the organisers), the demo video URL, and optionally a public repo or Drive folder (leave out `.env`, recordings and cache).
1. Record the demo video (`DEMO_VIDEO_SCRIPT.md`). Before recording, check the credit balance of the key in `SARVAM_API_KEY`; swap to `SARVAM_API_KEY_DEMO` if it's low.
2. Share the deck and the one-slide summary from their Share menus; download the one-slide as .pptx for submission.
3. Optional: calibrate the simulated buyers against real outcomes (the simulated A lead rate is ~34% vs 65% judged on real recordings; simulated buyers hang up more), then scale the pilot. A/B/n (more than two variants) is the brief's bonus item and isn't built.

## Source data location
- **Original Drive zip:** `/mnt/c/Users/Indiamart.imart/Downloads/Aids to solve Problems-20261009T070424Z-1-001.zip`.
- The recordings are copied to `data/recordings/`; the two category CSVs to `data/catalog/`.
