# Submission form copy (Ai Avengers 2)

## Short pitch (≤280)
VANI Prompt Lab: change one line of VANI's prompt, test it on simulated Sarvam buyers first, then A/B it on real traffic with sticky routing and group-sequential stats. It promotes winners, stops harmful prompts within a day, and sends guardrail conflicts to a named approver.

## Problem statement
PS05: Prompt A/B testing and auto-rollout for VANI

## Approach note
**Data used**
- `Call_Logs_Hackathon.csv`: 1,000 VANI calls, 825 buyers, 517 leads. This is the baseline and the bootstrap source for the simulator.
- 594 real VANI recordings, transcribed with Sarvam batch STT (`saaras:v3`, codemix, diarization).
- The VANI base prompt (a Jinja template), the Call Quality Matrix, and IndiaMART category data (MCAT → quantity units and spec fields).

**Design**
1. **Real-call audit:** `sarvam-105b` scores each transcript against the Call Quality Matrix and pulls out a buyer persona (591 calls).
2. **Prompt lab:** `sarvam-105b` rewrites one prompt section to create variant B. Persona buyers (`sarvam-105b-conversations`) call bot A and bot B, and the bot gets the product's real MCAT quantity units and spec options, as live VANI does. A blind judge scores every call. We ran 60 personas × 3 changes plus an A/A check (300 calls), and voiced 12 pairs with `bulbul:v3`.
3. **A/B engine:** every experiment has a versioned config hash. A sticky hash router assigns each buyer (10% configured gives 10.04% achieved; 0 mismatches over 50k buyers; an SRM check flags a broken split).
   - Primary metric: lead created per buyer, scored within a 2 h window.
   - Looks: 5 group-sequential looks. O'Brien-Fleming boundary for a win (2.5% one-sided), Pocock boundary for harm (5%).
   - Guardrail: call duration non-inferiority within +15%, with a hard stop at +30%.
4. **Decisions:** PROMOTE, STOP_HARM, INCONCLUSIVE, or HOLD_FOR_APPROVAL. A held win goes to a named approver, and the decision is logged with the approver's name and reason. All events go to an append-only SQLite log.
5. **Dashboard** (Streamlit, 10 tabs): experiments, split health, real-call audit, prompt lab, A/B voice pairs, and a live call.

**Expected outputs delivered:** the A/B framework, auto-rollout and auto-stop rules, guardrails, a dashboard, an audit trail, the 12-slide deck, a one-slide summary, and 66 passing tests.

## Sarvam agent ID / live link
TODO

## Demo video URL
TODO

## Impact on success metrics
**How we measured:** the A/B engine runs on the real call log (baseline 52.2% leads per buyer, 24,581 buyers planned over 132 h, 10% on B). The error rates come from 1,000 reruns of each scenario.

| Scenario | Outcome | 1,000 reruns |
|---|---|---|
| Win (+5 pp) | PROMOTE at 79 h, +4.2 pp | caught 92.8% (77.4% promoted, 15.4% held for approval) |
| Harm (−6 pp, calls ×1.3) | STOP_HARM at the first look (26 h) | stopped 98.8% |
| Null | INCONCLUSIVE | false win 1.7% (target 2.5%), false harm-stop 5.0% (target 5%) |
| Hold (+5 pp, calls ×1.25) | HOLD_FOR_APPROVAL, duration guardrail vetoes the win | held 92.7% |

- **Safe rollout:** only 10% of buyers see an untested prompt. A harmful prompt is stopped after about a day, before it reaches everyone.
- **Fewer false wins:** the 2 h scoring window cut false harm-stops from 7.1% to 5.0%.
- **Pre-screening before live traffic:** the simulated pilot caught a harmful change on behaviour alone. Buyers re-asked in 22% of calls (5% on A), calls ran ×1.17 longer, and Fatal calls rose from 43% to 53%. With 60 personas the lead effect is within noise (the A/A check shows it), so the pilot screens behaviour and the A/B test decides on leads.
- **Where to improve prompts:** the real-call audit (591 calls, model judgements, not human QA) found missed spec opportunities in 15% of calls as fatal and 21% as non-fatal, and 112 calls with more than 5 s of dead air. These are the next prompt changes to test.
- **At scale:** VANI can test prompt changes continuously without engineers reading results by hand. Buyers get better calls, sellers get more and better-enriched leads, and every rollout can be audited and reversed.
