# VANI Prompt Lab: demo video script

Target length: about 5½ minutes. Record the screen at 1920×1080 with the browser zoomed to 90%, so all dashboard tabs fit.

## Before you record (10 minutes)

1. Start the dashboard:
   ```bash
   cd ~/ab-hackathon
   .venv/bin/python scripts/run_scenarios.py     # rebuilds the demo experiments (~20 s, no API cost)
   .venv/bin/streamlit run app/dashboard.py
   ```
   `run_scenarios.py` also resets **exp-hold** to "awaiting approval", so you can approve it on camera.
2. Open **http://localhost:8501**. It must be `localhost`, or the browser blocks the microphone.
3. Click through every tab once, so the page is warm and nothing loads during recording.
4. **Do one short test call in the Live call tab before recording.** It uses Sarvam credits (a few rupees). The key in use is `SARVAM_API_KEY` in `.env`; the 270-credit demo key is kept there as `SARVAM_API_KEY_DEMO`.
5. Prepare the live-call lines. With product **TMT steel bars** the bot asks, in order: confirm the product → how many tons → grade (Fe 550D or Fe 550) → diameter → your name → your city → transfer. Say (Hinglish):
   - "Haan, mujhe TMT bars chahiye."
   - "Kareeb 2 ton chahiye."
   - "Fe 550D chahiye."
   - "12 mm wale."
   - "Mera naam Rohit hai."
   - "Pune se, Maharashtra."
   - "Haan, seller se connect kar do."

   Shorter take: answer several at once ("2 ton, Fe 550D, 12 mm"). The bot captures volunteered details and skips those questions.
6. The Real calls tab shows real buyer names and voices. Pick a call where no name is spoken, or blur it in editing.
7. Have the slide deck open in another tab: https://claude.ai/artifact/AYTNmqvrbbRxgZPBwsiEzs

---

## Scene 1: Hook (0:00–0:20)
**Screen:** deck slide 1 (cover).

**Say:**
> "VANI talks to thousands of IndiaMART buyers a day, and everything it says comes from one 163-kilobyte prompt. Change one line and every buyer hears it. We built the VANI Prompt Lab: a way to prove a prompt change works before everyone gets it."

## Scene 2: The problem and the plan (0:20–0:50)
**Screen:** slide 2 (problem), then slide 4 (three layers).

**Say:**
> "Today prompt changes ship blind. The effects are small, a few points of leads, easy to miss and easy to fake by peeking at results. And if a change hurts, nothing rolls it back.
> Our lab gives every change three layers of evidence. Learn from real calls, try the change in simulated calls, then prove it on live traffic with an A/B test that decides and rolls out by itself."

## Scene 3: Real data (0:50–1:10)
**Screen:** dashboard → **Call log** tab. Point at the funnel, then click any row.

**Say:**
> "This is the real VANI call log from the hackathon. 1,000 calls, 825 buyers, 517 leads. Click a row and you see exactly how we classified it: lead or not, connected or not, and the call duration."

## Scene 4: Real calls with Sarvam (1:10–1:45)
**Screen:** **Real calls** tab. Point at the failure chart, pick a call with "Fatal", press play for 5 seconds, then scroll to the audit.

**Say:**
> "We transcribed 594 real VANI recordings with Sarvam speech-to-text, bot and buyer separated, and audited 591 of them with sarvam-105b against VANI's own Call Quality Matrix.
> A third of calls fail it. The top failure is missing specifications the buyer gave. That tells us which prompt changes are worth testing."

## Scene 5: Prompt lab: A vs B in simulation (1:45–2:40)
**Screen:** **Prompt lab** tab. Choose `exp-win`, point at the metrics row, open "Persona … collection fields", pick a persona marked 🔊 and play a few seconds of A, then B.

**Say:**
> "Here's a prompt change: 'ask each spec once, then offer the live seller'. Sarvam rewrote exactly that section of VANI's prompt. Then two Sarvam agents talk: the VANI bot on prompt A or B, and a buyer modelled on a real recorded caller. Like live VANI, the bot gets the product's real spec fields, here from IndiaMART's own category data, so it asks the same questions a real call would. A blind judge scores both calls without knowing which prompt was used.
> We ran 60 buyer personas through every change: the interrogation prompt re-asks in four times as many calls and makes them 17% longer, exactly the behaviour we'd worry about. On leads, though, we also ran an A/A check: prompt A against itself moved 12 points by noise alone. So sixty simulated buyers can't measure a five-point lead effect. That's what live traffic is for."

**Screen tip:** choose `exp-aa` in the Prompt-change box for a second to show the A/A row, then go back to `exp-win`.

## Scene 6: Live call (2:40–3:30). The wow moment.
**Screen:** **Live call** tab.
1. Type a buyer ID, e.g. `98765`, and point at the router line ("bucket … → A").
2. Click **Start call**. Point at the "IndiaMART category: TMT Steel Bars · units … · specs Grade, Diameter" line. The bot greets you aloud.
3. Click the mic, speak your first line, then stop. The bot answers aloud.
4. Continue with the prepared lines until the bot transfers you.
5. **Skip "End call and audit it"** unless you have credits to spare; it's the most expensive click.

**Say (over step 1–2):**
> "Now live. My buyer ID goes through the same hash router as real traffic, so I always get the same prompt. The bot is given TMT steel bars' real spec fields: grade and diameter."

**Say (after the call):**
> "Sarvam heard me, the bot asked what a real VANI call asks, and Sarvam's voice spoke every reply. That's the whole loop."

## Scene 7: The A/B engine decides (3:30–4:40)
**Screen:** **Experiments** tab.
1. `exp-win`: point at the black Z line crossing the green line at look 3, then at the rollout chart jumping to 100%, then at the decision log.
2. Switch to `exp-harm`: stopped at look 1, B goes to 0%.
3. Switch to `exp-null`: inconclusive, stays on A.
4. Switch to `exp-hold`: point at the orange **Approval needed** box and the duration chart above the +15% line. Type "Priya, category lead" as approver and a reason, click **Approve: promote B**. Point at the new decision-log row with the approver's name.
5. Switch to `exp-pilot`: point at the blue box (its true effect is the pilot's estimate for exp-win, carried over as an odds ratio: +4.8 pp), then at the Z line: it crosses the win line at look 2, keeps running until duration is shown non-inferior, and is promoted at look 4.

**Say:**
> "On live traffic, 10% of buyers get B. The engine checks five times, at points fixed in advance. Each check has a win line and a harm line, so peeking can't create a fake winner.
> The good change crossed the win line at the third check and was promoted to everyone automatically. The harmful one was stopped at the first check. The cosmetic one: no difference, so we stay on A.
> This one wins on leads, but calls run 25% longer. Leads decide and duration can only veto, so the engine doesn't auto-ship it: B goes back to zero and a named person decides. I approve it, and the log records who, when and why.
> The last one closes the loop: its true effect is the pilot's estimate for the winning change. If the pilot is right, the engine confirms it at the fourth check, and only once the call length is shown to be fine."

## Scene 8: Proof it's trustworthy (4:40–5:15)
**Screen:** **Error rates** tab, then **Split evidence** tab.

**Say:**
> "We reran each scenario a thousand times. A real winner is caught 93% of the time, a harmful change is stopped 99% of the time, and when a change does nothing we declare a false win only 1.7% of the time, under our 2.5% target.
> These reruns also caught a bug in our own statistics: a buyer who called back could change their result between checks, and wrong harm-stops rose to 7.1%. Scoring each buyer on a fixed two-hour window from their first call fixed it: 5.0%, on target.
> And the router is exact: 10% configured, 10.04% achieved, and every buyer always gets the same variant."

## Scene 9: Close (5:15–5:30)
**Screen:** deck slide 12 (limits and next), then slide 1.

**Say:**
> "Every prompt change: learned from real calls, tried on simulated buyers, proven on live traffic, and rolled out or rolled back automatically, with a person in the loop when the evidence conflicts. VANI Prompt Lab."

---

## If something goes wrong while recording
- **The Live call shows an error:** click **New call** and retry once. If it fails again, the credits may be out: swap the two keys in `.env` (`SARVAM_API_KEY` ↔ `SARVAM_API_KEY_DEMO`), restart streamlit, and retry. If both are out, cut the scene and show the Prompt lab conversations instead.
- **The mic isn't picked up:** type the buyer line in the chat box under the mic. The bot still answers aloud.
- **exp-hold is already approved** (from a rehearsal): run `.venv/bin/python scripts/run_scenarios.py scenarios/hold.yaml` and reload the page.
- **The dashboard says "No experiments":** run `.venv/bin/python scripts/run_scenarios.py` again.
- **Port 8501 is busy:** add `--server.port 8502` to the streamlit command.
