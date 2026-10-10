# VANI Call Quality Matrix

Transcribed from "Current Definitions of Call Quality Matrix being used in VANI.png" (hackathon Drive).
Each parameter is judged Pass, Non-fatal or Fatal. Use N/A where the parameter cannot be judged from a
text transcript or does not apply to the call.

| Parameter | Definition | Fatal | Non-fatal |
|---|---|---|---|
| Outcome Accuracy | Buyer intent is buying or not, and the bot's outcome matches it | Always fatal | NA |
| Quantity Accuracy (Numeric) | Quantity value is correctly captured | Fatal if the wrong value affects the lead, subject to category | Minor difference (10-20%) that can be overlooked |
| Quantity Unit Accuracy | Quantity unit shared by the buyer is correctly captured | Fatal if it impacts the buy lead | Similar-value units |
| Quantity Probing Error | Quantity probed when not defined / not probed when defined / nudge not proper | Fatal if defined and not asked | Not defined and asked |
| Qty Missed Opportunity | Quantity was shared but not captured | Fatal | Non-fatal if the voice was not clear |
| Spec1 Accuracy | Input X captured as Y with impact on the buy lead | Subject to category | Minor difference that can be overlooked |
| Spec2 Accuracy | Input X captured as Y with impact on the buy lead | Subject to category | Minor difference that can be overlooked |
| Spec Missed Opportunity | Spec 1 or 2 shared by the buyer is not captured | Spec 1 or 2 shared but not captured | Spec 3 missed |
| Looping Behavior | Probing more than 1 + 2 times for any parameter such as quantity or spec | Buyer gets irritated, leading to disconnection or a missed opportunity for qty/spec | No impact on the call |
| WER | Words wrongly captured by speech recognition | Impacts the buy lead where a word is captured otherwise | Word wrongly captured, no major impact |
| Additional_details | Buyer-filled details captured | Grossly wrong / irrelevant details that impact the lead | Duplicate or very generic details |
| OnCall AstBuy Error | Live seller present but not pitched | Seller present but not pitched | Not pitched but disposition wrongly shared |
| Product Change / No Product Accuracy | Product wrongly captured | Wrong product approved | |
| Category ISQ's Error | Contradicting or confusing ISQ (spec) questions | ISQs contradicting | |
| MCAT / Search errors | Wrong MCAT (category) selected | | |
| Dead air | Dead air more than 5 s (8 s with noise disruption) | | |
| Overall Call | Pass, Fatal or Non-fatal based on the parameters above | | |
| Additional Observation | Any other observation, e.g. pronunciation | | |

Lead created = the call ends with a buy lead (BL Generated, BL Enriched or BL Approved): genuine buying
intent and enough requirement detail (product plus quantity and/or specs) for a seller to act.
