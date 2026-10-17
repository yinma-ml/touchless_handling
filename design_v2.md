# Company ABC: Auto Physical Damage Touchless Handling (STP) — Experiment Design v2

**Claim-lifecycle decision architecture, supervised gate specification, and counterfactual simulation blueprint**

- **Target environment:** Amazon SageMaker, Amazon Bedrock (Claude Sonnet 4.6 or later), LightGBM, CatBoost, Python
- **Audience:** Sections 1–3 for business and executive alignment; Sections 4–11 for data science and engineering implementation; Section 12 is the coding reference for LLM-assisted code generation.
- **Historical data:** ~300,000 closed auto physical damage claims (no bodily injury); ~8 months of FNOL call transcripts (~200,000 claims); FNOL stored as an immutable snapshot; all post-FNOL information arrives in an append-only table loaded in daily batches.
- **POC window:** 1 month.

---

## Table of Contents

1. Problem, Goal, and Scope
2. Design Principles
3. Claim Lifecycle and the Decision Timeline (core of the design)
4. System Architecture
5. Data Architecture and Point-in-Time Reconstruction
6. LLM Components (Claude Sonnet 4.6+)
7. Supervised Gates, Payment Rules, and Secondary Models
8. Counterfactual Simulation Engine
9. Validation, Metrics, and POC Acceptance Criteria
10. One-Month POC Plan
11. Path to Production and Governance
12. Repository Structure and Coding Guardrails
13. Appendix: Changes from v1, Assumptions, Open Items

---

## 1. Problem, Goal, and Scope

### 1.1 Baseline

Company ABC closes **20%** of auto physical damage claims straight-through at FNOL. Eligibility is determined by deterministic rules on the Claim Description Code (CDC) and intake checkboxes. Any ambiguity (unclear CDC, missing fields, two moving vehicles, non-standard endorsement) sends the claim to a desk adjuster, who spends days collecting statements, ordering police reports, reviewing the policy, and reconciling the appraisal. This drives Loss Adjustment Expense (LAE), cycle time, and inconsistent decisions.

### 1.2 Goal

Raise touchless handling from **20% to 80%** of in-scope claims without degrading decision accuracy or increasing payment error. 80% is the target; the POC's job is to build the pipeline that gets there and to measure, segment by segment, how much of the gap each evidence policy and decision timing closes.

### 1.3 Scope boundary: where automation ends

Automation covers everything from FNOL through **indemnity payment authorization and the subrogation trigger**. The following are explicitly outside the automated path and are **not counted as a human touch**:

- Total loss administration after indemnity is authorized: salvage, title transfer, lienholder payoff, owner-retained salvage.
- Supplement handling after the initial estimate is paid (existing downstream process).
- Rental / loss-of-use and diminished value (Phase 2; deterministic add-ons).

The following exit the automated path immediately and route to specialized units (counted as non-STP):

- Any bodily injury indication, litigation or attorney representation, SIU/fraud referral.

### 1.4 In-scope claim archetypes

Both insured-reported and claimant-reported claims are in scope. The archetype is set at FNOL from the reporter flag and extracted facts, and drives the evidence requirements and the payment rule.

| Archetype | Reporter | Adverse party | What we pay | Liability role |
| --- | --- | --- | --- | --- |
| **A1** First-party, no adverse party | Insured | None (single vehicle, parked and struck by unknown, animal, glass, weather) | Insured's repair or ACV minus deductible under collision/comprehensive | Not needed for payment; subrogation only if an adverse party is later identified |
| **A2** First-party with adverse party | Insured | Identified other driver | Insured's repair or ACV minus deductible under collision | Determines subrogation (and deductible recovery) and any later third-party exposure |
| **B** Third-party property damage | Claimant (or their carrier) | Our insured | Claimant's repair or ACV × insured liability %, capped at PD limit, no deductible | **Required for payment**: we pay the claimant only if our insured is at least partially liable under the loss state's negligence rule |

### 1.5 Definition of "touchless"

A claim is touchless if no desk adjuster makes or confirms a decision between FNOL and indemnity authorization. Automated field-appraiser dispatch (the preference of many senior customers) is an evidence-gathering step, not a touch. Because this definition differs from the current one, the system reports three rates side by side:

1. **Desk-touchless STP** — the headline 80% metric.
2. **Zero-human-cost STP** — desk-touchless and no field appraiser.
3. **Micro-touch STP** — all decisions automated except one human confirmation (typically the liability split); reported separately, never blended into (1).

---

## 2. Design Principles

1. **LLMs extract and reason; calibrated supervised models decide.** Claude Sonnet produces structured categorical and boolean facts and bounded recommendations. It never outputs a probability that gates money. Every payment gate is a LightGBM/CatBoost model calibrated on historical ground truth.
2. **Decision stability is the learning target.** For each decision (coverage, liability, damage amount), the question is: "Will the decision reachable with the evidence available today match the final adjudicated decision?" This is what makes human investigation provably unnecessary on a claim.
3. **Decisions are made at the right time, not on a fixed day.** The system re-scores a claim whenever new evidence lands (daily batch) and pays, waits, orders evidence, or escalates based on the gates and on whether the pending evidence is decision-relevant. The sweet spot per segment is discovered by simulation, not assumed.
4. **Evidence is ordered only when it can change the decision.** An LLM Evidence Need Agent reasons from the facts of the loss whether a police report or counterparty statement is needed. Clear-liability scenarios do not wait for a police report.
5. **First-party indemnity is decoupled from subrogation.** The insured is paid once coverage and damage are locked, regardless of fault; recovery is pursued asynchronously.
6. **Third-party payment requires established liability.** Claimants are paid only when the insured's liability clears the gate and the state negligence rule.
7. **Payment arithmetic is deterministic.** No model or LLM computes dollars.
8. **Pretrained LLM + GBDT, not custom deep learning.** 300k claims and 8 months of transcripts do not support training neural networks; auditability requires interpretable features.
9. **Payment error is two-sided.** Overpayment is leakage; underpayment is customer harm and regulatory exposure. Both are measured.
10. **Fairness by construction.** Driver age is not a routing feature. It is a monitoring dimension (STP rate, cycle time, payment error by age band), given the senior customer base.

---

## 3. Claim Lifecycle and the Decision Timeline

This section replaces the "Day 0 / Day 1–5" step list in v1. The timeline is not a schedule of what to do each day; it is the claim's information lifecycle, and the system's job is to decide at each point whether the information is sufficient.

### 3.1 Evidence events and arrival

| Event | Source | Typical arrival (to be replaced by empirical distributions in Week 1) | Observable in history? |
| --- | --- | --- | --- |
| FNOL snapshot + call transcript / free text | Intake (phone/web/app), reporter = insured or claimant | Day 0 | Yes, all claims; transcripts for ~8 months |
| Policy and endorsement check | Policy DB, endorsement documents | Day 0 (parallel) | Yes |
| Daily Total Loss ML score | Existing DS model, daily batch | Day 0–1, then daily | Yes (historical scores stored) |
| CCC estimate (repair $ or TL with ACV) | Customer photos or field appraiser via CCC | Photo: day 1–2; field: day 2–5 | Yes |
| Counterparty / insured statement | Digital questionnaire (future) or recorded statement (historical) | Day 1–4 | Only on claims where an adjuster ordered one; **insured-response event not logged** for claimant-reported claims |
| Police report | Automated request | Day 3–14, often never for minor losses | Only on claims where it was ordered |
| Supplement | Shop via CCC | Post-payment | Yes (downstream, out of scope) |
| Final adjudication (coverage, liability %, paid) | Closed-claim record | Day 30–90 | Yes, for mature claims |

### 3.2 Scoring epochs

The production system and the simulator share one cadence:

- **Epoch 0 (FNOL):** Near-real-time. Extract facts from the transcript/free text, run the policy check, run the gates on FNOL-only information, run the Evidence Need Agent, dispatch the appraisal per customer preference.
- **Epoch d (daily batch, d = 1 … T_max):** After the nightly append-table load, re-score only claims whose information state changed (new CCC estimate, TL score, statement, police report) plus claims that hit a timeout check. Because the information state changes only when evidence arrives, this is equivalent to event-triggered scoring with a daily clock.

### 3.3 Decision policy at each epoch

At every epoch, for every open claim in the automated track:

```
I       = information state (FNOL snapshot + append rows with insert_ts <= fnol_ts + d)
if guardrail(I):                     # BI hint, litigation, SIU, attorney
    -> EXIT to specialty unit
cov_hat, p_cov  = G1(I)              # rule/endorsement coverage decision + P(it is final)
L_hat,   p_liab = G2(I)              # liability bucket + P(bucket is final)
dmg_hat, p_dmg  = G3(I)              # CCC amount accepted + P(it is final)   (only once CCC present)

if pay_condition(archetype, cov_hat, p_cov, L_hat, p_liab, dmg_hat, p_dmg, tau):
    -> PAY (deterministic indemnity engine), trigger subrogation if applicable
else:
    pending = evidence not yet received that the Evidence Need Agent marked 'useful' or 'required'
             or that is already in flight (CCC, TL score)
    if pending is empty:
        -> ESCALATE the failing decision only (micro-touch) ; other decisions stay locked
    elif P(pending arrives before T_max - d) < min_arrival_prob:
        -> ESCALATE the failing decision
    else:
        -> WAIT (claim stays in the touchless holding queue)
if d == T_max and not terminal:
    -> ESCALATE
```

`pay_condition` by archetype:

- **A1:** `cov_hat ∈ {Cover, Partial}` and `p_cov ≥ τ_cov` and `p_dmg ≥ τ_dmg`. Liability is not required.
- **A2:** Same as A1 for the insured's payment. Liability is scored for subrogation routing and does not block payment to the insured.
- **B:** A1 conditions on the insured's policy (liability coverage active, PD limit) **plus** `p_liab ≥ τ_liab` and `L_hat` payable under the loss-state negligence rule (Section 7.5). If `L_hat` is an intermediate split in the POC, the claim goes to micro-touch for the split confirmation.

Coverage outcomes `Not Cover` and `Unable to Determine` always escalate to a human. The system never issues a denial.

### 3.4 Finding the sweet spot: value of waiting

The system does not choose a fixed waiting period. For a claim segment `s` (archetype × loss scenario × appraisal method × state group) and epoch `d`, define:

- `STP(s, d)` — share of claims in `s` whose gates clear by epoch `d`.
- `Err(s, d)` — two-sided payment error and decision-reversal rate among those paid at `d`.
- `Gain(s, d → d+1)` — increase in `STP` from waiting one more day given the evidence expected to arrive.
- `Cost(d)` — holding cost: cycle-time penalty, prompt-pay statute proximity, queue capacity.

The simulator (Section 8) produces, for each segment, the curve of `STP` and `Err` against `d` under each evidence policy. From those curves we derive the **Decision Timing Table**: the recommended `T_max`, the evidence to wait for, and the expected share decidable at Day 0 versus deferred. In production this table is what the policy in 3.3 consults, through `min_arrival_prob` and `T_max` set per segment.

Expected shape (to be measured, not assumed): A1 claims with photo appraisal decide at Epoch 0–2; A1 with field appraiser at Epoch 2–5; A2 payment to the insured at the same cadence as A1, with subrogation routing deferred; B claims split between Day-0-clear scenarios (rear-end where the insured struck the claimant and admits it) and scenarios that need a counterparty account or police report.

### 3.5 Lifecycle walk-throughs

**A1 — insured hits a deer, phone FNOL, prefers field appraiser.** Epoch 0: transcript extraction confirms single vehicle, animal strike, no injury; comprehensive coverage active; Evidence Need Agent returns `not_needed` for all evidence; G1 clears; appraiser dispatched. Epoch 3: CCC estimate arrives (repairable); TL score agrees; G3 clears; indemnity = estimate − comprehensive deductible; paid. Zero touches.

**A2 — insured reports being rear-ended at a light, photo appraisal.** Epoch 0: extraction shows two vehicles, insured stopped, struck from behind, adverse driver identified; G1 clears; G2 returns `0%` insured liability with high probability; Evidence Need Agent returns `not_needed` for police report (clear rear-end, no dispute signals). Epoch 1: photos → CCC estimate; G3 clears; insured paid under collision; subrogation model triggered against the adverse carrier for the paid amount plus deductible.

**B — claimant reports our insured changed lanes into them.** Epoch 0: claimant transcript extracted; reporter = claimant; insured's policy identified; liability PD coverage active; G2 shows a lane-change dispute scenario with probability mass split between `0%` and `100%`; Evidence Need Agent returns `required` for the insured's account and `useful` for a police report; digital questionnaire sent to the insured; police report requested. Epoch 2: CCC estimate on the claimant vehicle arrives; G3 clears; payment blocked on liability. Epoch 4: insured's account arrives (admits late lane change); G2 moves to `100%`, clears; claimant paid at PD limit cap, no deductible. If the account never arrives by `T_max`, only the liability decision escalates (micro-touch); coverage and damage remain locked.

---

## 4. System Architecture

**Layer 1 — Ingestion and deterministic baseline (Lambda / Step Functions / daily Glue or SageMaker Processing job)**

- Existing CDC rule engine (preserves the current 20%).
- Policy and standard-endorsement SQL engine (active on DOL, coverage, limits, deductibles; 80% of policies).
- Append-table loader with `insert_ts` on every row.
- Guardrail filter (BI, litigation, SIU) executed before any model.

**Layer 2 — GenAI extraction and reasoning (Bedrock, Claude Sonnet 4.6+)**

- Extractor 1: FNOL transcript and free text (insured-reported or claimant-reported) → fixed JSON schema.
- Extractor 2: statements and police reports → sanitized facts and conflict flags.
- Evidence Need Agent: categorical recommendation on police report / counterparty statement / insured account.
- Endorsement Agent (P1): non-standard endorsement text → deductible and coverage modifiers.
- Damage–narrative consistency check: CCC damaged panels vs. extracted point of impact.

**Layer 3 — Calibrated supervised gates (SageMaker endpoints or batch transform)**

- G1 Coverage Stability Gate, G2 Liability Stability Gate, G3 Damage/Payment Gate (P0).
- Reserve severity model, subrogation propensity refit, endorsement verifier (P1).
- Existing Daily Total Loss model consumed as a feature.

**Layer 4 — Decision loop (Step Functions over the daily batch)**

- Implements Section 3.3. State per claim: `OPEN → {PAID, WAIT, ESCALATED(decision), EXIT_SPECIALTY, TIMEOUT}`.
- Deterministic indemnity engine and subrogation trigger.
- Full audit record per epoch: information state hash, gate scores, agent outputs, policy action.

---

## 5. Data Architecture and Point-in-Time Reconstruction

### 5.1 Point-in-time principle

The FNOL record is immutable and all later information is inserted (never updated) with an `insert_ts`. The information state of claim `i` at epoch `d` is therefore exactly reproducible:

```
I(i, d) = fnol_snapshot[i]  ∪  { r ∈ append_table : r.claim_id = i and r.insert_ts ≤ fnol_ts[i] + d days }
```

No model may read a field that is not in `I(i, d)` when scoring epoch `d`. The snapshot builder enforces this by construction; a unit test asserts that no feature column has an `insert_ts` later than the epoch.

### 5.2 Datasets

| Dataset | Grain | Content and guardrails |
| --- | --- | --- |
| `df_fnol_snapshot` | 1 row per claim (~300k) | Immutable FNOL fields: CDC, channel, reporter type (insured / claimant / carrier), loss state, vehicles, policy snapshot, free text, appraisal preference, initial rule reserve. |
| `df_fnol_transcripts` | 1 row per phone FNOL (~200k, 8 months) | Raw ASR transcript. Speaker role (insured vs. claimant) attached from reporter type. |
| `df_daily_append` | 1 row per inserted fact | CCC estimate (amount, repair/TL flag, ACV, estimate version), daily TL score, statement text, police report text, reserve changes, supplements, each with `insert_ts`. |
| `df_statements_sanitized` | 1 row per statement | LLM-sanitized statements with adjuster opinions and fault conclusions removed. Flag `statement_exists` is itself a selection signal (Section 5.4). |
| `df_policy_endorsements` | 1 row per policy-form × endorsement | Indexed endorsement text for retrieval by the Endorsement Agent. |
| `df_closed_outcomes` | 1 row per mature claim | Final coverage status, final liability %, paid indemnity by estimate version (initial vs. supplements), reserve history, subrogation recovered, litigation/SIU flags, closed date. |
| `df_gold_audit` | 500–1,000 claims | Blind re-adjudication by two senior adjusters (coverage, liability %, payable amount) plus hand-labeled extraction fields. Stratified over archetypes and over historical manual claims. |

### 5.3 Maturity and censoring

Labels require a mature claim: closed ≥ 60 days for repairable, ≥ 90 days for total loss, measured against a fixed evaluation date. Complex claims close later, so any closed-only population over-represents easy claims. Mitigations:

- Report the open-at-evaluation share per FNOL month and per archetype alongside every STP figure.
- Carry open claims through the simulator as `UNRESOLVED` rather than dropping them; present STP as a range (resolved-only vs. all-open-counted-as-non-STP).

### 5.4 Selection bias in evidence features

Historically a statement or police report exists only because an adjuster judged the claim complex. `has_statement` and `has_police_report` therefore encode the historical routing decision. They are used as evidence-availability features in the gates (necessary for per-epoch scoring) but their effect is interpreted as conditional on the historical evidence policy. The simulator handles the counterfactual explicitly (Section 8.4).

### 5.5 Temporal splits

- Transcript-dependent gates (G1, G2, Evidence Need validation): transcript months M1–M8, of which ~M1–M6 are mature. Train M1–M4, calibrate M5, out-of-time test M6 (plus M7 with censoring flag).
- Structured-only models (G3 damage gate, reserve severity, subrogation): full multi-year 300k history. Transcript features are present only in the 8-month window; LightGBM handles the missing values natively with an explicit `has_transcript` flag.

---

## 6. LLM Components (Claude Sonnet 4.6+)

All prompts return strict JSON validated by Pydantic schemas held in `config/extraction_schemas.json`, versioned, and pinned to the model version that trained the gates. No prompt asks for a numeric confidence.

### 6.1 Extractor 1 — FNOL transcript and free text

Applied to insured-reported and claimant-reported calls; `speaker_role` is passed in the prompt.

| Field | Type | Notes |
| --- | --- | --- |
| `vehicle_count` | int | moving vehicles involved |
| `insured_vehicle_motion` | enum {parked, stopped, moving, unknown} | |
| `loss_scenario` | enum {rear_end_insured_struck, rear_end_insured_struck_other, parked_hit, single_vehicle, animal, glass_only, weather_hail_flood, intersection, lane_change_merge, backing, sideswipe, other} | primary liability driver |
| `insured_admits_fault`, `counterparty_admits_fault` | bool / null | |
| `point_of_impact` | enum {front, rear, left_front, …} | for damage consistency |
| `drivable`, `airbag_deployed` | bool / null | severity hints |
| `bodily_injury_hint` | bool | guardrail |
| `police_on_scene`, `citation_mentioned_to` | bool, enum {insured, other, none, unknown} | |
| `adverse_party_identified` | bool | plate, name, or carrier captured |
| `weather_or_road_hazard` | bool | |
| `dispute_signal` | bool | reporter disputes the other party's account or expects a dispute |
| `missing_critical_fact_count` | int | count of schema fields the call did not establish |

### 6.2 Extractor 2 — statements and police reports

`narrative_contradiction` (none / minor / material), `police_citation_issued_to`, `police_fault_narrative` (insured / other / shared / none), `physical_damage_matches_story`. Historical statements are first passed through the sanitizer prompt that removes adjuster opinions, coverage language, and fault conclusions.

### 6.3 Evidence Need Agent

**Purpose:** decide whether waiting for evidence can change the liability decision. If liability is clear from the loss facts, no police report is ordered.

**Inputs:** Extractor 1 output (both parties if available), archetype, loss state, G2's current bucket probabilities (the agent may read model probabilities; it may not emit one), list of evidence already received or in flight.

**Output schema:**

```json
{
  "police_report":           "not_needed | useful | required",
  "counterparty_statement":  "not_needed | useful | required",
  "insured_account":         "not_needed | useful | required",
  "reason_codes": ["clear_rear_end", "single_vehicle", "fault_admitted", "conflicting_accounts", "intersection_no_signal_info", "..."],
  "rationale": "two sentences max"
}
```

**Deterministic overrides (applied before the LLM):** `single_vehicle`, `animal`, `glass_only`, `weather`, `parked_hit` with no adverse party → everything `not_needed`. `rear_end_insured_struck_other` with `insured_admits_fault` → `not_needed`. Archetype B with no insured account yet → `insured_account = required` unless the claimant transcript includes an admission by the insured documented at the scene.

**Validation (Section 8.6):** on historical claims that did receive a police report, the liability decision change rate after the report must be materially higher for `required` than for `not_needed`. If it is not, the agent's categories carry no information and the rule overrides alone are used.

### 6.4 Endorsement Agent (P1)

Retrieval over `df_policy_endorsements` (indexed once per form, not re-read per claim) → `applicable_deductible`, `sub_limit`, `exclusion_hit`, `endorsement_ambiguity_flag`. Gated by the endorsement verifier model (Section 7.6).

### 6.5 Extraction quality

- Gold set: 300–500 transcripts hand-labeled for every Extractor 1 field; per-field precision/recall reported; fields below 0.85 F1 are excluded from gate features until the prompt is fixed.
- ASR quality: word error rate sampled on 100 calls, stratified by caller age band; if WER is materially worse for seniors, the extractor's `missing_critical_fact_count` is reviewed for age correlation.
- Cost and throughput: Bedrock Batch Inference; budget estimated in Week 1 from token counts on a 1k sample.

---

## 7. Supervised Gates, Payment Rules, and Secondary Models

All gates: LightGBM (CatBoost where categorical cardinality is high), isotonic calibration on the calibration split, ECE reported on the test split. Every gate takes an **evidence-availability vector** so one model serves every epoch: `has_transcript, has_ccc, has_tl_score, has_statement, has_police_report, has_insured_account, epoch_day`.

### 7.1 G1 — Coverage Stability Gate

- **Decision input:** `cov_hat` from the SQL engine (80%) or Endorsement Agent (20%), in {Cover, Partial, Not Cover, Unable}.
- **Target:** `Y = 1` if `cov_hat == final_coverage_status` on the closed claim.
- **Features:** `cov_hat`, policy tenure, days loss-to-report, coverage type, `endorsement_ambiguity_flag`, `num_endorsements`, loss state, `loss_scenario`, `vehicle_count`, reporter type, evidence vector.
- **Routing:** `Not Cover` and `Unable` force escalation regardless of score.
- **Phase 2:** when the existing Coverage Agent can be batch-run, its decision replaces or complements `cov_hat` and G1 becomes its meta-validator.

### 7.2 G2 — Liability Stability Gate

The Virtual Handler Liability Agent cannot be run on history in the POC, so G2 is a **direct liability model**.

- **Algorithm:** LightGBM multiclass over insured-liability buckets `{0%, 1–49%, 50%, 51–99%, 100%}`, isotonic-calibrated per class. `L_hat` is the top bucket; `p_liab` is its calibrated probability.
- **Target:** final closed insured liability %, bucketed.
- **Features:** Extractor 1 fields (both parties when available), `loss_scenario`, `insured_vehicle_motion`, admissions, `citation_mentioned_to`, `dispute_signal`, Extractor 2 fields when present, loss state negligence regime, `CDC`, time-aware `scenario_historical_dispute_rate` computed on the training window only, evidence vector.
- **Payable buckets in the POC:** `0%` and `100%`. Intermediate buckets route the liability decision to micro-touch (the split is confirmed by a human; everything else stays automated). This is a deliberate POC simplification; comparative splits can be automated in Phase 2 once the agent output is available.
- **Phase 2:** the agent's liability % becomes a feature and the target becomes `|agent_pct − final_pct| ≤ 5pp`, which is the original meta-validator design.

### 7.3 G3 — Damage and Payment Gate

- **Decision input:** initial CCC estimate (repair amount, or TL flag with ACV), appraisal method.
- **Target:** `Y = 1` if (a) the initial repair/TL designation matched the final designation and (b) the initial estimate (repair amount or ACV) was within ±10% of the final paid amount **attributable to the initial estimate version** (supplements excluded) with no desk-adjuster price override.
- **Features:** `ccc_amount`, `ccc_tl_flag`, `ccc_acv`, `repair_to_acv_ratio`, `daily_tl_prob`, `tl_disagreement = |1[ccc_tl] − daily_tl_prob|`, `appraisal_method`, `damage_narrative_consistency` (CCC panels vs. `point_of_impact`), vehicle age and make tier, `drivable`, `airbag_deployed`, estimate version count to date.
- **Trained on:** the full 300k structured history (transcript-derived features missing outside the 8-month window).

### 7.4 Deterministic indemnity engine

Pure Python; no model involvement.

```
First-party (A1, A2):
  base   = ccc_repair_amount            if repairable
         = ccc_acv                      if total loss
  payout = min(coverage_limit, max(0, base - deductible))
  (TL sales tax / fees applied per state table if available; otherwise the TL file continues
   to the human TL unit after indemnity authorization — out of scope, not a touch)

Third-party PD (B):
  base   = claimant repair amount or claimant ACV
  payout = min(pd_limit, base × insured_liability_pct)    if payable under state rule, else 0 / escalate
  no deductible
```

### 7.5 State negligence rule for third-party payment

| Regime | Claimant barred when | Insured liability required to pay |
| --- | --- | --- |
| Pure comparative | never | > 0% |
| Modified comparative, 50% bar | claimant fault ≥ 50% | > 50% |
| Modified comparative, 51% bar | claimant fault > 50% | ≥ 50% |
| Contributory | claimant any fault | 100% |

The regime table is a config file keyed by loss state. In the POC, with payable buckets `{0%, 100%}`, B claims pay at `100%` and never pay at `0%`; intermediate buckets go to micro-touch.

### 7.6 Secondary models (P1)

- **Reserve severity:** LightGBM Tweedie regressor on `final_paid` with FNOL features and, post-epoch-1, `daily_tl_prob` and `ccc_amount`; alert when predicted severity exceeds the rule reserve by a configurable factor. Not a payment gate; informs reserving only.
- **Subrogation propensity refit:** existing LightGBM model refit with `L_hat`, `adverse_party_identified`, `police_report_available`, payout amount, state regime; target `recovered > 0`. Triggered on A2 claims when `L_hat < 100%`.
- **Endorsement verifier:** LightGBM binary on `Y = 1` if the Endorsement Agent's deductible and coverage modifiers matched the final applied values; features are the agent's outputs, `endorsement_ambiguity_flag`, form codes, state.

---

## 8. Counterfactual Simulation Engine

### 8.1 Purpose

A static model evaluation cannot answer: "What STP rate and error do we get if we decide at Epoch 0 versus wait for the CCC estimate, the insured's account, or a police report, and for how long is waiting worth it, per segment?" The simulator replays every holdout claim through the decision policy under alternative evidence policies and timing parameters.

### 8.2 Inputs

- Holdout claims with full event timelines from `df_daily_append` (actual `insert_ts` per evidence item).
- Trained gates G1–G3 and the Evidence Need Agent outputs, **precomputed** per claim per distinct information state (the state changes only on evidence arrival, so LLM calls are bounded by the number of arrival events, not by days).
- Empirical arrival-time distributions per evidence type, by state and channel.
- Policy parameters (dials) below.
- Deterministic indemnity engine and state regime table.

### 8.3 Dials

| Dial | Values | Meaning |
| --- | --- | --- |
| `τ = (τ_cov, τ_liab, τ_dmg)` | shared sweep 0.80–0.99 with per-gate offsets | gate thresholds |
| Evidence policy `E` | E0 current rules (backtest); E1 FNOL + CCC only; E2 E1 + Evidence Need Agent ordering; E3 E1 + always order police report for 2+ vehicles | what gets ordered and awaited |
| `T_max` | {0, 2, 4, 6, 10} days, settable per segment | holding-queue patience |
| `min_arrival_prob` | {0.5, 0.7, 0.9} | wait only if pending evidence is likely to arrive in time |
| Granularity `L` | full touchless vs. micro-touch allowed | whether a single human confirmation is permitted |

### 8.4 Counterfactual evidence that was never collected

If policy `E` orders evidence that claim `i` never received historically, its content is unobservable. The simulator reports **bounds**:

- **Lower bound (uninformative):** arrival time drawn from the empirical distribution; gate scores unchanged on arrival. Captures the pure cost of waiting.
- **Upper bound (fully informative):** arrival reveals the final adjudicated liability bucket; G2 clears if that bucket is payable. Captures the maximum benefit.
- **Conditional estimate:** policy evaluated only on claims that historically had the evidence (biased toward complex claims; reported as a sensitivity).

Executive outputs show the lower and upper bound as a band; the decision to deploy an evidence policy is made on the lower bound.

### 8.5 Replay algorithm

```
for policy π in grid:
    for claim i in holdout:
        d = 0; state = OPEN; touches = 0; ordered = {}
        while state == OPEN:
            I = info_state(i, d, ordered, π)          # actual rows + simulated arrivals for ordered items
            if guardrail(I):            state = EXIT_SPECIALTY; break
            cov, p_cov = G1(I); L, p_liab = G2(I); dmg, p_dmg = G3(I) if has_ccc(I) else (None, 0)
            if pay_condition(arch(i), cov, p_cov, L, p_liab, dmg, p_dmg, π.τ):
                paid_sim = indemnity(arch(i), I, L); state = PAID; break
            need = evidence_need(I)                    # precomputed
            ordered |= π.E.order(need, I, d)           # may add simulated arrivals
            pending = inflight(I, ordered)
            failing = failing_gates(...)
            if not pending or P_arrive(pending, π.T_max - d) < π.min_arrival_prob or d == π.T_max:
                if π.L == micro_touch and failing == {liability}: touches = 1; state = PAID_MICRO
                else: state = ESCALATED(failing)
                break
            d += 1
        record(i, π, state, d, touches, paid_sim, paid_actual_initial_version, appraiser_dispatched, queue_days)
```

### 8.6 Mandatory validation runs

1. **Backtest (E0):** replay the current CDC rule engine. Must reproduce the historical ~20% STP, cycle-time distribution, and paid amounts within tolerance (±2pp STP). If it does not, the simulator is wrong and nothing else is reported.
2. **Oracle:** replace gates with the true labels. Gives the ceiling STP under each evidence policy and separates model shortfall from structural ceiling. This is the gap analysis toward 80%.
3. **Evidence value check:** on claims with historical police reports/statements, measure liability-decision change rate before vs. after arrival, by Evidence Need Agent category.

### 8.7 Outputs

- **Pareto fronts:** STP (three definitions) vs. decision reversal rate; STP vs. net dollar impact; STP vs. P90 cycle time — one per evidence policy and `T_max`.
- **Decision Timing Table:** per segment, share decidable at d = 0, 1, 2, 3, 5, 10, marginal error at each step, recommended `T_max` and evidence to await.
- **Evidence Value Table:** Δ liability stability from police report / statement by agent category and scenario.
- **Capacity view:** daily holding-queue occupancy and field-appraiser dispatch volume under each policy.
- **Uncertainty:** block bootstrap by FNOL week, B = 500, on the chosen operating points only; 95% intervals on STP and dollar metrics.

---

## 9. Validation, Metrics, and POC Acceptance Criteria

### 9.1 Calibration

Isotonic calibration on the calibration split for every gate (scikit-learn `FrozenEstimator` + `CalibratedClassifierCV`, or a direct `IsotonicRegression` wrapper). Report ECE and reliability diagrams on the out-of-time split; target ECE ≤ 0.03.

### 9.2 Business metrics (all via the simulator)

1. STP rate — desk-touchless, zero-human-cost, micro-touch — against the 80% goal.
2. Decision reversal rate — automated coverage and liability decision vs. `df_gold_audit` (primary) and vs. closed record (secondary).
3. Payment error — overpayment dollars (leakage) and underpayment dollars (customer harm), separately, against the initial-estimate-version paid amount; supplement deltas reported but neutral.
4. Cycle time — mean and P90 days to indemnity authorization, by archetype and appraisal method.
5. Human touches per 100 claims and field-appraiser dispatches per 100 claims.
6. Net financial impact — `N_new_STP × LAE_saved − overpayment − appraiser_cost − inference_cost − missed_subrogation`, with every input as an explicit assumption and a sensitivity table.
7. Wrongful denial rate — 0 by construction; verified by asserting no `PAID` state with `cov_hat ∉ {Cover, Partial}`.
8. Fairness — metrics 1, 3, 4 by driver age band and by channel; no routing feature may be age.

### 9.3 What "it works" means for the POC

The POC is successful if all of the following hold on the out-of-time split:

- Point-in-time reconstruction passes the leakage unit tests.
- Extractor 1 fields used by the gates reach F1 ≥ 0.85 on the gold set.
- G1–G3 calibrate to ECE ≤ 0.03.
- The simulator backtest reproduces the current 20% within ±2pp.
- The Decision Timing Table and Pareto fronts are produced with bootstrap intervals, and the lower-bound projection at the chosen error budget is reported honestly, whatever the number.

The POC does not pre-commit to an STP number. The 80% goal is the target the Pareto front and oracle run are measured against.

---

## 10. One-Month POC Plan

Workstreams run in parallel; weeks are checkpoints, not day-by-day tasks. **P0** items are required for the acceptance criteria; **P1** items drop first if a P0 slips.

### Week 1 — Data, labels, and prompts

- P0: Confirm access; build `df_fnol_snapshot`, `df_daily_append` extracts, closed outcomes with maturity flags; implement `info_state(i, d)` and its leakage tests.
- P0: Archetype assignment from reporter flag + extracted facts; state negligence config.
- P0: Labels v1 for G1–G3; measure open-at-evaluation share.
- P0: Kick off `df_gold_audit` with the claims team (adjudication takes the full month; first 200 by Week 3).
- P0: Extractor 1 and Evidence Need Agent prompts v1, tested on 200 transcripts; token/cost estimate.
- P1: Endorsement document indexing; reserve and subrogation refit data pulls.

### Week 2 — Extraction at scale and gates v1

- P0: Bedrock Batch extraction on a stratified ~30k transcripts (all archetypes, both appraisal methods, over-sample manual claims); schema validation; sanitizer on historical statements.
- P0: Train and calibrate G1–G3; feature importance review for leakage signatures.
- P0: Empirical arrival-time distributions per evidence type.
- P1: Endorsement Agent on a 1k sample; reserve Tweedie; subrogation refit.

### Week 3 — Simulator

- P0: Replay engine; E0 backtest against the current rule engine (gate: ±2pp); oracle run; policies E1–E3; counterfactual bounds; Decision Timing Table.
- P0: Extraction gold-set scoring; evidence value check for the agent categories.
- P1: Endorsement verifier gate.

### Week 4 — Tuning, uncertainty, readout

- P0: Threshold selection at the agreed error budget; bootstrap intervals; fairness cuts; ROI with stated assumptions.
- P0: Executive readout: Pareto fronts, Decision Timing Table, oracle gap to 80% by segment, evidence value, capacity view, and the production instrumentation list (Section 11).
- P1: Secondary model results appended.

### Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Coverage/Liability agents unavailable on history | G2 is a direct model; schema reserved for agent features in Phase 2 |
| Insured-account arrival not logged on B claims | Simulated from statement timestamps where present; flagged as a production logging requirement |
| Supplement versions not separable in CCC data | G3 target falls back to total paid with a wider ±15% band; reported as a limitation |
| ASR quality for senior callers | WER sample in Week 1; `missing_critical_fact_count` monitored by age band |
| Bedrock batch throughput | Submit Week 2 Monday; 30k cap; extraction on remaining claims is Phase 2 |
| Ground-truth noise | Gold audit as primary reference for reversal rate |

---

## 11. Path to Production and Governance

- **Shadow mode (minimum 6–8 weeks):** run the full loop on live claims with no actions taken; compare to adjuster decisions daily; monitor calibration drift and extraction schema violations.
- **Canary:** A1 claims with photo appraisal first; then A1 field appraiser; then A2; B last.
- **Instrumentation required before B goes live:** logged event for insured contact/account on claimant-reported claims; digital questionnaire delivery and response events; police report request/receipt timestamps.
- **Agent integration:** Coverage Agent and Virtual Handler outputs added as G1/G2 features once batch-runnable; gates retrained as meta-validators.
- **Model risk management:** model inventory entries for G1–G3 and the Evidence Need Agent prompt; validation report; champion/challenger; monthly fairness report by age band; prompt-pay statute checks on `T_max` per state.
- **Regulatory posture:** no automated denials; adverse decisions always human; audit trail per epoch retained.

---

## 12. Repository Structure and Coding Guardrails

```
abc_auto_stp_poc/
  config/
    model_params.yaml            # LightGBM/CatBoost params, tau defaults, T_max per segment
    extraction_schemas.json      # Pydantic/JSON schemas for all Sonnet prompts (versioned)
    state_negligence.yaml        # loss state -> regime
    prompts/                     # extractor_fnol.md, extractor_statement.md, sanitizer.md, evidence_need.md, endorsement.md
  src/
    data_pipeline/
      snapshot_builder.py        # info_state(i, d): FNOL snapshot + append rows with insert_ts <= fnol_ts + d
      archetypes.py              # A1 / A2 / B assignment
      target_engineering.py      # Y_cov, Y_liab_bucket, Y_dmg, maturity flags
      arrival_distributions.py   # empirical evidence arrival by type/state/channel
    llm/
      bedrock_client.py          # Sonnet 4.6+ batch/async client with schema validation
      extractor_fnol.py
      extractor_statement.py     # includes sanitizer
      evidence_need_agent.py     # rule overrides + LLM categorical output
      endorsement_agent.py       # P1
    models/
      train_gates.py             # G1, G2, G3
      calibration.py             # isotonic wrappers, ECE, reliability plots
      secondary_models.py        # reserve Tweedie, subro refit, endorsement verifier (P1)
    decision/
      policy.py                  # Section 3.3 loop; pay_condition by archetype
      indemnity.py               # deterministic payout; state regime
    simulation/
      replay.py                  # Section 8.5
      policies.py                # E0..E3, dials
      counterfactual.py          # lower/upper bound evidence handling
      backtest.py                # E0 vs history; oracle run
      metrics.py                 # three STP rates, two-sided payment error, cycle time, touches, fairness
      bootstrap.py               # block bootstrap by FNOL week
      plots.py                   # Pareto fronts, Decision Timing Table, capacity view
  tests/
    test_no_leakage.py           # asserts no feature insert_ts > epoch
    test_indemnity.py
    test_backtest_tolerance.py
  notebooks/
    poc_master_run.ipynb
```

### Coding guardrails

1. **No LLM probabilities for gating.** Prompts return categorical/boolean fields and `reason_codes` only. Any `"confidence"` field in a schema is a bug.
2. **Deterministic money.** `indemnity.py` is pure arithmetic with unit tests for every archetype × repair/TL × regime combination.
3. **Point-in-time everywhere.** Every feature builder takes `(claim_id, epoch_day)` and reads through `snapshot_builder.info_state`. No direct reads of `df_closed_outcomes` outside `target_engineering.py` and `metrics.py`.
4. **Field appraiser is a wait state, not a fallback.** `appraisal_method == 'FIELD'` keeps the claim in the holding queue awaiting `ccc_estimate`.
5. **Third-party payment requires liability.** `pay_condition` for archetype B must reference `p_liab`, `L_hat`, and the state regime; a unit test asserts no B claim pays at `L_hat == 0%`.
6. **Escalate the decision, not the claim.** `ESCALATED(failing)` carries the failing gate(s) so locked decisions are not re-worked by the human.
7. **Age is not a feature.** `driver_age_band` is permitted only in `metrics.py`.
8. **Model/prompt version pinning.** Every gate artifact records the extraction schema version and Sonnet model ID used to build its training features.

---

## 13. Appendix

### 13.1 Changes from v1 and why

| v1 | v2 | Reason |
| --- | --- | --- |
| Day-0 / Day 1–5 step list | Scoring epochs + per-epoch decision policy + simulator-derived Decision Timing Table | Timing is a claim-lifecycle question to be optimized, not a schedule |
| Model 1 (composite stability) + Model 4 + Model 5 | Three per-decision gates G1–G3 scored at every epoch | v1 label was circular (used an agent output that was also a feature), required `Paid > 0`, and was the AND of the other models; per-decision gates enable micro-touch routing |
| Coverage Agent / Virtual Handler as POC features | Direct liability model; agents as Phase 2 features | Agents cannot be batch-run on history |
| First-party formula with a 3rd-party line | Archetypes A1/A2/B with explicit third-party payment rule and state negligence regimes | Pay the claimant only if the insured is liable |
| Police report ordered by rule (`≥ 3 days`) | Evidence Need Agent with rule overrides; arrival from empirical distributions | Clear liability needs no report |
| TL treated like repair | TL indemnity via ACV; salvage/title out of scope | Per scope decision |
| ±10% vs. final paid | ±10% vs. initial-estimate-version paid | Pay-initial supplement policy |
| 1-week day-by-day plan | 4-week P0/P1 workstreams with de-scoping rule | 1-month POC |
| Claude 3.5 / 4.6 mixed | Sonnet 4.6+ throughout | |
| One-sided leakage, no backtest, no oracle | Two-sided payment error, mandatory backtest, oracle gap analysis, counterfactual bounds | Simulator must be validated and honest about unobserved evidence |
| `driver_age_band` as a feature | Fairness monitoring dimension only | Senior customer base; disparate-impact risk |

### 13.2 Assumptions to confirm in Week 1

- CCC data distinguishes estimate versions so supplements can be excluded from the G3 target.
- CCC TL valuations include ACV in `df_daily_append`.
- Claimant-reported claims carry the insured's policy identifier at FNOL (or a policy-match event exists).
- Final liability % and final coverage status are populated on closed claims for all archetypes.
- Historical Daily TL scores are stored with their scoring date.

### 13.3 Open items the business must supply

- LAE saved per touchless claim (desk vs. field), field-appraiser cost, Bedrock cost tolerance.
- Acceptable decision-reversal and payment-error budget at which the STP rate is read off the Pareto front.
- Prompt-pay deadlines by state to bound `T_max`.
- Which states' comparative-negligence splits the business wants automated in Phase 2.
