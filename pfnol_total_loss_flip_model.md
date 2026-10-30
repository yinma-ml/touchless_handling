# Technical Specification & Experiment Plan: Stage-2 pFNOL Total Loss (Supplement Flip) Model

**System Context:** Downstream pipeline step executing after Stage-1 Coverage Applicability (`M_cov`) and FNOL Initial Triage (CCC PMOI + Internal FNOL TL Model).  
**Trigger:** Arrival of Initial CCC Estimate (`E0`) and Inspection/Shop Photos prior to teardown or Supplement 1 (`S1`).  
**Objective:** Predict whether an exposure appearing repairable at `E0` will flip to a Total Loss by first exposure close date due to hidden teardown damage or supplement growth.

---

## 1. Pipeline Dependency, Data Volume, and Split Inheritance Constraints

### 1.1 Split Inheritance & Leakage Prevention (40 / 30 / 30 Split)

- **Total Population:** ~1,000,000 total historical exposures; filtered to covered (`Y1 == 1`), mature first-party Collision/Comprehensive exposures with `E0` estimates (~40,000 target training exposures in the active modeling sub-cohort).
- **Strict Split Inheritance:** Do **not** re-split data for Stage-2. Inherit the exact pre-computed `split_group` (`TRAIN = 40%`, `VAL_CALIB = 30%`, `TEST = 30%`) grouped by `(policy_root_id, loss_event_id, vehicle_id)` from the master pipeline.
- **Upstream Feature Leakage Rules:**
  1. **FNOL Internal TL Score (`fnol_internal_tl_prob`):** Must be populated in `TRAIN` using 5-fold Out-of-Fold (OOF) predictions fit strictly within the 40% `TRAIN` split. Populate `VAL_CALIB` and `TEST` using the model fit on `TRAIN`.
  2. **Historical Shop/Estimator Supplement Priors (`shop_hist_supp_multiplier`):** Fit Empirical Bayes (m-estimate) target statistics strictly on `TRAIN` (`E0` vs. final supplement totals). Join as a static lookup to `VAL_CALIB` and `TEST`. Never compute shop supplement history across the full dataset.
  3. **Estimate Stage Lock (`E0` Only):** Features must be extracted strictly from the **First CCC Estimate (`E0`, Supplement Sequence `0`)** and photos uploaded at or before `E0` completion timestamp. Exclude any line items, photos, or valuations updated during `S1+`.

---

## 2. Target Definition & Cohort Filtering ("The Flip Zone")

### 2.1 Entry Gate: Defining the Eligible pFNOL Flip Cohort

At pFNOL (`E0` arrival), deterministic rules first calculate the state-specific Constructive Total Loss dollar ceiling (`State_CTL_$`):

- **Statutory Percentage States:** `State_CTL_$ = State_Threshold_Pct * ACV_E0`
- **Total Loss Formula (TLF) States:** `State_CTL_$ = min(State_Threshold_Pct * ACV_E0, ACV_E0 - Expected_Salvage_E0)`

| Cohort Bucket | Entry Rule at `E0` | Pipeline Routing | Included in `M_tl_flip` Training? |
| :--- | :--- | :--- | :--- |
| **1. Deterministic Total Loss** | `E0_Total >= State_CTL_$` | Auto-route to Total Loss | **No** (Already totaled by rule) |
| **2. Trivial Cosmetic Repair** | `(E0_Total / State_CTL_$) < 0.20` AND `fnol_ccc_cat == 'Repairable'` AND `fnol_internal_tl_prob < 0.25` | Auto-route to Repair STP | **No** (`< 0.1%` flip rate; exclude to prevent tree dilution) |
| **3. Active Supplement Flip Zone** | `0.20 <= (E0_Total / State_CTL_$) < 1.00` **OR** (`fnol_ccc_cat in ['BTL', 'PTL']` OR `fnol_internal_tl_prob >= 0.25`) | **Score with `M_tl_flip`** | **Yes (Target Modeling Cohort)** |

### 2.2 Target Definitions (Evaluated as of First Exposure Close Date)

| Target ID | Target Name | Type | Ground Truth Calculation Logic |
| :--- | :--- | :--- | :--- |
| **`Y_flip`** | `is_total_loss_at_close` | Binary `{0, 1}` | **`1` (Flipped to Total Loss):** Exposure was `< State_CTL_$` at `E0`, and `Total_Loss_Indicator == 1` as of first exposure close date.<br>**`0` (Remained Repairable):** `Total_Loss_Indicator == 0` at first exposure close date (repaired and closed). |
| **`Y_supp_ratio`** | `log_supp_growth_ratio` | Float `[0, inf)` | _Auxiliary regression target for multi-task validation:_<br>`log( max(Final_Repair_Plus_Supp_Amount, E0_Total) / max(E0_Total, 500.0) )` |

---

## 3. Feature Engineering Specification (Available at `E0` + Photo Arrival)

All features are fit on the 40% `TRAIN` split and transformed on `VAL_CALIB` (30%) and `TEST` (30%).

| Feature Group | Feature Name | Source | Type | Missing Strategy | Formula / Extraction Logic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. FNOL Cascade Priors** | `fnol_ccc_pmoi_cat` | CCC FNOL | Cat | Fill `"UNKNOWN"` | Initial FNOL CCC classification: `Repairable`, `BTL` (Borderline), `PTL` (Potential TL). |
| | `fnol_internal_tl_prob` | Internal FNOL | Float `[0,1]` | `NaN` (Native) | OOF calibrated probability from in-house FNOL model across all claims. |
| | `fnol_internal_btl_override` | Internal FNOL | Cat | Fill `"NOT_RUN"` | Category assigned when internal model evaluates `BTL` cohort (`Escalated_PTL`, `Confirmed_BTL`, `Downgraded_Repairable`, `Not_Run`). |
| | `fnol_to_e0_severity_divergence` | FNOL + `E0` | Float | `NaN` (Native) | `fnol_internal_tl_prob - (E0_Total / State_CTL_$)`. High positive value flags an under-written `E0` relative to FNOL impact physics. |
| **B. State Threshold & `E0` Economics** | `normalized_ctl_proximity` | `E0` + State Table | Float `[0,1]` | `NaN` (Native) | `E0_Total / State_CTL_$`. Normalizes statutory rules across all 50 states onto `[0, 1]`. |
| | `ctl_dollar_headroom` | `E0` + State Table | Float | `NaN` (Native) | `max(0.0, State_CTL_$ - E0_Total)`. Exact dollar supplement required to trigger a Total Loss. |
| | `vehicle_acv_e0` | CCC Valuation | Float | `NaN` (Native) | Initial Actual Cash Value (`ACV_E0`) at `E0`. Low-ACV older vehicles flip on minor single-part supplements. |
| | `vehicle_age_years` | VIN / PAS | Int | `NaN` (Native) | `Year(DOL) - Vehicle_Model_Year`. |
| **C. CCC `E0` Parts & Labor Anatomy** | `e0_frame_structural_hrs` | CCC `E0` Lines | Float | Fill `0.0` | Sum of `Frame` + `Structural_Body` labor hours on `E0`. |
| | `e0_mech_elec_labor_share` | CCC `E0` Lines | Float `[0,1]` | Fill `0.0` | `(Mechanical_Labor_$ + Electrical_Labor_$) / max(Total_Labor_$, 1.0)`. |
| | `e0_oem_parts_dollar_share` | CCC `E0` Lines | Float `[0,1]` | Fill `0.0` | `OEM_Parts_$ / max(Total_Parts_$, 1.0)`. High OEM share indicates lack of cheap LKQ/Aftermarket alternatives during teardown. |
| | `e0_replace_to_repair_ratio` | CCC `E0` Lines | Float | Fill `0.0` | `Replace_Line_Count / max(Repair_Line_Count + Replace_Line_Count, 1)`. |
| | `e0_sublet_adas_missing_flag` | CCC `E0` + VIN | Int `{0,1}` | Fill `0` | `1` if `Vehicle_Model_Year >= 2019` AND `POI` is Front/Rear AND `Sublet/Calibration_Lines == 0` on `E0` (signals unpriced ADAS calibration). |
| **D. Historical Shop & Channel Priors** | `e0_author_channel` | CCC Metadata | Cat | Fill `"UNKNOWN"` | `Customer_Photo_Guided`, `DRP_Shop`, `Non_DRP_Independent`, `Staff_Appraiser`. |
| | `shop_hist_supp_multiplier` | `TRAIN` Lookup | Float | Fill `State_Mean` | Out-of-fold Empirical Bayes smoothed ratio: `(n * mean(Final_$/E0_$) + 20 * State_Mean) / (n + 20)` grouped by `Shop_ID` (fit strictly on 40% `TRAIN`). |
| | `shop_hist_flip_rate` | `TRAIN` Lookup | Float `[0,1]` | Fill `State_Mean` | Out-of-fold smoothed historical flip rate (`Y_flip == 1`) for the authoring shop/channel (fit strictly on 40% `TRAIN`). |
| **E. CCC POI & Image Caption Metadata** | `poi_structural_risk_tier` | CCC POI | Cat | Fill `"UNKNOWN"` | Mapped from CCC Point of Impact: `High_Front_Corner_11_01`, `High_Front_Center_12`, `High_Rollover_Undercarriage`, `Med_Rear_Side_Pillar`, `Low_Cosmetic_Door_Glass`. |
| | `poi_photo_concentration` | CCC Image Text | Float `[0,1]` | `NaN` (No Photos) | `Count(Photos tagged with POI zone) / Count(Non-Admin Photos)`. |
| | `missing_underhood_photo_flag` | CCC Image Text | Int `{0,1}` | Fill `0` | `1` if `POI` is Front (`11, 12, 01`) AND zero photo descriptions contain `["underhood", "engine", "radiator", "core", "inner"]` (signals jammed hood). |
| | `damage_photo_edge_density_max` | OpenCV / MobileNet | Float | `NaN` (No Photos) | Max central-crop Canny edge density / MobileNet damage score across top candidate damage photos. |
| **F. VLM Forensic Estimate Audit** | `vlm_unwritten_major_damage` | VLM Grid Audit | Int `{0,1,2}` | Fill `-1` (Skipped) | `0=Aligned with E0`, `1=Minor blend/trim missing`, `2=Visible major structural/suspension/LED/cooling damage in photos omitted from E0 lines`. |
| | `vlm_jammed_panel_hidden_crush` | VLM Grid Audit | Int `{0,1}` | Fill `-1` (Skipped) | `1` if hood/trunk/door is visibly buckled shut in photos hiding internal rail/apron crush. |
| | `vlm_unibody_seam_transfer` | VLM Grid Audit | Int `{0,1,2}` | Fill `-1` (Skipped) | `0=Outer skin only`, `1=Adjacent panel gap pinched`, `2=Buckle transferred to A/B-pillar, roof rail, or tilted wheel camber`. |
| | `vlm_upd_or_structural_rust` | VLM Grid Audit | Int `{0,1}` | Fill `-1` (Skipped) | `1` if photos show pre-existing rust rot or unrelated prior damage complicating repair. |

---

## 4. Two-Stage Photo Filtering & VLM Grid Architecture (Handling ~30 Photos/Exposure)

To process ~30 photos per exposure across ~40k training claims without prohibitive cost or blocked packages (no YOLO/AGPL), implement a 3-step funnel:

1. **Step 1 — Regex Text Filter on CCC Image Descriptions (`< 1 ms` CPU):**  
   Drop administrative photos where CCC caption matches `r"(?i)(vin|odom|mile|regis|plate|label|jamb|paper|doc|estimate|tread|key)"`.
2. **Step 2 — Lightweight CPU Ranking (`OpenCV` + `torchvision.models.mobilenet_v3_small`, `~30 ms` CPU):**
   - Deduplicate remaining photos via 8x8 Perceptual Hash (`Hamming_Distance <= 6`).
   - Filter out blurry images (`Laplacian_Variance < 80.0`).
   - Score remaining images using frozen `mobilenet_v3_small` embedding norm + central-crop Canny edge density + CCC POI text match boost. Select the **Top 6 Damage Photos**.
3. **Step 3 — `2x3` Grid Stitch & Single VLM Audit Call:**
   - Resize the Top 6 photos to `512x512`, overlay their CCC text caption on a top banner, and stitch into a single `1024x1536` JPEG grid.
   - Pass the **1 Grid Image + CCC `E0` Line Summary + FNOL Narrative** to the enterprise LLM (Batch API for training backfill) with `temperature = 0.0` to extract the 4 integer features in **Feature Group F**.

---

## 5. Experiment Design, Ablation Matrix, and Test Plan

### 5.1 Feature Ablation Matrix (Modular CLI Execution)

Following the modular pipeline standard, evaluate four progressive feature sets on the exact same **40% Train / 30% Val-Calib / 30% Test** splits to prove the incremental lift of each modality before enabling VLM calls in production:

| Experiment ID | Feature Groups Included | Purpose & Go/No-Go Criteria |
| :--- | :--- | :--- |
| **`Exp_F0` (Baseline)** | **Groups A + B** _(FNOL CCC/Internal Priors + State CTL Proximity)_ | Baseline benchmark. Tests whether simple economic proximity (`E0 / State_CTL_$`) + FNOL scores suffice. |
| **`Exp_F1` (`E0` Anatomy)** | **Groups A + B + C + D** _(Adds CCC `E0` Parts/Labor Ratios + OOF Shop Priors)_ | Quantifies lift from parsing CCC line-item structure and shop supplement history (zero image cost). |
| **`Exp_F2` (Zero-LLM Photo Meta)** | **`Exp_F1` + Group E** _(Adds CCC POI + Caption Metadata + OpenCV/MobileNet stats)_ | Tests how much visual signal is captured purely from CCC image captions (`missing_underhood_photo_flag`) and fast local CPU image stats. |
| **`Exp_F3` (Full VLM Audit)** | **`Exp_F2` + Group F** _(Adds 2x3 Grid VLM Forensic Audit features)_ | Must demonstrate statistically significant lift (`>= +0.025 PR-AUC` or `>= +5% Recall at Fixed FP Rate` on `TEST`) over `Exp_F2` to justify production VLM latency/cost. |

### 5.2 Calibration & Asymmetric Cost Threshold Tuning (On 30% `VAL_CALIB`)

1. **Model & Calibration:** Train a `K = 10` CatBoost ensemble on `TRAIN` (40%). Fit **Isotonic Regression** calibrators on `VAL_CALIB` (30%) to produce calibrated flip probability `mu_flip` and conservative upper risk bound `R_tractable = min(1.0, mu_flip + 1.645 * sigma_flip)`.
2. **Economic Cost Optimization:** Define asymmetric unit costs:
   - `Cost_FN` (Missed Total Loss Flip that goes to teardown/supplement): `$1,650` (teardown labor + shop storage + 14 days extra rental bleed).
   - `Cost_FP` (Repairable vehicle held for 24-hr desk/virtual teardown review): `$220` (appraiser review time + 1 day rental delay).

   Tune optimal threshold `tau_intercept` on `VAL_CALIB` to maximize **Net Dollar Savings** over deterministic rules.

### 5.3 Mandatory Validation & Test Plan (Evaluated on Locked 30% `TEST`)

The developer must generate a validation report on the 30% `TEST` split covering:

1. **Discrimination Metrics (Overall & BTL Sub-Cohort):**
   - **PR-AUC (Precision-Recall AUC)** and **ROC-AUC** across the full Flip Zone and specifically isolated on the **`fnol_ccc_pmoi_cat == 'BTL'`** sub-cohort.
   - **Recall @ 20% and 35% Precision:** Percentage of hidden supplement flippers intercepted at `E0` before teardown.
2. **Calibration Integrity:**
   - **Expected Calibration Error (ECE) `< 0.015`** and **Brier Score** reduction post-isotonic calibration across Statutory % states vs. TLF states.
3. **Counterfactual Economic Backtest:**
   - Calculate `Net_Savings_$ = (True_Positives * $1,650) - (False_Positives * $220)` and **Saved Rental Days** on the 30% `TEST` set (300k total exposures -> ~12k Flip-Zone test exposures).
4. **Leakage Sanity Checks (Automated CI Assertions):**
   - Assert zero overlap of `policy_root_id`, `loss_event_id`, or `vehicle_id` across `TRAIN`, `VAL_CALIB`, and `TEST`.
   - Assert `E0_Timestamp < First_Supplement_Timestamp` for 100% of scored test rows.
