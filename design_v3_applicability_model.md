# Technical Specification: Real-Time FNOL Coverage Applicability & Prefill Verification Model

**Scope:** First-Party Auto Claims POC (Collision, Comprehensive, Towing/ERS, Rental)  
**Trigger:** Real-time execution at First Notice of Loss (FNOL) assuming a matched policy (~95% cohort).  
**Objective:** Automate manual claim handler verification of policy prefill and predict coverage applicability without running full-document LLM screening on all policy contracts.

---

## 1. Targets, Feature Groups, Sources, and Calculations

### 1.1 Target Definitions (Exposure Level: Vehicle x Coverage)

- **Maturation Rule:** Exclude claims reported within 90 days of the training cutoff date unless already closed with a final adjudication code.
- **Censoring Rule:** Exclude claims closed as _Withdrawn/Abandoned prior to coverage determination_.

| Target ID | Target Name              | Data Type       | Ground Truth Calculation Logic                                                                                                                                                                                                                                                                                                                                                                 |
| :-------- | :----------------------- | :-------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`Y1`**  | `is_coverage_applicable` | Binary `{0, 1}` | **`1` (Covered):** Exposure closes with `Paid > $0` OR closure reason `Closed Without Payment (CWP) - Below Deductible` OR `CWP - Insured Pursued 3rd Party (Coverage Verified)`.<br>**`0` (Denied):** Exposure closes with formal denial code (`No Coverage on DOL`, `Excluded Driver`, `Excluded Peril/Endorsement`, `Lapse/Out of Force`, `Material Misrepresentation`).                    |
| **`Y2`**  | `is_prefill_exact`       | Binary `{0, 1}` | _Evaluated conditionally where `Y1 == 1`._<br>**`1` (Exact):** Initial FNOL PAS prefill tuple `(Coverage_Code, Per_Exposure_Ded, Per_Occurrence_Ded, Limit, Disappearing_Earned_Ded_Credit)` equals final adjudicated financial terms at closure.<br>**`0` (Modified):** Handler manually overrode coverage code, switched COLL <-> COMP, altered deductible/waiver, or remapped vehicle/term. |

---

### 1.2 Feature Engineering Specification (Strictly Available at T=0 FNOL)

_Note on Name Matching Consolidation:_ Driver matching is condensed into **2 representative features** (`driver_roster_match_tier` and `max_driver_exclusion_risk_score`) combining LLM nickname normalization, Jaro-Winkler, Double Metaphone, and suffix conflict logic upstream in Python.

| Group                                | Feature Name                      | Source           | Type           | Missing Value Strategy         | Calculation / Extraction Logic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| :----------------------------------- | :-------------------------------- | :--------------- | :------------- | :----------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Policy Match & Term Integrity** | `pas_match_method`                | System Log       | Cat            | Fill `"UNKNOWN"`               | Method used to link FNOL to PAS: `Exact_Pol_Num`, `SSN_DOB`, `Name_Address_Fuzzy`, `Manual_CSR`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|                                      | `dol_term_boundary_days`          | PAS + FNOL       | Float          | `NaN` (Native GBDT)            | `min(abs(DOL - Term_Start), abs(DOL - Term_End))` in days.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|                                      | `policy_term_version`             | PAS              | Cat            | Fill `"UNKNOWN"`               | `6_Month_Term` vs. `12_Month_Term` product architecture flag.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|                                      | `recent_endorsement_days`         | PAS              | Float          | `NaN` (= No Endorsement)       | `DOL - Last_Mod_Date` (coverage added, deductible lowered, or vehicle swapped prior to DOL). Leave `NaN` if no mid-term modification exists.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|                                      | `billing_risk_state`              | PAS Billing      | Cat            | Fill `"CURRENT"`               | As-of DOL billing status: `Current`, `In_Grace_Period`, `Pending_Cancel_NonPay`, `Reinstated_Last_30D`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|                                      | `vin_match_tier`                  | PAS + FNOL       | Cat            | Fill `"UNVERIFIED"`            | `Exact_VIN`, `Year_Make_Model_Match`, `Newly_Acquired_Unlisted`, `Temp_Rental_Sub`, `Mismatch`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **B. Exposure & Deductible Prefill** | `multi_car_occurrence_flag`       | FNOL             | Int            | Fill `0`                       | `1` if `>= 2` insured vehicles are claimed under the same occurrence; `0` otherwise.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|                                      | `special_ded_feature_code`        | PAS              | Cat            | Fill `"STANDARD"`              | `Disappearing_Ded`, `Earned_Ded`, `Glass_Full_Waiver`, `Collision_Ded_Waiver_CDW`, `Standard`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|                                      | `coverage_asymmetry_code`         | PAS              | Cat            | Fill `"NONE"`                  | Vehicle coverage profile: `Full_COMP_COLL`, `COMP_Only`, `Liability_Only`, `Storage_Comp_Only`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|                                      | `deductible_to_acv_ratio`         | PAS + VinBook    | Float          | `NaN` (Native GBDT)            | `Prefilled_Deductible / Estimated_Vehicle_ACV`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **C. Channel & Reporter**            | `intake_channel`                  | FNOL             | Cat            | Fill `"UNKNOWN"`               | `Phone_CSR`, `Web_Self_Service`, `Mobile_App`, `Third_Party_API`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|                                      | `reporter_persona`                | FNOL + LLM       | Cat            | Fill `"UNKNOWN"`               | `Named_Insured`, `Spouse`, `Unlisted_Driver`, `Body_Shop`, `Rental_Tow_Vendor`, `Attorney`, `Third_Party`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|                                      | `fnol_reporting_lag_hrs`          | FNOL             | Float          | `NaN` (Native GBDT)            | `(FNOL_Timestamp - DOL_Timestamp)` in hours. Cap lower bound at `0.0`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **D. Consolidated Driver Match**     | `driver_roster_match_tier`        | PAS + FNOL + LLM | Cat            | Fill `"NO_DRIVER_PARKED"`      | Upstream Python rule combining LLM-normalized first name (root + nickname variants), last name, suffix, and DOB against PAS Listed Drivers:<br>• `Tier_1_Exact_Or_Phonetic`: JW `>= 0.92` or Double Metaphone match, no suffix/DOB conflict.<br>• `Tier_2_Fuzzy_Listed`: `0.80 <= JW < 0.92`, no suffix conflict.<br>• `Tier_3_Suffix_Or_DOB_Conflict`: Name matches (`JW >= 0.85`) but Jr./Sr. or DOB mismatches.<br>• `Tier_4_Unlisted_Household`: Last name JW `>= 0.90` or same address, first name JW `< 0.60`.<br>• `Tier_5_Unlisted_External`: Both first and last name JW `< 0.60`, different address.<br>• `No_Driver_Parked`: Vehicle was parked/unoccupied. |
|                                      | `max_driver_exclusion_risk_score` | PAS + FNOL + LLM | Float `[0, 1]` | `NaN` (= No NDE on Policy)     | If policy has a Named Driver Exclusion (NDE) endorsement, compute `max(JaroWinkler, 0.95 * Metaphone_Match)` across raw and LLM-canonical driver names against all NDE names. Output `NaN` if policy has no excluded drivers.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|                                      | `llm_permissive_use_flag`         | LLM Fact         | Cat            | Fill `"NOT_STATED"`            | Extracted from transcript/notes: `Explicit_Permission`, `Implied_Family`, `Taken_Without_Permission`, `Shop_Valet_Custody`, `Not_Stated`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **E. Loss Peril & Exclusion Facts**  | `llm_true_peril_code`             | LLM Fact         | Cat            | Fill `"AMBIGUOUS"`             | `Veh_Collision`, `Fixed_Object`, `Pothole_Road_Debris`, `Animal_Strike`, `Weather_Hail_Flood`, `Glass_Only`, `Theft_Vandalism`, `Mechanical_Wear_Tear`, `Ambiguous`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|                                      | `peril_conflict_flag`             | FNOL + LLM       | Int            | Fill `-1` (If LLM `AMBIGUOUS`) | `1` if structured FNOL loss cause `!= llm_true_peril_code` (e.g., structured=`COMP`, narrative=`slid on ice hit tree`); `0` if aligned.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|                                      | `llm_exclusion_trigger_code`      | LLM Fact         | Cat            | Fill `"NONE_DETECTED"`         | Single consolidated exclusion trigger from text: `Commercial_Rideshare_Delivery`, `Wear_Tear_Mechanical`, `Racing_OffRoad`, `Out_Of_Territory_Mex`, `Intentional_Act`, `Aftermarket_Custom_Equip`, `None_Detected`.                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **F. State & Endorsement Rules**     | `jurisdiction_regime`             | PAS + FNOL       | Cat            | Fill `"STANDARD"`              | `MI_Broad_Coll`, `MI_Standard_Coll`, `MI_Limited_Coll`, `Out_Of_State_Loss_Deemer`, `Standard_In_State`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|                                      | `llm_fault_indicator`             | LLM Fact         | Cat            | Fill `"UNKNOWN"`               | `Clear_Not_At_Fault` (e.g., rear-ended, parked), `Clear_At_Fault`, `Single_Veh_At_Fault`, `Disputed_Comparative`, `Unknown`. _(Interacts with MI Collision tiers & CDW)._                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|                                      | `restrictive_form_count`          | PAS Forms        | Int            | Fill `0`                       | Count of active restrictive endorsement form IDs on policy (e.g., `Named_Operator_Only`, `Step_Down_Limit`, `Rideshare_Exclusion`, `ACV_Parts_Only`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

---

## 2. Model Architecture, Inputs, and Missing Value Handling

### 2.1 Upstream LLM Fact Extractor (Pre-Model Input Step)

- **Input:** Raw FNOL call transcript or web free-text notes (blind to PAS policy data to prevent confirmation bias).
- **Execution:** Deterministic schema-constrained JSON output (`temperature = 0.0`).
- **Output Fields:** `reported_driver_first`, `canonical_first_variants[]`, `reported_driver_last`, `driver_suffix`, `reporter_persona`, `llm_permissive_use_flag`, `llm_true_peril_code`, `llm_exclusion_trigger_code`, `llm_fault_indicator`.
- **Constraint:** Do not extract or use LLM token log-probs or LLM self-rated confidence scores.

### 2.2 Missing Value Strategy (Critical for FNOL Sparsity)

At FNOL, 30%–60% of claims will lack unstructured transcripts (e.g., terse web submissions), unlisted driver DOBs, or mid-term endorsements. **Do NOT use mean, median, or zero imputation for continuous features**, as `0` collides with true zero values (e.g., `dol_term_boundary_days = 0` means the crash happened on renewal day, whereas `NaN` means date unknown).

1. **Continuous Features (`Float`):** Pass raw `np.nan` directly into CatBoost/LightGBM. Configure tree splitters with `nan_mode='Min'` or `nan_mode='Max'` (learned optimal split direction per node).
2. **Categorical Features (`Cat`):** Never leave as `NaN` or drop rows. Map missing/unmentioned values to an explicit string token (`"UNKNOWN"`, `"NOT_STATED"`, or `"NO_DRIVER_PARKED"`). Missingness at FNOL is informative (Missing Not At Random) and must be a splittable category.
3. **Sparsity Indicator Feature:** Add a single integer feature `missing_critical_fnol_count` counting how many of `[VIN, Driver_Name, Narrative_Text, Exact_Loss_Location]` are missing at `T=0`.

### 2.3 Two-Stage Gradient Boosted Decision Tree (GBDT) Architecture

Use **CatBoostClassifier** (preferred for native target-statistics encoding of high-cardinality categorical combinations like `jurisdiction_regime` x `llm_fault_indicator`) or **LightGBM**.

- **Model 1 (`M_cov` — Coverage Applicability):**
  - **Target:** `Y1` (`is_coverage_applicable`)
  - **Training Population:** All mature matched-policy first-party exposures.
  - **Loss Function:** Logloss. If class weights are used during training to boost recall on `Y1 = 0`, post-hoc isotonic calibration (Section 3) is mandatory to restore true prior probabilities.
- **Model 2 (`M_prefill` — Prefill Verification):**
  - **Target:** `Y2` (`is_prefill_exact`)
  - **Training Population:** Subset of exposures where `Y1 == 1`.
  - **Loss Function:** Standard binary Logloss.

---

## 3. Tractable Probability & Confidence Calculation

To guarantee scientific, mathematically bounded probabilities and capture epistemic uncertainty (rare feature combinations where the model lacks historical support), implement a **Virtual Ensemble + Isotonic Calibration + Lower Confidence Bound** pipeline.

### 3.1 Dataset Splitting (Temporal Out-of-Time Split)

Do not use random K-fold across time (prevents policy-renewal leakage).

- **`D_train` (70%):** Oldest historical claims (minimum 24 months recommended to span full 6m/12m renewal cycles and seasonal CAT events).
- **`D_calib` (15%):** Subsequent 2–3 months of mature claims (strictly unseen during tree building; used _only_ for probability calibration and conformal quantile calculation).
- **`D_test` (15%):** Most recent mature claims (preceding the 90-day maturation buffer) for final evaluation.

### 3.2 Step-by-Step Mathematical Formulation

1. **Ensemble Training for Epistemic Uncertainty:**  
   Train a `K = 10` model ensemble (via 10 bootstrap/sub-sampled folds or CatBoost's native `VirtualEnsembles` method) for both `M_cov` and `M_prefill` on `D_train`.

2. **Post-Hoc Isotonic Calibration on `D_calib`:**  
   For each ensemble member `k in {1..K}`, fit a monotonic **Isotonic Regression** calibrator `g_1k(.)` for `M_cov` and `g_2k(.)` for `M_prefill` on `D_calib`.  
   For any new FNOL exposure `x`, the calibrated joint probability of valid coverage AND exact prefill for ensemble member `k` is:
   `p_hat_k(x) = g_1k(s_1k(x)) * g_2k(s_2k(x))`

3. **Point Estimate (`mu_p`) and Epistemic Uncertainty (`sigma_p`):**  
   Compute the empirical mean probability and standard deviation across the `K` calibrated ensemble predictions:
   - `mu_p(x) = (1 / K) * sum_{k=1..K} p_hat_k(x)`
   - `sigma_p(x) = sqrt( (1 / (K - 1)) * sum_{k=1..K} (p_hat_k(x) - mu_p(x))^2 )`

4. **Tractable Lower Confidence Bound (`C_tractable`):**  
   Combine point probability and epistemic variance into a single conservative confidence score at one-sided significance level `alpha = 0.05` (`z_0.95 = 1.645`):
   `C_tractable(x) = max(0.0, mu_p(x) - 1.645 * sigma_p(x))`

5. **Split Conformal Threshold Determination (`tau_green`):**  
   On `D_calib`, find the lowest threshold `tau` such that among all calibration exposures with `C_tractable(x) >= tau`, the empirical error rate (`1 - mean(Y1 == 1 and Y2 == 1)`) is `<= epsilon` (e.g., `epsilon = 0.01` or `1.0%` maximum false approval ceiling).

---

### 3.3 Reference Implementation Blueprint (Python)

```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

CAT_FEATURES = [
    "pas_match_method", "policy_term_version", "billing_risk_state",
    "vin_match_tier", "special_ded_feature_code", "coverage_asymmetry_code",
    "intake_channel", "reporter_persona", "driver_roster_match_tier",
    "llm_permissive_use_flag", "llm_true_peril_code",
    "llm_exclusion_trigger_code", "jurisdiction_regime", "llm_fault_indicator"
]

class FNOLCoverageConfidencePipeline:
    def __init__(self, n_estimators_ensemble: int = 10, z_alpha: float = 1.645):
        self.K = n_estimators_ensemble
        self.z_alpha = z_alpha
        self.models_y1, self.calibrators_y1 = [], []
        self.models_y2, self.calibrators_y2 = [], []
        self.tau_green = None

    def fit(self, df_train: pd.DataFrame, y1_train: np.ndarray, y2_train: np.ndarray,
            df_calib: pd.DataFrame, y1_calib: np.ndarray, y2_calib: np.ndarray,
            max_error_rate: float = 0.01):

        for k in range(self.K):
            # 1. Train Y1 (Coverage Applicability) on all exposures
            m1 = CatBoostClassifier(
                iterations=800, learning_rate=0.04, depth=6,
                cat_features=CAT_FEATURES, nan_mode="Min",
                subsample=0.8, random_seed=100 + k, verbose=0
            )
            m1.fit(df_train, y1_train)

            # Calibrate Y1 on D_calib
            raw_y1_calib = m1.predict_proba(df_calib)[:, 1]
            iso1 = IsotonicRegression(out_of_bounds="clip").fit(raw_y1_calib, y1_calib)
            self.models_y1.append(m1)
            self.calibrators_y1.append(iso1)

            # 2. Train Y2 (Prefill Exactness) conditioned on Y1 == 1
            mask_tr = (y1_train == 1)
            m2 = CatBoostClassifier(
                iterations=800, learning_rate=0.04, depth=6,
                cat_features=CAT_FEATURES, nan_mode="Min",
                subsample=0.8, random_seed=200 + k, verbose=0
            )
            m2.fit(df_train[mask_tr], y2_train[mask_tr])

            # Calibrate Y2 on D_calib (where Y1 == 1)
            mask_cal = (y1_calib == 1)
            raw_y2_calib = m2.predict_proba(df_calib[mask_cal])[:, 1]
            iso2 = IsotonicRegression(out_of_bounds="clip").fit(raw_y2_calib, y2_calib[mask_cal])
            self.models_y2.append(m2)
            self.calibrators_y2.append(iso2)

        # 3. Compute Conformal Threshold tau_green on D_calib
        res_cal = self.predict_confidence(df_calib)
        y_joint_true = (y1_calib == 1) & (y2_calib == 1)

        for tau in np.linspace(0.50, 0.995, 500):
            selected = res_cal["c_tractable"] >= tau
            if selected.sum() < 50:
                break
            empirical_error = 1.0 - y_joint_true[selected].mean()
            if empirical_error <= max_error_rate:
                self.tau_green = float(tau)
                break

    def predict_confidence(self, df_features: pd.DataFrame) -> pd.DataFrame:
        p1_preds = np.zeros((len(df_features), self.K))
        p2_preds = np.zeros((len(df_features), self.K))

        for k in range(self.K):
            s1 = self.models_y1[k].predict_proba(df_features)[:, 1]
            p1_preds[:, k] = self.calibrators_y1[k].predict(s1)

            s2 = self.models_y2[k].predict_proba(df_features)[:, 1]
            p2_preds[:, k] = self.calibrators_y2[k].predict(s2)

        joint_preds = p1_preds * p2_preds
        mu_p = np.mean(joint_preds, axis=1)
        sigma_p = np.std(joint_preds, axis=1, ddof=1)
        c_tractable = np.clip(mu_p - self.z_alpha * sigma_p, 0.0, 1.0)

        return pd.DataFrame({
            "prob_coverage_y1": np.mean(p1_preds, axis=1),
            "prob_prefill_y2": np.mean(p2_preds, axis=1),
            "joint_prob_mu": mu_p,
            "epistemic_std_sigma": sigma_p,
            "c_tractable": c_tractable
        })
```

---

### 3.4 Production Pipeline Routing Logic & Evaluation Metrics

#### Real-Time Routing Gate

| Routing Tier                                     | Condition                                              | Pipeline Action                                                                                                                                                                                           |
| :----------------------------------------------- | :----------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Green (Touchless STP)**                        | `C_tractable >= tau_green`                             | Lock prefilled coverages, deductibles, and limits automatically; bypass manual handler review.                                                                                                            |
| **Amber (Targeted Policy RAG / Handler Verify)** | `tau_amber <= C_tractable < tau_green`                 | Extract top 3 TreeSHAP features driving the drop in score. Trigger targeted LLM review on the specific endorsement form ID (if `restrictive_form_count > 0`) or route to handler with highlighted fields. |
| **Red (Coverage Investigation)**                 | `C_tractable < tau_amber` OR `prob_coverage_y1 < 0.50` | Suppress automated coverage commitment; route to coverage desk adjuster with SHAP reason codes.                                                                                                           |

#### Required Validation Metrics Before Deployment

1. **Expected Calibration Error (ECE):** Must be `< 0.015` on `D_test` across both `Y1` and `Y2`, stratified by `intake_channel` and `jurisdiction_regime` (specifically isolating MI vs. non-MI).
2. **Brier Score & Logloss:** Verify strictly lower Brier score post-isotonic calibration compared to uncalibrated GBDT output.
3. **STP Coverage vs. Precision At `tau_green`:** Measure `% of total claims routed to Green Lane` while asserting `False Approval Rate <= 1.0%` on `D_test`.
