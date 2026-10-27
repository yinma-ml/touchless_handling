# Technical Specification: Real-Time FNOL Coverage Applicability & Prefill Verification Model

**Scope:** First-Party Auto Claims POC (Collision, Comprehensive, Towing/ERS, Rental)  
**Trigger:** Real-time execution at First Notice of Loss (FNOL) assuming a matched policy (~95% cohort).  
**Objective:** Automate manual claim handler verification of policy prefill and predict coverage applicability without running full-document LLM screening on all policy contracts.  
**Delivery Phasing:**

- **Phase 1 (current):** Deploy Model 1 (`M_cov`, target `Y1`) only. Prefill correctness is handled by deterministic rules plus handler confirmation (Section 3.4). Start capturing FNOL-vs-closure financial-term snapshots to build the `Y2` label.
- **Phase 2 (conditional):** Once `Y2` labels have matured, add Model 2 (`M_prefill`) on top of the unchanged Model 1, but only if prefill overrides are frequent and not explained by the Phase 1 rules (Section 2.3 entry criteria).

---

## 1. Targets, Feature Groups, Sources, and Calculations

### 1.1 Target Definitions (Exposure Level: Vehicle x Coverage)

- **Maturation Rule:** Exclude claims reported within 90 days of the training cutoff date unless already closed with a final adjudication code.
- **Censoring Rule:** Exclude claims closed as _Withdrawn/Abandoned prior to coverage determination_.

| Target ID | Target Name | Phase | Data Type | Ground Truth Calculation Logic |
| :--- | :--- | :--- | :--- | :--- |
| **`Y1`** | `is_coverage_applicable` | 1 | Binary `{0, 1}` | **`1` (Covered):** Exposure closes with `Paid > $0` OR closure reason `Closed Without Payment (CWP) - Below Deductible` OR `CWP - Insured Pursued 3rd Party (Coverage Verified)`.<br>**`0` (Denied):** Exposure closes with formal denial code (`No Coverage on DOL`, `Excluded Driver`, `Excluded Peril/Endorsement`, `Lapse/Out of Force`, `Material Misrepresentation`). |
| **`Y2`** | `is_prefill_exact` | 2 | Binary `{0, 1}` | _Evaluated conditionally where `Y1 == 1`. Label collection starts in Phase 1; not used for training until Phase 2._<br>**`1` (Exact):** Snapshot of the exposure's financial-term tuple `(Coverage_Code, Per_Exposure_Ded, Disappearing_Earned_Ded_Credit)` taken at FNOL equals the snapshot taken at closure.<br>**`0` (Modified):** Any element of the tuple differs between the two snapshots (coverage code changed, COLL <-> COMP switch, deductible/waiver altered, vehicle/term remapped). |

**`Y2` label engineering note:** Handler changes to coverage and deductible typically appear first in claim notes and only later in the structured tables, so parsing notes or table change-timing is unreliable. `Y2` therefore compares only two structured snapshots: one written at FNOL and one at closure. Mid-life edits are ignored; whatever reached the tables by closure is the ground truth. `Limit` and `Per_Occurrence_Ded` are excluded from the tuple because for COLL/COMP they rarely differ from prefill (ACV-based limit; per-occurrence deductible only applies to multi-car occurrences, already covered by `multi_car_occurrence_flag`). Re-add them if Phase 1 snapshots show material change rates.

---

### 1.2 Feature Engineering Specification (Strictly Available at T=0 FNOL)

_Note on Name Matching Consolidation:_ Driver matching is condensed into **2 representative features** (`driver_roster_match_tier` and `max_driver_exclusion_risk_score`) combining LLM nickname normalization, Jaro-Winkler, Double Metaphone, and suffix conflict logic upstream in Python.

| Group | Feature Name | Source | Type | Missing Value Strategy | Calculation / Extraction Logic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Policy Match & Term Integrity** | `pas_match_method` | System Log | Cat | Fill `"UNKNOWN"` | Method used to link FNOL to PAS: `Exact_Pol_Num`, `SSN_DOB`, `Name_Address_Fuzzy`, `Manual_CSR`. |
| | `dol_term_boundary_days` | PAS + FNOL | Float | `NaN` (Native GBDT) | `min(abs(DOL - Term_Start), abs(DOL - Term_End))` in days. |
| | `policy_term_version` | PAS | Cat | Fill `"UNKNOWN"` | `6_Month_Term` vs. `12_Month_Term` product architecture flag. |
| | `recent_endorsement_days` | PAS | Float | `NaN` (= No Endorsement) | `DOL - Last_Mod_Date` (coverage added, deductible lowered, or vehicle swapped prior to DOL). Leave `NaN` if no mid-term modification exists. |
| | `billing_risk_state` | PAS Billing | Cat | Fill `"CURRENT"` | As-of DOL billing status: `Current`, `In_Grace_Period`, `Pending_Cancel_NonPay`, `Reinstated_Last_30D`. |
| | `vin_match_tier` | PAS + FNOL | Cat | Fill `"UNVERIFIED"` | `Exact_VIN`, `Year_Make_Model_Match`, `Newly_Acquired_Unlisted`, `Temp_Rental_Sub`, `Mismatch`. |
| **B. Exposure & Deductible Prefill** | `multi_car_occurrence_flag` | FNOL | Int | Fill `0` | `1` if `>= 2` insured vehicles are claimed under the same occurrence; `0` otherwise. |
| | `special_ded_feature_code` | PAS | Cat | Fill `"STANDARD"` | `Disappearing_Ded`, `Earned_Ded`, `Glass_Full_Waiver`, `Collision_Ded_Waiver_CDW`, `Standard`. |
| | `coverage_asymmetry_code` | PAS | Cat | Fill `"NONE"` | Vehicle coverage profile: `Full_COMP_COLL`, `COMP_Only`, `Liability_Only`, `Storage_Comp_Only`. |
| | `deductible_to_acv_ratio` | PAS + VinBook | Float | `NaN` (Native GBDT) | `Prefilled_Deductible / Estimated_Vehicle_ACV`. |
| **C. Channel & Reporter** | `intake_channel` | FNOL | Cat | Fill `"UNKNOWN"` | `Phone_CSR`, `Web_Self_Service`, `Mobile_App`, `Third_Party_API`. |
| | `reporter_persona` | FNOL + LLM | Cat | Fill `"UNKNOWN"` | `Named_Insured`, `Spouse`, `Unlisted_Driver`, `Body_Shop`, `Rental_Tow_Vendor`, `Attorney`, `Third_Party`. |
| | `narrative_source` | FNOL | Cat | Fill `"None"` | Text the LLM extractor read: `Call_Transcript`, `CSR_Notes`, `Web_Text`, `None`. When `None`, all `llm_*` features and `reporter_persona` are set to `"NO_NARRATIVE"` (Section 2.2). |
| | `fnol_reporting_lag_hrs` | FNOL | Float | `NaN` (Native GBDT) | `(FNOL_Timestamp - DOL_Timestamp)` in hours. Cap lower bound at `0.0`. |
| **D. Consolidated Driver Match** | `driver_roster_match_tier` | PAS + FNOL + LLM | Cat | Fill `"NO_DRIVER_PARKED"` | Upstream Python rule combining LLM-normalized first name (root + nickname variants), last name, suffix, and DOB against PAS Listed Drivers. Tier definitions below the table. |
| | `max_driver_exclusion_risk_score` | PAS + FNOL + LLM | Float `[0, 1]` | `NaN` (= No NDE on Policy) | If policy has a Named Driver Exclusion (NDE) endorsement, compute `max(JaroWinkler, 0.95 * Metaphone_Match)` across raw and LLM-canonical driver names against all NDE names. Output `NaN` if policy has no excluded drivers. |
| | `llm_permissive_use_flag` | LLM Fact | Cat | Fill `"NOT_STATED"` | Extracted from transcript/notes: `Explicit_Permission`, `Implied_Family`, `Taken_Without_Permission`, `Shop_Valet_Custody`, `Not_Stated`. |
| **E. Loss Peril & Exclusion Facts** | `llm_true_peril_code` | LLM Fact | Cat | Fill `"AMBIGUOUS"` | `Veh_Collision`, `Fixed_Object`, `Pothole_Road_Debris`, `Animal_Strike`, `Weather_Hail_Flood`, `Glass_Only`, `Theft_Vandalism`, `Mechanical_Wear_Tear`, `Ambiguous`. |
| | `peril_conflict_flag` | FNOL + LLM | Int | Fill `-1` (if LLM `AMBIGUOUS`) | `1` if structured FNOL loss cause `!= llm_true_peril_code` (e.g., structured=`COMP`, narrative=`slid on ice hit tree`); `0` if aligned. |
| | `llm_exclusion_trigger_code` | LLM Fact | Cat | Fill `"NONE_DETECTED"` | Single consolidated exclusion trigger from text: `Commercial_Rideshare_Delivery`, `Wear_Tear_Mechanical`, `Racing_OffRoad`, `Out_Of_Territory_Mex`, `Intentional_Act`, `Aftermarket_Custom_Equip`, `None_Detected`. |
| **F. State & Endorsement Rules** | `jurisdiction_regime` | PAS + FNOL | Cat | Fill `"STANDARD"` | `MI_Broad_Coll`, `MI_Standard_Coll`, `MI_Limited_Coll`, `Out_Of_State_Loss_Deemer`, `Standard_In_State`. |
| | `llm_fault_indicator` | LLM Fact | Cat | Fill `"UNKNOWN"` | `Clear_Not_At_Fault` (e.g., rear-ended, parked), `Clear_At_Fault`, `Single_Veh_At_Fault`, `Disputed_Comparative`, `Unknown`. Interacts with MI Collision tiers & CDW. |
| | `restrictive_form_count` | PAS Forms | Int | Fill `0` | Count of active restrictive endorsement form IDs on policy (e.g., `Named_Operator_Only`, `Step_Down_Limit`, `Rideshare_Exclusion`, `ACV_Parts_Only`). |

**`driver_roster_match_tier` definitions:**

- `Tier_1_Exact_Or_Phonetic`: JW `>= 0.92` or Double Metaphone match, no suffix/DOB conflict.
- `Tier_2_Fuzzy_Listed`: `0.80 <= JW < 0.92`, no suffix conflict.
- `Tier_3_Suffix_Or_DOB_Conflict`: Name matches (`JW >= 0.85`) but Jr./Sr. or DOB mismatches.
- `Tier_4_Unlisted_Household`: Last name JW `>= 0.90` or same address, first name JW `< 0.60`.
- `Tier_5_Unlisted_External`: Both first and last name JW `< 0.60`, different address.
- `No_Driver_Parked`: Vehicle was parked/unoccupied.

---

## 2. Model Architecture, Inputs, and Missing Value Handling

### 2.1 Upstream LLM Fact Extractor (Pre-Model Input Step)

- **Input:** Raw FNOL call transcript or web free-text notes (blind to PAS policy data to prevent confirmation bias).
- **Execution:** Deterministic schema-constrained JSON output (`temperature = 0.0`).
- **Output Fields:** `reported_driver_first`, `canonical_first_variants[]`, `reported_driver_last`, `driver_suffix`, `reporter_persona`, `llm_permissive_use_flag`, `llm_true_peril_code`, `llm_exclusion_trigger_code`, `llm_fault_indicator`.
- **Constraint:** Do not extract or use LLM token log-probs or LLM self-rated confidence scores.

### 2.2 Missing Value Strategy (Critical for FNOL Sparsity)

At FNOL, 30%–60% of claims will lack unstructured transcripts (e.g., terse web submissions), unlisted driver DOBs, or mid-term endorsements. **Do NOT use mean, median, or zero imputation for continuous features**, as `0` collides with true zero values (e.g., `dol_term_boundary_days = 0` means the crash happened on renewal day, whereas `NaN` means date unknown).

The historical data adds a second, structural source of missingness: call transcripts exist only for the most recent ~9 of 36 months (Regime B). The older ~27 months (Regime A) have at most brief CSR or web notes. Section 3.1 handles the split; this section handles the encoding.

1. **Continuous Features (`Float`):** Pass raw `np.nan` directly into CatBoost. CatBoost's `nan_mode` is a single global rule (`'Min'` sends all missing values to the smallest bin, `'Max'` to the largest), not a per-split learned direction. For continuous features where missingness itself is predictive (e.g., `recent_endorsement_days`, `deductible_to_acv_ratio`), add an explicit `is_missing_<feature>` integer flag so the trees can split on missingness independently of the value.
2. **Categorical Features (`Cat`):** Never leave as `NaN` or drop rows. Map missing/unmentioned values to an explicit string token (`"UNKNOWN"`, `"NOT_STATED"`, or `"NO_DRIVER_PARKED"`). Missingness at FNOL is informative (Missing Not At Random) and must be a splittable category.
3. **Narrative Source and "No Narrative" Token:** Add the categorical feature `narrative_source` (`Call_Transcript`, `CSR_Notes`, `Web_Text`, `None`) describing what text the LLM extractor (Section 2.1) actually read. When `narrative_source == "None"`, set every `llm_*` feature and `reporter_persona` to `"NO_NARRATIVE"`, and set `peril_conflict_flag = -1`. Reserve `"NOT_STATED"` / `"UNKNOWN"` / `"AMBIGUOUS"` for cases where text existed but did not state the fact.
   - This keeps "no text existed" (most of Regime A) distinct from "text existed but was silent" (an informative signal).
   - Production also has claims without transcripts (web and mobile submissions), so `NO_NARRATIVE` and `CSR_Notes` are real production states, not only a legacy artifact.
   - Do **not** add the regime itself (A/B) or a date feature as a model input: every production claim is Regime B, so the model would learn nothing transferable from it.
4. **Sparsity Indicator Feature:** Add a single integer feature `missing_critical_fnol_count` counting how many of `[VIN, Driver_Name, Narrative_Text, Exact_Loss_Location]` are missing at `T=0`.

### 2.3 Two-Stage Gradient Boosted Decision Tree (GBDT) Architecture (Phased)

Use **CatBoostClassifier**. LightGBM is not recommended for this problem:

- **Categorical-heavy feature set:** About 15 of ~23 features are categorical, and key signals are combinations (e.g., `jurisdiction_regime` x `llm_fault_indicator`). CatBoost's ordered target statistics encode these without target leakage. LightGBM's native categorical splits overfit rare levels, and the minority class (`Y1 = 0`, denials) is rare.
- **Small-data robustness:** Ordered boosting reduces prediction shift on the comparatively small Regime B sample that drives calibration.
- **Native uncertainty:** `posterior_sampling=True` with `virtual_ensembles_predict` gives an ensemble spread from a single model. Benchmark this against the explicit `K = 10` ensemble (Section 3.2) and keep the explicit ensemble only if its spread is materially better at ranking errors.
- **Speed:** LightGBM's main advantage, training speed, is not a constraint at FNOL claim volumes.

**Two-Regime Sample Weighting (Training Only):** `D_train` contains both regimes (Section 3.1). Regime A teaches policy-system edge cases (rare endorsements, term boundaries, Michigan collision tiers, multi-car occurrences, driver name/suffix mismatches). Regime B teaches transcript-based signals and matches production. Up-weight Regime B so it carries a target share `s` of the total training weight:

- `w_A = 1.0`
- `w_B = (s / (1 - s)) * (N_A / N_B)`, where `N_A` and `N_B` are the Regime A and Regime B row counts in `D_train`
- Default `s = 0.5`; tune `s in {0.3, 0.5, 0.7}` by logloss on `D_calib`.
- Weights apply to tree building only. Calibration and threshold setting are unweighted and use Regime B data only.

The architecture remains two-stage, but only Model 1 is trained and deployed in Phase 1. Because Model 2 estimates `P(Y2 = 1 | Y1 = 1, x)`, it can be added later by multiplying onto Model 1's calibrated output (chain rule) without retraining Model 1.

- **Model 1 (`M_cov` — Coverage Applicability) — Phase 1:**
  - **Target:** `Y1` (`is_coverage_applicable`)
  - **Training Population:** All mature matched-policy first-party exposures from Train households, both regimes, with two-regime sample weights.
  - **Loss Function:** Logloss. If class weights are used during training to boost recall on `Y1 = 0`, post-hoc isotonic calibration (Section 3) is mandatory to restore true prior probabilities.
- **Model 2 (`M_prefill` — Prefill Verification) — Phase 2, conditional:**
  - **Target:** `Y2` (`is_prefill_exact`), built from FNOL-vs-closure snapshots (Section 1.1).
  - **Training Population:** Subset of exposures where `Y1 == 1`.
  - **Loss Function:** Standard binary Logloss.

#### Phase 1 Prefill Handling (Deterministic Rules)

In Phase 1, prefill correctness is not scored by a model. Instead, a deterministic flag `prefill_rule_pass` is computed at FNOL and is `1` only when all of the following hold:

- `multi_car_occurrence_flag == 0`
- `special_ded_feature_code == "STANDARD"`
- `peril_conflict_flag == 0` (an ambiguous peril, `-1`, fails the rule)
- `vin_match_tier == "Exact_VIN"`
- `recent_endorsement_days` is `NaN` or `> 30`

`prefill_rule_pass` is a routing input (Section 3.4), not a model feature.

#### Phase 2 Entry Criteria

Train and deploy `M_prefill` only when all of the following are met:

1. At least 6 months of matured FNOL/closure snapshot pairs are available (enough to fill `D_train`, `D_calib`, and `D_test` for `Y2`).
2. The `Y2 = 0` rate among `Y1 = 1` exposures is material (e.g., `>= 2%`); below that, `M_prefill` mostly learns noise.
3. `prefill_rule_pass` fails to explain a meaningful share of `Y2 = 0` cases (e.g., `> 30%` of overrides occur on rule-pass claims). If the rules already catch most overrides, keep the rules and skip `M_prefill`.

---

## 3. Tractable Probability & Confidence Calculation

To guarantee scientific, mathematically bounded probabilities and capture epistemic uncertainty (rare feature combinations where the model lacks historical support), implement a **Virtual Ensemble + Isotonic Calibration + Lower Confidence Bound** pipeline.

### 3.1 Dataset Splitting (Household-Grouped Stratified Split, Two Regimes)

**Why not a temporal split:** With 36 months of history and transcripts only in the most recent ~9 months, a 70/15/15 time split would put almost no transcripts in `D_train`. The trees would never split on the `llm_*` features. It would also tie `D_calib` and `D_test` to a single 2–3 month season (hail and deer vs. ice) and to whatever operational push was running at the time. Calibration and conformal thresholds require `D_calib`, `D_test`, and production to be exchangeable, so they must come from the transcript period (Regime B).

**Data regimes:**

- **Regime A (months 1–27, ~75% of volume):** No call transcripts (brief CSR or web notes may exist). Rich in policy-system edge cases.
- **Regime B (months 28–36, ~25% of volume):** Call transcripts plus PAS data, equivalent to live production.

**Split procedure:**

1. **Group key:** `Household_ID`, falling back to `Root_Policy_ID` where it is missing. Household is preferred because a policy rewrite issues a new policy number while keeping the same drivers and vehicles. All claims and exposures of a group across all 36 months and all 6m/12m renewal terms land in the same bucket.
2. **Assign groups once:** Use `StratifiedGroupKFold` to assign groups to Train / Calib / Test at 70 / 15 / 15. Stratify on the combination of regime, `Y1`, Michigan vs. non-Michigan, and `intake_channel`, so seasons, renewal cycles, and both regimes spread evenly across buckets.
3. **Filter by regime per bucket:**
   - **`D_train`:** All claims of Train groups, Regime A and B, with two-regime sample weights (Section 2.3).
   - **`D_calib`:** Regime B claims of Calib groups only. Used _only_ for isotonic calibration, tuning the regime weight `s`, and the conformal threshold.
   - **`D_test`:** Regime B claims of Test groups only. Used for final evaluation.
   - **Discarded:** Regime A claims of Calib and Test groups. Putting them in `D_train` would leak those households back into training.
4. **Strict maturation for Regime B in `D_calib` and `D_test`:** The general maturation rule (Section 1.1) keeps recent claims that closed early. Early closures are mostly simple payments, because denials take longer, which would make calibration optimistic in exactly the sets that set thresholds. For `D_calib` and `D_test`, require the full 90-day maturation window with no early-closure exception. Verify that the denial rate by report-age cohort matches older claims at the same age.

**Forward backtest (validation run, not a data bucket):** The grouped split spreads time evenly, so it hides drift (form filings, rate changes, operational pushes). Before deployment, re-run the full pipeline with a cutoff 3 months before the last matured month. Train, calibrate, and set `tau_green` on data before the cutoff only, then measure the Green-lane error rate on the 3 held-out months (`D_oot`, groups not in that run's `D_train`). The production model is then refit on all data with the standard split. A live shadow-mode period is the final out-of-time check.

**Sample-size check:** Regime B leaves roughly 6 matured months, and `D_calib` gets 15% of those groups. With a denial rate around 2%, there may be too few denials above `tau_green` to certify a 1% error ceiling. Before fixing `epsilon`, count the expected Green-lane exposures in `D_calib`. Certifying `epsilon = 0.01` with the Clopper–Pearson rule in Section 3.2 needs roughly 3,000 or more. If that count isn't reachable, shift the split toward Calib (e.g., 65 / 25 / 10) or start with a looser `epsilon` (e.g., 0.02) and tighten it as data accumulates.

### 3.2 Step-by-Step Mathematical Formulation

1. **Ensemble Training for Epistemic Uncertainty:**  
   Train a `K = 10` model ensemble (via 10 bootstrap/sub-sampled folds or CatBoost's native `VirtualEnsembles` method) for `M_cov` on `D_train` (Phase 1). In Phase 2, also train a `K = 10` ensemble for `M_prefill`.

2. **Post-Hoc Isotonic Calibration on `D_calib`:**  
   For each ensemble member `k in {1..K}`, fit a monotonic **Isotonic Regression** calibrator `g_1k(.)` for `M_cov` (and, in Phase 2, `g_2k(.)` for `M_prefill`) on `D_calib`.  
   For any new FNOL exposure `x`, the calibrated probability for ensemble member `k` is:
   - **Phase 1 (coverage only):** `p_hat_k(x) = g_1k(s_1k(x))`
   - **Phase 2 (coverage AND exact prefill):** `p_hat_k(x) = g_1k(s_1k(x)) * g_2k(s_2k(x))`

3. **Point Estimate (`mu_p`) and Epistemic Uncertainty (`sigma_p`):**  
   Compute the empirical mean probability and standard deviation across the `K` calibrated ensemble predictions:
   - `mu_p(x) = (1 / K) * sum_{k=1..K} p_hat_k(x)`
   - `sigma_p(x) = sqrt( (1 / (K - 1)) * sum_{k=1..K} (p_hat_k(x) - mu_p(x))^2 )`

4. **Tractable Lower Confidence Bound (`C_tractable`):**  
   Combine point probability and epistemic variance into a single conservative confidence score at one-sided significance level `alpha = 0.05` (`z_0.95 = 1.645`):
   `C_tractable(x) = max(0.0, mu_p(x) - 1.645 * sigma_p(x))`

5. **Split Conformal Threshold Determination (`tau_green`):**  
   On `D_calib`, find the lowest threshold `tau` such that among all calibration exposures with `C_tractable(x) >= tau`, the one-sided 95% Clopper–Pearson upper bound on the error rate is `<= epsilon`. Using the upper bound rather than the raw empirical rate guards against small Regime B calibration samples (e.g., `epsilon = 0.01` or `1.0%` maximum false approval ceiling). The error rate is `1 - mean(Y1 == 1)` in Phase 1 and `1 - mean(Y1 == 1 and Y2 == 1)` in Phase 2. Re-derive `tau_green` when moving to Phase 2; the Phase 1 threshold does not transfer.

---

### 3.3 Reference Implementation Blueprint (Python)

```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

CAT_FEATURES = [
    "pas_match_method", "policy_term_version", "billing_risk_state",
    "vin_match_tier", "special_ded_feature_code", "coverage_asymmetry_code",
    "intake_channel", "narrative_source", "reporter_persona",
    "driver_roster_match_tier", "llm_permissive_use_flag", "llm_true_peril_code",
    "llm_exclusion_trigger_code", "jurisdiction_regime", "llm_fault_indicator"
]


def assign_group_buckets(df: pd.DataFrame, n_folds: int = 20,
                         train_folds: int = 14, calib_folds: int = 3,
                         seed: int = 42) -> pd.Series:
    """Assign every household to Train / Calib / Test (default 70/15/15).

    Expects columns: household_id (fallback already applied from root_policy_id),
    regime ('A' or 'B'), y1, jurisdiction_regime, intake_channel.
    """
    is_mi = df["jurisdiction_regime"].str.startswith("MI_").map({True: "MI", False: "NonMI"})
    coarse = df["regime"] + "|" + df["y1"].astype(str)
    strata = coarse + "|" + is_mi + "|" + df["intake_channel"]
    rare = strata.map(strata.value_counts()) < n_folds
    strata = strata.where(~rare, coarse)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for i, (_, idx) in enumerate(sgkf.split(df, strata, groups=df["household_id"])):
        fold.iloc[idx] = i

    return pd.Series(
        np.select([fold < train_folds, fold < train_folds + calib_folds], ["train", "calib"], "test"),
        index=df.index,
    )


def build_datasets(df: pd.DataFrame, bucket: pd.Series):
    """Train: both regimes. Calib/Test: Regime B, fully matured (no early-closure exception)."""
    d_train = df[bucket == "train"]
    regime_b_strict = (df["regime"] == "B") & df["is_fully_matured_90d"]
    d_calib = df[(bucket == "calib") & regime_b_strict]
    d_test = df[(bucket == "test") & regime_b_strict]
    return d_train, d_calib, d_test


def regime_sample_weights(regime: pd.Series, target_share_b: float = 0.5) -> np.ndarray:
    n_a = (regime == "A").sum()
    n_b = (regime == "B").sum()
    w_b = (target_share_b / (1.0 - target_share_b)) * (n_a / n_b)
    return np.where(regime == "B", w_b, 1.0)


def clopper_pearson_upper(n_errors: int, n: int, alpha: float = 0.05) -> float:
    if n_errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, n_errors + 1, n - n_errors))


class FNOLCoverageConfidencePipeline:
    def __init__(self, n_estimators_ensemble: int = 10, z_alpha: float = 1.645):
        self.K = n_estimators_ensemble
        self.z_alpha = z_alpha
        self.models_y1, self.calibrators_y1 = [], []
        self.models_y2, self.calibrators_y2 = [], []
        self.use_prefill_model = False
        self.tau_green = None

    @staticmethod
    def _new_model(seed: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=800, learning_rate=0.04, depth=6,
            cat_features=CAT_FEATURES, nan_mode="Min",
            bootstrap_type="Bernoulli", subsample=0.8,
            random_seed=seed, verbose=0
        )

    def fit(self, X_train: pd.DataFrame, y1_train: np.ndarray, w_train: np.ndarray,
            X_calib: pd.DataFrame, y1_calib: np.ndarray,
            y2_train: np.ndarray | None = None, y2_calib: np.ndarray | None = None,
            max_error_rate: float = 0.01, min_selected: int = 50):
        # X_calib must be Regime B only (exchangeable with production).
        # Phase 1: y2_* is None -> M_cov only. Phase 2: pass y2_* to add M_prefill.
        self.use_prefill_model = y2_train is not None and y2_calib is not None

        for k in range(self.K):
            # 1. Train Y1 (Coverage Applicability) on both regimes, Regime B up-weighted
            m1 = self._new_model(seed=100 + k)
            m1.fit(X_train, y1_train, sample_weight=w_train)

            # Calibrate Y1 on D_calib (unweighted, Regime B only)
            raw_y1_calib = m1.predict_proba(X_calib)[:, 1]
            iso1 = IsotonicRegression(out_of_bounds="clip").fit(raw_y1_calib, y1_calib)
            self.models_y1.append(m1)
            self.calibrators_y1.append(iso1)

            if not self.use_prefill_model:
                continue

            # 2. (Phase 2) Train Y2 (Prefill Exactness) conditioned on Y1 == 1
            mask_tr = (y1_train == 1)
            m2 = self._new_model(seed=200 + k)
            m2.fit(X_train[mask_tr], y2_train[mask_tr], sample_weight=w_train[mask_tr])

            # Calibrate Y2 on D_calib (where Y1 == 1)
            mask_cal = (y1_calib == 1)
            raw_y2_calib = m2.predict_proba(X_calib[mask_cal])[:, 1]
            iso2 = IsotonicRegression(out_of_bounds="clip").fit(raw_y2_calib, y2_calib[mask_cal])
            self.models_y2.append(m2)
            self.calibrators_y2.append(iso2)

        # 3. Conformal threshold tau_green on D_calib using the Clopper-Pearson upper bound
        res_cal = self.predict_confidence(X_calib)
        y_true = (y1_calib == 1)
        if self.use_prefill_model:
            y_true = y_true & (y2_calib == 1)

        for tau in np.linspace(0.50, 0.995, 500):
            selected = (res_cal["c_tractable"] >= tau).to_numpy()
            n_sel = int(selected.sum())
            if n_sel < min_selected:
                break
            n_err = int((~y_true[selected]).sum())
            if clopper_pearson_upper(n_err, n_sel) <= max_error_rate:
                self.tau_green = float(tau)
                break

        if self.tau_green is None:
            # Not certifiable at this epsilon with the available Regime B sample (Section 3.1).
            print(f"WARNING: no tau certifies error <= {max_error_rate}; Green lane disabled.")

    def predict_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        p1_preds = np.zeros((len(X), self.K))
        p2_preds = np.ones((len(X), self.K))

        for k in range(self.K):
            s1 = self.models_y1[k].predict_proba(X)[:, 1]
            p1_preds[:, k] = self.calibrators_y1[k].predict(s1)

            if self.use_prefill_model:
                s2 = self.models_y2[k].predict_proba(X)[:, 1]
                p2_preds[:, k] = self.calibrators_y2[k].predict(s2)

        joint_preds = p1_preds * p2_preds
        mu_p = np.mean(joint_preds, axis=1)
        sigma_p = np.std(joint_preds, axis=1, ddof=1)
        c_tractable = np.clip(mu_p - self.z_alpha * sigma_p, 0.0, 1.0)

        return pd.DataFrame({
            "prob_coverage_y1": np.mean(p1_preds, axis=1),
            "prob_prefill_y2": np.mean(p2_preds, axis=1) if self.use_prefill_model else np.nan,
            "joint_prob_mu": mu_p,
            "epistemic_std_sigma": sigma_p,
            "c_tractable": c_tractable
        }, index=X.index)


# --- Usage (Phase 1) ---
# FEATURES = [...]  # all Section 1.2 features + Section 2.2 missing flags
# bucket = assign_group_buckets(df)
# d_train, d_calib, d_test = build_datasets(df, bucket)
# w_train = regime_sample_weights(d_train["regime"], target_share_b=0.5)
# pipe = FNOLCoverageConfidencePipeline()
# pipe.fit(d_train[FEATURES], d_train["y1"].to_numpy(), w_train,
#          d_calib[FEATURES], d_calib["y1"].to_numpy())
# scores_test = pipe.predict_confidence(d_test[FEATURES])
```

---

### 3.4 Production Pipeline Routing Logic & Evaluation Metrics

#### Real-Time Routing Gate

`C_tractable` is coverage-only in Phase 1 and joint (coverage AND exact prefill) in Phase 2.

| Routing Tier | Condition | Phase 1 Action (`M_cov` only) | Phase 2 Action (`M_cov` + `M_prefill`) |
| :--- | :--- | :--- | :--- |
| **Green (Touchless STP)** | `C_tractable >= tau_green` | Auto-confirm coverage. If `prefill_rule_pass == 1`, also lock prefilled coverage code, deductible, and limits. Otherwise, show the handler a one-click confirmation of the prefilled terms (no coverage review). | Lock prefilled coverages, deductibles, and limits automatically; bypass manual handler review. |
| **Amber (Targeted Policy RAG / Handler Verify)** | `tau_amber <= C_tractable < tau_green` | Extract top 3 TreeSHAP features driving the drop in score. Trigger targeted LLM review on the specific endorsement form ID (if `restrictive_form_count > 0`) or route to handler with highlighted fields. | Same as Phase 1; SHAP reason codes come from whichever model (`M_cov` or `M_prefill`) drives the drop. |
| **Red (Coverage Investigation)** | `C_tractable < tau_amber` OR `prob_coverage_y1 < 0.50` | Suppress automated coverage commitment; route to coverage desk adjuster with SHAP reason codes. | Same as Phase 1. |

#### Required Validation Metrics Before Deployment

1. **Expected Calibration Error (ECE):** Must be `< 0.015` on `D_test` for `Y1` (Phase 1), and also for `Y2` in Phase 2, stratified by `intake_channel`, `narrative_source`, and `jurisdiction_regime` (specifically isolating MI vs. non-MI).
2. **Brier Score & Logloss:** Verify strictly lower Brier score post-isotonic calibration compared to uncalibrated GBDT output.
3. **STP Coverage vs. Precision At `tau_green`:** Measure `% of total claims routed to Green Lane` while asserting `False Approval Rate <= 1.0%` on `D_test` (coverage errors in Phase 1; coverage or prefill errors in Phase 2).
4. **Phase 1 Prefill Rule Monitoring:** For Green claims with `prefill_rule_pass == 1` (auto-locked prefill), track the FNOL-vs-closure snapshot override rate. If it exceeds `1.0%`, tighten the rules or route those claims to one-click confirmation. This monitoring also supplies the evidence for the Phase 2 entry criteria (Section 2.3).
5. **Forward Backtest:** On the 3 held-out months (`D_oot`, Section 3.1), the Green-lane error rate at the backtest's own `tau_green` must stay `<= epsilon`. A breach indicates temporal drift that the grouped split cannot reveal; investigate before deployment.
