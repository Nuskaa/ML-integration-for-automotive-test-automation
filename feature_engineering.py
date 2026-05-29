"""
feature_engineering.py
========================
Computes ML-ready features from the unified test DataFrame.

Two main outputs:
  1. build_static_features(df)  → features derivable from test design alone
                                   (no execution history needed)
  2. build_execution_features(df, exec_history_df)  → richer features when
                                   you have historical pass/fail runs

The static features are ready to use RIGHT NOW with your existing data.
The execution features unlock once you accumulate run history from TestRunner.py.

Usage:
    from data_loader import load_all_suites
    from feature_engineering import build_static_features, get_feature_matrix

    df = load_all_suites("DM_All_TestSuites/")
    features = build_static_features(df)
    X, y, feature_names = get_feature_matrix(features)
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


# ── Risk weights ────────────────────────────────────────────────────────────
# Domain knowledge encoded as multipliers (from automotive QA experience).
# Negative tests are 2.5× more likely to catch regressions than positive tests.
# High-priority tests catch high-severity failures.
# These are STARTING WEIGHTS — the ML model will refine them.

PRIORITY_RISK = {"High": 3.0, "Medium": 1.5, "Low": 0.5, None: 1.0}
TEST_TYPE_RISK = {"Negative": 2.5, "Positive": 1.0, "Automated": 1.8, None: 1.2}
LEVEL_RISK = {"Data Validation": 2.0, "Integration": 1.8, "System": 1.5, None: 1.0}

# Modules historically associated with more failures (from defect analysis)
# TSK and SANITY tests touch the most complex system interactions
MODULE_COMPLEXITY = {
    "TSK": 3.0,      # Task management — complex state machine
    "SANITY": 2.5,   # End-to-end system tests — fragile
    "SYS_SANITY": 2.5,
    "OPS": 2.0,      # UI operations — UI-sensitive
    "GGP_F": 1.8,    # GGP with implements — hardware interaction
    "GGP_S": 1.5,
    "DEFECT": 2.8,   # Tests created from defects — guaranteed to have failed before
    "PDT": 1.3,
    "PFD": 1.2,
    "GPN": 1.2,
    "CTR": 1.2,
    "FRM": 1.1,
    "SHP": 1.5,      # ShapeFile tests — file format fragility
    "PROD_SETUP": 1.3,
    "MADSW": 2.8,    # MADSW tickets = known defects
}

# Keywords in description that signal higher failure risk
HIGH_RISK_KEYWORDS = [
    "delete", "conflict", "error", "invalid", "missing", "maximum", "minimum",
    "merge", "import", "export", "factory reset", "nogfft", "asapplied",
    "overflow", "timeout", "boundary", "edge", "null", "empty",
]

MEDIUM_RISK_KEYWORDS = [
    "verify", "update", "modify", "create", "navigate",
    "product", "task", "field", "coverage",
]


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from test design metadata alone.
    No execution history required — works with data you have right now.

    Returns a copy of df with new feature columns added.
    """
    feat = df.copy()

    # ── 1. Priority encoding ────────────────────────────────────────────────
    feat["priority_score"] = feat["priority"].map(PRIORITY_RISK).fillna(1.0)

    # Ordinal encoding: High=2, Medium=1, Low=0, Unknown=-1
    priority_ord = {"High": 2, "Medium": 1, "Low": 0}
    feat["priority_ordinal"] = feat["priority"].map(priority_ord).fillna(-1).astype(int)

    # ── 2. Test type encoding ────────────────────────────────────────────────
    feat["type_risk_score"] = feat["test_type"].map(TEST_TYPE_RISK).fillna(1.2)
    feat["is_negative_test"] = (feat["test_type"] == "Negative").astype(int)
    feat["is_positive_test"] = (feat["test_type"] == "Positive").astype(int)
    feat["is_automated_type"] = (feat["test_type"] == "Automated").astype(int)

    # ── 3. Test level encoding ───────────────────────────────────────────────
    feat["level_risk_score"] = feat["test_level"].map(LEVEL_RISK).fillna(1.0)
    feat["is_data_validation"] = (feat["test_level"] == "Data Validation").astype(int)

    # ── 4. Module / suite encoding ───────────────────────────────────────────
    feat["module_complexity"] = feat["suite"].map(MODULE_COMPLEXITY).fillna(1.0)

    # One-hot encode suite (the 14 suites become binary features)
    suite_dummies = pd.get_dummies(feat["suite"], prefix="suite")
    feat = pd.concat([feat, suite_dummies], axis=1)

    # ── 5. Automation status ─────────────────────────────────────────────────
    feat["has_script_int"] = feat["has_script"].fillna(False).astype(int)
    # Non-automated tests are riskier to run late (no CI protection)
    feat["no_automation_risk"] = (feat["has_script_int"] == 0).astype(int)

    # ── 6. Step complexity ───────────────────────────────────────────────────
    # More steps = more complex = higher failure surface
    feat["step_count_filled"] = feat["step_count"].fillna(feat["step_count"].median())
    feat["is_multi_step"] = (feat["step_count_filled"] > 3).astype(int)
    feat["step_complexity_log"] = np.log1p(feat["step_count_filled"])

    # ── 7. Defect linkage ────────────────────────────────────────────────────
    # A test born from a defect ticket has historically failed — HIGH priority
    feat["has_defect_link"] = feat["has_defect"].astype(int)
    feat["is_defect_test_int"] = feat["is_defect_test"].astype(int)

    # ── 8. Work item linkage ─────────────────────────────────────────────────
    feat["has_work_item"] = feat["linked_work_item"].notna().astype(int)
    # Count multiple work items (some cells contain comma-separated IDs)
    feat["work_item_count"] = feat["linked_work_item"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )

    # ── 9. NLP features from description ─────────────────────────────────────
    feat["desc_len"] = feat["description"].fillna("").apply(len)
    feat["desc_word_count"] = feat["description"].fillna("").apply(lambda x: len(x.split()))

    # Keyword risk scores
    feat["high_risk_keyword_count"] = feat["description"].fillna("").apply(
        lambda d: sum(1 for kw in HIGH_RISK_KEYWORDS if kw in d.lower())
    )
    feat["medium_risk_keyword_count"] = feat["description"].fillna("").apply(
        lambda d: sum(1 for kw in MEDIUM_RISK_KEYWORDS if kw in d.lower())
    )

    # Specific high-signal keywords as binary features
    for kw in ["import", "export", "delete", "conflict", "error", "maximum", "minimum", "merge"]:
        feat[f"desc_has_{kw}"] = feat["description"].fillna("").str.lower().str.contains(kw).astype(int)

    # ── 10. Test number (within module) ──────────────────────────────────────
    # Higher-numbered tests are often more complex / edge-case
    feat["test_number_filled"] = feat["test_number"].fillna(0)

    # ── 11. Composite risk score ─────────────────────────────────────────────
    # Weighted combination — interpretable starting point for ranking
    feat["composite_risk_score"] = (
        feat["priority_score"] * 0.25
        + feat["type_risk_score"] * 0.20
        + feat["module_complexity"] * 0.20
        + feat["level_risk_score"] * 0.10
        + feat["has_defect_link"] * 1.5          # strong signal
        + feat["is_defect_test_int"] * 1.8        # very strong signal
        + feat["no_automation_risk"] * 0.3
        + feat["high_risk_keyword_count"] * 0.15
        + feat["is_negative_test"] * 0.40
        + feat["is_multi_step"] * 0.20
    )

    return feat


def build_execution_features(
    feat: pd.DataFrame,
    exec_history: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Add execution-history features when you have run history.

    Parameters
    ----------
    feat : DataFrame
        Output of build_static_features()
    exec_history : DataFrame
        One row per test execution with columns:
            test_id, run_date, result ('Passed'/'Failed'), build_id, duration_sec
    window : int
        Number of most recent runs to use for rolling stats

    Returns
    -------
    feat with additional columns:
        historical_fail_rate, recent_fail_rate, days_since_last_fail,
        exec_time_mean, exec_time_std, consecutive_passes, flakiness_score
    """
    if exec_history is None or len(exec_history) == 0:
        # Fill with neutral values when no history exists
        for col in ["historical_fail_rate", "recent_fail_rate", "days_since_last_fail",
                    "exec_time_mean", "exec_time_std", "consecutive_passes", "flakiness_score"]:
            feat[col] = 0.5 if "rate" in col else 0.0
        return feat

    history = exec_history.copy()
    history["result_binary"] = (history["result"] == "Failed").astype(int)
    history = history.sort_values("run_date")

    stats = {}
    for tc_id, group in history.groupby("test_id"):
        results = group["result_binary"].tolist()
        recent = results[-window:]

        # Historical fail rate
        hist_fail = sum(results) / len(results) if results else 0.5

        # Recent fail rate (last N runs — more weight to recent)
        recent_fail = sum(recent) / len(recent) if recent else hist_fail

        # Days since last failure
        failures = group[group["result_binary"] == 1]
        if len(failures) > 0 and "run_date" in group.columns:
            last_fail = failures["run_date"].max()
            import datetime
            days_since = (datetime.datetime.now() - pd.to_datetime(last_fail)).days
        else:
            days_since = 9999  # never failed → low risk

        # Execution time stats
        time_mean = group["duration_sec"].mean() if "duration_sec" in group.columns else 0
        time_std = group["duration_sec"].std() if "duration_sec" in group.columns else 0

        # Consecutive passes from the end (long streaks may hide fragility)
        consecutive = 0
        for r in reversed(results):
            if r == 0:
                consecutive += 1
            else:
                break

        # Flakiness score: count of sign changes (pass→fail or fail→pass) / total runs
        changes = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
        flakiness = changes / max(len(results) - 1, 1)

        stats[tc_id] = {
            "historical_fail_rate": hist_fail,
            "recent_fail_rate": recent_fail,
            "days_since_last_fail": min(days_since, 9999),
            "exec_time_mean": time_mean,
            "exec_time_std": time_std if not pd.isna(time_std) else 0,
            "consecutive_passes": consecutive,
            "flakiness_score": flakiness,
        }

    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.index.name = "test_id"
    stats_df = stats_df.reset_index()

    feat = feat.merge(stats_df, on="test_id", how="left")

    # Fill unknowns (new tests with no history) with conservative values
    feat["historical_fail_rate"] = feat["historical_fail_rate"].fillna(0.5)
    feat["recent_fail_rate"] = feat["recent_fail_rate"].fillna(0.5)
    feat["days_since_last_fail"] = feat["days_since_last_fail"].fillna(9999)
    feat["exec_time_mean"] = feat["exec_time_mean"].fillna(0)
    feat["exec_time_std"] = feat["exec_time_std"].fillna(0)
    feat["consecutive_passes"] = feat["consecutive_passes"].fillna(0)
    feat["flakiness_score"] = feat["flakiness_score"].fillna(0)

    return feat


# ── Feature matrix builder ───────────────────────────────────────────────────

STATIC_FEATURE_COLS = [
    # Priority & type
    "priority_score", "priority_ordinal",
    "type_risk_score", "is_negative_test", "is_positive_test", "is_automated_type",
    # Level
    "level_risk_score", "is_data_validation",
    # Module
    "module_complexity",
    # Automation
    "has_script_int", "no_automation_risk",
    # Complexity
    "step_count_filled", "is_multi_step", "step_complexity_log",
    # Defect linkage
    "has_defect_link", "is_defect_test_int",
    # Work items
    "has_work_item", "work_item_count",
    # NLP
    "desc_len", "desc_word_count",
    "high_risk_keyword_count", "medium_risk_keyword_count",
    "desc_has_import", "desc_has_export", "desc_has_delete",
    "desc_has_conflict", "desc_has_error", "desc_has_maximum",
    "desc_has_minimum", "desc_has_merge",
    # Test ordering
    "test_number_filled",
]

EXECUTION_FEATURE_COLS = [
    "historical_fail_rate", "recent_fail_rate", "days_since_last_fail",
    "exec_time_mean", "exec_time_std", "consecutive_passes", "flakiness_score",
]


def get_feature_matrix(
    feat: pd.DataFrame,
    include_suite_dummies: bool = True,
    use_execution: bool = False,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract the feature matrix X and target vector y from the feature DataFrame.

    For supervised learning, y = 1 if test has ever failed or has defect link, else 0.
    (This is a proxy label — replace with actual run results when available.)

    Returns
    -------
    X : np.ndarray  (n_tests, n_features)
    y : np.ndarray  (n_tests,)  — proxy failure labels
    feature_names : list[str]
    """
    cols = STATIC_FEATURE_COLS.copy()
    if use_execution:
        cols += EXECUTION_FEATURE_COLS

    if include_suite_dummies:
        suite_cols = [c for c in feat.columns if c.startswith("suite_")]
        cols += suite_cols

    # Keep only columns that exist
    cols = [c for c in cols if c in feat.columns]

    X = feat[cols].fillna(0).values.astype(float)

    # Proxy label: 1 = "risky" (has failed, has defect, or is defect-origin test)
    y_parts = [
        feat["result_clean"].isin(["Failed", "Blocked"]).astype(int),
        feat["has_defect"].astype(int),
        feat["is_defect_test"].astype(int),
    ]
    y = (y_parts[0] | y_parts[1] | y_parts[2]).values.astype(int)

    return X, y, cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_all_suites

    folder = sys.argv[1] if len(sys.argv) > 1 else "/tmp/DM_All_TestSuites"
    print(f"Loading from {folder}...\n")
    df = load_all_suites(folder)
    feat = build_static_features(df)

    print("\nTop 20 riskiest tests by composite score:")
    top = feat.nlargest(20, "composite_risk_score")[
        ["test_id", "suite", "priority", "test_type", "composite_risk_score",
         "high_risk_keyword_count", "has_defect_link", "is_defect_test_int"]
    ]
    print(top.to_string(index=False))

    print(f"\nLabel distribution:")
    X, y, cols = get_feature_matrix(feat)
    print(f"  Risky (y=1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Safe  (y=0): {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features ({len(cols)}): {cols}")
