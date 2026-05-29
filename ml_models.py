"""
ml_models.py
=============
Three ML modules for the DM test pipeline:

  1. TestPrioritizer      — ranks tests by predicted failure probability
  2. BenchmarkAnomalyDetector — flags timing regressions across builds
  3. FlakyTestDetector    — identifies non-deterministic tests

All models are designed to:
  - Work with small datasets (no GPU needed)
  - Be fully explainable (SHAP, feature importance)
  - Be serializable (save/load with joblib)
  - Degrade gracefully: if no training data → use rule-based ranking

Usage:
    from ml_models import TestPrioritizer, BenchmarkAnomalyDetector

    # Prioritization (rule-based works right now, ML improves with history)
    prioritizer = TestPrioritizer()
    prioritizer.fit_rule_based(feature_df)
    ranked = prioritizer.rank(feature_df)

    # Anomaly detection (works immediately on your benchmark data)
    detector = BenchmarkAnomalyDetector()
    detector.fit(benchmark_df)
    anomalies = detector.detect(new_benchmark_df)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 1: Test Case Prioritizer
# ════════════════════════════════════════════════════════════════════════════

class TestPrioritizer:
    """
    Ranks test cases by predicted failure probability.

    Two modes:
      - Rule-based (works right now): uses composite_risk_score from features
      - ML-based (needs training data): Random Forest on feature matrix

    The rule-based mode is already meaningful and can be used in production
    while you accumulate execution history for the ML model.
    """

    def __init__(self, mode: str = "rule_based"):
        """
        Parameters
        ----------
        mode : "rule_based" or "ml"
        """
        self.mode = mode
        self.model = None
        self.feature_names = None
        self.label_encoder = None
        self.is_fitted = False

    def fit_rule_based(self, feature_df: pd.DataFrame) -> "TestPrioritizer":
        """
        No training needed — uses composite_risk_score directly.
        Call this when you don't have execution history yet.
        """
        assert "composite_risk_score" in feature_df.columns, \
            "Run build_static_features() first"
        self.mode = "rule_based"
        self.is_fitted = True
        print("[TestPrioritizer] Rule-based mode ready. "
              "Ranking by composite_risk_score.")
        return self

    def fit_ml(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_estimators: int = 100,
        class_weight: str = "balanced",
    ) -> "TestPrioritizer":
        """
        Train a Random Forest classifier on labelled execution history.

        Parameters
        ----------
        X : feature matrix from get_feature_matrix()
        y : binary labels (1 = failed/risky, 0 = passed/safe)
        feature_names : column names for X
        n_estimators : number of trees (100 is good for small datasets)
        class_weight : 'balanced' handles class imbalance automatically

        Notes
        -----
        Minimum ~30 labelled examples to train. With <30, use rule_based.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        if len(y) < 30:
            print(f"[TestPrioritizer] Only {len(y)} labelled samples — "
                  "using rule_based mode. Collect more execution history.")
            self.mode = "rule_based"
            self.is_fitted = True
            return self

        print(f"[TestPrioritizer] Training Random Forest on {len(y)} samples...")
        print(f"  Class balance: {y.sum()} risky ({y.mean()*100:.1f}%), "
              f"{(1-y).sum()} safe")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            max_depth=6,             # prevents overfitting on small datasets
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.feature_names = feature_names
        self.mode = "ml"
        self.is_fitted = True

        # Cross-validation (stratified, 5-fold)
        if len(y) >= 50:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc")
            print(f"  Cross-val AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        else:
            print("  (Cross-val skipped — need 50+ samples for reliable estimate)")

        # Feature importance
        self._print_top_features(n=10)
        return self

    def _print_top_features(self, n: int = 10):
        if self.model is None or self.feature_names is None:
            return
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        print(f"\n  Top {n} most important features:")
        for feat, imp in importances.head(n).items():
            bar = "█" * int(imp * 50)
            print(f"    {feat:<35} {bar} {imp:.4f}")

    def rank(self, feature_df: pd.DataFrame, X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Return the test DataFrame sorted by failure risk (highest risk first).

        Parameters
        ----------
        feature_df : full feature DataFrame
        X : feature matrix (required for ml mode)

        Returns
        -------
        DataFrame with columns: test_id, suite, description,
                                 risk_score, risk_label, rank
        """
        assert self.is_fitted, "Call fit_rule_based() or fit_ml() first"

        result = feature_df[["test_id", "suite", "priority", "test_type",
                              "description", "composite_risk_score",
                              "has_defect_link", "is_defect_test_int"]].copy()

        if self.mode == "rule_based":
            result["risk_score"] = feature_df["composite_risk_score"]
            result["risk_pct"] = (result["risk_score"] / result["risk_score"].max() * 100).round(1)

        elif self.mode == "ml" and X is not None and self.model is not None:
            proba = self.model.predict_proba(X)
            fail_class_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else -1
            if fail_class_idx >= 0:
                result["risk_score"] = proba[:, fail_class_idx]
            else:
                result["risk_score"] = feature_df["composite_risk_score"]
            result["risk_pct"] = (result["risk_score"] * 100).round(1)
        else:
            result["risk_score"] = feature_df["composite_risk_score"]
            result["risk_pct"] = (result["risk_score"] / result["risk_score"].max() * 100).round(1)

        # Sort by risk descending
        result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1

        # Human-readable label
        result["risk_label"] = pd.cut(
            result["risk_pct"],
            bins=[0, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"],
            include_lowest=True,
        )

        return result

    def save(self, path: str):
        import joblib
        joblib.dump({"model": self.model, "mode": self.mode,
                     "feature_names": self.feature_names}, path)
        print(f"[TestPrioritizer] Saved to {path}")

    def load(self, path: str) -> "TestPrioritizer":
        import joblib
        data = joblib.load(path)
        self.model = data["model"]
        self.mode = data["mode"]
        self.feature_names = data["feature_names"]
        self.is_fitted = True
        return self


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 2: Benchmark Anomaly Detector
# ════════════════════════════════════════════════════════════════════════════

class BenchmarkAnomalyDetector:
    """
    Detects timing regressions in benchmark metrics across builds.

    Benchmark metrics: import_time, export_time, ui_launch_time,
                       discovery_time (or any numeric columns you have)

    Two detection methods (both run, results combined):
      1. Z-score (rolling window) — simple, interpretable
      2. Isolation Forest — unsupervised ML, catches subtle multivariate anomalies

    Expected input DataFrame columns:
        build_id    : string identifier for the build
        timestamp   : datetime of the benchmark run
        metric_name : e.g. "import_time", "ui_launch_time"
        value_ms    : the timing in milliseconds

    Or wide format: one column per metric.
    """

    def __init__(self, z_threshold: float = 2.5, contamination: float = 0.05,
                 window: int = 20):
        """
        Parameters
        ----------
        z_threshold : Z-score threshold for flagging (2.5 → ~1.2% false positive rate)
        contamination : expected fraction of anomalies (5% is a safe default)
        window : rolling window size for Z-score (use 20 if you have 20+ builds)
        """
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.window = window
        self.iso_forest = None
        self.metric_columns = None
        self.scaler = None
        self.is_fitted = False

    def fit(self, benchmark_df: pd.DataFrame, metric_cols: Optional[List[str]] = None):
        """
        Fit the anomaly detector on historical benchmark data.

        Parameters
        ----------
        benchmark_df : DataFrame with benchmark data (one row per build)
        metric_cols : list of numeric column names to monitor.
                      If None, uses all numeric columns except 'build_id'.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        df = benchmark_df.copy()

        if metric_cols is None:
            metric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c not in ("build_id", "build_number")]

        self.metric_columns = metric_cols
        print(f"[BenchmarkAnomalyDetector] Fitting on metrics: {metric_cols}")
        print(f"  Training samples: {len(df)}")

        # StandardScaler before Isolation Forest
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df[metric_cols].fillna(df[metric_cols].median()))

        # Isolation Forest (works well even with 10-20 builds)
        n_est = min(100, max(10, len(df) * 2))
        self.iso_forest = IsolationForest(
            n_estimators=n_est,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.iso_forest.fit(X)
        self.is_fitted = True

        # Store training stats for Z-score comparison
        self._train_means = df[metric_cols].mean()
        self._train_stds = df[metric_cols].std().replace(0, 1)

        print(f"  Isolation Forest fitted with {n_est} estimators")
        print(f"  Z-score threshold: {self.z_threshold}σ")
        return self

    def detect(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new benchmark data.

        Parameters
        ----------
        new_df : DataFrame with same metric columns as training data

        Returns
        -------
        DataFrame with columns:
            build_id, <metric_cols>,
            iso_anomaly (bool), z_score_max (float),
            z_anomaly (bool), anomaly (bool), anomalous_metrics (list),
            severity ("Normal" / "Warning" / "Critical")
        """
        assert self.is_fitted, "Call fit() first"

        result = new_df.copy()
        mc = self.metric_columns

        # ── Isolation Forest ──────────────────────────────────────────────
        X = self.scaler.transform(result[mc].fillna(self._train_means))
        iso_pred = self.iso_forest.predict(X)  # -1 = anomaly, 1 = normal
        iso_scores = self.iso_forest.score_samples(X)  # lower = more anomalous

        result["iso_anomaly"] = (iso_pred == -1)
        result["iso_score"] = iso_scores

        # ── Z-score ───────────────────────────────────────────────────────
        z_scores = (result[mc] - self._train_means) / self._train_stds
        result["z_score_max"] = z_scores.abs().max(axis=1)
        result["z_anomaly"] = result["z_score_max"] > self.z_threshold

        # Per-metric Z-score columns for diagnosis
        for col in mc:
            result[f"z_{col}"] = ((result[col] - self._train_means[col])
                                  / self._train_stds[col]).round(2)

        # ── Combined decision ─────────────────────────────────────────────
        result["anomaly"] = result["iso_anomaly"] | result["z_anomaly"]

        # Which metrics are anomalous?
        def find_anomalous_metrics(row):
            return [col for col in mc
                    if abs((row[col] - self._train_means[col])
                           / self._train_stds[col]) > self.z_threshold]

        result["anomalous_metrics"] = result.apply(find_anomalous_metrics, axis=1)

        # Severity
        def severity(row):
            if not row["anomaly"]:
                return "Normal"
            z = row["z_score_max"]
            if z > self.z_threshold * 2 or (row["iso_anomaly"] and row["z_anomaly"]):
                return "Critical"
            return "Warning"

        result["severity"] = result.apply(severity, axis=1)

        n_anomalies = result["anomaly"].sum()
        print(f"[BenchmarkAnomalyDetector] Detected {n_anomalies}/{len(result)} anomalies")

        return result

    def fit_detect(self, history_df: pd.DataFrame, new_df: pd.DataFrame,
                   metric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Convenience: fit on history, detect on new."""
        return self.fit(history_df, metric_cols).detect(new_df)

    def save(self, path: str):
        import joblib
        joblib.dump(self, path)
        print(f"[BenchmarkAnomalyDetector] Saved to {path}")

    @staticmethod
    def load(path: str) -> "BenchmarkAnomalyDetector":
        import joblib
        return joblib.load(path)


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 3: Flaky Test Detector
# ════════════════════════════════════════════════════════════════════════════

class FlakyTestDetector:
    """
    Identifies tests that produce inconsistent pass/fail results
    on the same or equivalent builds.

    A "flaky" test is one that:
      - Switches between Pass/Fail on identical builds (pure flakiness)
      - Has unusually high variance relative to its module peers

    This module requires execution history (multiple runs per test).
    Until then, it falls back to keyword-based flakiness heuristics.
    """

    # Keywords in test descriptions that correlate with flakiness
    FLAKINESS_KEYWORDS = [
        "timeout", "timing", "async", "wait", "delay", "concurrent",
        "race", "navigation", "launch", "startup", "load",
        "network", "connection", "intermittent",
    ]

    def __init__(self, min_runs: int = 5, flakiness_threshold: float = 0.2):
        """
        Parameters
        ----------
        min_runs : minimum number of runs to label a test as flaky
        flakiness_threshold : fraction of result changes that = flaky (0.2 = 20%)
        """
        self.min_runs = min_runs
        self.flakiness_threshold = flakiness_threshold

    def detect_from_history(self, exec_history: pd.DataFrame) -> pd.DataFrame:
        """
        Detect flaky tests from execution history.

        Parameters
        ----------
        exec_history : DataFrame with columns:
            test_id, build_id, result ('Passed'/'Failed'), run_date

        Returns
        -------
        DataFrame with flakiness classification per test_id
        """
        history = exec_history.copy()
        history["result_binary"] = (history["result"] == "Failed").astype(int)
        history = history.sort_values("run_date")

        rows = []
        for tc_id, group in history.groupby("test_id"):
            results = group["result_binary"].tolist()
            n = len(results)

            if n < self.min_runs:
                flakiness_type = "insufficient_data"
                flakiness_score = 0.0
            else:
                # Count sign changes
                changes = sum(1 for i in range(1, n) if results[i] != results[i-1])
                flakiness_score = changes / max(n - 1, 1)

                # Classify
                if flakiness_score == 0:
                    flakiness_type = "stable"
                elif flakiness_score < self.flakiness_threshold:
                    flakiness_type = "mostly_stable"
                elif flakiness_score < 0.5:
                    flakiness_type = "flaky"
                else:
                    flakiness_type = "highly_flaky"

            rows.append({
                "test_id": tc_id,
                "total_runs": n,
                "fail_count": sum(results),
                "pass_count": n - sum(results),
                "flakiness_score": round(flakiness_score, 3),
                "flakiness_type": flakiness_type,
                "is_flaky": flakiness_type in ("flaky", "highly_flaky"),
            })

        result_df = pd.DataFrame(rows)
        n_flaky = result_df["is_flaky"].sum()
        print(f"[FlakyTestDetector] Found {n_flaky} flaky tests "
              f"({n_flaky/len(result_df)*100:.1f}% of suite)")
        return result_df

    def detect_heuristic(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keyword-based flakiness heuristic (no history needed).
        Returns a risk score for each test based on description keywords.
        """
        result = feature_df[["test_id", "suite", "description"]].copy()

        result["flakiness_keyword_count"] = result["description"].fillna("").apply(
            lambda d: sum(1 for kw in self.FLAKINESS_KEYWORDS if kw in d.lower())
        )

        # Tests in suites that interact with navigation/UI are more flaky
        FLAKY_SUITES = {"SANITY", "SYS_SANITY", "OPS", "GGP_F", "GGP_S"}
        result["suite_flakiness_risk"] = result["suite"].isin(FLAKY_SUITES).astype(int)

        result["heuristic_flakiness_score"] = (
            result["flakiness_keyword_count"] * 0.4
            + result["suite_flakiness_risk"] * 0.6
        )

        result["flakiness_risk"] = pd.cut(
            result["heuristic_flakiness_score"],
            bins=[-1, 0, 0.5, 1.0, 999],
            labels=["Low", "Medium", "High", "Very High"],
        )

        return result.sort_values("heuristic_flakiness_score", ascending=False)


# ════════════════════════════════════════════════════════════════════════════
#  SHAP Explainability (for interviews — very impressive)
# ════════════════════════════════════════════════════════════════════════════

def explain_predictions(
    prioritizer: TestPrioritizer,
    X: np.ndarray,
    feature_names: List[str],
    test_ids: List[str],
    n_samples: int = 50,
    save_path: Optional[str] = None,
):
    """
    Generate SHAP explanations for why specific tests were ranked high/low.

    Shows: "TC_DM_TSK_005 was ranked #1 because:
            - has_defect_link contributed +0.35
            - is_negative_test contributed +0.22
            - ..."

    Parameters
    ----------
    prioritizer : fitted TestPrioritizer in 'ml' mode
    X : feature matrix
    feature_names : feature column names
    test_ids : list of test IDs (for labelling)
    n_samples : SHAP background samples (50 is fine for small datasets)
    save_path : if provided, save SHAP summary plot as PNG

    Returns
    -------
    shap_df : DataFrame with per-test SHAP values
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] Install with: pip install shap")
        return None

    if prioritizer.model is None or prioritizer.mode != "ml":
        print("[SHAP] Only available in ML mode after fit_ml()")
        return None

    print(f"[SHAP] Computing explanations for {len(X)} tests...")

    # Use TreeExplainer (fast, exact for Random Forest)
    explainer = shap.TreeExplainer(prioritizer.model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values is a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class (failure probability)
    else:
        sv = shap_values

    shap_df = pd.DataFrame(sv, columns=feature_names)
    shap_df["test_id"] = test_ids

    # Summary: which features drove each prediction?
    shap_df["top_positive_feature"] = shap_df[feature_names].idxmax(axis=1)
    shap_df["top_negative_feature"] = shap_df[feature_names].idxmin(axis=1)

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(sv, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Summary plot saved to {save_path}")

    return shap_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import load_all_suites
    from feature_engineering import build_static_features, get_feature_matrix

    folder = sys.argv[1] if len(sys.argv) > 1 else "/tmp/DM_All_TestSuites"
    df = load_all_suites(folder)
    feat = build_static_features(df)
    X, y, cols = get_feature_matrix(feat)

    print("\n" + "="*60)
    print("MODULE 1: Test Prioritizer (rule-based)")
    print("="*60)
    p = TestPrioritizer()
    p.fit_rule_based(feat)
    ranked = p.rank(feat)
    print("\nTop 15 highest-risk tests to run FIRST:")
    print(ranked[["rank", "test_id", "suite", "priority", "risk_label", "risk_pct"]].head(15).to_string(index=False))

    print("\n" + "="*60)
    print("MODULE 1b: Test Prioritizer (ML-based, proxy labels)")
    print("="*60)
    p_ml = TestPrioritizer()
    p_ml.fit_ml(X, y, cols)
    ranked_ml = p_ml.rank(feat, X)
    print("\nTop 15 highest-risk tests (ML model):")
    print(ranked_ml[["rank", "test_id", "suite", "priority", "risk_label", "risk_pct"]].head(15).to_string(index=False))

    print("\n" + "="*60)
    print("MODULE 3: Flaky Test Detector (heuristic, no history needed)")
    print("="*60)
    flaky = FlakyTestDetector()
    fdf = flaky.detect_heuristic(feat)
    print("\nTop 10 most likely flaky tests:")
    print(fdf[["test_id", "suite", "flakiness_keyword_count",
               "suite_flakiness_risk", "flakiness_risk"]].head(10).to_string(index=False))
