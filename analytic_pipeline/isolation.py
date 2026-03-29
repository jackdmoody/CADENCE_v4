"""
Isolation Forest Anomaly Pre-Filter — Pair Level
==================================================
Scores (src_ip, dst_ip) pairs rather than individual flows.

Architecture note
-----------------
Flow-level IForest (prior to v6) consistently scored beacon flows as *normal*
because individual beacon connections are volumetrically unremarkable — small
bytes, short duration, standard ports. The anomaly signal only becomes visible
when flows are aggregated to the pair level: a pair with 1400 identical flows
at 5-minute intervals has a very low IAT CV and bytes CV, which is anomalous
relative to background pairs.

By shifting IForest input from flows to pairs, the contamination parameter
controls the fraction of *pairs* flagged rather than flows, which is both
more meaningful and more stable across different data volumes.

Output columns added to pair_df:
    iforest_score   decision_function value (lower = more anomalous)
    iforest_label   1 = normal, -1 = anomaly
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from .config import BDPConfig
from .features import IFOREST_FEATURES

log = logging.getLogger(__name__)


def run_isolation_forest(
    pair_df: pd.DataFrame,
    cfg: BDPConfig,
    feature_suffix: str = "_stdz",
) -> tuple[pd.DataFrame, pd.DataFrame, IsolationForest, float]:
    """
    Fit Isolation Forest on scaled pair features and annotate pair_df.

    Parameters
    ----------
    pair_df        : Scaled pair DataFrame (output of features.process_features).
                     Must contain *_stdz columns for each IForest feature.
    cfg            : Pipeline configuration.
    feature_suffix : Column suffix identifying scaled modelling features.

    Returns
    -------
    df_annotated : Full pair DataFrame with iforest_score and iforest_label.
    anomalies_df : Subset of df_annotated where iforest_label == -1.
    model        : Fitted IsolationForest object.
    stability    : Float in [0, 1]; train/test score-quantile agreement.
    """
    iso_cfg = cfg.isolation

    features = [c for c in pair_df.columns if c.endswith(feature_suffix)]
    if not features:
        # Fallback: CadenceScaler was bypassed — use raw IFOREST_FEATURES.
        # fillna(0) here is a last-resort safety net; median imputation should
        # have already run in scale_pair_features() via CadenceScaler.
        features = [f for f in IFOREST_FEATURES if f in pair_df.columns]
        log.warning(
            "run_isolation_forest(): no %s columns found — falling back to raw "
            "IFOREST_FEATURES with fillna(0). Ensure CadenceScaler ran first.",
            feature_suffix,
        )

    log.info("IsolationForest: %d pair-level features, %d pairs.", len(features), len(pair_df))

    MIN_PAIRS_FOR_IFOREST = 20
    if len(pair_df) < MIN_PAIRS_FOR_IFOREST:
        log.warning(
            "IForest bypass: only %d pairs in pool (< %d). "
            "Passing all pairs through to SAX/periodicity.",
            len(pair_df), MIN_PAIRS_FOR_IFOREST,
        )
        df_out = pair_df.copy()
        df_out["iforest_score"] = 0.0
        df_out["iforest_label"] = -1   # flag all as "anomalous" → proceed
        dummy_model = IsolationForest(
            n_estimators=10, contamination=0.5, random_state=iso_cfg.random_state
        ).fit(pair_df[features].fillna(0).astype(float))  # fillna(0) safe: bypass path only
        return df_out, df_out.copy(), dummy_model, 1.0

    # _stdz columns from CadenceScaler are already median-imputed; fillna(0) is
    # a safety net for any residual NaNs introduced after scaling (should be none).
    df_model = pair_df[features].fillna(0).astype(float)

    model = IsolationForest(
        n_estimators=iso_cfg.n_estimators,
        max_samples=min(iso_cfg.max_samples, len(df_model)),
        contamination=iso_cfg.contamination,
        random_state=iso_cfg.random_state,
    ).fit(df_model)

    df_out = pair_df.copy()
    df_out["iforest_score"] = model.decision_function(df_model)

    # Hard percentile cut on the full pair set.
    # IsolationForest.predict() sets its threshold on the training sample
    # (max_samples rows). When max_samples < n_pairs the effective anomaly
    # rate on the full dataset can differ substantially from contamination.
    # Cutting at the contamination percentile of all pair scores guarantees
    # exactly contamination * n_pairs anomalous pairs regardless of max_samples.
    score_threshold = np.percentile(df_out["iforest_score"], iso_cfg.contamination * 100)
    df_out["iforest_label"] = np.where(
        df_out["iforest_score"] <= score_threshold, -1, 1
    ).astype(int)

    # --- Stability check ---
    if len(df_model) >= 10:
        X_train, X_test = train_test_split(
            df_model, test_size=iso_cfg.test_size, random_state=iso_cfg.random_state
        )
        train_scores = -model.decision_function(X_train)
        test_scores  = -model.decision_function(X_test)
        qs = np.linspace(0, 1, 11)
        stability = float(
            1.0 - np.mean(
                np.abs(np.quantile(train_scores, qs) - np.quantile(test_scores, qs))
                / (np.std(train_scores) + 1e-8)
            )
        )
    else:
        stability = 1.0

    n_anom = int((df_out["iforest_label"] == -1).sum())
    log.info(
        "Pair-level IForest: %d / %d pairs anomalous (%.1f%%)  stability=%.3f%s",
        n_anom, len(df_out), n_anom / len(df_out) * 100, stability,
        "  ⚠ below threshold" if stability < iso_cfg.stability_threshold else "",
    )

    return df_out, df_out[df_out["iforest_label"] == -1].copy(), model, stability


# ---------------------------------------------------------------------------
# Concentration analysis
# ---------------------------------------------------------------------------

def concentration_report(
    df: pd.DataFrame,
    key_col,
    topk: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.Series:
    """HHI and top-k share percentages for anomalous pairs grouped by key_col."""
    if isinstance(key_col, (list, tuple)):
        key_series = df[key_col].astype(str).agg("→".join, axis=1)
        key_name   = " → ".join(key_col)
    else:
        key_series = df[key_col].astype(str)
        key_name   = key_col

    counts = key_series.value_counts(dropna=False)
    total  = counts.sum()
    shares = (counts / total).values

    topk_shares = {
        f"top_{k}_share_pct": round(float(shares[:k].sum()) * 100, 2)
        if len(shares) >= k else round(float(shares.sum()) * 100, 2)
        for k in topk
    }
    hhi = float(np.sum(shares ** 2))

    return pd.Series({
        "key":              key_name,
        "unique_keys":      int(len(counts)),
        "total_anomalies":  int(total),
        "hhi":              round(hhi, 4),
        "hhi_interpretation": (
            "unconcentrated"          if hhi < 0.15 else
            "moderately concentrated" if hhi < 0.25 else
            "highly concentrated"
        ),
        **topk_shares,
    })


def print_concentration_summary(
    df: pd.DataFrame,
    label_col: str = "iforest_label",
) -> None:
    """Print HHI / top-k concentration report for anomalous pairs."""
    anomalies_df = df[df[label_col] == -1]

    src_s  = concentration_report(anomalies_df, "src_ip")
    dst_s  = concentration_report(anomalies_df, "dst_ip")

    anomaly_rate = len(anomalies_df) / len(df)
    s_top5 = float(src_s.get("top_5_share_pct", 0))
    d_top5 = float(dst_s.get("top_5_share_pct", 0))
    s_hhi  = float(src_s.get("hhi", 0))
    d_hhi  = float(dst_s.get("hhi", 0))

    verdicts = []
    if s_hhi < 0.15 and d_hhi < 0.15:
        verdicts.append("Anomalies spread across many pairs (HHI < 0.15) — broadly representative.")
    if s_top5 > 80:
        verdicts.append("Anomalies highly concentrated in src IPs (top-5 > 80%).")
    if d_top5 > 80:
        verdicts.append("Anomalies highly concentrated in dst IPs (top-5 > 80%).")
    if not verdicts:
        verdicts.append("No strong concentration signals.")

    print("=== Anomaly Rate Diagnosis ===")
    print(f"  Rate: {anomaly_rate:.2%}  |  src top-5={s_top5}%  HHI={s_hhi}"
          f"  |  dst top-5={d_top5}%  HHI={d_hhi}")
    for v in verdicts:
        print(f"  → {v}")


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def explain_with_shap(
    model: IsolationForest,
    pair_df: pd.DataFrame,
    feature_suffix: str = "_stdz",
    max_display: int = 14,
) -> pd.DataFrame:
    """
    Compute SHAP values for every anomalous pair using TreeExplainer.

    SHAP decomposes each pair's IForest anomaly score into per-feature
    contributions, answering: "which behavioral signals drove this pair's
    anomaly score?"

    Important scope note
    --------------------
    SHAP here explains the *IForest anomaly score*, not downstream beacon
    confidence or triage score. A high SHAP value for ``iat_cv`` means the
    pair's IAT regularity drove the anomaly detection — it does not directly
    confirm beaconing. The corroboration stage (H1–H6) provides that
    confirmation. Both signals are complementary: SHAP explains *why* a pair
    was surfaced; corroboration explains *why* it is believed to be C2.

    Parameters
    ----------
    model          : Fitted IsolationForest from run_isolation_forest().
    pair_df        : Pair DataFrame containing *_stdz feature columns.
    feature_suffix : Column suffix for scaled feature columns.
    max_display    : Number of top features to include in summary plot.

    Returns
    -------
    shap_df : DataFrame indexed like pair_df, one SHAP-value column per
              feature. Column names are the clean feature names (no suffix).
              Also contains ``shap_sum`` (sum of absolute values — overall
              anomaly explanation magnitude) and identity columns
              (src_ip, dst_ip, dst_port, proto, channel_id) where present.
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP explanation. pip install shap")
        return pd.DataFrame()

    feature_cols = [c for c in pair_df.columns if c.endswith(feature_suffix)]
    if not feature_cols:
        log.warning("No %s columns found in pair_df — skipping SHAP.", feature_suffix)
        return pd.DataFrame()

    X = pair_df[feature_cols].fillna(0).astype(float)
    clean_names = [c.replace(feature_suffix, "") for c in feature_cols]

    log.info("Computing SHAP values for %d pairs × %d features...", len(X), len(feature_cols))

    # TreeExplainer works natively with sklearn IsolationForest —
    # no kernel approximation needed, exact Shapley values.
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)   # shape: (n_pairs, n_features)

    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{n}" for n in clean_names],
        index=pair_df.index,
    )

    # Overall explanation magnitude per pair (sum of |SHAP|)
    shap_df["shap_sum"] = np.abs(shap_values).sum(axis=1)

    # Re-attach identity columns for downstream joining / display
    for col in ("src_ip", "dst_ip", "dst_port", "proto", "channel_id"):
        if col in pair_df.columns:
            shap_df[col] = pair_df[col].values

    log.info(
        "SHAP complete. Top anomaly driver: %s",
        shap_df[[c for c in shap_df.columns if c.startswith("shap_") and c != "shap_sum"]]
          .abs().mean().idxmax().replace("shap_", ""),
    )
    return shap_df


def plot_shap_beeswarm(
    model: IsolationForest,
    pair_df: pd.DataFrame,
    feature_suffix: str = "_stdz",
    max_display: int = 14,
    title: str = "SHAP — IForest Feature Importance (Pair Level)",
) -> None:
    """
    Render a SHAP beeswarm summary plot for the IForest anomaly scores.

    Each dot is one pair. Horizontal position = SHAP value (impact on anomaly
    score). Color = feature value (red = high, blue = low). Features are sorted
    by mean |SHAP| descending so the most influential signal is at the top.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("shap or matplotlib not installed — skipping beeswarm plot.")
        return

    feature_cols = [c for c in pair_df.columns if c.endswith(feature_suffix)]
    if not feature_cols:
        return

    X = pair_df[feature_cols].fillna(0).astype(float)
    X.columns = [c.replace(feature_suffix, "") for c in feature_cols]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    explanation = shap.Explanation(
        values=shap_values,
        data=X.values,
        feature_names=list(X.columns),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()


def plot_shap_waterfall(
    shap_df: pd.DataFrame,
    pair_id: str,
    model: IsolationForest,
    pair_df: pd.DataFrame,
    feature_suffix: str = "_stdz",
    title: str | None = None,
) -> None:
    """
    Render a SHAP waterfall plot for a single pair, identified by channel_id.

    Shows exactly how each feature pushed the anomaly score up or down from
    the expected value — the analyst-facing "why was this pair flagged?" view.

    Parameters
    ----------
    shap_df   : Output of explain_with_shap().
    pair_id   : channel_id string to look up in shap_df.
    model     : Fitted IsolationForest (for expected_value).
    pair_df   : Scaled pair DataFrame (for raw feature values).
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("shap or matplotlib not installed.")
        return

    if "channel_id" not in shap_df.columns:
        log.warning("shap_df has no channel_id column — cannot look up pair.")
        return

    row = shap_df[shap_df["channel_id"] == pair_id]
    if row.empty:
        log.warning("channel_id %r not found in shap_df.", pair_id)
        return

    feature_cols  = [c for c in pair_df.columns if c.endswith(feature_suffix)]
    clean_names   = [c.replace(feature_suffix, "") for c in feature_cols]
    shap_cols     = [f"shap_{n}" for n in clean_names]

    idx = row.index[0]
    sv  = row[shap_cols].values[0]

    pair_row  = pair_df.loc[idx]
    X         = pair_df[feature_cols].fillna(0).astype(float)
    explainer = shap.TreeExplainer(model)
    base_val  = explainer.expected_value

    feature_vals = pair_row[feature_cols].fillna(0).astype(float).values

    explanation = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=feature_vals,
        feature_names=clean_names,
    )

    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(explanation, show=False)
    t = title or f"SHAP Waterfall — {pair_id}"
    plt.title(t, fontsize=11, pad=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Visualisation stubs (kept for pipeline.py compatibility)
# ---------------------------------------------------------------------------

def plot_iforest_score_distribution(df, *args, **kwargs):
    """Plot IForest score distribution for pair-level scores."""
    try:
        import matplotlib.pyplot as plt
        scores = df["iforest_score"].astype(float)
        threshold = np.percentile(scores, 5)
        plt.figure(figsize=(9, 5))
        plt.hist(scores, bins=60, color="steelblue", edgecolor="black", alpha=0.7)
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2,
                    label=f"5th pct cutoff ({threshold:.4f})")
        plt.title("Isolation Forest Score Distribution — Pair Level")
        plt.xlabel("Anomaly Score (lower = more anomalous)")
        plt.ylabel("Pair count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


def plot_iforest_analysis(df, *args, **kwargs):
    """Stub — pair-level IForest has no per-flow polar clock."""
    plot_iforest_score_distribution(df)
