"""
Feature Engineering — Channel Aggregation
==========================================
Transforms a flow-level Zeek conn log DataFrame into a channel-level
behavioral summary DataFrame, one row per channel.

Point 1: Channel key
--------------------
The grouping unit is now configurable via cfg.pair.channel_key. The default
is (src_ip, dst_ip, dst_port, proto), which separates services on the same
destination host. All downstream stages (SAX, periodicity, corroboration,
triage) receive a 'channel_id' string key that encodes all key components.

Legacy (src_ip, dst_ip) behaviour is available by setting:
    cfg.pair.channel_key = ("src_ip", "dst_ip")

Point 4: New beacon-discriminating features
-------------------------------------------
In addition to the original 8 IForest features, the following are computed:

    iat_mad_s           Median absolute deviation of IAT (robust jitter measure).
    iat_ratio           Median / mean IAT; near 1.0 = symmetric, low = skewed.
    missing_beat_rate   Fraction of expected beacon slots with no connection.
                        Requires enough observations to estimate period.
    persistence_ratio   Fraction of observation days with at least one flow.
                        Beacons are active every day; sporadic traffic is not.
    req_resp_asymmetry  |src_bytes - dst_bytes| / (src_bytes + dst_bytes + 1).
                        C2 polling: tiny request, small response → near 0.
                        Exfiltration: large src → near 1.
    zero_payload_frac   Fraction of flows with zero destination bytes.
                        High in keepalive/heartbeat channels.

Output columns (all existing + new)
-------------------------------------
    Identity
        src_ip, dst_ip, dst_port, proto, channel_id

    Volume
        n_flows, log_n_flows

    Scheduling regularity  (primary beacon signal)
        iat_mean_s, iat_cv, iat_log_mean
        iat_mad_s (new), iat_ratio (new), missing_beat_rate (new)

    Payload uniformity
        bytes_mean, bytes_cv

    Temporal / persistence
        persistence_ratio (new)
        sin_time_mean, cos_time_mean

    Payload character
        req_resp_asymmetry (new), zero_payload_frac (new)

    Connection consistency
        duration_cv, conn_state_entropy

    Metadata
        first_seen, last_seen, scenario
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import BDPConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(series: pd.Series) -> float:
    """Shannon entropy of a categorical series."""
    probs = series.value_counts(normalize=True).values
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _ts_to_float(ts: pd.Series) -> pd.Series:
    """Convert timestamp series to float Unix seconds, handling both dtypes."""
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.values.astype("datetime64[ns]").astype(np.float64) / 1e9
    return ts.astype(float)


def _channel_id(row_keys: tuple, key_names: tuple) -> str:
    """Build a human-readable channel ID string from key component values."""
    return "→".join(str(v) for v in row_keys)


# ---------------------------------------------------------------------------
# Channel aggregation
# ---------------------------------------------------------------------------

def aggregate_pairs(
    df: pd.DataFrame,
    cfg: BDPConfig,
) -> pd.DataFrame:
    """
    Aggregate flow-level DataFrame to one row per channel.

    Point 1: Groups by cfg.pair.channel_key (default: src_ip, dst_ip,
    dst_port, proto) rather than just (src_ip, dst_ip).

    Point 4: Computes persistence_ratio, iat_mad_s, iat_ratio,
    missing_beat_rate, req_resp_asymmetry, and zero_payload_frac
    in addition to the original 8 IForest features.

    Parameters
    ----------
    df  : Output of loaders.load_and_prepare() — one row per flow.
    cfg : Pipeline configuration.

    Returns
    -------
    DataFrame with one row per channel.
    """
    min_flows   = cfg.pair.min_pair_flows
    channel_key = list(cfg.pair.channel_key)

    # Resolve which key columns are actually present in df.
    # dst_port may appear as 'dst_p' and proto as 'service' depending on schema.
    col_aliases = {
        "dst_port": ["dst_port", "dst_p", "id.resp_p", "destination.port"],
        "proto":    ["proto", "service", "network.protocol", "network.transport"],
    }
    resolved_key: list[str] = []
    for k in channel_key:
        if k in df.columns:
            resolved_key.append(k)
        elif k in col_aliases:
            for alias in col_aliases[k]:
                if alias in df.columns:
                    resolved_key.append(alias)
                    break
            else:
                log.debug("channel_key component '%s' not found in df — omitting", k)
        else:
            log.debug("channel_key component '%s' not found in df — omitting", k)

    # Ensure src_ip and dst_ip are always in the key
    for required in ("src_ip", "dst_ip"):
        if required not in resolved_key:
            resolved_key.insert(0, required)

    # Deduplicate while preserving order
    seen: set = set()
    resolved_key = [k for k in resolved_key if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]

    log.info("aggregate_pairs(): channel key = %s", resolved_key)

    records = []
    grouped = df.groupby(resolved_key, sort=False)

    for key_vals, grp in grouped:
        n = len(grp)
        if n < min_flows:
            continue

        grp = grp.sort_values("timestamp")

        # --- Channel identity ---
        if isinstance(key_vals, tuple):
            kv = key_vals
        else:
            kv = (key_vals,)

        rec: dict = {}
        for col, val in zip(resolved_key, kv):
            rec[col] = val

        # Ensure src_ip and dst_ip are always present for downstream
        rec.setdefault("src_ip", kv[0])
        rec.setdefault("dst_ip", kv[1] if len(kv) > 1 else "")

        # Channel ID encodes all key components
        rec["channel_id"] = _channel_id(kv, tuple(resolved_key))
        # Legacy pair_id alias (src→dst) kept for backward compat
        rec["pair_id"]    = f"{rec['src_ip']}→{rec['dst_ip']}"

        # --- IAT sequence ---
        ts_num = _ts_to_float(grp["timestamp"])
        iat    = np.diff(np.sort(ts_num.values))
        iat    = iat[iat > 0]

        if len(iat) >= 2:
            iat_mean = float(np.mean(iat))
            iat_med  = float(np.median(iat))
            iat_cv   = float(np.std(iat) / iat_mean) if iat_mean > 0 else np.nan
            # Point 4: MAD and ratio
            iat_mad  = float(np.median(np.abs(iat - iat_med)))
            iat_ratio = float(iat_med / iat_mean) if iat_mean > 0 else np.nan
        else:
            iat_mean = iat_med = iat_cv = iat_mad = iat_ratio = np.nan

        # --- Point 4: persistence ratio (fraction of window days with activity) ---
        first_ts = float(ts_num.iloc[0])
        last_ts  = float(ts_num.iloc[-1])
        window_days = max((last_ts - first_ts) / 86400.0, 1.0)
        active_days = ts_num.apply(lambda t: int(t // 86400)).nunique()
        persistence_ratio = float(active_days / window_days)

        # --- Point 4: missing beat rate ---
        # Estimate using period ≈ median IAT; count beacon slots with no firing.
        if iat_med and iat_med > 0 and len(iat) >= 4:
            expected_beats = max(int(round((last_ts - first_ts) / iat_med)), 1)
            actual_beats   = len(grp)
            missing_beat_rate = float(max(0, expected_beats - actual_beats) / expected_beats)
        else:
            missing_beat_rate = np.nan

        # --- Bytes ---
        b = (grp["destination.bytes"] if "destination.bytes" in grp.columns
             else grp.get("total_bytes", pd.Series([0] * n, index=grp.index)))
        bytes_mean = float(b.mean())
        bytes_cv   = float(b.std() / bytes_mean) if bytes_mean > 0 else 0.0

        # --- Point 4: request/response asymmetry ---
        src_bytes_col = next(
            (c for c in ("source.bytes", "src_bytes") if c in grp.columns), None
        )
        if src_bytes_col and bytes_mean > 0:
            src_b  = grp[src_bytes_col].fillna(0).values.astype(float)
            dst_b  = b.fillna(0).values.astype(float)
            asym   = np.abs(src_b - dst_b) / (src_b + dst_b + 1.0)
            req_resp_asymmetry = float(np.median(asym))
        else:
            req_resp_asymmetry = np.nan

        # --- Point 4: zero-payload fraction ---
        zero_payload_frac = float((b == 0).mean()) if len(b) > 0 else 0.0

        # --- Duration ---
        dur_col = next(
            (c for c in ("duration", "event.duration") if c in grp.columns), None
        )
        if dur_col:
            dur_vals = grp[dur_col].dropna()
            dur_mean = float(dur_vals.mean()) if len(dur_vals) > 0 else 0.0
            dur_cv   = float(dur_vals.std() / dur_mean) if dur_mean > 0 else 0.0
        else:
            dur_cv = np.nan

        # --- Connection state entropy ---
        state_col = next(
            (c for c in ("conn_state", "network.connection.state") if c in grp.columns), None
        )
        conn_entropy = _shannon_entropy(grp[state_col]) if state_col else 0.0

        # --- Time-of-day encoding ---
        sin_mean = float(grp["sin_time"].mean()) if "sin_time" in grp.columns else 0.0
        cos_mean = float(grp["cos_time"].mean()) if "cos_time" in grp.columns else 1.0

        rec.update({
            "n_flows":             n,
            "log_n_flows":         float(np.log1p(n)),
            "iat_mean_s":          iat_mean,
            "iat_log_mean":        float(np.log1p(iat_mean)) if not np.isnan(iat_mean) else np.nan,
            "iat_cv":              iat_cv,
            "iat_mad_s":           iat_mad,           # Point 4
            "iat_ratio":           iat_ratio,          # Point 4
            "missing_beat_rate":   missing_beat_rate,  # Point 4
            "persistence_ratio":   persistence_ratio,  # Point 4
            "bytes_mean":          bytes_mean,
            "log_bytes_mean":      float(np.log1p(bytes_mean)),   # log-transform: bytes_mean is heavily right-skewed
            "bytes_cv":            bytes_cv,
            "req_resp_asymmetry":  req_resp_asymmetry, # Point 4
            "zero_payload_frac":   zero_payload_frac,  # Point 4
            "duration_cv":         dur_cv,
            "conn_state_entropy":  conn_entropy,
            "sin_time_mean":       sin_mean,
            "cos_time_mean":       cos_mean,
            "first_seen":          first_ts,
            "last_seen":           last_ts,
        })

        if "scenario" in grp.columns:
            rec["scenario"] = grp["scenario"].iloc[0]

        records.append(rec)

    pair_df = pd.DataFrame(records)
    log.info(
        "aggregate_pairs(): %d flows → %d channels (key=%s, min_flows=%d)",
        len(df), len(pair_df), resolved_key, min_flows,
    )
    return pair_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# IForest feature set
# ---------------------------------------------------------------------------

IFOREST_FEATURES = [
    # Volume
    "log_n_flows",          # log1p(n_flows) — right-skew corrected at source

    # IAT regularity  (primary beacon discriminator)
    "iat_cv",               # coefficient of variation — bounded [0, ∞), typically < 3
    "iat_log_mean",         # log1p(iat_mean_s) — right-skew corrected at source
    "iat_mad_s",            # median absolute deviation — robust to burst outliers
    "iat_ratio",            # median/mean IAT — near 1.0 = symmetric timing

    # Beacon gap signal
    "missing_beat_rate",    # fraction of expected slots with no firing

    # Persistence
    "persistence_ratio",    # fraction of observation days with activity

    # Payload
    "log_bytes_mean",       # log1p(bytes_mean) — right-skew corrected at source
                            # replaces raw bytes_mean which spans 0–10M+
    "bytes_cv",

    # Payload character
    "req_resp_asymmetry",   # |src-dst|/(src+dst+1) — naturally [0, 1]
    "zero_payload_frac",    # fraction of zero-byte flows — naturally [0, 1]

    # Connection consistency
    "duration_cv",
    "conn_state_entropy",

    # Temporal (cyclic, naturally bounded [-1, 1])
    "sin_time_mean",
    "cos_time_mean",
]

# Features that are protected from skew-transform (already bounded or cyclic)
_PROTECTED_FROM_TRANSFORM = frozenset({
    "iat_cv", "iat_ratio", "missing_beat_rate", "persistence_ratio",
    "bytes_cv", "req_resp_asymmetry", "zero_payload_frac",
    "duration_cv", "conn_state_entropy",
    "sin_time_mean", "cos_time_mean",
    # already log-transformed at aggregation time:
    "log_n_flows", "iat_log_mean", "log_bytes_mean",
})


# ---------------------------------------------------------------------------
# EDA validation and pre-scaling transforms  (Fix 1 — wires ScalingConfig)
# ---------------------------------------------------------------------------

def validate_and_transform_features(
    pair_df: pd.DataFrame,
    cfg: BDPConfig,
    feature_medians: Optional[dict] = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Inspect each candidate IForest feature and apply necessary transforms
    before StandardScaler.  Implements the ScalingConfig thresholds that
    previously existed in config but were never called.

    Steps (in order)
    ----------------
    1. Median imputation  — replace NaN with per-feature median rather than 0.
       Avoids treating missing values as a strong signal (Fix 3).

    2. Skewness check  — features with |skew| > cfg.scaling.skew_threshold
       that are not already log-transformed receive an additional log1p pass.
       Applied only to non-negative features not in _PROTECTED_FROM_TRANSFORM
       (Fix 2 generalised).

    3. Near-zero variance drop  — features where variance < cfg.scaling.binary_threshold
       carry no discriminative signal and waste IForest splits.

    4. Range ratio check  — features where max/min > cfg.scaling.range_ratio_threshold
       after skew correction are flagged. If they also have fewer than
       cfg.scaling.min_unique unique values the column is dropped as near-constant.

    Parameters
    ----------
    pair_df        : Output of aggregate_pairs().
    cfg            : Pipeline configuration.
    feature_medians: If provided (inference mode), use stored medians for
                     imputation rather than computing from data.

    Returns
    -------
    out            : DataFrame with transformed feature columns.
    active_features: List of feature names that survived all filters
                     (used as IForest input).
    medians        : Dict mapping feature → median value (store for inference).
    """
    scl = cfg.scaling
    candidates = [f for f in IFOREST_FEATURES if f in pair_df.columns]
    out = pair_df.copy()

    dropped:     list[str] = []
    transformed: list[str] = []
    medians:     dict      = {}

    for feat in candidates:
        col = out[feat].astype(float)

        # ── Step 1: Median imputation ──────────────────────────────────────
        if feature_medians is not None:
            med = feature_medians.get(feat, 0.0)
        else:
            med = float(col.median()) if col.notna().any() else 0.0
        medians[feat] = med

        n_nan = col.isna().sum()
        if n_nan > 0:
            col = col.fillna(med)
            log.debug(
                "  [impute] %s: filled %d NaNs with median=%.4f",
                feat, n_nan, med,
            )
        out[feat] = col

        # ── Step 2: Skewness check ─────────────────────────────────────────
        if feat not in _PROTECTED_FROM_TRANSFORM and col.notna().sum() >= 4:
            skew = float(col.skew())
            if abs(skew) > scl.skew_threshold and col.min() >= 0:
                new_feat = f"{feat}_log"
                out[new_feat] = np.log1p(col)
                # replace original in active set
                candidates[candidates.index(feat)] = new_feat
                medians[new_feat] = float(out[new_feat].median())
                del medians[feat]
                out.drop(columns=[feat], inplace=True, errors="ignore")
                transformed.append(f"{feat} → {new_feat}  (skew={skew:.2f})")
                feat = new_feat
                col  = out[feat]

        # ── Step 3: Near-zero variance drop ───────────────────────────────
        variance = float(col.var())
        if variance < scl.binary_threshold:
            dropped.append(f"{feat}  (var={variance:.6f} < {scl.binary_threshold})")
            out.drop(columns=[feat], inplace=True, errors="ignore")
            candidates.remove(feat)
            continue

        # ── Step 4: Range ratio + unique-value check ───────────────────────
        col_min = float(col.min())
        col_max = float(col.max())
        if col_min > 0:
            rr = col_max / col_min
        else:
            rr = col_max - col_min   # use range if min == 0

        n_unique = col.nunique()
        if rr > scl.range_ratio_threshold and n_unique < scl.min_unique:
            dropped.append(
                f"{feat}  (range_ratio={rr:.1f} > {scl.range_ratio_threshold},"
                f" unique={n_unique} < {scl.min_unique})"
            )
            out.drop(columns=[feat], inplace=True, errors="ignore")
            candidates.remove(feat)
            continue

    # ── Summary log ───────────────────────────────────────────────────────
    if transformed:
        log.info("validate_and_transform_features(): log-transformed %d feature(s):", len(transformed))
        for t in transformed:
            log.info("  ↳ %s", t)
    if dropped:
        log.info("validate_and_transform_features(): dropped %d feature(s):", len(dropped))
        for d in dropped:
            log.info("  ✗ %s", d)
    log.info(
        "validate_and_transform_features(): %d features active after EDA  "
        "(%d transformed, %d dropped)",
        len(candidates), len(transformed), len(dropped),
    )

    return out, candidates, medians


# ---------------------------------------------------------------------------
# Scaling for IForest  (Fix 3 — median imputation + wired ScalingConfig)
# ---------------------------------------------------------------------------

class CadenceScaler:
    """
    Wraps StandardScaler with per-feature median imputation and the EDA
    transform record so the same decisions can be replayed at inference time.

    Attributes
    ----------
    scaler          : Fitted StandardScaler.
    active_features : Feature names passed to the scaler (post-EDA).
    medians         : Per-feature median values for NaN imputation.
    """

    def __init__(self) -> None:
        self.scaler:          StandardScaler   = StandardScaler()
        self.active_features: list[str]        = []
        self.medians:         dict             = {}

    def fit_transform(
        self,
        pair_df: pd.DataFrame,
        cfg: BDPConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run EDA validation, fit StandardScaler, return scaled DataFrame.

        Returns
        -------
        out        : pair_df with *_stdz columns appended.
        X_raw      : The un-scaled but EDA-cleaned feature matrix
                     (useful for SHAP with raw feature values).
        """
        out, self.active_features, self.medians = validate_and_transform_features(
            pair_df, cfg, feature_medians=None
        )

        X = out[self.active_features].fillna(0).astype(float)
        X_scaled = self.scaler.fit_transform(X)

        for i, feat in enumerate(self.active_features):
            out[f"{feat}_stdz"] = X_scaled[:, i]

        log.info(
            "CadenceScaler.fit_transform(): %d channels, %d features → StandardScaler",
            len(out), len(self.active_features),
        )
        return out, X

    def transform(
        self,
        pair_df: pd.DataFrame,
        cfg: BDPConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply stored EDA decisions and scaler to new data (inference)."""
        out, _, _ = validate_and_transform_features(
            pair_df, cfg, feature_medians=self.medians
        )
        # Align to features seen at fit time
        present = [f for f in self.active_features if f in out.columns]
        X = out[present].fillna(0).astype(float)

        if len(present) < len(self.active_features):
            missing = set(self.active_features) - set(present)
            log.warning("transform(): %d features absent at inference: %s", len(missing), missing)

        X_scaled = self.scaler.transform(X)
        for i, feat in enumerate(present):
            out[f"{feat}_stdz"] = X_scaled[:, i]

        return out, X


def scale_pair_features(
    pair_df: pd.DataFrame,
    cfg: BDPConfig,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, CadenceScaler]:
    """
    EDA validation + Z-score standardisation of channel-level IForest features.

    Replaces the old fillna(0) + bare StandardScaler approach with:
      - Median imputation  (NaN → per-feature median, not 0)
      - Skewness-triggered log1p transform for non-protected features
      - Near-zero variance drop
      - Range ratio / cardinality guard
      - StandardScaler on the surviving, clean feature set

    Parameters
    ----------
    scaler : If None, fit a new CadenceScaler (training mode).
             If a CadenceScaler, apply stored transforms (inference mode).

    Returns
    -------
    out            : pair_df with *_stdz columns appended.
    cadence_scaler : Fitted CadenceScaler (store on BDPArtifacts).
    """
    if scaler is None:
        cadence_scaler = CadenceScaler()
        out, _ = cadence_scaler.fit_transform(pair_df, cfg)
    else:
        # Accept either the new CadenceScaler or legacy sklearn StandardScaler
        if isinstance(scaler, CadenceScaler):
            cadence_scaler = scaler
            out, _ = cadence_scaler.transform(pair_df, cfg)
        else:
            # Legacy path: bare StandardScaler passed in, wrap it minimally
            log.warning("scale_pair_features(): received legacy StandardScaler — "
                        "skipping EDA validation, falling back to fillna(0).")
            cadence_scaler = CadenceScaler()
            cadence_scaler.scaler = scaler
            features = [f for f in IFOREST_FEATURES if f in pair_df.columns]
            cadence_scaler.active_features = features
            X = pair_df[features].fillna(0).astype(float)
            X_scaled = scaler.transform(X)
            out = pair_df.copy()
            for i, feat in enumerate(features):
                out[f"{feat}_stdz"] = X_scaled[:, i]

    return out, cadence_scaler


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------

def process_features(
    df: pd.DataFrame,
    cfg: BDPConfig,
    heavy_tailed=None,
) -> tuple[pd.DataFrame, CadenceScaler, list, list]:
    """Compatibility shim: aggregate flows to channels, run EDA, and scale."""
    pair_df = aggregate_pairs(df, cfg)
    pair_df_scaled, cadence_scaler = scale_pair_features(pair_df, cfg)
    return pair_df_scaled, cadence_scaler, [], []
