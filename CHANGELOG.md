# Changelog

All notable changes to CADENCE are documented here.

---

## [4.0.0] — Current

### Added

- **Streamlit GUI (`cadence_app.py`)** — Browser-based interface exposing all ~40 `BDPConfig`
  fields across 11 tabbed config sections (I/O, Features, Isolation Forest, Pair/SAX,
  Periodicity, PELT, Corroboration, TLS, Prefilter, Triage, Scaling). Includes live log
  console, pipeline funnel metrics, per-output download buttons, and inline SHAP visualizations.
  Run with `streamlit run cadence_app.py`.

- **Parquet ingestion** — `loaders.py` adds `load_parquet()`. `IOConfig` adds `input_parquet`
  field. `load_and_prepare()` priority order: unified → parquet → CSV → ISF.

- **Unified single-file ingestion** — One CSV or Parquet file with a `log_type` column
  (`conn`, `dns`, `http`, `ssl`) can now replace four separate log files. `load_unified()`
  splits by `log_type` and routes each slice automatically. The conn slice flows through the
  standard pipeline; dns/http/ssl slices are stashed on `cfg._unified_slices` and consumed
  by `corroborate()` without requiring separate path arguments.

- **DataFrame-based corroboration loaders** — `load_dns_from_df()`, `load_http_from_df()`,
  `load_ssl_from_df()` in `corroboration.py`. These normalise in-memory DataFrames using the
  same schema/rename/time-filter logic as the path-based loaders, enabling SQL and notebook
  ingestion patterns without writing intermediate files.

- **SHAP explainability** — `isolation.py` adds three functions:
  - `explain_with_shap(model, pair_df)` — runs `shap.TreeExplainer` on the fitted
    IsolationForest and returns a per-pair DataFrame of SHAP values. Explains *why* each
    channel was flagged as anomalous (anomaly detection), distinct from corroboration which
    explains *why* it is believed to be C2 (beacon attribution).
  - `plot_shap_beeswarm(model, pair_df)` — mean |SHAP| summary plot across all pairs.
  - `plot_shap_waterfall(shap_df, pair_id, ...)` — single-pair drill-down waterfall chart.
  - `BDPArtifacts.shap_values` field added to store SHAP output across the pipeline.
  - SHAP CSV added as a download in the GUI.

- **EDA validation and pre-scaling transforms (`CadenceScaler`)** — New
  `validate_and_transform_features()` function and `CadenceScaler` class in `features.py`.
  Previously the four `ScalingConfig` fields (`skew_threshold`, `binary_threshold`,
  `range_ratio_threshold`, `min_unique`) existed in config but were never called. They are
  now wired and enforced:
  1. **Median imputation** — NaN values filled with per-feature median (not 0). Prevents
     `missing_beat_rate = NaN` from being treated as "never misses a beat" (a strong signal).
  2. **Skewness-triggered log1p** — Features with `|skew| > skew_threshold` and non-negative
     values receive an additional log1p pass before StandardScaler. Protected features
     (ratios, fractions, cyclic encodings) are excluded.
  3. **Near-zero variance drop** — Features where `variance < binary_threshold` are removed
     before IForest. They consume tree splits without contributing signal.
  4. **Range ratio guard** — Features where `max/min > range_ratio_threshold` with fewer
     than `min_unique` unique values are dropped as near-constant with extreme outliers.
  - All EDA decisions logged at INFO level for auditability.
  - `CadenceScaler` stores `active_features` and `medians` so the same decisions can be
    replayed at inference time without data leakage.

- **`log_bytes_mean` replaces `bytes_mean` in `IFOREST_FEATURES`** — `bytes_mean` spans
  0 to hundreds of MB and is the most skewed raw feature passed to IForest. It is now
  log-transformed at aggregation time (`log_bytes_mean = log1p(bytes_mean)`) alongside
  the existing `log_n_flows` and `iat_log_mean`. Raw `bytes_mean` is retained in the
  DataFrame for analyst readability but does not enter the IForest feature matrix.

- **`BDPArtifacts.scaler` typed as `CadenceScaler`** — replaces the former
  `Optional[StandardScaler]` annotation. `CadenceScaler` wraps `StandardScaler` and
  exposes `.active_features` and `.medians` for downstream use (e.g., SHAP waterfall
  feature value lookup).

- **`recover_raw_features()` fixed** — Previously used a two-condition strip
  (`not endswith("_stdz") and not endswith("log_stdz")`) that would pass through
  `feature_log_stdz` columns produced by the EDA skew transform. Simplified to
  `not endswith("_stdz")` which correctly catches all scaled variants.

- **Six new tests in `test_pipeline.py`**:
  - `test_log_bytes_mean_replaces_raw_bytes_mean` — asserts IFOREST_FEATURES contains
    `log_bytes_mean` and not `bytes_mean`.
  - `test_cadence_scaler_type` — asserts `art.scaler` is a `CadenceScaler` with populated
    `active_features` and `medians`.
  - `test_no_nan_in_stdz_columns` — asserts median imputation left no NaNs in scaled output.
  - `test_validate_and_transform_drops_zero_variance` — unit test for the variance filter.
  - `test_median_imputation_not_zero_fill` — regression guard against `fillna(0)`.
  - `test_log_bytes_mean_replaces_raw_bytes_mean` — IFOREST_FEATURES contents guard.

### Changed

- `IOConfig` adds `input_parquet` and `input_unified` fields alongside `input_csv`.
- `BDPPipeline.filter()` returns a 5-tuple `(df_annotated, anomalies, model, stability,
  shap_df)`, up from 4. `run()` updated to unpack and store `shap_values` on artifacts.
- `BDPPipeline.corroborate()` accepts both file paths and pre-loaded DataFrames
  (`dns_df`, `http_df`, `ssl_df` kwargs), enabling unified ingestion without temp files.
- `BDPPipeline.run()` auto-detects `cfg._unified_slices` and routes to DataFrame loaders.
- `pyproject.toml` adds `pyarrow>=14.0` and `shap>=0.44` as core dependencies;
  `streamlit>=1.32` moved to `[gui]` optional extra.
- `isolation.py` fallback path (no `_stdz` columns) now emits a warning rather than
  silently proceeding, making CadenceScaler bypass visible in logs.
- `__init__.py` version bumped to `4.0.0`.

### Dependency additions

| Package | Version | Purpose |
|---|---|---|
| `pyarrow` | ≥14.0 | Parquet read/write via `pd.read_parquet()` |
| `shap` | ≥0.44 | IForest TreeExplainer, beeswarm + waterfall plots |
| `streamlit` | ≥1.32 | GUI (optional extra: `pip install cadence-analytic[gui]`) |

---

## [3.0.0]

### Added
- **Channel-level grouping keys** — The analysis unit is now `(src_ip, dst_ip, dst_port, proto)`
  rather than `(src_ip, dst_ip)`. Configurable via `cfg.pair.channel_key`. Downstream stages
  (SAX, periodicity, scoring, corroboration, changepoint) all propagate `channel_id`. Legacy
  `(src_ip, dst_ip)` behaviour is available by setting `channel_key = ("src_ip", "dst_ip")`.
- **Six new IForest features** — `iat_mad_s`, `iat_ratio`, `missing_beat_rate`,
  `persistence_ratio`, `req_resp_asymmetry`, `zero_payload_frac`. Total feature count raised
  from 8 to 14 (15 after `log_bytes_mean` added in v4).
- **TLS/SSL corroboration (H5–H6)** — `load_ssl_logs()`, `score_tls_hypothesis()`. H5 tests
  SNI stability, JA3 monotony, certificate reuse. H6 tests self-signed certs, known C2 JA3
  fingerprints, absent SNI, session resumption abuse. Adaptive weight redistribution across
  available log types.
- **Synthetic SSL log generation** — `SyntheticDataGenerator.generate()` returns a 5-tuple
  `(conn, dns, http, ssl, labels)`.
- **`PrefilterConfig`** — `dst_fanin_threshold` and `failed_conn_threshold` promoted from
  hardcoded constants to `BDPConfig.prefilter`.
- **`TLSCorroborationConfig`** — Nested config dataclass under `corroboration.tls`.
- **Reweighted triage scoring** — beacon confidence ×4, payload stability ×2, persistence ×2,
  period agreement ×1, temporal ×1. Uncommon ports and high volume dropped from score.
- **`acf_nlags` raised from 20 to 40** — Improves ACF coverage for slow beacons (6h+).
- **`min_pair_flows` raised from 3 to 8** — Aligned with `sax.min_observations`.

### Fixed
- `KeyError: 'corroborated'` when no DNS or HTTP logs provided.
- Millisecond timestamp handling in `loaders.py`.
- RFC 5737 documentation ranges treated correctly as external IPs in pre-filter.

---

## [2.0.0]

### Added
- **v8 corroboration robustness fixes** — H1 median IAT + CV gate, best-delta domain
  selection, duty-cycle-scaled DNS observation gate, DNS answer IP validation. H2 digit-run
  DGA detection, rate-normalised NXDomain, fast-flux detection. H3 weighted consistency
  score, trimmed body CV, path-component URI CV. H4 global UA rarity, UA monotony,
  fraction-based URI entropy. H1+H2 overlap bonus. Configurable benign domain allowlist.

### Fixed
- `KeyError: 'corroborated'` when no DNS or HTTP logs provided.
- Millisecond timestamp handling in `loaders.py`.
- RFC 5737 documentation ranges (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`)
  treated correctly as external IPs in pre-filter.

---

## [1.0.0]

- Initial pipeline: Isolation Forest → SAX → ACF/FFT → PELT → DNS/HTTP corroboration
  (H1–H4) → MITRE ATT&CK annotation.
- Synthetic data generator with 4 malicious scenarios + 2 decoys.
- Self-contained HTML report with plot capture.
