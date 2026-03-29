# CADENCE
### C2 Anomaly Detection via Ensemble Network Correlation Evidence

> A multi-stage behavioral analytic for detecting C2 beaconing in Zeek network telemetry. Reduces hundreds of thousands of connection log events to a small set of high-confidence, analyst-actionable leads by sequencing independent detection techniques — each answering a question the previous stage cannot.

---

## The Problem

C2 beacons are designed to look normal. A single beacon connection has unremarkable byte counts, a standard port, and a plausible connection state. No threshold or signature catches it. What makes a beacon detectable is its **behavior over time**: it fires on a fixed schedule, to the same destination, with uniform payloads, and leaves correlated evidence across multiple protocol layers simultaneously.

CADENCE is built around that observation. It doesn't look for a single anomalous flow — it looks for the pattern that only automated, scheduled, malicious processes produce.

---

## How It Works

Each stage answers a specific question that the previous stage cannot:

```
Zeek conn.log, http.log, dns.log, ssl.log  (30 days, ~30k rows/day)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1–2 — Ingest, EDA, Feature Engineering & Scaling         │
│  Schema normalisation · Channel-level aggregation               │
│  EDA validation: median imputation, skew transforms,            │
│  variance filtering, range ratio guards (CadenceScaler)         │
│  15 IForest features (log_bytes_mean, IAT MAD, persistence...)  │
│  945k flows → 166k (src, dst, port, proto) channels             │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: Which channels are obviously benign?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — Domain-Knowledge Pre-Filter                          │
│  RFC 1918 internal-to-internal · Known CDN/DNS/NTP infra        │
│  High-fanin shared services · Dead connection states            │
│  Configurable thresholds via PrefilterConfig                    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: Which channels are volumetrically anomalous?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — Isolation Forest + SHAP                              │
│  Joint multivariate anomaly scoring on 15 channel-level features│
│  SHAP TreeExplainer: per-feature anomaly attribution            │
│  Stability validation · HHI concentration analysis              │
│  ~5% of channels pass through                                   │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: Does this channel's timing show symbolic regularity?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — SAX Pre-Screening                                    │
│  Symbolic Aggregate approXimation on inter-arrival times        │
│  Fast O(N) elimination of non-periodic channels                 │
│  ~35% of anomalous channels pass through                        │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: Is this channel statistically periodic?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6 — Binned-Count ACF + Welch PSD Periodicity Analysis    │
│  Per-channel IAT autocorrelation (binned for jitter robustness) │
│  Spectral density estimation · Corrected period estimation      │
│  Composite beacon confidence scoring (threshold ≥ 0.45)         │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: When did beaconing start? Did the interval change?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 8 — PELT Changepoint Detection                           │
│  Estimated beacon start time · Interval shift detection         │
│  Operator interaction flagging                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: Is there independent cross-layer evidence of C2?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 9 — Multi-Source Corroboration (H1–H6)                   │
│                                                                 │
│  DNS:   H1 Regularity · H2 DGA/NXDomain/FastFlux               │
│  HTTP:  H3 Behavioral Consistency · H4 Evasion Indicators       │
│  TLS:   H5 Session Consistency · H6 TLS Evasion Indicators      │
│                                                                 │
│  Adaptive weight redistribution across available log types      │
│  ~1–5 high-confidence leads                                     │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: What ATT&CK techniques does this map to?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 10 — MITRE ATT&CK Annotation                             │
│  Evidence-based technique mapping (T1071, T1568, T1573, etc.)   │
│  Tactic-grouped output per corroborated lead                    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
  Analyst Brief — corroborated channels with full H1–H6 evidence
                  chain, SHAP anomaly attribution, and ATT&CK annotations
```

### Why this sequence

| Stage | Technique | Beacon property tested | Why the previous stage couldn't answer this |
|---|---|---|---|
| Pre-Filter | Domain knowledge allowlists | Is it obviously benign infrastructure? | Without removing known-benign internal and infrastructure channels first, IForest wastes its anomaly budget on traffic that could never be C2. |
| Isolation Forest | Ensemble anomaly scoring | Is it behaviorally unusual across 15 dimensions? | Thresholding on individual features misses multivariate anomalies. A beacon with moderate bytes AND moderate duration AND moderate frequency looks normal on any single axis. |
| SAX Pre-Screening | Symbolic time-series encoding | Does its timing show regularity? | IForest scores feature distributions, not temporal patterns. A channel can be volumetrically anomalous but completely aperiodic. SAX eliminates these cheaply before expensive ACF. |
| Binned-Count ACF + Welch PSD | Spectral periodicity analysis | Is it statistically periodic with quantifiable confidence? | SAX is a fast heuristic filter, not a statistical test. Two channels can both pass SAX but one has a weak, noisy pattern and the other has a strong, jitter-tolerant periodicity. ACF + PSD provides the confidence score. |
| PELT Changepoint | Changepoint detection | When did it start? Did the operator interact? | Periodicity analysis confirms the beacon exists but doesn't tell you when it activated or whether the interval shifted mid-campaign. |
| DNS + HTTP + TLS Corroboration | Cross-layer hypothesis testing (H1–H6) | Is it C2 specifically, not just automated? | Periodicity alone cannot distinguish C2 beacons from Windows Update, NTP, or health monitors. Independent protocol-layer evidence filters legitimate automated services. |
| MITRE ATT&CK Mapping | Evidence-to-technique annotation | What tradecraft does this represent? | Corroboration confirms a lead is malicious, but doesn't tell the analyst which ATT&CK techniques are in play or how to frame the finding for IR handoff. |

---

## Installation

```bash
git clone https://github.com/jackdmoody/CADENCE.git
cd CADENCE
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Core pipeline
pip install -e .

# Core + Streamlit GUI
pip install -e ".[gui]"
```

**Requirements:** Python 3.10+

---

## Quickstart

### GUI — Streamlit

```bash
pip install -e ".[gui-streamlit]"
streamlit run cadence_app.py
```

### GUI — Dash

```bash
pip install -e ".[gui-dash]"
python cadence_app_dash.py
# Opens at http://127.0.0.1:8050

# Custom port or debug mode
python cadence_app_dash.py --port 8051 --debug
```

Both GUIs expose identical features: three input modes (synthetic data, unified file, separate
files), all ~40 `BDPConfig` fields, live log streaming, pipeline funnel metrics, SHAP
visualizations, and download buttons for all outputs.

| | Streamlit (`cadence_app.py`) | Dash (`cadence_app_dash.py`) |
|---|---|---|
| Start command | `streamlit run cadence_app.py` | `python cadence_app_dash.py` |
| Install | `pip install -e ".[gui-streamlit]"` | `pip install -e ".[gui-dash]"` |
| Execution model | Reruns full script on each interaction | Callback-only updates, no full rerun |
| Long runs | Blocks UI thread during pipeline | Background thread + interval polling |
| Debug mode | `streamlit run cadence_app.py --logger.level debug` | `python cadence_app_dash.py --debug` |

| Mode | Description |
|---|---|
| **Synthetic data** | Generate conn, dns, http, ssl in-memory — no files needed. Includes ground-truth evaluation. |
| **Unified file** | One CSV or Parquet with a `log_type` column (`conn`, `dns`, `http`, `ssl`). All log slices are extracted automatically. |
| **Separate files** | Individual CSV or Parquet files per log type. |

### Command line (synthetic validation)

```bash
# Default: 30 days, 30k background rows/day
python run_full_scale.py

# With HTML analyst report
python run_full_scale.py --report

# Auto-open report in browser
python run_full_scale.py --report --browser

# Quick smoke test
python run_full_scale.py --days 5 --bg-rows 3000 --report
```

### Python API

```python
from analytic_pipeline import BDPPipeline, BDPConfig
from pathlib import Path

cfg = BDPConfig()
cfg.io.input_csv = Path("data/conn.csv")   # or input_parquet / input_unified

art = BDPPipeline(cfg).run(
    dns_log_path  = "data/dns.csv",
    http_log_path = "data/http.csv",
    ssl_log_path  = "data/ssl.csv",
)

from analytic_pipeline.corroboration import print_analyst_brief
print_analyst_brief(art.corroboration)

# SHAP — explain why IForest flagged each channel
print(art.shap_values.sort_values("shap_sum", ascending=False).head())
```

### Unified single-file ingestion

If your data pipeline produces a single table with a `log_type` column:

```python
cfg = BDPConfig()
cfg.io.input_unified = Path("data/all_logs.parquet")  # or .csv

art = BDPPipeline(cfg).run()  # no separate log path args needed
```

The `log_type` column must contain the values `conn`, `dns`, `http`, and/or `ssl`.
Missing log types are handled gracefully — corroboration weight redistributes automatically.

### HTML report

```python
from analytic_pipeline.report import ReportContext

with ReportContext(output_dir="results", open_browser=True) as report:
    art = BDPPipeline(cfg).run(
        dns_log_path  = "data/dns.csv",
        http_log_path = "data/http.csv",
        ssl_log_path  = "data/ssl.csv",
    )
    report.finalise(art)
```

---

## Data Sources

CADENCE ingests four Zeek log types exported as CSV or Parquet.

---

### conn.log — Connection log (required)

**Zeek export:**
```bash
zeek-cut ts id.orig_h id.orig_p id.resp_h id.resp_p proto service duration \
    orig_bytes resp_bytes orig_pkts resp_pkts conn_state \
    < conn.log > conn.csv
```

| Zeek column | ISF/ECS alias | Required | Used for |
|---|---|---|---|
| `ts` | `timestamp` | **Yes** | Timestamp; Unix float seconds |
| `id.orig_h` | `source.ip` | **Yes** | Source IP |
| `id.orig_p` | `source.port` | **Yes** | Source port |
| `id.resp_h` | `destination.ip` | **Yes** | Destination IP |
| `id.resp_p` | `destination.port` | **Yes** | Destination port; part of channel key |
| `proto` | `network.transport` | No | Protocol label; part of channel key if present |
| `service` | `network.protocol` | **Yes** | Application protocol; OHE features |
| `duration` | `event.duration` | **Yes** | Flow duration; `duration_cv` IForest feature |
| `orig_bytes` | `source.bytes` | No | `req_resp_asymmetry` |
| `resp_bytes` | `destination.bytes` | No | `log_bytes_mean`, `bytes_cv`, `zero_payload_frac` |
| `orig_pkts` | `source.packets` | No | Packet count |
| `resp_pkts` | `destination.packets` | No | Packet count |
| `conn_state` | `network.connection.state` | No | `conn_state_entropy`, dead-pair pre-filter |

**Minimum viable:** `ts`, `id.orig_h`, `id.orig_p`, `id.resp_h`, `id.resp_p`, `service`, `duration`.

**Window size:** 30 days minimum. Slow beacons (6h interval) produce ~120 firings over 30 days; shorter windows degrade recall.

---

### dns.log — DNS log (optional, enables H1 + H2)

```bash
zeek-cut ts id.orig_h query rcode_name answers TTLs < dns.log > dns.csv
```

| Zeek column | Required | Used for |
|---|---|---|
| `ts` | **Yes** | Window filtering, DNS IAT computation |
| `id.orig_h` | **Yes** | Matched against beacon channel `src_ip` |
| `query` | **Yes** | DGA detection, period matching |
| `rcode_name` | No | H2 NXDomain rate |
| `answers` | No | H1 IP validation, H2 fast-flux detection |
| `TTLs` | No | H2 short-TTL detection |

---

### http.log — HTTP log (optional, enables H3 + H4)

```bash
zeek-cut ts id.orig_h id.resp_h uri user_agent method \
    request_body_len response_body_len status_code < http.log > http.csv
```

| Zeek column | Required | Used for |
|---|---|---|
| `id.orig_h` | **Yes** | Matched against beacon channel `src_ip` |
| `uri` | No | H3 URI length CV, path CV; H4 high-entropy URI |
| `user_agent` | No | H4 rare UA and UA monotony |
| `method` | No | H4 abnormal HTTP method |
| `response_body_len` | No | H3 response body CV (trimmed 5th–95th pct) |

---

### ssl.log — TLS/SSL log (optional, enables H5 + H6)

```bash
zeek-cut ts id.orig_h id.resp_h server_name ja3 ja3s \
    cert_chain_fuids validation_status resumed established < ssl.log > ssl.csv
```

| Zeek column | Required | Used for |
|---|---|---|
| `ts` | **Yes** | Window filtering |
| `id.orig_h` | No* | Matched against beacon channel `src_ip` |
| `server_name` | No | H5 SNI stability; H6 absent SNI |
| `ja3` | No | H5 JA3 monotony; H6 known C2 JA3 matching |
| `cert_chain_fuids` | No | H5 certificate reuse |
| `validation_status` | No | H6 self-signed cert |
| `resumed` | No | H6 high session resumption |

**JA3/JA3S requires:** `zkg install zeek/salesforce/ja3`

---

## CLI Reference

```
python run_full_scale.py [OPTIONS]

  --output      Output directory           (default: ./results)
  --days        Simulation days            (default: 30)
  --bg-rows     Background rows per day    (default: 30000)
  --noisy-rows  Noisy rows per day         (default: 1000)
  --seed        RNG seed                   (default: 42)
  --report      Generate HTML report       (off by default)
  --browser     Auto-open report           (off by default)
  --visualize   Render matplotlib plots    (off by default)
```

---

## Configuration

Export defaults to JSON and edit:

```python
from analytic_pipeline import BDPConfig
BDPConfig().to_json("config.json")
```

Load a saved config:

```python
cfg = BDPConfig.from_json("config.json")
```

The GUI also provides **Save config** and **Load config** buttons that serialize the current
widget state to/from JSON.

---

### Feature Engineering & Scaling (`scaling`)

EDA validation runs between channel aggregation and StandardScaler. All four `ScalingConfig`
thresholds are active — previously they existed in config but were never called.

| Parameter | Default | Justification |
|---|---|---|
| `scaling.skew_threshold` | `2.0` | Features with `\|skew\| > 2.0` receive an additional `log1p` pass before StandardScaler. Byte counts and raw packet counts are heavily right-skewed; without this, a single whale channel with 500MB traffic dominates IForest splits. Protected features (ratios, fractions, cyclic encodings, already-log-transformed columns) are excluded. |
| `scaling.binary_threshold` | `0.001` | Features with variance below this are dropped. Near-zero-variance columns carry no discriminative signal but consume IForest tree splits. |
| `scaling.range_ratio_threshold` | `100.0` | Features where `max/min > 100` AND fewer than `min_unique` unique values are dropped. Guards against outlier-dominated columns that StandardScaler cannot normalise effectively. |
| `scaling.min_unique` | `10` | Minimum unique values required alongside a high range ratio to retain a feature. |

**NaN imputation:** NaN values are filled with the per-feature median (not 0). `missing_beat_rate = NaN` imputing to 0 was wrong — 0 means "never misses a beat," which is a strong beacon signal, not a neutral default. Medians are stored on `CadenceScaler` for inference-time consistency.

**`log_bytes_mean`:** Raw `bytes_mean` spans 0–hundreds of MB and is the most skewed feature entering IForest. It is log-transformed at aggregation time (`log_bytes_mean = log1p(bytes_mean)`) alongside `log_n_flows` and `iat_log_mean`. Raw `bytes_mean` is retained for analyst readability but does not enter the IForest feature matrix.

---

### Channel Grouping (`pair`)

| Parameter | Default | Justification |
|---|---|---|
| `pair.channel_key` | `(src_ip, dst_ip, dst_port, proto)` | 4-tuple prevents mixing traffic to different services on the same destination host. Revert to `(src_ip, dst_ip)` only if you want coarser grouping. |
| `pair.min_pair_flows` | `8` | Minimum flows for a meaningful SAX word and IAT sequence. Aligned with `sax.min_observations` to prevent a dead 3–7 flow range. |
| `pair.max_pairs` | `5000` | Safety cap; applied after sorting by flow count descending so the richest channels are always evaluated. |

---

### Pre-Filter (`prefilter`)

| Parameter | Default | Justification |
|---|---|---|
| `prefilter.dst_fanin_threshold` | `0.50` | Destinations contacted by >50% of unique source IPs are treated as shared infrastructure. C2 servers are contacted by one or a small number of infected hosts. |
| `prefilter.failed_conn_threshold` | `0.90` | Channels with >90% failed connection states are dead channels. Real C2 requires successful handshakes. |

---

### Isolation Forest (`isolation`)

| Parameter | Default | Justification |
|---|---|---|
| `isolation.n_estimators` | `200` | 200 trees provide stable scores. Above 200, returns diminish while training time grows linearly. |
| `isolation.max_samples` | `3000` | Per-tree training subsample. Large enough to capture distributional shape across 15 features; small enough for fast fit. |
| `isolation.contamination` | `0.05` | Fixed at 5% rather than `"auto"`. Fixed contamination ensures a known, bounded anomaly set size regardless of `max_samples` and data distribution. Validate against score distribution density at the threshold. |
| `isolation.stability_threshold` | `0.80` | Minimum train/test score-quantile agreement. Below 0.80 triggers a warning — the anomaly boundary is unstable and results may not reproduce. |
| `isolation.random_state` | `42` | Fixed for reproducibility across runs and validation runs. |

---

### SAX Pre-Screening (`sax`)

SAX is intentionally permissive — its role is fast elimination, not final decision.

| Parameter | Default | Justification |
|---|---|---|
| `sax.word_length` | `20` | Resolution for detecting 2–3 motif repetitions in a 30-day beacon sequence. |
| `sax.alphabet_size` | `4` | Four symbols map IAT values to quartiles. Standard SAX formulation (Lin et al., 2003). |
| `sax.cv_threshold` | `0.60` | Shared with `periodicity.cv_threshold` — a channel failing CV at SAX would also fail at periodicity, so early elimination saves computation. |
| `sax.acf_threshold` | `0.30` | Minimum SAX-level lag-1 autocorrelation. Lower than downstream threshold because discretization attenuates autocorrelation. |
| `sax.motif_threshold` | `0.40` | Fraction of SAX positions forming a repeated motif. Catches channels with brief periodic bursts amid otherwise random traffic. |
| `sax.min_tests_passing` | `2` | 2-of-3 majority vote (CV gate, ACF, motif). Tolerates one weak indicator. |

---

### Periodicity (`periodicity`)

| Parameter | Default | Justification |
|---|---|---|
| `periodicity.acf_nlags` | `40` | Raised from 20. Ensures ACF window spans ≥2 full cycles for slow beacons (6h+ interval). At 20 lags, a 15-min beacon with ~900s median IAT needs 20+ firings to show a significant lag. |
| `periodicity.confidence_threshold` | `0.45` | Intentionally permissive — false positives here get filtered by corroboration. Raising above 0.60 risks missing jittered beacons. |
| `periodicity.min_period_s` | `60` | Sub-minute intervals overlap with legitimate keepalives (TCP keepalive at 30s, DNS TTL refreshes). |
| `periodicity.cv_threshold` | `0.60` | IAT CV gate. Shared with SAX for consistency. |
| `periodicity.acf_significance_threshold` | `0.25` | For 100-observation sequences, 95% confidence bound on white-noise ACF ≈ 0.20. 0.25 provides a small margin above noise. |
| `periodicity.fft_power_ratio_threshold` | `0.15` | ~7.5× concentration above uniform noise floor across N=50 bins. |

---

### PELT Changepoint Detection (`pelt`)

| Parameter | Default | Justification |
|---|---|---|
| `pelt.penalty` | `"bic"` | BIC adapts to sequence length — longer sequences tolerate more changepoints. Preferred over fixed penalties for variable-length IAT sequences. |
| `pelt.min_segment_length` | `5` | Prevents micro-segment noise. A real interval shift needs ≥5 observations at the new rate. |
| `pelt.min_observations` | `15` | Below 15 IAT observations there is insufficient data to estimate both pre- and post-change distributions reliably. |
| `pelt.max_changepoints` | `10` | Real operators reconfigure intervals rarely. More than 10 changepoints almost certainly indicates non-stationary background traffic. |

---

### Corroboration (`corroboration`)

| Parameter | Default | Justification |
|---|---|---|
| `corroboration.min_score` | `0.55` | Final gate. At 0.55, a channel needs evidence from ~two hypotheses (H1+H3, or H2+H4). Lower to 0.45 to surface weak-signal leads; raise to 0.65 for higher precision. |
| `corroboration.dga_entropy_threshold` | `3.5` | Human-readable domains score ~2.5; DGA typically exceeds 3.5. Validated against `fast_https_dga` synthetic scenario. |
| `corroboration.rare_ua_threshold` | `0.05` | User-agents seen in <5% of global HTTP traffic. Measured against the full HTTP log (global frequency), not just the pair. |
| `corroboration.period_tolerance_pct` | `0.15` | DNS query IAT must match conn beacon period within ±15%. Tolerates NTP jitter and DNS caching effects. |
| `corroboration.nxdomain_rate_threshold` | `0.10` | Rate-normalised NXDomain fraction. Avoids false positives from absolute counts on high-query-rate sources. |

---

### TLS Corroboration (`corroboration.tls`)

| Parameter | Default | Justification |
|---|---|---|
| `tls.ja3_monotony_threshold` | `0.90` | >90% of sessions using the same JA3 fingerprint indicates a non-browser client (consistent TLS stack). Cobalt Strike and most C2 frameworks show 100%. |
| `tls.h5_weight` | `0.30` | TLS evidence contributes 30% of the corroboration score when present. Lower than DNS/HTTP because SSL logs are often incomplete in real deployments. |
| `tls.h6_weight` | `0.30` | Same rationale as `h5_weight`. |
| `tls.cert_age_new_days` | `30` | Certificates issued within 30 days flag infrastructure churn common in attacker-controlled C2. Benign services typically carry longer-lived certs. |

---

### Triage (`triage`)

| Parameter | Default | Justification |
|---|---|---|
| `triage.beaconing_std_thresh` | `0.5` | IAT standard deviation below this (in normalised units) is scored as highly regular. Tighter than the periodicity CV gate by design. |
| `triage.rare_dst_thresh` | `25` | Destinations seen in fewer than 25 unique connections across the dataset are flagged as rare. Calibrated against the synthetic 30-day background. |
| `triage.off_hour_range` | `(6, 22)` | Traffic outside 06:00–22:00 scored as suspicious. Adjust for night-shift environments. |

**Triage score weights:** beacon confidence ×4, payload stability ×2, persistence ×2, period agreement ×1, temporal ×1. Maximum score: 10. Uncommon ports and high volume removed from scoring — modern C2 intentionally uses port 443 and small payloads.

---

## SHAP Explainability

SHAP (`shap.TreeExplainer`) runs on the fitted Isolation Forest immediately after Stage 4,
decomposing each channel's anomaly score into per-feature contributions.

```python
# After pipeline.run():
shap_df = art.shap_values

# Channels with the highest total anomaly explanation
top = shap_df.sort_values("shap_sum", ascending=False).head(10)

# Top driver across all anomalous channels
top_feature = shap_df[[c for c in shap_df.columns if c.startswith("shap_")]].abs().mean().idxmax()
```

**Important scope note:** SHAP explains *why a channel was flagged by Isolation Forest* (anomaly
detection), not why it is believed to be C2 (beacon attribution). These are complementary signals:

- **SHAP** → "This channel scored anomalous primarily because `persistence_ratio` and `iat_cv`
  were both in the extreme tail of the distribution."
- **Corroboration (H1–H6)** → "This channel is attributed as C2 because DNS queries match the
  beacon period (H1), the domain is DGA-like (H2), and HTTP payloads are uniform (H3)."

The Streamlit GUI renders both a beeswarm summary plot (mean |SHAP| across all pairs) and a
per-pair waterfall chart (how each feature pushed the anomaly score for a selected lead).

---

## Module Reference

| Module | Stage | Description |
|---|---|---|
| `loaders.py` | 1 | CSV/Parquet/unified ingestion. Schema normalisation, timestamp parsing, cyclic time encoding, OHE. Supports `load_unified()` for single-file multi-log-type ingestion. |
| `features.py` | 2 | Channel aggregation, 15 IForest feature computation, `CadenceScaler` (EDA validation + StandardScaler). |
| `prefilter.py` | 3 | Domain-knowledge pre-filter. RFC 1918, CDN/DNS/NTP infra, fanin, dead connections. |
| `isolation.py` | 4 | Isolation Forest fit, SHAP TreeExplainer, beeswarm + waterfall plots, stability check, HHI concentration analysis. |
| `sax_screening.py` | 5 | SAX symbolic pre-screening on inter-arrival time sequences. |
| `periodicity.py` | 6 | Binned-count ACF and Welch PSD. Jitter-robust beacon confidence scoring. |
| `scoring.py` | 7 | Channel priority scoring: beacon confidence ×4 + payload stability ×2 + persistence ×2 + period agreement ×1 + temporal ×1. |
| `changepoint.py` | 8 | PELT changepoint detection for beacon start time and interval shifts. |
| `corroboration.py` | 9 | DNS (H1–H2), HTTP (H3–H4), TLS (H5–H6) hypothesis testing. DataFrame-based loaders for unified ingestion. Adaptive weight redistribution. |
| `mitre_mapping.py` | 10 | Evidence-based MITRE ATT&CK technique annotation on corroborated leads. |
| `pipeline.py` | — | `BDPPipeline.run()` orchestrator. `BDPArtifacts` dataclass. |
| `config.py` | — | All `BDPConfig` sub-configs as dataclasses. JSON serialization. |
| `report.py` | — | Self-contained HTML report generator. H1–H6 evidence cards, SHAP, channel key display. |
| `generate_synthetic_data.py` | — | Synthetic Zeek log generator (conn + dns + http + ssl). Ground-truth labels. |
| `cadence_app.py` | — | Streamlit GUI. All config fields, file upload, live logs, SHAP plots, downloads. |
| `cadence_app_dash.py` | — | Dash GUI. Identical features to Streamlit; background-thread pipeline, interval-polled log console. Run with `python cadence_app_dash.py`. |

---

## MITRE ATT&CK Coverage

| Technique ID | Name | Tactic | Triggered By |
|---|---|---|---|
| T1029 | Scheduled Transfer | Exfiltration | Every corroborated lead with a confirmed beacon interval |
| T1041 | Exfiltration Over C2 Channel | Exfiltration | Corroborated lead with >500 flows |
| T1071.001 | Application Layer Protocol: Web Protocols | C2 | H3 or H4 |
| T1071.004 | Application Layer Protocol: DNS | C2 | H1 or NXDomain responses |
| T1568.001 | Dynamic Resolution: Fast Flux DNS | C2 | H2 short TTLs |
| T1568.002 | Dynamic Resolution: DGA | C2 | H2 DGA detection |
| T1573 | Encrypted Channel | C2 | Periodic beacon with no HTTP evidence |
| T1571 | Non-Standard Port | C2 | PELT-detected interval shift |
| T1001 | Data Obfuscation | C2 | H4 high-entropy URI |
| T1132 | Data Encoding | C2 | H4 high-entropy URI |
| T1036 | Masquerading | Defense Evasion | H4 rare or absent User-Agent |

**Coverage gaps:** T1090 proxy/redirector chaining, T1102 web service C2, T1219 remote access tools require log sources beyond Zeek. Initial access and lateral movement are out of scope.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest analytic_pipeline/test_pipeline.py -v
```

The test suite uses seed=42 and a 3-day, 2000-row/day synthetic dataset so the full suite completes in under 60 seconds.

---

## Known Limitations

- **Synthetic-only validation:** Pipeline performance is validated against synthetic ground truth. Real-traffic precision/recall depends on threat mix and log completeness.
- **Slow fixed-C2 at minimum threshold:** `exfil_slow` (6h interval) corroborates at approximately `min_score` due to limited DGA and HTTP evidence. Expected behavior for slow fixed-domain beacons.
- **Encrypted C2 without ssl.log:** H5 and H6 do not fire without SSL logs. H3 and H4 require http.log. Absent log types redistribute corroboration weight to DNS hypotheses.
- **Sub-minute beacons:** Intervals below 60s are filtered by default. Adjust `periodicity.min_period_s` if your threat model requires it, but expect more false positives.
- **No threat intelligence enrichment:** A beacon to a newly-registered domain without DGA characteristics may score lower than expected. Integrating a passive DNS or TI feed would strengthen H2 coverage.
- **SHAP scope:** SHAP explains IForest anomaly scoring, not beacon attribution. See the SHAP section above for the distinction.

---

## License

MIT
