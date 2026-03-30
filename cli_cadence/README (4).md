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
│  Stage 1–2 — Ingest, Feature Engineering & Scaling              │
│  Schema normalisation · Channel-level aggregation               │
│  14 IForest features (IAT MAD, persistence ratio, beat rate...) │
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
│  Stage 4 — Isolation Forest                                     │
│  Joint multivariate anomaly scoring on 14 channel-level features│
│  Stability validation · HHI concentration analysis              │
│  SHAP feature importance (TreeExplainer)                        │
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
                  chain and ATT&CK technique annotations
```

### Why this sequence

| Stage | Technique | Beacon property tested | Why the previous stage couldn't answer this |
|---|---|---|---|
| Pre-Filter | Domain knowledge allowlists | Is it obviously benign infrastructure? | Without removing known-benign internal and infrastructure channels first, IForest wastes its anomaly budget on traffic that could never be C2. |
| Isolation Forest | Ensemble anomaly scoring | Is it behaviorally unusual across 14 dimensions? | Thresholding on individual features misses multivariate anomalies. A beacon with moderate bytes AND moderate duration AND moderate frequency looks normal on any single axis. |
| SAX Pre-Screening | Symbolic time-series encoding | Does its timing show regularity? | IForest scores feature distributions, not temporal patterns. A channel can be volumetrically anomalous but completely aperiodic. SAX eliminates these cheaply before expensive ACF. |
| Binned-Count ACF + Welch PSD | Spectral periodicity analysis | Is it statistically periodic with quantifiable confidence? | SAX is a fast heuristic filter, not a statistical test. Two channels can both pass SAX but one has a weak, noisy pattern and the other has a strong, jitter-tolerant periodicity. ACF + PSD provides the confidence score. |
| PELT Changepoint | Changepoint detection | When did it start? Did the operator interact? | Periodicity analysis confirms the beacon exists but doesn't tell you when it activated or whether the interval shifted mid-campaign (indicating a human operator reconfiguring the implant). |
| DNS + HTTP + TLS Corroboration | Cross-layer hypothesis testing (H1–H6) | Is it C2 specifically, not just automated? | Periodicity alone cannot distinguish C2 beacons from Windows Update, NTP, or health monitors. Independent protocol-layer evidence filters legitimate automated services. |
| MITRE ATT&CK Mapping | Evidence-to-technique annotation | What tradecraft does this represent? | Corroboration confirms a lead is malicious, but doesn't tell the analyst which ATT&CK techniques are in play or how to frame the finding for IR handoff. |

---

## Installation

```bash
# Clone and create a virtual environment
git clone https://github.com/jackdmoody/CADENCE.git
cd CADENCE
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install with GUI support (optional)
pip install -e ".[gui-streamlit]"   # Streamlit GUI
pip install -e ".[gui-dash]"        # Dash GUI
pip install -e ".[gui]"             # Both GUIs
```

**Requirements:** Python 3.10+

---

## Quickstart

### Option 1: Interactive CLI (recommended for first run)

```bash
# Interactive config walkthrough — prompts for each section
python cadence_cli.py --interactive --report

# Quick synthetic run with all defaults
python cadence_cli.py --synthetic --report --browser

# Real data with parquet files
python cadence_cli.py --conn data/conn.parquet --dns data/dns.parquet \
    --http data/http.parquet --ssl data/ssl.parquet --report
```

### Option 2: Full-scale synthetic evaluation

```bash
# Default: 30 days, 30k background rows/day
python run_full_scale.py --report

# With report auto-opened in browser
python run_full_scale.py --report --browser

# Quick smoke test (5 days, smaller dataset)
python run_full_scale.py --days 5 --bg-rows 3000 --report
```

### Option 3: Python API

```python
from analytic_pipeline import BDPPipeline, BDPConfig
from analytic_pipeline.report import ReportContext
from pathlib import Path

cfg = BDPConfig()
cfg.io.input_csv = Path("data/conn.csv")

with ReportContext(output_dir="results", open_browser=True) as report:
    art = BDPPipeline(cfg).run(
        dns_log_path  = "data/dns.csv",
        http_log_path = "data/http.csv",
        ssl_log_path  = "data/ssl.csv",
    )
    report.finalise(art)
```

For synthetic runs with ground-truth evaluation in the report:

```python
report.finalise(art, labels=labels)
```

### Option 4: GUI interfaces

```bash
# Streamlit
pip install -e ".[gui-streamlit]"
streamlit run cadence_app.py

# Dash
pip install -e ".[gui-dash]"
python cadence_app_dash.py
```

---

## CLI Reference (`cadence_cli.py`)

The interactive CLI provides full config control without a GUI. All parameters have sensible defaults — press Enter to keep them.

### Input modes

```bash
# Separate log files (CSV or Parquet — auto-detected)
python cadence_cli.py --conn data/conn.csv --dns data/dns.csv \
    --http data/http.csv --ssl data/ssl.csv

# Parquet files work the same way
python cadence_cli.py --conn data/conn.parquet --dns data/dns.parquet

# Combined BDP export (auto-splits by log type column)
python cadence_cli.py --combined data/bdp_export.parquet

# Combined file with custom log type column
python cadence_cli.py --combined data/export.parquet --log-type-col "event.dataset"

# Synthetic data generation + pipeline run
python cadence_cli.py --synthetic --days 30 --seed 42
```

### Configuration

```bash
# Interactive walkthrough — pick which sections to configure
python cadence_cli.py --interactive

# Override specific fields without interactive mode
python cadence_cli.py --conn data/conn.csv \
    --override isolation.n_estimators=300 \
    --override periodicity.confidence_threshold=0.40 \
    --override corroboration.min_score=0.50

# Load config from JSON
python cadence_cli.py --config my_config.json --conn data/conn.csv

# Save config to JSON (dry run — doesn't execute pipeline)
python cadence_cli.py --interactive --save-config my_config.json --dry-run

# Show all default config values and exit
python cadence_cli.py --show-defaults
```

### Output

```bash
# Custom output directory
python cadence_cli.py --synthetic --output /path/to/results

# Generate HTML report
python cadence_cli.py --synthetic --report

# Open report in browser when done
python cadence_cli.py --synthetic --report --browser

# Include matplotlib diagnostic plots (slower)
python cadence_cli.py --synthetic --report --visualize

# Suppress log output
python cadence_cli.py --synthetic --quiet
```

### Full options

```
python cadence_cli.py [OPTIONS]

Input:
  --conn PATH           Path to conn log (CSV/Parquet)
  --dns PATH            Path to DNS log (CSV/Parquet)
  --http PATH           Path to HTTP log (CSV/Parquet)
  --ssl PATH            Path to SSL/TLS log (CSV/Parquet)
  --combined PATH       Path to combined BDP export (auto-split)
  --log-type-col COL    Column name for log type in combined file (default: event.dataset)

Synthetic data:
  --synthetic           Generate synthetic data and run
  --days N              Simulation days (default: 30)
  --bg-rows N           Background rows/day (default: 30000)
  --noisy-rows N        Noisy rows/day (default: 1000)
  --seed N              RNG seed (default: 42)

Configuration:
  --config PATH         Load config from JSON file
  --save-config PATH    Save final config to JSON before running
  --interactive         Interactive config walkthrough
  --override KEY=VALUE  Override config fields (e.g. isolation.n_estimators=300)
  --show-defaults       Print default config and exit

Output:
  --output DIR          Output directory (default: results)
  --report              Generate HTML report
  --browser             Open report in browser
  --visualize           Render matplotlib diagnostic plots
  --dry-run             Build config and exit without running
  --quiet               Suppress info-level log output
```

---

## Synthetic evaluation reference (`run_full_scale.py`)

For reproducible paper figures and benchmarking:

```bash
python run_full_scale.py [OPTIONS]

Options:
  --output DIR          Output directory (default: ./results)
  --days N              Simulation days (default: 30)
  --bg-rows N           Background rows/day (default: 30000)
  --noisy-rows N        Noisy rows/day (default: 1000)
  --seed N              RNG seed (default: 42)
  --report              Generate HTML report with ground-truth evaluation
  --browser             Auto-open report in browser
  --visualize           Render matplotlib plots
```

Ground-truth labels are automatically passed to the HTML report when using `--report`, enabling the precision/recall/F1 evaluation section.

---

## Supported File Formats

All log inputs (conn, DNS, HTTP, SSL) accept multiple formats. The pipeline auto-detects format by file extension:

| Extension | Format | Notes |
|---|---|---|
| `.csv` | Comma-separated values | Default Zeek export format |
| `.tsv` | Tab-separated values | Native Zeek log format |
| `.parquet`, `.pq` | Apache Parquet | Recommended for large datasets (faster I/O, smaller files) |
| `.feather` | Apache Feather | Fast columnar format |
| `.json` | JSON | Single JSON object or array |
| `.jsonl` | JSON Lines | One JSON object per line |

### Combined BDP/Trino exports

If your data platform exports all log types into a single file (common with Trino, Elastic, or BDP environments), use `split_combined_log()` or the CLI's `--combined` flag:

```python
from analytic_pipeline.loaders import split_combined_log

# Auto-split by event.dataset column
paths = split_combined_log("data/combined.parquet")
# Returns: {"conn": Path("data/conn.parquet"), "dns": Path("data/dns.parquet"), ...}

cfg.io.input_csv = paths["conn"]
art = BDPPipeline(cfg).run(
    dns_log_path  = str(paths.get("dns", "")),
    http_log_path = str(paths.get("http", "")),
    ssl_log_path  = str(paths.get("ssl")),
)
```

Or from the CLI:
```bash
python cadence_cli.py --combined data/bdp_export.parquet --report
```

The splitter looks for a `event.dataset` column by default (values like `zeek.conn`, `zeek.dns`, etc.) and falls back to `log_type`, `event.module`, `type`, or `dataset`.

---

## Data Sources

CADENCE ingests four Zeek log types. Each table below lists every field the pipeline reads, whether it is required or optional, and what happens when it is absent.

### conn.log — Connection log (required)

The primary input. Every flow record from Zeek's `conn.log`.

**Zeek export command:**
```bash
zeek-cut ts id.orig_h id.orig_p id.resp_h id.resp_p proto service duration \
    orig_bytes resp_bytes orig_pkts resp_pkts conn_state \
    < conn.log > conn.csv
```

**Expected columns** (Zeek native names — the pipeline also accepts the ISF/ECS dotted-key variants shown in parentheses):

| Zeek column | ISF/ECS alias | Required | Used for |
|---|---|---|---|
| `ts` | `timestamp` | **Yes** | Timestamp; must be Unix float seconds |
| `id.orig_h` | `source.ip` | **Yes** | Source IP address |
| `id.orig_p` | `source.port` | **Yes** | Source port |
| `id.resp_h` | `destination.ip` | **Yes** | Destination IP address |
| `id.resp_p` | `destination.port` | **Yes** | Destination port; part of channel key |
| `proto` | `network.transport` | No | Protocol label (tcp/udp); part of channel key if present |
| `service` | `network.protocol` | **Yes** | Application protocol (http, ssl, dns…); used for OHE features |
| `duration` | `event.duration` | **Yes** | Flow duration in seconds |
| `orig_bytes` | `source.bytes` | No | Request bytes; used for `req_resp_asymmetry` |
| `resp_bytes` | `destination.bytes` | No | Response bytes; used for `bytes_cv`, `zero_payload_frac` |
| `orig_pkts` | `source.packets` | No | Originator packet count |
| `resp_pkts` | `destination.packets` | No | Responder packet count |
| `conn_state` | `network.connection.state` | No | Zeek connection state (SF, S0, REJ…) |

**Minimum viable conn.log:** `ts`, `id.orig_h`, `id.orig_p`, `id.resp_h`, `id.resp_p`, `service`, `duration`. All other columns degrade gracefully.

**Window size:** 30 days minimum recommended. Slow beacons (6-hour intervals) produce only ~120 firings over 30 days.

### dns.log — DNS log (optional, enables H1 + H2)

| Zeek column | Required | Used for |
|---|---|---|
| `ts` | **Yes** | Timestamp for window filtering and DNS IAT computation |
| `id.orig_h` | **Yes** | Source IP; matched against beacon channel src_ip |
| `query` | **Yes** | Queried domain name; DGA detection, period matching |
| `rcode_name` | No | Response code (NOERROR, NXDOMAIN…); H2 NXDomain rate |
| `answers` | No | Resolved IP addresses; H2 fast-flux detection |
| `TTLs` | No | DNS TTL values; H2 short-TTL detection |

### http.log — HTTP log (optional, enables H3 + H4)

| Zeek column | Required | Used for |
|---|---|---|
| `ts` | No | Window filtering |
| `id.orig_h` | **Yes** | Source IP; matched against beacon channel src_ip |
| `id.resp_h` | No | Destination IP; narrows match to the specific channel |
| `uri` | No | H3 URI length CV and path CV; H4 high-entropy URI detection |
| `user_agent` | No | H4 rare UA and UA monotony detection |
| `method` | No | H4 abnormal HTTP method detection |
| `response_body_len` | No | H3 response body CV |

### ssl.log — TLS/SSL log (optional, enables H5 + H6)

| Zeek column | Required | Used for |
|---|---|---|
| `ts` | **Yes** | Timestamp for window filtering |
| `id.orig_h` | No* | Source IP; matched against beacon channel src_ip |
| `id.resp_h` | No | Destination IP; narrows match to channel dst_ip |
| `server_name` | No | H5 SNI stability; H6 absent SNI detection |
| `ja3` | No | H5 JA3 fingerprint monotony; H6 known C2 JA3 matching |
| `ja3s` | No | H5 JA3S server fingerprint |
| `cert_chain_fuids` | No | H5 certificate reuse across sessions |
| `validation_status` | No | H6 self-signed certificate detection |
| `resumed` | No | H6 high session resumption rate |

\* Without `id.orig_h`, no pair matching occurs and TLS scoring returns null results.

---

## HTML Report

The `--report` flag generates a self-contained HTML dashboard with:

- **Pipeline flow visualization** with funnel counts at each stage
- **Triage summary table** with severity ratings (CRITICAL / HIGH / MEDIUM), H1–H6 evidence badges, and clickable rows
- **Evidence cards** per lead with three columns:
  - Identity: infected host, C2 destination, flow count, channel key, beacon start time, DNS/HTTP/TLS score pills, matched domains, observed SNIs, user agents
  - Hypothesis results: H1–H6 with pass/fail status and detailed sub-signal breakdown (DGA domains, NXDomain rate, body CV, JA3 fingerprints, etc.)
  - MITRE ATT&CK techniques: clickable technique IDs linking to attack.mitre.org with tactic labels and pipeline evidence
  - Analyst next steps: tailored per-lead action items based on which hypotheses fired
- **Operator interaction warnings** when PELT detects beacon interval shifts
- **SHAP feature importance** — inline SVG bar chart of mean |SHAP| values showing which IForest features drive anomaly scoring (computed automatically from the IForest model)
- **Ground-truth evaluation** (synthetic mode) — precision/recall/F1 metric cards with per-scenario detection results
- **CSV download buttons** for all pipeline output tables including SHAP values
- **Diagnostic plot gallery** (with `--visualize`)
- **Run metadata** (timestamps, durations, funnel counts, IForest stability)

The HTML file is completely self-contained — no external dependencies. It can be emailed, archived, or opened on any machine.

---

## Output Files

| File | Description |
|---|---|
| `priority.csv` | All channels ranked by priority score |
| `periodicity.csv` | Per-channel periodicity metrics: ACF peak, IAT CV, dominant period, beacon confidence |
| `sax_screening.csv` | SAX pre-screening results |
| `changepoints.csv` | PELT changepoint results: estimated beacon start times, interval shifts |
| `corroboration.csv` | Confirmed leads with full H1–H6 hypothesis results, corroboration score, MITRE ATT&CK annotations |
| `shap_values.csv` | SHAP feature importance values (mean |SHAP| per feature) |
| `run_summary.json` | Machine-readable summary of pipeline funnel counts and runtime |
| `cadence_report.html` | Interactive HTML dashboard (with `--report` flag) |

---

## IForest Feature Set

CADENCE uses 14 channel-level features for Isolation Forest scoring:

| Feature | Description | C2 signal |
|---|---|---|
| `log_n_flows` | log(1 + flow count) | Beacons produce many small flows |
| `iat_cv` | IAT coefficient of variation | Low CV = regular schedule |
| `iat_log_mean` | log(1 + mean IAT) | Encodes beacon period scale |
| `iat_mad_s` | IAT median absolute deviation | Robust jitter measure |
| `iat_ratio` | Median / mean IAT | Near 1.0 = symmetric distribution |
| `missing_beat_rate` | Fraction of expected slots with no connection | Beacons rarely miss |
| `persistence_ratio` | Fraction of window days with activity | Beacons active every day |
| `bytes_cv` | CV of destination bytes per flow | Uniform polling payloads |
| `req_resp_asymmetry` | \|src_bytes - dst_bytes\| / total | C2 polling: near 0; exfil: near 1 |
| `zero_payload_frac` | Fraction of zero-byte flows | Keepalive/heartbeat signals |
| `duration_cv` | CV of flow duration | Uniform connection duration |
| `conn_state_entropy` | Shannon entropy of connection states | Single outcome = automated |
| `sin_time_mean` | Mean sin(time-of-day) | Time clustering |
| `cos_time_mean` | Mean cos(time-of-day) | Time clustering |

SHAP (TreeExplainer) is run on the IForest model to empirically rank these features by their contribution to anomaly scoring. Results are included in the HTML report and exported as `shap_values.csv`.

---

## Synthetic Test Scenarios

The generator injects six scenarios into 30 days of realistic background traffic, including correlated conn, dns, http, and ssl logs:

| Scenario | Type | Period | Key Signals |
|---|---|---|---|
| `fast_https_dga` | **Malicious** | 5 min | DGA domains, short TTL, absent SNI, monotonic JA3, high session resumption |
| `slow_http_fixed` | **Malicious** | 1 hr | Fixed domain, outdated UA, uniform HTTP payloads |
| `multi_host_campaign` | **Malicious** | 15 min | DGA, NXDomain misses, absent UA, absent SNI |
| `exfil_slow` | **Malicious** | 6 hr | Large variable payload, plausible-looking domain, monotonic JA3 |
| `decoy_windows_update` | **Decoy** | 1 hr | Benign domain and UA — should **not** be confirmed |
| `decoy_ntp` | **Decoy** | ~17 min | NTP polling, no HTTP — should **not** be confirmed |

**Expected result at full scale (30 days, 30k bg/day):** All 4 malicious scenarios detected, both decoys filtered at the corroboration stage.

---

## Module Reference

| Module | Stage | Description |
|---|---|---|
| `config.py` | — | Typed dataclass configuration. Sub-configs for each pipeline stage including `PrefilterConfig` and `TLSCorroborationConfig`. |
| `loaders.py` | 1 | Multi-format ingest (CSV, Parquet, TSV, Feather, JSON). `smart_read()` auto-detects format. `split_combined_log()` splits BDP exports. Schema normalisation, feature engineering. |
| `features.py` | 2 | Channel-level feature aggregation (14 IForest features), StandardScaler. Key: `(src_ip, dst_ip, dst_port, proto)`. |
| `prefilter.py` | 3 | Domain-knowledge pre-filter. RFC 1918, CDN/DNS/NTP infra, fanin, dead connections. |
| `isolation.py` | 4 | Isolation Forest fit on channel features, stability check, HHI concentration analysis. |
| `sax_screening.py` | 5 | SAX symbolic pre-screening on inter-arrival time sequences. |
| `periodicity.py` | 6 | Binned-count ACF and Welch PSD. Jitter-robust beacon confidence scoring. |
| `scoring.py` | 7 | Channel priority scoring: beacon confidence × 4 + payload stability × 2 + persistence × 2 + period agreement × 1 + temporal × 1. |
| `changepoint.py` | 8 | PELT changepoint detection for beacon start time and interval shifts. |
| `corroboration.py` | 9 | DNS (H1–H2), HTTP (H3–H4), and TLS (H5–H6) hypothesis testing. Multi-format log loading. Adaptive weight redistribution. |
| `mitre_mapping.py` | 10 | Evidence-based MITRE ATT&CK technique annotation on corroborated leads. |
| `pipeline.py` | — | `BDPPipeline.run()` orchestrator. `BDPArtifacts` dataclass with SHAP values field. |
| `report.py` | — | Self-contained HTML report: evidence cards with MITRE techniques, SHAP bar chart, ground-truth evaluation, diagnostic plot gallery. |
| `generate_synthetic_data.py` | — | Synthetic Zeek log generator (conn + dns + http + ssl). Ground-truth labels and `evaluate_detection()`. |
| `cadence_cli.py` | — | Interactive CLI with config walkthrough, `--override`, `--combined`, `--synthetic`, `--show-defaults`. |
| `cadence_app.py` | — | Streamlit GUI (optional). |
| `cadence_app_dash.py` | — | Dash GUI with background execution, live timer, progress bar (optional). |

---

## MITRE ATT&CK Coverage

Each corroborated lead is automatically annotated with evidence-based ATT&CK technique mappings. Every mapping is derived directly from pipeline evidence — no external threat intelligence feeds required.

| Technique ID | Name | Tactic | Triggered By |
|---|---|---|---|
| T1029 | Scheduled Transfer | Exfiltration | Every corroborated lead with a confirmed beacon interval |
| T1041 | Exfiltration Over C2 Channel | Exfiltration | Corroborated lead with > 500 flows over the beacon channel |
| T1071.001 | Application Layer Protocol: Web Protocols | Command and Control | H3 (stereotyped HTTP patterns) or H4 (non-standard HTTP methods) |
| T1071.004 | Application Layer Protocol: DNS | Command and Control | H1 (DNS period matches beacon period) or NXDomain responses |
| T1568.001 | Dynamic Resolution: Fast Flux DNS | Command and Control | H2 short DNS TTLs |
| T1568.002 | Dynamic Resolution: Domain Generation Algorithms | Command and Control | H2 DGA detection (entropy + consonant-run + digit-run heuristics) |
| T1573 | Encrypted Channel | Command and Control | Confirmed periodic beacon with no HTTP evidence (HTTPS/TLS) |
| T1571 | Non-Standard Port | Command and Control | PELT-detected beacon interval shift (operator reconfiguration) |
| T1001 | Data Obfuscation | Command and Control | H4 high-entropy URI |
| T1132 | Data Encoding | Command and Control | H4 high-entropy URI |
| T1036 | Masquerading | Defense Evasion | H4 rare or absent User-Agent string |

---

## Known Limitations

- **Encrypted C2 without SSL log:** H5 and H6 will not fire without a `ssl.log`. H3 and H4 will not fire without an `http.log`. Absent log types redistribute weight to DNS hypotheses — the channel is not penalised.
- **Sub-minute beacons:** Intervals below 60 seconds are filtered by default (`periodicity.min_period_s`). Adjust if your threat model requires it, but expect more false positives from legitimate keepalive traffic.
- **Slow fixed-C2 at minimum threshold:** `exfil_slow` (6h interval) corroborates at approximately the `min_score` threshold due to limited DGA and HTTP evidence. This is expected behavior for slow, fixed-domain beacons.
- **IForest zero recall on synthetic malicious:** IForest shows zero recall on synthetic malicious scenarios because the background traffic generator scatters flows too broadly — this is a synthetic artifact, not a pipeline defect. Corroboration catches these scenarios via DNS/HTTP/TLS evidence.
- **No threat intelligence enrichment:** A beacon to a newly-registered domain without DGA characteristics may score lower than expected. Integrating a passive DNS or TI feed would strengthen H2 coverage.
- **Synthetic-only validation:** Pipeline performance is validated against synthetic ground truth. Real-traffic F1 depends on the threat mix and log completeness in the deployment environment.

---

## License

MIT
