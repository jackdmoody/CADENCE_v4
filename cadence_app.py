"""
cadence_app.py
==============
Streamlit GUI for the CADENCE C2 Beacon Detection Pipeline.

Run with:
    streamlit run cadence_app.py

Features
--------
- Upload conn log as CSV or Parquet (separate) OR a unified file with a
  log_type column containing conn / dns / http / ssl rows.
- Expose all BDPConfig fields across collapsible sections.
- Save / load config as JSON.
- Run pipeline with live log streaming.
- View and download HTML analyst report.
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CADENCE",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dependency check — runs before any pipeline import
# ---------------------------------------------------------------------------

_REQUIRED = {
    "pandas":       ("pandas",       "pandas>=2.0"),
    "numpy":        ("numpy",        "numpy>=1.24"),
    "sklearn":      ("scikit-learn", "scikit-learn>=1.3"),
    "scipy":        ("scipy",        "scipy>=1.10"),
    "statsmodels":  ("statsmodels",  "statsmodels>=0.14"),
    "matplotlib":   ("matplotlib",   "matplotlib>=3.7"),
    "pyarrow":      ("pyarrow",      "pyarrow>=14.0"),
    "shap":         ("shap",         "shap>=0.44"),
}

_CADENCE_PKG = Path(__file__).parent / "analytic_pipeline" / "__init__.py"


def _check_dependencies() -> tuple[list[str], list[str]]:
    """Return (missing_packages, install_commands)."""
    missing, cmds = [], []
    for import_name, (pkg_name, pin) in _REQUIRED.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
            cmds.append(pin)
    return missing, cmds


def _check_cadence_package() -> bool:
    return _CADENCE_PKG.exists()


def _auto_install(pins: list[str]) -> tuple[bool, str]:
    """Attempt pip install; return (success, output)."""
    cmd = [sys.executable, "-m", "pip", "install"] + pins
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


missing_pkgs, install_pins = _check_dependencies()
cadence_present = _check_cadence_package()

if missing_pkgs or not cadence_present:
    st.markdown("## ⚙️ CADENCE — Setup Required")

    if not cadence_present:
        st.error(
            "**`analytic_pipeline` package not found.**\n\n"
            "Make sure you are running from the CADENCE repo root:\n"
            "```bash\n"
            "cd /path/to/CADENCE\n"
            "streamlit run cadence_app.py\n"
            "```"
        )

    if missing_pkgs:
        st.warning(
            f"**{len(missing_pkgs)} missing dependenc{'y' if len(missing_pkgs) == 1 else 'ies'} "
            f"detected:** {', '.join(missing_pkgs)}"
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            auto_btn = st.button("⬇️  Install automatically", use_container_width=True)
        with col2:
            st.code("pip install " + " ".join(install_pins), language="bash")

        if auto_btn:
            with st.spinner(f"Installing {', '.join(missing_pkgs)}…"):
                ok, output = _auto_install(install_pins)
            if ok:
                st.success("✅ Installation complete — reloading…")
                st.code(output[-2000:] if len(output) > 2000 else output)
                st.rerun()
            else:
                st.error("❌ Installation failed. Run the command above manually.")
                st.code(output)

    if not cadence_present or missing_pkgs:
        st.stop()   # halt — do not render the rest of the app

# ---------------------------------------------------------------------------
# All deps confirmed present — safe to import
# ---------------------------------------------------------------------------

import pandas as pd

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Main background */
.stApp {
    background-color: #0b0f17;
    color: #c9d1d9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}

/* Headers */
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.5px;
}

h1 { color: #58a6ff; font-size: 1.6rem !important; }
h2 { color: #79c0ff; font-size: 1.1rem !important; }
h3 { color: #8b949e; font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 1px; }

/* Cards / expanders */
[data-testid="stExpander"] {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    margin-bottom: 0.5rem;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: #2ea043;
}

/* Secondary buttons */
.stDownloadButton > button {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
}

/* Log console */
.log-console {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    background-color: #010409;
    color: #58a6ff;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem;
    height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.55;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.75rem 1rem;
}

/* Inputs */
input, select, textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Status tags */
.tag-ok   { color: #3fb950; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.tag-warn { color: #d29922; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.tag-err  { color: #f85149; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }

/* Dividers */
hr { border-color: #21262d; }

/* Number inputs — tighten */
[data-testid="stNumberInput"] > div { gap: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="display:flex; align-items:baseline; gap:12px; margin-bottom:4px;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:1.7rem; color:#58a6ff; font-weight:600;">CADENCE</span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#484f58; letter-spacing:2px;">C2 ANOMALY DETECTION VIA ENSEMBLE NETWORK CORRELATION EVIDENCE</span>
</div>
<hr style="margin-top:6px; margin-bottom:18px;">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "run_logs" not in st.session_state:
    st.session_state.run_logs = ""
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "report_html" not in st.session_state:
    st.session_state.report_html = None
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "syn_labels" not in st.session_state:
    st.session_state.syn_labels = None   # ground-truth labels from synthetic run

# ---------------------------------------------------------------------------
# Sidebar — Data Input
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📂 Data Input")

    input_mode = st.radio(
        "Input mode",
        ["Synthetic data", "Unified file (conn + dns + http + ssl)", "Separate files"],
        help=(
            "Synthetic: generate test data in-memory — no files needed.\n"
            "Unified: one CSV/Parquet with a log_type column.\n"
            "Separate: individual files per log type."
        ),
    )

    if input_mode == "Synthetic data":
        unified_file = conn_file = dns_file = http_file = ssl_file = None
        st.markdown("**Synthetic data parameters**")
        syn_days      = st.slider("Days",                  3,  90, 30,  1, key="syn_days")
        syn_bg_rows   = st.slider("Background rows / day", 1000, 50000, 10000, 1000, key="syn_bg_rows")
        syn_noisy     = st.slider("Noisy rows / day",      100, 5000, 500, 100,  key="syn_noisy")
        syn_seed      = st.number_input("Seed", 0, 9999, 42, 1, key="syn_seed")
        st.caption(
            "Generates conn, dns, http, and ssl logs with 4 malicious "
            "scenarios + 2 decoys. Ground-truth evaluation is shown after the run."
        )

    elif input_mode == "Unified file (conn + dns + http + ssl)":
        syn_days = syn_bg_rows = syn_noisy = syn_seed = None
        unified_file = st.file_uploader(
            "Unified log file (CSV or Parquet)",
            type=["csv", "parquet", "pq"],
            key="unified_upload",
        )
        conn_file = dns_file = http_file = ssl_file = None

    else:
        syn_days = syn_bg_rows = syn_noisy = syn_seed = None
        unified_file = None
        conn_file  = st.file_uploader("Conn log (CSV or Parquet)",  type=["csv","parquet","pq"], key="conn_up")
        dns_file   = st.file_uploader("DNS log (CSV or Parquet)",   type=["csv","parquet","pq"], key="dns_up")
        http_file  = st.file_uploader("HTTP log (CSV or Parquet)",  type=["csv","parquet","pq"], key="http_up")
        ssl_file   = st.file_uploader("SSL log (CSV or Parquet)",   type=["csv","parquet","pq"], key="ssl_up")

    st.divider()
    st.markdown("## ⚙️ Config I/O")

    config_upload = st.file_uploader("Load config JSON", type=["json"], key="cfg_upload")
    if config_upload is not None:
        try:
            loaded_cfg_dict = json.load(config_upload)
            st.session_state["loaded_cfg"] = loaded_cfg_dict
            st.success("Config loaded.")
        except Exception as e:
            st.error(f"Failed to parse config: {e}")

    st.divider()
    st.markdown("## 🚀 Run")

    generate_report = st.checkbox("Generate HTML analyst report", value=True)
    suppress_plots  = st.checkbox("Suppress matplotlib plots",    value=True)

    run_btn = st.button("▶  Run Pipeline", use_container_width=True)


# ---------------------------------------------------------------------------
# Helper: build loaded default or override from uploaded JSON
# ---------------------------------------------------------------------------

def _get_init(section: str, key: str, default):
    """Return value from loaded config JSON if present, else default."""
    loaded = st.session_state.get("loaded_cfg", {})
    return loaded.get(section, {}).get(key, default)


# ---------------------------------------------------------------------------
# Main area — Config tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "I/O", "Features", "Isolation Forest", "Pair / SAX",
    "Periodicity", "PELT", "Corroboration", "TLS", "Prefilter", "Triage", "Scaling",
])

# ── Tab 0: I/O ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### I/O Configuration")
    c1, c2 = st.columns(2)
    with c1:
        io_output_dir  = st.text_input("Output directory",  _get_init("io","output_dir","results"),  key="io_output_dir")
        io_table_name  = st.text_input("Table name",        _get_init("io","table_name","Conn_logs"),key="io_table_name")
        io_query_limit = st.number_input("ISF query limit", min_value=1000, max_value=50_000_000,
                                         value=_get_init("io","query_limit",1_000_000), step=100_000, key="io_query_limit")
    with c2:
        io_query_start = st.text_input("Query start (UTC)", _get_init("io","query_start","2025-10-22 00:00:00"), key="io_qstart")
        io_query_end   = st.text_input("Query end (UTC)",   _get_init("io","query_end",  "2025-10-22 23:59:59"), key="io_qend")
        io_debug       = st.checkbox("Debug mode",          _get_init("io","debug", False), key="io_debug")

# ── Tab 1: Features ─────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Feature Configuration")
    st.caption("These are advanced / rarely changed. Edit only if your schema differs from standard Zeek output.")
    feat_keep_cols = st.text_area(
        "keep_cols (comma-separated)",
        value=", ".join(_get_init("features","keep_cols",(
            "timestamp","datetime","src_ip","src_p","src_pkts",
            "dst_ip","dst_p","resp_pkts","duration",
            "conn_state","service","total_bytes",
            "sin_time","cos_time","hour","minute","scenario",
        ))),
        height=80, key="feat_keep_cols",
    )

# ── Tab 2: Isolation Forest ─────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Isolation Forest")
    c1, c2 = st.columns(2)
    with c1:
        iso_n_est    = st.number_input("n_estimators",   10, 2000,  _get_init("isolation","n_estimators",200),   50,  key="iso_n_est")
        iso_max_samp = st.number_input("max_samples",    100,50000, _get_init("isolation","max_samples",3000),   500, key="iso_max_samp")
        iso_rs       = st.number_input("random_state",   0,  9999,  _get_init("isolation","random_state",42),    1,   key="iso_rs")
    with c2:
        iso_contam   = st.slider("contamination",  0.001, 0.20,  float(_get_init("isolation","contamination",0.05)),  0.005, key="iso_contam")
        iso_test_sz  = st.slider("test_size",      0.1,   0.5,   float(_get_init("isolation","test_size",0.3)),        0.05,  key="iso_test_sz")
        iso_stab     = st.slider("stability_threshold", 0.5, 1.0, float(_get_init("isolation","stability_threshold",0.80)), 0.05, key="iso_stab")

# ── Tab 3: Pair / SAX ───────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Pair Config")
    c1, c2 = st.columns(2)
    with c1:
        pair_min_obs   = st.number_input("min_observations",  2, 100,  _get_init("pair","min_observations",8),  1, key="pair_min_obs")
        pair_min_flows = st.number_input("min_pair_flows",    2, 100,  _get_init("pair","min_pair_flows",8),    1, key="pair_min_flows")
        pair_max_pairs = st.number_input("max_pairs",         10,50000,_get_init("pair","max_pairs",5000),     100,key="pair_max_pairs")
    with c2:
        pair_channel_key = st.multiselect(
            "channel_key",
            ["src_ip","dst_ip","dst_port","proto","src_port"],
            default=list(_get_init("pair","channel_key",["src_ip","dst_ip","dst_port","proto"])),
            key="pair_channel_key",
        )

    st.markdown("### SAX Config")
    c1, c2 = st.columns(2)
    with c1:
        sax_word_len  = st.number_input("word_length",      4, 100, _get_init("sax","word_length",20),  1, key="sax_word_len")
        sax_alpha     = st.number_input("alphabet_size",    2, 26,  _get_init("sax","alphabet_size",4), 1, key="sax_alpha")
        sax_min_obs   = st.number_input("min_observations", 2, 100, _get_init("sax","min_observations",8), 1, key="sax_min_obs")
        sax_max_lag   = st.number_input("max_acf_lag",      2, 50,  _get_init("sax","max_acf_lag",10), 1, key="sax_max_lag")
    with c2:
        sax_cv    = st.slider("cv_threshold",     0.0, 2.0, float(_get_init("sax","cv_threshold",0.60)),  0.05, key="sax_cv")
        sax_acf   = st.slider("acf_threshold",    0.0, 1.0, float(_get_init("sax","acf_threshold",0.30)), 0.05, key="sax_acf")
        sax_motif = st.slider("motif_threshold",  0.0, 1.0, float(_get_init("sax","motif_threshold",0.40)),0.05, key="sax_motif")
        sax_min_t = st.number_input("min_tests_passing", 1, 5, _get_init("sax","min_tests_passing",2), 1, key="sax_min_t")

# ── Tab 4: Periodicity ───────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### Periodicity Config")
    c1, c2 = st.columns(2)
    with c1:
        per_min_obs  = st.number_input("min_observations",           5,  500, _get_init("periodicity","min_observations",10), 1, key="per_min_obs")
        per_nlags    = st.number_input("acf_nlags",                  5,  200, _get_init("periodicity","acf_nlags",40),         1, key="per_nlags")
        per_min_per  = st.number_input("min_period_s (seconds)",     1, 3600, int(_get_init("periodicity","min_period_s",60)), 1, key="per_min_per")
    with c2:
        per_acf_sig  = st.slider("acf_significance_threshold", 0.0, 1.0, float(_get_init("periodicity","acf_significance_threshold",0.25)),0.05, key="per_acf_sig")
        per_cv       = st.slider("cv_threshold",               0.0, 2.0, float(_get_init("periodicity","cv_threshold",0.60)),              0.05, key="per_cv")
        per_fft      = st.slider("fft_power_ratio_threshold",  0.0, 1.0, float(_get_init("periodicity","fft_power_ratio_threshold",0.15)), 0.05, key="per_fft")
        per_conf     = st.slider("confidence_threshold",       0.0, 1.0, float(_get_init("periodicity","confidence_threshold",0.45)),      0.05, key="per_conf")

# ── Tab 5: PELT ──────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### PELT Changepoint Config")
    c1, c2 = st.columns(2)
    with c1:
        pelt_penalty     = st.text_input("penalty",            str(_get_init("pelt","penalty","bic")),    key="pelt_penalty")
        pelt_min_seg     = st.number_input("min_segment_length", 2, 100, _get_init("pelt","min_segment_length",5),  1, key="pelt_min_seg")
    with c2:
        pelt_min_obs     = st.number_input("min_observations", 5, 500,  _get_init("pelt","min_observations",15), 1, key="pelt_min_obs")
        pelt_max_cp      = st.number_input("max_changepoints", 1, 100,  _get_init("pelt","max_changepoints",10), 1, key="pelt_max_cp")

# ── Tab 6: Corroboration ─────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### Corroboration Config")
    c1, c2 = st.columns(2)
    with c1:
        corr_period_tol   = st.slider("period_tolerance_pct",  0.0, 0.5,  float(_get_init("corroboration","period_tolerance_pct",0.15)),  0.01, key="corr_period_tol")
        corr_short_ttl    = st.number_input("short_ttl_threshold_s", 10, 3600, int(_get_init("corroboration","short_ttl_threshold_s",300)), 10,   key="corr_short_ttl")
        corr_dga_ent      = st.slider("dga_entropy_threshold", 1.0, 6.0,  float(_get_init("corroboration","dga_entropy_threshold",3.5)),   0.1,  key="corr_dga_ent")
        corr_dga_min_len  = st.number_input("dga_min_label_len", 2, 30,   _get_init("corroboration","dga_min_label_len",8),               1,    key="corr_dga_min_len")
        corr_min_score    = st.slider("min_score",             0.0, 1.0,  float(_get_init("corroboration","min_score",0.55)),              0.01, key="corr_min_score")
    with c2:
        corr_body_cv      = st.slider("http_body_cv_threshold",     0.0, 2.0,  float(_get_init("corroboration","http_body_cv_threshold",0.30)),  0.05, key="corr_body_cv")
        corr_uri_cv       = st.slider("http_uri_cv_threshold",      0.0, 2.0,  float(_get_init("corroboration","http_uri_cv_threshold",0.40)),   0.05, key="corr_uri_cv")
        corr_rare_ua      = st.slider("rare_ua_threshold",          0.0, 0.5,  float(_get_init("corroboration","rare_ua_threshold",0.05)),       0.01, key="corr_rare_ua")
        corr_uri_ent      = st.slider("uri_entropy_threshold",      0.0, 8.0,  float(_get_init("corroboration","uri_entropy_threshold",4.0)),    0.1,  key="corr_uri_ent")
        corr_nxdomain     = st.slider("nxdomain_rate_threshold",    0.0, 1.0,  float(_get_init("corroboration","nxdomain_rate_threshold",0.10)), 0.01, key="corr_nxdomain")
        corr_body_trim    = st.slider("body_cv_trim_pct",           0.0, 0.2,  float(_get_init("corroboration","body_cv_trim_pct",0.05)),        0.01, key="corr_body_trim")
    corr_extra_benign = st.text_input(
        "Extra benign domain suffixes (comma-separated)",
        value=", ".join(_get_init("corroboration","extra_benign_domain_suffixes",[])),
        key="corr_extra_benign",
    )

# ── Tab 7: TLS ───────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### TLS Corroboration Config (H5 / H6)")
    c1, c2 = st.columns(2)
    with c1:
        tls_sni_ent    = st.slider("sni_entropy_threshold",   0.0, 5.0, float(_get_init("corroboration",{}).get("tls",{}).get("sni_entropy_threshold", 1.0) if isinstance(_get_init("corroboration",{}),dict) else 1.0), 0.1, key="tls_sni_ent")
        tls_ja3_mono   = st.slider("ja3_monotony_threshold",  0.0, 1.0, float(_get_init("corroboration",{}).get("tls",{}).get("ja3_monotony_threshold",0.90) if isinstance(_get_init("corroboration",{}),dict) else 0.90), 0.05, key="tls_ja3_mono")
        tls_cert_reuse = st.number_input("cert_reuse_min_sessions", 1, 50, 3, 1, key="tls_cert_reuse")
    with c2:
        tls_cert_age   = st.number_input("cert_age_new_days", 1, 365, 30, 1, key="tls_cert_age")
        tls_h5_weight  = st.slider("h5_weight", 0.0, 1.0, 0.30, 0.05, key="tls_h5_weight")
        tls_h6_weight  = st.slider("h6_weight", 0.0, 1.0, 0.30, 0.05, key="tls_h6_weight")
    tls_known_c2 = st.text_area(
        "Known C2 JA3 fingerprints (one per line)",
        value="\n".join(_get_init("corroboration",{}).get("tls",{}).get("ja3_known_c2",
            ["e7d705a3286e19ea42f587b344ee6865","6d4e5b73a8e1c8a0f9c6e62f7b2d1a9c"])
            if isinstance(_get_init("corroboration",{}),dict) else
            ["e7d705a3286e19ea42f587b344ee6865","6d4e5b73a8e1c8a0f9c6e62f7b2d1a9c"]),
        height=80, key="tls_known_c2",
    )

# ── Tab 8: Prefilter ─────────────────────────────────────────────────────────
with tabs[8]:
    st.markdown("### Pre-filter Config")
    c1, c2 = st.columns(2)
    with c1:
        pf_fanin   = st.slider("dst_fanin_threshold",   0.0, 1.0, float(_get_init("prefilter","dst_fanin_threshold",0.50)),   0.05, key="pf_fanin")
    with c2:
        pf_failed  = st.slider("failed_conn_threshold", 0.0, 1.0, float(_get_init("prefilter","failed_conn_threshold",0.90)), 0.05, key="pf_failed")

# ── Tab 9: Triage ────────────────────────────────────────────────────────────
with tabs[9]:
    st.markdown("### Triage / Priority Scoring Config")
    c1, c2 = st.columns(2)
    with c1:
        tri_beacon_std  = st.slider("beaconing_std_thresh",  0.0, 5.0,  float(_get_init("triage","beaconing_std_thresh",0.5)),  0.1,  key="tri_beacon_std")
        tri_rare_dst    = st.number_input("rare_dst_thresh", 1, 1000,   _get_init("triage","rare_dst_thresh",25),               1,    key="tri_rare_dst")
    with c2:
        tri_high_vol    = st.slider("high_volume_pct",       0.0, 0.5,  float(_get_init("triage","high_volume_pct",0.05)),       0.01, key="tri_high_vol")
        tri_off_start   = st.number_input("off_hour_start",  0,   23,   _get_init("triage","off_hour_range",[6,22])[0],          1,    key="tri_off_start")
        tri_off_end     = st.number_input("off_hour_end",    0,   23,   _get_init("triage","off_hour_range",[6,22])[1],          1,    key="tri_off_end")

# ── Tab 10: Scaling ──────────────────────────────────────────────────────────
with tabs[10]:
    st.markdown("### Scaling / Variance Filter Config")
    c1, c2 = st.columns(2)
    with c1:
        scl_thresh      = st.slider("threshold",             0.0, 10.0, float(_get_init("scaling","threshold",1.0)),             0.1,  key="scl_thresh")
        scl_bin_thresh  = st.slider("binary_threshold",      0.0, 0.1,  float(_get_init("scaling","binary_threshold",0.001)),    0.001,key="scl_bin_thresh")
        scl_skew        = st.slider("skew_threshold",        0.0, 10.0, float(_get_init("scaling","skew_threshold",2.0)),        0.1,  key="scl_skew")
    with c2:
        scl_rr          = st.slider("range_ratio_threshold", 1.0, 1000.0,float(_get_init("scaling","range_ratio_threshold",100.0)),1.0, key="scl_rr")
        scl_min_uniq    = st.number_input("min_unique",      2,   100,  _get_init("scaling","min_unique",10),                    1,    key="scl_min_uniq")
        scl_min_max     = st.slider("min_max_threshold",     1.0, 1000.0,float(_get_init("scaling","min_max_threshold",100.0)),  1.0,  key="scl_min_max")


# ---------------------------------------------------------------------------
# Build BDPConfig from UI values
# ---------------------------------------------------------------------------

def build_config_from_ui() -> dict:
    """Assemble config dict from current widget values."""
    extra_benign = [s.strip() for s in st.session_state.corr_extra_benign.split(",") if s.strip()]
    ja3_list     = [s.strip() for s in st.session_state.tls_known_c2.splitlines() if s.strip()]
    keep_cols    = tuple(s.strip() for s in st.session_state.feat_keep_cols.split(",") if s.strip())
    channel_key  = tuple(st.session_state.pair_channel_key)

    try:
        pelt_penalty = float(st.session_state.pelt_penalty)
    except ValueError:
        pelt_penalty = st.session_state.pelt_penalty

    return {
        "io": {
            "output_dir":  st.session_state.io_output_dir,
            "table_name":  st.session_state.io_table_name,
            "query_start": st.session_state.io_qstart,
            "query_end":   st.session_state.io_qend,
            "query_limit": int(st.session_state.io_query_limit),
            "debug":       st.session_state.io_debug,
        },
        "features": {"keep_cols": list(keep_cols)},
        "scaling": {
            "threshold":             st.session_state.scl_thresh,
            "binary_threshold":      st.session_state.scl_bin_thresh,
            "skew_threshold":        st.session_state.scl_skew,
            "range_ratio_threshold": st.session_state.scl_rr,
            "min_unique":            int(st.session_state.scl_min_uniq),
            "min_max_threshold":     st.session_state.scl_min_max,
        },
        "isolation": {
            "n_estimators":        int(st.session_state.iso_n_est),
            "max_samples":         int(st.session_state.iso_max_samp),
            "test_size":           st.session_state.iso_test_sz,
            "random_state":        int(st.session_state.iso_rs),
            "contamination":       st.session_state.iso_contam,
            "stability_threshold": st.session_state.iso_stab,
        },
        "pair": {
            "min_observations": int(st.session_state.pair_min_obs),
            "min_pair_flows":   int(st.session_state.pair_min_flows),
            "max_pairs":        int(st.session_state.pair_max_pairs),
            "channel_key":      list(channel_key),
        },
        "sax": {
            "word_length":       int(st.session_state.sax_word_len),
            "alphabet_size":     int(st.session_state.sax_alpha),
            "cv_threshold":      st.session_state.sax_cv,
            "acf_threshold":     st.session_state.sax_acf,
            "motif_threshold":   st.session_state.sax_motif,
            "min_tests_passing": int(st.session_state.sax_min_t),
            "min_observations":  int(st.session_state.sax_min_obs),
            "max_acf_lag":       int(st.session_state.sax_max_lag),
        },
        "periodicity": {
            "min_observations":           int(st.session_state.per_min_obs),
            "acf_nlags":                  int(st.session_state.per_nlags),
            "acf_significance_threshold": st.session_state.per_acf_sig,
            "cv_threshold":               st.session_state.per_cv,
            "fft_power_ratio_threshold":  st.session_state.per_fft,
            "min_period_s":               float(st.session_state.per_min_per),
            "confidence_threshold":       st.session_state.per_conf,
        },
        "pelt": {
            "penalty":            pelt_penalty,
            "min_segment_length": int(st.session_state.pelt_min_seg),
            "min_observations":   int(st.session_state.pelt_min_obs),
            "max_changepoints":   int(st.session_state.pelt_max_cp),
        },
        "corroboration": {
            "period_tolerance_pct":        st.session_state.corr_period_tol,
            "short_ttl_threshold_s":       float(st.session_state.corr_short_ttl),
            "dga_entropy_threshold":       st.session_state.corr_dga_ent,
            "dga_min_label_len":           int(st.session_state.corr_dga_min_len),
            "http_body_cv_threshold":      st.session_state.corr_body_cv,
            "http_uri_cv_threshold":       st.session_state.corr_uri_cv,
            "rare_ua_threshold":           st.session_state.corr_rare_ua,
            "uri_entropy_threshold":       st.session_state.corr_uri_ent,
            "min_score":                   st.session_state.corr_min_score,
            "nxdomain_rate_threshold":     st.session_state.corr_nxdomain,
            "body_cv_trim_pct":            st.session_state.corr_body_trim,
            "extra_benign_domain_suffixes": extra_benign,
            "tls": {
                "sni_entropy_threshold":   st.session_state.tls_sni_ent,
                "ja3_monotony_threshold":  st.session_state.tls_ja3_mono,
                "cert_reuse_min_sessions": int(st.session_state.tls_cert_reuse),
                "cert_age_new_days":       int(st.session_state.tls_cert_age),
                "ja3_known_c2":            ja3_list,
                "h5_weight":               st.session_state.tls_h5_weight,
                "h6_weight":               st.session_state.tls_h6_weight,
            },
        },
        "prefilter": {
            "dst_fanin_threshold":   st.session_state.pf_fanin,
            "failed_conn_threshold": st.session_state.pf_failed,
        },
        "triage": {
            "beaconing_std_thresh": st.session_state.tri_beacon_std,
            "rare_dst_thresh":      int(st.session_state.tri_rare_dst),
            "high_volume_pct":      st.session_state.tri_high_vol,
            "off_hour_range":       [int(st.session_state.tri_off_start), int(st.session_state.tri_off_end)],
        },
    }


# ---------------------------------------------------------------------------
# Config download button
# ---------------------------------------------------------------------------

cfg_dict = build_config_from_ui()
cfg_json_str = json.dumps(cfg_dict, indent=2, default=str)

st.sidebar.download_button(
    "💾  Download current config",
    data=cfg_json_str,
    file_name="cadence_config.json",
    mime="application/json",
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Log capture handler
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    """Push log records into st.session_state.run_logs."""
    def emit(self, record):
        msg = self.format(record)
        st.session_state.run_logs += msg + "\n"


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline(cfg_dict: dict, tmpdir: str):
    """Run in the current thread (Streamlit callback)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from analytic_pipeline import BDPConfig, BDPPipeline
    from analytic_pipeline.config import (
        IOConfig, FeatureConfig, ScalingConfig, IsolationConfig,
        PairConfig, SAXConfig, PeriodicityConfig, PELTConfig,
        CorroborationConfig, TLSCorroborationConfig, PrefilterConfig, TriageConfig,
    )
    from analytic_pipeline.report import ReportContext

    # Attach log handler
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", "%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        # Build config
        cd = cfg_dict
        tls_d  = cd["corroboration"].pop("tls", {})
        cfg = BDPConfig(
            io=IOConfig(**{k: v for k, v in cd["io"].items() if k not in ("input_csv","input_parquet","input_unified")}),
            features=FeatureConfig(keep_cols=tuple(cd["features"]["keep_cols"])),
            scaling=ScalingConfig(**cd["scaling"]),
            isolation=IsolationConfig(**cd["isolation"]),
            pair=PairConfig(**{**cd["pair"], "channel_key": tuple(cd["pair"]["channel_key"])}),
            sax=SAXConfig(**cd["sax"]),
            periodicity=PeriodicityConfig(**cd["periodicity"]),
            pelt=PELTConfig(**cd["pelt"]),
            corroboration=CorroborationConfig(
                **cd["corroboration"],
                tls=TLSCorroborationConfig(**tls_d),
            ),
            prefilter=PrefilterConfig(**cd["prefilter"]),
            triage=TriageConfig(**{k: v for k, v in cd["triage"].items() if k != "off_hour_range"},
                                off_hour_range=tuple(cd["triage"]["off_hour_range"])),
        )
        cfg.io.output_dir = Path(tmpdir) / "output"
        cfg.io.output_dir.mkdir(parents=True, exist_ok=True)

        # Set input paths from session state temp files
        if st.session_state.get("_unified_path"):
            cfg.io.input_unified = Path(st.session_state["_unified_path"])
        elif st.session_state.get("_conn_path"):
            p = Path(st.session_state["_conn_path"])
            if p.suffix.lower() in (".parquet", ".pq"):
                cfg.io.input_parquet = p
            else:
                cfg.io.input_csv = p

        dns_path  = st.session_state.get("_dns_path")
        http_path = st.session_state.get("_http_path")
        ssl_path  = st.session_state.get("_ssl_path")

        st.session_state.run_logs += "▶  Starting CADENCE pipeline...\n"

        report_dir = cfg.io.output_dir / "report"
        report_dir.mkdir(exist_ok=True)

        if generate_report:
            with ReportContext(output_dir=report_dir, open_browser=False) as report:
                pipe = BDPPipeline(cfg)
                art  = pipe.run(
                    dns_log_path  = dns_path,
                    http_log_path = http_path,
                    ssl_log_path  = ssl_path,
                    visualize     = not suppress_plots,
                )
                rpt_path = report.finalise(art)
            if rpt_path and Path(rpt_path).exists():
                st.session_state.report_html = Path(rpt_path).read_text(encoding="utf-8")
        else:
            pipe = BDPPipeline(cfg)
            art  = pipe.run(
                dns_log_path  = dns_path,
                http_log_path = http_path,
                ssl_log_path  = ssl_path,
                visualize     = not suppress_plots,
            )

        st.session_state.artifacts = art
        st.session_state.run_complete = True
        st.session_state.run_logs += "\n✅  Pipeline complete.\n"

    except Exception as exc:
        import traceback
        st.session_state.run_logs += f"\n❌  ERROR: {exc}\n{traceback.format_exc()}\n"
        st.session_state.run_complete = True
    finally:
        root_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Synthetic pipeline execution
# ---------------------------------------------------------------------------

def run_pipeline_synthetic(cfg_dict: dict, days: int, bg_rows: int, noisy_rows: int, seed: int):
    """
    Generate synthetic Zeek logs in-memory and run the full pipeline —
    no file uploads or temp files required.

    Uses cfg._unified_slices to pass DataFrames directly to corroboration,
    bypassing all file I/O. Ground-truth labels are stored in
    st.session_state.syn_labels for the evaluation table.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from analytic_pipeline import BDPConfig, BDPPipeline
    from analytic_pipeline.config import (
        IOConfig, FeatureConfig, ScalingConfig, IsolationConfig,
        PairConfig, SAXConfig, PeriodicityConfig, PELTConfig,
        CorroborationConfig, TLSCorroborationConfig, PrefilterConfig, TriageConfig,
    )
    from analytic_pipeline.generate_synthetic_data import SyntheticDataGenerator
    from analytic_pipeline.report import ReportContext

    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", "%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        # ── Step 1: generate synthetic data ───────────────────────────────
        st.session_state.run_logs += f"🧪  Generating synthetic data ({days} days, {bg_rows:,} bg rows/day, seed={seed})...\n"
        gen = SyntheticDataGenerator(seed=seed)
        conn, dns, http, ssl, labels = gen.generate(
            days=days,
            bg_rows_per_day=bg_rows,
            noisy_rows_per_day=noisy_rows,
        )
        st.session_state.syn_labels = labels
        st.session_state.run_logs += (
            f"   conn: {len(conn):,} rows  |  dns: {len(dns):,}  |  "
            f"http: {len(http):,}  |  ssl: {len(ssl):,}\n"
            f"   ground-truth scenarios: {labels['scenario'].tolist()}\n"
        )

        # ── Step 2: build config ──────────────────────────────────────────
        cd = cfg_dict
        tls_d = cd["corroboration"].pop("tls", {})
        cfg = BDPConfig(
            io=IOConfig(**{k: v for k, v in cd["io"].items()
                           if k not in ("input_csv", "input_parquet", "input_unified")}),
            features=FeatureConfig(keep_cols=tuple(cd["features"]["keep_cols"])),
            scaling=ScalingConfig(**cd["scaling"]),
            isolation=IsolationConfig(**cd["isolation"]),
            pair=PairConfig(**{**cd["pair"], "channel_key": tuple(cd["pair"]["channel_key"])}),
            sax=SAXConfig(**cd["sax"]),
            periodicity=PeriodicityConfig(**cd["periodicity"]),
            pelt=PELTConfig(**cd["pelt"]),
            corroboration=CorroborationConfig(
                **cd["corroboration"],
                tls=TLSCorroborationConfig(**tls_d),
            ),
            prefilter=PrefilterConfig(**cd["prefilter"]),
            triage=TriageConfig(
                **{k: v for k, v in cd["triage"].items() if k != "off_hour_range"},
                off_hour_range=tuple(cd["triage"]["off_hour_range"]),
            ),
        )

        # Set query window from the generated data
        cfg.io.query_start = str(pd.to_datetime(conn["timestamp"].min(), unit="s", utc=True))[:19]
        cfg.io.query_end   = str(pd.to_datetime(conn["timestamp"].max(), unit="s", utc=True))[:19]

        tmpdir = tempfile.mkdtemp(prefix="cadence_syn_")
        cfg.io.output_dir = Path(tmpdir) / "output"
        cfg.io.output_dir.mkdir(parents=True, exist_ok=True)

        # Inject DataFrames directly via _unified_slices — no file I/O
        cfg._unified_slices = {"conn": conn, "dns": dns, "http": http, "ssl": ssl}

        # Write conn to a temp CSV so load_and_prepare() can read it
        # (unified mode reads conn from _unified_slices, so this sets input_unified
        #  to a sentinel path that triggers the unified branch in loaders.py)
        conn_path = Path(tmpdir) / "conn.csv"
        conn.to_csv(conn_path, index=False)
        cfg.io.input_csv = conn_path   # used as fallback; unified slices take priority

        st.session_state.run_logs += "▶  Starting CADENCE pipeline...\n"

        # ── Step 3: run pipeline ──────────────────────────────────────────
        generate_report = st.session_state.get("_generate_report", True)
        suppress_plots  = st.session_state.get("_suppress_plots",  True)

        report_dir = cfg.io.output_dir / "report"
        report_dir.mkdir(exist_ok=True)

        if generate_report:
            with ReportContext(output_dir=report_dir, open_browser=False) as report:
                pipe = BDPPipeline(cfg)
                art  = pipe.run(
                    dns_log_path  = None,   # passed via cfg._unified_slices
                    http_log_path = None,
                    ssl_log_path  = None,
                    visualize     = not suppress_plots,
                )
                rpt_path = report.finalise(art)
            if rpt_path and Path(rpt_path).exists():
                st.session_state.report_html = Path(rpt_path).read_text(encoding="utf-8")
        else:
            pipe = BDPPipeline(cfg)
            art  = pipe.run(
                dns_log_path  = None,
                http_log_path = None,
                ssl_log_path  = None,
                visualize     = not suppress_plots,
            )

        # ── Step 4: ground-truth evaluation ──────────────────────────────
        if not art.corroboration.empty:
            from analytic_pipeline.generate_synthetic_data import evaluate_detection
            gt_results = evaluate_detection(art.corroboration, labels, art.anomalies)
            st.session_state.syn_eval = gt_results
            p   = float(gt_results["precision"].iloc[0])
            r   = float(gt_results["recall"].iloc[0])
            f1  = float(gt_results["f1"].iloc[0])
            st.session_state.run_logs += (
                f"\n📊  Ground-truth evaluation:\n"
                f"   Precision: {p:.3f}  |  Recall: {r:.3f}  |  F1: {f1:.3f}\n"
            )
        else:
            st.session_state.syn_eval = None

        st.session_state.artifacts    = art
        st.session_state.run_complete = True
        st.session_state.run_logs    += "\n✅  Pipeline complete.\n"

    except Exception as exc:
        import traceback
        st.session_state.run_logs += f"\n❌  ERROR: {exc}\n{traceback.format_exc()}\n"
        st.session_state.run_complete = True
        st.session_state.syn_eval = None
    finally:
        root_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Handle Run button
# ---------------------------------------------------------------------------

if run_btn:
    st.session_state.run_logs     = ""
    st.session_state.artifacts    = None
    st.session_state.report_html  = None
    st.session_state.run_complete = False
    st.session_state.syn_labels   = None
    st.session_state.syn_eval     = None
    # Store checkbox states so run_pipeline_synthetic can read them
    st.session_state["_generate_report"] = generate_report
    st.session_state["_suppress_plots"]  = suppress_plots

    if input_mode == "Synthetic data":
        cfg_dict_run = build_config_from_ui()
        run_pipeline_synthetic(
            cfg_dict_run,
            days      = syn_days,
            bg_rows   = syn_bg_rows,
            noisy_rows= syn_noisy,
            seed      = syn_seed,
        )
        st.rerun()

    else:
        has_input = unified_file is not None or conn_file is not None
        if not has_input:
            st.error("Please upload at least a conn log (or a unified file) before running.")
        else:
            tmpdir = tempfile.mkdtemp(prefix="cadence_")

            def _save(f, suffix):
                ext = Path(f.name).suffix or suffix
                p = Path(tmpdir) / f"upload{ext}"
                p.write_bytes(f.read())
                return str(p)

            if unified_file:
                st.session_state["_unified_path"] = _save(unified_file, ".csv")
                st.session_state["_conn_path"]    = None
                st.session_state["_dns_path"]     = None
                st.session_state["_http_path"]    = None
                st.session_state["_ssl_path"]     = None
            else:
                st.session_state["_unified_path"] = None
                st.session_state["_conn_path"]  = _save(conn_file,  ".csv") if conn_file  else None
                st.session_state["_dns_path"]   = _save(dns_file,   ".csv") if dns_file   else None
                st.session_state["_http_path"]  = _save(http_file,  ".csv") if http_file  else None
                st.session_state["_ssl_path"]   = _save(ssl_file,   ".csv") if ssl_file   else None

            cfg_dict_run = build_config_from_ui()
            run_pipeline(cfg_dict_run, tmpdir)
            st.rerun()


# ---------------------------------------------------------------------------
# Results section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 📋 Run Output")

if st.session_state.run_logs:
    st.markdown(
        f'<div class="log-console">{st.session_state.run_logs}</div>',
        unsafe_allow_html=True,
    )

art = st.session_state.artifacts
if art is not None:
    st.markdown("### Pipeline Funnel")

    n_raw     = len(art.raw)         if art.raw is not None         else 0
    n_pre     = len(art.prefiltered) if art.prefiltered is not None and not art.prefiltered.empty else 0
    n_anom    = len(art.anomalies)   if art.anomalies is not None   else 0
    n_sax     = len(art.sax)         if art.sax is not None and not art.sax.empty else 0
    n_sax_p   = int(art.sax["sax_prescreen_pass"].sum()) if art.sax is not None and not art.sax.empty else 0
    n_beacon  = int(art.periodicity["is_beacon_pair"].sum()) if art.periodicity is not None and not art.periodicity.empty else 0
    n_corr    = int(art.corroboration["corroborated"].sum()) if art.corroboration is not None and not art.corroboration.empty else 0

    cols = st.columns(7)
    labels = ["Raw rows", "Pre-filtered", "IForest anomalies", "Pairs eval'd", "SAX pass", "Beacon pairs", "Corroborated"]
    values = [n_raw, n_pre, n_anom, n_sax, n_sax_p, n_beacon, n_corr]
    for col, lbl, val in zip(cols, labels, values):
        col.metric(lbl, f"{val:,}")

    # Download buttons
    st.markdown("### Downloads")
    dl_cols = st.columns(5)

    if art.priority is not None and not art.priority.empty:
        dl_cols[0].download_button(
            "📥 Priority CSV",
            data=art.priority.to_csv(index=False),
            file_name="priority.csv", mime="text/csv",
        )
    if art.periodicity is not None and not art.periodicity.empty:
        dl_cols[1].download_button(
            "📥 Periodicity CSV",
            data=art.periodicity.to_csv(index=False),
            file_name="periodicity.csv", mime="text/csv",
        )
    if art.corroboration is not None and not art.corroboration.empty:
        dl_cols[2].download_button(
            "📥 Corroboration CSV",
            data=art.corroboration.to_csv(index=False),
            file_name="corroboration.csv", mime="text/csv",
        )
    if hasattr(art, "shap_values") and art.shap_values is not None and not art.shap_values.empty:
        dl_cols[3].download_button(
            "📥 SHAP Values CSV",
            data=art.shap_values.to_csv(index=False),
            file_name="shap_values.csv", mime="text/csv",
        )
    if st.session_state.report_html:
        dl_cols[4].download_button(
            "📥 HTML Report",
            data=st.session_state.report_html,
            file_name="cadence_report.html", mime="text/html",
        )

    # ── SHAP section ────────────────────────────────────────────────────────
    if hasattr(art, "shap_values") and art.shap_values is not None and not art.shap_values.empty:
        st.markdown("### SHAP Feature Importance")
        st.caption(
            "SHAP values explain **why each pair was flagged by Isolation Forest** — "
            "not why it is believed to be C2. For that, see the corroboration hypotheses (H1–H6). "
            "These two signals are complementary: SHAP explains anomaly detection; "
            "corroboration explains beacon attribution."
        )

        shap_df = art.shap_values
        shap_cols = [c for c in shap_df.columns if c.startswith("shap_") and c != "shap_sum"]

        # ── Beeswarm (mean |SHAP| bar chart as Streamlit-native substitute) ──
        if shap_cols:
            mean_abs = shap_df[shap_cols].abs().mean().sort_values(ascending=False)
            mean_abs.index = [c.replace("shap_", "") for c in mean_abs.index]
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#0b0f17")
            ax.set_facecolor("#161b22")
            bars = ax.barh(
                mean_abs.index[::-1],
                mean_abs.values[::-1],
                color="#1f6feb", edgecolor="#21262d", linewidth=0.5,
            )
            ax.set_xlabel("Mean |SHAP value|", color="#8b949e", fontsize=10)
            ax.set_title("IForest Feature Importance — Mean |SHAP|", color="#58a6ff", fontsize=11, pad=10)
            ax.tick_params(colors="#8b949e", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#21262d")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Per-pair waterfall — select a corroborated lead ──────────────────
        if "channel_id" in shap_df.columns and art.corroboration is not None and not art.corroboration.empty:
            st.markdown("#### Waterfall — single pair explanation")
            st.caption("Select a corroborated lead to see exactly which features drove its anomaly score.")

            # Build display labels from corroboration table
            corr = art.corroboration.copy()
            if "channel_id" in corr.columns:
                id_col = "channel_id"
            else:
                # synthesise a channel_id label from IP/port if needed
                corr["_label"] = (
                    corr.get("src_ip", "?").astype(str) + " → " +
                    corr.get("dst_ip", "?").astype(str) + ":" +
                    corr.get("dst_port", "").astype(str)
                )
                id_col = "_label"

            available_ids = [
                cid for cid in corr[id_col].tolist()
                if cid in shap_df.get("channel_id", pd.Series()).values
            ]

            if available_ids:
                selected = st.selectbox("Channel", available_ids, key="shap_waterfall_sel")

                if selected:
                    from analytic_pipeline.isolation import plot_shap_waterfall
                    import io

                    try:
                        import shap
                        row = shap_df[shap_df["channel_id"] == selected]
                        sv  = row[shap_cols].values[0]
                        feat_names = [c.replace("shap_", "") for c in shap_cols]

                        explainer = shap.TreeExplainer(art.iforest_model)
                        base_val  = explainer.expected_value

                        # Feature values: look up in art.scaled
                        if "channel_id" in art.scaled.columns:
                            pair_row = art.scaled[art.scaled["channel_id"] == selected]
                            stdz_cols = [f"{n}_stdz" for n in feat_names]
                            stdz_cols = [c for c in stdz_cols if c in art.scaled.columns]
                            feat_vals = pair_row[stdz_cols].fillna(0).values[0] if not pair_row.empty else sv
                        else:
                            feat_vals = sv   # fallback

                        explanation = shap.Explanation(
                            values=sv,
                            base_values=base_val,
                            data=feat_vals,
                            feature_names=feat_names,
                        )

                        fig2, ax2 = plt.subplots(figsize=(9, 4))
                        fig2.patch.set_facecolor("#0b0f17")
                        shap.plots.waterfall(explanation, show=False)
                        plt.title(f"SHAP Waterfall — {selected}", color="#58a6ff", fontsize=11)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)

                    except Exception as e:
                        st.warning(f"Could not render waterfall: {e}")
            else:
                st.info("No matching channel IDs between corroborated leads and SHAP output.")

    # Corroboration table
    if art.corroboration is not None and not art.corroboration.empty:
        st.markdown("### Corroborated Leads")
        display_cols = [c for c in [
            "src_ip","dst_ip","dst_port","proto",
            "corroborated","triage_score","beacon_confidence",
            "dominant_period_s","mitre_techniques",
        ] if c in art.corroboration.columns]
        st.dataframe(
            art.corroboration[display_cols]
              .sort_values("triage_score", ascending=False)
              .reset_index(drop=True),
            use_container_width=True,
        )
    elif st.session_state.run_complete:
        st.info("No corroborated leads. Check that DNS/HTTP/SSL logs are provided and min_score isn't too high.")

    # ── Ground-truth evaluation (synthetic mode only) ────────────────────
    syn_eval = st.session_state.get("syn_eval")
    if syn_eval is not None and not syn_eval.empty:
        st.markdown("### 🎯 Ground-Truth Evaluation")
        st.caption(
            "Synthetic mode only — compares corroborated leads against known malicious "
            "scenarios. Decoy scenarios should NOT appear in corroborated leads."
        )

        # Summary metrics
        p  = float(syn_eval["precision"].iloc[0])
        r  = float(syn_eval["recall"].iloc[0])
        f1 = float(syn_eval["f1"].iloc[0])
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{p:.3f}", delta="≥ target" if p >= 0.75 else "below target",
                  delta_color="normal" if p >= 0.75 else "inverse")
        m2.metric("Recall",    f"{r:.3f}", delta="≥ 0.75 ✓" if r >= 0.75 else "< 0.75 ✗",
                  delta_color="normal" if r >= 0.75 else "inverse")
        m3.metric("F1",        f"{f1:.3f}")

        # Per-scenario table
        display_cols = [c for c in ["scenario", "malicious", "detected"] if c in syn_eval.columns]
        if display_cols:
            styled = syn_eval[display_cols].copy()
            styled["result"] = styled.apply(
                lambda row: "✅ Detected" if row.get("detected") else (
                    "❌ Missed" if row.get("malicious") else "✓ Correctly suppressed"
                ), axis=1
            )
            st.dataframe(
                styled[["scenario", "malicious", "result"]].reset_index(drop=True),
                use_container_width=True,
            )
