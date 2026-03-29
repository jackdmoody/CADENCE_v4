"""
cadence_app_dash.py
====================
Dash GUI for the CADENCE C2 Beacon Detection Pipeline.

Run with:
    python cadence_app_dash.py
    python cadence_app_dash.py --port 8051 --debug

Features
--------
- Three input modes: synthetic data (in-memory), unified file, separate files
- All ~40 BDPConfig fields across 11 tabbed config sections
- Background-thread pipeline execution with live log polling
- Pipeline funnel metrics, SHAP bar chart, corroborated leads table
- Ground-truth evaluation table (synthetic mode only)
- Save / load config JSON
- Download all output CSVs and HTML report
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check — before any heavy import
# ---------------------------------------------------------------------------

_REQUIRED = {
    "pandas":      ("pandas",        "pandas>=2.0"),
    "numpy":       ("numpy",         "numpy>=1.24"),
    "sklearn":     ("scikit-learn",  "scikit-learn>=1.3"),
    "scipy":       ("scipy",         "scipy>=1.10"),
    "statsmodels": ("statsmodels",   "statsmodels>=0.14"),
    "matplotlib":  ("matplotlib",    "matplotlib>=3.7"),
    "pyarrow":     ("pyarrow",       "pyarrow>=14.0"),
    "shap":        ("shap",          "shap>=0.44"),
    "dash":        ("dash",          "dash>=2.14"),
}
_CADENCE_PKG = Path(__file__).parent / "analytic_pipeline" / "__init__.py"

_missing, _pins = [], []
for _imp, (_pkg, _pin) in _REQUIRED.items():
    try:
        __import__(_imp)
    except ImportError:
        _missing.append(_pkg)
        _pins.append(_pin)

if not _CADENCE_PKG.exists():
    print(
        "\n❌  analytic_pipeline package not found.\n"
        "   Run from the CADENCE repo root:\n"
        "       cd /path/to/CADENCE && python cadence_app_dash.py\n"
    )
    sys.exit(1)

if _missing:
    print(f"\n⚠  Missing dependencies: {', '.join(_missing)}")
    answer = input("   Install now? [y/N] ").strip().lower()
    if answer == "y":
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + _pins,
            capture_output=False,
        )
        if result.returncode != 0:
            print("❌  Installation failed. Run manually:")
            print("    pip install " + " ".join(_pins))
            sys.exit(1)
        print("✅  Installed. Restarting…\n")
        import os
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        print("   Run:  pip install " + " ".join(_pins))
        sys.exit(1)

# ---------------------------------------------------------------------------
# All deps present — safe to import
# ---------------------------------------------------------------------------

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

# ---------------------------------------------------------------------------
# Colour palette & shared styles
# ---------------------------------------------------------------------------

C = dict(
    bg="#0b0f17", sidebar="#0d1117", surface="#161b22",
    border="#21262d", border2="#30363d",
    blue="#58a6ff", blue2="#79c0ff", blue3="#1f6feb",
    green="#238636", green2="#3fb950",
    text="#c9d1d9", muted="#8b949e", dim="#484f58",
    warn="#d29922", err="#f85149", console="#010409",
)

MONO = "'IBM Plex Mono','Fira Code',monospace"
SANS = "'IBM Plex Sans',system-ui,sans-serif"

GLOBAL_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
body, * {{ box-sizing: border-box; }}
body {{ background: {C['bg']}; color: {C['text']}; font-family: {SANS}; margin: 0; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border2']}; border-radius: 3px; }}
input, select, textarea {{
    background: {C['sidebar']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    font-family: {MONO} !important;
    font-size: 12px !important;
    padding: 6px 10px !important;
    width: 100%;
    outline: none !important;
}}
input[type=range] {{
    padding: 0 !important;
    accent-color: {C['blue3']};
}}
input[type=checkbox] {{
    width: auto !important;
    accent-color: {C['blue3']};
}}
label {{ font-family: {MONO}; font-size: 11px; color: {C['muted']}; display: block; margin-bottom: 4px; }}
hr {{ border: none; border-top: 1px solid {C['border']}; margin: 12px 0; }}
"""

def _s(**kwargs):
    """Build an inline style dict."""
    return kwargs

def card(children, **style):
    base = dict(
        background=C["surface"], border=f"1px solid {C['border']}",
        borderRadius=8, padding="16px 18px", marginBottom=16,
    )
    base.update(style)
    return html.Div(children, style=base)

def label(text):
    return html.Label(text, style=dict(fontFamily=MONO, fontSize=11, color=C["muted"], marginBottom=4))

def section_head(text):
    return html.Div(text, style=dict(
        fontFamily=MONO, fontSize=10, color=C["muted"],
        textTransform="uppercase", letterSpacing="1.5px",
        borderBottom=f"1px solid {C['border']}", paddingBottom=6, marginBottom=14,
    ))

def two_col(left, right):
    return html.Div([
        html.Div(left,  style=dict(flex=1, paddingRight=10)),
        html.Div(right, style=dict(flex=1, paddingLeft=10)),
    ], style=dict(display="flex", gap=0))

def field(lbl, *children):
    return html.Div([label(lbl), *children], style=dict(marginBottom=14))

def txt(id_, value="", placeholder=""):
    return dcc.Input(id=id_, type="text", value=value, placeholder=placeholder,
                     debounce=True, style=dict(width="100%"))

def num(id_, value=0, min_=0, max_=None, step=1):
    kw = dict(id=id_, type="number", value=value, min=min_, step=step, debounce=True,
              style=dict(width="100%"))
    if max_ is not None:
        kw["max"] = max_
    return dcc.Input(**kw)

def sldr(id_, min_, max_, value, step=0.05):
    return html.Div([
        dcc.Slider(id=id_, min=min_, max=max_, value=value, step=step,
                   marks=None, tooltip={"placement": "bottom", "always_visible": True}),
    ], style=dict(paddingTop=4, paddingBottom=8))

def btn(id_, label_text, color=C["green"], text_color="#fff"):
    return html.Button(label_text, id=id_, n_clicks=0, style=dict(
        fontFamily=MONO, fontWeight=700, fontSize=13,
        background=color, color=text_color, border="none",
        borderRadius=6, padding="9px 18px", cursor="pointer",
        width="100%", marginBottom=8, transition="background 0.15s",
    ))

# ---------------------------------------------------------------------------
# Global pipeline state (shared between callback thread and main thread)
# ---------------------------------------------------------------------------

_state = dict(
    running=False,
    logs="",
    complete=False,
    artifacts=None,
    report_html=None,
    shap_df=None,
    syn_eval=None,
    error=None,
    start_time=None,    # float timestamp when run began
    current_stage=0,    # 0-10 index of last detected pipeline stage
)
_state_lock = threading.Lock()

# Pipeline stage labels and their log keywords — used for progress detection
_STAGES = [
    (1,  "Stage 1",    "Ingest & feature engineering"),
    (2,  "Stage 2",    "Channel aggregation & EDA scaling"),
    (3,  "Stage 3",    "Domain-knowledge pre-filter"),
    (4,  "Stage 4",    "Isolation Forest + SHAP"),
    (5,  "Stage 5",    "SAX pre-screening"),
    (6,  "Stage 6",    "ACF + Welch PSD periodicity"),
    (7,  "Stage 7",    "Priority triage scoring"),
    (8,  "Stage 8",    "PELT changepoint detection"),
    (9,  "Stage 9",    "Multi-source corroboration"),
    (10, "Stage 10",   "MITRE ATT&CK annotation"),
]

# Approximate fraction of total wall time each stage consumes (synthetic 30d run)
# Used to estimate time remaining. Tuned against real runs.
_STAGE_WEIGHT = [0.10, 0.12, 0.04, 0.18, 0.06, 0.22, 0.04, 0.06, 0.14, 0.04]


def _log(msg: str):
    with _state_lock:
        _state["logs"] += msg + "\n"
        # Detect stage transitions from log messages
        for stage_num, keyword, _ in _STAGES:
            if keyword in msg:
                if stage_num > _state["current_stage"]:
                    _state["current_stage"] = stage_num
                break


class _LogHandler(logging.Handler):
    def emit(self, record):
        _log(self.format(record))


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

CFG_DEFAULTS = dict(
    io_output_dir="results", io_table_name="Conn_logs",
    io_query_start="2025-10-22 00:00:00", io_query_end="2025-10-22 23:59:59",
    io_query_limit=1_000_000, io_debug=False,
    feat_keep_cols=(
        "timestamp,datetime,src_ip,src_p,src_pkts,"
        "dst_ip,dst_p,resp_pkts,duration,conn_state,service,"
        "total_bytes,sin_time,cos_time,hour,minute,scenario"
    ),
    iso_n_est=200, iso_max_samp=3000, iso_rs=42,
    iso_contam=0.05, iso_test_sz=0.3, iso_stab=0.80,
    pair_min_obs=8, pair_min_flows=8, pair_max_pairs=5000,
    pair_channel_key=["src_ip", "dst_ip", "dst_port", "proto"],
    sax_word_len=20, sax_alpha=4, sax_min_obs=8, sax_max_lag=10,
    sax_cv=0.60, sax_acf=0.30, sax_motif=0.40, sax_min_t=2,
    per_min_obs=10, per_nlags=40, per_min_per=60,
    per_acf_sig=0.25, per_cv=0.60, per_fft=0.15, per_conf=0.45,
    pelt_penalty="bic", pelt_min_seg=5, pelt_min_obs=15, pelt_max_cp=10,
    corr_period_tol=0.15, corr_short_ttl=300, corr_dga_ent=3.5,
    corr_dga_min_len=8, corr_min_score=0.55,
    corr_body_cv=0.30, corr_uri_cv=0.40, corr_rare_ua=0.05,
    corr_uri_ent=4.0, corr_nxdomain=0.10, corr_body_trim=0.05,
    corr_extra_benign="",
    tls_sni_ent=1.0, tls_ja3_mono=0.90, tls_cert_reuse=3,
    tls_cert_age=30, tls_h5_weight=0.30, tls_h6_weight=0.30,
    tls_known_c2="e7d705a3286e19ea42f587b344ee6865\n6d4e5b73a8e1c8a0f9c6e62f7b2d1a9c",
    pf_fanin=0.50, pf_failed=0.90,
    tri_beacon_std=0.5, tri_rare_dst=25, tri_high_vol=0.05,
    tri_off_start=6, tri_off_end=22,
    scl_thresh=1.0, scl_bin_thresh=0.001, scl_skew=2.0,
    scl_rr=100.0, scl_min_uniq=10, scl_min_max=100.0,
    # synthetic
    syn_days=30, syn_bg_rows=10000, syn_noisy=500, syn_seed=42,
)

# ---------------------------------------------------------------------------
# Build BDPConfig dict from callback values
# ---------------------------------------------------------------------------

def _build_cfg_dict(vals: dict) -> dict:
    g = vals.get
    extra_benign = [s.strip() for s in (g("corr_extra_benign") or "").split(",") if s.strip()]
    ja3_list     = [s.strip() for s in (g("tls_known_c2") or "").splitlines() if s.strip()]
    keep_cols    = [s.strip() for s in (g("feat_keep_cols") or "").split(",") if s.strip()]
    channel_key  = g("pair_channel_key") or ["src_ip","dst_ip","dst_port","proto"]

    penalty = g("pelt_penalty") or "bic"
    try:
        penalty = float(penalty)
    except (ValueError, TypeError):
        pass

    return {
        "io": {
            "output_dir":  g("io_output_dir") or "results",
            "table_name":  g("io_table_name") or "Conn_logs",
            "query_start": g("io_query_start") or "2025-10-22 00:00:00",
            "query_end":   g("io_query_end")   or "2025-10-22 23:59:59",
            "query_limit": int(g("io_query_limit") or 1_000_000),
            "debug":       bool(g("io_debug")),
        },
        "features": {"keep_cols": keep_cols},
        "scaling": {
            "threshold":             float(g("scl_thresh") or 1.0),
            "binary_threshold":      float(g("scl_bin_thresh") or 0.001),
            "skew_threshold":        float(g("scl_skew") or 2.0),
            "range_ratio_threshold": float(g("scl_rr") or 100.0),
            "min_unique":            int(g("scl_min_uniq") or 10),
            "min_max_threshold":     float(g("scl_min_max") or 100.0),
        },
        "isolation": {
            "n_estimators":        int(g("iso_n_est") or 200),
            "max_samples":         int(g("iso_max_samp") or 3000),
            "test_size":           float(g("iso_test_sz") or 0.3),
            "random_state":        int(g("iso_rs") or 42),
            "contamination":       float(g("iso_contam") or 0.05),
            "stability_threshold": float(g("iso_stab") or 0.80),
        },
        "pair": {
            "min_observations": int(g("pair_min_obs") or 8),
            "min_pair_flows":   int(g("pair_min_flows") or 8),
            "max_pairs":        int(g("pair_max_pairs") or 5000),
            "channel_key":      list(channel_key),
        },
        "sax": {
            "word_length":       int(g("sax_word_len") or 20),
            "alphabet_size":     int(g("sax_alpha") or 4),
            "cv_threshold":      float(g("sax_cv") or 0.60),
            "acf_threshold":     float(g("sax_acf") or 0.30),
            "motif_threshold":   float(g("sax_motif") or 0.40),
            "min_tests_passing": int(g("sax_min_t") or 2),
            "min_observations":  int(g("sax_min_obs") or 8),
            "max_acf_lag":       int(g("sax_max_lag") or 10),
        },
        "periodicity": {
            "min_observations":           int(g("per_min_obs") or 10),
            "acf_nlags":                  int(g("per_nlags") or 40),
            "acf_significance_threshold": float(g("per_acf_sig") or 0.25),
            "cv_threshold":               float(g("per_cv") or 0.60),
            "fft_power_ratio_threshold":  float(g("per_fft") or 0.15),
            "min_period_s":               float(g("per_min_per") or 60),
            "confidence_threshold":       float(g("per_conf") or 0.45),
        },
        "pelt": {
            "penalty":            penalty,
            "min_segment_length": int(g("pelt_min_seg") or 5),
            "min_observations":   int(g("pelt_min_obs") or 15),
            "max_changepoints":   int(g("pelt_max_cp") or 10),
        },
        "corroboration": {
            "period_tolerance_pct":         float(g("corr_period_tol") or 0.15),
            "short_ttl_threshold_s":        float(g("corr_short_ttl") or 300),
            "dga_entropy_threshold":        float(g("corr_dga_ent") or 3.5),
            "dga_min_label_len":            int(g("corr_dga_min_len") or 8),
            "http_body_cv_threshold":       float(g("corr_body_cv") or 0.30),
            "http_uri_cv_threshold":        float(g("corr_uri_cv") or 0.40),
            "rare_ua_threshold":            float(g("corr_rare_ua") or 0.05),
            "uri_entropy_threshold":        float(g("corr_uri_ent") or 4.0),
            "min_score":                    float(g("corr_min_score") or 0.55),
            "nxdomain_rate_threshold":      float(g("corr_nxdomain") or 0.10),
            "body_cv_trim_pct":             float(g("corr_body_trim") or 0.05),
            "extra_benign_domain_suffixes": extra_benign,
            "tls": {
                "sni_entropy_threshold":   float(g("tls_sni_ent") or 1.0),
                "ja3_monotony_threshold":  float(g("tls_ja3_mono") or 0.90),
                "cert_reuse_min_sessions": int(g("tls_cert_reuse") or 3),
                "cert_age_new_days":       int(g("tls_cert_age") or 30),
                "ja3_known_c2":            ja3_list,
                "h5_weight":               float(g("tls_h5_weight") or 0.30),
                "h6_weight":               float(g("tls_h6_weight") or 0.30),
            },
        },
        "prefilter": {
            "dst_fanin_threshold":   float(g("pf_fanin") or 0.50),
            "failed_conn_threshold": float(g("pf_failed") or 0.90),
        },
        "triage": {
            "beaconing_std_thresh": float(g("tri_beacon_std") or 0.5),
            "rare_dst_thresh":      int(g("tri_rare_dst") or 25),
            "high_volume_pct":      float(g("tri_high_vol") or 0.05),
            "off_hour_range":       [int(g("tri_off_start") or 6), int(g("tri_off_end") or 22)],
        },
    }


# ---------------------------------------------------------------------------
# Pipeline execution helpers
# ---------------------------------------------------------------------------

def _make_pipeline(cfg_dict: dict, tmpdir: str):
    """Build BDPConfig and BDPPipeline from a config dict."""
    sys.path.insert(0, str(Path(__file__).parent))
    from analytic_pipeline import BDPConfig, BDPPipeline
    from analytic_pipeline.config import (
        IOConfig, FeatureConfig, ScalingConfig, IsolationConfig,
        PairConfig, SAXConfig, PeriodicityConfig, PELTConfig,
        CorroborationConfig, TLSCorroborationConfig, PrefilterConfig, TriageConfig,
    )

    cd   = cfg_dict
    tls_d = cd["corroboration"].pop("tls", {})
    cfg = BDPConfig(
        io=IOConfig(**{k: v for k, v in cd["io"].items()
                       if k not in ("input_csv","input_parquet","input_unified")}),
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
    cfg.io.output_dir = Path(tmpdir) / "output"
    cfg.io.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg, BDPPipeline(cfg)


def _attach_log_handler() -> _LogHandler:
    handler = _LogHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", "%H:%M:%S"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler


def _detach_log_handler(handler: _LogHandler):
    logging.getLogger().removeHandler(handler)


def _run_file_pipeline(cfg_dict: dict, paths: dict, generate_report: bool):
    """Background thread: file-based pipeline execution."""
    handler = _attach_log_handler()
    try:
        tmpdir = tempfile.mkdtemp(prefix="cadence_")
        cfg, pipe = _make_pipeline(cfg_dict, tmpdir)

        conn_path = paths.get("conn")
        if conn_path:
            p = Path(conn_path)
            if p.suffix.lower() in (".parquet", ".pq"):
                cfg.io.input_parquet = p
            else:
                cfg.io.input_csv = p

        unified = paths.get("unified")
        if unified:
            cfg.io.input_unified = Path(unified)

        dns_path  = paths.get("dns")
        http_path = paths.get("http")
        ssl_path  = paths.get("ssl")

        _log("▶  Starting CADENCE pipeline…")

        if generate_report:
            from analytic_pipeline.report import ReportContext
            report_dir = cfg.io.output_dir / "report"
            report_dir.mkdir(exist_ok=True)
            with ReportContext(output_dir=report_dir, open_browser=False) as report:
                art = pipe.run(dns_log_path=dns_path, http_log_path=http_path,
                               ssl_log_path=ssl_path, visualize=False)
                rpt_path = report.finalise(art)
            if rpt_path and Path(rpt_path).exists():
                with _state_lock:
                    _state["report_html"] = Path(rpt_path).read_text(encoding="utf-8")
        else:
            art = pipe.run(dns_log_path=dns_path, http_log_path=http_path,
                           ssl_log_path=ssl_path, visualize=False)

        with _state_lock:
            _state["artifacts"] = art
            _state["shap_df"]   = art.shap_values if not art.shap_values.empty else None
            _state["complete"]  = True
        _log("\n✅  Pipeline complete.")

    except Exception as exc:
        import traceback
        _log(f"\n❌  ERROR: {exc}\n{traceback.format_exc()}")
        with _state_lock:
            _state["complete"] = True
            _state["error"]    = str(exc)
    finally:
        _detach_log_handler(handler)
        with _state_lock:
            _state["running"] = False


def _run_synthetic_pipeline(cfg_dict: dict, days: int, bg_rows: int,
                             noisy_rows: int, seed: int, generate_report: bool):
    """Background thread: synthetic data pipeline execution."""
    handler = _attach_log_handler()
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from analytic_pipeline.generate_synthetic_data import (
            SyntheticDataGenerator, evaluate_detection,
        )
        from analytic_pipeline.report import ReportContext

        _log(f"🧪  Generating synthetic data ({days} days, {bg_rows:,} bg rows/day, seed={seed})…")
        gen = SyntheticDataGenerator(seed=seed)
        conn, dns, http, ssl, labels = gen.generate(
            days=days, bg_rows_per_day=bg_rows, noisy_rows_per_day=noisy_rows,
        )
        _log(f"   conn: {len(conn):,}  dns: {len(dns):,}  http: {len(http):,}  ssl: {len(ssl):,}")

        tmpdir = tempfile.mkdtemp(prefix="cadence_syn_")
        cfg, pipe = _make_pipeline(cfg_dict, tmpdir)
        cfg.io.query_start = str(pd.to_datetime(conn["timestamp"].min(), unit="s", utc=True))[:19]
        cfg.io.query_end   = str(pd.to_datetime(conn["timestamp"].max(), unit="s", utc=True))[:19]

        # Inject DataFrames directly — no file I/O for log slices
        cfg._unified_slices = {"conn": conn, "dns": dns, "http": http, "ssl": ssl}
        conn_path = Path(tmpdir) / "conn.csv"
        conn.to_csv(conn_path, index=False)
        cfg.io.input_csv = conn_path

        _log("▶  Starting CADENCE pipeline…")

        if generate_report:
            report_dir = cfg.io.output_dir / "report"
            report_dir.mkdir(exist_ok=True)
            with ReportContext(output_dir=report_dir, open_browser=False) as report:
                art = pipe.run(visualize=False)
                rpt_path = report.finalise(art)
            if rpt_path and Path(rpt_path).exists():
                with _state_lock:
                    _state["report_html"] = Path(rpt_path).read_text(encoding="utf-8")
        else:
            art = pipe.run(visualize=False)

        syn_eval = None
        if not art.corroboration.empty:
            syn_eval = evaluate_detection(art.corroboration, labels, art.anomalies)
            p  = float(syn_eval["precision"].iloc[0])
            r  = float(syn_eval["recall"].iloc[0])
            f1 = float(syn_eval["f1"].iloc[0])
            _log(f"\n📊  Ground-truth:  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")

        with _state_lock:
            _state["artifacts"] = art
            _state["shap_df"]   = art.shap_values if not art.shap_values.empty else None
            _state["syn_eval"]  = syn_eval
            _state["complete"]  = True
        _log("\n✅  Pipeline complete.")

    except Exception as exc:
        import traceback
        _log(f"\n❌  ERROR: {exc}\n{traceback.format_exc()}")
        with _state_lock:
            _state["complete"] = True
            _state["error"]    = str(exc)
    finally:
        _detach_log_handler(handler)
        with _state_lock:
            _state["running"] = False


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _sidebar():
    return html.Div([
        # Logo
        html.Div([
            html.Div("CADENCE", style=dict(fontFamily=MONO, fontSize=18,
                                            color=C["blue"], fontWeight=700, letterSpacing=-1)),
            html.Div("C2 ANOMALY DETECTION", style=dict(
                fontFamily=MONO, fontSize=9, color=C["dim"], letterSpacing="1.5px", marginTop=2,
            )),
        ], style=dict(marginBottom=20)),
        html.Hr(),

        # Input mode
        section_head("📂 Data Input"),
        field("Input mode",
            dcc.RadioItems(
                id="input_mode",
                options=[
                    {"label": " Synthetic data",                       "value": "synthetic"},
                    {"label": " Unified file (conn+dns+http+ssl)",      "value": "unified"},
                    {"label": " Separate files",                        "value": "separate"},
                ],
                value="synthetic",
                labelStyle=dict(display="block", fontFamily=SANS, fontSize=12,
                                color=C["text"], marginBottom=8, cursor="pointer"),
                inputStyle=dict(marginRight=6, accentColor=C["blue3"]),
            )
        ),

        # Synthetic params (shown/hidden by callback)
        html.Div(id="syn_params", children=[
            field("Days", dcc.Slider(id="syn_days", min=3, max=90, value=30, step=1, marks=None,
                                     tooltip={"placement":"bottom","always_visible":True})),
            field("Background rows / day",
                  dcc.Slider(id="syn_bg_rows", min=1000, max=50000, value=10000, step=1000,
                             marks=None, tooltip={"placement":"bottom","always_visible":True})),
            field("Noisy rows / day",
                  dcc.Slider(id="syn_noisy", min=100, max=5000, value=500, step=100,
                             marks=None, tooltip={"placement":"bottom","always_visible":True})),
            field("Seed", num("syn_seed", 42, 0, 9999)),
            html.Div("4 malicious scenarios + 2 decoys. Ground-truth shown after run.",
                     style=dict(fontFamily=MONO, fontSize=10, color=C["dim"],
                                lineHeight=1.5, marginBottom=12)),
        ]),

        # Unified upload (hidden by default)
        html.Div(id="unified_upload_div", style=dict(display="none"), children=[
            field("Unified log file (CSV or Parquet)",
                  dcc.Upload(id="upload_unified",
                             children=html.Div(["📎 Browse or drag & drop"],
                                               style=dict(fontFamily=MONO, fontSize=11, color=C["muted"])),
                             style=dict(border=f"1px dashed {C['border2']}", borderRadius=6,
                                        padding="10px 12px", cursor="pointer",
                                        background="#0d1117", textAlign="center"),
                             multiple=False)),
            html.Div(id="upload_unified_name", style=dict(fontFamily=MONO, fontSize=10, color=C["dim"])),
        ]),

        # Separate file uploads (hidden by default)
        html.Div(id="separate_upload_div", style=dict(display="none"), children=[
            *[field(f"{t} log (CSV or Parquet)",
                    dcc.Upload(id=f"upload_{t}",
                               children=html.Div([f"📎 {t}.csv / {t}.parquet"],
                                                  style=dict(fontFamily=MONO, fontSize=11, color=C["muted"])),
                               style=dict(border=f"1px dashed {C['border2']}", borderRadius=6,
                                          padding="8px 10px", cursor="pointer",
                                          background="#0d1117", textAlign="center"),
                               multiple=False),
                    html.Div(id=f"upload_{t}_name",
                             style=dict(fontFamily=MONO, fontSize=10, color=C["dim"])),
              ) for t in ["conn", "dns", "http", "ssl"]],
        ]),

        html.Hr(),
        section_head("⚙️ Config I/O"),
        dcc.Upload(id="upload_config",
                   children=html.Div("📂 Load config JSON",
                                      style=dict(fontFamily=MONO, fontSize=11, color=C["muted"])),
                   style=dict(border=f"1px dashed {C['border2']}", borderRadius=6,
                              padding="8px 10px", cursor="pointer",
                              background="#0d1117", textAlign="center", marginBottom=8),
                   multiple=False),
        html.Div(id="config_load_status", style=dict(fontFamily=MONO, fontSize=10, marginBottom=8)),
        html.Button("💾 Download config", id="btn_download_config", n_clicks=0,
                    style=dict(fontFamily=MONO, fontSize=11, background=C["surface"],
                               color=C["muted"], border=f"1px solid {C['border2']}",
                               borderRadius=6, padding="7px 0", cursor="pointer",
                               width="100%", marginBottom=8)),
        dcc.Download(id="download_config"),

        html.Hr(),
        section_head("🚀 Run"),
        dcc.Checklist(id="run_options",
                      options=[
                          {"label": " Generate HTML report", "value": "report"},
                          {"label": " Suppress matplotlib plots", "value": "no_plots"},
                      ],
                      value=["report", "no_plots"],
                      labelStyle=dict(display="block", fontFamily=SANS, fontSize=12,
                                      color=C["text"], marginBottom=8),
                      inputStyle=dict(marginRight=6, accentColor=C["blue3"])),
        btn("btn_run", "▶  Run Pipeline"),
        html.Div(id="run_status_badge", style=dict(fontFamily=MONO, fontSize=11,
                                                    textAlign="center", marginTop=4)),

        dcc.Store(id="store_file_paths", data={}),
        dcc.Store(id="store_config",     data={}),
        dcc.Interval(id="log_poll", interval=800, n_intervals=0, disabled=True),

    ], style=dict(
        width=270, flexShrink=0, background=C["sidebar"],
        borderRight=f"1px solid {C['border']}",
        padding="20px 16px", overflowY="auto", height="100vh",
        position="fixed", top=0, left=0,
    ))


def _config_tabs():
    TAB_STYLE = dict(fontFamily=MONO, fontSize=12, color=C["muted"],
                     background="transparent", border="none",
                     borderBottom=f"2px solid transparent", padding="8px 14px",
                     cursor="pointer")
    SEL_STYLE = dict(**TAB_STYLE, color=C["blue"], borderBottom=f"2px solid {C['blue']}")

    def tab_io():
        return card([
            section_head("I/O Configuration"),
            two_col(
                [field("Output directory", txt("io_output_dir", CFG_DEFAULTS["io_output_dir"])),
                 field("Table name",       txt("io_table_name", CFG_DEFAULTS["io_table_name"])),
                 field("ISF query limit",  num("io_query_limit", CFG_DEFAULTS["io_query_limit"], 1000, 50_000_000, 100_000))],
                [field("Query start (UTC)", txt("io_query_start", CFG_DEFAULTS["io_query_start"])),
                 field("Query end (UTC)",   txt("io_query_end",   CFG_DEFAULTS["io_query_end"])),
                 field("Debug mode",
                       dcc.Checklist(id="io_debug", options=[{"label":" Enable","value":"on"}],
                                     value=[], inputStyle=dict(marginRight=6, accentColor=C["blue3"]),
                                     labelStyle=dict(fontFamily=SANS, fontSize=12, color=C["text"])))],
            )
        ])

    def tab_features():
        return card([
            section_head("Feature Configuration"),
            html.Div("Advanced — edit only if your schema differs from standard Zeek output.",
                     style=dict(fontFamily=MONO, fontSize=11, color=C["dim"], marginBottom=10)),
            field("keep_cols (comma-separated)",
                  dcc.Textarea(id="feat_keep_cols", value=CFG_DEFAULTS["feat_keep_cols"],
                               style=dict(width="100%", minHeight=70, background=C["sidebar"],
                                          color=C["text"], border=f"1px solid {C['border']}",
                                          borderRadius=6, fontFamily=MONO, fontSize=11,
                                          padding="8px 10px", resize="vertical"))),
        ])

    def tab_iforest():
        return card([
            section_head("Isolation Forest"),
            two_col(
                [field("n_estimators",   num("iso_n_est",    CFG_DEFAULTS["iso_n_est"],   10, 2000, 50)),
                 field("max_samples",    num("iso_max_samp", CFG_DEFAULTS["iso_max_samp"], 100, 50000, 500)),
                 field("random_state",   num("iso_rs",       CFG_DEFAULTS["iso_rs"],       0, 9999))],
                [field("contamination",       sldr("iso_contam",  0.001, 0.20,  CFG_DEFAULTS["iso_contam"],  0.005)),
                 field("test_size",           sldr("iso_test_sz", 0.1,   0.5,   CFG_DEFAULTS["iso_test_sz"], 0.05)),
                 field("stability_threshold", sldr("iso_stab",    0.5,   1.0,   CFG_DEFAULTS["iso_stab"],    0.05))],
            )
        ])

    def tab_pair_sax():
        return card([
            section_head("Pair Config"),
            two_col(
                [field("min_observations", num("pair_min_obs",   CFG_DEFAULTS["pair_min_obs"],   2, 100)),
                 field("min_pair_flows",   num("pair_min_flows", CFG_DEFAULTS["pair_min_flows"], 2, 100)),
                 field("max_pairs",        num("pair_max_pairs", CFG_DEFAULTS["pair_max_pairs"], 10, 50000, 100))],
                [field("channel_key",
                       dcc.Checklist(id="pair_channel_key",
                                     options=[{"label": f" {k}", "value": k}
                                              for k in ["src_ip","dst_ip","dst_port","proto","src_port"]],
                                     value=CFG_DEFAULTS["pair_channel_key"],
                                     labelStyle=dict(display="block", fontFamily=MONO, fontSize=11,
                                                     color=C["text"], marginBottom=6),
                                     inputStyle=dict(marginRight=6, accentColor=C["blue3"])))],
            ),
            html.Hr(),
            section_head("SAX Config"),
            two_col(
                [field("word_length",      num("sax_word_len", CFG_DEFAULTS["sax_word_len"], 4, 100)),
                 field("alphabet_size",    num("sax_alpha",    CFG_DEFAULTS["sax_alpha"],    2, 26)),
                 field("min_observations", num("sax_min_obs",  CFG_DEFAULTS["sax_min_obs"],  2, 100)),
                 field("max_acf_lag",      num("sax_max_lag",  CFG_DEFAULTS["sax_max_lag"],  2, 50))],
                [field("cv_threshold",     sldr("sax_cv",    0.0, 2.0, CFG_DEFAULTS["sax_cv"],    0.05)),
                 field("acf_threshold",    sldr("sax_acf",   0.0, 1.0, CFG_DEFAULTS["sax_acf"],   0.05)),
                 field("motif_threshold",  sldr("sax_motif", 0.0, 1.0, CFG_DEFAULTS["sax_motif"], 0.05)),
                 field("min_tests_passing", num("sax_min_t", CFG_DEFAULTS["sax_min_t"], 1, 5))],
            )
        ])

    def tab_periodicity():
        return card([
            section_head("Periodicity Config"),
            two_col(
                [field("min_observations",  num("per_min_obs", CFG_DEFAULTS["per_min_obs"], 5, 500)),
                 field("acf_nlags",         num("per_nlags",   CFG_DEFAULTS["per_nlags"],   5, 200)),
                 field("min_period_s",      num("per_min_per", CFG_DEFAULTS["per_min_per"], 1, 3600))],
                [field("acf_significance_threshold", sldr("per_acf_sig", 0.0, 1.0, CFG_DEFAULTS["per_acf_sig"], 0.05)),
                 field("cv_threshold",               sldr("per_cv",      0.0, 2.0, CFG_DEFAULTS["per_cv"],      0.05)),
                 field("fft_power_ratio_threshold",  sldr("per_fft",     0.0, 1.0, CFG_DEFAULTS["per_fft"],     0.05)),
                 field("confidence_threshold",       sldr("per_conf",    0.0, 1.0, CFG_DEFAULTS["per_conf"],    0.05))],
            )
        ])

    def tab_pelt():
        return card([
            section_head("PELT Changepoint Config"),
            two_col(
                [field("penalty",            txt("pelt_penalty", str(CFG_DEFAULTS["pelt_penalty"]))),
                 field("min_segment_length", num("pelt_min_seg", CFG_DEFAULTS["pelt_min_seg"], 2, 100))],
                [field("min_observations",   num("pelt_min_obs", CFG_DEFAULTS["pelt_min_obs"], 5, 500)),
                 field("max_changepoints",   num("pelt_max_cp",  CFG_DEFAULTS["pelt_max_cp"],  1, 100))],
            )
        ])

    def tab_corroboration():
        return card([
            section_head("Corroboration Config"),
            two_col(
                [field("period_tolerance_pct", sldr("corr_period_tol", 0.0, 0.5,  CFG_DEFAULTS["corr_period_tol"], 0.01)),
                 field("short_ttl_threshold_s", num("corr_short_ttl", CFG_DEFAULTS["corr_short_ttl"], 10, 3600, 10)),
                 field("dga_entropy_threshold", sldr("corr_dga_ent",   1.0, 6.0,  CFG_DEFAULTS["corr_dga_ent"],   0.1)),
                 field("dga_min_label_len",     num("corr_dga_min_len", CFG_DEFAULTS["corr_dga_min_len"], 2, 30)),
                 field("min_score",             sldr("corr_min_score", 0.0, 1.0,  CFG_DEFAULTS["corr_min_score"], 0.01))],
                [field("http_body_cv_threshold", sldr("corr_body_cv",   0.0, 2.0, CFG_DEFAULTS["corr_body_cv"],   0.05)),
                 field("http_uri_cv_threshold",  sldr("corr_uri_cv",    0.0, 2.0, CFG_DEFAULTS["corr_uri_cv"],    0.05)),
                 field("rare_ua_threshold",      sldr("corr_rare_ua",   0.0, 0.5, CFG_DEFAULTS["corr_rare_ua"],   0.01)),
                 field("uri_entropy_threshold",  sldr("corr_uri_ent",   0.0, 8.0, CFG_DEFAULTS["corr_uri_ent"],   0.1)),
                 field("nxdomain_rate_threshold",sldr("corr_nxdomain",  0.0, 1.0, CFG_DEFAULTS["corr_nxdomain"],  0.01)),
                 field("body_cv_trim_pct",       sldr("corr_body_trim", 0.0, 0.2, CFG_DEFAULTS["corr_body_trim"], 0.01))],
            ),
            field("Extra benign domain suffixes (comma-separated)",
                  txt("corr_extra_benign", CFG_DEFAULTS["corr_extra_benign"])),
        ])

    def tab_tls():
        return card([
            section_head("TLS Corroboration Config (H5 / H6)"),
            two_col(
                [field("sni_entropy_threshold",   sldr("tls_sni_ent",   0.0, 5.0, CFG_DEFAULTS["tls_sni_ent"],   0.1)),
                 field("ja3_monotony_threshold",  sldr("tls_ja3_mono",  0.0, 1.0, CFG_DEFAULTS["tls_ja3_mono"],  0.05)),
                 field("cert_reuse_min_sessions", num("tls_cert_reuse", CFG_DEFAULTS["tls_cert_reuse"], 1, 50))],
                [field("cert_age_new_days", num("tls_cert_age",    CFG_DEFAULTS["tls_cert_age"],    1, 365)),
                 field("h5_weight",         sldr("tls_h5_weight",  0.0, 1.0, CFG_DEFAULTS["tls_h5_weight"], 0.05)),
                 field("h6_weight",         sldr("tls_h6_weight",  0.0, 1.0, CFG_DEFAULTS["tls_h6_weight"], 0.05))],
            ),
            field("Known C2 JA3 fingerprints (one per line)",
                  dcc.Textarea(id="tls_known_c2", value=CFG_DEFAULTS["tls_known_c2"],
                               style=dict(width="100%", minHeight=60, background=C["sidebar"],
                                          color=C["text"], border=f"1px solid {C['border']}",
                                          borderRadius=6, fontFamily=MONO, fontSize=11,
                                          padding="8px 10px", resize="vertical"))),
        ])

    def tab_prefilter():
        return card([
            section_head("Pre-filter Config"),
            two_col(
                [field("dst_fanin_threshold",   sldr("pf_fanin",  0.0, 1.0, CFG_DEFAULTS["pf_fanin"],  0.05))],
                [field("failed_conn_threshold", sldr("pf_failed", 0.0, 1.0, CFG_DEFAULTS["pf_failed"], 0.05))],
            )
        ])

    def tab_triage():
        return card([
            section_head("Triage / Priority Scoring"),
            two_col(
                [field("beaconing_std_thresh", sldr("tri_beacon_std", 0.0, 5.0, CFG_DEFAULTS["tri_beacon_std"], 0.1)),
                 field("rare_dst_thresh",      num("tri_rare_dst",    CFG_DEFAULTS["tri_rare_dst"], 1, 1000))],
                [field("high_volume_pct",  sldr("tri_high_vol",  0.0, 0.5, CFG_DEFAULTS["tri_high_vol"], 0.01)),
                 field("off_hour_start",   num("tri_off_start",  CFG_DEFAULTS["tri_off_start"],  0, 23)),
                 field("off_hour_end",     num("tri_off_end",    CFG_DEFAULTS["tri_off_end"],    0, 23))],
            )
        ])

    def tab_scaling():
        return card([
            section_head("Scaling / Variance Filter"),
            two_col(
                [field("threshold",             sldr("scl_thresh",     0.0,  10.0,  CFG_DEFAULTS["scl_thresh"],     0.1)),
                 field("binary_threshold",      sldr("scl_bin_thresh", 0.0,  0.1,   CFG_DEFAULTS["scl_bin_thresh"], 0.001)),
                 field("skew_threshold",        sldr("scl_skew",       0.0,  10.0,  CFG_DEFAULTS["scl_skew"],       0.1))],
                [field("range_ratio_threshold", sldr("scl_rr",         1.0,  1000.0,CFG_DEFAULTS["scl_rr"],         1.0)),
                 field("min_unique",            num("scl_min_uniq",    CFG_DEFAULTS["scl_min_uniq"], 2, 100)),
                 field("min_max_threshold",     sldr("scl_min_max",    1.0,  1000.0,CFG_DEFAULTS["scl_min_max"],    1.0))],
            )
        ])

    tabs_content = {
        "IO": tab_io(), "Features": tab_features(), "Isolation Forest": tab_iforest(),
        "Pair / SAX": tab_pair_sax(), "Periodicity": tab_periodicity(), "PELT": tab_pelt(),
        "Corroboration": tab_corroboration(), "TLS": tab_tls(),
        "Prefilter": tab_prefilter(), "Triage": tab_triage(), "Scaling": tab_scaling(),
    }

    return html.Div([
        # Tab bar
        html.Div([
            html.Button(name, id=f"tab_btn_{i}", n_clicks=0,
                        style=dict(fontFamily=MONO, fontSize=12, background="transparent",
                                   color=C["blue"] if i == 0 else C["muted"], border="none",
                                   borderBottom=f"2px solid {C['blue'] if i == 0 else 'transparent'}",
                                   padding="8px 12px", cursor="pointer", whiteSpace="nowrap",
                                   marginBottom=-1, transition="color 0.15s"))
            for i, name in enumerate(tabs_content)
        ], style=dict(display="flex", borderBottom=f"1px solid {C['border']}",
                      marginBottom=16, overflowX="auto", flexWrap="nowrap")),

        # Tab panels
        *[html.Div(content, id=f"tab_panel_{i}",
                   style=dict(display="block" if i == 0 else "none"))
          for i, content in enumerate(tabs_content.values())],

        dcc.Store(id="active_tab", data=0),
    ])


def _results_panel():
    return html.Div([
        html.Hr(),
        html.Div([
            html.Div("📋 Run Output", style=dict(fontFamily=MONO, fontSize=10, color=C["muted"],
                                                  textTransform="uppercase", letterSpacing="1.5px")),
        ], style=dict(marginBottom=10)),

        # Timer + progress bar (visible while running)
        html.Div(id="timer_display", style=dict(marginBottom=14)),

        # Log console
        html.Div(id="log_console", style=dict(
            fontFamily=MONO, fontSize=11, background=C["console"],
            color=C["blue"], border=f"1px solid {C['border']}",
            borderRadius=6, padding="12px 14px", height=160, overflowY="auto",
            whiteSpace="pre-wrap", wordBreak="break-all", lineHeight=1.6,
            marginBottom=20,
        )),

        # Funnel metrics
        html.Div(id="funnel_metrics"),

        # Downloads
        html.Div(id="download_buttons"),
        dcc.Download(id="dl_priority"),
        dcc.Download(id="dl_periodicity"),
        dcc.Download(id="dl_corroboration"),
        dcc.Download(id="dl_shap"),
        dcc.Download(id="dl_report"),

        # SHAP bar chart
        html.Div(id="shap_section"),

        # Leads table
        html.Div(id="leads_table"),

        # Ground-truth evaluation
        html.Div(id="gt_eval_section"),
    ])


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    title="CADENCE",
    update_title=None,
    suppress_callback_exceptions=True,
)

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>{{%title%}}</title>
  {{%favicon%}}
  {{%css%}}
  <style>{GLOBAL_CSS}</style>
</head>
<body>
  {{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""

app.layout = html.Div([
    _sidebar(),

    # Main content
    html.Div([
        # Header
        html.Div([
            html.Span("CADENCE", style=dict(fontFamily=MONO, fontSize=20,
                                             color=C["blue"], fontWeight=700, letterSpacing=-1)),
            html.Span("C2 ANOMALY DETECTION VIA ENSEMBLE NETWORK CORRELATION EVIDENCE",
                      style=dict(fontFamily=MONO, fontSize=9, color=C["dim"],
                                 letterSpacing="2px", marginLeft=14)),
        ], style=dict(display="flex", alignItems="baseline", marginBottom=8)),
        html.Hr(),

        _config_tabs(),
        _results_panel(),

    ], style=dict(marginLeft=290, padding="20px 28px", overflowY="auto", minHeight="100vh")),

], style=dict(display="flex", background=C["bg"], color=C["text"]))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ── Tab switching ────────────────────────────────────────────────────────────

TAB_NAMES = ["IO","Features","Isolation Forest","Pair / SAX","Periodicity",
             "PELT","Corroboration","TLS","Prefilter","Triage","Scaling"]

@app.callback(
    [Output(f"tab_panel_{i}", "style") for i in range(len(TAB_NAMES))]
    + [Output(f"tab_btn_{i}", "style") for i in range(len(TAB_NAMES))]
    + [Output("active_tab", "data")],
    [Input(f"tab_btn_{i}", "n_clicks") for i in range(len(TAB_NAMES))],
    State("active_tab", "data"),
    prevent_initial_call=True,
)
def switch_tab(*args):
    current = args[-1]
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    btn_id  = ctx.triggered[0]["prop_id"].split(".")[0]
    new_tab = int(btn_id.replace("tab_btn_", ""))

    panels = [dict(display="block" if i == new_tab else "none") for i in range(len(TAB_NAMES))]

    def _btn_style(i):
        active = i == new_tab
        return dict(fontFamily=MONO, fontSize=12, background="transparent",
                    color=C["blue"] if active else C["muted"], border="none",
                    borderBottom=f"2px solid {C['blue'] if active else 'transparent'}",
                    padding="8px 12px", cursor="pointer", whiteSpace="nowrap",
                    marginBottom=-1)

    btns = [_btn_style(i) for i in range(len(TAB_NAMES))]
    return panels + btns + [new_tab]


# ── Input mode visibility ────────────────────────────────────────────────────

@app.callback(
    Output("syn_params",         "style"),
    Output("unified_upload_div", "style"),
    Output("separate_upload_div","style"),
    Input("input_mode", "value"),
)
def toggle_input_mode(mode):
    show  = dict(display="block")
    hide  = dict(display="none")
    if mode == "synthetic":
        return show, hide, hide
    elif mode == "unified":
        return hide, show, hide
    return hide, hide, show


# ── File upload filename display ─────────────────────────────────────────────

def _upload_name_callback(upload_id, name_id, store_key):
    @app.callback(
        Output(name_id, "children"),
        Output("store_file_paths", "data", allow_duplicate=True),
        Input(upload_id, "filename"),
        Input(upload_id, "contents"),
        State("store_file_paths", "data"),
        prevent_initial_call=True,
    )
    def _cb(filename, contents, paths):
        if not filename or not contents:
            raise PreventUpdate
        _, content_str = contents.split(",")
        raw = base64.b64decode(content_str)
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(filename).suffix or ".csv"
        )
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        paths = paths or {}
        paths[store_key] = tmp.name
        return f"✓ {filename}", paths

for _t in ["conn", "dns", "http", "ssl"]:
    _upload_name_callback(f"upload_{_t}", f"upload_{_t}_name", _t)

@app.callback(
    Output("upload_unified_name", "children"),
    Output("store_file_paths", "data", allow_duplicate=True),
    Input("upload_unified", "filename"),
    Input("upload_unified", "contents"),
    State("store_file_paths", "data"),
    prevent_initial_call=True,
)
def _unified_upload(filename, contents, paths):
    if not filename or not contents:
        raise PreventUpdate
    _, content_str = contents.split(",")
    raw = base64.b64decode(content_str)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".csv")
    tmp.write(raw); tmp.flush(); tmp.close()
    paths = paths or {}
    paths["unified"] = tmp.name
    return f"✓ {filename}", paths


# ── Config load ──────────────────────────────────────────────────────────────

@app.callback(
    Output("store_config", "data"),
    Output("config_load_status", "children"),
    Output("config_load_status", "style"),
    Input("upload_config", "contents"),
    Input("upload_config", "filename"),
    prevent_initial_call=True,
)
def load_config(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        _, content_str = contents.split(",")
        raw = base64.b64decode(content_str)
        cfg = json.loads(raw)
        return cfg, f"✓ {filename} loaded", dict(color=C["green2"], fontFamily=MONO, fontSize=10)
    except Exception as e:
        return {}, f"❌ {e}", dict(color=C["err"], fontFamily=MONO, fontSize=10)


# ── Config download ──────────────────────────────────────────────────────────

# Collect all config input IDs
_CFG_INPUT_IDS = [
    "io_output_dir","io_table_name","io_query_start","io_query_end",
    "io_query_limit","io_debug","feat_keep_cols",
    "iso_n_est","iso_max_samp","iso_rs","iso_contam","iso_test_sz","iso_stab",
    "pair_min_obs","pair_min_flows","pair_max_pairs","pair_channel_key",
    "sax_word_len","sax_alpha","sax_min_obs","sax_max_lag",
    "sax_cv","sax_acf","sax_motif","sax_min_t",
    "per_min_obs","per_nlags","per_min_per","per_acf_sig","per_cv","per_fft","per_conf",
    "pelt_penalty","pelt_min_seg","pelt_min_obs","pelt_max_cp",
    "corr_period_tol","corr_short_ttl","corr_dga_ent","corr_dga_min_len","corr_min_score",
    "corr_body_cv","corr_uri_cv","corr_rare_ua","corr_uri_ent","corr_nxdomain","corr_body_trim",
    "corr_extra_benign","tls_sni_ent","tls_ja3_mono","tls_cert_reuse","tls_cert_age",
    "tls_h5_weight","tls_h6_weight","tls_known_c2",
    "pf_fanin","pf_failed","tri_beacon_std","tri_rare_dst","tri_high_vol",
    "tri_off_start","tri_off_end","scl_thresh","scl_bin_thresh","scl_skew",
    "scl_rr","scl_min_uniq","scl_min_max",
]

@app.callback(
    Output("download_config", "data"),
    Input("btn_download_config", "n_clicks"),
    [State(id_, "value") for id_ in _CFG_INPUT_IDS],
    prevent_initial_call=True,
)
def download_config(n_clicks, *vals):
    if not n_clicks:
        raise PreventUpdate
    vals_dict = dict(zip(_CFG_INPUT_IDS, vals))
    cfg_dict  = _build_cfg_dict(vals_dict)
    return dict(
        content=json.dumps(cfg_dict, indent=2, default=str),
        filename="cadence_config.json",
        type="application/json",
    )


# ── Run pipeline ─────────────────────────────────────────────────────────────

@app.callback(
    Output("log_poll",          "disabled"),
    Output("run_status_badge",  "children"),
    Output("run_status_badge",  "style"),
    Input("btn_run", "n_clicks"),
    State("input_mode", "value"),
    State("store_file_paths", "data"),
    State("run_options", "value"),
    State("syn_days",    "value"),
    State("syn_bg_rows", "value"),
    State("syn_noisy",   "value"),
    State("syn_seed",    "value"),
    *[State(id_, "value") for id_ in _CFG_INPUT_IDS],
    prevent_initial_call=True,
)
def start_run(n_clicks, mode, file_paths, run_opts, syn_days, syn_bg_rows, syn_noisy, syn_seed, *cfg_vals):
    if not n_clicks:
        raise PreventUpdate

    with _state_lock:
        if _state["running"]:
            raise PreventUpdate
        _state.update(running=True, logs="", complete=False,
                      artifacts=None, report_html=None, shap_df=None,
                      syn_eval=None, error=None,
                      start_time=time.time(), current_stage=0)

    vals_dict  = dict(zip(_CFG_INPUT_IDS, cfg_vals))
    cfg_dict   = _build_cfg_dict(vals_dict)
    gen_report = "report"   in (run_opts or [])

    if mode == "synthetic":
        t = threading.Thread(
            target=_run_synthetic_pipeline,
            args=(cfg_dict, int(syn_days or 30), int(syn_bg_rows or 10000),
                  int(syn_noisy or 500), int(syn_seed or 42), gen_report),
            daemon=True,
        )
    else:
        paths = file_paths or {}
        if mode == "unified":
            paths = {"unified": paths.get("unified")}
        t = threading.Thread(
            target=_run_file_pipeline,
            args=(cfg_dict, paths, gen_report),
            daemon=True,
        )

    t.start()

    badge_style = dict(color=C["warn"], fontFamily=MONO, fontSize=11)
    return False, "⏳ Running…", badge_style


# ── Poll logs and update results ─────────────────────────────────────────────

@app.callback(
    Output("log_console",       "children"),
    Output("log_poll",          "disabled",    allow_duplicate=True),
    Output("run_status_badge",  "children",    allow_duplicate=True),
    Output("run_status_badge",  "style",       allow_duplicate=True),
    Output("timer_display",     "children"),
    Output("funnel_metrics",    "children"),
    Output("download_buttons",  "children"),
    Output("shap_section",      "children"),
    Output("leads_table",       "children"),
    Output("gt_eval_section",   "children"),
    Input("log_poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_results(n):
    with _state_lock:
        logs          = _state["logs"]
        complete      = _state["complete"]
        running       = _state["running"]
        art           = _state["artifacts"]
        shap_df       = _state["shap_df"]
        syn_eval      = _state["syn_eval"]
        err           = _state["error"]
        start_time    = _state["start_time"]
        current_stage = _state["current_stage"]

    # Log console is always updated
    log_div = logs or "— pipeline output will appear here —"

    # ── Timer and progress bar ────────────────────────────────────────────
    def _fmt_time(seconds):
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m {seconds % 60:02d}s"

    if start_time is not None:
        elapsed = time.time() - start_time

        # Fraction complete based on stage weights
        completed_weight = sum(_STAGE_WEIGHT[:current_stage])
        pct = min(completed_weight * 100, 99 if not complete else 100)

        if complete:
            pct = 100
            stage_label = "Complete"
            elapsed_str = _fmt_time(elapsed)
            est_str = ""
            bar_color = C["green2"]
        else:
            # Estimate remaining time using elapsed / completed_fraction
            if completed_weight > 0.05:
                total_est = elapsed / completed_weight
                remaining = max(0, total_est - elapsed)
                est_str = f"  ~{_fmt_time(remaining)} remaining"
            else:
                est_str = "  estimating…"
            stage_label = (
                _STAGES[current_stage - 1][2] if 0 < current_stage <= len(_STAGES)
                else "Initialising…"
            )
            elapsed_str = _fmt_time(elapsed)
            bar_color = C["blue3"]

        timer_div = html.Div([
            # Top row: stage name + elapsed + remaining
            html.Div([
                html.Span(
                    f"Stage {current_stage}/10 — {stage_label}" if not complete else "✅ Pipeline complete",
                    style=dict(fontFamily=MONO, fontSize=11,
                               color=C["green2"] if complete else C["blue2"])
                ),
                html.Span(
                    f"  {elapsed_str}{est_str}",
                    style=dict(fontFamily=MONO, fontSize=11, color=C["muted"])
                ),
            ], style=dict(display="flex", justifyContent="space-between",
                          alignItems="center", marginBottom=6)),

            # Progress bar
            html.Div(
                html.Div(style=dict(
                    width=f"{pct:.1f}%",
                    height="100%",
                    background=bar_color,
                    borderRadius=3,
                    transition="width 0.8s ease",
                )),
                style=dict(
                    height=6,
                    background=C["surface"],
                    border=f"1px solid {C['border']}",
                    borderRadius=3,
                    overflow="hidden",
                )
            ),

            # Stage tick marks
            html.Div([
                html.Div(style=dict(
                    width=f"{100/10:.1f}%",
                    borderRight=f"1px solid {C['border']}",
                    height=4,
                    display="inline-block",
                    background=bar_color if i < current_stage else "transparent",
                    transition="background 0.8s ease",
                ))
                for i in range(10)
            ], style=dict(display="flex", marginTop=2, height=4)),

        ], style=dict(
            background=C["surface"], border=f"1px solid {C['border']}",
            borderRadius=6, padding="10px 14px", marginBottom=14,
        ))
    else:
        timer_div = html.Div()

    if not complete and running:
        return (log_div, no_update, no_update, no_update,
                timer_div,
                no_update, no_update, no_update, no_update, no_update)

    # Pipeline is done
    poll_disabled = True
    if err:
        badge = "❌ Error"
        badge_style = dict(color=C["err"], fontFamily=MONO, fontSize=11)
    else:
        badge = "✅ Complete"
        badge_style = dict(color=C["green2"], fontFamily=MONO, fontSize=11)

    if art is None:
        return (log_div, poll_disabled, badge, badge_style,
                timer_div,
                no_update, no_update, no_update, no_update, no_update)

    # ── Funnel metrics ────────────────────────────────────────────────────
    def _safe_len(df):
        return len(df) if df is not None and not df.empty else 0

    def _safe_sum(df, col):
        try:
            return int(df[col].sum()) if df is not None and not df.empty and col in df.columns else 0
        except Exception:
            return 0

    metrics = [
        ("Raw rows",         _safe_len(art.raw)),
        ("Pre-filtered",     _safe_len(art.prefiltered)),
        ("IForest anomalies",_safe_len(art.anomalies)),
        ("Pairs eval'd",     _safe_len(art.sax)),
        ("SAX pass",         _safe_sum(art.sax, "sax_prescreen_pass")),
        ("Beacon pairs",     _safe_sum(art.periodicity, "is_beacon_pair")),
        ("Corroborated",     _safe_sum(art.corroboration, "corroborated")),
    ]

    funnel = html.Div([
        html.Div("Pipeline Funnel", style=dict(fontFamily=MONO, fontSize=10, color=C["muted"],
                                               textTransform="uppercase", letterSpacing="1.5px",
                                               marginBottom=10)),
        html.Div([
            html.Div([
                html.Div(f"{v:,}", style=dict(fontFamily=MONO, fontSize=18, color=C["blue"],
                                              fontWeight=600, marginBottom=4)),
                html.Div(lbl, style=dict(fontFamily=SANS, fontSize=10, color=C["muted"],
                                         lineHeight=1.3)),
            ], style=dict(background=C["surface"], border=f"1px solid {C['border']}",
                          borderRadius=6, padding="10px 8px", flex=1))
            for lbl, v in metrics
        ], style=dict(display="flex", gap=8, marginBottom=20)),
    ])

    # ── Download buttons ──────────────────────────────────────────────────
    dl_btns = []
    dl_style = dict(fontFamily=MONO, fontSize=11, background=C["surface"],
                    color=C["text"], border=f"1px solid {C['border2']}",
                    borderRadius=6, padding="7px 14px", cursor="pointer", marginRight=8, marginBottom=8)

    if art.priority is not None and not art.priority.empty:
        dl_btns.append(html.Button("📥 Priority CSV",      id="btn_dl_priority",      n_clicks=0, style=dl_style))
    if art.periodicity is not None and not art.periodicity.empty:
        dl_btns.append(html.Button("📥 Periodicity CSV",   id="btn_dl_periodicity",   n_clicks=0, style=dl_style))
    if art.corroboration is not None and not art.corroboration.empty:
        dl_btns.append(html.Button("📥 Corroboration CSV", id="btn_dl_corroboration", n_clicks=0, style=dl_style))
    if shap_df is not None:
        dl_btns.append(html.Button("📥 SHAP CSV",          id="btn_dl_shap",          n_clicks=0,
                                   style=dict(**dl_style, color=C["blue2"],
                                              border=f"1px solid {C['blue3']}")))
    if _state["report_html"]:
        dl_btns.append(html.Button("📥 HTML Report",       id="btn_dl_report",        n_clicks=0, style=dl_style))

    downloads = html.Div([
        html.Div("Downloads", style=dict(fontFamily=MONO, fontSize=10, color=C["muted"],
                                         textTransform="uppercase", letterSpacing="1.5px",
                                         marginBottom=10)),
        html.Div(dl_btns, style=dict(display="flex", flexWrap="wrap")),
    ], style=dict(marginBottom=20)) if dl_btns else html.Div()

    # ── SHAP bar chart ────────────────────────────────────────────────────
    shap_section = html.Div()
    if shap_df is not None:
        shap_cols   = [c for c in shap_df.columns if c.startswith("shap_") and c != "shap_sum"]
        mean_abs    = shap_df[shap_cols].abs().mean().sort_values(ascending=False)
        mean_abs.index = [c.replace("shap_", "") for c in mean_abs.index]
        max_val = float(mean_abs.max()) if len(mean_abs) else 1.0

        bars = [
            html.Div([
                html.Div(feat, style=dict(width=170, fontFamily=MONO, fontSize=10,
                                          color=C["muted"], textAlign="right",
                                          flexShrink=0, paddingRight=10)),
                html.Div(html.Div(style=dict(
                    width=f"{(val/max_val)*100:.1f}%", height="100%",
                    background=f"linear-gradient(90deg,{C['blue3']},{C['blue']})",
                    borderRadius=3,
                )), style=dict(flex=1, height=14, background="#0d1117", borderRadius=3)),
                html.Div(f"{val:.3f}", style=dict(width=36, fontFamily=MONO, fontSize=10,
                                                   color=C["blue"], textAlign="right", flexShrink=0)),
            ], style=dict(display="flex", alignItems="center", gap=8, marginBottom=6))
            for feat, val in mean_abs.items()
        ]

        shap_section = card([
            html.Div("SHAP Feature Importance", style=dict(fontFamily=MONO, fontSize=11,
                                                            color=C["blue2"], marginBottom=14)),
            html.Div([
                html.Span("ℹ️ ", style=dict(flexShrink=0)),
                html.Span([
                    "SHAP values explain ", html.Strong("why each channel was flagged by Isolation Forest",
                                                         style=dict(color=C["blue"])),
                    " — not why it is believed to be C2. "
                    "Corroboration hypotheses H1–H6 explain beacon attribution."
                ], style=dict(fontFamily=SANS, fontSize=11, color=C["muted"], lineHeight=1.6)),
            ], style=dict(display="flex", gap=8, background="#0d1f38",
                          border=f"1px solid {C['blue3']}", borderRadius=6,
                          padding="10px 14px", marginBottom=16)),
            html.Div("Mean |SHAP| — IForest feature importance across all pairs",
                     style=dict(fontFamily=MONO, fontSize=10, color=C["dim"], marginBottom=12)),
            *bars,
        ], marginBottom=20)

    # ── Leads table ───────────────────────────────────────────────────────
    leads_div = html.Div()
    if art.corroboration is not None and not art.corroboration.empty:
        DISPLAY_COLS = ["src_ip","dst_ip","dst_port","proto",
                        "triage_score","beacon_confidence","dominant_period_s","mitre_techniques"]
        show_cols = [c for c in DISPLAY_COLS if c in art.corroboration.columns]
        df = (art.corroboration[show_cols]
              .sort_values("triage_score", ascending=False)
              .reset_index(drop=True))

        def _score_color(s):
            try:
                v = float(s)
                return C["err"] if v > 0.85 else (C["warn"] if v > 0.7 else C["green2"])
            except Exception:
                return C["text"]

        th_style = dict(padding="8px 12px", textAlign="left", color=C["muted"],
                        fontWeight=600, fontSize=10, letterSpacing="0.5px",
                        background=C["surface"], borderBottom=f"1px solid {C['border']}")
        td_style = dict(padding="8px 12px", fontSize=11, fontFamily=MONO)

        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            bg = "transparent" if i % 2 == 0 else "#0f1318"
            cells = []
            for col in show_cols:
                val = row[col]
                style = dict(**td_style, background=bg)
                if col == "triage_score":
                    style["color"] = _score_color(val)
                    style["fontWeight"] = 600
                elif col in ("src_ip","dst_ip"):
                    style["color"] = C["err"] if col == "dst_ip" else C["text"]
                elif col == "dominant_period_s":
                    style["color"] = C["blue"]
                elif col == "mitre_techniques":
                    style["color"] = C["dim"]
                    style["fontSize"] = 10
                else:
                    style["color"] = C["muted"]
                cells.append(html.Td(str(val) if val is not None else "", style=style))
            rows.append(html.Tr(cells, style=dict(borderBottom=f"1px solid {C['border']}")))

        leads_div = html.Div([
            html.Div("Corroborated Leads", style=dict(fontFamily=MONO, fontSize=10, color=C["muted"],
                                                       textTransform="uppercase", letterSpacing="1.5px",
                                                       marginBottom=10)),
            html.Div(html.Table([
                html.Thead(html.Tr([html.Th(c, style=th_style) for c in show_cols])),
                html.Tbody(rows),
            ], style=dict(width="100%", borderCollapse="collapse", fontFamily=MONO)),
            style=dict(border=f"1px solid {C['border']}", borderRadius=6,
                       overflow="hidden", overflowX="auto", marginBottom=20)),
        ])

    # ── Ground-truth evaluation ───────────────────────────────────────────
    gt_div = html.Div()
    if syn_eval is not None and not syn_eval.empty:
        p  = float(syn_eval["precision"].iloc[0])
        r  = float(syn_eval["recall"].iloc[0])
        f1 = float(syn_eval["f1"].iloc[0])

        def _metric_card(label_text, value, good):
            return html.Div([
                html.Div(f"{value:.3f}", style=dict(fontFamily=MONO, fontSize=22,
                                                     color=C["green2"] if good else C["err"],
                                                     fontWeight=700, marginBottom=4)),
                html.Div(label_text, style=dict(fontFamily=SANS, fontSize=11, color=C["muted"])),
                html.Div("≥ target" if good else "below target",
                         style=dict(fontFamily=MONO, fontSize=9,
                                    color=C["green2"] if good else C["err"], marginTop=2)),
            ], style=dict(background=C["surface"], border=f"1px solid {C['border']}",
                          borderRadius=6, padding="12px 16px", flex=1))

        display_cols = [c for c in ["scenario","malicious","detected"] if c in syn_eval.columns]
        df_gt = syn_eval[display_cols].copy()
        df_gt["result"] = df_gt.apply(
            lambda row: "✅ Detected" if row.get("detected") else (
                "❌ Missed" if row.get("malicious") else "✓ Correctly suppressed"
            ), axis=1
        )

        def _gt_color(result):
            if "✅" in result: return C["green2"]
            if "❌" in result: return C["err"]
            return C["muted"]

        th_s = dict(padding="8px 12px", textAlign="left", color=C["muted"], fontWeight=600,
                    fontSize=10, background=C["surface"], borderBottom=f"1px solid {C['border']}")
        gt_rows = [
            html.Tr([
                html.Td(str(row["scenario"]),
                        style=dict(padding="8px 12px", fontFamily=MONO, fontSize=11,
                                   color=C["text"], background="transparent" if i%2==0 else "#0f1318")),
                html.Td("malicious" if row.get("malicious") else "decoy",
                        style=dict(padding="8px 12px", fontFamily=MONO, fontSize=11,
                                   color=C["warn"] if row.get("malicious") else C["muted"],
                                   background="transparent" if i%2==0 else "#0f1318")),
                html.Td(row["result"],
                        style=dict(padding="8px 12px", fontFamily=MONO, fontSize=11,
                                   color=_gt_color(row["result"]),
                                   background="transparent" if i%2==0 else "#0f1318")),
            ], style=dict(borderBottom=f"1px solid {C['border']}"))
            for i, (_, row) in enumerate(df_gt.iterrows())
        ]

        gt_div = card([
            html.Div("🎯 Ground-Truth Evaluation", style=dict(fontFamily=MONO, fontSize=11,
                                                               color=C["blue2"], marginBottom=14)),
            html.Div("Synthetic mode only — compares corroborated leads against known scenarios.",
                     style=dict(fontFamily=MONO, fontSize=10, color=C["dim"], marginBottom=14)),
            html.Div([
                _metric_card("Precision", p,  p  >= 0.75),
                _metric_card("Recall",    r,  r  >= 0.75),
                _metric_card("F1",        f1, f1 >= 0.60),
            ], style=dict(display="flex", gap=12, marginBottom=16)),
            html.Table([
                html.Thead(html.Tr([html.Th(c, style=th_s)
                                    for c in ["scenario","type","result"]])),
                html.Tbody(gt_rows),
            ], style=dict(width="100%", borderCollapse="collapse", fontFamily=MONO,
                          border=f"1px solid {C['border']}", borderRadius=6)),
        ])

    return (log_div, poll_disabled, badge, badge_style,
            timer_div,
            funnel, downloads, shap_section, leads_div, gt_div)


# ── CSV / report downloads ────────────────────────────────────────────────────

def _make_csv_download(btn_id, dl_id, attr):
    @app.callback(
        Output(dl_id, "data"),
        Input(btn_id, "n_clicks"),
        prevent_initial_call=True,
    )
    def _dl(n):
        if not n:
            raise PreventUpdate
        with _state_lock:
            art = _state["artifacts"]
        if art is None:
            raise PreventUpdate
        df = getattr(art, attr, None)
        if df is None or df.empty:
            raise PreventUpdate
        return dcc.send_data_frame(df.to_csv, f"{attr}.csv", index=False)

for _btn, _dl, _attr in [
    ("btn_dl_priority",      "dl_priority",      "priority"),
    ("btn_dl_periodicity",   "dl_periodicity",   "periodicity"),
    ("btn_dl_corroboration", "dl_corroboration", "corroboration"),
]:
    _make_csv_download(_btn, _dl, _attr)


@app.callback(
    Output("dl_shap", "data"),
    Input("btn_dl_shap", "n_clicks"),
    prevent_initial_call=True,
)
def dl_shap(n):
    if not n:
        raise PreventUpdate
    with _state_lock:
        shap_df = _state["shap_df"]
    if shap_df is None:
        raise PreventUpdate
    return dcc.send_data_frame(shap_df.to_csv, "shap_values.csv", index=False)


@app.callback(
    Output("dl_report", "data"),
    Input("btn_dl_report", "n_clicks"),
    prevent_initial_call=True,
)
def dl_report(n):
    if not n:
        raise PreventUpdate
    with _state_lock:
        html_str = _state["report_html"]
    if not html_str:
        raise PreventUpdate
    return dict(content=html_str, filename="cadence_report.html", type="text/html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CADENCE Dash GUI")
    parser.add_argument("--port",  type=int, default=8050, help="Port (default 8050)")
    parser.add_argument("--debug", action="store_true",    help="Enable Dash debug mode")
    parser.add_argument("--host",  default="127.0.0.1",    help="Host (default 127.0.0.1)")
    args = parser.parse_args()
    print(f"\n🔭  CADENCE Dash GUI  →  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
