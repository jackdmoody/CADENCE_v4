"""
CADENCE: C2 Anomaly Detection via Ensemble Network Correlation Evidence.

A multi-stage behavioral analytic pipeline for detecting C2 beaconing from
Zeek conn, dns, http, and ssl logs. Combines unsupervised machine learning
(Isolation Forest + SHAP) with statistical time-series analysis (binned-count
ACF, Welch PSD, PELT) and six-layer cross-protocol hypothesis testing (H1-H6)
to surface high-confidence analyst-actionable leads.

Architecture (v4)
------------------
- Unified ingestion: single CSV/Parquet with log_type column, or separate files.
- Parquet support alongside CSV.
- CadenceScaler: EDA validation (median imputation, skew transforms, variance
  filtering, range ratio guards) before StandardScaler. ScalingConfig fields
  are now fully wired.
- 15 IForest features: log_bytes_mean replaces raw bytes_mean.
- SHAP TreeExplainer on IForest: per-pair anomaly attribution.
- Streamlit GUI (cadence_app.py): all config fields, live logs, SHAP plots.
- Corroboration via DNS (H1-H2), HTTP (H3-H4), and TLS/SSL (H5-H6).

Quick start
-----------
    from analytic_pipeline import BDPPipeline, BDPConfig
    from pathlib import Path

    cfg = BDPConfig()
    cfg.io.input_csv = Path("data/conn.csv")

    art = BDPPipeline(cfg).run(
        dns_log_path  = "data/dns.csv",
        http_log_path = "data/http.csv",
        ssl_log_path  = "data/ssl.csv",
    )

    from analytic_pipeline.corroboration import print_analyst_brief
    print_analyst_brief(art.corroboration)

    # SHAP: why was each channel flagged by IForest?
    print(art.shap_values.sort_values("shap_sum", ascending=False).head())
"""
from .config import (
    BDPConfig,
    IOConfig,
    IsolationConfig,
    PairConfig,
    PrefilterConfig,
    TriageConfig,
    PeriodicityConfig,
    CorroborationConfig,
    TLSCorroborationConfig,
    SAXConfig,
    PELTConfig,
    ScalingConfig,
)
from .pipeline import BDPPipeline, BDPArtifacts
from .features import CadenceScaler
from .report import ReportContext

__version__ = "4.0.0"

__all__ = [
    "BDPPipeline", "BDPArtifacts", "ReportContext", "CadenceScaler",
    "BDPConfig", "IOConfig", "IsolationConfig", "PairConfig",
    "PrefilterConfig", "TriageConfig", "PeriodicityConfig",
    "CorroborationConfig", "TLSCorroborationConfig",
    "SAXConfig", "PELTConfig", "ScalingConfig",
]
