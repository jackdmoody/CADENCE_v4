#!/usr/bin/env python3
"""
cadence_cli.py
===============
Interactive CLI for the CADENCE C2 Beacon Detection Pipeline.

Provides three modes:
  1. Interactive  — walk through config sections, change what you want, run
  2. Config file  — load a JSON config, optionally override specific fields
  3. Quick run    — all defaults with just file paths specified

Usage
------
    # Interactive mode — prompts for every section
    python cadence_cli.py --interactive

    # Quick run with defaults
    python cadence_cli.py --conn data/conn.csv --dns data/dns.csv --http data/http.csv

    # Quick run with parquet
    python cadence_cli.py --conn data/conn.parquet --dns data/dns.parquet

    # Load config JSON, override output dir
    python cadence_cli.py --config my_config.json --output results/run_42

    # Synthetic mode — generate data and run
    python cadence_cli.py --synthetic --days 30 --report

    # Combined BDP file — auto-split and run
    python cadence_cli.py --combined data/bdp_export.parquet --report

    # Save current config to JSON (dry run)
    python cadence_cli.py --interactive --save-config my_config.json --dry-run

    # Show all defaults
    python cadence_cli.py --show-defaults
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cadence_cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"

def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"

def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m"

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _banner(title: str) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {_bold(title)}")
    print(f"{'=' * w}")


def _prompt(label: str, default: Any, type_hint: type = str) -> Any:
    """Prompt for a single config value with type coercion."""
    default_str = str(default)
    if isinstance(default, bool):
        default_str = "yes" if default else "no"

    raw = input(f"  {label} [{_dim(default_str)}]: ").strip()

    if not raw:
        return default

    # Type coercion
    if isinstance(default, bool):
        return raw.lower() in ("yes", "y", "true", "1")
    elif isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            print(f"    {_yellow('Invalid integer, keeping default')}")
            return default
    elif isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            print(f"    {_yellow('Invalid float, keeping default')}")
            return default
    elif isinstance(default, tuple):
        # For tuples of strings/ints, accept comma-separated
        items = [x.strip() for x in raw.split(",")]
        if all(isinstance(x, int) for x in default):
            try:
                return tuple(int(x) for x in items)
            except ValueError:
                print(f"    {_yellow('Invalid, keeping default')}")
                return default
        return tuple(items)
    elif isinstance(default, Path):
        return Path(raw)
    else:
        return raw


def _prompt_section(name: str, dc_instance: Any) -> Any:
    """Interactively prompt for all fields in a dataclass section."""
    _banner(f"Config: {name}")

    # Skip tuple fields that are long lists (keep_cols, common_ports, etc.)
    skip_fields = {"keep_cols", "protected_features", "meta_cols", "common_ports",
                   "ja3_known_c2", "extra_benign_domain_suffixes"}

    fields = dataclasses.fields(dc_instance)
    print(f"  {_dim(f'{len(fields)} parameters — press Enter to keep default')}\n")

    changes = {}
    for f in fields:
        if f.name in skip_fields:
            continue
        # Handle nested dataclass (TLSCorroborationConfig)
        val = getattr(dc_instance, f.name)
        if dataclasses.is_dataclass(val):
            val = _prompt_section(f"  {name}.{f.name}", val)
            changes[f.name] = val
        else:
            new_val = _prompt(f.name, val)
            if new_val != val:
                changes[f.name] = new_val

    if changes:
        # Reconstruct the dataclass with changes
        d = dataclasses.asdict(dc_instance)
        for k, v in changes.items():
            if dataclasses.is_dataclass(v):
                d[k] = dataclasses.asdict(v)
            else:
                d[k] = v
        # Handle nested TLS config
        if "tls" in d and isinstance(d["tls"], dict):
            from analytic_pipeline.config import TLSCorroborationConfig
            tls_d = d.pop("tls")
            return type(dc_instance)(**d, tls=TLSCorroborationConfig(**tls_d))
        return type(dc_instance)(**d)

    return dc_instance


def _prompt_yes_no(question: str, default: bool = True) -> bool:
    """Simple yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {question} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1")


# ---------------------------------------------------------------------------
# Config sections menu
# ---------------------------------------------------------------------------

SECTION_NAMES = [
    ("io",            "I/O paths and query settings"),
    ("isolation",     "Isolation Forest parameters"),
    ("pair",          "Channel grouping and filtering"),
    ("sax",           "SAX pre-screening"),
    ("periodicity",   "ACF + FFT periodicity analysis"),
    ("pelt",          "PELT changepoint detection"),
    ("corroboration", "Multi-source corroboration (H1-H6)"),
    ("prefilter",     "Domain-knowledge pre-filter"),
    ("scaling",       "Variance filtering and scaling"),
    ("triage",        "Priority scoring"),
]


def _interactive_config() -> "BDPConfig":
    """Walk through config sections interactively."""
    from analytic_pipeline.config import BDPConfig

    cfg = BDPConfig()

    _banner("CADENCE Configuration")
    print(f"  {_dim('Select which sections to configure.')}")
    print(f"  {_dim('Press Enter to skip a section (keeps defaults).')}\n")

    print(f"  {'#':<4} {'Section':<18} Description")
    print(f"  {'—'*4} {'—'*18} {'—'*35}")
    for i, (key, desc) in enumerate(SECTION_NAMES, 1):
        print(f"  {i:<4} {_cyan(key):<27} {desc}")

    print()
    raw = input(f"  Sections to configure (e.g. 1,3,7 or 'all') [{_dim('Enter=skip all')}]: ").strip()

    if not raw:
        print(f"\n  {_green('Using all defaults.')}")
        return cfg

    if raw.lower() == "all":
        indices = list(range(len(SECTION_NAMES)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            indices = [i for i in indices if 0 <= i < len(SECTION_NAMES)]
        except ValueError:
            print(f"  {_yellow('Invalid input, using defaults.')}")
            return cfg

    for idx in indices:
        key, _ = SECTION_NAMES[idx]
        section = getattr(cfg, key)
        updated = _prompt_section(key, section)
        setattr(cfg, key, updated)

    return cfg


def _show_config(cfg: "BDPConfig") -> None:
    """Pretty-print the current config."""
    _banner("Current Configuration")
    d = cfg.as_dict()
    for section_name, section_dict in d.items():
        print(f"\n  {_cyan(section_name)}:")
        if isinstance(section_dict, dict):
            for k, v in section_dict.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        val_str = str(v2)
                        if len(val_str) > 60:
                            val_str = val_str[:57] + "..."
                        print(f"      {k2:<35} = {val_str}")
                else:
                    val_str = str(v)
                    if len(val_str) > 60:
                        val_str = val_str[:57] + "..."
                    print(f"    {k:<35} = {val_str}")
        else:
            print(f"    {section_dict}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cadence_cli",
        description="CADENCE — C2 Beacon Detection Pipeline (CLI Interface)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive
  %(prog)s --conn data/conn.csv --dns data/dns.csv --http data/http.csv --report
  %(prog)s --synthetic --days 30 --report --browser
  %(prog)s --combined data/bdp_export.parquet --output results/
  %(prog)s --config my_config.json --override isolation.n_estimators=300
  %(prog)s --show-defaults
        """,
    )

    # Input modes
    inp = p.add_argument_group("Input")
    inp.add_argument("--conn",       type=str, help="Path to conn log (CSV/Parquet)")
    inp.add_argument("--dns",        type=str, help="Path to DNS log (CSV/Parquet)")
    inp.add_argument("--http",       type=str, help="Path to HTTP log (CSV/Parquet)")
    inp.add_argument("--ssl",        type=str, help="Path to SSL/TLS log (CSV/Parquet)")
    inp.add_argument("--combined",   type=str, help="Path to combined BDP export (auto-split)")
    inp.add_argument("--log-type-col", type=str, default="event.dataset",
                     help="Column name for log type in combined file (default: event.dataset)")

    # Synthetic mode
    syn = p.add_argument_group("Synthetic data")
    syn.add_argument("--synthetic",  action="store_true", help="Generate synthetic data and run")
    syn.add_argument("--days",       type=int, default=30, help="Simulation days (default: 30)")
    syn.add_argument("--bg-rows",    type=int, default=30_000, help="Background rows/day (default: 30000)")
    syn.add_argument("--noisy-rows", type=int, default=1_000, help="Noisy rows/day (default: 1000)")
    syn.add_argument("--seed",       type=int, default=42, help="RNG seed (default: 42)")

    # Config
    conf = p.add_argument_group("Configuration")
    conf.add_argument("--config",       type=str, help="Load config from JSON file")
    conf.add_argument("--save-config",  type=str, help="Save final config to JSON before running")
    conf.add_argument("--interactive",  action="store_true", help="Interactive config walkthrough")
    conf.add_argument("--override",     type=str, nargs="*", metavar="KEY=VALUE",
                      help="Override config fields (e.g. isolation.n_estimators=300)")
    conf.add_argument("--show-defaults", action="store_true", help="Print default config and exit")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--output",     type=str, default="results", help="Output directory (default: results)")
    out.add_argument("--report",     action="store_true", help="Generate HTML report")
    out.add_argument("--browser",    action="store_true", help="Open report in browser")
    out.add_argument("--visualize",  action="store_true", help="Render matplotlib diagnostic plots")
    out.add_argument("--dry-run",    action="store_true", help="Build config and exit without running")
    out.add_argument("--quiet",      action="store_true", help="Suppress info-level log output")

    return p


# ---------------------------------------------------------------------------
# Config overrides from CLI flags
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: "BDPConfig", overrides: list[str]) -> "BDPConfig":
    """
    Apply key=value overrides like 'isolation.n_estimators=300'.
    Supports dotted paths into nested config sections.
    """
    for item in overrides:
        if "=" not in item:
            print(f"  {_yellow(f'Skipping invalid override (no =): {item}')}")
            continue

        key, val = item.split("=", 1)
        parts = key.strip().split(".")

        if len(parts) == 1:
            print(f"  {_yellow(f'Override must be section.field=value: {item}')}")
            continue

        section_name = parts[0]
        field_name = parts[1]

        section = getattr(cfg, section_name, None)
        if section is None:
            print(f"  {_yellow(f'Unknown config section: {section_name}')}")
            continue

        # Handle nested TLS config
        if len(parts) == 3 and parts[1] == "tls":
            field_name = parts[2]
            section = section.tls

        if not hasattr(section, field_name):
            print(f"  {_yellow(f'Unknown field: {key}')}")
            continue

        current = getattr(section, field_name)

        # Type coerce
        try:
            if isinstance(current, bool):
                coerced = val.lower() in ("true", "yes", "1")
            elif isinstance(current, int):
                coerced = int(val)
            elif isinstance(current, float):
                coerced = float(val)
            elif isinstance(current, Path):
                coerced = Path(val)
            else:
                coerced = val
        except (ValueError, TypeError):
            print(f"  {_yellow(f'Type error for {key}={val}, keeping default')}")
            continue

        setattr(section, field_name, coerced)
        print(f"  {_green(f'Override: {key} = {coerced}')}")

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    from analytic_pipeline.config import BDPConfig

    # Show defaults and exit
    if args.show_defaults:
        cfg = BDPConfig()
        _show_config(cfg)
        return

    # Build config
    if args.config:
        print(f"  Loading config from {_cyan(args.config)}")
        cfg = BDPConfig.from_json(args.config)
    elif args.interactive:
        cfg = _interactive_config()
    else:
        cfg = BDPConfig()

    # Apply CLI overrides
    if args.override:
        cfg = _apply_overrides(cfg, args.override)

    # Set output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.io.output_dir = output_dir

    # Show final config
    if args.interactive or args.config:
        if _prompt_yes_no("Show final config?", default=False):
            _show_config(cfg)

    # Save config if requested
    if args.save_config:
        cfg.to_json(args.save_config)
        print(f"\n  {_green(f'Config saved to {args.save_config}')}")

    if args.dry_run:
        print(f"\n  {_yellow('Dry run — exiting without running pipeline.')}")
        return

    # ── Resolve input files ───────────────────────────────────────────────

    conn_path = args.conn
    dns_path  = args.dns
    http_path = args.http
    ssl_path  = args.ssl
    labels    = None

    if args.combined:
        _banner("Splitting combined log file")
        from analytic_pipeline.loaders import split_combined_log
        paths = split_combined_log(
            args.combined,
            log_type_col=args.log_type_col,
            output_dir=output_dir / "split_logs",
        )
        conn_path = str(paths["conn"]) if "conn" in paths else conn_path
        dns_path  = str(paths["dns"])  if "dns"  in paths else dns_path
        http_path = str(paths["http"]) if "http" in paths else http_path
        ssl_path  = str(paths["ssl"])  if "ssl"  in paths else ssl_path

        if conn_path:
            cfg.io.input_csv = Path(conn_path)

    elif args.synthetic:
        _banner("Generating synthetic data")
        t0 = time.perf_counter()

        from analytic_pipeline.generate_synthetic_data import SyntheticDataGenerator

        gen = SyntheticDataGenerator(seed=args.seed)
        conn, dns, http, ssl_df, labels = gen.generate(
            days=args.days,
            bg_rows_per_day=args.bg_rows,
            noisy_rows_per_day=args.noisy_rows,
        )

        data_dir = output_dir / "synthetic_data"
        data_dir.mkdir(exist_ok=True)

        conn_path = str(data_dir / "conn.csv")
        dns_path  = str(data_dir / "dns.csv")
        http_path = str(data_dir / "http.csv")
        ssl_path  = str(data_dir / "ssl.csv")

        conn.to_csv(conn_path, index=False)
        dns.to_csv(dns_path, index=False)
        http.to_csv(http_path, index=False)
        ssl_df.to_csv(ssl_path, index=False)

        elapsed = time.perf_counter() - t0
        print(f"  Generated {len(conn):,} conn, {len(dns):,} dns, "
              f"{len(http):,} http, {len(ssl_df):,} ssl rows in {elapsed:.1f}s")

        cfg.io.input_csv = Path(conn_path)

        import pandas as pd
        cfg.io.query_start = str(pd.to_datetime(conn["timestamp"].min(), unit="s", utc=True))[:19]
        cfg.io.query_end   = str(pd.to_datetime(conn["timestamp"].max(), unit="s", utc=True))[:19]

    else:
        if conn_path:
            cfg.io.input_csv = Path(conn_path)

    # Validate we have at least a conn log
    if cfg.io.input_csv is None:
        print(f"\n  {_red('No conn log specified.')}")
        print(f"  Use --conn, --synthetic, or --combined to provide input data.")
        print(f"  Run with --help for usage examples.")
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────

    _banner("Running CADENCE Pipeline")
    print(f"  Conn:   {cfg.io.input_csv}")
    print(f"  DNS:    {dns_path or 'none'}")
    print(f"  HTTP:   {http_path or 'none'}")
    print(f"  SSL:    {ssl_path or 'none'}")
    print(f"  Output: {output_dir.resolve()}")
    print()

    from analytic_pipeline import BDPPipeline

    t1 = time.perf_counter()

    if args.report:
        from analytic_pipeline.report import ReportContext
        report_dir = output_dir / "report"
        report_dir.mkdir(exist_ok=True)

        with ReportContext(
            output_dir=report_dir,
            open_browser=args.browser,
        ) as report:
            pipe = BDPPipeline(cfg)
            art  = pipe.run(
                dns_log_path  = dns_path,
                http_log_path = http_path,
                ssl_log_path  = ssl_path,
                visualize     = args.visualize,
            )
            report_path = report.finalise(art, labels=labels)
        print(f"\n  HTML report: {report_path.resolve()}")
    else:
        pipe = BDPPipeline(cfg)
        art  = pipe.run(
            dns_log_path  = dns_path,
            http_log_path = http_path,
            ssl_log_path  = ssl_path,
            visualize     = args.visualize,
        )

    elapsed = time.perf_counter() - t1
    print(f"\n  Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Print results ─────────────────────────────────────────────────────

    _banner("Results")

    # Funnel
    n_raw    = len(art.raw)
    n_pre    = len(art.prefiltered) if not art.prefiltered.empty else 0
    n_anom   = len(art.anomalies)
    n_pairs  = len(art.sax) if not art.sax.empty else 0
    n_sax    = int(art.sax["sax_prescreen_pass"].sum()) if not art.sax.empty else 0
    n_beacon = int(art.periodicity["is_beacon_pair"].sum()) if not art.periodicity.empty else 0
    n_corr   = int(art.corroboration["corroborated"].sum()) if not art.corroboration.empty else 0

    print(f"\n  {'Stage':<40} {'Count':>10}")
    print(f"  {'—'*40} {'—'*10}")
    print(f"  {'Conn log rows':<40} {n_raw:>10,}")
    print(f"  {'Pre-filter removed':<40} {n_pre:>10,}")
    print(f"  {'IForest anomalies':<40} {n_anom:>10,}")
    print(f"  {'Channels evaluated':<40} {n_pairs:>10,}")
    print(f"  {'SAX-passing → full ACF':<40} {n_sax:>10,}")
    print(f"  {'Beacon channels':<40} {n_beacon:>10,}")
    print(f"  {_bold('Corroborated leads'):<49} {_bold(str(n_corr)):>10}")

    # Analyst brief
    if not art.corroboration.empty and n_corr > 0:
        from analytic_pipeline.corroboration import print_analyst_brief
        print()
        print_analyst_brief(art.corroboration)

    # Ground truth (synthetic mode)
    if labels is not None and not art.corroboration.empty:
        from analytic_pipeline.generate_synthetic_data import evaluate_detection
        results = evaluate_detection(art.corroboration, labels, art.anomalies)

        if not results.empty:
            precision = float(results["precision"].iloc[0])
            recall    = float(results["recall"].iloc[0])
            f1        = float(results["f1"].iloc[0])

            _banner("Ground-Truth Evaluation")
            print(f"\n  {'Scenario':<30} {'Type':>10} {'Result':>10}")
            print(f"  {'—'*30} {'—'*10} {'—'*10}")
            for _, row in results.iterrows():
                detected_str = _green("YES ✓") if row.get("detected") else _red("MISS ✗")
                mal_str = "malicious" if row.get("malicious") else "decoy"
                print(f"  {row['scenario']:<30} {mal_str:>10} {detected_str:>10}")

            p_color = _green if precision >= 0.75 else _red
            r_color = _green if recall >= 0.75 else _red
            f_color = _green if f1 >= 0.60 else _red

            print(f"\n  Precision : {p_color(f'{precision:.3f}')}")
            print(f"  Recall    : {r_color(f'{recall:.3f}')}")
            print(f"  F1        : {f_color(f'{f1:.3f}')}")

    total = time.perf_counter() - t1
    _banner(f"Done — total wall time {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
