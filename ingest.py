"""
Incremental Trade Ingestion Pipeline

CLI entry point for discovering daily CSV exports, ingesting into SQLite,
deduplicating, and regenerating the dashboard.

Usage:
    python ingest.py                  # Scan D:\\futures\\daily\\ for new CSVs
    python ingest.py file.csv         # Ingest specific file(s)
    python ingest.py --bootstrap      # One-time load from all-trades.csv
    python ingest.py --executions F   # Ingest from execution log CSV
    python ingest.py --ninjatrader    # Sync from NinjaTrader SQLite database
    python ingest.py --history        # Show ingestion log
    python ingest.py --regenerate     # Regenerate dashboard without ingesting
    python ingest.py --no-regime      # Skip regime_analysis.py after regeneration
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import execution_converter
import nt_connector
import process_trades
import trade_store

DAILY_DIR = Path(r"D:\futures\daily")
PROCESSED_DIR = DAILY_DIR / "processed"
BOOTSTRAP_CSV = Path(r"D:\futures\code\all-trades.csv")
OUTPUT_JS = Path(__file__).parent / "data" / "trades.js"


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV (excludes header)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return sum(1 for _ in f) - 1


def count_sim_rows(path: Path) -> int:
    """Count rows with Sim-* accounts in a CSV."""
    count = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 3 and row[2].strip().startswith("Sim-"):
                count += 1
    return count


def ingest_file(conn, csv_path: Path, move_to_processed: bool = True) -> tuple[int, int]:
    """Ingest a single CSV file into the trade store.
    Returns (trades_new, trades_dup).
    """
    path_str = str(csv_path.resolve())
    file_name = csv_path.name

    # Check if already ingested
    if trade_store.is_file_ingested(conn, path_str):
        print(f"  SKIP (already ingested): {file_name}")
        return 0, 0

    # Count raw rows for logging
    rows_raw = count_csv_rows(csv_path)
    rows_sim = count_sim_rows(csv_path)

    # Read and consolidate trades using existing logic
    trades = process_trades.read_trades(str(csv_path))

    if not trades:
        print(f"  {file_name}: 0 Sim trades found")
        trade_store.log_ingestion(conn, path_str, file_name,
                                  rows_raw, rows_sim, 0, 0)
        if move_to_processed:
            _move_to_processed(csv_path)
        return 0, 0

    # Insert with dedup
    new_count, dup_count = trade_store.insert_trades(conn, trades, source_file=file_name)

    # Log ingestion
    trade_store.log_ingestion(conn, path_str, file_name,
                              rows_raw, rows_sim, new_count, dup_count)

    print(f"  {file_name}: {new_count} new trades, {dup_count} duplicates skipped")

    # Move to processed directory
    if move_to_processed:
        _move_to_processed(csv_path)

    return new_count, dup_count


def _move_to_processed(csv_path: Path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / csv_path.name
    # Handle name collision
    if dest.exists():
        stem = csv_path.stem
        suffix = csv_path.suffix
        i = 1
        while dest.exists():
            dest = PROCESSED_DIR / f"{stem}_{i}{suffix}"
            i += 1
    shutil.move(str(csv_path), str(dest))
    print(f"    -> moved to {dest}")


def discover_new_csvs() -> list[Path]:
    """Find CSV files in the daily directory that haven't been ingested."""
    if not DAILY_DIR.exists():
        print(f"Daily directory not found: {DAILY_DIR}")
        return []
    csvs = sorted(DAILY_DIR.glob("*.csv"))
    return csvs


def regenerate_dashboard(conn, run_regime: bool = True) -> None:
    """Read all trades from SQLite and regenerate data/trades.js."""
    trades = trade_store.read_all_trades(conn)
    total_in_db = trade_store.get_trade_count(conn)

    if not trades:
        print("No trades in database — nothing to regenerate.")
        return

    print(f"\nRegenerating dashboard from {total_in_db} trades in DB...")
    output, *_ = process_trades.build_dashboard_output(
        trades, source_label="trades.db", total_trades_raw=total_in_db
    )
    process_trades.write_trades_js(output, OUTPUT_JS)

    if run_regime:
        regime_script = Path(__file__).parent / "regime_analysis.py"
        if regime_script.exists():
            print("\nRunning regime analysis...")
            subprocess.run([sys.executable, str(regime_script)], check=False)


def show_history(conn) -> None:
    """Print the ingestion log."""
    log = trade_store.get_ingestion_log(conn)
    if not log:
        print("No ingestion history.")
        return

    print(f"\n{'Ingested At':<22} {'File':<40} {'Raw':>6} {'Sim':>6} {'New':>6} {'Dup':>6}")
    print("-" * 92)
    for entry in log:
        ts = entry["ingested_at"][:19]
        print(f"{ts:<22} {entry['file_name']:<40} {entry['rows_raw']:>6} "
              f"{entry['rows_sim']:>6} {entry['trades_new']:>6} {entry['trades_dup']:>6}")

    total = trade_store.get_trade_count(conn)
    print(f"\nTotal trades in DB: {total}")


def cmd_bootstrap(conn, run_regime: bool = True) -> None:
    """One-time load from the master all-trades.csv."""
    if not BOOTSTRAP_CSV.exists():
        print(f"Bootstrap CSV not found: {BOOTSTRAP_CSV}")
        sys.exit(1)

    print(f"Bootstrapping from {BOOTSTRAP_CSV}...")
    ingest_file(conn, BOOTSTRAP_CSV, move_to_processed=False)
    regenerate_dashboard(conn, run_regime=run_regime)


def cmd_ingest_files(conn, files: list[str], run_regime: bool = True) -> None:
    """Ingest specific files provided as CLI arguments."""
    total_new = 0
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"  File not found: {f}")
            continue
        new, _ = ingest_file(conn, path, move_to_processed=False)
        total_new += new

    if total_new > 0:
        regenerate_dashboard(conn, run_regime=run_regime)
    else:
        print("No new trades ingested — skipping regeneration.")


def cmd_ingest_executions(conn, csv_path: str, run_regime: bool = True) -> None:
    """Ingest trades reconstructed from an execution log CSV."""
    path = Path(csv_path)
    if not path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    path_str = str(path.resolve())
    file_name = path.name

    print(f"Converting executions from {file_name}...")
    trades = execution_converter.read_executions(str(path))

    if not trades:
        print("  No trades reconstructed — nothing to ingest.")
        return

    # Insert with dedup
    new_count, dup_count = trade_store.insert_trades(conn, trades, source_file=file_name)

    # Log ingestion
    rows_raw = sum(1 for _ in open(path, encoding="utf-8-sig")) - 1
    trade_store.log_ingestion(conn, path_str, file_name,
                              rows_raw, len(trades), new_count, dup_count)

    print(f"  {new_count} new trades, {dup_count} duplicates skipped")

    if new_count > 0:
        regenerate_dashboard(conn, run_regime=run_regime)
    else:
        print("No new trades ingested — skipping regeneration.")


def cmd_ingest_ninjatrader(conn, nt_db_path: str, nt_accounts: str,
                           run_regime: bool = True) -> None:
    """Ingest trades directly from NinjaTrader's SQLite database."""
    print(f"Syncing from NinjaTrader database...")
    trades = nt_connector.read_nt_trades(nt_db_path, accounts=nt_accounts)

    if not trades:
        print("  No trades reconstructed — nothing to ingest.")
        return

    # Insert with dedup
    source_label = f"NinjaTrader:{Path(nt_db_path).name}"
    new_count, dup_count = trade_store.insert_trades(conn, trades,
                                                     source_file=source_label)

    # Log ingestion
    trade_store.log_ingestion(conn, nt_db_path, source_label,
                              len(trades), len(trades), new_count, dup_count)

    print(f"\n  {new_count} new trades, {dup_count} duplicates skipped")

    if new_count > 0:
        regenerate_dashboard(conn, run_regime=run_regime)
    else:
        print("No new trades ingested — skipping regeneration.")


def cmd_scan_daily(conn, run_regime: bool = True) -> None:
    """Scan the daily directory for new CSV files and ingest them."""
    csvs = discover_new_csvs()
    if not csvs:
        print(f"No new CSV files in {DAILY_DIR}")
        return

    print(f"Found {len(csvs)} CSV file(s) in {DAILY_DIR}")
    total_new = 0
    for csv_path in csvs:
        new, _ = ingest_file(conn, csv_path, move_to_processed=True)
        total_new += new

    if total_new > 0:
        regenerate_dashboard(conn, run_regime=run_regime)
    else:
        print("No new trades ingested — skipping regeneration.")


def main():
    parser = argparse.ArgumentParser(
        description="Incremental trade ingestion pipeline"
    )
    parser.add_argument("files", nargs="*", help="Specific CSV files to ingest")
    parser.add_argument("--bootstrap", action="store_true",
                        help="One-time load from all-trades.csv")
    parser.add_argument("--history", action="store_true",
                        help="Show ingestion log")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate dashboard without ingesting")
    parser.add_argument("--executions", type=str, metavar="CSV",
                        help="Ingest trades reconstructed from an execution log CSV")
    parser.add_argument("--ninjatrader", "--nt", action="store_true",
                        help="Sync trades from NinjaTrader's SQLite database")
    parser.add_argument("--nt-db", type=str, default=nt_connector.DEFAULT_NT_DB,
                        help="Path to NinjaTrader.sqlite (default: NT8 user docs)")
    parser.add_argument("--nt-accounts", type=str, default="Sim*",
                        help="Account name filter pattern (default: Sim*)")
    parser.add_argument("--no-regime", action="store_true",
                        help="Skip regime_analysis.py after regeneration")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite database (default: data/trades.db)")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else trade_store.DB_PATH
    conn = trade_store.get_connection(db_path)
    trade_store.init_db(conn)

    run_regime = not args.no_regime

    try:
        if args.history:
            show_history(conn)
        elif args.regenerate:
            regenerate_dashboard(conn, run_regime=run_regime)
        elif args.bootstrap:
            cmd_bootstrap(conn, run_regime=run_regime)
        elif args.executions:
            cmd_ingest_executions(conn, args.executions, run_regime=run_regime)
        elif args.ninjatrader:
            cmd_ingest_ninjatrader(conn, args.nt_db, args.nt_accounts,
                                   run_regime=run_regime)
        elif args.files:
            cmd_ingest_files(conn, args.files, run_regime=run_regime)
        else:
            cmd_scan_daily(conn, run_regime=run_regime)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
