"""
ES Levels Strategy — Technical Indicator Impact Analysis

Analyzes ES trades from all Levels strategy variants against technical indicators
from multiple ES bar timeframes. Determines which indicators affect long and short
trade outcomes. Uses walk-forward validation to avoid overfitting.

Analysis matrix: 4 strategies × 4 timeframes = 16 combinations,
each analyzed 3 ways (combined / long-only / short-only) = up to 48 analyses.
"""

import csv
import math
import re
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from itertools import combinations

TRADES_CSV = r"D:\futures\code\all-trades.csv"
MARKET_DIR = Path(r"D:\futures\data\Market")
OUTPUT_MD = Path(__file__).parent / "reports" / "es_levels_indicator_analysis.md"
OUTPUT_HTML = Path(__file__).parent / "reports" / "es_levels_indicator_analysis.html"

BAR_FILES = {
    "1M": MARKET_DIR / "Snapshot_1M_ES 03-26.csv",
    "2M": MARKET_DIR / "Snapshot_2M_ES 03-26.csv",
    "3M": MARKET_DIR / "Snapshot_3M_ES 03-26.csv",
    "5M": MARKET_DIR / "Snapshot_5M_ES 03-26.csv",
}

BAR_MINUTES = {"1M": 1, "2M": 2, "3M": 3, "5M": 5}

ACCOUNT_MAP = {
    "Sim-Levels": "Levels 1M",
    "Sim-Levels-2M": "Levels 2M",
    "Sim-Levels-3M": "Levels 3M",
    "Sim-Levels-5M": "Levels 5M",
}

SKIP_COLUMNS = {"DateTime", "Close", "Symbol"}
NUM_BUCKETS = 5
TRAIN_RATIO = 0.70
MIN_TRADES = 20  # Skip any analysis pass with fewer trades


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def parse_profit(raw: str) -> float:
    """Parse NinjaTrader profit format: ($175.00) or $125.00"""
    raw = raw.strip()
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.replace("(", "").replace(")", "").replace("$", "").replace(",", "")
    try:
        val = float(cleaned)
        return -val if negative else val
    except ValueError:
        return 0.0


def parse_entry_time(raw: str) -> datetime:
    """Parse US date format: 1/2/2026 8:48:11 AM"""
    raw = raw.strip()
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {raw}")


def load_trades() -> dict:
    """Load trades from all-trades.csv, filtered to ES 03-26 and Levels accounts.
    Returns dict keyed by strategy name -> list of trade dicts.
    """
    print("  Loading trade data...")
    strategies = defaultdict(list)
    total = 0
    skipped = 0

    with open(TRADES_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]

        idx = {h: i for i, h in enumerate(header)}

        for row in reader:
            if len(row) < len(header) - 1:
                continue
            instrument = row[idx["Instrument"]].strip()
            account = row[idx["Account"]].strip()

            if instrument != "ES 03-26":
                skipped += 1
                continue
            if account not in ACCOUNT_MAP:
                skipped += 1
                continue

            strategy = ACCOUNT_MAP[account]
            direction = row[idx["Market pos."]].strip()
            profit = parse_profit(row[idx["Profit"]])
            entry_time = parse_entry_time(row[idx["Entry time"]])

            trade = {
                "strategy": strategy,
                "account": account,
                "direction": direction,
                "profit": profit,
                "entryTime": entry_time,
                "entryDate": entry_time.strftime("%Y-%m-%d"),
            }
            strategies[strategy].append(trade)
            total += 1

    for strat, trades in strategies.items():
        trades.sort(key=lambda t: t["entryTime"])
        longs = sum(1 for t in trades if t["direction"] == "Long")
        shorts = sum(1 for t in trades if t["direction"] == "Short")
        print(f"    {strat}: {len(trades)} trades ({longs} long, {shorts} short)")

    print(f"    Total ES Levels trades: {total} (skipped {skipped} non-matching)")
    return dict(strategies)


def load_bar_data(timeframe: str) -> tuple:
    """Load bar CSV for a timeframe. Returns (indicators_dict, indicator_info).
    indicators_dict: keyed by datetime string -> {col_name: parsed_value}
    indicator_info: {col_name: 'numeric' or 'boolean'}
    """
    csv_path = BAR_FILES[timeframe]
    print(f"  Loading {timeframe} bar data from {csv_path.name}...")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]

        # Read a sample of rows to classify columns
        sample_rows = []
        all_rows = []
        for i, row in enumerate(reader):
            all_rows.append(row)
            if i < 200:
                sample_rows.append(row)

    # Classify columns
    indicator_info = {}
    col_indices = {}

    for ci, col_name in enumerate(header):
        if col_name in SKIP_COLUMNS:
            continue

        # Sample values to determine type
        sample_vals = []
        for row in sample_rows:
            if ci < len(row):
                v = row[ci].strip()
                if v:
                    sample_vals.append(v)

        if not sample_vals:
            continue

        # Check if boolean
        unique_lower = set(v.lower() for v in sample_vals[:100])
        if unique_lower <= {"true", "false"}:
            indicator_info[col_name] = "boolean"
            col_indices[col_name] = ci
            continue

        # Check if numeric
        numeric_count = 0
        null_count = 0
        for v in sample_vals[:100]:
            if v.lower() in ("", "nan", "null", "none"):
                null_count += 1
                continue
            try:
                fv = float(v)
                if not math.isinf(fv) and abs(fv) < 1e15:
                    numeric_count += 1
            except (ValueError, OverflowError):
                pass

        total_sampled = len(sample_vals[:100])
        if null_count > total_sampled * 0.5:
            continue  # Too many nulls
        if numeric_count >= (total_sampled - null_count) * 0.8:
            indicator_info[col_name] = "numeric"
            col_indices[col_name] = ci

    # Now parse all rows
    indicators = {}
    for row in all_rows:
        dt_str = row[0].strip()
        snapshot = {}
        for col_name, ci in col_indices.items():
            raw = row[ci].strip() if ci < len(row) else ""
            ind_type = indicator_info[col_name]
            if ind_type == "boolean":
                if raw.lower() == "true":
                    snapshot[col_name] = True
                elif raw.lower() == "false":
                    snapshot[col_name] = False
                else:
                    snapshot[col_name] = None
            else:
                try:
                    val = float(raw)
                    if math.isinf(val) or abs(val) > 1e15:
                        snapshot[col_name] = None
                    else:
                        snapshot[col_name] = val
                except (ValueError, OverflowError):
                    snapshot[col_name] = None
        indicators[dt_str] = snapshot

    print(f"    Loaded {len(all_rows)} bars, {len(col_indices)} indicators "
          f"({sum(1 for v in indicator_info.values() if v == 'numeric')} numeric, "
          f"{sum(1 for v in indicator_info.values() if v == 'boolean')} boolean)")
    return indicators, indicator_info


# ──────────────────────────────────────────────
# Time-Based Join (No Data Leakage)
# ──────────────────────────────────────────────

def floor_to_bar(dt: datetime, bar_minutes: int) -> datetime:
    """Floor a datetime to the nearest bar boundary."""
    minute = (dt.minute // bar_minutes) * bar_minutes
    return dt.replace(minute=minute, second=0, microsecond=0)


def join_trades_with_bars(trades: list, indicators: dict, timeframe: str) -> list:
    """Join trades with bar indicator snapshots using last COMPLETED bar.

    Algorithm:
    1. Floor trade entry time to bar boundary
    2. Subtract one bar period to get last completed bar
    3. Look up that bar's snapshot
    4. Fallback: try one additional bar back
    """
    bar_min = BAR_MINUTES[timeframe]
    joined = []
    matched = 0
    unmatched = 0

    for t in trades:
        entry_dt = t["entryTime"]
        # Floor to bar boundary
        bar_boundary = floor_to_bar(entry_dt, bar_min)
        # Subtract one bar period to get last completed bar
        completed_bar = bar_boundary - timedelta(minutes=bar_min)

        dt_key = completed_bar.strftime("%Y-%m-%d %H:%M:%S")
        snapshot = indicators.get(dt_key)

        if not snapshot:
            # Fallback: try one more bar back
            fallback = completed_bar - timedelta(minutes=bar_min)
            dt_key = fallback.strftime("%Y-%m-%d %H:%M:%S")
            snapshot = indicators.get(dt_key)

        if snapshot:
            joined.append({**t, "_indicators": snapshot, "_bar_time": dt_key})
            matched += 1
        else:
            unmatched += 1

    match_rate = (matched / (matched + unmatched) * 100) if (matched + unmatched) > 0 else 0
    print(f"    {timeframe} join: {matched} matched, {unmatched} unmatched ({match_rate:.1f}% match rate)")
    return joined


# ──────────────────────────────────────────────
# Analysis Functions (adapted from regime_analysis.py)
# ──────────────────────────────────────────────

def compute_stats(trades: list) -> dict:
    """Compute win rate, PF, P&L for a group of trades."""
    if not trades:
        return {"count": 0, "winRate": 0, "pf": 0, "pnl": 0, "avgTrade": 0}
    profits = [t["profit"] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    w, l = len(wins), len(losses)
    d = w + l
    gp = sum(wins)
    gl = abs(sum(losses))
    return {
        "count": len(trades),
        "winRate": round((w / d) * 100, 1) if d > 0 else 0,
        "pf": round(gp / gl, 2) if gl > 0 else (9999 if gp > 0 else 0),
        "pnl": round(sum(profits), 2),
        "avgTrade": round(statistics.mean(profits), 2) if profits else 0,
    }


def analyze_indicator(joined_trades: list, indicator_name: str, ind_type: str) -> dict | None:
    """Analyze one indicator's predictive power by bucketing trades."""
    if ind_type == "boolean":
        true_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is True]
        false_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is False]
        if not true_trades or not false_trades:
            return None
        true_stats = compute_stats(true_trades)
        false_stats = compute_stats(false_trades)
        pf_spread = abs((true_stats["pf"] if true_stats["pf"] < 9999 else 3)
                        - (false_stats["pf"] if false_stats["pf"] < 9999 else 3))
        return {
            "name": indicator_name,
            "type": "boolean",
            "buckets": [
                {**false_stats, "label": "False", "range": [False, False]},
                {**true_stats, "label": "True", "range": [True, True]},
            ],
            "pfSpread": round(pf_spread, 2),
            "correlation": None,
        }

    # Numeric: get valid values
    valid_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is not None]
    if len(valid_trades) < 20:
        return None

    values = [t["_indicators"][indicator_name] for t in valid_trades]
    # Check for zero variance
    if len(set(values)) < 3:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Compute quintile boundaries
    boundaries = []
    for i in range(1, NUM_BUCKETS):
        idx = int(n * i / NUM_BUCKETS)
        boundaries.append(sorted_vals[idx])

    # Check for degenerate boundaries (all same value)
    if len(set(boundaries)) < 2:
        return None

    # Assign trades to buckets
    buckets = [[] for _ in range(NUM_BUCKETS)]
    for t in valid_trades:
        v = t["_indicators"][indicator_name]
        assigned = False
        for i, b in enumerate(boundaries):
            if v <= b:
                buckets[i].append(t)
                assigned = True
                break
        if not assigned:
            buckets[-1].append(t)

    bucket_stats = []
    pfs = []
    for i, bucket in enumerate(buckets):
        stats = compute_stats(bucket)
        lo = sorted_vals[0] if i == 0 else boundaries[i - 1]
        hi = boundaries[i] if i < len(boundaries) else sorted_vals[-1]
        stats["label"] = f"Q{i+1}"
        stats["range"] = [round(lo, 4), round(hi, 4)]
        bucket_stats.append(stats)
        pfs.append(stats["pf"] if stats["pf"] < 9999 else 3)

    pf_spread = max(pfs) - min(pfs) if pfs else 0

    # Pearson correlation between indicator value and profit
    profits = [t["profit"] for t in valid_trades]
    corr = None
    if len(values) > 10:
        try:
            mean_v = statistics.mean(values)
            mean_p = statistics.mean(profits)
            cov = sum((v - mean_v) * (p - mean_p) for v, p in zip(values, profits)) / len(values)
            std_v = statistics.stdev(values)
            std_p = statistics.stdev(profits)
            if std_v > 0 and std_p > 0:
                corr = round(cov / (std_v * std_p), 4)
        except Exception:
            pass

    return {
        "name": indicator_name,
        "type": "numeric",
        "buckets": bucket_stats,
        "pfSpread": round(pf_spread, 2),
        "correlation": corr,
    }


def find_optimal_threshold(joined_trades: list, indicator_name: str, ind_type: str) -> dict | None:
    """Find the best threshold to use as a no-trade filter."""
    if ind_type == "boolean":
        true_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is True]
        false_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is False]
        if not true_trades or not false_trades:
            return None
        true_s = compute_stats(true_trades)
        false_s = compute_stats(false_trades)
        if true_s["pf"] < false_s["pf"] and true_s["pf"] < 1.0:
            filtered, kept = true_trades, false_trades
            rule = f"{indicator_name} = True"
        elif false_s["pf"] < true_s["pf"] and false_s["pf"] < 1.0:
            filtered, kept = false_trades, true_trades
            rule = f"{indicator_name} = False"
        else:
            return None
        return {
            "rule": rule,
            "indicator": indicator_name,
            "tradesFiltered": len(filtered),
            "tradesKept": len(kept),
            "pnlFiltered": round(sum(t["profit"] for t in filtered), 2),
            "pnlKept": round(sum(t["profit"] for t in kept), 2),
            "pfFiltered": compute_stats(filtered)["pf"],
            "pfKept": compute_stats(kept)["pf"],
            "wrFiltered": compute_stats(filtered)["winRate"],
            "wrKept": compute_stats(kept)["winRate"],
        }

    # Numeric
    valid_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is not None]
    if len(valid_trades) < 20:
        return None

    values = [t["_indicators"][indicator_name] for t in valid_trades]
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    best_filter = None
    best_improvement = 0

    for pct in [10, 15, 20, 25, 30, 33]:
        # Low threshold: filter trades where indicator <= threshold
        lo_idx = int(n * pct / 100)
        if lo_idx >= n:
            continue
        lo_thresh = sorted_vals[lo_idx]
        lo_filtered = [t for t in valid_trades if t["_indicators"][indicator_name] <= lo_thresh]
        lo_kept = [t for t in valid_trades if t["_indicators"][indicator_name] > lo_thresh]

        if lo_filtered and lo_kept:
            fs = compute_stats(lo_filtered)
            ks = compute_stats(lo_kept)
            if fs["pf"] < 1.0 and ks["pf"] > fs["pf"]:
                saved = -fs["pnl"] if fs["pnl"] < 0 else 0
                if saved > best_improvement:
                    best_improvement = saved
                    best_filter = {
                        "rule": f"{indicator_name} < {round(lo_thresh, 4)}",
                        "direction": "below",
                        "threshold": round(lo_thresh, 4),
                        "percentile": pct,
                        "indicator": indicator_name,
                        "tradesFiltered": len(lo_filtered),
                        "tradesKept": len(lo_kept),
                        "pnlFiltered": fs["pnl"],
                        "pnlKept": ks["pnl"],
                        "pfFiltered": fs["pf"],
                        "pfKept": ks["pf"],
                        "wrFiltered": fs["winRate"],
                        "wrKept": ks["winRate"],
                    }

        # High threshold: filter trades where indicator >= threshold
        hi_idx = int(n * (100 - pct) / 100)
        if hi_idx >= n:
            continue
        hi_thresh = sorted_vals[hi_idx]
        hi_filtered = [t for t in valid_trades if t["_indicators"][indicator_name] >= hi_thresh]
        hi_kept = [t for t in valid_trades if t["_indicators"][indicator_name] < hi_thresh]

        if hi_filtered and hi_kept:
            fs = compute_stats(hi_filtered)
            ks = compute_stats(hi_kept)
            if fs["pf"] < 1.0 and ks["pf"] > fs["pf"]:
                saved = -fs["pnl"] if fs["pnl"] < 0 else 0
                if saved > best_improvement:
                    best_improvement = saved
                    best_filter = {
                        "rule": f"{indicator_name} > {round(hi_thresh, 4)}",
                        "direction": "above",
                        "threshold": round(hi_thresh, 4),
                        "percentile": 100 - pct,
                        "indicator": indicator_name,
                        "tradesFiltered": len(hi_filtered),
                        "tradesKept": len(hi_kept),
                        "pnlFiltered": fs["pnl"],
                        "pnlKept": ks["pnl"],
                        "pfFiltered": fs["pf"],
                        "pfKept": ks["pf"],
                        "wrFiltered": fs["winRate"],
                        "wrKept": ks["winRate"],
                    }

    return best_filter


def apply_filter(trades: list, filt: dict) -> tuple:
    """Apply a single filter, returning (filtered_out, kept)."""
    filtered, kept = [], []
    for t in trades:
        ind = t["_indicators"]
        remove = False
        if filt.get("direction"):
            v = ind.get(filt["indicator"])
            if v is not None:
                if filt["direction"] == "below" and v <= filt["threshold"]:
                    remove = True
                elif filt["direction"] == "above" and v >= filt["threshold"]:
                    remove = True
        else:
            # Boolean filter
            v = ind.get(filt["indicator"])
            if "True" in filt["rule"] and v is True:
                remove = True
            elif "False" in filt["rule"] and v is False:
                remove = True
        if remove:
            filtered.append(t)
        else:
            kept.append(t)
    return filtered, kept


def validate_filter(test_trades: list, filt: dict) -> dict:
    """Validate a filter on out-of-sample test trades."""
    filtered, kept = apply_filter(test_trades, filt)
    test_baseline = compute_stats(test_trades)
    fs = compute_stats(filtered) if filtered else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
    ks = compute_stats(kept) if kept else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
    return {
        "testBaseline": test_baseline,
        "testTradesFiltered": len(filtered),
        "testTradesKept": len(kept),
        "testPnlFiltered": fs["pnl"],
        "testPnlKept": ks["pnl"],
        "testPfFiltered": fs["pf"],
        "testPfKept": ks["pf"],
        "testWrFiltered": fs["winRate"],
        "testWrKept": ks["winRate"],
    }


def apply_combo_filter(trades: list, f1: dict, f2: dict) -> tuple:
    """Apply two filters with OR logic."""
    filtered, kept = [], []
    for t in trades:
        f1_out, _ = apply_filter([t], f1)
        if f1_out:
            filtered.append(t)
            continue
        f2_out, _ = apply_filter([t], f2)
        if f2_out:
            filtered.append(t)
        else:
            kept.append(t)
    return filtered, kept


def test_combo_filters(train_trades: list, test_trades: list, single_filters: list) -> list:
    """Test combinations of 2 top filters."""
    top_filters = [f for f in single_filters if f is not None and f["pnlFiltered"] < -100][:8]
    if len(top_filters) < 2:
        top_filters = [f for f in single_filters if f is not None][:6]
    if len(top_filters) < 2:
        return []

    combo_results = []
    for f1, f2 in combinations(top_filters, 2):
        # Skip if same indicator
        if f1["indicator"] == f2["indicator"]:
            continue
        filtered, kept = apply_combo_filter(train_trades, f1, f2)
        if filtered and kept:
            fs = compute_stats(filtered)
            ks = compute_stats(kept)
            if fs["pf"] < ks["pf"]:
                t_filtered, t_kept = apply_combo_filter(test_trades, f1, f2)
                tfs = compute_stats(t_filtered) if t_filtered else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
                tks = compute_stats(t_kept) if t_kept else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
                combo_results.append({
                    "rule": f"({f1['rule']}) OR ({f2['rule']})",
                    "tradesFiltered": len(filtered),
                    "tradesKept": len(kept),
                    "pnlFiltered": fs["pnl"],
                    "pnlKept": ks["pnl"],
                    "pfFiltered": fs["pf"],
                    "pfKept": ks["pf"],
                    "wrFiltered": fs["winRate"],
                    "wrKept": ks["winRate"],
                    "testTradesFiltered": len(t_filtered),
                    "testTradesKept": len(t_kept),
                    "testPnlFiltered": tfs["pnl"],
                    "testPnlKept": tks["pnl"],
                    "testPfFiltered": tfs["pf"],
                    "testPfKept": tks["pf"],
                    "testWrFiltered": tfs["winRate"],
                    "testWrKept": tks["winRate"],
                })

    combo_results.sort(key=lambda x: x["pnlFiltered"])
    return combo_results[:10]


# ──────────────────────────────────────────────
# Walk-Forward Analysis per Combination
# ──────────────────────────────────────────────

def run_analysis_pass(joined_trades: list, indicator_info: dict, label: str) -> dict | None:
    """Run a single analysis pass (combined, long-only, or short-only).

    Returns dict with rankings, filters, combo filters, and split info.
    """
    if len(joined_trades) < MIN_TRADES:
        return None

    # Chronological split
    joined_trades.sort(key=lambda t: t["entryTime"])
    split_idx = int(len(joined_trades) * TRAIN_RATIO)
    if split_idx < 10 or (len(joined_trades) - split_idx) < 5:
        return None

    train = joined_trades[:split_idx]
    test = joined_trades[split_idx:]

    train_baseline = compute_stats(train)
    test_baseline = compute_stats(test)
    full_baseline = compute_stats(joined_trades)

    split_info = {
        "label": label,
        "totalTrades": len(joined_trades),
        "trainCount": len(train),
        "testCount": len(test),
        "trainStart": train[0]["entryDate"],
        "trainEnd": train[-1]["entryDate"],
        "testStart": test[0]["entryDate"],
        "testEnd": test[-1]["entryDate"],
        "trainBaseline": train_baseline,
        "testBaseline": test_baseline,
        "fullBaseline": full_baseline,
    }

    # Analyze all indicators on full data for rankings
    results = []
    for name, ind_type in indicator_info.items():
        r = analyze_indicator(joined_trades, name, ind_type)
        if r:
            results.append(r)
    results.sort(key=lambda x: x["pfSpread"], reverse=True)

    # Find optimal thresholds on TRAIN data, validate on TEST
    single_filters = []
    for name, ind_type in indicator_info.items():
        f = find_optimal_threshold(train, name, ind_type)
        if f:
            test_result = validate_filter(test, f)
            f.update(test_result)
            single_filters.append(f)
    single_filters.sort(key=lambda x: x["pnlFiltered"])

    # Combo filters
    combo_filters = test_combo_filters(train, test, single_filters)

    return {
        "splitInfo": split_info,
        "rankings": results,
        "singleFilters": single_filters,
        "comboFilters": combo_filters,
    }


# ──────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────

def fmt_pnl(val: float) -> str:
    return f"${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"


def _is_holds(f: dict) -> bool:
    """Check if a filter holds on out-of-sample data."""
    return (f.get("testPnlFiltered", 0) < 0
            and f.get("testPfKept", 0) > f.get("testPfFiltered", 0))


def _collect_holds_filters(all_results: dict) -> dict:
    """Collect all HOLDS filters organized by indicator.
    Returns {indicator: [{"rule", "strat", "tf", "direction", "trainPnl", "testPnl", ...}, ...]}
    """
    by_indicator = defaultdict(list)
    for key, result_set in all_results.items():
        strat, tf = key
        for pass_result in result_set:
            if pass_result is None:
                continue
            direction = pass_result["splitInfo"]["label"]  # "Combined", "Long Only", "Short Only"
            for f in pass_result["singleFilters"]:
                entry = {
                    "rule": f["rule"],
                    "indicator": f["indicator"],
                    "strat": strat,
                    "tf": tf,
                    "direction": direction,
                    "trainPnlFiltered": f["pnlFiltered"],
                    "trainPfKept": f["pfKept"],
                    "testPnlFiltered": f.get("testPnlFiltered", 0),
                    "testPfKept": f.get("testPfKept", 0),
                    "testPfFiltered": f.get("testPfFiltered", 0),
                    "holds": _is_holds(f),
                }
                by_indicator[f["indicator"]].append(entry)
    return by_indicator


def write_report(all_results: dict, strategy_trades: dict, cross_comparison: dict):
    """Write the full markdown report."""
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("# ES Levels Strategy — Technical Indicator Impact Analysis")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("**Method:** Auto-discovered indicators from ES bar snapshots (1M, 2M, 3M, 5M)")
    lines.append(f"**Validation:** Walk-forward ({TRAIN_RATIO:.0%} train / {1-TRAIN_RATIO:.0%} test, chronological)")
    lines.append("")

    # Collect all filter data upfront
    all_filters = _collect_holds_filters(all_results)

    # ── Executive Summary ──
    lines.append("## Executive Summary")
    lines.append("")

    # Total stats
    total_filters = sum(len(entries) for entries in all_filters.values())
    total_holds = sum(1 for entries in all_filters.values() for e in entries if e["holds"])
    total_overfit = total_filters - total_holds
    lines.append(f"**Total filters tested:** {total_filters} | "
                 f"**HOLDS:** {total_holds} ({total_holds*100//total_filters if total_filters else 0}%) | "
                 f"**OVERFIT:** {total_overfit} ({total_overfit*100//total_filters if total_filters else 0}%)")
    lines.append("")

    # ── Global Best: Top indicators by HOLDS count ──
    lines.append("## Global Best Indicators (All Strategies, All Directions)")
    lines.append("")
    lines.append("Ranked by how many (strategy × timeframe × direction) combinations each indicator's")
    lines.append("filter **HOLDS** on out-of-sample test data. Higher = more robust signal.")
    lines.append("")

    indicator_holds_count = {}
    for ind, entries in all_filters.items():
        holds_entries = [e for e in entries if e["holds"]]
        if holds_entries:
            total_test_pnl = sum(e["testPnlFiltered"] for e in holds_entries)
            total_train_pnl = sum(e["trainPnlFiltered"] for e in holds_entries)
            indicator_holds_count[ind] = {
                "count": len(holds_entries),
                "total": len(entries),
                "trainPnlSaved": -total_train_pnl,
                "testPnlSaved": -total_test_pnl,
                "contexts": holds_entries,
            }

    sorted_global = sorted(indicator_holds_count.items(),
                           key=lambda x: x[1]["count"], reverse=True)

    lines.append("| Rank | Indicator | HOLDS / Tested | Train P&L Saved | Test P&L Saved | Strategies | Timeframes |")
    lines.append("|-----:|-----------|:--------------:|-----------:|-----------:|------------|------------|")
    for rank, (ind, info) in enumerate(sorted_global[:25], 1):
        strats = sorted(set(e["strat"] for e in info["contexts"]))
        tfs = sorted(set(e["tf"] for e in info["contexts"]))
        lines.append(f"| {rank} | {ind} | {info['count']}/{info['total']} | "
                     f"{fmt_pnl(info['trainPnlSaved'])} | {fmt_pnl(info['testPnlSaved'])} | "
                     f"{'、'.join(s.replace('Levels ', '') for s in strats)} | "
                     f"{'、'.join(tfs)} |")
    lines.append("")

    # ── Global Long vs Short ──
    lines.append("## Global Long vs Short Indicator Breakdown")
    lines.append("")
    lines.append("Which indicators produce HOLDS filters for longs, shorts, or both across all strategies.")
    lines.append("")

    # Build per-indicator long/short/combined stats
    indicator_by_dir = {}
    for ind, entries in all_filters.items():
        long_holds = [e for e in entries if e["holds"] and e["direction"] == "Long Only"]
        short_holds = [e for e in entries if e["holds"] and e["direction"] == "Short Only"]
        combined_holds = [e for e in entries if e["holds"] and e["direction"] == "Combined"]
        any_holds = long_holds or short_holds or combined_holds
        if any_holds:
            indicator_by_dir[ind] = {
                "long": long_holds,
                "short": short_holds,
                "combined": combined_holds,
                "total_holds": len(long_holds) + len(short_holds) + len(combined_holds),
            }

    # Sort by total HOLDS
    sorted_by_dir = sorted(indicator_by_dir.items(),
                           key=lambda x: x[1]["total_holds"], reverse=True)

    lines.append("| Indicator | Long HOLDS | Short HOLDS | Combined HOLDS | Long Test P&L | Short Test P&L | Verdict |")
    lines.append("|-----------|:----------:|:-----------:|:--------------:|--------------:|---------------:|---------|")
    for ind, info in sorted_by_dir[:30]:
        long_test = sum(e["testPnlFiltered"] for e in info["long"])
        short_test = sum(e["testPnlFiltered"] for e in info["short"])
        nl, ns, nc = len(info["long"]), len(info["short"]), len(info["combined"])

        if nl > 0 and ns > 0:
            verdict = "Both L&S"
        elif nl > 0:
            verdict = "Long only"
        elif ns > 0:
            verdict = "Short only"
        else:
            verdict = "Combined only"

        lines.append(f"| {ind} | {nl} | {ns} | {nc} | "
                     f"{fmt_pnl(-long_test) if nl else '—'} | "
                     f"{fmt_pnl(-short_test) if ns else '—'} | {verdict} |")
    lines.append("")

    # ── Top Filters: Global Long ──
    lines.append("### Best Global Filters — Longs")
    lines.append("")
    lines.append("Top HOLDS filters discovered across all strategies for **long trades**, ranked by test P&L saved.")
    lines.append("")

    all_long_holds = []
    for ind, entries in all_filters.items():
        for e in entries:
            if e["holds"] and e["direction"] == "Long Only":
                all_long_holds.append(e)
    all_long_holds.sort(key=lambda e: e["testPnlFiltered"])

    if all_long_holds:
        lines.append("| Rule | Strategy | TF | Train P&L Saved | Test P&L Saved | Test PF→ |")
        lines.append("|------|----------|:--:|-----------:|-----------:|--------:|")
        for e in all_long_holds[:20]:
            lines.append(f"| {e['rule']} | {e['strat']} | {e['tf']} | "
                         f"{fmt_pnl(-e['trainPnlFiltered'])} | {fmt_pnl(-e['testPnlFiltered'])} | "
                         f"{e['testPfKept']:.2f} |")
        lines.append("")
    else:
        lines.append("No HOLDS filters found for longs.")
        lines.append("")

    # ── Top Filters: Global Short ──
    lines.append("### Best Global Filters — Shorts")
    lines.append("")
    lines.append("Top HOLDS filters discovered across all strategies for **short trades**, ranked by test P&L saved.")
    lines.append("")

    all_short_holds = []
    for ind, entries in all_filters.items():
        for e in entries:
            if e["holds"] and e["direction"] == "Short Only":
                all_short_holds.append(e)
    all_short_holds.sort(key=lambda e: e["testPnlFiltered"])

    if all_short_holds:
        lines.append("| Rule | Strategy | TF | Train P&L Saved | Test P&L Saved | Test PF→ |")
        lines.append("|------|----------|:--:|-----------:|-----------:|--------:|")
        for e in all_short_holds[:20]:
            lines.append(f"| {e['rule']} | {e['strat']} | {e['tf']} | "
                         f"{fmt_pnl(-e['trainPnlFiltered'])} | {fmt_pnl(-e['testPnlFiltered'])} | "
                         f"{e['testPfKept']:.2f} |")
        lines.append("")
    else:
        lines.append("No HOLDS filters found for shorts.")
        lines.append("")

    # ── Per-Strategy Sections ──
    for strat_name in ["Levels 1M", "Levels 2M", "Levels 3M", "Levels 5M"]:
        trades = strategy_trades.get(strat_name, [])
        if not trades:
            continue

        longs = sum(1 for t in trades if t["direction"] == "Long")
        shorts = sum(1 for t in trades if t["direction"] == "Short")
        total_stats = compute_stats(trades)

        lines.append(f"## {strat_name}")
        lines.append("")
        lines.append(f"**Trades:** {len(trades)} total ({longs} long, {shorts} short) | "
                     f"**WR:** {total_stats['winRate']}% | **PF:** {total_stats['pf']} | "
                     f"**P&L:** {fmt_pnl(total_stats['pnl'])}")
        lines.append("")

        # ── Per-Strategy Long vs Short Summary ──
        strat_long_holds = []
        strat_short_holds = []
        strat_combined_holds = []
        for ind, entries in all_filters.items():
            for e in entries:
                if e["strat"] != strat_name or not e["holds"]:
                    continue
                if e["direction"] == "Long Only":
                    strat_long_holds.append(e)
                elif e["direction"] == "Short Only":
                    strat_short_holds.append(e)
                else:
                    strat_combined_holds.append(e)

        # Aggregate by indicator for this strategy
        strat_ind_summary = defaultdict(lambda: {"long": [], "short": [], "combined": []})
        for e in strat_long_holds:
            strat_ind_summary[e["indicator"]]["long"].append(e)
        for e in strat_short_holds:
            strat_ind_summary[e["indicator"]]["short"].append(e)
        for e in strat_combined_holds:
            strat_ind_summary[e["indicator"]]["combined"].append(e)

        if strat_ind_summary:
            lines.append(f"### {strat_name} — Long vs Short Summary")
            lines.append("")
            lines.append("Indicators with HOLDS filters for this strategy across all timeframes.")
            lines.append("")
            lines.append("| Indicator | Long HOLDS (TFs) | Short HOLDS (TFs) | Combined HOLDS (TFs) | Best Long Rule | Best Short Rule |")
            lines.append("|-----------|:----------------:|:-----------------:|:--------------------:|----------------|-----------------|")

            # Sort by total holds for this strategy
            sorted_strat_inds = sorted(strat_ind_summary.items(),
                                       key=lambda x: len(x[1]["long"]) + len(x[1]["short"]) + len(x[1]["combined"]),
                                       reverse=True)
            for ind, dirs in sorted_strat_inds[:20]:
                long_tfs = sorted(set(e["tf"] for e in dirs["long"]))
                short_tfs = sorted(set(e["tf"] for e in dirs["short"]))
                comb_tfs = sorted(set(e["tf"] for e in dirs["combined"]))

                # Pick best rule per direction (most test P&L saved)
                best_long_rule = ""
                if dirs["long"]:
                    best_l = min(dirs["long"], key=lambda e: e["testPnlFiltered"])
                    best_long_rule = f"`{best_l['rule']}`"
                best_short_rule = ""
                if dirs["short"]:
                    best_s = min(dirs["short"], key=lambda e: e["testPnlFiltered"])
                    best_short_rule = f"`{best_s['rule']}`"

                lines.append(f"| {ind} | "
                             f"{len(dirs['long'])} ({','.join(long_tfs) if long_tfs else '—'}) | "
                             f"{len(dirs['short'])} ({','.join(short_tfs) if short_tfs else '—'}) | "
                             f"{len(dirs['combined'])} ({','.join(comb_tfs) if comb_tfs else '—'}) | "
                             f"{best_long_rule or '—'} | {best_short_rule or '—'} |")
            lines.append("")

        for tf in ["1M", "2M", "3M", "5M"]:
            key = (strat_name, tf)
            if key not in all_results:
                continue
            result_set = all_results[key]

            lines.append(f"### {strat_name} × ES {tf} Bars")
            lines.append("")

            for pass_result in result_set:
                if pass_result is None:
                    continue

                si = pass_result["splitInfo"]
                tb = si["trainBaseline"]
                tsb = si["testBaseline"]
                fb = si["fullBaseline"]

                lines.append(f"#### {si['label']} ({si['totalTrades']} trades)")
                lines.append("")

                # Walk-forward split
                lines.append("| Set | Period | Trades | Win Rate | PF | P&L |")
                lines.append("|-----|--------|------:|--------:|---:|----:|")
                lines.append(f"| **Train** | {si['trainStart']} to {si['trainEnd']} | "
                             f"{si['trainCount']} | {tb['winRate']}% | {tb['pf']:.2f} | {fmt_pnl(tb['pnl'])} |")
                lines.append(f"| **Test** | {si['testStart']} to {si['testEnd']} | "
                             f"{si['testCount']} | {tsb['winRate']}% | {tsb['pf']:.2f} | {fmt_pnl(tsb['pnl'])} |")
                lines.append(f"| **Full** | {si['trainStart']} to {si['testEnd']} | "
                             f"{si['totalTrades']} | {fb['winRate']}% | {fb['pf']:.2f} | {fmt_pnl(fb['pnl'])} |")
                lines.append("")

                # Top 15 indicators by PF spread
                rankings = pass_result["rankings"]
                if rankings:
                    lines.append("**Top 15 Indicators by PF Spread:**")
                    lines.append("")
                    lines.append("| Rank | Indicator | Type | PF Spread | Correlation | Best Q PF | Worst Q PF |")
                    lines.append("|-----:|-----------|------|----------:|------------:|----------:|-----------:|")
                    for i, r in enumerate(rankings[:15], 1):
                        corr_str = f"{r['correlation']:.4f}" if r["correlation"] is not None else "N/A"
                        pfs = [b["pf"] for b in r["buckets"] if b["pf"] < 9999]
                        best_pf = max(pfs) if pfs else 0
                        worst_pf = min(pfs) if pfs else 0
                        lines.append(f"| {i} | {r['name']} | {r['type']} | {r['pfSpread']:.2f} | "
                                     f"{corr_str} | {best_pf:.2f} | {worst_pf:.2f} |")
                    lines.append("")

                # Top 10 detail with quintile breakdowns
                for r in rankings[:10]:
                    lines.append(f"<details><summary><b>{r['name']}</b> (PF spread: {r['pfSpread']:.2f})</summary>")
                    lines.append("")
                    if r["type"] == "boolean":
                        lines.append("| Value | Trades | Win Rate | P&L | PF | Avg Trade |")
                        lines.append("|-------|------:|--------:|----:|---:|----------:|")
                    else:
                        lines.append("| Bucket | Range | Trades | Win Rate | P&L | PF | Avg Trade |")
                        lines.append("|--------|-------|------:|--------:|----:|---:|----------:|")
                    for b in r["buckets"]:
                        pf_str = f"{b['pf']:.2f}" if b["pf"] < 9999 else "Inf"
                        if r["type"] == "boolean":
                            lines.append(f"| {b['label']} | {b['count']} | {b['winRate']}% | "
                                         f"{fmt_pnl(b['pnl'])} | {pf_str} | {fmt_pnl(b['avgTrade'])} |")
                        else:
                            lines.append(f"| {b['label']} | {b['range'][0]:.2f} — {b['range'][1]:.2f} | "
                                         f"{b['count']} | {b['winRate']}% | "
                                         f"{fmt_pnl(b['pnl'])} | {pf_str} | {fmt_pnl(b['avgTrade'])} |")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

                # Single filter recommendations
                single_filters = pass_result["singleFilters"]
                if single_filters:
                    lines.append("**Recommended Filters (Walk-Forward Validated):**")
                    lines.append("")
                    lines.append("| Rule | Train Out | Train P&L | Train PF→ | Test Out | Test P&L | Test PF→ | Verdict |")
                    lines.append("|------|--------:|---------:|--------:|--------:|---------:|-------:|---------|")
                    for f in single_filters[:10]:
                        verdict = "**HOLDS**" if _is_holds(f) else "OVERFIT"
                        lines.append(f"| {f['rule']} | {f['tradesFiltered']} | {fmt_pnl(f['pnlFiltered'])} | "
                                     f"{f['pfKept']:.2f} | {f.get('testTradesFiltered', 0)} | "
                                     f"{fmt_pnl(f.get('testPnlFiltered', 0))} | "
                                     f"{f.get('testPfKept', 0):.2f} | {verdict} |")
                    lines.append("")

                # Combo filters
                combo_filters = pass_result["comboFilters"]
                if combo_filters:
                    lines.append("**Combination Filters:**")
                    lines.append("")
                    lines.append("| Rules | Train Out | Train P&L | Test Out | Test P&L | Test PF→ | Verdict |")
                    lines.append("|-------|--------:|---------:|--------:|---------:|-------:|---------|")
                    for cf in combo_filters[:5]:
                        verdict = "**HOLDS**" if _is_holds(cf) else "OVERFIT"
                        lines.append(f"| {cf['rule']} | {cf['tradesFiltered']} | {fmt_pnl(cf['pnlFiltered'])} | "
                                     f"{cf.get('testTradesFiltered', 0)} | "
                                     f"{fmt_pnl(cf.get('testPnlFiltered', 0))} | "
                                     f"{cf.get('testPfKept', 0):.2f} | {verdict} |")
                    lines.append("")

                lines.append("---")
                lines.append("")

    # ── Cross-Comparison Matrix ──
    lines.append("## Cross-Comparison Matrix")
    lines.append("")
    lines.append("Indicators appearing in top 10 by PF spread across strategy/timeframe/direction combinations.")
    lines.append("")

    if cross_comparison:
        lines.append("| Indicator | Appearances | Long | Short | Combined |")
        lines.append("|-----------|:----------:|:----:|:-----:|:--------:|")

        sorted_cross = sorted(cross_comparison.items(), key=lambda x: len(x[1]), reverse=True)
        for ind, contexts in sorted_cross[:30]:
            n_long = sum(1 for c in contexts if "Long" in c)
            n_short = sum(1 for c in contexts if "Short" in c)
            n_comb = sum(1 for c in contexts if "Combined" in c)
            lines.append(f"| {ind} | {len(contexts)} | {n_long} | {n_short} | {n_comb} |")
        lines.append("")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report written: {OUTPUT_MD}")


def write_html_report(all_results: dict, strategy_trades: dict, cross_comparison: dict):
    """Write an interactive HTML report with charts."""
    import json as _json

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    all_filters = _collect_holds_filters(all_results)

    # ── Prepare chart data ──

    # 1. Global top indicators (HOLDS count)
    indicator_holds = {}
    for ind, entries in all_filters.items():
        holds = [e for e in entries if e["holds"]]
        if holds:
            indicator_holds[ind] = {
                "total": len(entries),
                "holds": len(holds),
                "holdsPct": round(len(holds) / len(entries) * 100, 1),
                "trainSaved": round(-sum(e["trainPnlFiltered"] for e in holds), 2),
                "testSaved": round(-sum(e["testPnlFiltered"] for e in holds), 2),
            }
    sorted_global = sorted(indicator_holds.items(), key=lambda x: x[1]["holds"], reverse=True)[:25]
    global_chart_labels = [x[0] for x in sorted_global]
    global_chart_holds = [x[1]["holds"] for x in sorted_global]
    global_chart_total = [x[1]["total"] for x in sorted_global]
    global_chart_test_saved = [x[1]["testSaved"] for x in sorted_global]

    # 2. Long vs Short breakdown
    ls_data = {}
    for ind, entries in all_filters.items():
        long_h = [e for e in entries if e["holds"] and e["direction"] == "Long Only"]
        short_h = [e for e in entries if e["holds"] and e["direction"] == "Short Only"]
        comb_h = [e for e in entries if e["holds"] and e["direction"] == "Combined"]
        total_h = len(long_h) + len(short_h) + len(comb_h)
        if total_h >= 3:
            ls_data[ind] = {
                "long": len(long_h),
                "short": len(short_h),
                "combined": len(comb_h),
                "longTestSaved": round(-sum(e["testPnlFiltered"] for e in long_h), 2),
                "shortTestSaved": round(-sum(e["testPnlFiltered"] for e in short_h), 2),
            }
    sorted_ls = sorted(ls_data.items(), key=lambda x: x[1]["long"] + x[1]["short"] + x[1]["combined"], reverse=True)[:20]
    ls_labels = [x[0] for x in sorted_ls]
    ls_long = [x[1]["long"] for x in sorted_ls]
    ls_short = [x[1]["short"] for x in sorted_ls]
    ls_combined = [x[1]["combined"] for x in sorted_ls]

    # 3. Per-strategy data
    strat_sections = {}
    for strat_name in ["Levels 1M", "Levels 2M", "Levels 3M", "Levels 5M"]:
        trades = strategy_trades.get(strat_name, [])
        if not trades:
            continue
        # Collect this strategy's indicator L/S data
        strat_ind = defaultdict(lambda: {"long": 0, "short": 0, "combined": 0,
                                         "longTFs": [], "shortTFs": [],
                                         "bestLongRule": "", "bestShortRule": "",
                                         "longTestSaved": 0, "shortTestSaved": 0})
        for ind, entries in all_filters.items():
            for e in entries:
                if e["strat"] != strat_name or not e["holds"]:
                    continue
                d = e["direction"]
                si = strat_ind[ind]
                if d == "Long Only":
                    si["long"] += 1
                    si["longTFs"].append(e["tf"])
                    si["longTestSaved"] += -e["testPnlFiltered"]
                    if not si["bestLongRule"] or e["testPnlFiltered"] < si.get("_bestLongVal", 0):
                        si["bestLongRule"] = e["rule"]
                        si["_bestLongVal"] = e["testPnlFiltered"]
                elif d == "Short Only":
                    si["short"] += 1
                    si["shortTFs"].append(e["tf"])
                    si["shortTestSaved"] += -e["testPnlFiltered"]
                    if not si["bestShortRule"] or e["testPnlFiltered"] < si.get("_bestShortVal", 0):
                        si["bestShortRule"] = e["rule"]
                        si["_bestShortVal"] = e["testPnlFiltered"]
                else:
                    si["combined"] += 1

        # Clean up temp keys and sort
        sorted_strat = sorted(
            [(k, v) for k, v in strat_ind.items()
             if v["long"] + v["short"] + v["combined"] >= 2],
            key=lambda x: x[1]["long"] + x[1]["short"] + x[1]["combined"],
            reverse=True)[:20]

        strat_sections[strat_name] = {
            "totalTrades": len(trades),
            "longs": sum(1 for t in trades if t["direction"] == "Long"),
            "shorts": sum(1 for t in trades if t["direction"] == "Short"),
            "stats": compute_stats(trades),
            "indicators": [(k, {
                "long": v["long"], "short": v["short"], "combined": v["combined"],
                "longTFs": sorted(set(v["longTFs"])), "shortTFs": sorted(set(v["shortTFs"])),
                "bestLongRule": v["bestLongRule"], "bestShortRule": v["bestShortRule"],
                "longTestSaved": round(v["longTestSaved"], 2),
                "shortTestSaved": round(v["shortTestSaved"], 2),
            }) for k, v in sorted_strat],
        }

    # 4. Best global filters for longs and shorts
    best_long_filters = []
    best_short_filters = []
    for ind, entries in all_filters.items():
        for e in entries:
            if not e["holds"]:
                continue
            entry = {
                "rule": e["rule"], "indicator": e["indicator"],
                "strat": e["strat"], "tf": e["tf"],
                "trainSaved": round(-e["trainPnlFiltered"], 2),
                "testSaved": round(-e["testPnlFiltered"], 2),
                "testPfKept": e["testPfKept"],
            }
            if e["direction"] == "Long Only":
                best_long_filters.append(entry)
            elif e["direction"] == "Short Only":
                best_short_filters.append(entry)
    best_long_filters.sort(key=lambda x: x["testSaved"], reverse=True)
    best_short_filters.sort(key=lambda x: x["testSaved"], reverse=True)

    # 5. Per-combination top indicators and filters for detail tabs
    combo_details = {}
    for key, result_set in sorted(all_results.items()):
        strat, tf = key
        passes = []
        for pr in result_set:
            if pr is None:
                passes.append(None)
                continue
            si = pr["splitInfo"]
            # Top 15 rankings
            rankings = []
            for r in pr["rankings"][:15]:
                pfs = [b["pf"] for b in r["buckets"] if b["pf"] < 9999]
                rankings.append({
                    "name": r["name"], "type": r["type"],
                    "pfSpread": r["pfSpread"],
                    "corr": r["correlation"],
                    "bestPF": max(pfs) if pfs else 0,
                    "worstPF": min(pfs) if pfs else 0,
                    "buckets": [{
                        "label": b["label"],
                        "range": b["range"],
                        "count": b["count"],
                        "winRate": b["winRate"],
                        "pnl": b["pnl"],
                        "pf": b["pf"] if b["pf"] < 9999 else None,
                        "avgTrade": b["avgTrade"],
                    } for b in r["buckets"]],
                })
            # Top 10 filters
            filters = []
            for f in pr["singleFilters"][:10]:
                filters.append({
                    "rule": f["rule"],
                    "trainOut": f["tradesFiltered"],
                    "trainPnl": f["pnlFiltered"],
                    "trainPfKept": f["pfKept"],
                    "testOut": f.get("testTradesFiltered", 0),
                    "testPnl": f.get("testPnlFiltered", 0),
                    "testPfKept": f.get("testPfKept", 0),
                    "holds": _is_holds(f),
                })
            passes.append({
                "label": si["label"],
                "totalTrades": si["totalTrades"],
                "trainCount": si["trainCount"],
                "testCount": si["testCount"],
                "trainPeriod": f"{si['trainStart']} to {si['trainEnd']}",
                "testPeriod": f"{si['testStart']} to {si['testEnd']}",
                "trainBaseline": si["trainBaseline"],
                "testBaseline": si["testBaseline"],
                "fullBaseline": si["fullBaseline"],
                "rankings": rankings,
                "filters": filters,
            })
        combo_details[f"{strat}|{tf}"] = passes

    # 6. Verdict totals
    total_filters = sum(len(entries) for entries in all_filters.values())
    total_holds = sum(1 for entries in all_filters.values() for e in entries if e["holds"])

    # ── Build JSON data blob ──
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    report_data = sanitize({
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "totalFilters": total_filters,
        "totalHolds": total_holds,
        "globalChart": {
            "labels": global_chart_labels,
            "holds": global_chart_holds,
            "total": global_chart_total,
            "testSaved": global_chart_test_saved,
        },
        "lsChart": {
            "labels": ls_labels,
            "long": ls_long,
            "short": ls_short,
            "combined": ls_combined,
        },
        "lsDetail": {k: v for k, v in sorted_ls},
        "stratSections": strat_sections,
        "bestLongFilters": best_long_filters[:20],
        "bestShortFilters": best_short_filters[:20],
        "comboDetails": combo_details,
    })

    data_json = _json.dumps(report_data, separators=(",", ":"))

    # Build HTML using string concatenation — avoids f-string escaping issues with JS
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ES Levels — Indicator Impact Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a; --text: #e0e0e0;
         --dim: #888; --green: #22c55e; --red: #ef4444; --blue: #3b82f6;
         --amber: #f59e0b; --purple: #a855f7; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text);
        line-height: 1.5; padding: 0; }
.header { background: linear-gradient(135deg, #1e293b, #0f172a); padding: 24px 32px;
           border-bottom: 1px solid var(--border); }
.header h1 { font-size: 1.5rem; font-weight: 600; }
.header .meta { color: var(--dim); font-size: 0.85rem; margin-top: 4px; }
.kpi-row { display: flex; gap: 16px; padding: 16px 32px; flex-wrap: wrap; }
.kpi { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
        padding: 16px 20px; min-width: 160px; flex: 1; }
.kpi .label { color: var(--dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
.kpi .value.green { color: var(--green); }
.kpi .value.red { color: var(--red); }
.kpi .value.blue { color: var(--blue); }
.tabs { display: flex; gap: 0; padding: 0 32px; border-bottom: 1px solid var(--border);
         background: var(--card); overflow-x: auto; }
.tab { padding: 12px 20px; cursor: pointer; color: var(--dim); border-bottom: 2px solid transparent;
        white-space: nowrap; font-size: 0.9rem; transition: all 0.15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--blue); border-bottom-color: var(--blue); }
.content { padding: 24px 32px; }
.section { display: none; }
.section.active { display: block; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
         padding: 20px; margin-bottom: 20px; }
.card h3 { font-size: 1.1rem; margin-bottom: 12px; }
.card h4 { font-size: 0.95rem; color: var(--dim); margin: 16px 0 8px; }
.chart-container { position: relative; height: 450px; margin: 8px 0; }
.chart-container.tall { height: 600px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border);
     color: var(--dim); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }
th.r { text-align: right; }
td { padding: 7px 10px; border-bottom: 1px solid var(--border); }
td.r { text-align: right; font-variant-numeric: tabular-nums; }
td.mono { font-family: 'Consolas', monospace; font-size: 0.82rem; }
tr:hover { background: rgba(59,130,246,0.06); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.badge.holds { background: rgba(34,197,94,0.15); color: var(--green); }
.badge.overfit { background: rgba(239,68,68,0.15); color: var(--red); }
.badge.both { background: rgba(168,85,247,0.15); color: var(--purple); }
.badge.long { background: rgba(34,197,94,0.15); color: var(--green); }
.badge.short { background: rgba(239,68,68,0.15); color: var(--red); }
.pnl-pos { color: var(--green); }
.pnl-neg { color: var(--red); }
.sub-tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.sub-tab { padding: 6px 14px; border-radius: 6px; cursor: pointer; background: var(--bg);
            border: 1px solid var(--border); font-size: 0.82rem; color: var(--dim); transition: all 0.15s; }
.sub-tab:hover { border-color: var(--blue); color: var(--text); }
.sub-tab.active { background: var(--blue); color: white; border-color: var(--blue); }
.sub-section { display: none; }
.sub-section.active { display: block; }
details { margin: 4px 0; }
details summary { cursor: pointer; padding: 6px 0; color: var(--blue); font-size: 0.85rem; }
details summary:hover { text-decoration: underline; }
@media (max-width: 768px) {
  .content, .header, .kpi-row { padding-left: 16px; padding-right: 16px; }
  .tabs { padding: 0 16px; }
}
</style>
</head>
<body>
<div class="header">
  <h1>ES Levels Strategy — Technical Indicator Impact Analysis</h1>
  <div class="meta" id="meta"></div>
</div>
<div class="kpi-row" id="kpis"></div>
<div class="tabs" id="tabs"></div>
<div class="content" id="content"></div>

<script>
const D = """)
    html_parts.append(data_json)
    html_parts.append(""";
const fmtPnl = v => v >= 0 ? '$' + v.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})
  : '-$' + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
const fmtPnlHtml = v => '<span class="'+(v>=0?'pnl-pos':'pnl-neg')+'">'+fmtPnl(v)+'</span>';
const badge = (holds) => holds ? '<span class="badge holds">HOLDS</span>' : '<span class="badge overfit">OVERFIT</span>';

// Meta
document.getElementById('meta').innerHTML = 'Generated: '+D.generated+' &middot; Walk-forward 70/30 chronological &middot; ~296 auto-discovered indicators &times; 4 bar timeframes';

// KPIs
const kpis = document.getElementById('kpis');
kpis.innerHTML = '<div class="kpi"><div class="label">Total Filters Tested</div><div class="value blue">'+D.totalFilters.toLocaleString()+'</div></div>'
  +'<div class="kpi"><div class="label">Walk-Forward HOLDS</div><div class="value green">'+D.totalHolds.toLocaleString()+' ('+Math.round(D.totalHolds/D.totalFilters*100)+'%)</div></div>'
  +'<div class="kpi"><div class="label">OVERFIT</div><div class="value red">'+(D.totalFilters-D.totalHolds).toLocaleString()+' ('+Math.round((D.totalFilters-D.totalHolds)/D.totalFilters*100)+'%)</div></div>'
  +'<div class="kpi"><div class="label">Strategies</div><div class="value">4</div></div>'
  +'<div class="kpi"><div class="label">Bar Timeframes</div><div class="value">1M, 2M, 3M, 5M</div></div>';

// Tabs
const tabDefs = [
  {id:'global', label:'Global Best'},
  {id:'longshort', label:'Long vs Short'},
  {id:'longs', label:'Best Longs'},
  {id:'shorts', label:'Best Shorts'},
];
const stratNames = Object.keys(D.stratSections);
stratNames.forEach(s => tabDefs.push({id:'strat_'+s.replace(/\\s/g,'_'), label:s}));
tabDefs.push({id:'detail', label:'All Combinations'});

const tabsEl = document.getElementById('tabs');
const contentEl = document.getElementById('content');
tabDefs.forEach((t,i) => {
  const el = document.createElement('div');
  el.className = 'tab' + (i===0?' active':'');
  el.textContent = t.label;
  el.onclick = () => switchTab(t.id);
  el.dataset.id = t.id;
  tabsEl.appendChild(el);
  const sec = document.createElement('div');
  sec.className = 'section' + (i===0?' active':'');
  sec.id = 'sec_'+t.id;
  contentEl.appendChild(sec);
});

function switchTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.id===id));
  document.querySelectorAll('.section').forEach(s => s.classList.toggle('active', s.id==='sec_'+id));
}

// ── Global Best tab ──
(function() {
  const sec = document.getElementById('sec_global');
  sec.innerHTML = '<div class="card">'
    +'<h3>Top 25 Indicators by HOLDS Count (All Strategies, All Directions)</h3>'
    +'<p style="color:var(--dim);font-size:0.85rem;margin-bottom:12px">How many (strategy &times; timeframe &times; direction) combinations each indicator filter holds on out-of-sample test data.</p>'
    +'<div class="chart-container tall"><canvas id="globalChart"></canvas></div>'
    +'</div>'
    +'<div class="card">'
    +'<h3>Test P&amp;L Saved by Top Indicators</h3>'
    +'<p style="color:var(--dim);font-size:0.85rem;margin-bottom:12px">Sum of out-of-sample P&amp;L saved across all HOLDS combinations per indicator.</p>'
    +'<div class="chart-container tall"><canvas id="globalSavedChart"></canvas></div>'
    +'</div>';

  new Chart(document.getElementById('globalChart'), {
    type: 'bar',
    data: {
      labels: D.globalChart.labels,
      datasets: [
        { label: 'HOLDS', data: D.globalChart.holds, backgroundColor: 'rgba(34,197,94,0.7)' },
        { label: 'OVERFIT', data: D.globalChart.total.map((t,i) => t - D.globalChart.holds[i]), backgroundColor: 'rgba(239,68,68,0.3)' },
      ]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#888' } } },
      scales: {
        x: { stacked: true, ticks: { color: '#888' }, grid: { color: '#2a2d3a' } },
        y: { stacked: true, ticks: { color: '#e0e0e0', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });

  new Chart(document.getElementById('globalSavedChart'), {
    type: 'bar',
    data: {
      labels: D.globalChart.labels,
      datasets: [{ label: 'Test P&L Saved', data: D.globalChart.testSaved,
        backgroundColor: D.globalChart.testSaved.map(v => v > 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)') }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#888', callback: v => '$'+v.toLocaleString() }, grid: { color: '#2a2d3a' } },
        y: { ticks: { color: '#e0e0e0', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
})();

// ── Long vs Short tab ──
(function() {
  const sec = document.getElementById('sec_longshort');
  sec.innerHTML = '<div class="card">'
    +'<h3>HOLDS Count: Long vs Short vs Combined</h3>'
    +'<p style="color:var(--dim);font-size:0.85rem;margin-bottom:12px">How many times each indicator filter HOLDS, broken down by trade direction.</p>'
    +'<div class="chart-container tall"><canvas id="lsChart"></canvas></div>'
    +'</div>'
    +'<div class="card"><h3>Long vs Short Detail</h3>'
    +'<table><thead><tr><th>Indicator</th><th class="r">Long HOLDS</th><th class="r">Short HOLDS</th><th class="r">Combined</th><th class="r">Long Test Saved</th><th class="r">Short Test Saved</th><th>Type</th></tr></thead>'
    +'<tbody id="lsTable"></tbody></table></div>';

  new Chart(document.getElementById('lsChart'), {
    type: 'bar',
    data: {
      labels: D.lsChart.labels,
      datasets: [
        { label: 'Long', data: D.lsChart.long, backgroundColor: 'rgba(34,197,94,0.7)' },
        { label: 'Short', data: D.lsChart.short, backgroundColor: 'rgba(239,68,68,0.7)' },
        { label: 'Combined', data: D.lsChart.combined, backgroundColor: 'rgba(59,130,246,0.5)' },
      ]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#888' } } },
      scales: {
        x: { stacked: true, ticks: { color: '#888' }, grid: { color: '#2a2d3a' } },
        y: { stacked: true, ticks: { color: '#e0e0e0', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });

  const tbody = document.getElementById('lsTable');
  const lsEntries = Object.entries(D.lsDetail).sort((a,b) =>
    (b[1].long+b[1].short+b[1].combined) - (a[1].long+a[1].short+a[1].combined));
  lsEntries.forEach(([ind, v]) => {
    const type = v.long > 0 && v.short > 0 ? '<span class="badge both">Both L&amp;S</span>'
      : v.long > 0 ? '<span class="badge long">Long</span>'
      : v.short > 0 ? '<span class="badge short">Short</span>'
      : '<span class="badge">Combined</span>';
    tbody.innerHTML += '<tr><td>'+ind+'</td><td class="r">'+v.long+'</td><td class="r">'+v.short+'</td>'
      +'<td class="r">'+v.combined+'</td><td class="r">'+fmtPnlHtml(v.longTestSaved)+'</td>'
      +'<td class="r">'+fmtPnlHtml(v.shortTestSaved)+'</td><td>'+type+'</td></tr>';
  });
})();

// ── Best Longs / Shorts tabs ──
function renderFilterTable(secId, filters, dirLabel) {
  const sec = document.getElementById(secId);
  let html = '<div class="card"><h3>Top HOLDS Filters &mdash; '+dirLabel+' Trades</h3>'
    +'<p style="color:var(--dim);font-size:0.85rem;margin-bottom:12px">Walk-forward validated filters ranked by test P&amp;L saved.</p>'
    +'<table><thead><tr><th>Rule</th><th>Strategy</th><th>TF</th><th class="r">Train Saved</th><th class="r">Test Saved</th><th class="r">Test PF&rarr;</th></tr></thead><tbody>';
  filters.forEach(f => {
    html += '<tr><td class="mono">'+f.rule+'</td><td>'+f.strat+'</td><td>'+f.tf+'</td>'
      +'<td class="r">'+fmtPnlHtml(f.trainSaved)+'</td><td class="r">'+fmtPnlHtml(f.testSaved)+'</td>'
      +'<td class="r">'+f.testPfKept.toFixed(2)+'</td></tr>';
  });
  html += '</tbody></table></div>';
  sec.innerHTML = html;
}
renderFilterTable('sec_longs', D.bestLongFilters, 'Long');
renderFilterTable('sec_shorts', D.bestShortFilters, 'Short');

// ── Per-Strategy tabs ──
stratNames.forEach(strat => {
  const sid = 'sec_strat_' + strat.replace(/\\s/g, '_');
  const sec = document.getElementById(sid);
  const s = D.stratSections[strat];
  const st = s.stats;
  let html = '<div class="kpi-row" style="padding:0;margin-bottom:16px;">'
    +'<div class="kpi"><div class="label">Trades</div><div class="value">'+s.totalTrades+' ('+s.longs+'L / '+s.shorts+'S)</div></div>'
    +'<div class="kpi"><div class="label">Win Rate</div><div class="value">'+st.winRate+'%</div></div>'
    +'<div class="kpi"><div class="label">Profit Factor</div><div class="value">'+st.pf+'</div></div>'
    +'<div class="kpi"><div class="label">P&amp;L</div><div class="value '+(st.pnl>=0?'green':'red')+'">'+fmtPnl(st.pnl)+'</div></div>'
    +'</div>';

  const inds = s.indicators;
  if (inds.length > 0) {
    const chartId = 'strat_chart_'+strat.replace(/\\s/g,'_');
    html += '<div class="card"><h3>Long vs Short HOLDS &mdash; '+strat+'</h3>'
      +'<div class="chart-container" style="height:'+Math.max(300, inds.length*28)+'px"><canvas id="'+chartId+'"></canvas></div></div>';

    html += '<div class="card"><h3>Indicator Detail &mdash; '+strat+'</h3>'
      +'<table><thead><tr><th>Indicator</th><th class="r">Long HOLDS</th><th>Long TFs</th><th class="r">Short HOLDS</th><th>Short TFs</th><th class="r">Combined</th><th>Best Long Rule</th><th>Best Short Rule</th></tr></thead><tbody>';
    inds.forEach(([ind, v]) => {
      html += '<tr><td>'+ind+'</td><td class="r">'+v.long+'</td><td>'+(v.longTFs.join(',')||'\\u2014')+'</td>'
        +'<td class="r">'+v.short+'</td><td>'+(v.shortTFs.join(',')||'\\u2014')+'</td>'
        +'<td class="r">'+v.combined+'</td>'
        +'<td class="mono" style="font-size:0.78rem">'+(v.bestLongRule||'\\u2014')+'</td>'
        +'<td class="mono" style="font-size:0.78rem">'+(v.bestShortRule||'\\u2014')+'</td></tr>';
    });
    html += '</tbody></table></div>';
  }
  sec.innerHTML = html;

  if (inds.length > 0) {
    setTimeout(() => {
      const canvas = document.getElementById('strat_chart_'+strat.replace(/\\s/g,'_'));
      if (!canvas) return;
      new Chart(canvas, {
        type: 'bar',
        data: {
          labels: inds.map(x => x[0]),
          datasets: [
            { label: 'Long', data: inds.map(x => x[1].long), backgroundColor: 'rgba(34,197,94,0.7)' },
            { label: 'Short', data: inds.map(x => x[1].short), backgroundColor: 'rgba(239,68,68,0.7)' },
            { label: 'Combined', data: inds.map(x => x[1].combined), backgroundColor: 'rgba(59,130,246,0.5)' },
          ]
        },
        options: {
          indexAxis: 'y', responsive: true, maintainAspectRatio: false,
          plugins: { legend: { labels: { color: '#888' } } },
          scales: {
            x: { stacked: true, ticks: { color: '#888' }, grid: { color: '#2a2d3a' } },
            y: { stacked: true, ticks: { color: '#e0e0e0', font: { size: 10 } }, grid: { display: false } }
          }
        }
      });
    }, 50);
  }
});

// ── All Combinations detail tab ──
(function() {
  const sec = document.getElementById('sec_detail');
  const combos = Object.keys(D.comboDetails).sort();
  let subTabHtml = '<div class="sub-tabs" id="combo_tabs">';
  combos.forEach((k, i) => {
    const parts = k.split('|');
    subTabHtml += '<div class="sub-tab'+(i===0?' active':'')+'" data-combo="'+k+'" onclick="switchCombo(&apos;'+k+'&apos;)">'+parts[0]+' &times; '+parts[1]+'</div>';
  });
  subTabHtml += '</div>';

  let detailHtml = '';
  combos.forEach((k, i) => {
    const passes = D.comboDetails[k];
    detailHtml += '<div class="sub-section'+(i===0?' active':'')+'" id="combo_'+k.replace('|','_')+'">';
    passes.forEach(p => {
      if (!p) return;
      const tb = p.trainBaseline, tsb = p.testBaseline, fb = p.fullBaseline;
      detailHtml += '<div class="card"><h3>'+p.label+' ('+p.totalTrades+' trades)</h3>'
        +'<table style="margin-bottom:12px"><thead><tr><th>Set</th><th>Period</th><th class="r">Trades</th><th class="r">WR</th><th class="r">PF</th><th class="r">P&amp;L</th></tr></thead><tbody>'
        +'<tr><td><b>Train</b></td><td>'+p.trainPeriod+'</td><td class="r">'+p.trainCount+'</td><td class="r">'+tb.winRate+'%</td><td class="r">'+tb.pf.toFixed(2)+'</td><td class="r">'+fmtPnlHtml(tb.pnl)+'</td></tr>'
        +'<tr><td><b>Test</b></td><td>'+p.testPeriod+'</td><td class="r">'+p.testCount+'</td><td class="r">'+tsb.winRate+'%</td><td class="r">'+tsb.pf.toFixed(2)+'</td><td class="r">'+fmtPnlHtml(tsb.pnl)+'</td></tr>'
        +'<tr><td><b>Full</b></td><td></td><td class="r">'+p.totalTrades+'</td><td class="r">'+fb.winRate+'%</td><td class="r">'+fb.pf.toFixed(2)+'</td><td class="r">'+fmtPnlHtml(fb.pnl)+'</td></tr>'
        +'</tbody></table>';

      if (p.rankings.length) {
        detailHtml += '<h4>Top 15 Indicators by PF Spread</h4><table><thead><tr><th>#</th><th>Indicator</th><th>Type</th><th class="r">PF Spread</th><th class="r">Corr</th><th class="r">Best PF</th><th class="r">Worst PF</th></tr></thead><tbody>';
        p.rankings.forEach((r,ri) => {
          detailHtml += '<tr><td>'+(ri+1)+'</td><td>'+r.name+'</td><td>'+r.type+'</td><td class="r">'+r.pfSpread.toFixed(2)+'</td><td class="r">'+(r.corr!==null?r.corr.toFixed(4):'N/A')+'</td><td class="r">'+r.bestPF.toFixed(2)+'</td><td class="r">'+r.worstPF.toFixed(2)+'</td></tr>';
        });
        detailHtml += '</tbody></table>';
      }

      p.rankings.slice(0, 5).forEach(r => {
        detailHtml += '<details><summary>'+r.name+' \\u2014 quintile breakdown</summary><table><thead><tr>';
        if (r.type === 'boolean') detailHtml += '<th>Value</th>';
        else detailHtml += '<th>Bucket</th><th>Range</th>';
        detailHtml += '<th class="r">Trades</th><th class="r">WR</th><th class="r">PF</th><th class="r">P&amp;L</th><th class="r">Avg</th></tr></thead><tbody>';
        r.buckets.forEach(b => {
          const pfStr = b.pf !== null ? b.pf.toFixed(2) : 'Inf';
          if (r.type === 'boolean') {
            detailHtml += '<tr><td>'+b.label+'</td>';
          } else {
            const r0 = typeof b.range[0]==='number'?b.range[0].toFixed(2):b.range[0];
            const r1 = typeof b.range[1]==='number'?b.range[1].toFixed(2):b.range[1];
            detailHtml += '<tr><td>'+b.label+'</td><td>'+r0+' \\u2014 '+r1+'</td>';
          }
          detailHtml += '<td class="r">'+b.count+'</td><td class="r">'+b.winRate+'%</td><td class="r">'+pfStr+'</td><td class="r">'+fmtPnlHtml(b.pnl)+'</td><td class="r">'+fmtPnlHtml(b.avgTrade)+'</td></tr>';
        });
        detailHtml += '</tbody></table></details>';
      });

      if (p.filters.length) {
        detailHtml += '<h4>Recommended Filters</h4><table><thead><tr><th>Rule</th><th class="r">Train Out</th><th class="r">Train P&amp;L</th><th class="r">Train PF&rarr;</th><th class="r">Test Out</th><th class="r">Test P&amp;L</th><th class="r">Test PF&rarr;</th><th>Verdict</th></tr></thead><tbody>';
        p.filters.forEach(f => {
          detailHtml += '<tr><td class="mono">'+f.rule+'</td><td class="r">'+f.trainOut+'</td><td class="r">'+fmtPnlHtml(f.trainPnl)+'</td><td class="r">'+f.trainPfKept.toFixed(2)+'</td><td class="r">'+f.testOut+'</td><td class="r">'+fmtPnlHtml(f.testPnl)+'</td><td class="r">'+f.testPfKept.toFixed(2)+'</td><td>'+badge(f.holds)+'</td></tr>';
        });
        detailHtml += '</tbody></table>';
      }
      detailHtml += '</div>';
    });
    detailHtml += '</div>';
  });

  sec.innerHTML = subTabHtml + detailHtml;
})();

function switchCombo(k) {
  document.querySelectorAll('#combo_tabs .sub-tab').forEach(t => t.classList.toggle('active', t.dataset.combo===k));
  document.querySelectorAll('.sub-section').forEach(s => s.classList.toggle('active', s.id==='combo_'+k.replace('|','_')));
}
</script>
</body>
</html>""")
    html = "".join(html_parts)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report: {OUTPUT_HTML}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ES Levels Strategy — Indicator Impact Analysis")
    print("=" * 60)
    print()

    # Step 1: Load trades
    strategy_trades = load_trades()
    print()

    # Step 2: Load all bar timeframes
    bar_data = {}
    for tf in BAR_FILES:
        indicators, indicator_info = load_bar_data(tf)
        bar_data[tf] = (indicators, indicator_info)
        print()

    # Step 3: Run analysis for each (strategy, timeframe) combination
    all_results = {}  # (strategy, timeframe) -> [combined_result, long_result, short_result]
    cross_comparison = defaultdict(list)  # indicator -> list of context strings

    total_combos = 0
    skipped_combos = 0

    for strat_name, trades in sorted(strategy_trades.items()):
        for tf in BAR_FILES:
            indicators, indicator_info = bar_data[tf]

            print(f"  Analyzing: {strat_name} × {tf} bars...")

            # Join trades with bar data
            joined = join_trades_with_bars(trades, indicators, tf)

            if len(joined) < MIN_TRADES:
                print(f"    SKIP: only {len(joined)} matched trades (< {MIN_TRADES})")
                skipped_combos += 1
                continue

            # Verify no data leakage
            leaks = 0
            for t in joined:
                bar_time = datetime.strptime(t["_bar_time"], "%Y-%m-%d %H:%M:%S")
                if bar_time >= t["entryTime"]:
                    leaks += 1
            if leaks > 0:
                print(f"    WARNING: {leaks} potential data leaks detected!")

            result_set = []

            # Pass 1: Combined (all trades)
            print(f"    Combined ({len(joined)} trades)...")
            combined = run_analysis_pass(joined, indicator_info, "Combined")
            result_set.append(combined)
            if combined:
                for r in combined["rankings"][:10]:
                    cross_comparison[r["name"]].append(f"{strat_name}/{tf}/Combined")

            # Pass 2: Longs only
            long_trades = [t for t in joined if t["direction"] == "Long"]
            if len(long_trades) >= MIN_TRADES:
                print(f"    Longs ({len(long_trades)} trades)...")
                long_result = run_analysis_pass(long_trades, indicator_info, "Long Only")
                result_set.append(long_result)
                if long_result:
                    for r in long_result["rankings"][:10]:
                        cross_comparison[r["name"]].append(f"{strat_name}/{tf}/Long")
            else:
                result_set.append(None)
                print(f"    Longs: SKIP ({len(long_trades)} < {MIN_TRADES})")

            # Pass 3: Shorts only
            short_trades = [t for t in joined if t["direction"] == "Short"]
            if len(short_trades) >= MIN_TRADES:
                print(f"    Shorts ({len(short_trades)} trades)...")
                short_result = run_analysis_pass(short_trades, indicator_info, "Short Only")
                result_set.append(short_result)
                if short_result:
                    for r in short_result["rankings"][:10]:
                        cross_comparison[r["name"]].append(f"{strat_name}/{tf}/Short")
            else:
                result_set.append(None)
                print(f"    Shorts: SKIP ({len(short_trades)} < {MIN_TRADES})")

            all_results[(strat_name, tf)] = result_set
            total_combos += 1
            print()

    print(f"\n  Completed {total_combos} combinations, skipped {skipped_combos}")

    # Step 4: Write reports
    print("\n  Writing reports...")
    write_report(all_results, strategy_trades, cross_comparison)
    write_html_report(all_results, strategy_trades, cross_comparison)

    print("\nDone!")


if __name__ == "__main__":
    main()
