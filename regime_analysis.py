"""
Market Regime Analysis — No-Trade Signal Finder
Joins trade data with ES 1-minute indicator snapshots to find market conditions
where trades consistently lose. Uses ES as a broad market regime proxy for ALL instruments.

Walk-forward validation: thresholds are fit on the first 70% of trades (chronologically)
and validated on the remaining 30% to detect overfitting.
"""

import csv
import json
import math
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from itertools import combinations

ES_CSV = r"C:\temp\zonebot_snapshot_STATS-LONG-ONL_Snapshot_1M_ES 03-26.csv"
TRADES_JS = Path(__file__).parent / "data" / "trades.js"
OUTPUT_JS = Path(__file__).parent / "data" / "regime.js"
OUTPUT_MD = Path(__file__).parent / "reports" / "regime_analysis.md"

# Indicators to analyze as regime signals (market-level, not trade-level)
REGIME_INDICATORS = {
    # Volatility
    "ATR_14_Bar": "numeric",
    "ATR_1H": "numeric",
    "ATR_PERCENTILE": "numeric",
    "VOLATILITY_EXPANDING": "boolean",
    "BollingerBandWidthSlope": "numeric",
    "RECENT_RANGE_SIZE": "numeric",
    # Trend
    "ADX": "numeric",
    "Chop": "numeric",
    "TREND_ALIGNMENT": "numeric",
    "MULTI_TIMEFRAME_ALIGNMENT": "numeric",
    "HMA8_Slope": "numeric",
    "HMA21_Slope": "numeric",
    # Momentum
    "LRSI": "numeric",
    "LRSI_HTF": "numeric",
    "KST_Bullish": "boolean",
    "KST_Bearish": "boolean",
    # Structure
    "EMA3_ABOVE_8": "boolean",
    "EMA8_ABOVE_21": "boolean",
    "EMA21_ABOVE_55": "boolean",
    "HTF_EMA8_ABOVE_21": "boolean",
    "HTF_STRUCTURE_ALIGNED": "boolean",
    "HTF_MOMENTUM_ALIGNED": "boolean",
    # Volume
    "VOLUME_RELATIVE_TO_AVERAGE": "numeric",
    # Session
    "IS_RTH": "boolean",
    "IS_FIRST_HOUR": "boolean",
    "IS_POWER_HOUR": "boolean",
    "MINUTES_INTO_SESSION": "numeric",
    # VWAP
    "DIST_FROM_VWAP_BANDS": "numeric",
}

NUM_BUCKETS = 5
TRAIN_RATIO = 0.70  # 70% train, 30% test (chronological split)


def load_es_indicators(csv_path: str) -> dict:
    """Load ES 1-min indicator CSV into a dict keyed by datetime string."""
    print("  Loading ES indicator data...")
    indicators = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]

        # Find column indices for our regime indicators
        col_indices = {}
        for name in REGIME_INDICATORS:
            if name in header:
                col_indices[name] = header.index(name)
            else:
                print(f"    WARNING: indicator '{name}' not found in CSV")

        row_count = 0
        for row in reader:
            dt_str = row[0].strip()
            snapshot = {}
            for name, idx in col_indices.items():
                raw = row[idx].strip() if idx < len(row) else ""
                ind_type = REGIME_INDICATORS[name]
                if ind_type == "boolean":
                    snapshot[name] = raw.lower() == "true"
                else:
                    try:
                        val = float(raw)
                        # Skip infinity and extreme values
                        if math.isinf(val) or abs(val) > 1e15:
                            snapshot[name] = None
                        else:
                            snapshot[name] = val
                    except (ValueError, OverflowError):
                        snapshot[name] = None
            indicators[dt_str] = snapshot
            row_count += 1

    print(f"    Loaded {row_count} bars, {len(col_indices)} indicators")
    return indicators


def load_trades(js_path: Path) -> list:
    """Load trades from the existing data/trades.js."""
    print("  Loading trade data...")
    with open(js_path, "r", encoding="utf-8") as f:
        content = f.read()
    json_str = content.replace("const TRADE_DATA = ", "").rstrip(";\n")
    data = json.loads(json_str)
    trades = data["trades"]
    print(f"    Loaded {len(trades)} trades")
    return trades


def join_trades_with_indicators(trades: list, indicators: dict) -> list:
    """For each trade, find the ES indicator snapshot at entry time."""
    print("  Joining trades with ES indicators...")
    joined = []
    matched = 0
    unmatched = 0

    for t in trades:
        entry_dt = datetime.fromisoformat(t["entryTime"])
        # Round to nearest minute
        dt_key = entry_dt.strftime("%Y-%m-%d %H:%M:00")

        snapshot = indicators.get(dt_key)
        if not snapshot:
            # Try +/- 1 minute
            for delta in [1, -1, 2, -2]:
                alt = (entry_dt + timedelta(minutes=delta)).strftime("%Y-%m-%d %H:%M:00")
                snapshot = indicators.get(alt)
                if snapshot:
                    break

        if snapshot:
            joined.append({**t, "_indicators": snapshot})
            matched += 1
        else:
            unmatched += 1

    print(f"    Matched: {matched}, Unmatched: {unmatched}")
    return joined


def compute_bucket_stats(trades: list) -> dict:
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


def analyze_indicator(joined_trades: list, indicator_name: str, ind_type: str) -> dict:
    """Analyze one indicator's predictive power by bucketing trades."""
    if ind_type == "boolean":
        true_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is True]
        false_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is False]
        if not true_trades or not false_trades:
            return None
        true_stats = compute_bucket_stats(true_trades)
        false_stats = compute_bucket_stats(false_trades)
        # PF spread
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
    if len(valid_trades) < 50:
        return None

    values = [t["_indicators"][indicator_name] for t in valid_trades]
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Compute quintile boundaries
    boundaries = []
    for i in range(1, NUM_BUCKETS):
        idx = int(n * i / NUM_BUCKETS)
        boundaries.append(sorted_vals[idx])

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
        stats = compute_bucket_stats(bucket)
        lo = sorted_vals[0] if i == 0 else boundaries[i - 1]
        hi = boundaries[i] if i < len(boundaries) else sorted_vals[-1]
        stats["label"] = f"Q{i+1}"
        stats["range"] = [round(lo, 4), round(hi, 4)]
        bucket_stats.append(stats)
        pfs.append(stats["pf"] if stats["pf"] < 9999 else 3)

    pf_spread = max(pfs) - min(pfs) if pfs else 0

    # Simple correlation between indicator value and profit
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
    """Find the best threshold to use as a no-trade filter for a numeric indicator."""
    if ind_type == "boolean":
        # For booleans, test both directions
        true_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is True]
        false_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is False]
        if not true_trades or not false_trades:
            return None
        true_s = compute_bucket_stats(true_trades)
        false_s = compute_bucket_stats(false_trades)
        # Which side is worse?
        if true_s["pf"] < false_s["pf"] and true_s["pf"] < 1.0:
            filtered = true_trades
            kept = false_trades
            rule = f"{indicator_name} = True"
        elif false_s["pf"] < true_s["pf"] and false_s["pf"] < 1.0:
            filtered = false_trades
            kept = true_trades
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
            "pfFiltered": compute_bucket_stats(filtered)["pf"],
            "pfKept": compute_bucket_stats(kept)["pf"],
            "wrFiltered": compute_bucket_stats(filtered)["winRate"],
            "wrKept": compute_bucket_stats(kept)["winRate"],
        }

    # Numeric: test various percentile thresholds
    valid_trades = [t for t in joined_trades if t["_indicators"].get(indicator_name) is not None]
    if len(valid_trades) < 50:
        return None

    values = [t["_indicators"][indicator_name] for t in valid_trades]
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    best_filter = None
    best_improvement = 0

    # Test lower thresholds (filter out below X) and upper (filter out above X)
    for pct in [10, 15, 20, 25, 30, 33]:
        # Low threshold: filter trades where indicator < threshold
        lo_idx = int(n * pct / 100)
        lo_thresh = sorted_vals[lo_idx]
        lo_filtered = [t for t in valid_trades if t["_indicators"][indicator_name] <= lo_thresh]
        lo_kept = [t for t in valid_trades if t["_indicators"][indicator_name] > lo_thresh]

        if lo_filtered and lo_kept:
            fs = compute_bucket_stats(lo_filtered)
            ks = compute_bucket_stats(lo_kept)
            # The filter is good if filtered trades have low PF and kept have higher PF
            if fs["pf"] < 1.0 and ks["pf"] > fs["pf"]:
                improvement = ks["pnl"] - sum(t["profit"] for t in valid_trades)
                # Actually, improvement = -sum of filtered P&L (since it's negative, removing it helps)
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

        # High threshold: filter trades where indicator > threshold
        hi_idx = int(n * (100 - pct) / 100)
        hi_thresh = sorted_vals[hi_idx]
        hi_filtered = [t for t in valid_trades if t["_indicators"][indicator_name] >= hi_thresh]
        hi_kept = [t for t in valid_trades if t["_indicators"][indicator_name] < hi_thresh]

        if hi_filtered and hi_kept:
            fs = compute_bucket_stats(hi_filtered)
            ks = compute_bucket_stats(hi_kept)
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
    """Apply a single filter to a set of trades, returning (filtered, kept)."""
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
    """Validate a filter (trained elsewhere) on out-of-sample test trades."""
    filtered, kept = apply_filter(test_trades, filt)
    test_baseline = compute_bucket_stats(test_trades)
    fs = compute_bucket_stats(filtered) if filtered else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
    ks = compute_bucket_stats(kept) if kept else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
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
    """Apply two filters with OR logic, returning (filtered, kept)."""
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
    """Test combinations of 2 top filters. Train on train_trades, validate on test_trades."""
    # Only combine the top filters that have real impact on training data
    top_filters = [f for f in single_filters if f is not None and f["pnlFiltered"] < -500][:8]

    combo_results = []
    for f1, f2 in combinations(top_filters, 2):
        # Evaluate on training set
        filtered, kept = apply_combo_filter(train_trades, f1, f2)

        if filtered and kept:
            fs = compute_bucket_stats(filtered)
            ks = compute_bucket_stats(kept)
            if fs["pf"] < ks["pf"]:
                # Validate on test set
                t_filtered, t_kept = apply_combo_filter(test_trades, f1, f2)
                tfs = compute_bucket_stats(t_filtered) if t_filtered else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
                tks = compute_bucket_stats(t_kept) if t_kept else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}

                combo_results.append({
                    "rule": f"({f1['rule']}) OR ({f2['rule']})",
                    # Training (in-sample) results
                    "tradesFiltered": len(filtered),
                    "tradesKept": len(kept),
                    "pnlFiltered": fs["pnl"],
                    "pnlKept": ks["pnl"],
                    "pfFiltered": fs["pf"],
                    "pfKept": ks["pf"],
                    "wrFiltered": fs["winRate"],
                    "wrKept": ks["winRate"],
                    # Test (out-of-sample) results
                    "testTradesFiltered": len(t_filtered),
                    "testTradesKept": len(t_kept),
                    "testPnlFiltered": tfs["pnl"],
                    "testPnlKept": tks["pnl"],
                    "testPfFiltered": tfs["pf"],
                    "testPfKept": tks["pf"],
                    "testWrFiltered": tfs["winRate"],
                    "testWrKept": tks["winRate"],
                })

    # Sort by P&L improvement (most negative filtered P&L = most saved)
    combo_results.sort(key=lambda x: x["pnlFiltered"])
    return combo_results[:10]


MIN_STRATEGY_TRADES = 30  # Skip strategies with fewer trades for per-strategy analysis


def analyze_per_strategy(train_trades: list, test_trades: list, single_filters: list) -> list:
    """Evaluate top single filters on each individual strategy.

    Returns a list of per-strategy result dicts, each containing baseline stats
    and the impact of every top filter on that strategy.
    """
    # Use the top filters that had real impact globally (same criteria as combo selection)
    top_filters = [f for f in single_filters if f is not None and f["pnlFiltered"] < -500][:8]
    if not top_filters:
        top_filters = [f for f in single_filters if f is not None][:8]

    # Group trades by strategy
    train_by_strat = defaultdict(list)
    test_by_strat = defaultdict(list)
    for t in train_trades:
        train_by_strat[t.get("strategy", "Unknown")].append(t)
    for t in test_trades:
        test_by_strat[t.get("strategy", "Unknown")].append(t)

    results = []
    for strat in sorted(train_by_strat.keys()):
        s_train = train_by_strat[strat]
        s_test = test_by_strat.get(strat, [])
        total = len(s_train) + len(s_test)
        if total < MIN_STRATEGY_TRADES:
            continue

        train_baseline = compute_bucket_stats(s_train)
        test_baseline = compute_bucket_stats(s_test) if s_test else None

        filter_results = []
        for filt in top_filters:
            # Apply filter on this strategy's train trades
            train_filtered, train_kept = apply_filter(s_train, filt)
            train_fs = compute_bucket_stats(train_filtered) if train_filtered else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
            train_ks = compute_bucket_stats(train_kept) if train_kept else {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}

            # Apply filter on this strategy's test trades
            test_fs = {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
            test_ks = {"count": 0, "pnl": 0, "pf": 0, "winRate": 0}
            if s_test:
                test_filtered, test_kept = apply_filter(s_test, filt)
                test_fs = compute_bucket_stats(test_filtered) if test_filtered else test_fs
                test_ks = compute_bucket_stats(test_kept) if test_kept else test_ks

            # Verdict: HOLDS if filtered trades are net losers on both train and test
            holds = (train_fs["pnl"] < 0 and test_fs["pnl"] < 0
                     and test_ks["pf"] >= test_fs.get("pf", 0))

            filter_results.append({
                "rule": filt["rule"],
                "trainFiltered": train_fs["count"],
                "trainPnlFiltered": train_fs["pnl"],
                "trainPfKept": train_ks["pf"],
                "testFiltered": test_fs["count"],
                "testPnlFiltered": test_fs["pnl"],
                "testPfKept": test_ks["pf"],
                "holds": holds,
            })

        results.append({
            "strategy": strat,
            "trainTrades": len(s_train),
            "testTrades": len(s_test),
            "trainBaseline": train_baseline,
            "testBaseline": test_baseline,
            "filters": filter_results,
        })

    # Sort by total trade count descending
    results.sort(key=lambda x: x["trainTrades"] + x["testTrades"], reverse=True)
    return results


def _discover_filters_for_group(train: list, test: list, top_n: int = 5) -> list:
    """Run threshold discovery on all indicators for a group of trades.

    Returns the top_n filters sorted by train P&L saved, each with test validation.
    """
    _empty_test = {
        "testBaseline": None,
        "testTradesFiltered": 0, "testTradesKept": 0,
        "testPnlFiltered": 0, "testPnlKept": 0,
        "testPfFiltered": 0, "testPfKept": 0,
        "testWrFiltered": 0, "testWrKept": 0,
    }
    discovered = []
    for name, ind_type in REGIME_INDICATORS.items():
        filt = find_optimal_threshold(train, name, ind_type)
        if filt is None:
            continue
        test_val = validate_filter(test, filt) if test else dict(_empty_test)
        filt.update(test_val)
        holds = (filt["testPnlFiltered"] < 0
                 and filt.get("testPfKept", 0) > filt.get("testPfFiltered", 0))
        filt["holds"] = holds
        discovered.append(filt)

    discovered.sort(key=lambda x: x["pnlFiltered"])
    return discovered[:top_n]


def discover_strategy_filters(train_trades: list, test_trades: list) -> list:
    """Discover strategy-specific indicator thresholds.

    Runs find_optimal_threshold() on each strategy's trades independently
    three ways: combined, long-only, and short-only. Validates on that
    strategy's test trades. Returns top 5 filters per group.
    """
    # Group trades by strategy
    train_by_strat = defaultdict(list)
    test_by_strat = defaultdict(list)
    for t in train_trades:
        train_by_strat[t.get("strategy", "Unknown")].append(t)
    for t in test_trades:
        test_by_strat[t.get("strategy", "Unknown")].append(t)

    results = []
    for strat in sorted(train_by_strat.keys()):
        s_train = train_by_strat[strat]
        s_test = test_by_strat.get(strat, [])
        total = len(s_train) + len(s_test)
        if total < MIN_STRATEGY_TRADES:
            continue

        # Combined (all directions)
        filters_all = _discover_filters_for_group(s_train, s_test)

        # Per-direction splits
        directions = []
        for direction in ("Long", "Short"):
            d_train = [t for t in s_train if t.get("direction") == direction]
            d_test = [t for t in s_test if t.get("direction") == direction]
            d_total = len(d_train) + len(d_test)
            if d_total < MIN_STRATEGY_TRADES:
                continue
            d_filters = _discover_filters_for_group(d_train, d_test)
            if d_filters:
                directions.append({
                    "direction": direction,
                    "trainTrades": len(d_train),
                    "testTrades": len(d_test),
                    "tradeCount": d_total,
                    "filters": d_filters,
                })

        results.append({
            "strategy": strat,
            "tradeCount": total,
            "trainTrades": len(s_train),
            "testTrades": len(s_test),
            "filters": filters_all,
            "directions": directions,
        })

    # Sort by total trade count descending
    results.sort(key=lambda x: x["tradeCount"], reverse=True)
    return results


def build_regime_timeline(joined_trades: list, indicators: dict, top_indicators: list) -> list:
    """Build a daily regime score timeline."""
    # Group indicator data by date
    daily_scores = defaultdict(lambda: {"values": [], "trades": [], "wins": 0, "losses": 0, "pnl": 0})

    # Use the top 5 numeric indicators to build a composite score
    score_indicators = [r["name"] for r in top_indicators if r["type"] == "numeric"][:5]

    # Get percentile distributions for normalization
    all_values = defaultdict(list)
    for dt_str, snapshot in indicators.items():
        for ind in score_indicators:
            v = snapshot.get(ind)
            if v is not None:
                all_values[ind].append(v)

    percentile_ranges = {}
    for ind, vals in all_values.items():
        s = sorted(vals)
        percentile_ranges[ind] = (s[len(s) // 4], s[3 * len(s) // 4])  # Q1, Q3

    # Build daily timeline
    for t in joined_trades:
        date = t["entryDate"]
        daily_scores[date]["pnl"] += t["profit"]
        daily_scores[date]["trades"].append(t["profit"])
        if t["profit"] > 0:
            daily_scores[date]["wins"] += 1
        elif t["profit"] < 0:
            daily_scores[date]["losses"] += 1

    # For each day, sample the indicator values at midday (or average across the day)
    for dt_str, snapshot in indicators.items():
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        # Only sample at 10:00 for a representative daily reading
        if dt.hour == 10 and dt.minute == 0:
            date = dt.strftime("%Y-%m-%d")
            score = 0
            count = 0
            for ind in score_indicators:
                v = snapshot.get(ind)
                if v is not None and ind in percentile_ranges:
                    q1, q3 = percentile_ranges[ind]
                    iqr = q3 - q1
                    if iqr > 0:
                        # Normalize to 0-1 range based on IQR
                        normalized = (v - q1) / iqr
                        score += max(0, min(1, normalized))
                        count += 1
            if count > 0 and date in daily_scores:
                daily_scores[date]["score"] = round(score / count, 3)

    timeline = []
    for date in sorted(daily_scores.keys()):
        d = daily_scores[date]
        tc = len(d["trades"])
        entry = {
            "date": date,
            "trades": tc,
            "pnl": round(d["pnl"], 2),
            "wins": d["wins"],
            "losses": d["losses"],
            "winRate": round(d["wins"] / (d["wins"] + d["losses"]) * 100, 1) if (d["wins"] + d["losses"]) > 0 else 0,
        }
        if "score" in d:
            entry["score"] = d["score"]
        timeline.append(entry)

    return timeline


def fmt_pnl(val: float) -> str:
    return f"${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"


def write_markdown_report(results: list, single_filters: list, combo_filters: list, baseline: dict, split_info: dict, per_strategy: list = None, strategy_discovery: list = None):
    """Write the regime analysis markdown report with walk-forward validation."""
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    tb = split_info["trainBaseline"]
    tsb = split_info["testBaseline"]

    lines = ["# Market Regime Analysis — No-Trade Filter Report", ""]
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Method:** ES 1-minute indicators as market regime proxy for all instruments")
    lines.append(f"**Validation:** Walk-forward ({TRAIN_RATIO:.0%} train / {1-TRAIN_RATIO:.0%} test, chronological)")
    lines.append(f"**Baseline:** {baseline['count']} trades, {baseline['winRate']}% WR, "
                 f"PF {baseline['pf']}, P&L {fmt_pnl(baseline['pnl'])}")
    lines.append("")

    # Walk-forward split details
    lines.append("## Walk-Forward Split")
    lines.append("")
    lines.append("| Set | Period | Trades | Win Rate | PF | P&L |")
    lines.append("|-----|--------|------:|--------:|---:|----:|")
    lines.append(f"| **Train** | {split_info['trainStart']} to {split_info['trainEnd']} | "
                 f"{split_info['trainCount']} | {tb['winRate']}% | {tb['pf']:.2f} | {fmt_pnl(tb['pnl'])} |")
    lines.append(f"| **Test** | {split_info['testStart']} to {split_info['testEnd']} | "
                 f"{split_info['testCount']} | {tsb['winRate']}% | {tsb['pf']:.2f} | {fmt_pnl(tsb['pnl'])} |")
    lines.append("")

    # Top indicators ranked
    lines.append("## Indicator Rankings (by PF spread across quintiles)")
    lines.append("")
    lines.append("| Rank | Indicator | Type | PF Spread | Correlation | Best Bucket PF | Worst Bucket PF |")
    lines.append("|-----:|-----------|------|----------:|------------:|---------------:|----------------:|")

    for i, r in enumerate(results[:20], 1):
        corr_str = f"{r['correlation']:.4f}" if r["correlation"] is not None else "N/A"
        pfs = [b["pf"] for b in r["buckets"] if b["pf"] < 9999]
        best_pf = max(pfs) if pfs else 0
        worst_pf = min(pfs) if pfs else 0
        lines.append(f"| {i} | {r['name']} | {r['type']} | {r['pfSpread']:.2f} | "
                     f"{corr_str} | {best_pf:.2f} | {worst_pf:.2f} |")
    lines.append("")

    # Detailed bucket breakdowns for top 10
    lines.append("## Top 10 Indicator Detail")
    lines.append("")
    for r in results[:10]:
        lines.append(f"### {r['name']} (PF spread: {r['pfSpread']:.2f})")
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
                lines.append(f"| {b['label']} | {b['range'][0]:.2f} - {b['range'][1]:.2f} | "
                             f"{b['count']} | {b['winRate']}% | "
                             f"{fmt_pnl(b['pnl'])} | {pf_str} | {fmt_pnl(b['avgTrade'])} |")
        lines.append("")

    # Single filter recommendations with walk-forward validation
    valid_filters = [f for f in single_filters if f is not None]
    valid_filters.sort(key=lambda x: x["pnlFiltered"])

    lines.append("## Recommended Single Filters (Walk-Forward Validated)")
    lines.append("")
    lines.append("Thresholds found on training data, then tested on unseen test data.")
    lines.append("**HOLDS** = filter works out-of-sample (filtered trades are net losers). **OVERFIT** = filter doesn't generalize.")
    lines.append("")
    lines.append("| Rule | Train Trades Out | Train P&L Saved | Train PF-> | Test Trades Out | Test P&L Saved | Test PF-> | Verdict |")
    lines.append("|------|----------------:|--------------:|--------:|---------------:|-------------:|-------:|---------|")
    for f in valid_filters[:15]:
        holds = f.get("testPnlFiltered", 0) < 0 and f.get("testPfKept", 0) > f.get("testPfFiltered", 0)
        verdict = "**HOLDS**" if holds else "OVERFIT"
        test_pnl = f.get("testPnlFiltered", 0)
        test_pf_kept = f.get("testPfKept", 0)
        test_out = f.get("testTradesFiltered", 0)
        lines.append(f"| {f['rule']} | {f['tradesFiltered']} | {fmt_pnl(f['pnlFiltered'])} | "
                     f"{f['pfKept']:.2f} | {test_out} | {fmt_pnl(test_pnl)} | "
                     f"{test_pf_kept:.2f} | {verdict} |")
    lines.append("")

    # Combo filters with walk-forward
    if combo_filters:
        lines.append("## Best Combination Filters (Walk-Forward Validated)")
        lines.append("")
        lines.append("| Rules | Train Out | Train P&L | Train PF-> | Test Out | Test P&L | Test PF-> | Verdict |")
        lines.append("|-------|--------:|---------:|--------:|--------:|---------:|-------:|---------|")
        for cf in combo_filters[:10]:
            holds = cf.get("testPnlFiltered", 0) < 0 and cf.get("testPfKept", 0) > cf.get("testPfFiltered", 0)
            verdict = "**HOLDS**" if holds else "OVERFIT"
            lines.append(f"| {cf['rule']} | {cf['tradesFiltered']} | {fmt_pnl(cf['pnlFiltered'])} | "
                         f"{cf['pfKept']:.2f} | {cf.get('testTradesFiltered', 0)} | "
                         f"{fmt_pnl(cf.get('testPnlFiltered', 0))} | "
                         f"{cf.get('testPfKept', 0):.2f} | {verdict} |")
        lines.append("")

    # Per-strategy filter breakdown
    if per_strategy:
        # Get the filter rules used
        filter_rules = []
        if per_strategy[0]["filters"]:
            filter_rules = [fr["rule"] for fr in per_strategy[0]["filters"]]

        lines.append("## Per-Strategy Filter Breakdown")
        lines.append("")
        lines.append(f"Evaluating top {len(filter_rules)} global filters on each strategy individually.")
        lines.append(f"Strategies with fewer than {MIN_STRATEGY_TRADES} total trades are excluded.")
        lines.append("")

        # Per-filter detail tables
        for fi, rule in enumerate(filter_rules):
            lines.append(f"### Filter: `{rule}`")
            lines.append("")
            lines.append("| Strategy | Train Trades | Train P&L Saved | Test P&L Saved | Verdict |")
            lines.append("|----------|------------:|--------------:|--------------:|---------|")
            for ps in per_strategy:
                fr = ps["filters"][fi]
                verdict = "**HOLDS**" if fr["holds"] else "OVERFIT"
                if fr["trainFiltered"] == 0 and fr["testFiltered"] == 0:
                    verdict = "n/a"
                lines.append(f"| {ps['strategy']} | {ps['trainTrades']} | "
                             f"{fmt_pnl(fr['trainPnlFiltered'])} | "
                             f"{fmt_pnl(fr['testPnlFiltered'])} | {verdict} |")
            lines.append("")

        # Summary matrix
        lines.append("### Summary Matrix")
        lines.append("")
        # Header: Strategy | Filter1 | Filter2 | ...
        short_rules = [r.replace("_", "\\_") for r in filter_rules]
        header = "| Strategy | " + " | ".join(short_rules) + " |"
        sep = "|----------|" + "|".join(["--------:" for _ in filter_rules]) + "|"
        lines.append(header)
        lines.append(sep)
        for ps in per_strategy:
            cells = []
            for fr in ps["filters"]:
                if fr["trainFiltered"] == 0 and fr["testFiltered"] == 0:
                    cells.append("n/a")
                elif fr["holds"]:
                    cells.append("**HOLDS**")
                else:
                    cells.append("OVERFIT")
            lines.append(f"| {ps['strategy']} | " + " | ".join(cells) + " |")
        lines.append("")

    # Per-strategy indicator discovery
    if strategy_discovery:
        lines.append("## Per-Strategy Indicator Discovery")
        lines.append("")
        lines.append("Indicators discovered independently for each strategy using walk-forward validation.")
        lines.append(f"Strategies with fewer than {MIN_STRATEGY_TRADES} total trades are excluded.")
        lines.append("Fewer trades per strategy means more OVERFIT verdicts — the validation flags this honestly.")
        lines.append("")

        for sd in strategy_discovery:
            if not sd["filters"] and not sd.get("directions"):
                continue
            holds_count = sum(1 for f in sd["filters"] if f.get("holds"))
            lines.append(f"### {sd['strategy']} ({sd['tradeCount']} trades, {holds_count}/{len(sd['filters'])} HOLD)")
            lines.append("")

            # Combined (all directions)
            if sd["filters"]:
                lines.append("**All Trades**")
                lines.append("")
                lines.append("| Rule | Train Trades Out | Train P&L Saved | Test P&L Saved | Verdict |")
                lines.append("|------|----------------:|--------------:|--------------:|---------|")
                for f in sd["filters"]:
                    verdict = "**HOLDS**" if f.get("holds") else "OVERFIT"
                    test_pnl = f.get("testPnlFiltered", 0)
                    lines.append(f"| {f['rule']} | {f['tradesFiltered']} | "
                                 f"{fmt_pnl(f['pnlFiltered'])} | {fmt_pnl(test_pnl)} | {verdict} |")
                lines.append("")

            # Per-direction splits
            for dd in sd.get("directions", []):
                if not dd["filters"]:
                    continue
                d_holds = sum(1 for f in dd["filters"] if f.get("holds"))
                lines.append(f"**{dd['direction']} Only** ({dd['tradeCount']} trades, {d_holds}/{len(dd['filters'])} HOLD)")
                lines.append("")
                lines.append("| Rule | Train Trades Out | Train P&L Saved | Test P&L Saved | Verdict |")
                lines.append("|------|----------------:|--------------:|--------------:|---------|")
                for f in dd["filters"]:
                    verdict = "**HOLDS**" if f.get("holds") else "OVERFIT"
                    test_pnl = f.get("testPnlFiltered", 0)
                    lines.append(f"| {f['rule']} | {f['tradesFiltered']} | "
                                 f"{fmt_pnl(f['pnlFiltered'])} | {fmt_pnl(test_pnl)} | {verdict} |")
                lines.append("")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report: {OUTPUT_MD}")


def main():
    print("=== Market Regime Analysis (Walk-Forward Validation) ===")
    print()

    # Step 1: Load data
    indicators = load_es_indicators(ES_CSV)
    trades = load_trades(TRADES_JS)

    # Step 2: Join trades with indicators
    joined = join_trades_with_indicators(trades, indicators)
    if not joined:
        print("ERROR: No trades matched with indicator data")
        return

    # Step 2b: Chronological train/test split
    joined.sort(key=lambda t: t["entryTime"])
    split_idx = int(len(joined) * TRAIN_RATIO)
    train_trades = joined[:split_idx]
    test_trades = joined[split_idx:]

    train_baseline = compute_bucket_stats(train_trades)
    test_baseline = compute_bucket_stats(test_trades)
    full_baseline = compute_bucket_stats(joined)

    train_start = train_trades[0]["entryDate"]
    train_end = train_trades[-1]["entryDate"]
    test_start = test_trades[0]["entryDate"]
    test_end = test_trades[-1]["entryDate"]

    print(f"\n  Walk-forward split ({TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}):")
    print(f"    Train: {len(train_trades)} trades ({train_start} to {train_end})")
    print(f"      WR={train_baseline['winRate']}%, PF={train_baseline['pf']}, P&L=${train_baseline['pnl']:,.2f}")
    print(f"    Test:  {len(test_trades)} trades ({test_start} to {test_end})")
    print(f"      WR={test_baseline['winRate']}%, PF={test_baseline['pf']}, P&L=${test_baseline['pnl']:,.2f}")

    # Step 3: Analyze each indicator (on ALL data for bucket charts)
    print("\n  Analyzing indicators...")
    results = []
    for name, ind_type in REGIME_INDICATORS.items():
        r = analyze_indicator(joined, name, ind_type)
        if r:
            results.append(r)

    # Step 4: Rank by PF spread
    results.sort(key=lambda x: x["pfSpread"], reverse=True)

    print(f"\n  Top 10 indicators by PF spread:")
    for i, r in enumerate(results[:10], 1):
        corr_str = f"corr={r['correlation']:.3f}" if r["correlation"] is not None else ""
        print(f"    {i}. {r['name']:<30} spread={r['pfSpread']:.2f}  {corr_str}")

    # Step 5: Find optimal thresholds on TRAIN data, validate on TEST data
    print("\n  Finding optimal filter thresholds (training on first 70%)...")
    single_filters = []
    for name, ind_type in REGIME_INDICATORS.items():
        f = find_optimal_threshold(train_trades, name, ind_type)
        if f:
            # Validate on test set
            test_result = validate_filter(test_trades, f)
            f.update(test_result)
            single_filters.append(f)

    single_filters.sort(key=lambda x: x["pnlFiltered"])
    print(f"\n  Top single filters (train -> test):")
    for f in single_filters[:8]:
        holds = "HOLDS" if f["testPnlFiltered"] < 0 and f["testPfKept"] > f["testPfFiltered"] else "OVERFIT"
        print(f"    {f['rule']:<40} train: ${f['pnlFiltered']:,.0f} saved, PF->{f['pfKept']:.2f}  |  "
              f"test: ${f['testPnlFiltered']:,.0f}, PF->{f['testPfKept']:.2f}  [{holds}]")

    # Step 6: Test combos (train on train, validate on test)
    print("\n  Testing combination filters...")
    combo_filters = test_combo_filters(train_trades, test_trades, single_filters)
    if combo_filters:
        print(f"\n  Top combo filters (train -> test):")
        for cf in combo_filters[:5]:
            holds = "HOLDS" if cf["testPnlFiltered"] < 0 and cf["testPfKept"] > cf["testPfFiltered"] else "OVERFIT"
            print(f"    {cf['rule']}")
            print(f"      train: filtered={cf['tradesFiltered']}  P&L=${cf['pnlFiltered']:,.0f}  PF->{cf['pfKept']:.2f}  |  "
                  f"test: ${cf['testPnlFiltered']:,.0f}  PF->{cf['testPfKept']:.2f}  [{holds}]")

    # Step 6b: Per-strategy filter breakdown
    print("\n  Analyzing per-strategy filter impact...")
    per_strategy = analyze_per_strategy(train_trades, test_trades, single_filters)
    if per_strategy:
        print(f"    {len(per_strategy)} strategies with >= {MIN_STRATEGY_TRADES} trades")
        for ps in per_strategy[:5]:
            holds_count = sum(1 for fr in ps["filters"] if fr["holds"])
            print(f"    {ps['strategy']:<25} {ps['trainTrades']+ps['testTrades']:>4} trades, "
                  f"{holds_count}/{len(ps['filters'])} filters HOLD")

    # Step 6c: Per-strategy indicator discovery
    print("\n  Discovering strategy-specific indicators...")
    strategy_discovery = discover_strategy_filters(train_trades, test_trades)
    if strategy_discovery:
        print(f"    {len(strategy_discovery)} strategies analyzed")
        for sd in strategy_discovery[:5]:
            holds_count = sum(1 for f in sd["filters"] if f.get("holds"))
            top3 = sd["filters"][:3]
            print(f"    {sd['strategy']:<25} {sd['tradeCount']:>4} trades, "
                  f"{holds_count}/{len(sd['filters'])} filters HOLD")
            for f in top3:
                verdict = "HOLDS" if f.get("holds") else "OVERFIT"
                print(f"      {f['rule']:<36} train: {fmt_pnl(f['pnlFiltered'])}  "
                      f"test: {fmt_pnl(f.get('testPnlFiltered', 0))}  [{verdict}]")
            for dd in sd.get("directions", []):
                d_holds = sum(1 for f in dd["filters"] if f.get("holds"))
                print(f"      {dd['direction']:<6} ({dd['tradeCount']} trades): "
                      f"{d_holds}/{len(dd['filters'])} HOLD")
                for f in dd["filters"][:2]:
                    verdict = "HOLDS" if f.get("holds") else "OVERFIT"
                    print(f"        {f['rule']:<34} train: {fmt_pnl(f['pnlFiltered'])}  "
                          f"test: {fmt_pnl(f.get('testPnlFiltered', 0))}  [{verdict}]")

    # Step 7: Build timeline (on all data)
    print("\n  Building regime timeline...")
    timeline = build_regime_timeline(joined, indicators, results)

    # Step 8: Output
    print("\n  Writing outputs...")

    # Markdown report
    split_info = {
        "trainCount": len(train_trades), "testCount": len(test_trades),
        "trainStart": train_start, "trainEnd": train_end,
        "testStart": test_start, "testEnd": test_end,
        "trainBaseline": train_baseline, "testBaseline": test_baseline,
    }
    write_markdown_report(results, single_filters, combo_filters, full_baseline, split_info, per_strategy, strategy_discovery)

    # JS data file for dashboard
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

    regime_data = {
        "baseline": full_baseline,
        "trainBaseline": train_baseline,
        "testBaseline": test_baseline,
        "split": {
            "trainCount": len(train_trades), "testCount": len(test_trades),
            "trainRange": f"{train_start} to {train_end}",
            "testRange": f"{test_start} to {test_end}",
        },
        "indicators": results[:20],
        "singleFilters": [f for f in single_filters if f is not None][:15],
        "comboFilters": combo_filters[:10],
        "perStrategy": per_strategy,
        "strategyDiscovery": strategy_discovery,
        "timeline": timeline,
    }
    regime_data = sanitize(regime_data)

    OUTPUT_JS.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(regime_data, separators=(",", ":"))
    with open(OUTPUT_JS, "w", encoding="utf-8") as f:
        f.write(f"const REGIME_DATA = {json_str};\n")

    size_kb = OUTPUT_JS.stat().st_size / 1024
    print(f"  Dashboard data: {OUTPUT_JS} ({size_kb:.0f} KB)")
    print("\nDone!")


if __name__ == "__main__":
    main()
