"""
Futures Strategy Analysis - Trade Data Processor
Reads all-trades.csv, filters Sim-* strategies, computes comprehensive metrics,
and outputs data/trades.js for the web dashboard.
"""

import csv
import json
import math
import re
import statistics
from datetime import datetime
from collections import defaultdict
from pathlib import Path

INPUT_CSV = r"D:\futures\code\all-trades.csv"
OUTPUT_JS = Path(__file__).parent / "data" / "trades.js"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

FILTER_OUTLIERS = False


def parse_currency(val: str) -> float:
    val = val.strip()
    if not val or val == "":
        return 0.0
    negative = val.startswith("(") or val.startswith("-")
    cleaned = val.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
    if cleaned.startswith("-"):
        cleaned = cleaned[1:]
    if not cleaned:
        return 0.0
    amount = float(cleaned)
    return -amount if negative else amount


def parse_datetime(val: str) -> datetime:
    val = val.strip()
    # Support both NinjaTrader export formats:
    #   Old: "1/2/2026 7:00:00 AM"  (12-hour with AM/PM)
    #   New: "2025-10-27 15:35:59"  (ISO-like 24-hour)
    if "AM" in val or "PM" in val:
        return datetime.strptime(val, "%m/%d/%Y %I:%M:%S %p")
    return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")


def normalize_instrument(full_name: str) -> str:
    return full_name.strip().split(" ")[0]


def filter_outliers(trades: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove per-instrument outliers using IQR method.
    Returns (kept_trades, excluded_trades)."""
    by_inst = defaultdict(list)
    for t in trades:
        by_inst[t["instrument"]].append(t["profit"])

    # Compute bounds per instrument
    bounds = {}
    for inst, profits in by_inst.items():
        profits_sorted = sorted(profits)
        n = len(profits_sorted)
        q1 = profits_sorted[n // 4]
        q3 = profits_sorted[(3 * n) // 4]
        iqr = q3 - q1
        lower = q1 - IQR_MULT * iqr
        upper = q3 + IQR_MULT * iqr
        bounds[inst] = (lower, upper)

    kept = []
    excluded = []
    for t in trades:
        lower, upper = bounds[t["instrument"]]
        if t["profit"] < lower or t["profit"] > upper:
            excluded.append(t)
        else:
            kept.append(t)

    return kept, excluded


def derive_strategy_family(account: str) -> str:
    """Derive strategy family from account name.
    Sim-EMA-Runner-AH -> EMA Runner
    Sim-Levels-2M     -> Levels
    Sim-Snappy-3M-AH  -> Snappy
    Sim-VWMA-Wick-1M  -> VWMA Wick
    """
    name = account.replace("Sim-", "", 1)
    name = re.sub(r"-AH$", "", name)
    name = re.sub(r"-\d+M$", "", name)
    return name.replace("-", " ")


def read_trades(csv_path: str) -> list[dict]:
    # Read all qualifying rows first
    raw_rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header if h.strip()]

        for row in reader:
            if len(row) < 23:
                continue
            account = row[2].strip()
            if not account.startswith("Sim-"):
                continue
            try:
                entry_time = parse_datetime(row[8])
                exit_time = parse_datetime(row[9])
            except (ValueError, IndexError):
                continue
            raw_rows.append((row, entry_time, exit_time))

    # Group by (account, direction, entryTime, exitTime) to consolidate
    # multi-contract fills that NinjaTrader splits into separate rows
    groups = defaultdict(list)
    for row, entry_time, exit_time in raw_rows:
        key = (row[2].strip(), row[4].strip(), row[8].strip(), row[9].strip())
        groups[key].append((row, entry_time, exit_time))

    trades = []
    for key, group_rows in groups.items():
        first_row, entry_time, exit_time = group_rows[0]
        account = first_row[2].strip()
        holding_seconds = (exit_time - entry_time).total_seconds()

        # Sum across all contracts in this fill
        total_qty = sum(int(r[5].strip()) for r, _, _ in group_rows)
        total_profit = sum(parse_currency(r[12]) for r, _, _ in group_rows)
        total_commission = sum(parse_currency(r[14]) for r, _, _ in group_rows)
        total_mae = sum(parse_currency(r[19]) for r, _, _ in group_rows)
        total_mfe = sum(parse_currency(r[20]) for r, _, _ in group_rows)
        total_etd = sum(parse_currency(r[21]) for r, _, _ in group_rows)
        max_bars = max((int(r[22].strip()) if r[22].strip() else 0) for r, _, _ in group_rows)

        # Use first row's trade ID as the canonical ID
        trade = {
            "id": int(first_row[0].strip()),
            "instrument": normalize_instrument(first_row[1]),
            "instrumentFull": first_row[1].strip(),
            "strategy": account.replace("Sim-", ""),
            "subStrategy": account,
            "direction": first_row[4].strip(),
            "qty": total_qty,
            "entryPrice": float(first_row[6].strip()),
            "exitPrice": float(first_row[7].strip()),
            "entryTime": entry_time.isoformat(),
            "exitTime": exit_time.isoformat(),
            "entryName": first_row[10].strip(),
            "exitName": first_row[11].strip(),
            "profit": round(total_profit, 2),
            "commission": round(total_commission, 2),
            "mae": round(total_mae, 2),
            "mfe": round(total_mfe, 2),
            "etd": round(total_etd, 2),
            "bars": max_bars,
            "holdingMinutes": round(holding_seconds / 60, 2),
            "entryHour": entry_time.hour,
            "entryDayOfWeek": entry_time.weekday(),
            "entryDate": entry_time.strftime("%Y-%m-%d"),
        }
        trades.append(trade)

    # Sort by exit time
    trades.sort(key=lambda t: t["exitTime"])
    return trades


def compute_metrics(trades: list[dict], strategy_name: str) -> dict:
    if not trades:
        return {"strategyName": strategy_name, "tradeCount": 0}

    profits = [t["profit"] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    breakevens = [p for p in profits if p == 0]

    win_count = len(wins)
    loss_count = len(losses)
    be_count = len(breakevens)
    total = len(trades)
    decisions = win_count + loss_count  # exclude breakevens from win rate

    win_rate = round((win_count / decisions) * 100, 2) if decisions > 0 else 0
    total_pnl = round(sum(profits), 2)
    avg_win = round(statistics.mean(wins), 2) if wins else 0
    avg_loss = round(statistics.mean(losses), 2) if losses else 0
    max_win = round(max(wins), 2) if wins else 0
    max_loss = round(min(losses), 2) if losses else 0
    avg_trade = round(statistics.mean(profits), 2) if profits else 0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    # Equity curve & max drawdown
    equity_curve = []
    equity = 0
    peak = 0
    max_dd = 0
    for i, t in enumerate(trades):
        equity += t["profit"]
        equity = round(equity, 2)
        equity_curve.append([i + 1, equity])
        peak = max(peak, equity)
        dd = equity - peak
        max_dd = min(max_dd, dd)
    max_dd = round(max_dd, 2)

    # Daily P&L
    daily = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        d = t["entryDate"]
        daily[d]["pnl"] += t["profit"]
        daily[d]["trades"] += 1
        if t["profit"] > 0:
            daily[d]["wins"] += 1
    daily_pnl = [
        {"date": d, "pnl": round(v["pnl"], 2), "trades": v["trades"], "wins": v["wins"]}
        for d, v in sorted(daily.items())
    ]

    # Sharpe ratio (daily returns, annualized)
    sharpe = None
    daily_returns = [dp["pnl"] for dp in daily_pnl]
    if len(daily_returns) >= 10:
        mean_ret = statistics.mean(daily_returns)
        std_ret = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
        sharpe = round((mean_ret / std_ret) * math.sqrt(252), 2) if std_ret > 0 else 0

    # Profit distribution (histogram bins at $50 increments)
    bin_size = 50
    min_p = max(min(profits), -2000)
    max_p = min(max(profits), 2000)
    bin_start = (int(min_p) // bin_size) * bin_size
    bin_end = ((int(max_p) // bin_size) + 1) * bin_size
    bins = list(range(bin_start, bin_end + bin_size, bin_size))
    counts = [0] * (len(bins) - 1)
    for p in profits:
        clamped = max(min_p, min(max_p, p))
        idx = int((clamped - bin_start) / bin_size)
        idx = max(0, min(idx, len(counts) - 1))
        counts[idx] += 1
    profit_distribution = {"bins": bins, "counts": counts}

    # Long vs Short breakdown
    longs = [t for t in trades if t["direction"] == "Long"]
    shorts = [t for t in trades if t["direction"] == "Short"]
    long_wins = sum(1 for t in longs if t["profit"] > 0)
    short_wins = sum(1 for t in shorts if t["profit"] > 0)
    long_decisions = sum(1 for t in longs if t["profit"] != 0)
    short_decisions = sum(1 for t in shorts if t["profit"] != 0)

    long_short = {
        "longCount": len(longs),
        "shortCount": len(shorts),
        "longWinRate": round((long_wins / long_decisions) * 100, 2) if long_decisions > 0 else 0,
        "shortWinRate": round((short_wins / short_decisions) * 100, 2) if short_decisions > 0 else 0,
        "longPnL": round(sum(t["profit"] for t in longs), 2),
        "shortPnL": round(sum(t["profit"] for t in shorts), 2),
        "longAvg": round(statistics.mean([t["profit"] for t in longs]), 2) if longs else 0,
        "shortAvg": round(statistics.mean([t["profit"] for t in shorts]), 2) if shorts else 0,
    }

    # Hourly P&L and win rate
    hourly = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        h = t["entryHour"]
        hourly[h]["pnl"] += t["profit"]
        hourly[h]["trades"] += 1
        if t["profit"] > 0:
            hourly[h]["wins"] += 1
    hourly_pnl = {}
    hourly_win_rate = {}
    hourly_trade_count = {}
    for h in sorted(hourly.keys()):
        hourly_pnl[str(h)] = round(hourly[h]["pnl"], 2)
        hourly_trade_count[str(h)] = hourly[h]["trades"]
        decisions_h = sum(1 for t in trades if t["entryHour"] == h and t["profit"] != 0)
        hourly_win_rate[str(h)] = round((hourly[h]["wins"] / decisions_h) * 100, 1) if decisions_h > 0 else 0

    # Day-of-week P&L
    dow = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        d = DAY_NAMES[t["entryDayOfWeek"]]
        dow[d]["pnl"] += t["profit"]
        dow[d]["trades"] += 1
        if t["profit"] > 0:
            dow[d]["wins"] += 1
    dow_pnl = {}
    dow_trade_count = {}
    dow_win_rate = {}
    for d in DAY_NAMES[:5]:  # Mon-Fri
        if d in dow:
            dow_pnl[d] = round(dow[d]["pnl"], 2)
            dow_trade_count[d] = dow[d]["trades"]
            decisions_d = sum(1 for t in trades if DAY_NAMES[t["entryDayOfWeek"]] == d and t["profit"] != 0)
            dow_win_rate[d] = round((dow[d]["wins"] / decisions_d) * 100, 1) if decisions_d > 0 else 0

    # Hour x Day-of-week matrix (for heatmap)
    hour_day_matrix = {}
    for d in DAY_NAMES[:5]:
        hour_day_matrix[d] = {}
    for t in trades:
        d = DAY_NAMES[t["entryDayOfWeek"]]
        h = str(t["entryHour"])
        if d in hour_day_matrix:
            hour_day_matrix[d][h] = round(hour_day_matrix[d].get(h, 0) + t["profit"], 2)

    # Instrument breakdown
    inst = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "losses": 0, "win_profits": [], "loss_profits": []})
    for t in trades:
        i = t["instrument"]
        inst[i]["pnl"] += t["profit"]
        inst[i]["trades"] += 1
        if t["profit"] > 0:
            inst[i]["wins"] += 1
            inst[i]["win_profits"].append(t["profit"])
        elif t["profit"] < 0:
            inst[i]["losses"] += 1
            inst[i]["loss_profits"].append(t["profit"])
    instrument_pnl = {}
    instrument_trade_count = {}
    instrument_details = []
    for i_name in sorted(inst.keys()):
        v = inst[i_name]
        instrument_pnl[i_name] = round(v["pnl"], 2)
        instrument_trade_count[i_name] = v["trades"]
        i_decisions = v["wins"] + v["losses"]
        gp = sum(v["win_profits"])
        gl = abs(sum(v["loss_profits"]))
        instrument_details.append({
            "instrument": i_name,
            "trades": v["trades"],
            "winRate": round((v["wins"] / i_decisions) * 100, 1) if i_decisions > 0 else 0,
            "avgWin": round(statistics.mean(v["win_profits"]), 2) if v["win_profits"] else 0,
            "avgLoss": round(statistics.mean(v["loss_profits"]), 2) if v["loss_profits"] else 0,
            "pnl": round(v["pnl"], 2),
            "profitFactor": round(gp / gl, 2) if gl > 0 else (float("inf") if gp > 0 else 0),
        })

    # MAE vs Profit, MFE vs Profit (for scatter plots)
    mae_vs_profit = [[t["mae"], t["profit"]] for t in trades]
    mfe_vs_profit = [[t["mfe"], t["profit"]] for t in trades]

    # Rolling metrics
    rolling_pnl_20 = []
    rolling_wr_50 = []
    if len(trades) >= 20:
        for i in range(19, len(trades)):
            window = profits[i - 19 : i + 1]
            rolling_pnl_20.append([i + 1, round(statistics.mean(window), 2)])
    if len(trades) >= 50:
        for i in range(49, len(trades)):
            window = profits[i - 49 : i + 1]
            w = sum(1 for p in window if p > 0)
            d = sum(1 for p in window if p != 0)
            rolling_wr_50.append([i + 1, round((w / d) * 100, 1) if d > 0 else 0])

    # Consecutive win/loss streaks
    max_consec_wins = 0
    max_consec_losses = 0
    current_streak = 0
    current_type = None
    streaks = []
    for i, p in enumerate(profits):
        if p > 0:
            if current_type == "win":
                current_streak += 1
            else:
                if current_type and current_streak > 0:
                    streaks.append({"type": current_type, "length": current_streak})
                current_type = "win"
                current_streak = 1
        elif p < 0:
            if current_type == "loss":
                current_streak += 1
            else:
                if current_type and current_streak > 0:
                    streaks.append({"type": current_type, "length": current_streak})
                current_type = "loss"
                current_streak = 1
        # breakeven doesn't break streaks
    if current_type and current_streak > 0:
        streaks.append({"type": current_type, "length": current_streak})

    win_streaks = [s["length"] for s in streaks if s["type"] == "win"]
    loss_streaks = [s["length"] for s in streaks if s["type"] == "loss"]
    max_consec_wins = max(win_streaks) if win_streaks else 0
    max_consec_losses = max(loss_streaks) if loss_streaks else 0

    # Average holding time
    avg_holding = round(statistics.mean([t["holdingMinutes"] for t in trades]), 2) if trades else 0

    return {
        "strategyName": strategy_name,
        "tradeCount": total,
        "winCount": win_count,
        "lossCount": loss_count,
        "breakEvenCount": be_count,
        "winRate": win_rate,
        "totalPnL": total_pnl,
        "avgWin": avg_win,
        "avgLoss": avg_loss,
        "maxWin": max_win,
        "maxLoss": max_loss,
        "avgTrade": avg_trade,
        "profitFactor": profit_factor,
        "sharpeRatio": sharpe,
        "maxDrawdown": max_dd,
        "avgHoldingMinutes": avg_holding,
        "maxConsecWins": max_consec_wins,
        "maxConsecLosses": max_consec_losses,
        "longShort": long_short,
        "equityCurve": equity_curve,
        "dailyPnL": daily_pnl,
        "profitDistribution": profit_distribution,
        "hourlyPnL": hourly_pnl,
        "hourlyWinRate": hourly_win_rate,
        "hourlyTradeCount": hourly_trade_count,
        "dowPnL": dow_pnl,
        "dowTradeCount": dow_trade_count,
        "dowWinRate": dow_win_rate,
        "hourDayMatrix": hour_day_matrix,
        "instrumentPnL": instrument_pnl,
        "instrumentTradeCount": instrument_trade_count,
        "instrumentDetails": instrument_details,
        "maeVsProfit": mae_vs_profit,
        "mfeVsProfit": mfe_vs_profit,
        "rollingPnL20": rolling_pnl_20,
        "rollingWinRate50": rolling_wr_50,
        "streaks": streaks,
    }


def compute_sub_strategy_summary(trades: list[dict]) -> list[dict]:
    """Compute summary metrics for each sub-strategy (account) within a family."""
    by_sub = defaultdict(list)
    for t in trades:
        by_sub[t["subStrategy"]].append(t)

    summaries = []
    for name in sorted(by_sub.keys()):
        sub_trades = by_sub[name]
        profits = [t["profit"] for t in sub_trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        w = len(wins)
        l = len(losses)
        decisions = w + l
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        summaries.append({
            "name": name,
            "trades": len(sub_trades),
            "winRate": round((w / decisions) * 100, 1) if decisions > 0 else 0,
            "totalPnL": round(sum(profits), 2),
            "avgTrade": round(statistics.mean(profits), 2) if profits else 0,
            "avgWin": round(statistics.mean(wins), 2) if wins else 0,
            "avgLoss": round(statistics.mean(losses), 2) if losses else 0,
            "profitFactor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0),
            "maxDrawdown": 0,  # computed below
        })
        # Compute max drawdown for sub-strategy
        equity = 0
        peak = 0
        max_dd = 0
        for t in sub_trades:
            equity += t["profit"]
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
        summaries[-1]["maxDrawdown"] = round(max_dd, 2)

    return summaries


def sanitize(obj):
    """Handle infinity/NaN values for JSON serialization."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return 9999.99
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def build_dashboard_output(trades: list[dict], source_label: str = "all-trades.csv",
                           total_trades_raw: int | None = None) -> dict:
    """Run outlier filtering + all metric computation on consolidated trades.
    Returns (output_dict, families_sorted, strategy_metrics, all_metrics, by_family, kept_trades).
    """
    if FILTER_OUTLIERS:
        kept, excluded = filter_outliers(trades)
        if excluded:
            print(f"  Excluded {len(excluded)} outlier trades:")
            from collections import Counter
            by_inst = Counter(t["instrument"] for t in excluded)
            for inst, cnt in by_inst.most_common():
                ex_trades = [t for t in excluded if t["instrument"] == inst]
                pnls = [t["profit"] for t in ex_trades]
                print(f"    {inst}: {cnt} trades (P&L range: ${min(pnls):,.2f} to ${max(pnls):,.2f})")
        print(f"  Remaining: {len(kept)} trades")
    else:
        kept = trades
    print(f"  Processing {len(kept)} trades")

    # Group by strategy family (derived from account name)
    by_family = defaultdict(list)
    for t in kept:
        by_family[t["strategy"]].append(t)

    families_sorted = sorted(by_family.keys())
    print(f"  Found {len(families_sorted)} strategy families:")

    # Compute per-family metrics + sub-strategy comparisons
    strategy_metrics = {}
    for name in families_sorted:
        family_trades = by_family[name]
        metrics = compute_metrics(family_trades, name)
        metrics["subStrategies"] = compute_sub_strategy_summary(family_trades)
        strategy_metrics[name] = metrics
        subs = len(metrics["subStrategies"])
        pnl_str = f"${metrics['totalPnL']:>10,.2f}"
        print(f"    {name:<20} {metrics['tradeCount']:>5} trades  {pnl_str}  WR: {metrics['winRate']:>5.1f}%  ({subs} variants)")

    # Compute aggregate
    all_metrics = compute_metrics(kept, "All Strategies")
    all_metrics["subStrategies"] = []
    # For _ALL, sub-strategies are the families themselves
    for name in families_sorted:
        m = strategy_metrics[name]
        all_metrics["subStrategies"].append({
            "name": name,
            "trades": m["tradeCount"],
            "winRate": m["winRate"],
            "totalPnL": m["totalPnL"],
            "avgTrade": m["avgTrade"],
            "avgWin": m["avgWin"],
            "avgLoss": m["avgLoss"],
            "profitFactor": m["profitFactor"],
            "maxDrawdown": m["maxDrawdown"],
        })
    strategy_metrics["_ALL"] = all_metrics
    print(f"\n  TOTAL: {all_metrics['tradeCount']} trades, ${all_metrics['totalPnL']:,.2f} P&L, {all_metrics['winRate']:.1f}% WR")

    # Build dates list
    all_dates = sorted(set(t["entryDate"] for t in kept))

    # Build output
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "sourceFile": source_label,
            "totalTradesRaw": total_trades_raw,
            "totalTradesFiltered": len(kept),
            "dateRange": {"start": all_dates[0], "end": all_dates[-1]} if all_dates else {},
            "tradingDays": len(all_dates),
            "strategies": families_sorted,
            "instruments": sorted(set(t["instrument"] for t in kept)),
        },
        "strategies": strategy_metrics,
        "trades": kept,
    }

    output = sanitize(output)
    return output, families_sorted, strategy_metrics, all_metrics, by_family, kept


def write_trades_js(output: dict, path: Path = OUTPUT_JS) -> None:
    """Serialize output dict and write as a JS file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(output, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"const TRADE_DATA = {json_str};\n")
    size_kb = path.stat().st_size / 1024
    print(f"\n  Output: {path} ({size_kb:.0f} KB)")


def main():
    print("Reading trades from CSV...")
    trades = read_trades(INPUT_CSV)
    print(f"  Consolidated to {len(trades)} Sim-* trades")

    # Count raw trades
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        raw_count = sum(1 for _ in f) - 1  # subtract header

    output, families_sorted, strategy_metrics, all_metrics, by_family, trades = \
        build_dashboard_output(trades, source_label="all-trades.csv", total_trades_raw=raw_count)

    write_trades_js(output)

    # ---- Generate summary CSV and Markdown reports ----
    all_dates = sorted(set(t["entryDate"] for t in trades))
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    write_summary_csv(reports_dir, families_sorted, strategy_metrics, all_metrics)
    write_summary_markdown(reports_dir, families_sorted, strategy_metrics, all_metrics, all_dates, trades)
    write_per_strategy_reports(reports_dir, families_sorted, strategy_metrics, by_family)

    print("Done!")


def fmt_pnl(val: float) -> str:
    return f"${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"


def fmt_pf(val: float) -> str:
    if math.isinf(val) or val >= 9999:
        return "Inf"
    return f"{val:.2f}"


def write_summary_csv(reports_dir: Path, names: list, metrics: dict, all_m: dict):
    """One CSV with every strategy as a row — easy to sort/filter/analyze."""
    csv_path = reports_dir / "strategy_summary.csv"
    fields = [
        "Strategy", "Trades", "Wins", "Losses", "BreakEven",
        "WinRate%", "TotalPnL", "AvgTrade", "AvgWin", "AvgLoss",
        "MaxWin", "MaxLoss", "ProfitFactor", "SharpeRatio",
        "MaxDrawdown", "MaxConsecWins", "MaxConsecLosses",
        "AvgHoldingMin", "LongTrades", "ShortTrades",
        "LongWinRate%", "ShortWinRate%", "LongPnL", "ShortPnL",
        "BestHour", "WorstHour", "BestDay", "WorstDay",
        "Instruments",
    ]

    rows = []
    for name in names:
        m = metrics[name]
        ls = m.get("longShort", {})

        # Best/worst hour
        hp = m.get("hourlyPnL", {})
        best_h = max(hp, key=hp.get) if hp else ""
        worst_h = min(hp, key=hp.get) if hp else ""

        # Best/worst day of week
        dp = m.get("dowPnL", {})
        best_d = max(dp, key=dp.get) if dp else ""
        worst_d = min(dp, key=dp.get) if dp else ""

        insts = " ".join(sorted(m.get("instrumentTradeCount", {}).keys()))

        rows.append([
            name, m["tradeCount"], m["winCount"], m["lossCount"], m["breakEvenCount"],
            m["winRate"], m["totalPnL"], m["avgTrade"], m["avgWin"], m["avgLoss"],
            m["maxWin"], m["maxLoss"],
            m["profitFactor"] if not math.isinf(m["profitFactor"]) else 9999.99,
            m["sharpeRatio"] if m["sharpeRatio"] is not None else "",
            m["maxDrawdown"], m["maxConsecWins"], m["maxConsecLosses"],
            m["avgHoldingMinutes"],
            ls.get("longCount", 0), ls.get("shortCount", 0),
            ls.get("longWinRate", 0), ls.get("shortWinRate", 0),
            ls.get("longPnL", 0), ls.get("shortPnL", 0),
            best_h, worst_h, best_d, worst_d, insts,
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"  Summary CSV: {csv_path}")


def write_summary_markdown(reports_dir: Path, names: list, metrics: dict, all_m: dict, dates: list, trades: list):
    """Overall summary markdown with rankings and key insights."""
    md_path = reports_dir / "summary.md"

    lines = ["# Futures Strategy Analysis — Summary Report", ""]
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Date range:** {dates[0]} to {dates[-1]} ({len(dates)} trading days)")
    lines.append(f"**Total trades:** {all_m['tradeCount']:,} across {len(names)} strategies")
    lines.append(f"**Net P&L:** {fmt_pnl(all_m['totalPnL'])}  |  **Win rate:** {all_m['winRate']:.1f}%  |  **Profit factor:** {fmt_pf(all_m['profitFactor'])}")
    lines.append("")

    # Ranked table
    lines.append("## Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | Trades | Win Rate | Avg Trade | P&L | PF | Sharpe | Max DD |")
    lines.append("|----------|-------:|--------:|---------:|----:|---:|------:|------:|")

    ranked = sorted(names, key=lambda n: metrics[n]["totalPnL"], reverse=True)
    for name in ranked:
        m = metrics[name]
        sharpe = f"{m['sharpeRatio']:.2f}" if m["sharpeRatio"] is not None else "N/A"
        lines.append(
            f"| {name} | {m['tradeCount']} | {m['winRate']:.1f}% "
            f"| {fmt_pnl(m['avgTrade'])} | {fmt_pnl(m['totalPnL'])} "
            f"| {fmt_pf(m['profitFactor'])} | {sharpe} | {fmt_pnl(m['maxDrawdown'])} |"
        )
    lines.append("")

    # Top/bottom rankings
    lines.append("## Rankings")
    lines.append("")

    def top_n(key, n=5, reverse=True, fmt=None, filter_fn=None):
        pool = [name for name in names if (filter_fn is None or filter_fn(metrics[name]))]
        ranked_list = sorted(pool, key=lambda nm: metrics[nm].get(key, 0) or 0, reverse=reverse)[:n]
        result = []
        for i, nm in enumerate(ranked_list, 1):
            val = metrics[nm].get(key, 0)
            if val is None:
                val = 0
            display = fmt(val) if fmt else str(val)
            result.append(f"{i}. **{nm}** — {display}")
        return result

    lines.append("### Best P&L")
    lines.extend(top_n("totalPnL", fmt=fmt_pnl))
    lines.append("")

    lines.append("### Worst P&L")
    lines.extend(top_n("totalPnL", reverse=False, fmt=fmt_pnl))
    lines.append("")

    lines.append("### Highest Win Rate (min 20 trades)")
    lines.extend(top_n("winRate", fmt=lambda v: f"{v:.1f}%", filter_fn=lambda m: m["tradeCount"] >= 20))
    lines.append("")

    lines.append("### Best Profit Factor (min 20 trades)")
    lines.extend(top_n("profitFactor", fmt=fmt_pf, filter_fn=lambda m: m["tradeCount"] >= 20))
    lines.append("")

    lines.append("### Best Sharpe Ratio")
    lines.extend(top_n("sharpeRatio", fmt=lambda v: f"{v:.2f}" if v else "N/A",
                        filter_fn=lambda m: m.get("sharpeRatio") is not None))
    lines.append("")

    lines.append("### Smallest Max Drawdown (min 20 trades)")
    lines.extend(top_n("maxDrawdown", reverse=True, fmt=fmt_pnl, filter_fn=lambda m: m["tradeCount"] >= 20))
    lines.append("")

    # Long vs Short aggregate
    ls = all_m.get("longShort", {})
    lines.append("## Long vs Short (All Strategies)")
    lines.append("")
    lines.append(f"| | Trades | Win Rate | Avg P&L | Total P&L |")
    lines.append(f"|---|------:|--------:|-------:|--------:|")
    lines.append(f"| Long | {ls.get('longCount',0)} | {ls.get('longWinRate',0):.1f}% | {fmt_pnl(ls.get('longAvg',0))} | {fmt_pnl(ls.get('longPnL',0))} |")
    lines.append(f"| Short | {ls.get('shortCount',0)} | {ls.get('shortWinRate',0):.1f}% | {fmt_pnl(ls.get('shortAvg',0))} | {fmt_pnl(ls.get('shortPnL',0))} |")
    lines.append("")

    # Instrument summary
    lines.append("## Instrument Summary")
    lines.append("")
    lines.append("| Instrument | Trades | Win Rate | P&L | PF |")
    lines.append("|-----------|------:|--------:|----:|---:|")
    for d in sorted(all_m.get("instrumentDetails", []), key=lambda x: x["pnl"], reverse=True):
        lines.append(f"| {d['instrument']} | {d['trades']} | {d['winRate']:.1f}% | {fmt_pnl(d['pnl'])} | {fmt_pf(d['profitFactor'])} |")
    lines.append("")

    # Time analysis
    lines.append("## Time of Day (All Strategies)")
    lines.append("")
    lines.append("| Hour | Trades | P&L | Win Rate |")
    lines.append("|------|------:|----:|--------:|")
    for h in sorted(all_m.get("hourlyPnL", {}).keys(), key=int):
        hr = int(h)
        label = f"{hr}:00" if hr < 12 else (f"12:00" if hr == 12 else f"{hr}:00")
        pnl = all_m["hourlyPnL"][h]
        cnt = all_m["hourlyTradeCount"].get(h, 0)
        wr = all_m["hourlyWinRate"].get(h, 0)
        lines.append(f"| {label} | {cnt} | {fmt_pnl(pnl)} | {wr:.1f}% |")
    lines.append("")

    lines.append("## Day of Week (All Strategies)")
    lines.append("")
    lines.append("| Day | Trades | P&L | Win Rate |")
    lines.append("|-----|------:|----:|--------:|")
    for d in DAY_NAMES[:5]:
        if d in all_m.get("dowPnL", {}):
            lines.append(f"| {d} | {all_m['dowTradeCount'].get(d,0)} | {fmt_pnl(all_m['dowPnL'][d])} | {all_m['dowWinRate'].get(d,0):.1f}% |")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary MD:  {md_path}")


def write_per_strategy_reports(reports_dir: Path, names: list, metrics: dict, by_family: dict):
    """One markdown file per strategy with detailed analysis."""
    strat_dir = reports_dir / "strategies"
    strat_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        m = metrics[name]
        trades = by_family[name]
        safe_name = name.replace(" ", "-").replace("/", "-")
        md_path = strat_dir / f"{safe_name}.md"

        lines = [f"# {name}", ""]
        lines.append(f"**Trades:** {m['tradeCount']}  |  **Win rate:** {m['winRate']:.1f}%  |  **P&L:** {fmt_pnl(m['totalPnL'])}")
        lines.append("")

        # Core metrics table
        lines.append("## Key Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Total Trades | {m['tradeCount']} |")
        lines.append(f"| Wins / Losses / BE | {m['winCount']} / {m['lossCount']} / {m['breakEvenCount']} |")
        lines.append(f"| Win Rate | {m['winRate']:.1f}% |")
        lines.append(f"| Total P&L | {fmt_pnl(m['totalPnL'])} |")
        lines.append(f"| Avg Trade | {fmt_pnl(m['avgTrade'])} |")
        lines.append(f"| Avg Win | {fmt_pnl(m['avgWin'])} |")
        lines.append(f"| Avg Loss | {fmt_pnl(m['avgLoss'])} |")
        lines.append(f"| Max Win | {fmt_pnl(m['maxWin'])} |")
        lines.append(f"| Max Loss | {fmt_pnl(m['maxLoss'])} |")
        lines.append(f"| Profit Factor | {fmt_pf(m['profitFactor'])} |")
        sharpe_str = f"{m['sharpeRatio']:.2f}" if m["sharpeRatio"] is not None else "N/A"
        lines.append(f"| Sharpe Ratio | {sharpe_str} |")
        lines.append(f"| Max Drawdown | {fmt_pnl(m['maxDrawdown'])} |")
        lines.append(f"| Max Consec Wins | {m['maxConsecWins']} |")
        lines.append(f"| Max Consec Losses | {m['maxConsecLosses']} |")
        lines.append(f"| Avg Holding (min) | {m['avgHoldingMinutes']:.1f} |")
        lines.append("")

        # Long vs Short
        ls = m.get("longShort", {})
        lines.append("## Long vs Short")
        lines.append("")
        lines.append("| | Trades | Win Rate | Avg P&L | Total P&L |")
        lines.append("|---|------:|--------:|-------:|--------:|")
        lines.append(f"| Long | {ls.get('longCount',0)} | {ls.get('longWinRate',0):.1f}% | {fmt_pnl(ls.get('longAvg',0))} | {fmt_pnl(ls.get('longPnL',0))} |")
        lines.append(f"| Short | {ls.get('shortCount',0)} | {ls.get('shortWinRate',0):.1f}% | {fmt_pnl(ls.get('shortAvg',0))} | {fmt_pnl(ls.get('shortPnL',0))} |")
        lines.append("")

        # Instruments
        inst_details = m.get("instrumentDetails", [])
        if inst_details:
            lines.append("## By Instrument")
            lines.append("")
            lines.append("| Instrument | Trades | Win Rate | Avg Win | Avg Loss | P&L | PF |")
            lines.append("|-----------|------:|--------:|-------:|-------:|----:|---:|")
            for d in sorted(inst_details, key=lambda x: x["pnl"], reverse=True):
                lines.append(
                    f"| {d['instrument']} | {d['trades']} | {d['winRate']:.1f}% "
                    f"| {fmt_pnl(d['avgWin'])} | {fmt_pnl(d['avgLoss'])} "
                    f"| {fmt_pnl(d['pnl'])} | {fmt_pf(d['profitFactor'])} |"
                )
            lines.append("")

        # Hourly
        hp = m.get("hourlyPnL", {})
        if hp:
            lines.append("## By Hour")
            lines.append("")
            lines.append("| Hour | Trades | P&L | Win Rate |")
            lines.append("|------|------:|----:|--------:|")
            for h in sorted(hp.keys(), key=int):
                pnl = hp[h]
                cnt = m["hourlyTradeCount"].get(h, 0)
                wr = m["hourlyWinRate"].get(h, 0)
                lines.append(f"| {int(h):02d}:00 | {cnt} | {fmt_pnl(pnl)} | {wr:.1f}% |")
            lines.append("")

        # Day of week
        dp = m.get("dowPnL", {})
        if dp:
            lines.append("## By Day of Week")
            lines.append("")
            lines.append("| Day | Trades | P&L | Win Rate |")
            lines.append("|-----|------:|----:|--------:|")
            for d in DAY_NAMES[:5]:
                if d in dp:
                    lines.append(f"| {d} | {m['dowTradeCount'].get(d,0)} | {fmt_pnl(dp[d])} | {m['dowWinRate'].get(d,0):.1f}% |")
            lines.append("")

        # Daily P&L series
        daily = m.get("dailyPnL", [])
        if daily:
            lines.append("## Daily P&L")
            lines.append("")
            lines.append("| Date | Trades | P&L | Cum P&L |")
            lines.append("|------|------:|----:|-------:|")
            cum = 0
            for dp_entry in daily:
                cum += dp_entry["pnl"]
                lines.append(f"| {dp_entry['date']} | {dp_entry['trades']} | {fmt_pnl(dp_entry['pnl'])} | {fmt_pnl(round(cum, 2))} |")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"  Per-strategy: {strat_dir}/ ({len(names)} files)")


if __name__ == "__main__":
    main()
