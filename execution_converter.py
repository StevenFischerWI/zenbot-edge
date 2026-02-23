"""
Execution-to-Trade Converter

Reads NinjaTrader execution logs (individual fills), deduplicates by execution ID,
reconstructs flat-to-flat trades, computes P&L from fill prices x instrument multiplier,
and returns trade dicts compatible with process_trades.read_trades() schema.

Usage:
    from execution_converter import read_executions
    trades = read_executions(r"D:\futures\Stats\Daily\inbound\combined.csv")
"""

import csv
import fnmatch
from collections import defaultdict
from datetime import datetime

POINT_VALUES = {
    "ES": 50, "MES": 5, "NQ": 20, "MNQ": 2,
    "RTY": 50, "M2K": 5, "YM": 5, "MYM": 0.5,
    "GC": 100, "MGC": 10, "SI": 5000, "HG": 25000,
    "CL": 1000, "MCL": 100, "NG": 10000, "PL": 50,
    "NKD": 5, "FDAX": 25, "ZS": 50, "6E": 125000,
    "MBT": 5,
}


def _parse_time(val: str) -> datetime:
    """Parse execution timestamp. Handles M/D/YYYY H:MM and M/D/YYYY H:MM:SS."""
    val = val.strip()
    try:
        return datetime.strptime(val, "%m/%d/%Y %H:%M:%S")
    except ValueError:
        return datetime.strptime(val, "%m/%d/%Y %H:%M")


def _parse_commission(val: str) -> float:
    """Parse commission like '$0.00' or '$1.23'."""
    val = val.strip().replace("$", "").replace(",", "")
    if not val:
        return 0.0
    return float(val)


def _parse_position_signed(pos_str: str) -> int:
    """Parse position to signed int: '3 L' -> +3, '2 S' -> -2, '-' -> 0."""
    pos = pos_str.strip()
    if pos == "-" or not pos:
        return 0
    parts = pos.split()
    if len(parts) == 2:
        qty = int(parts[0])
        return qty if parts[1] == "L" else -qty
    return 0


def _chain_order_fills(minute_fills: list[dict], starting_pos: int) -> list[dict]:
    """Order fills within a minute by following the position chain.

    Each fill's Position column gives the resulting position after that fill.
    We follow the chain: current_pos + delta -> fill's resulting position.
    """
    ordered = []
    remaining = list(minute_fills)
    current_pos = starting_pos

    while remaining:
        found = False
        for i, fill in enumerate(remaining):
            delta = fill["qty"] if fill["action"] == "Buy" else -fill["qty"]
            expected_pos = current_pos + delta
            actual_pos = _parse_position_signed(fill["position"])
            if expected_pos == actual_pos:
                ordered.append(fill)
                current_pos = actual_pos
                remaining.pop(i)
                found = True
                break
        if not found:
            # Fallback: append remaining sorted by Exit-first (best effort)
            remaining.sort(key=lambda f: (0 if f["ex"] == "Exit" else 1))
            ordered.extend(remaining)
            break

    return ordered


def _finalize_trade(fills: list[dict], instrument_base: str, instrument_full: str,
                    account: str, trade_id: int) -> dict | None:
    """Convert accumulated fills into a single trade dict."""
    if not fills:
        return None

    entry_fills = [f for f in fills if f["ex"] == "Entry"]
    exit_fills = [f for f in fills if f["ex"] == "Exit"]

    if not entry_fills or not exit_fills:
        return None

    # Direction from first entry fill
    direction = "Long" if entry_fills[0]["action"] == "Buy" else "Short"

    # Weighted average prices
    entry_qty = sum(f["qty"] for f in entry_fills)
    exit_qty = sum(f["qty"] for f in exit_fills)
    if entry_qty == 0 or exit_qty == 0:
        return None

    entry_price = sum(f["price"] * f["qty"] for f in entry_fills) / entry_qty
    exit_price = sum(f["price"] * f["qty"] for f in exit_fills) / exit_qty

    # P&L: (sum_sell_value - sum_buy_value) * point_value
    # Works for both Long (buy entry, sell exit) and Short (sell entry, buy exit)
    sum_buy = sum(f["price"] * f["qty"] for f in fills if f["action"] == "Buy")
    sum_sell = sum(f["price"] * f["qty"] for f in fills if f["action"] == "Sell")
    point_value = POINT_VALUES.get(instrument_base, 1)
    profit = round((sum_sell - sum_buy) * point_value, 2)

    commission = round(sum(f["commission"] for f in fills), 2)

    entry_time = entry_fills[0]["time"]
    exit_time = exit_fills[-1]["time"]
    holding_seconds = (exit_time - entry_time).total_seconds()

    strategy = account.replace("Sim-", "", 1)

    return {
        "id": trade_id,
        "instrument": instrument_base,
        "instrumentFull": instrument_full,
        "strategy": strategy,
        "subStrategy": account,
        "direction": direction,
        "qty": entry_qty,
        "entryPrice": round(entry_price, 10),
        "exitPrice": round(exit_price, 10),
        "entryTime": entry_time.isoformat(),
        "exitTime": exit_time.isoformat(),
        "entryName": entry_fills[0]["name"],
        "exitName": exit_fills[-1]["name"],
        "profit": profit,
        "commission": commission,
        "mae": 0,
        "mfe": 0,
        "etd": 0,
        "bars": 0,
        "holdingMinutes": round(holding_seconds / 60, 2),
        "entryHour": entry_time.hour,
        "entryHalfHour": f"{entry_time.hour:02d}:{'00' if entry_time.minute < 30 else '30'}",
        "entryDayOfWeek": entry_time.weekday(),
        "entryDate": entry_time.strftime("%Y-%m-%d"),
    }


def read_executions(csv_path: str, accounts: str = "Sim-*") -> list[dict]:
    """Read execution CSV, dedup, reconstruct trades, return trade dicts.

    Args:
        csv_path: Path to execution CSV file
        accounts: fnmatch pattern for account filtering (default: "Sim-*")

    Returns:
        List of trade dicts matching read_trades() schema, sorted by exitTime
    """
    # Step 1: Read & dedup by execution ID
    seen_ids = set()
    fills = []
    skipped_instruments = set()
    total_rows = 0
    dup_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            total_rows += 1
            if len(row) < 13:
                continue

            exec_id = row[5].strip()
            if exec_id in seen_ids:
                dup_rows += 1
                continue
            seen_ids.add(exec_id)

            instrument_full = row[0].strip()
            instrument_base = instrument_full.split()[0] if instrument_full else ""

            # Skip garbage rows (strategy names or numbers as instruments)
            if instrument_base not in POINT_VALUES:
                skipped_instruments.add(instrument_full)
                continue

            account = row[12].strip()
            if not fnmatch.fnmatch(account, accounts):
                continue

            try:
                time = _parse_time(row[4])
            except (ValueError, IndexError):
                continue

            fills.append({
                "instrument_full": instrument_full,
                "instrument_base": instrument_base,
                "account": account,
                "action": row[1].strip(),
                "qty": int(row[2].strip()),
                "price": float(row[3].strip()),
                "time": time,
                "ex": row[6].strip(),       # "Entry" or "Exit"
                "position": row[7].strip(),  # "2 L", "1 S", "-"
                "name": row[9].strip(),
                "commission": _parse_commission(row[10]),
            })

    print(f"  {total_rows:,} rows, {dup_rows:,} duplicates removed, "
          f"{len(fills):,} fills for {accounts}")
    if skipped_instruments:
        sample = sorted(skipped_instruments)[:10]
        print(f"  Skipped {len(skipped_instruments)} unrecognized instruments: {sample}")

    # Step 2: Group by (account, instrument_base)
    groups = defaultdict(list)
    for fill in fills:
        key = (fill["account"], fill["instrument_base"])
        groups[key].append(fill)

    print(f"  {len(groups)} (account, instrument) groups")

    # Step 3: Reconstruct trades per group using position-chain ordering
    all_trades = []
    trade_id = 1
    imbalanced = 0
    incomplete = 0
    chain_fallbacks = 0

    for (account, instrument_base), group_fills in sorted(groups.items()):
        # Sort by time ascending
        group_fills.sort(key=lambda f: f["time"])

        # Group fills by minute
        by_minute = defaultdict(list)
        for fill in group_fills:
            minute_key = fill["time"].replace(second=0, microsecond=0)
            by_minute[minute_key].append(fill)

        # Walk through minutes, tracking signed position
        # positive = long, negative = short, 0 = flat
        current_pos = 0
        current_fills = []
        instrument_full = group_fills[0]["instrument_full"]

        for minute_key in sorted(by_minute.keys()):
            minute_fills = by_minute[minute_key]

            # Order fills within minute by following position chain
            ordered = _chain_order_fills(minute_fills, current_pos)
            if len(ordered) != len(minute_fills):
                chain_fallbacks += 1

            for fill in ordered:
                delta = fill["qty"] if fill["action"] == "Buy" else -fill["qty"]
                new_pos = current_pos + delta

                if current_pos == 0 and new_pos != 0:
                    # Transition flat -> non-flat: start new trade
                    current_fills = [fill]
                    instrument_full = fill["instrument_full"]
                elif current_pos != 0:
                    # In a trade: accumulate fill
                    current_fills.append(fill)
                    if new_pos == 0:
                        # Position went flat — finalize trade
                        trade = _finalize_trade(
                            current_fills, instrument_base, instrument_full,
                            account, trade_id
                        )
                        if trade:
                            buy_q = sum(f["qty"] for f in current_fills if f["action"] == "Buy")
                            sell_q = sum(f["qty"] for f in current_fills if f["action"] == "Sell")
                            if buy_q != sell_q:
                                imbalanced += 1
                            all_trades.append(trade)
                            trade_id += 1
                        current_fills = []
                # else: flat -> flat (shouldn't happen, skip)

                current_pos = new_pos

        # Discard incomplete trade at end of group
        if current_pos != 0 and current_fills:
            incomplete += 1
            current_pos = 0

    # Sort by exit time
    all_trades.sort(key=lambda t: t["exitTime"])

    print(f"  Reconstructed {len(all_trades):,} trades")
    if imbalanced:
        print(f"  WARNING: {imbalanced} trades with buy/sell qty imbalance")
    if incomplete:
        print(f"  Discarded {incomplete} incomplete trades (position not flat at end)")
    if chain_fallbacks:
        print(f"  Note: {chain_fallbacks} minutes used fallback ordering")

    return all_trades
