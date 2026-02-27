"""
NinjaTrader SQLite Database Connector

Reads NinjaTrader 8's SQLite database (NinjaTrader.sqlite), reconstructs
flat-to-flat trades from execution fills, and returns trade dicts compatible
with the Zenbot Edge pipeline (process_trades / trade_store schema).

Usage:
    from nt_connector import read_nt_trades
    trades = read_nt_trades(r"C:\\Users\\steve\\Documents\\NinjaTrader 8\\db\\NinjaTrader.sqlite")
"""

import fnmatch
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from execution_converter import POINT_VALUES, _finalize_trade

# .NET epoch: 0001-01-01 00:00:00 UTC
# .NET ticks = 100-nanosecond intervals since that epoch
_DOTNET_EPOCH = datetime(1, 1, 1)

# Try zoneinfo first (works on Linux / Python with tzdata), fall back to
# dateutil, and finally to a manual EST/EDT implementation for Windows
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except (ImportError, KeyError):
    try:
        from dateutil import tz
        _ET = tz.gettz("America/New_York")
    except ImportError:
        # Manual Eastern time: determine EST vs EDT from the datetime itself
        _ET = None

_EST = timezone(timedelta(hours=-5))
_EDT = timezone(timedelta(hours=-4))


def _is_dst(dt_utc: datetime) -> bool:
    """Check if a UTC datetime falls in US Eastern Daylight Time.
    DST: 2nd Sunday of March 2:00 AM ET to 1st Sunday of November 2:00 AM ET.
    """
    year = dt_utc.year
    # 2nd Sunday of March
    march1 = datetime(year, 3, 1)
    dst_start = march1 + timedelta(days=(6 - march1.weekday()) % 7 + 7)
    dst_start = dst_start.replace(hour=7)  # 2 AM ET = 7 AM UTC
    # 1st Sunday of November
    nov1 = datetime(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    dst_end = dst_end.replace(hour=6)  # 2 AM ET = 6 AM UTC
    return dst_start <= dt_utc < dst_end

# NinjaTrader OrderAction enum
_ORDER_ACTION_BUY = {0, 1}        # Buy, BuyToCover
_ORDER_ACTION_SELL = {2, 3}       # Sell, SellShort

# Sentinel values NinjaTrader uses for unset MaxPrice/MinPrice
_SENTINEL_MAX = -1.7976931348623157e+308
_SENTINEL_MIN = 1.7976931348623157e+308

# Default NT8 database path
DEFAULT_NT_DB = r"C:\Users\steve\Documents\NinjaTrader 8\db\NinjaTrader.sqlite"

# Entry name patterns (used as fallback when Order join fails)
_ENTRY_NAMES = {"Entry", "L1", "L2", "L3", "S1", "1S", "S2"}
_EXIT_NAMES = {"Stop loss", "Profit target", "Close", "Exit", "Stop1", "Stop2",
               "Stop3", "Target1", "Target2", "Sell"}


def _ticks_to_datetime(ticks: int) -> datetime:
    """Convert .NET ticks (100ns intervals since 0001-01-01 UTC) to Eastern naive datetime."""
    utc_dt = _DOTNET_EPOCH + timedelta(microseconds=ticks // 10)
    # .NET ticks are UTC — convert to Eastern, then strip tzinfo for consistency
    # with the CSV pipeline which stores naive Eastern datetimes
    if _ET is not None:
        et_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(_ET)
    else:
        # Fallback: manual EST/EDT calculation
        tz_offset = _EDT if _is_dst(utc_dt) else _EST
        et_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(tz_offset)
    return et_dt.replace(tzinfo=None)


def _derive_strategy(account_name: str) -> tuple[str, str]:
    """Derive (strategy, subStrategy) from a NinjaTrader account name.

    Normalizes spaces to hyphens so NT accounts like "Sim-Levels 2M" produce
    the same strategy name ("Levels-2M") as the CSV pipeline's "Sim-Levels-2M".

    Returns:
        (strategy, subStrategy) tuple
    """
    # Normalize: replace spaces with hyphens for consistency with CSV pipeline
    normalized = account_name.replace(" ", "-")
    sub_strategy = normalized

    # Strip "Sim" prefix variations
    if normalized.startswith("Sim-"):
        strategy = normalized[4:]
    elif normalized.startswith("Sim"):
        strategy = normalized[3:]
    elif normalized.startswith("Playback"):
        strategy = normalized
    else:
        strategy = normalized

    # If strategy is empty or just digits, keep the full name
    if not strategy or strategy.isdigit():
        strategy = normalized

    return strategy, sub_strategy



def _determine_action(order_action, fill_name: str) -> str | None:
    """Determine 'Buy' or 'Sell' from OrderAction enum or fill name fallback.

    Returns 'Buy', 'Sell', or None if undetermined.
    """
    if order_action is not None:
        if order_action in _ORDER_ACTION_BUY:
            return "Buy"
        if order_action in _ORDER_ACTION_SELL:
            return "Sell"

    # Fallback: strip any order-id suffix from the name (e.g. "L1-20162-036886B2" → "L1")
    base_name = fill_name.split("-")[0] if fill_name else ""

    if base_name in _ENTRY_NAMES:
        return None  # Can't determine Buy/Sell from entry name alone
    if base_name in _EXIT_NAMES:
        return None  # Can't determine Buy/Sell from exit name alone

    return None


def _read_fills(nt_conn: sqlite3.Connection,
                accounts: str = "Sim*") -> list[dict]:
    """Read execution fills from NinjaTrader SQLite with full JOINs.

    Args:
        nt_conn: Connection to NinjaTrader.sqlite
        accounts: fnmatch pattern for account filtering

    Returns:
        List of normalized fill dicts sorted by (account, instrument, time)
    """
    cursor = nt_conn.execute("""
        SELECT e.Id, e.ExecutionId, e.Price, e.Quantity, e.Time,
               e.Name, e.Commission, e.Fee, e.MaxPrice, e.MinPrice,
               e.IsEntry, e.IsExit, e.Position,
               o.OrderAction,
               a.Name AS AccountName,
               mi.Name AS InstrumentName, mi.PointValue, mi.InstrumentType
        FROM Executions e
        JOIN Accounts a ON e.Account = a.Id
        JOIN Instruments i ON e.Instrument = i.Id
        JOIN MasterInstruments mi ON i.MasterInstrument = mi.Id
        LEFT JOIN Orders o ON e.OrderId = o.OrderId
        WHERE mi.InstrumentType = 0
        ORDER BY a.Name, mi.Name, e.Time
    """)

    fills = []
    skipped_accounts = set()
    skipped_instruments = set()

    for row in cursor.fetchall():
        (exec_id, execution_id, price, quantity, time_ticks,
         name, commission, fee, max_price, min_price,
         is_entry, is_exit, position,
         order_action,
         account_name,
         instrument_name, point_value, instrument_type) = row

        # Filter by account pattern
        if not fnmatch.fnmatch(account_name, accounts):
            skipped_accounts.add(account_name)
            continue

        # Skip instruments we don't have point values for
        if instrument_name not in POINT_VALUES:
            skipped_instruments.add(instrument_name)
            continue

        # Determine Buy/Sell
        action = _determine_action(order_action, name or "")

        # Convert timestamp
        try:
            dt = _ticks_to_datetime(time_ticks)
        except (OverflowError, ValueError):
            continue

        # Parse commission + fee
        total_commission = (commission or 0.0) + (fee or 0.0)

        # Determine entry/exit from flags
        entry_exit = "Entry" if is_entry else ("Exit" if is_exit else None)

        # MAE/MFE from MaxPrice/MinPrice (only valid on entry fills)
        has_mfe_mae = (max_price is not None and max_price != _SENTINEL_MAX
                       and min_price is not None and min_price != _SENTINEL_MIN)

        fills.append({
            "id": exec_id,
            "execution_id": execution_id,
            "instrument_base": instrument_name,
            "instrument_full": instrument_name,
            "account": account_name,
            "action": action,
            "qty": quantity,
            "price": price,
            "time": dt,
            "name": name or "",
            "commission": total_commission,
            "ex": entry_exit,
            "position": position,
            "max_price": max_price if has_mfe_mae else None,
            "min_price": min_price if has_mfe_mae else None,
        })

    if skipped_instruments:
        sample = sorted(skipped_instruments)[:10]
        print(f"  Skipped {len(skipped_instruments)} unrecognized instruments: {sample}")

    return fills


def _infer_actions(fills: list[dict]) -> None:
    """Infer Buy/Sell action for fills missing OrderAction using the Position column.

    The Position column in NinjaTrader's Executions table gives the signed resulting
    position after each fill (positive=long, negative=short, 0=flat).
    By tracking previous position, we determine: pos increased → Buy, decreased → Sell.
    """
    groups = defaultdict(list)
    for fill in fills:
        key = (fill["account"], fill["instrument_base"])
        groups[key].append(fill)

    for group_fills in groups.values():
        unknowns = [f for f in group_fills if f["action"] is None]
        if not unknowns:
            continue

        # Use Position column to infer Buy/Sell via position deltas
        prev_pos = 0
        for fill in group_fills:
            if fill["action"] is not None:
                # Already known — update position tracking from the known action
                if fill["action"] == "Buy":
                    prev_pos += fill["qty"]
                else:
                    prev_pos -= fill["qty"]
                continue

            current_pos = fill.get("position")
            if current_pos is not None:
                delta = current_pos - prev_pos
                if delta > 0:
                    fill["action"] = "Buy"
                elif delta < 0:
                    fill["action"] = "Sell"
                prev_pos = current_pos
            else:
                # No Position data — try name-based heuristic
                name = fill["name"].split("-")[0] if fill["name"] else ""
                if name in ("L1", "L2", "L3", "Entry"):
                    fill["action"] = "Buy"
                    prev_pos += fill["qty"]
                elif name in ("S1", "1S", "S2"):
                    fill["action"] = "Sell"
                    prev_pos -= fill["qty"]

        # Mark any remaining unknowns for removal
        for f in group_fills:
            if f["action"] is None:
                f["_skip"] = True


def _extract_mae_mfe(fills: list[dict], direction: str,
                     entry_price: float, instrument: str) -> tuple[float, float]:
    """Extract MAE and MFE from entry fills' MaxPrice/MinPrice.

    Returns (mae, mfe) as dollar values (price difference × point value).
    """
    point_value = POINT_VALUES.get(instrument, 1)

    # Collect valid MaxPrice/MinPrice from entry fills
    max_prices = [f["max_price"] for f in fills
                  if f.get("max_price") is not None and f["ex"] == "Entry"]
    min_prices = [f["min_price"] for f in fills
                  if f.get("min_price") is not None and f["ex"] == "Entry"]

    if not max_prices or not min_prices:
        return 0, 0

    trade_high = max(max_prices)
    trade_low = min(min_prices)

    if direction == "Long":
        mae = round((entry_price - trade_low) * point_value, 2)
        mfe = round((trade_high - entry_price) * point_value, 2)
    else:  # Short
        mae = round((trade_high - entry_price) * point_value, 2)
        mfe = round((entry_price - trade_low) * point_value, 2)

    return max(mae, 0), max(mfe, 0)


def read_nt_strategy_configs(db_path: str = DEFAULT_NT_DB) -> list[dict]:
    """Extract current NinjaScript strategy configurations from NinjaTrader SQLite.

    Parses the Userdata XML blob from each non-ATM strategy row and returns
    a list of config dicts with key parameters (name, instrument, timeframes,
    session hours, direction filters, etc.).

    Returns:
        List of strategy config dicts sorted by name.
    """
    import html
    import re
    from xml.etree import ElementTree as ET

    db_file = Path(db_path)
    if not db_file.exists():
        print(f"  Strategy configs: DB not found: {db_path}")
        return []

    uri_path = quote(str(db_file.resolve()).replace("\\", "/"), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?immutable=1", uri=True)

    try:
        cursor = conn.execute("""
            SELECT s.Id, s.Name, s.Classname, s.IsTerminal, s.Userdata,
                   a.Name AS AccountName
            FROM Strategies s
            LEFT JOIN Strategy2Account s2a ON s.Id = s2a.Strategy
            LEFT JOIN Accounts a ON s2a.Account = a.Id
            WHERE s.Classname != 'NinjaTrader.NinjaScript.AtmStrategy'
        """)

        seen_ids = set()
        configs = []

        for row in cursor.fetchall():
            sid, name, classname, is_terminal, userdata_blob, account = row
            # Deduplicate (join can produce multiple rows per strategy)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)

            if not userdata_blob:
                continue

            try:
                userdata = userdata_blob.decode("utf-16-le")
            except (UnicodeDecodeError, AttributeError):
                continue

            impl_match = re.search(r"<_Impl>(.*?)</_Impl>", userdata, re.DOTALL)
            if not impl_match:
                continue

            try:
                xml_text = html.unescape(impl_match.group(1))
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                continue

            def _get(tag, default=""):
                el = root.find(tag)
                return el.text if el is not None and el.text else default

            # Extract session times (datetime strings → HH:MM)
            session_start = _get("SESSION_START")
            session_end = _get("SESSION_END")
            try:
                session_start = session_start.split("T")[1][:5] if "T" in session_start else session_start
            except (IndexError, AttributeError):
                session_start = ""
            try:
                session_end = session_end.split("T")[1][:5] if "T" in session_end else session_end
            except (IndexError, AttributeError):
                session_end = ""

            # Build timeframe string
            ltf = _get("LOWER_TIMEFRAME", "")
            htf = _get("HIGHER_TIMEFRAME", "")
            htf_type = _get("HIGHER_TIMEFRAME_PERIOD_TYPE", "Minute")
            unit = "min" if htf_type == "Minute" else htf_type
            timeframe = ""
            if ltf and htf:
                timeframe = f"{ltf} / {htf} {unit}"
            elif ltf:
                timeframe = f"{ltf} {unit}"

            # Short classname: strip namespace
            short_class = classname.rsplit(".", 1)[-1] if classname else ""

            configs.append({
                "name": name or "",
                "className": short_class,
                "account": account or "",
                "instrument": _get("InstrumentOrInstrumentList"),
                "timeframe": timeframe,
                "allowLongs": _get("ALLOW_LONGS", "true") == "true",
                "allowShorts": _get("ALLOW_SHORTS", "true") == "true",
                "sessionStart": session_start,
                "sessionEnd": session_end,
                "qty": int(_get("FIRST_TARGET_CONTRACT_SIZE", "0") or "0")
                    or int(_get("DefaultQuantity", "1") or "1"),
                "isTerminal": bool(is_terminal),
            })

        configs.sort(key=lambda c: c["name"])
        print(f"  Strategy configs: {len(configs)} NinjaScript strategies found")
        return configs

    finally:
        conn.close()


def read_nt_trades(db_path: str = DEFAULT_NT_DB,
                   accounts: str = "Sim*") -> list[dict]:
    """Read NinjaTrader SQLite database and reconstruct trades.

    Args:
        db_path: Path to NinjaTrader.sqlite
        accounts: fnmatch pattern for account filtering (default: "Sim*")

    Returns:
        List of trade dicts matching the standard schema, sorted by exitTime
    """
    print(f"  Reading NinjaTrader database: {db_path}")
    print(f"  Account filter: {accounts}")

    # Verify the file exists before attempting connection
    db_file = Path(db_path)
    if not db_file.exists():
        print(f"  ERROR: Database file not found: {db_path}")
        return []

    # Build a proper file URI — forward slashes, URL-encode spaces/special chars
    # so the SQLite URI parser handles paths like "NinjaTrader 8" correctly
    uri_path = quote(str(db_file.resolve()).replace("\\", "/"), safe="/:")
    nt_conn = sqlite3.connect(f"file:{uri_path}?immutable=1", uri=True)

    try:
        fills = _read_fills(nt_conn, accounts)
    finally:
        nt_conn.close()

    if not fills:
        print("  No fills found matching account filter.")
        return []

    # Infer Buy/Sell for fills missing OrderAction
    _infer_actions(fills)

    # Remove fills that couldn't be resolved
    fills = [f for f in fills if not f.get("_skip")]

    print(f"  {len(fills):,} fills for {accounts}")

    # Group by (account, instrument)
    groups = defaultdict(list)
    for fill in fills:
        key = (fill["account"], fill["instrument_base"])
        groups[key].append(fill)

    print(f"  {len(groups)} (account, instrument) groups")

    # Reconstruct trades using flat-to-flat position tracking
    all_trades = []
    trade_id = 1
    imbalanced = 0
    incomplete = 0

    for (account, instrument_base), group_fills in sorted(groups.items()):
        # Already sorted by time from SQL ORDER BY
        current_pos = 0
        current_fills = []
        instrument_full = instrument_base

        for fill in group_fills:
            delta = fill["qty"] if fill["action"] == "Buy" else -fill["qty"]
            new_pos = current_pos + delta

            if current_pos == 0 and new_pos != 0:
                # flat → position: start new trade
                current_fills = [fill]
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
                        buy_q = sum(f["qty"] for f in current_fills
                                    if f["action"] == "Buy")
                        sell_q = sum(f["qty"] for f in current_fills
                                     if f["action"] == "Sell")
                        if buy_q != sell_q:
                            imbalanced += 1

                        # Derive strategy from account name only
                        strategy, sub_strategy = _derive_strategy(account)
                        trade["strategy"] = strategy
                        trade["subStrategy"] = sub_strategy

                        # Extract MAE/MFE from MaxPrice/MinPrice
                        entry_fills = [f for f in current_fills
                                       if f["ex"] == "Entry"]
                        if entry_fills:
                            mae, mfe = _extract_mae_mfe(
                                current_fills, trade["direction"],
                                trade["entryPrice"], instrument_base
                            )
                            trade["mae"] = mae
                            trade["mfe"] = mfe

                        all_trades.append(trade)
                        trade_id += 1
                    current_fills = []

            current_pos = new_pos

        # Discard incomplete trade at end of group
        if current_pos != 0 and current_fills:
            incomplete += 1

    # Sort by exit time
    all_trades.sort(key=lambda t: t["exitTime"])

    # Print summary by account
    account_counts = defaultdict(int)
    for t in all_trades:
        account_counts[t["subStrategy"]] += 1

    print(f"\n  Reconstructed {len(all_trades):,} trades:")
    for acct in sorted(account_counts.keys()):
        print(f"    {acct:30s}  {account_counts[acct]:>4} trades")

    if imbalanced:
        print(f"  WARNING: {imbalanced} trades with buy/sell qty imbalance")
    if incomplete:
        print(f"  Discarded {incomplete} incomplete trades (position not flat at end)")

    return all_trades
