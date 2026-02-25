# ZENBOTEDGE

Futures trading analysis dashboard powered by NinjaTrader data.

## Setup

1. Clone the repo
2. Ensure Python 3.10+ is installed

## Importing Trades from NinjaTrader

The import reads directly from NinjaTrader 8's SQLite database, reconstructs trades from execution fills, and loads them into the dashboard.

### Find your NinjaTrader database

The database is typically at:
```
C:\Users\<username>\Documents\NinjaTrader 8\db\NinjaTrader.sqlite
```

If you use OneDrive, it may be at:
```
C:\Users\<username>\OneDrive\Documents\NinjaTrader 8\db\NinjaTrader.sqlite
```

### Run the import

```powershell
python ingest.py --ninjatrader --nt-accounts "*" --no-regime --nt-db "C:\Users\steve\OneDrive\_old\Documents\NinjaTrader 8\db\NinjaTrader.sqlite"
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--ninjatrader` | Sync trades from NinjaTrader's SQLite database |
| `--nt-db <path>` | Path to `NinjaTrader.sqlite` (required if not at the default location) |
| `--nt-accounts <pattern>` | Account filter pattern. `"*"` for all accounts, `"Sim*"` for sim only (default) |
| `--no-regime` | Skip regime analysis (requires a separate ES indicator export) |

### What it does

1. Opens the NinjaTrader database (read-only, safe to run while NinjaTrader is open)
2. Reads execution fills and reconstructs flat-to-flat trades
3. Computes P&L, MAE/MFE, holding time, and other metrics
4. Deduplicates against previously imported trades
5. Regenerates the dashboard data file

You can run the import repeatedly — it only adds new trades.

### Re-importing from scratch

To clear all NinjaTrader-imported trades and start fresh:

```powershell
# Delete the trade database and re-import
del data\trades.db
python ingest.py --ninjatrader --nt-accounts "*" --no-regime --nt-db "C:\Users\steve\OneDrive\_old\Documents\NinjaTrader 8\db\NinjaTrader.sqlite"
```

## Opening the Dashboard

Open `index.html` in any browser:

```powershell
start index.html
```

No server required — the dashboard runs entirely in the browser.

### Dashboard Features

- **Strategy sidebar** — select/compare strategies, sort by P&L, win rate, or trade count
- **Global filters** — filter by direction (Long/Short), instruments, time of day, date range
- **Instrument presets** — quick-select All, Usual (majors + micros), or Micros only
- **Tabs** — Overview, Calendar, Time Analysis, Instruments, Risk & Efficiency, Trade Log
- **Themes** — toggle light/dark mode

All times are displayed in Eastern (US/Eastern).
