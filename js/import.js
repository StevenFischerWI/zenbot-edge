/* Browser-based CSV import for NinjaTrader and thinkorswim trade exports */

// ============================================================
// Futures point-value multipliers (per full point move)
// ============================================================

const FUTURES_POINT_VALUES = {
    '/ES': 50, '/MES': 5,
    '/NQ': 20, '/MNQ': 2,
    '/RTY': 50, '/M2K': 5,
    '/YM': 5,  '/MYM': 0.5,
    '/CL': 1000, '/MCL': 100,
    '/GC': 100, '/MGC': 10,
    '/SI': 5000, '/SIL': 1000,
    '/HG': 25000,
    '/NG': 10000, '/QM': 5000,
    '/ZB': 1000, '/ZN': 1000, '/ZF': 1000, '/ZT': 2000,
    '/6E': 125000, '/6J': 12500000, '/6B': 62500, '/6A': 100000,
    '/HE': 400, '/LE': 400,
    '/ZC': 50, '/ZS': 50, '/ZW': 50,
};

function getFuturesRoot(symbol) {
    // Strip month+year suffix: /ESZ25 → /ES, /MESH25 → /MES, /NQZ5 → /NQ
    return symbol.replace(/[FGHJKMNQUVXZ]\d{1,2}$/, '');
}

function getPointValue(symbol) {
    const root = getFuturesRoot(symbol);
    return FUTURES_POINT_VALUES[root] || 50; // default to 50 if unknown
}

// ============================================================
// CSV Parsing
// ============================================================

function parseCSVLine(line) {
    const fields = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (inQuotes) {
            if (ch === '"' && i + 1 < line.length && line[i + 1] === '"') {
                current += '"';
                i++;
            } else if (ch === '"') {
                inQuotes = false;
            } else {
                current += ch;
            }
        } else {
            if (ch === '"') {
                inQuotes = true;
            } else if (ch === ',') {
                fields.push(current);
                current = '';
            } else {
                current += ch;
            }
        }
    }
    fields.push(current);
    return fields;
}

function parseCurrency(val) {
    val = val.trim();
    if (!val) return 0;
    const negative = val.startsWith('(') || val.startsWith('-');
    let cleaned = val.replace(/[$,()]/g, '');
    if (cleaned.startsWith('-')) cleaned = cleaned.slice(1);
    if (!cleaned) return 0;
    const amount = parseFloat(cleaned);
    return isNaN(amount) ? 0 : (negative ? -amount : amount);
}

function parseNTDateTime(val) {
    val = val.trim();
    if (val.includes('AM') || val.includes('PM')) {
        // "1/2/2026 7:00:00 AM"
        const [datePart, timePart, ampm] = val.split(' ');
        const [month, day, year] = datePart.split('/').map(Number);
        let [hours, minutes, seconds] = timePart.split(':').map(Number);
        if (ampm === 'PM' && hours !== 12) hours += 12;
        if (ampm === 'AM' && hours === 12) hours = 0;
        return new Date(year, month - 1, day, hours, minutes, seconds);
    }
    // "2026-01-02 07:00:00"
    const [datePart, timePart] = val.split(' ');
    const [year, month, day] = datePart.split('-').map(Number);
    const [hours, minutes, seconds] = timePart.split(':').map(Number);
    return new Date(year, month - 1, day, hours, minutes, seconds);
}

function normalizeInstrument(fullName) {
    return fullName.trim().split(' ')[0];
}

function formatISOLocal(d) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function formatDateOnly(d) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/**
 * Parse a NinjaTrader CSV export into an array of trade objects.
 * Mirrors process_trades.read_trades() but accepts ALL accounts.
 */
function parseNinjaTraderCSV(csvText) {
    const lines = csvText.split(/\r?\n/);
    if (lines.length < 2) return [];

    // Skip header
    const rawRows = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const row = parseCSVLine(line);
        if (row.length < 23) continue;

        const account = row[2].trim();
        if (!account) continue;

        let entryTime, exitTime;
        try {
            entryTime = parseNTDateTime(row[8]);
            exitTime = parseNTDateTime(row[9]);
        } catch (e) {
            continue;
        }
        if (isNaN(entryTime.getTime()) || isNaN(exitTime.getTime())) continue;

        rawRows.push({ row, entryTime, exitTime });
    }

    // Group by (account, direction, entryTime, exitTime) to consolidate multi-fill rows
    const groups = {};
    for (const item of rawRows) {
        const key = `${item.row[2].trim()}|${item.row[4].trim()}|${item.row[8].trim()}|${item.row[9].trim()}`;
        if (!groups[key]) groups[key] = [];
        groups[key].push(item);
    }

    const trades = [];
    for (const key of Object.keys(groups)) {
        const groupRows = groups[key];
        const first = groupRows[0];
        const { row: firstRow, entryTime, exitTime } = first;
        const account = firstRow[2].trim();
        const holdingSeconds = (exitTime - entryTime) / 1000;

        // Sum across all contracts in this fill
        let totalQty = 0, totalProfit = 0, totalCommission = 0;
        let totalMAE = 0, totalMFE = 0, totalETD = 0, maxBars = 0;

        for (const item of groupRows) {
            const r = item.row;
            totalQty += parseInt(r[5].trim()) || 0;
            totalProfit += parseCurrency(r[12]);
            totalCommission += parseCurrency(r[14]);
            totalMAE += parseCurrency(r[19]);
            totalMFE += parseCurrency(r[20]);
            totalETD += parseCurrency(r[21]);
            const bars = parseInt(r[22].trim()) || 0;
            if (bars > maxBars) maxBars = bars;
        }

        // JS weekday: 0=Sun..6=Sat → convert to Python convention: 0=Mon..6=Sun
        const jsDay = entryTime.getDay(); // 0=Sun
        const pyDay = jsDay === 0 ? 6 : jsDay - 1; // 0=Mon..6=Sun

        const trade = {
            id: parseInt(firstRow[0].trim()) || 0,
            instrument: normalizeInstrument(firstRow[1]),
            instrumentFull: firstRow[1].trim(),
            strategy: account,
            subStrategy: account,
            direction: firstRow[4].trim(),
            qty: totalQty,
            entryPrice: parseFloat(firstRow[6].trim()) || 0,
            exitPrice: parseFloat(firstRow[7].trim()) || 0,
            entryTime: formatISOLocal(entryTime),
            exitTime: formatISOLocal(exitTime),
            entryName: firstRow[10].trim(),
            exitName: firstRow[11].trim(),
            profit: Math.round(totalProfit * 100) / 100,
            commission: Math.round(totalCommission * 100) / 100,
            mae: Math.round(totalMAE * 100) / 100,
            mfe: Math.round(totalMFE * 100) / 100,
            etd: Math.round(totalETD * 100) / 100,
            bars: maxBars,
            holdingMinutes: Math.round((holdingSeconds / 60) * 100) / 100,
            entryHour: entryTime.getHours(),
            entryHalfHour: String(entryTime.getHours()).padStart(2, '0') + ':' + (entryTime.getMinutes() < 30 ? '00' : '30'),
            entryDayOfWeek: pyDay,
            entryDate: formatDateOnly(entryTime),
        };
        trades.push(trade);
    }

    // Sort by exitTime
    trades.sort((a, b) => a.exitTime.localeCompare(b.exitTime));

    // Assign sequential IDs if originals were 0 or duplicated
    for (let i = 0; i < trades.length; i++) {
        trades[i].id = i + 1;
    }

    return trades;
}

// ============================================================
// thinkorswim (Schwab) CSV Parser
// ============================================================

/**
 * Parse a thinkorswim Account Statement CSV into an array of trade objects.
 * thinkorswim exports individual fills; we FIFO-match them into round-trip trades.
 */
function parseSchwabCSV(csvText) {
    const lines = csvText.split(/\r?\n/);

    // --- Find all Account Trade History sections (one per account) ---
    const sections = []; // { account, headerIdx }
    let lastAccount = 'Default';

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Look for "Account Statement for XXXX" to capture account name
        const acctMatch = line.match(/Account Statement for\s+(.+)/i);
        if (acctMatch) {
            lastAccount = acctMatch[1].trim().replace(/,+$/, '');
            continue;
        }

        if (line.indexOf('Account Trade History') !== -1) {
            // Next non-blank line is column headers
            for (let j = i + 1; j < lines.length; j++) {
                if (lines[j].trim()) {
                    sections.push({ account: lastAccount, headerIdx: j });
                    break;
                }
            }
        }
    }
    if (sections.length === 0) return [];

    // --- Parse fills from each section ---
    const allFills = [];

    for (const section of sections) {
        const { account, headerIdx } = section;

        // Parse column headers
        const headers = parseCSVLine(lines[headerIdx]).map(h => h.trim());
        const col = (name) => headers.indexOf(name);
        const iExecTime = col('Exec Time');
        const iSide = col('Side');
        const iQty = col('Qty');
        const iPosEffect = col('Pos Effect');
        const iSymbol = col('Symbol');
        const iPrice = col('Price');

        if (iSide === -1 || iQty === -1 || iSymbol === -1 || iPrice === -1) continue;

        let lastExecTime = null;

        for (let i = headerIdx + 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const row = parseCSVLine(line);
            // Stop at next section header (fewer fields or doesn't look like data)
            if (row.length < headers.length - 1) break;

            // Exec Time: forward-fill for combo/spread legs
            let execTimeStr = (iExecTime !== -1 && row[iExecTime]) ? row[iExecTime].trim() : '';
            if (execTimeStr) {
                lastExecTime = execTimeStr;
            } else {
                execTimeStr = lastExecTime;
            }
            if (!execTimeStr) continue;

            const symbol = (row[iSymbol] || '').trim();
            // Filter to futures only (starts with /)
            if (!symbol.startsWith('/')) continue;

            const side = (row[iSide] || '').trim().toUpperCase();
            if (side !== 'BUY' && side !== 'SELL') continue;

            const qty = parseInt((row[iQty] || '').trim()) || 0;
            if (qty <= 0) continue;

            const posEffect = (row[iPosEffect] || '').trim().toUpperCase();
            const price = parseFloat((row[iPrice] || '').trim()) || 0;

            const execTime = parseSchwabDateTime(execTimeStr);
            if (!execTime || isNaN(execTime.getTime())) continue;

            allFills.push({ execTime, side, qty, posEffect, symbol, price, account });
        }
    }

    // --- FIFO match fills into round-trip trades ---
    // Group fills by account + root symbol (FIFO queues are per-account per-symbol)
    const fillsByKey = {};
    for (const fill of allFills) {
        const root = getFuturesRoot(fill.symbol);
        const key = fill.account + '|' + root;
        if (!fillsByKey[key]) fillsByKey[key] = [];
        fillsByKey[key].push(fill);
    }

    const trades = [];

    for (const key of Object.keys(fillsByKey)) {
        const keyFills = fillsByKey[key];
        const account = keyFills[0].account;
        const root = getFuturesRoot(keyFills[0].symbol);
        const longQueue = [];
        const shortQueue = [];

        for (const fill of keyFills) {
            if (fill.side === 'BUY' && fill.posEffect === 'TO OPEN') {
                longQueue.push({ ...fill });
            } else if (fill.side === 'SELL' && fill.posEffect === 'TO OPEN') {
                shortQueue.push({ ...fill });
            } else if (fill.side === 'SELL' && fill.posEffect === 'TO CLOSE') {
                let remaining = fill.qty;
                while (remaining > 0 && longQueue.length > 0) {
                    const open = longQueue[0];
                    const matchQty = Math.min(remaining, open.qty);
                    trades.push(buildSchwabTrade(open, fill, matchQty, 'Long', root, account));
                    remaining -= matchQty;
                    open.qty -= matchQty;
                    if (open.qty <= 0) longQueue.shift();
                }
            } else if (fill.side === 'BUY' && fill.posEffect === 'TO CLOSE') {
                let remaining = fill.qty;
                while (remaining > 0 && shortQueue.length > 0) {
                    const open = shortQueue[0];
                    const matchQty = Math.min(remaining, open.qty);
                    trades.push(buildSchwabTrade(open, fill, matchQty, 'Short', root, account));
                    remaining -= matchQty;
                    open.qty -= matchQty;
                    if (open.qty <= 0) shortQueue.shift();
                }
            }
        }
    }

    // Sort by exitTime, assign sequential IDs
    trades.sort((a, b) => a.exitTime.localeCompare(b.exitTime));
    for (let i = 0; i < trades.length; i++) {
        trades[i].id = i + 1;
    }

    return trades;
}

function parseSchwabDateTime(val) {
    // M/d/yy HH:mm:ss — e.g., "1/15/25 10:30:45"
    val = val.trim();
    const parts = val.split(' ');
    if (parts.length < 2) return null;
    const dateParts = parts[0].split('/');
    if (dateParts.length < 3) return null;
    const month = parseInt(dateParts[0]);
    const day = parseInt(dateParts[1]);
    let year = parseInt(dateParts[2]);
    if (year < 100) year += 2000;
    const timeParts = parts[1].split(':');
    const hours = parseInt(timeParts[0]) || 0;
    const minutes = parseInt(timeParts[1]) || 0;
    const seconds = parseInt(timeParts[2]) || 0;
    return new Date(year, month - 1, day, hours, minutes, seconds);
}

function buildSchwabTrade(openFill, closeFill, qty, direction, root, account) {
    const entryTime = openFill.execTime;
    const exitTime = closeFill.execTime;
    const entryPrice = openFill.price;
    const exitPrice = closeFill.price;
    const pointValue = getPointValue(openFill.symbol);

    let profit;
    if (direction === 'Long') {
        profit = (exitPrice - entryPrice) * qty * pointValue;
    } else {
        profit = (entryPrice - exitPrice) * qty * pointValue;
    }
    profit = Math.round(profit * 100) / 100;

    const holdingSeconds = (exitTime - entryTime) / 1000;
    const holdingMinutes = Math.round((holdingSeconds / 60) * 100) / 100;

    // JS weekday: 0=Sun..6=Sat → Python convention: 0=Mon..6=Sun
    const jsDay = entryTime.getDay();
    const pyDay = jsDay === 0 ? 6 : jsDay - 1;

    // Strategy = account, subStrategy = root symbol (e.g., "ES", "NQ")
    const subStrategy = root.replace(/^\//, '');

    return {
        id: 0, // assigned later
        instrument: root,
        instrumentFull: openFill.symbol,
        strategy: account,
        subStrategy: subStrategy,
        direction: direction,
        qty: qty,
        entryPrice: entryPrice,
        exitPrice: exitPrice,
        entryTime: formatISOLocal(entryTime),
        exitTime: formatISOLocal(exitTime),
        entryName: 'Market',
        exitName: 'Market',
        profit: profit,
        commission: 0,
        mae: 0,
        mfe: 0,
        etd: 0,
        bars: 0,
        holdingMinutes: holdingMinutes,
        entryHour: entryTime.getHours(),
        entryHalfHour: String(entryTime.getHours()).padStart(2, '0') + ':' + (entryTime.getMinutes() < 30 ? '00' : '30'),
        entryDayOfWeek: pyDay,
        entryDate: formatDateOnly(entryTime),
    };
}

// ============================================================
// Build TRADE_DATA structure from parsed trades
// ============================================================

function buildTradeData(trades) {
    if (!trades || trades.length === 0) return null;

    // Extract unique values
    const strategySet = new Set();
    const instrumentSet = new Set();
    const dateSet = new Set();

    for (const t of trades) {
        strategySet.add(t.strategy);
        instrumentSet.add(t.instrument);
        dateSet.add(t.entryDate);
    }

    const strategiesSorted = [...strategySet].sort();
    const instrumentsSorted = [...instrumentSet].sort();
    const allDates = [...dateSet].sort();

    // Group trades by strategy
    const byFamily = {};
    for (const t of trades) {
        if (!byFamily[t.strategy]) byFamily[t.strategy] = [];
        byFamily[t.strategy].push(t);
    }

    // Compute per-strategy metrics using existing computeMetrics + sub-strategy summaries
    const strategyMetrics = {};
    for (const name of strategiesSorted) {
        const familyTrades = byFamily[name];
        const metrics = computeMetrics(familyTrades, name);
        metrics.subStrategies = computeSubStrategySummaries(familyTrades);
        strategyMetrics[name] = metrics;
    }

    // Compute _ALL aggregate
    const allMetrics = computeMetrics(trades, 'All Strategies');
    allMetrics.subStrategies = [];
    for (const name of strategiesSorted) {
        const m = strategyMetrics[name];
        allMetrics.subStrategies.push({
            name: name,
            trades: m.tradeCount,
            winRate: m.winRate,
            totalPnL: m.totalPnL,
            avgTrade: m.avgTrade,
            avgWin: m.avgWin,
            avgLoss: m.avgLoss,
            profitFactor: m.profitFactor,
            maxDrawdown: m.maxDrawdown,
        });
    }
    strategyMetrics['_ALL'] = allMetrics;

    return {
        metadata: {
            generated: new Date().toISOString(),
            sourceFile: 'CSV Import',
            totalTradesRaw: trades.length,
            totalTradesFiltered: trades.length,
            dateRange: {
                start: allDates[0],
                end: allDates[allDates.length - 1],
            },
            tradingDays: allDates.length,
            strategies: strategiesSorted,
            instruments: instrumentsSorted,
        },
        strategies: strategyMetrics,
        trades: trades,
    };
}

// ============================================================
// Import handler
// ============================================================

function handleImport(file) {
    const statusEl = document.getElementById('import-status');

    if (!file || !file.name.toLowerCase().endsWith('.csv')) {
        if (statusEl) {
            statusEl.textContent = 'Please select a .csv file.';
            statusEl.className = 'import-status error';
        }
        return;
    }

    if (statusEl) {
        statusEl.textContent = 'Reading file...';
        statusEl.className = 'import-status';
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        try {
            const text = e.target.result;
            if (statusEl) statusEl.textContent = 'Parsing trades...';

            // Auto-detect format: thinkorswim vs NinjaTrader
            let trades;
            if (text.indexOf('Account Trade History') !== -1) {
                trades = parseSchwabCSV(text);
            } else {
                trades = parseNinjaTraderCSV(text);
            }

            if (trades.length === 0) {
                if (statusEl) {
                    statusEl.textContent = 'No valid trades found. Make sure this is a NinjaTrader or thinkorswim trade export CSV.';
                    statusEl.className = 'import-status error';
                }
                return;
            }

            if (statusEl) statusEl.textContent = `Found ${trades.length} trades. Building dashboard...`;

            const data = buildTradeData(trades);
            if (!data) {
                if (statusEl) {
                    statusEl.textContent = 'Failed to build trade data.';
                    statusEl.className = 'import-status error';
                }
                return;
            }

            // Set as global and re-init
            window.TRADE_DATA = data;
            initFromImport();
        } catch (err) {
            console.error('Import error:', err);
            if (statusEl) {
                statusEl.textContent = 'Error parsing CSV: ' + err.message;
                statusEl.className = 'import-status error';
            }
        }
    };

    reader.onerror = function () {
        if (statusEl) {
            statusEl.textContent = 'Error reading file.';
            statusEl.className = 'import-status error';
        }
    };

    reader.readAsText(file);
}
