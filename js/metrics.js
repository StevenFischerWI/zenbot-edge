/* Client-side metrics computation, trade filtering, and trade log rendering */

const DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const TRADES_PER_PAGE = 50;

let tradeLogState = {
    page: 1,
    sortCol: 'exitTime',
    sortDir: 'desc',
    filterVariant: '',
    filteredTrades: [],
};

// ============================================================
// Client-side metrics computation (mirrors Python compute_metrics)
// ============================================================

function computeMetrics(trades, name) {
    if (!trades || trades.length === 0) {
        return { strategyName: name, tradeCount: 0, subStrategies: [] };
    }

    const profits = trades.map(t => t.profit);
    const wins = profits.filter(p => p > 0);
    const losses = profits.filter(p => p < 0);
    const beCount = profits.filter(p => p === 0).length;
    const winCount = wins.length;
    const lossCount = losses.length;
    const decisions = winCount + lossCount;
    const total = trades.length;

    const sum = arr => arr.reduce((a, b) => a + b, 0);
    const mean = arr => arr.length ? sum(arr) / arr.length : 0;
    const r2 = v => Math.round(v * 100) / 100;

    const winRate = decisions > 0 ? r2((winCount / decisions) * 100) : 0;
    const totalPnL = r2(sum(profits));
    const avgWin = r2(mean(wins));
    const avgLoss = r2(mean(losses));
    const maxWin = wins.length ? r2(Math.max(...wins)) : 0;
    const maxLoss = losses.length ? r2(Math.min(...losses)) : 0;
    const avgTrade = r2(mean(profits));

    const grossProfit = sum(wins);
    const grossLoss = Math.abs(sum(losses));
    const profitFactor = grossLoss > 0 ? r2(grossProfit / grossLoss) : (grossProfit > 0 ? 9999.99 : 0);

    // Equity curve & max drawdown
    const equityCurve = [];
    let equity = 0, peak = 0, maxDD = 0;
    for (let i = 0; i < trades.length; i++) {
        equity = r2(equity + trades[i].profit);
        equityCurve.push([i + 1, equity]);
        peak = Math.max(peak, equity);
        maxDD = Math.min(maxDD, equity - peak);
    }
    maxDD = r2(maxDD);

    // Daily P&L
    const dailyMap = {};
    for (const t of trades) {
        const d = t.entryDate;
        if (!dailyMap[d]) dailyMap[d] = { pnl: 0, trades: 0, wins: 0 };
        dailyMap[d].pnl += t.profit;
        dailyMap[d].trades++;
        if (t.profit > 0) dailyMap[d].wins++;
    }
    const dailyPnL = Object.keys(dailyMap).sort().map(d => ({
        date: d, pnl: r2(dailyMap[d].pnl), trades: dailyMap[d].trades, wins: dailyMap[d].wins
    }));

    // Sharpe (daily, annualized)
    let sharpeRatio = null;
    const dailyReturns = dailyPnL.map(d => d.pnl);
    if (dailyReturns.length >= 10) {
        const mr = mean(dailyReturns);
        const variance = dailyReturns.reduce((s, v) => s + (v - mr) ** 2, 0) / (dailyReturns.length - 1);
        const std = Math.sqrt(variance);
        sharpeRatio = std > 0 ? r2((mr / std) * Math.sqrt(252)) : 0;
    }

    // Profit distribution
    const binSize = 50;
    const minP = Math.max(Math.min(...profits), -2000);
    const maxP = Math.min(Math.max(...profits), 2000);
    const binStart = Math.floor(minP / binSize) * binSize;
    const binEnd = (Math.floor(maxP / binSize) + 1) * binSize;
    const bins = [];
    for (let b = binStart; b <= binEnd; b += binSize) bins.push(b);
    const counts = new Array(bins.length - 1).fill(0);
    for (const p of profits) {
        const clamped = Math.max(minP, Math.min(maxP, p));
        let idx = Math.floor((clamped - binStart) / binSize);
        idx = Math.max(0, Math.min(idx, counts.length - 1));
        counts[idx]++;
    }

    // Long vs Short
    const longs = trades.filter(t => t.direction === 'Long');
    const shorts = trades.filter(t => t.direction === 'Short');
    const longWins = longs.filter(t => t.profit > 0).length;
    const shortWins = shorts.filter(t => t.profit > 0).length;
    const longDec = longs.filter(t => t.profit !== 0).length;
    const shortDec = shorts.filter(t => t.profit !== 0).length;
    const longShort = {
        longCount: longs.length, shortCount: shorts.length,
        longWinRate: longDec > 0 ? r2((longWins / longDec) * 100) : 0,
        shortWinRate: shortDec > 0 ? r2((shortWins / shortDec) * 100) : 0,
        longPnL: r2(sum(longs.map(t => t.profit))),
        shortPnL: r2(sum(shorts.map(t => t.profit))),
        longAvg: longs.length ? r2(mean(longs.map(t => t.profit))) : 0,
        shortAvg: shorts.length ? r2(mean(shorts.map(t => t.profit))) : 0,
    };

    // Half-hour P&L, win rate, trade count
    const hourlyMap = {};
    for (const t of trades) {
        const h = t.entryHalfHour || (String(t.entryHour).padStart(2, '0') + ':00');
        if (!hourlyMap[h]) hourlyMap[h] = { pnl: 0, trades: 0, wins: 0, decisions: 0 };
        hourlyMap[h].pnl += t.profit;
        hourlyMap[h].trades++;
        if (t.profit > 0) hourlyMap[h].wins++;
        if (t.profit !== 0) hourlyMap[h].decisions++;
    }
    const hourlyPnL = {}, hourlyWinRate = {}, hourlyTradeCount = {};
    for (const h of Object.keys(hourlyMap).sort()) {
        const v = hourlyMap[h];
        hourlyPnL[h] = r2(v.pnl);
        hourlyTradeCount[h] = v.trades;
        hourlyWinRate[h] = v.decisions > 0 ? r2((v.wins / v.decisions) * 100) : 0;
    }

    // Day-of-week P&L
    const dowMap = {};
    for (const t of trades) {
        const d = DAY_NAMES[t.entryDayOfWeek];
        if (!dowMap[d]) dowMap[d] = { pnl: 0, trades: 0, wins: 0, decisions: 0 };
        dowMap[d].pnl += t.profit;
        dowMap[d].trades++;
        if (t.profit > 0) dowMap[d].wins++;
        if (t.profit !== 0) dowMap[d].decisions++;
    }
    const dowPnL = {}, dowTradeCount = {}, dowWinRate = {};
    for (const d of DAY_NAMES.slice(0, 5)) {
        if (dowMap[d]) {
            dowPnL[d] = r2(dowMap[d].pnl);
            dowTradeCount[d] = dowMap[d].trades;
            dowWinRate[d] = dowMap[d].decisions > 0 ? r2((dowMap[d].wins / dowMap[d].decisions) * 100) : 0;
        }
    }

    // Half-hour x Day matrix
    const hourDayMatrix = {};
    for (const d of DAY_NAMES.slice(0, 5)) hourDayMatrix[d] = {};
    for (const t of trades) {
        const d = DAY_NAMES[t.entryDayOfWeek];
        const h = t.entryHalfHour || (String(t.entryHour).padStart(2, '0') + ':00');
        if (hourDayMatrix[d]) {
            hourDayMatrix[d][h] = r2((hourDayMatrix[d][h] || 0) + t.profit);
        }
    }

    // Instrument breakdown
    const instMap = {};
    for (const t of trades) {
        const i = t.instrument;
        if (!instMap[i]) instMap[i] = { pnl: 0, trades: 0, wins: 0, losses: 0, winP: [], lossP: [] };
        instMap[i].pnl += t.profit;
        instMap[i].trades++;
        if (t.profit > 0) { instMap[i].wins++; instMap[i].winP.push(t.profit); }
        else if (t.profit < 0) { instMap[i].losses++; instMap[i].lossP.push(t.profit); }
    }
    const instrumentPnL = {}, instrumentTradeCount = {}, instrumentDetails = [];
    for (const iName of Object.keys(instMap).sort()) {
        const v = instMap[iName];
        instrumentPnL[iName] = r2(v.pnl);
        instrumentTradeCount[iName] = v.trades;
        const iDec = v.wins + v.losses;
        const gp = sum(v.winP), gl = Math.abs(sum(v.lossP));
        instrumentDetails.push({
            instrument: iName, trades: v.trades,
            winRate: iDec > 0 ? r2((v.wins / iDec) * 100) : 0,
            avgWin: v.winP.length ? r2(mean(v.winP)) : 0,
            avgLoss: v.lossP.length ? r2(mean(v.lossP)) : 0,
            pnl: r2(v.pnl),
            profitFactor: gl > 0 ? r2(gp / gl) : (gp > 0 ? 9999.99 : 0),
        });
    }

    // MAE/MFE scatter data
    const maeVsProfit = trades.map(t => [t.mae, t.profit]);
    const mfeVsProfit = trades.map(t => [t.mfe, t.profit]);

    // Rolling metrics
    const rollingPnL20 = [];
    const rollingWinRate50 = [];
    if (trades.length >= 20) {
        for (let i = 19; i < trades.length; i++) {
            const window = profits.slice(i - 19, i + 1);
            rollingPnL20.push([i + 1, r2(mean(window))]);
        }
    }
    if (trades.length >= 50) {
        for (let i = 49; i < trades.length; i++) {
            const window = profits.slice(i - 49, i + 1);
            const w = window.filter(p => p > 0).length;
            const d = window.filter(p => p !== 0).length;
            rollingWinRate50.push([i + 1, d > 0 ? r2((w / d) * 100) : 0]);
        }
    }

    // Streaks
    const streaks = [];
    let currentStreak = 0, currentType = null;
    for (const p of profits) {
        const type = p > 0 ? 'win' : p < 0 ? 'loss' : null;
        if (type === null) continue; // breakeven doesn't break streaks
        if (type === currentType) {
            currentStreak++;
        } else {
            if (currentType && currentStreak > 0) streaks.push({ type: currentType, length: currentStreak });
            currentType = type;
            currentStreak = 1;
        }
    }
    if (currentType && currentStreak > 0) streaks.push({ type: currentType, length: currentStreak });
    const winStreaks = streaks.filter(s => s.type === 'win').map(s => s.length);
    const lossStreaks = streaks.filter(s => s.type === 'loss').map(s => s.length);
    const maxConsecWins = winStreaks.length ? Math.max(...winStreaks) : 0;
    const maxConsecLosses = lossStreaks.length ? Math.max(...lossStreaks) : 0;

    // Sub-strategy comparison
    const subStrats = computeSubStrategySummaries(trades);

    return {
        strategyName: name, tradeCount: total,
        winCount, lossCount, breakEvenCount: beCount, winRate,
        totalPnL, avgWin, avgLoss, maxWin, maxLoss, avgTrade,
        profitFactor, sharpeRatio, maxDrawdown: maxDD,
        avgHoldingMinutes: r2(mean(trades.map(t => t.holdingMinutes))),
        maxConsecWins, maxConsecLosses,
        longShort, equityCurve, dailyPnL,
        profitDistribution: { bins, counts },
        hourlyPnL, hourlyWinRate, hourlyTradeCount,
        dowPnL, dowTradeCount, dowWinRate, hourDayMatrix,
        instrumentPnL, instrumentTradeCount, instrumentDetails,
        maeVsProfit, mfeVsProfit,
        rollingPnL20, rollingWinRate50, streaks,
        subStrategies: subStrats,
    };
}

function computeSubStrategySummaries(trades) {
    const byKey = {};
    for (const t of trades) {
        const k = t.subStrategy;
        if (!byKey[k]) byKey[k] = [];
        byKey[k].push(t);
    }
    const summaries = [];
    for (const name of Object.keys(byKey).sort()) {
        const sub = byKey[name];
        const profits = sub.map(t => t.profit);
        const wins = profits.filter(p => p > 0);
        const losses = profits.filter(p => p < 0);
        const w = wins.length, l = losses.length, dec = w + l;
        const sum = arr => arr.reduce((a, b) => a + b, 0);
        const mean = arr => arr.length ? sum(arr) / arr.length : 0;
        const r2 = v => Math.round(v * 100) / 100;
        const gp = sum(wins), gl = Math.abs(sum(losses));

        let equity = 0, peak = 0, maxDD = 0;
        for (const t of sub) {
            equity += t.profit;
            peak = Math.max(peak, equity);
            maxDD = Math.min(maxDD, equity - peak);
        }

        summaries.push({
            name, trades: sub.length,
            winRate: dec > 0 ? r2((w / dec) * 100) : 0,
            totalPnL: r2(sum(profits)),
            avgTrade: r2(mean(profits)),
            avgWin: wins.length ? r2(mean(wins)) : 0,
            avgLoss: losses.length ? r2(mean(losses)) : 0,
            profitFactor: gl > 0 ? r2(gp / gl) : (gp > 0 ? 9999.99 : 0),
            maxDrawdown: r2(maxDD),
        });
    }
    return summaries;
}

// ============================================================
// Global filter: get trades after applying global direction/instrument
// ============================================================

function getGlobalFilteredTrades() {
    let trades = TRADE_DATA.trades;
    if (appState.hideApex) {
        trades = trades.filter(t => !t.strategy.startsWith('APEX'));
    }
    if (appState.globalDirection) {
        trades = trades.filter(t => t.direction === appState.globalDirection);
    }
    if (appState.globalInstruments.size > 0) {
        trades = trades.filter(t => appState.globalInstruments.has(t.instrument));
    }
    if (appState.globalHours.size > 0) {
        trades = trades.filter(t => {
            const hh = t.entryHalfHour || (String(t.entryHour).padStart(2, '0') + ':00');
            return appState.globalHours.has(hh);
        });
    }
    if (appState.globalDateFrom) {
        trades = trades.filter(t => t.entryDate >= appState.globalDateFrom);
    }
    if (appState.globalDateTo) {
        trades = trades.filter(t => t.entryDate <= appState.globalDateTo);
    }
    return trades;
}

function hasAnyGlobalFilter() {
    const dr = appState.data.metadata.dateRange;
    const hasDateFilter = (appState.globalDateFrom && appState.globalDateFrom > dr.start)
        || (appState.globalDateTo && appState.globalDateTo < dr.end);
    return appState.hideApex
        || appState.globalDirection
        || appState.globalInstruments.size > 0
        || appState.globalHours.size > 0
        || hasDateFilter;
}

function getActiveMetrics() {
    const hasGlobalFilter = hasAnyGlobalFilter();
    const sel = appState.selectedStrategies;
    const isAll = sel.has('_ALL');
    const isMulti = !isAll && sel.size > 1;
    const isSingle = !isAll && sel.size === 1;

    if (!hasGlobalFilter && isAll) {
        // No filter, all selected — use pre-computed _ALL
        return appState.data.strategies['_ALL'];
    }

    if (!hasGlobalFilter && isSingle) {
        // No filter, single strategy — use pre-computed
        const key = [...sel][0];
        return appState.data.strategies[key];
    }

    // Multi-select or global filter active — recompute from trades
    let trades = getGlobalFilteredTrades();
    if (!isAll) {
        trades = trades.filter(t => sel.has(t.strategy));
    }
    const label = isAll ? 'All Strategies' : (isSingle ? [...sel][0] : `${sel.size} Strategies`);
    return computeMetrics(trades, label);
}

// For the sidebar, we need per-family trade counts/PnL under global filters
function getSidebarMetrics() {
    if (!hasAnyGlobalFilter()) return null; // use pre-computed

    const trades = getGlobalFilteredTrades();
    const byFamily = {};
    let allPnL = 0, allCount = 0;
    for (const t of trades) {
        if (!byFamily[t.strategy]) byFamily[t.strategy] = { pnl: 0, count: 0, wins: 0, decisions: 0 };
        byFamily[t.strategy].pnl += t.profit;
        byFamily[t.strategy].count++;
        if (t.profit > 0) byFamily[t.strategy].wins++;
        if (t.profit !== 0) byFamily[t.strategy].decisions++;
        allPnL += t.profit;
        allCount++;
    }
    // compute win rate for each
    for (const k of Object.keys(byFamily)) {
        const v = byFamily[k];
        v.winRate = v.decisions > 0 ? Math.round((v.wins / v.decisions) * 1000) / 10 : 0;
    }
    const allDec = trades.filter(t => t.profit !== 0).length;
    const allWins = trades.filter(t => t.profit > 0).length;
    return {
        _ALL: { pnl: Math.round(allPnL * 100) / 100, count: allCount, winRate: allDec > 0 ? Math.round((allWins / allDec) * 1000) / 10 : 0 },
        families: byFamily,
    };
}

// ============================================================
// Trade log (tab-level filters are separate from global)
// ============================================================

function getStrategyTrades(strategyKey) {
    // Start with globally-filtered trades
    let trades = getGlobalFilteredTrades();
    // strategyKey is kept for backward compat but we use selectedStrategies
    const sel = appState.selectedStrategies;
    if (!sel.has('_ALL')) {
        trades = trades.filter(t => sel.has(t.strategy));
    }
    return trades;
}

function applyTradeLogFilters(trades) {
    let result = trades;
    if (tradeLogState.filterVariant) {
        result = result.filter(t => t.subStrategy === tradeLogState.filterVariant);
    }
    return result;
}

function sortTrades(trades) {
    const col = tradeLogState.sortCol;
    const dir = tradeLogState.sortDir === 'asc' ? 1 : -1;
    return [...trades].sort((a, b) => {
        let va = a[col], vb = b[col];
        if (typeof va === 'string') return va.localeCompare(vb) * dir;
        return (va - vb) * dir;
    });
}

function renderTradeLog(containerId) {
    const container = document.getElementById(containerId);
    const allTrades = getStrategyTrades();

    const variants = [...new Set(allTrades.map(t => t.subStrategy))].sort();
    const instruments = [...new Set(allTrades.map(t => t.instrument))].sort();

    // Filters row (only variant — direction & instrument are global now)
    let html = '<div class="filters-row">';
    if (variants.length > 1) {
        html += `<select class="filter-select" id="filter-variant" onchange="onTradeLogFilterChange()">
            <option value="">All Variants</option>
            ${variants.map(v => `<option value="${v}" ${tradeLogState.filterVariant === v ? 'selected' : ''}>${v.replace('Sim-', '')}</option>`).join('')}
        </select>`;
    }
    html += `<span style="color:var(--text-muted);font-size:13px" id="trade-count-label"></span></div>`;

    let filtered = applyTradeLogFilters(allTrades);
    filtered = sortTrades(filtered);
    tradeLogState.filteredTrades = filtered;

    const totalPages = Math.max(1, Math.ceil(filtered.length / TRADES_PER_PAGE));
    tradeLogState.page = Math.min(tradeLogState.page, totalPages);
    const start = (tradeLogState.page - 1) * TRADES_PER_PAGE;
    const pageData = filtered.slice(start, start + TRADES_PER_PAGE);

    const columns = [
        { key: 'id', label: '#', cls: 'num' },
        { key: 'exitTime', label: 'Time', cls: '' },
        { key: 'instrument', label: 'Instrument', cls: '' },
        { key: 'strategy', label: 'Account', cls: '' },
        { key: 'subStrategy', label: 'Account', cls: '' },
        { key: 'direction', label: 'Dir', cls: '' },
        { key: 'qty', label: 'Qty', cls: 'num' },
        { key: 'entryPrice', label: 'Entry', cls: 'num' },
        { key: 'exitPrice', label: 'Exit', cls: 'num' },
        { key: 'exitName', label: 'Exit Type', cls: '' },
        { key: 'profit', label: 'Profit', cls: 'num' },
        { key: 'mae', label: 'MAE', cls: 'num' },
        { key: 'mfe', label: 'MFE', cls: 'num' },
        { key: 'bars', label: 'Bars', cls: 'num' },
    ];

    html += '<div class="data-table-container"><table class="data-table"><thead><tr>';
    columns.forEach(col => {
        const sorted = tradeLogState.sortCol === col.key;
        const cls = sorted ? (tradeLogState.sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc') : '';
        html += `<th class="${col.cls} ${cls}" onclick="onTradeSort('${col.key}')">${col.label}</th>`;
    });
    html += '</tr></thead><tbody>';

    pageData.forEach(t => {
        const dirClass = t.direction === 'Long' ? 'text-accent' : 'text-neutral';
        html += `<tr>
            <td class="num">${t.id}</td>
            <td>${formatDateTime(t.exitTime)}</td>
            <td class="fw-600">${t.instrument}</td>
            <td>${t.strategy}</td>
            <td>${t.subStrategy}</td>
            <td class="${dirClass}">${t.direction}</td>
            <td class="num">${t.qty}</td>
            <td class="num">${formatNumber(t.entryPrice)}</td>
            <td class="num">${formatNumber(t.exitPrice)}</td>
            <td>${t.exitName}</td>
            <td class="num ${profitTextClass(t.profit)} fw-600">${formatCurrency(t.profit)}</td>
            <td class="num text-loss">${formatCurrency(t.mae)}</td>
            <td class="num text-profit">${formatCurrency(t.mfe)}</td>
            <td class="num">${t.bars}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';

    html += `<div class="pagination">
        <span>Showing ${filtered.length === 0 ? 0 : start + 1}-${Math.min(start + TRADES_PER_PAGE, filtered.length)} of ${filtered.length} trades</span>
        <div class="pagination-btns">
            <button class="page-btn" onclick="onPageChange(1)" ${tradeLogState.page <= 1 ? 'disabled' : ''}>&laquo;</button>
            <button class="page-btn" onclick="onPageChange(${tradeLogState.page - 1})" ${tradeLogState.page <= 1 ? 'disabled' : ''}>&lsaquo; Prev</button>
            <span style="padding:6px 12px;color:var(--text-secondary)">Page ${tradeLogState.page} of ${totalPages}</span>
            <button class="page-btn" onclick="onPageChange(${tradeLogState.page + 1})" ${tradeLogState.page >= totalPages ? 'disabled' : ''}>Next &rsaquo;</button>
            <button class="page-btn" onclick="onPageChange(${totalPages})" ${tradeLogState.page >= totalPages ? 'disabled' : ''}>&raquo;</button>
        </div>
    </div>`;

    container.innerHTML = html;
    const label = document.getElementById('trade-count-label');
    if (label) label.textContent = `${filtered.length} trades`;
}

function onTradeLogFilterChange() {
    const variantEl = document.getElementById('filter-variant');
    tradeLogState.filterVariant = variantEl ? variantEl.value : '';
    tradeLogState.page = 1;
    renderTradeLog('tab-trades');
}

function onTradeSort(col) {
    if (tradeLogState.sortCol === col) {
        tradeLogState.sortDir = tradeLogState.sortDir === 'asc' ? 'desc' : 'asc';
    } else {
        tradeLogState.sortCol = col;
        tradeLogState.sortDir = col === 'profit' || col === 'exitTime' ? 'desc' : 'asc';
    }
    renderTradeLog('tab-trades');
}

function onPageChange(page) {
    tradeLogState.page = Math.max(1, page);
    renderTradeLog('tab-trades');
}
