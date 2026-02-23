/* Formatting and color utility functions */

function getCSSVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function formatCurrency(val) {
    if (val == null) return '-';
    const abs = Math.abs(val);
    const formatted = abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return val < 0 ? `-$${formatted}` : `$${formatted}`;
}

function formatCurrencyShort(val) {
    if (val == null) return '-';
    const abs = Math.abs(val);
    let formatted;
    if (abs >= 1000) {
        formatted = (abs / 1000).toFixed(1) + 'k';
    } else {
        formatted = abs.toFixed(0);
    }
    return val < 0 ? `-$${formatted}` : `$${formatted}`;
}

function formatPercent(val) {
    if (val == null) return '-';
    return val.toFixed(1) + '%';
}

function formatNumber(val, decimals = 2) {
    if (val == null) return '-';
    return val.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function profitColor(val) {
    if (val > 0) return getComputedStyle(document.documentElement).getPropertyValue('--profit').trim();
    if (val < 0) return getComputedStyle(document.documentElement).getPropertyValue('--loss').trim();
    return getComputedStyle(document.documentElement).getPropertyValue('--neutral').trim();
}

function profitClass(val) {
    if (val > 0) return 'pnl-positive';
    if (val < 0) return 'pnl-negative';
    return 'pnl-zero';
}

function profitTextClass(val) {
    if (val > 0) return 'text-profit';
    if (val < 0) return 'text-loss';
    return 'text-neutral';
}

function barColors(values) {
    const profit = getCSSVar('--profit');
    const loss = getCSSVar('--loss');
    return values.map(v => v >= 0 ? profit : loss);
}

function formatDate(iso) {
    const d = new Date(iso);
    return `${d.getMonth() + 1}/${d.getDate()}`;
}

function formatDateFull(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatTime(iso) {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', second: '2-digit' });
}

function formatDateTime(iso) {
    return formatDateFull(iso) + ' ' + formatTime(iso);
}

function formatHalfHourLabel(hh) {
    // hh is "HH:MM" like "09:00" or "14:30"
    const [hourStr, minStr] = hh.split(':');
    const hr = parseInt(hourStr);
    const min = minStr || '00';
    const suffix = hr < 12 ? 'A' : 'P';
    const display = hr === 0 ? 12 : hr > 12 ? hr - 12 : hr;
    return `${display}:${min}${suffix}`;
}

function heatmapColor(val, maxAbs) {
    if (val === 0 || maxAbs === 0) return 'transparent';
    const intensity = Math.min(Math.abs(val) / maxAbs, 1);
    const alpha = 0.15 + intensity * 0.65;
    return val > 0
        ? `rgba(34, 197, 94, ${alpha})`
        : `rgba(239, 68, 68, ${alpha})`;
}

function pfColor(val) {
    if (val >= 1.5) return getCSSVar('--profit');
    if (val >= 1.0) return '#eab308';
    return getCSSVar('--loss');
}

function sharpeColor(val) {
    if (val == null) return getCSSVar('--neutral');
    if (val >= 1.0) return getCSSVar('--profit');
    if (val >= 0) return '#eab308';
    return getCSSVar('--loss');
}
