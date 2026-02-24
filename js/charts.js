/* Chart.js rendering factory functions */

// Global Chart.js defaults — theme-aware
function initChartDefaults() {
    Chart.defaults.color = getCSSVar('--text-secondary');
    Chart.defaults.borderColor = getCSSVar('--chart-grid');
    Chart.defaults.font.family = "'Inter', 'Segoe UI', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.display = false;
    // Keep tooltips dark in both themes for readability
    Chart.defaults.plugins.tooltip.backgroundColor = '#1c1f2e';
    Chart.defaults.plugins.tooltip.borderColor = '#2a2d3a';
    Chart.defaults.plugins.tooltip.titleColor = '#e4e4e7';
    Chart.defaults.plugins.tooltip.bodyColor = '#d1d5db';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleFont = { weight: '600' };
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 6;
}

// --- KPI SPARKLINES ---

function renderSparkline(canvasId, data, color, tooltipLabel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i + 1),
            datasets: [{
                data: data,
                borderColor: color,
                backgroundColor: color + '1A',
                borderWidth: 1.5,
                pointRadius: 0,
                pointHitRadius: 10,
                tension: 0.4,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            events: ['mousemove', 'mouseout'],
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    displayColors: false,
                    padding: { top: 4, bottom: 4, left: 8, right: 8 },
                    bodyFont: { size: 11 },
                    callbacks: {
                        title: () => '',
                        label: tooltipLabel || ((ctx) => String(ctx.parsed.y))
                    }
                }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            interaction: { intersect: false, mode: 'nearest' }
        }
    });
}

function downsampleArray(arr, maxPoints) {
    if (arr.length <= maxPoints) return arr;
    const step = Math.ceil(arr.length / maxPoints);
    const result = [];
    for (let i = 0; i < arr.length; i += step) result.push(arr[i]);
    if (result.length === 0 || result[result.length - 1] !== arr[arr.length - 1]) {
        result.push(arr[arr.length - 1]);
    }
    return result;
}

// --- OVERVIEW TAB ---

function renderEquityCurve(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    // Downsample if more than 1000 points
    let points = data;
    let step = 1;
    if (points.length > 1000) {
        step = Math.ceil(points.length / 1000);
        points = points.filter((_, i) => i % step === 0 || i === points.length - 1);
    }
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: points.map(d => d[0]),
            datasets: [{
                label: 'Cumulative P&L',
                data: points.map(d => d[1]),
                borderColor: '#60a5fa',
                backgroundColor: (context) => {
                    const chart = context.chart;
                    const { ctx: c, chartArea } = chart;
                    if (!chartArea) return 'rgba(96,165,250,0.1)';
                    const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                    gradient.addColorStop(0, 'rgba(96,165,250,0.25)');
                    gradient.addColorStop(1, 'rgba(96,165,250,0.02)');
                    return gradient;
                },
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `P&L: ${formatCurrency(ctx.parsed.y)}`,
                        title: (items) => `Trade #${items[0].label}`
                    }
                },
                annotation: {
                    annotations: {
                        zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: getCSSVar('--chart-zero'), borderDash: [5, 5], borderWidth: 1 }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Trade #' }, ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'Cumulative P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
            }
        }
    });
}

function renderDailyPnL(canvasId, dailyData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const values = dailyData.map(d => d.pnl);
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dailyData.map(d => formatDate(d.date)),
            datasets: [{
                label: 'Daily P&L',
                data: values,
                backgroundColor: barColors(values),
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `P&L: ${formatCurrency(ctx.parsed.y)}`,
                        afterLabel: (ctx) => `Trades: ${dailyData[ctx.dataIndex].trades}`
                    }
                }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 15, maxRotation: 45 } },
                y: { title: { display: true, text: 'P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
            }
        }
    });
}

function renderProfitDistribution(canvasId, distData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const { bins, counts } = distData;
    const labels = [];
    for (let i = 0; i < bins.length - 1; i++) {
        labels.push(`${bins[i]} to ${bins[i + 1]}`);
    }
    const midpoints = bins.slice(0, -1).map((b, i) => (b + bins[i + 1]) / 2);
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Trade Count',
                data: counts,
                backgroundColor: midpoints.map(m => m >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'),
                borderRadius: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: { title: (items) => items[0].label, label: (ctx) => `${ctx.parsed.y} trades` }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Profit ($)' }, ticks: { maxTicksLimit: 12, maxRotation: 45 } },
                y: { title: { display: true, text: 'Count' } }
            }
        }
    });
}

function renderLongVsShort(canvasId, lsData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Long', 'Short'],
            datasets: [{
                label: 'Total P&L',
                data: [lsData.longPnL, lsData.shortPnL],
                backgroundColor: [
                    lsData.longPnL >= 0 ? 'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)',
                    lsData.shortPnL >= 0 ? 'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)',
                ],
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => formatCurrency(ctx.parsed.y) } }
            },
            scales: {
                y: { ticks: { callback: (v) => formatCurrency(v) } }
            }
        }
    });
}

function renderLongVsShortWinRate(canvasId, lsData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Long', 'Short'],
            datasets: [{
                label: 'Win Rate',
                data: [lsData.longWinRate, lsData.shortWinRate],
                backgroundColor: ['rgba(96,165,250,0.7)', 'rgba(251,146,60,0.7)'],
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.y.toFixed(1)}%` } }
            },
            scales: {
                y: { ticks: { callback: (v) => `${v}%` }, suggestedMin: 0, suggestedMax: 100 }
            }
        }
    });
}

function onInstrumentBarClick(event, elements, chart) {
    if (!elements.length) return;
    const idx = elements[0].index;
    const instrument = chart.data.labels[idx];
    if (!instrument) return;
    // Set filter to just this instrument
    appState.globalInstruments.clear();
    appState.globalInstruments.add(instrument);
    // Sync checkboxes
    document.querySelectorAll('#instrument-dropdown input[value]').forEach(cb => {
        cb.checked = cb.value === instrument;
    });
    syncAllInstrumentsPreset();
    updateInstrumentBtn();
    saveState();
    applyGlobalFilters();
}

function renderInstrumentPnLBar(canvasId, instPnL, vertical, allInstruments) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const allKeys = allInstruments ? [...allInstruments].sort() : Object.keys(instPnL).sort();
    const labels = allKeys;
    const values = allKeys.map(k => instPnL[k] || 0);
    const clickHandler = { onClick: onInstrumentBarClick, onHover: (e, els) => { e.native.target.style.cursor = els.length ? 'pointer' : 'default'; } };
    if (vertical) {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'P&L',
                    data: values,
                    backgroundColor: barColors(values),
                    borderRadius: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...clickHandler,
                plugins: {
                    tooltip: { callbacks: { label: (ctx) => formatCurrency(ctx.parsed.y) } }
                },
                scales: {
                    x: {},
                    y: { title: { display: true, text: 'P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
                }
            }
        });
    }
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'P&L',
                data: values,
                backgroundColor: barColors(values),
                borderRadius: 3,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            ...clickHandler,
            plugins: {
                tooltip: { callbacks: { label: (ctx) => formatCurrency(ctx.parsed.x) } }
            },
            scales: {
                x: { title: { display: true, text: 'P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } },
                y: {}
            }
        }
    });
}

// --- TIME ANALYSIS TAB ---

function renderHourlyPnL(canvasId, hourlyPnL, hourlyCount) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const hours = Object.keys(hourlyPnL).sort();
    const pnlVals = hours.map(h => hourlyPnL[h]);
    const countVals = hours.map(h => hourlyCount[h] || 0);
    const labels = hours.map(h => formatHalfHourLabel(h));
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'P&L',
                    data: pnlVals,
                    backgroundColor: barColors(pnlVals),
                    borderRadius: 3,
                    yAxisID: 'y',
                    order: 2,
                },
                {
                    label: 'Trade Count',
                    data: countVals,
                    type: 'line',
                    borderColor: 'rgba(96,165,250,0.8)',
                    backgroundColor: 'rgba(96,165,250,0.1)',
                    pointRadius: 3,
                    pointBackgroundColor: '#60a5fa',
                    borderWidth: 2,
                    yAxisID: 'y1',
                    order: 1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { boxWidth: 12, padding: 16 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.dataset.label === 'P&L') return `P&L: ${formatCurrency(ctx.parsed.y)}`;
                            return `Trades: ${ctx.parsed.y}`;
                        }
                    }
                }
            },
            scales: {
                y: { title: { display: true, text: 'P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } },
                y1: { position: 'right', title: { display: true, text: 'Trades' }, grid: { display: false } }
            }
        }
    });
}

function renderHourlyWinRate(canvasId, hourlyWR) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const hours = Object.keys(hourlyWR).sort();
    const values = hours.map(h => hourlyWR[h]);
    const labels = hours.map(h => formatHalfHourLabel(h));
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Win Rate',
                data: values,
                backgroundColor: values.map(v => v >= 50 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)'),
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: { callbacks: { label: (ctx) => `Win Rate: ${ctx.parsed.y.toFixed(1)}%` } },
                annotation: {
                    annotations: {
                        fiftyLine: { type: 'line', yMin: 50, yMax: 50, borderColor: getCSSVar('--chart-zero'), borderDash: [5, 5], borderWidth: 1, label: { display: true, content: '50%', position: 'start', font: { size: 10 } } }
                    }
                }
            },
            scales: {
                y: { min: 0, max: 100, title: { display: true, text: 'Win Rate %' } }
            }
        }
    });
}

function renderDowPnL(canvasId, dowPnL) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
    const values = days.map(d => dowPnL[d] || 0);
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: days,
            datasets: [{
                label: 'P&L',
                data: values,
                backgroundColor: barColors(values),
                borderRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: { callbacks: { label: (ctx) => formatCurrency(ctx.parsed.y) } }
            },
            scales: {
                y: { title: { display: true, text: 'P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
            }
        }
    });
}

function renderHeatmapTable(containerId, matrix) {
    const container = document.getElementById(containerId);
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
    // Collect all hours across all days
    const hoursSet = new Set();
    days.forEach(d => {
        if (matrix[d]) Object.keys(matrix[d]).forEach(h => hoursSet.add(h));
    });
    const hours = Array.from(hoursSet).sort();
    if (hours.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted);">No data available</p>';
        return null;
    }

    // Find max absolute value for color scaling
    let maxAbs = 0;
    days.forEach(d => {
        hours.forEach(h => {
            const val = matrix[d]?.[String(h)] || 0;
            maxAbs = Math.max(maxAbs, Math.abs(val));
        });
    });

    let html = '<table class="heatmap-table"><thead><tr><th></th>';
    days.forEach(d => html += `<th>${d}</th>`);
    html += '</tr></thead><tbody>';

    hours.forEach(h => {
        const label = formatHalfHourLabel(h);
        html += `<tr><td class="row-label">${label}</td>`;
        days.forEach(d => {
            const val = matrix[d]?.[h] || 0;
            const bg = heatmapColor(val, maxAbs);
            const textColor = val === 0 ? 'var(--text-muted)' : (val > 0 ? 'var(--profit)' : 'var(--loss)');
            html += `<td style="background:${bg};color:${textColor}" title="${d} ${label}: ${formatCurrency(val)}">${val === 0 ? '-' : formatCurrencyShort(val)}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
    return null; // No Chart.js instance
}

// --- INSTRUMENTS TAB ---

function renderInstrumentTradeCount(canvasId, instCount) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const sorted = Object.entries(instCount).sort((a, b) => b[1] - a[1]);
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(s => s[0]),
            datasets: [{
                label: 'Trades',
                data: sorted.map(s => s[1]),
                backgroundColor: 'rgba(96,165,250,0.6)',
                borderRadius: 3,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.x} trades` } }
            },
            scales: {
                x: { title: { display: true, text: 'Trade Count' } }
            }
        }
    });
}

function renderInstrumentTable(containerId, details) {
    const container = document.getElementById(containerId);
    const sorted = [...details].sort((a, b) => b.pnl - a.pnl);
    let html = `<div class="data-table-container"><table class="data-table">
        <thead><tr>
            <th>Instrument</th><th class="num">Trades</th><th class="num">Win Rate</th>
            <th class="num">Avg Win</th><th class="num">Avg Loss</th><th class="num">P&L</th><th class="num">Profit Factor</th>
        </tr></thead><tbody>`;
    sorted.forEach(d => {
        const pf = d.profitFactor >= 9999 ? 'Inf' : d.profitFactor.toFixed(2);
        html += `<tr>
            <td class="fw-600">${d.instrument}</td>
            <td class="num">${d.trades}</td>
            <td class="num ${d.winRate >= 50 ? 'text-profit' : 'text-loss'}">${d.winRate.toFixed(1)}%</td>
            <td class="num text-profit">${formatCurrency(d.avgWin)}</td>
            <td class="num text-loss">${formatCurrency(d.avgLoss)}</td>
            <td class="num ${profitTextClass(d.pnl)}">${formatCurrency(d.pnl)}</td>
            <td class="num" style="color:${pfColor(d.profitFactor)}">${pf}</td>
        </tr>`;
    });
    html += '</tbody></table></div>';
    container.innerHTML = html;
    return null;
}

// --- RISK TAB ---

function renderScatterMAE(canvasId, maeData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const wins = maeData.filter(d => d[1] > 0).map(d => ({ x: d[0], y: d[1] }));
    const losses = maeData.filter(d => d[1] <= 0).map(d => ({ x: d[0], y: d[1] }));
    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Winners', data: wins, backgroundColor: 'rgba(34,197,94,0.4)', pointRadius: 3 },
                { label: 'Losers', data: losses, backgroundColor: 'rgba(239,68,68,0.4)', pointRadius: 3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { boxWidth: 10, padding: 12 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `MAE: ${formatCurrency(ctx.parsed.x)}, Profit: ${formatCurrency(ctx.parsed.y)}`
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'MAE ($)' }, ticks: { callback: v => formatCurrencyShort(v) } },
                y: { title: { display: true, text: 'Profit ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
            }
        }
    });
}

function renderScatterMFE(canvasId, mfeData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const wins = mfeData.filter(d => d[1] > 0).map(d => ({ x: d[0], y: d[1] }));
    const losses = mfeData.filter(d => d[1] <= 0).map(d => ({ x: d[0], y: d[1] }));
    return new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Winners', data: wins, backgroundColor: 'rgba(34,197,94,0.4)', pointRadius: 3 },
                { label: 'Losers', data: losses, backgroundColor: 'rgba(239,68,68,0.4)', pointRadius: 3 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { boxWidth: 10, padding: 12 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `MFE: ${formatCurrency(ctx.parsed.x)}, Profit: ${formatCurrency(ctx.parsed.y)}`
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'MFE ($)' }, ticks: { callback: v => formatCurrencyShort(v) } },
                y: { title: { display: true, text: 'Profit ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
            }
        }
    });
}

function renderRollingPnL(canvasId, rollingData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (!rollingData || rollingData.length === 0) {
        return renderEmptyChart(ctx, 'Not enough trades for rolling 20-trade average');
    }
    let points = rollingData;
    if (points.length > 800) {
        const step = Math.ceil(points.length / 800);
        points = points.filter((_, i) => i % step === 0 || i === points.length - 1);
    }
    const values = points.map(d => d[1]);
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: points.map(d => d[0]),
            datasets: [{
                label: 'Rolling 20-Trade Avg P&L',
                data: values,
                borderColor: '#60a5fa',
                pointRadius: 0,
                borderWidth: 2,
                tension: 0.2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                tooltip: { callbacks: { label: (ctx) => `Avg: ${formatCurrency(ctx.parsed.y)}` } },
                annotation: {
                    annotations: {
                        zeroLine: { type: 'line', yMin: 0, yMax: 0, borderColor: getCSSVar('--chart-zero'), borderDash: [5, 5], borderWidth: 1 }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Trade #' }, ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'Avg P&L ($)' }, ticks: { callback: v => formatCurrencyShort(v) } }
            }
        }
    });
}

function renderRollingWinRate(canvasId, rollingData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (!rollingData || rollingData.length === 0) {
        return renderEmptyChart(ctx, 'Not enough trades for rolling 50-trade win rate');
    }
    let points = rollingData;
    if (points.length > 800) {
        const step = Math.ceil(points.length / 800);
        points = points.filter((_, i) => i % step === 0 || i === points.length - 1);
    }
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: points.map(d => d[0]),
            datasets: [{
                label: 'Rolling 50-Trade Win Rate',
                data: points.map(d => d[1]),
                borderColor: '#a78bfa',
                pointRadius: 0,
                borderWidth: 2,
                tension: 0.2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                tooltip: { callbacks: { label: (ctx) => `Win Rate: ${ctx.parsed.y.toFixed(1)}%` } },
                annotation: {
                    annotations: {
                        fiftyLine: { type: 'line', yMin: 50, yMax: 50, borderColor: getCSSVar('--chart-zero'), borderDash: [5, 5], borderWidth: 1 }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Trade #' }, ticks: { maxTicksLimit: 10 } },
                y: { min: 0, max: 100, title: { display: true, text: 'Win Rate %' } }
            }
        }
    });
}

function renderStreaks(canvasId, streaks) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (!streaks || streaks.length === 0) {
        return renderEmptyChart(ctx, 'No streak data available');
    }
    // Get top streaks sorted by length
    const top = [...streaks].sort((a, b) => b.length - a.length).slice(0, 15);
    const labels = top.map((s, i) => `#${i + 1} (${s.type})`);
    const values = top.map(s => s.type === 'win' ? s.length : -s.length);
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Streak Length',
                data: values,
                backgroundColor: values.map(v => v > 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'),
                borderRadius: 3,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const abs = Math.abs(ctx.parsed.x);
                            return `${abs} consecutive ${ctx.parsed.x > 0 ? 'wins' : 'losses'}`;
                        }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Streak Length' } }
            }
        }
    });
}

// Helper for empty state
function renderEmptyChart(ctx, message) {
    return new Chart(ctx, {
        type: 'bar',
        data: { labels: [''], datasets: [{ data: [0], backgroundColor: 'transparent' }] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: message, color: getCSSVar('--text-muted'), font: { size: 14 } },
                legend: { display: false }
            },
            scales: { x: { display: false }, y: { display: false } }
        }
    });
}


// --- CALENDAR TAB ---

function renderCalendar(containerId, dailyPnL) {
    const container = document.getElementById(containerId);
    if (!dailyPnL || dailyPnL.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted);padding:20px">No daily data available</p>';
        return;
    }

    // Build lookup: "YYYY-MM-DD" → { pnl, trades, wins }
    const dayMap = {};
    let maxAbsPnl = 0;
    dailyPnL.forEach(d => {
        dayMap[d.date] = d;
        maxAbsPnl = Math.max(maxAbsPnl, Math.abs(d.pnl));
    });

    // Determine month range from data
    const dates = dailyPnL.map(d => d.date).sort();
    const firstDate = new Date(dates[0] + 'T00:00:00');
    const lastDate = new Date(dates[dates.length - 1] + 'T00:00:00');
    const startMonth = new Date(firstDate.getFullYear(), firstDate.getMonth(), 1);
    const endMonth = new Date(lastDate.getFullYear(), lastDate.getMonth(), 1);

    // Collect all months
    const months = [];
    const cur = new Date(startMonth);
    while (cur <= endMonth) {
        months.push(new Date(cur));
        cur.setMonth(cur.getMonth() + 1);
    }

    // Grand totals
    let grandPnl = 0, grandTrades = 0, grandWins = 0, grandDays = 0, grandGreenDays = 0, grandRedDays = 0;
    dailyPnL.forEach(d => {
        grandPnl += d.pnl;
        grandTrades += d.trades;
        grandWins += d.wins;
        grandDays++;
        if (d.pnl > 0) grandGreenDays++;
        else if (d.pnl < 0) grandRedDays++;
    });
    const grandWinRate = grandDays > 0 ? ((grandGreenDays / grandDays) * 100).toFixed(1) : '0.0';

    let html = `<div class="calendar-summary">
        <div class="calendar-summary-item">
            <span class="calendar-summary-label">Total P&L</span>
            <span class="calendar-summary-value ${profitTextClass(grandPnl)}">${formatCurrency(Math.round(grandPnl * 100) / 100)}</span>
        </div>
        <div class="calendar-summary-item">
            <span class="calendar-summary-label">Trading Days</span>
            <span class="calendar-summary-value text-accent">${grandDays}</span>
        </div>
        <div class="calendar-summary-item">
            <span class="calendar-summary-label">Green / Red Days</span>
            <span class="calendar-summary-value"><span class="text-profit">${grandGreenDays}</span> / <span class="text-loss">${grandRedDays}</span></span>
        </div>
        <div class="calendar-summary-item">
            <span class="calendar-summary-label">Win Day Rate</span>
            <span class="calendar-summary-value" style="color:${grandGreenDays / grandDays >= 0.5 ? 'var(--profit)' : 'var(--loss)'}">${grandWinRate}%</span>
        </div>
        <div class="calendar-summary-item">
            <span class="calendar-summary-label">Avg Day</span>
            <span class="calendar-summary-value ${profitTextClass(grandPnl / grandDays)}">${formatCurrency(Math.round((grandPnl / grandDays) * 100) / 100)}</span>
        </div>
    </div>`;

    html += '<div class="calendar-grid">';

    months.forEach(monthDate => {
        const year = monthDate.getFullYear();
        const month = monthDate.getMonth();
        const monthName = monthDate.toLocaleString('en-US', { month: 'long', year: 'numeric' });

        // First day of month (0=Sun, 1=Mon, ..., 6=Sat)
        const firstDay = new Date(year, month, 1).getDay();
        // Shift to Mon-based: Mon=0, Tue=1, ..., Sun=6
        const startOffset = (firstDay + 6) % 7;
        const daysInMonth = new Date(year, month + 1, 0).getDate();

        // Monthly totals
        let monthPnl = 0, monthTrades = 0, monthDays = 0, monthGreen = 0, monthRed = 0;
        for (let d = 1; d <= daysInMonth; d++) {
            const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
            if (dayMap[key]) {
                monthPnl += dayMap[key].pnl;
                monthTrades += dayMap[key].trades;
                monthDays++;
                if (dayMap[key].pnl > 0) monthGreen++;
                else if (dayMap[key].pnl < 0) monthRed++;
            }
        }

        html += `<div class="calendar-month">
            <div class="calendar-month-header">
                <span class="calendar-month-name">${monthName}</span>
                <span class="calendar-month-pnl ${profitTextClass(monthPnl)}">${formatCurrency(Math.round(monthPnl * 100) / 100)}</span>
            </div>
            <div class="calendar-dow-row">
                <span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span><span>Sat</span><span>Sun</span>
            </div>
            <div class="calendar-days">`;

        // Empty cells before first day
        for (let i = 0; i < startOffset; i++) {
            html += '<div class="calendar-day empty"></div>';
        }

        // Day cells
        for (let d = 1; d <= daysInMonth; d++) {
            const key = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
            const dayData = dayMap[key];
            const dow = new Date(year, month, d).getDay();
            const isWeekend = (dow === 0 || dow === 6);

            if (dayData) {
                const intensity = maxAbsPnl > 0 ? Math.min(Math.abs(dayData.pnl) / maxAbsPnl, 1) : 0;
                const alpha = 0.12 + intensity * 0.55;
                const bg = dayData.pnl > 0
                    ? `rgba(34, 197, 94, ${alpha})`
                    : dayData.pnl < 0
                    ? `rgba(239, 68, 68, ${alpha})`
                    : 'rgba(107, 114, 128, 0.1)';
                const textColor = dayData.pnl > 0 ? 'var(--profit)' : dayData.pnl < 0 ? 'var(--loss)' : 'var(--neutral)';
                html += `<div class="calendar-day has-data" style="background:${bg}" title="${monthDate.toLocaleString('en-US', { month: 'short' })} ${d}: ${formatCurrency(dayData.pnl)} (${dayData.trades} trades, ${dayData.wins}W)">
                    <span class="calendar-day-num">${d}</span>
                    <span class="calendar-day-pnl" style="color:${textColor}">${formatCurrencyShort(dayData.pnl)}</span>
                    <span class="calendar-day-trades">${dayData.trades}t</span>
                </div>`;
            } else {
                html += `<div class="calendar-day${isWeekend ? ' weekend' : ''}">
                    <span class="calendar-day-num">${d}</span>
                </div>`;
            }
        }

        // Trailing empty cells to complete the grid
        const totalCells = startOffset + daysInMonth;
        const remainder = totalCells % 7;
        if (remainder > 0) {
            for (let i = 0; i < 7 - remainder; i++) {
                html += '<div class="calendar-day empty"></div>';
            }
        }

        html += `</div>
            <div class="calendar-month-footer">
                <span>${monthDays} days</span>
                <span><span class="text-profit">${monthGreen}W</span> / <span class="text-loss">${monthRed}L</span></span>
                <span>${monthTrades} trades</span>
            </div>
        </div>`;
    });

    html += '</div>';
    container.innerHTML = html;
}

