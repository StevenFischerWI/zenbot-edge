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

function renderInstrumentPnLBar(canvasId, instPnL, vertical) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const sorted = Object.entries(instPnL).sort((a, b) => a[0].localeCompare(b[0]));
    const labels = sorted.map(s => s[0]);
    const values = sorted.map(s => s[1]);
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
    const hours = Object.keys(hourlyPnL).sort((a, b) => +a - +b);
    const pnlVals = hours.map(h => hourlyPnL[h]);
    const countVals = hours.map(h => hourlyCount[h] || 0);
    const labels = hours.map(h => {
        const hr = +h;
        return hr === 0 ? '12 AM' : hr < 12 ? `${hr} AM` : hr === 12 ? '12 PM' : `${hr - 12} PM`;
    });
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
    const hours = Object.keys(hourlyWR).sort((a, b) => +a - +b);
    const values = hours.map(h => hourlyWR[h]);
    const labels = hours.map(h => {
        const hr = +h;
        return hr === 0 ? '12A' : hr < 12 ? `${hr}A` : hr === 12 ? '12P' : `${hr - 12}P`;
    });
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
        if (matrix[d]) Object.keys(matrix[d]).forEach(h => hoursSet.add(+h));
    });
    const hours = Array.from(hoursSet).sort((a, b) => a - b);
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
        const label = h === 0 ? '12 AM' : h < 12 ? `${h} AM` : h === 12 ? '12 PM' : `${h - 12} PM`;
        html += `<tr><td class="row-label">${label}</td>`;
        days.forEach(d => {
            const val = matrix[d]?.[String(h)] || 0;
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

// --- REGIME TAB ---

function renderRegimeTab() {
    const data = REGIME_DATA;

    // 1. Filter rules table (with walk-forward validation)
    renderRegimeFiltersTable('regime-filters-container', data.singleFilters, data.baseline, data.split);

    // 2. Combo filters table
    renderRegimeCombosTable('regime-combos-container', data.comboFilters, data.baseline, data.split);

    // 3. Per-strategy filter impact matrix
    renderPerStrategyTable('regime-per-strategy-container', data.perStrategy, data.singleFilters);

    // 3b. Per-strategy indicator discovery
    renderStrategyDiscoveryTable('regime-strategy-discovery-container', data.strategyDiscovery);

    // 4. Indicator rankings table
    renderRegimeRankingsTable('regime-rankings-container', data.indicators);

    // 5. Bucket charts for top indicators
    renderRegimeBucketCharts('regime-bucket-charts', data.indicators);

    // 6. Timeline chart
    appState.chartInstances.regimeTimeline = renderRegimeTimeline('chart-regime-timeline', data.timeline);
}

function renderRegimeFiltersTable(containerId, filters, baseline, split) {
    const container = document.getElementById(containerId);
    if (!filters || filters.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No single filters found</p>';
        return;
    }

    let html = `<p style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">
        Baseline: ${baseline.count} trades, ${baseline.winRate}% WR, PF ${baseline.pf.toFixed(2)}, P&L ${formatCurrency(baseline.pnl)}
    </p>`;
    if (split) {
        html += `<p style="font-size:11px;color:var(--text-muted);margin-bottom:12px">
            Walk-forward: Train ${split.trainCount} trades (${split.trainRange}) | Test ${split.testCount} trades (${split.testRange})
        </p>`;
    }
    html += `<div class="data-table-container"><table class="data-table">
        <thead><tr>
            <th rowspan="2" style="vertical-align:bottom">No-Trade Rule</th>
            <th colspan="3" class="num" style="text-align:center;border-bottom:1px solid var(--border);font-size:11px;color:var(--text-muted)">Train (In-Sample)</th>
            <th colspan="3" class="num" style="text-align:center;border-bottom:1px solid var(--border);font-size:11px;color:var(--text-muted)">Test (Out-of-Sample)</th>
            <th rowspan="2" class="num" style="vertical-align:bottom">Verdict</th>
        </tr><tr>
            <th class="num">Out</th>
            <th class="num">P&L Saved</th>
            <th class="num">PF Kept</th>
            <th class="num">Out</th>
            <th class="num">P&L Saved</th>
            <th class="num">PF Kept</th>
        </tr></thead><tbody>`;

    filters.forEach(f => {
        const trainPnlClass = f.pnlFiltered < 0 ? 'text-loss' : 'text-profit';
        const testPnl = f.testPnlFiltered || 0;
        const testPnlClass = testPnl < 0 ? 'text-loss' : 'text-profit';
        const testPfKept = f.testPfKept || 0;
        const testOut = f.testTradesFiltered || 0;
        const holds = testPnl < 0 && testPfKept > (f.testPfFiltered || 0);
        const verdictClass = holds ? 'text-profit' : 'text-loss';
        const verdict = holds ? 'HOLDS' : 'OVERFIT';
        html += `<tr>
            <td class="fw-600">${f.rule}</td>
            <td class="num">${f.tradesFiltered}</td>
            <td class="num ${trainPnlClass}">${formatCurrency(f.pnlFiltered)}</td>
            <td class="num fw-600">${f.pfKept.toFixed(2)}</td>
            <td class="num">${testOut}</td>
            <td class="num ${testPnlClass}">${formatCurrency(testPnl)}</td>
            <td class="num fw-600">${testPfKept.toFixed(2)}</td>
            <td class="num ${verdictClass} fw-600">${verdict}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function renderRegimeCombosTable(containerId, combos, baseline, split) {
    const container = document.getElementById(containerId);
    if (!combos || combos.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No combination filters found</p>';
        return;
    }

    let html = `<div class="data-table-container"><table class="data-table">
        <thead><tr>
            <th rowspan="2" style="vertical-align:bottom">Filter Rules (OR)</th>
            <th colspan="3" class="num" style="text-align:center;border-bottom:1px solid var(--border);font-size:11px;color:var(--text-muted)">Train (In-Sample)</th>
            <th colspan="3" class="num" style="text-align:center;border-bottom:1px solid var(--border);font-size:11px;color:var(--text-muted)">Test (Out-of-Sample)</th>
            <th rowspan="2" class="num" style="vertical-align:bottom">Verdict</th>
        </tr><tr>
            <th class="num">Out</th>
            <th class="num">P&L Saved</th>
            <th class="num">PF Kept</th>
            <th class="num">Out</th>
            <th class="num">P&L Saved</th>
            <th class="num">PF Kept</th>
        </tr></thead><tbody>`;

    combos.forEach(cf => {
        const testPnl = cf.testPnlFiltered || 0;
        const testPnlClass = testPnl < 0 ? 'text-loss' : 'text-profit';
        const testPfKept = cf.testPfKept || 0;
        const testOut = cf.testTradesFiltered || 0;
        const holds = testPnl < 0 && testPfKept > (cf.testPfFiltered || 0);
        const verdictClass = holds ? 'text-profit' : 'text-loss';
        const verdict = holds ? 'HOLDS' : 'OVERFIT';
        html += `<tr>
            <td class="fw-600" style="white-space:normal;max-width:350px">${cf.rule}</td>
            <td class="num">${cf.tradesFiltered}</td>
            <td class="num text-loss">${formatCurrency(cf.pnlFiltered)}</td>
            <td class="num fw-600">${cf.pfKept.toFixed(2)}</td>
            <td class="num">${testOut}</td>
            <td class="num ${testPnlClass}">${formatCurrency(testPnl)}</td>
            <td class="num fw-600">${testPfKept.toFixed(2)}</td>
            <td class="num ${verdictClass} fw-600">${verdict}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function renderPerStrategyTable(containerId, perStrategy, singleFilters) {
    const container = document.getElementById(containerId);
    if (!perStrategy || perStrategy.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No per-strategy data available</p>';
        return;
    }

    // Get filter rules from the first strategy's filter list
    const filterRules = perStrategy[0].filters.map(f => f.rule);

    // Count how many strategies each filter helps
    const filterSummary = filterRules.map((rule, fi) => {
        let holds = 0, overfit = 0, na = 0;
        perStrategy.forEach(ps => {
            const fr = ps.filters[fi];
            if (fr.trainFiltered === 0 && fr.testFiltered === 0) na++;
            else if (fr.holds) holds++;
            else overfit++;
        });
        return { rule, holds, overfit, na };
    });

    // Shorten rule names for column headers
    const shortRule = (rule) => {
        return rule.length > 22 ? rule.substring(0, 20) + '..' : rule;
    };

    let html = `<p style="font-size:12px;color:var(--text-secondary);margin-bottom:8px">
        Matrix: top global filters applied to each strategy individually. Green = filter HOLDS (helps), Red = OVERFIT (hurts), Gray = n/a (no trades filtered).
    </p>`;

    html += `<div class="data-table-container" style="overflow-x:auto"><table class="data-table" style="font-size:12px">
        <thead><tr>
            <th style="position:sticky;left:0;background:var(--bg-card);z-index:2">Strategy</th>
            <th class="num">Trades</th>`;

    filterRules.forEach(rule => {
        html += `<th class="num" style="white-space:nowrap;max-width:140px;overflow:hidden;text-overflow:ellipsis" title="${rule}">${shortRule(rule)}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Data rows — one per strategy
    perStrategy.forEach(ps => {
        const totalTrades = ps.trainTrades + ps.testTrades;
        html += `<tr>
            <td class="fw-600" style="position:sticky;left:0;background:var(--bg-card);z-index:1;white-space:nowrap">${ps.strategy}</td>
            <td class="num">${totalTrades}</td>`;

        ps.filters.forEach(fr => {
            let cellContent, cellStyle;
            if (fr.trainFiltered === 0 && fr.testFiltered === 0) {
                cellContent = 'n/a';
                cellStyle = 'color:var(--text-muted)';
            } else if (fr.holds) {
                const saved = Math.abs(fr.testPnlFiltered);
                cellContent = saved > 0 ? formatCurrencyShort(saved) : 'HOLDS';
                cellStyle = 'color:var(--profit);font-weight:600';
            } else {
                const lost = fr.testPnlFiltered;
                cellContent = lost !== 0 ? formatCurrencyShort(Math.abs(lost)) : 'OVERFIT';
                cellStyle = 'color:var(--loss)';
            }
            html += `<td class="num" style="${cellStyle}" title="${fr.rule}: test P&L filtered=${formatCurrency(fr.testPnlFiltered)}, test PF kept=${fr.testPfKept.toFixed(2)}">${cellContent}</td>`;
        });
        html += '</tr>';
    });

    // Summary row
    html += `<tr style="border-top:2px solid var(--border);font-weight:600">
        <td style="position:sticky;left:0;background:var(--bg-card);z-index:1">Summary</td>
        <td class="num"></td>`;
    filterSummary.forEach(fs => {
        const ratio = fs.holds + fs.overfit > 0
            ? Math.round(fs.holds / (fs.holds + fs.overfit) * 100)
            : 0;
        const color = ratio >= 60 ? '#22c55e' : ratio >= 40 ? '#eab308' : '#ef4444';
        html += `<td class="num" style="color:${color}" title="${fs.holds} HOLDS, ${fs.overfit} OVERFIT, ${fs.na} n/a">${fs.holds}/${fs.holds + fs.overfit}</td>`;
    });
    html += '</tr>';

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function _renderDiscoveryFilterRows(filters) {
    let html = '';
    filters.forEach(f => {
        const trainPnlClass = f.pnlFiltered < 0 ? 'text-loss' : 'text-profit';
        const testPnl = f.testPnlFiltered || 0;
        const testPnlClass = testPnl < 0 ? 'text-loss' : 'text-profit';
        const verdictClass = f.holds ? 'text-profit' : 'text-loss';
        const verdict = f.holds ? 'HOLDS' : 'OVERFIT';
        html += `<tr>
            <td class="fw-600">${f.rule}</td>
            <td class="num">${f.tradesFiltered}</td>
            <td class="num ${trainPnlClass}">${formatCurrency(f.pnlFiltered)}</td>
            <td class="num ${testPnlClass}">${formatCurrency(testPnl)}</td>
            <td class="num ${verdictClass} fw-600">${verdict}</td>
        </tr>`;
    });
    return html;
}

function _renderDiscoveryTable(filters, label) {
    if (!filters || filters.length === 0) return '';
    const holdsCount = filters.filter(f => f.holds).length;
    const labelColor = holdsCount > 0 ? '#22c55e' : 'var(--text-muted)';
    let html = `<div style="margin-bottom:10px">
        <div style="font-size:12px;font-weight:600;margin-bottom:4px;display:flex;gap:8px;align-items:center">
            <span>${label}</span>
            <span style="color:${labelColor};font-size:11px">${holdsCount}/${filters.length} HOLD</span>
        </div>
        <div class="data-table-container"><table class="data-table" style="font-size:12px">
            <thead><tr>
                <th>Rule</th>
                <th class="num">Train Out</th>
                <th class="num">Train P&L Saved</th>
                <th class="num">Test P&L Saved</th>
                <th class="num">Verdict</th>
            </tr></thead><tbody>`;
    html += _renderDiscoveryFilterRows(filters);
    html += '</tbody></table></div></div>';
    return html;
}

function renderStrategyDiscoveryTable(containerId, strategyDiscovery) {
    const container = document.getElementById(containerId);
    if (!strategyDiscovery || strategyDiscovery.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No strategy-specific discovery data available</p>';
        return;
    }

    let html = `<p style="font-size:12px;color:var(--text-secondary);margin-bottom:12px">
        Indicators discovered independently for each strategy (combined + per-direction). Fewer trades means more OVERFIT verdicts — the walk-forward validation flags this honestly.
    </p>`;

    strategyDiscovery.forEach((sd, idx) => {
        const hasFilters = (sd.filters && sd.filters.length > 0);
        const hasDirections = (sd.directions && sd.directions.length > 0);
        if (!hasFilters && !hasDirections) return;

        const holdsCount = sd.filters ? sd.filters.filter(f => f.holds).length : 0;
        const totalFilters = sd.filters ? sd.filters.length : 0;
        // Count direction holds too
        let dirHolds = 0, dirTotal = 0;
        if (sd.directions) {
            sd.directions.forEach(dd => {
                dd.filters.forEach(f => { dirTotal++; if (f.holds) dirHolds++; });
            });
        }
        const allHolds = holdsCount + dirHolds;
        const allTotal = totalFilters + dirTotal;
        const summaryColor = allHolds > 0 ? '#22c55e' : '#6b7280';
        const isExpanded = idx < 3;

        html += `<div class="discovery-strategy-section" style="margin-bottom:8px">
            <div class="discovery-header" style="cursor:pointer;display:flex;align-items:center;gap:8px;padding:8px 12px;background:var(--bg-card-hover);border-radius:6px;border:1px solid var(--border)"
                 onclick="this.parentElement.classList.toggle('collapsed');this.querySelector('.chevron').textContent=this.parentElement.classList.contains('collapsed')?'\\u25B6':'\\u25BC'">
                <span class="chevron" style="font-size:10px;color:var(--text-muted);width:12px">${isExpanded ? '\u25BC' : '\u25B6'}</span>
                <span class="fw-600" style="flex:1">${sd.strategy}</span>
                <span style="font-size:12px;color:var(--text-muted)">${sd.tradeCount} trades</span>
                <span style="font-size:12px;color:${summaryColor};font-weight:600">${allHolds}/${allTotal} HOLD</span>
            </div>
            <div class="discovery-body" style="padding:8px 0 0 0;${isExpanded ? '' : 'display:none'}">`;

        // Combined table
        if (hasFilters) {
            html += _renderDiscoveryTable(sd.filters, 'All Trades');
        }

        // Per-direction tables
        if (hasDirections) {
            sd.directions.forEach(dd => {
                const dirLabel = `${dd.direction} Only (${dd.tradeCount} trades)`;
                html += _renderDiscoveryTable(dd.filters, dirLabel);
            });
        }

        html += `</div></div>`;
    });

    html += `<style>
        .discovery-strategy-section.collapsed .discovery-body { display: none !important; }
        .discovery-strategy-section:not(.collapsed) .discovery-body { display: block !important; }
        .discovery-header:hover { background: var(--bg-card-hover) !important; }
    </style>`;

    container.innerHTML = html;

    // Set initial collapsed state for items after the first 3
    container.querySelectorAll('.discovery-strategy-section').forEach((el, i) => {
        if (i >= 3) el.classList.add('collapsed');
    });
}

function renderRegimeRankingsTable(containerId, indicators) {
    const container = document.getElementById(containerId);
    if (!indicators || indicators.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No indicator data</p>';
        return;
    }

    let html = `<div class="data-table-container"><table class="data-table">
        <thead><tr>
            <th>#</th><th>Indicator</th><th>Type</th>
            <th class="num">PF Spread</th><th class="num">Correlation</th>
            <th class="num">Best PF</th><th class="num">Worst PF</th>
        </tr></thead><tbody>`;

    indicators.forEach((r, i) => {
        const pfs = r.buckets.map(b => b.pf).filter(p => p < 9999);
        const best = pfs.length ? Math.max(...pfs) : 0;
        const worst = pfs.length ? Math.min(...pfs) : 0;
        const corr = r.correlation != null ? r.correlation.toFixed(4) : 'N/A';
        html += `<tr>
            <td>${i + 1}</td>
            <td class="fw-600">${r.name}</td>
            <td>${r.type}</td>
            <td class="num fw-600" style="color:${r.pfSpread > 0.5 ? 'var(--accent)' : 'var(--text-secondary)'}">${r.pfSpread.toFixed(2)}</td>
            <td class="num">${corr}</td>
            <td class="num text-profit">${best.toFixed(2)}</td>
            <td class="num text-loss">${worst.toFixed(2)}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function renderRegimeBucketCharts(containerId, indicators) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    // Render bucket charts for top 6 numeric indicators
    const numericIndicators = indicators.filter(r => r.type === 'numeric').slice(0, 6);

    numericIndicators.forEach(r => {
        const card = document.createElement('div');
        card.className = 'chart-card';
        card.innerHTML = `<h4>${r.name} — PF by Quintile</h4><div class="chart-container-sm"><canvas></canvas></div>`;
        container.appendChild(card);

        const canvas = card.querySelector('canvas');
        const ctx = canvas.getContext('2d');

        const labels = r.buckets.map(b => b.label);
        const pfData = r.buckets.map(b => Math.min(b.pf, 5));
        const wrData = r.buckets.map(b => b.winRate);
        const colors = pfData.map(pf => pf >= 1.0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)');

        appState.chartInstances['regime_bucket_' + r.name] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Profit Factor',
                    data: pfData,
                    backgroundColor: colors,
                    borderRadius: 4,
                    yAxisID: 'y',
                }, {
                    label: 'Win Rate %',
                    data: wrData,
                    type: 'line',
                    borderColor: '#60a5fa',
                    backgroundColor: 'transparent',
                    pointRadius: 4,
                    pointBackgroundColor: '#60a5fa',
                    tension: 0.3,
                    yAxisID: 'y1',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(ctx) {
                                const b = r.buckets[ctx.dataIndex];
                                return `Range: ${b.range[0].toFixed(2)} – ${b.range[1].toFixed(2)}\nTrades: ${b.count}\nP&L: ${formatCurrency(b.pnl)}`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            breakeven: {
                                type: 'line', yMin: 1, yMax: 1, yScaleID: 'y',
                                borderColor: getCSSVar('--chart-zero'), borderDash: [4, 4], borderWidth: 1,
                            }
                        }
                    }
                },
                scales: {
                    y: { title: { display: true, text: 'PF' }, beginAtZero: true },
                    y1: { position: 'right', title: { display: true, text: 'WR%' }, min: 30, max: 70, grid: { display: false } },
                }
            }
        });
    });
}

function renderRegimeTimeline(canvasId, timeline) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const dates = timeline.map(d => d.date);
    const pnlData = timeline.map(d => d.pnl);
    const tradeData = timeline.map(d => d.trades);
    const colors = pnlData.map(v => v >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)');

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Daily P&L',
                data: pnlData,
                backgroundColor: colors,
                borderRadius: 3,
                yAxisID: 'y',
            }, {
                label: 'Trade Count',
                data: tradeData,
                type: 'line',
                borderColor: 'rgba(96,165,250,0.5)',
                backgroundColor: 'transparent',
                pointRadius: 2,
                tension: 0.3,
                yAxisID: 'y1',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        afterLabel: function(ctx) {
                            const d = timeline[ctx.dataIndex];
                            return `W/L: ${d.wins}/${d.losses}  WR: ${d.winRate}%`;
                        }
                    }
                },
                annotation: {
                    annotations: {
                        zero: { type: 'line', yMin: 0, yMax: 0, yScaleID: 'y', borderColor: getCSSVar('--chart-zero'), borderWidth: 1 }
                    }
                }
            },
            scales: {
                x: { ticks: { maxTicksAuto: true, maxRotation: 45, font: { size: 10 } } },
                y: { title: { display: true, text: 'P&L ($)' } },
                y1: { position: 'right', title: { display: true, text: 'Trades' }, grid: { display: false }, beginAtZero: true },
            }
        }
    });
}
