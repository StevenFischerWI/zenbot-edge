/* Application entry point, state management, rendering */

const appState = {
    data: null,
    selectedStrategies: new Set(['_ALL']),  // Multi-select: Set of strategy keys
    activeTab: 'overview',
    chartInstances: {},
    sidebarSort: 'name',
    sidebarFilter: '',
    // Global filters
    globalDirection: '',          // '' | 'Long' | 'Short'
    globalInstruments: new Set(), // Set of instrument strings
    globalHours: new Set(),      // Set of entry hour numbers
    globalDateFrom: '',           // '' | 'YYYY-MM-DD'
    globalDateTo: '',             // '' | 'YYYY-MM-DD'
    theme: 'dark',                // 'dark' | 'light'
};

const STATE_KEY = 'futures-analysis-state';
const DATA_KEY = 'futures-analysis-data';
const IDB_NAME = 'futures-analysis-db';
const IDB_STORE = 'tradedata';

function idbOpen() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(IDB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

function idbSaveData(data) {
    return idbOpen().then(db => new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readwrite');
        tx.objectStore(IDB_STORE).put(data, DATA_KEY);
        tx.oncomplete = () => { db.close(); resolve(); };
        tx.onerror = () => { db.close(); reject(tx.error); };
    }));
}

function idbLoadData() {
    return idbOpen().then(db => new Promise((resolve, reject) => {
        const tx = db.transaction(IDB_STORE, 'readonly');
        const req = tx.objectStore(IDB_STORE).get(DATA_KEY);
        req.onsuccess = () => { db.close(); resolve(req.result); };
        req.onerror = () => { db.close(); reject(req.error); };
    }));
}

function saveState() {
    try {
        const s = {
            selectedStrategies: [...appState.selectedStrategies],
            activeTab: appState.activeTab,
            sidebarSort: appState.sidebarSort,
            sidebarFilter: appState.sidebarFilter,
            globalDirection: appState.globalDirection,
            globalInstruments: [...appState.globalInstruments],
            globalHours: [...appState.globalHours],
            globalDateFrom: appState.globalDateFrom,
            globalDateTo: appState.globalDateTo,
            theme: appState.theme,
        };
        localStorage.setItem(STATE_KEY, JSON.stringify(s));
    } catch (e) { /* quota or private mode — ignore */ }
}

function loadState() {
    try {
        const raw = localStorage.getItem(STATE_KEY);
        if (!raw) return;
        const s = JSON.parse(raw);

        // Validate & restore strategies
        if (Array.isArray(s.selectedStrategies) && s.selectedStrategies.length > 0) {
            const validStrategies = new Set(appState.data.metadata.strategies);
            const restored = s.selectedStrategies.filter(k => k === '_ALL' || validStrategies.has(k));
            if (restored.length > 0) {
                appState.selectedStrategies = new Set(restored);
            }
        }

        // Restore tab
        const validTabs = ['overview', 'calendar', 'time', 'instruments', 'risk', 'trades', 'strategies'];
        if (validTabs.includes(s.activeTab)) {
            appState.activeTab = s.activeTab;
        }

        // Restore sidebar sort & filter
        if (['name', 'pnl', 'trades', 'wr'].includes(s.sidebarSort)) {
            appState.sidebarSort = s.sidebarSort;
        }
        if (typeof s.sidebarFilter === 'string') {
            appState.sidebarFilter = s.sidebarFilter;
        }

        // Restore global filters
        if (s.globalDirection === 'Long' || s.globalDirection === 'Short') {
            appState.globalDirection = s.globalDirection;
        }
        if (Array.isArray(s.globalInstruments)) {
            const validInstruments = new Set(appState.data.metadata.instruments);
            const restored = s.globalInstruments.filter(i => validInstruments.has(i));
            appState.globalInstruments = new Set(restored);
        }
        if (Array.isArray(s.globalHours)) {
            appState.globalHours = new Set(s.globalHours.filter(h => typeof h === 'string' && /^\d{2}:\d{2}$/.test(h)));
        }

        // Restore theme
        if (s.theme === 'light' || s.theme === 'dark') {
            appState.theme = s.theme;
        }

        // Restore date range
        const dr = appState.data.metadata.dateRange;
        if (s.globalDateFrom && s.globalDateFrom >= dr.start && s.globalDateFrom <= dr.end) {
            appState.globalDateFrom = s.globalDateFrom;
        }
        if (s.globalDateTo && s.globalDateTo >= dr.start && s.globalDateTo <= dr.end) {
            appState.globalDateTo = s.globalDateTo;
        }
    } catch (e) { /* corrupted data — ignore, use defaults */ }
}

function applyRestoredState() {
    // Sync direction buttons
    document.querySelectorAll('.direction-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.value === appState.globalDirection);
    });

    // Sync instrument checkboxes and button
    appState.globalInstruments.forEach(inst => {
        const cb = document.querySelector(`#instrument-dropdown input[value="${inst}"]`);
        if (cb) cb.checked = true;
    });
    syncInstrumentPresets();
    updateInstrumentBtn();

    // Sync half-hour checkboxes and button
    appState.globalHours.forEach(h => {
        const cb = document.querySelector(`#hour-dropdown input[value="${h}"]`);
        if (cb) cb.checked = true;
    });
    syncRTHPreset();
    syncAllHoursPreset();
    updateHourBtn();

    // Sync date inputs
    document.getElementById('global-date-from').value = appState.globalDateFrom;
    document.getElementById('global-date-to').value = appState.globalDateTo;

    // Sync sidebar sort buttons
    document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
    const sortBtn = document.querySelector(`.sort-btn[data-sort="${appState.sidebarSort}"]`);
    if (sortBtn) sortBtn.classList.add('active');

    // Sync search input
    const searchInput = document.querySelector('.sidebar-search');
    if (searchInput && appState.sidebarFilter) {
        searchInput.value = appState.sidebarFilter;
    }

    // Sync tab
    if (appState.activeTab !== 'overview') {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        const tabBtn = document.querySelector(`.tab-btn[data-tab="${appState.activeTab}"]`);
        if (tabBtn) tabBtn.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        const tabContent = document.getElementById(`tab-${appState.activeTab}`);
        if (tabContent) tabContent.classList.add('active');
    }

    // Sync filter badge
    updateFilterBadge();
}

function applyTheme() {
    if (appState.theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    document.getElementById('theme-toggle').textContent = appState.theme === 'dark' ? '\u263C' : '\u263E';
}

function toggleTheme() {
    appState.theme = appState.theme === 'dark' ? 'light' : 'dark';
    applyTheme();
    initChartDefaults();
    destroyAllCharts();
    renderTab(appState.activeTab);
    saveState();
}

function initDashboard() {
    appState.data = TRADE_DATA;
    document.getElementById('header-import-btn').style.display = '';
    loadState();
    applyTheme();
    initChartDefaults();
    renderHeader();
    populateGlobalFilters();
    applyRestoredState();
    renderSidebar();
    renderKPIs();
    renderTab(appState.activeTab);
}

function showImportOverlay() {
    document.querySelector('.sidebar').style.display = 'none';
    document.querySelector('.main').style.display = 'none';
    document.getElementById('import-overlay').style.display = 'flex';
    setupDropzone();
    applyTheme();
    renderFilterSummary();
}

function init() {
    if (typeof TRADE_DATA !== 'undefined') {
        initDashboard();
        return;
    }

    // Try restoring last imported data from IndexedDB
    idbLoadData().then(cached => {
        if (cached) {
            window.TRADE_DATA = cached;
            initDashboard();
        } else {
            showImportOverlay();
        }
    }).catch(() => {
        showImportOverlay();
    });
}

function initFromImport() {
    // Hide import panel, show dashboard
    document.getElementById('import-overlay').style.display = 'none';
    document.querySelector('.sidebar').style.display = '';
    document.querySelector('.main').style.display = '';
    document.getElementById('header-import-btn').style.display = '';

    // Save imported data to IndexedDB for next reload
    idbSaveData(TRADE_DATA).catch(() => { /* storage error — ignore */ });

    // Set new data, then restore saved filters/state
    appState.data = TRADE_DATA;
    appState.selectedStrategies = new Set(['_ALL']);
    appState.activeTab = 'overview';
    appState.sidebarSort = 'name';
    appState.sidebarFilter = '';
    appState.globalDirection = '';
    appState.globalInstruments = new Set();
    appState.globalHours = new Set();
    appState.globalDateFrom = '';
    appState.globalDateTo = '';
    loadState();
    destroyAllCharts();

    // Re-sync UI
    initChartDefaults();
    renderHeader();
    populateGlobalFilters();
    applyRestoredState();

    renderSidebar();
    renderKPIs();
    renderTab(appState.activeTab);
    saveState();
}

function showImportPanel() {
    destroyAllCharts();
    document.querySelector('.sidebar').style.display = 'none';
    document.querySelector('.main').style.display = 'none';
    document.getElementById('import-overlay').style.display = 'flex';
    document.getElementById('import-status').textContent = '';
    document.getElementById('import-status').className = 'import-status';
    // Reset file input
    const fileInput = document.getElementById('import-file-input');
    if (fileInput) fileInput.value = '';
    setupDropzone();
}

function setupDropzone() {
    const dropzone = document.getElementById('import-dropzone');
    if (!dropzone || dropzone._dropzoneSetup) return;
    dropzone._dropzoneSetup = true;

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleImport(file);
    });
}

// --- Header ---
function renderHeader() {
    const meta = appState.data.metadata;
    const dr = meta.dateRange;
    document.getElementById('date-range').textContent =
        `${formatDateFull(dr.start)} - ${formatDateFull(dr.end)}  |  ${meta.tradingDays} trading days  |  ${meta.totalTradesFiltered.toLocaleString()} trades`;
}

// --- Global Filters ---
function populateGlobalFilters() {
    // Build instrument checkbox list with preferred order
    const instruments = appState.data.metadata.instruments;
    const pinned = ['ES', 'MES', 'NQ', 'MNQ', 'RTY', 'CL', 'GC'];
    const pinnedSet = new Set(pinned);
    const sortedInstruments = [
        ...pinned.filter(i => instruments.includes(i)),
        ...instruments.filter(i => !pinnedSet.has(i)).sort((a, b) => a.localeCompare(b))
    ];
    const dd = document.getElementById('instrument-dropdown');
    dd.innerHTML = `<label class="hour-preset"><input type="checkbox" id="instrument-preset-all" onchange="onInstrumentPresetAll(this.checked)"> <strong>All</strong></label>`
        + `<label class="hour-preset"><input type="checkbox" id="instrument-preset-usual" onchange="onInstrumentPresetUsual(this.checked)"> <strong>Usual</strong></label>`
        + `<label class="hour-preset"><input type="checkbox" id="instrument-preset-micros" onchange="onInstrumentPresetMicros(this.checked)"> <strong>Micros</strong></label><div class="hour-preset-divider"></div>`
        + sortedInstruments.map(inst =>
        `<label><input type="checkbox" value="${inst}" onchange="onInstrumentToggle('${inst}', this.checked)"> ${inst}</label>`
    ).join('');

    // Build half-hour checkbox list from half-hours present in data
    const allTrades = appState.data.trades;
    const halfHoursPresent = [...new Set(allTrades.map(t => t.entryHalfHour || (String(t.entryHour).padStart(2, '0') + ':00')))].sort();
    const hourDD = document.getElementById('hour-dropdown');
    hourDD.innerHTML = `<label class="hour-preset"><input type="checkbox" id="hour-preset-all" onchange="onHourPresetAll(this.checked)"> <strong>All</strong></label>`
        + `<label class="hour-preset"><input type="checkbox" id="hour-preset-rth" onchange="onHourPresetRTH(this.checked)"> <strong>RTH (9:30–16:00)</strong></label><div class="hour-preset-divider"></div>`
        + halfHoursPresent.map(h =>
        `<label><input type="checkbox" value="${h}" onchange="onHourToggle('${h}', this.checked)"> ${formatHalfHourLabel(h)}</label>`
    ).join('');

    // Set date input bounds and defaults from data range
    const dr = appState.data.metadata.dateRange;
    const dateFrom = document.getElementById('global-date-from');
    const dateTo = document.getElementById('global-date-to');
    dateFrom.min = dr.start;
    dateFrom.max = dr.end;
    dateTo.min = dr.start;
    dateTo.max = dr.end;
    dateTo.value = dr.end;
    appState.globalDateTo = dr.end;
}

function onDirectionFilterChange() {
    setDirectionFilter(appState.globalDirection);
}

function setDirectionFilter(value) {
    appState.globalDirection = value;
    document.querySelectorAll('.direction-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.value === value);
    });
    saveState();
    applyGlobalFilters();
}

function toggleInstrumentDropdown() {
    document.getElementById('instrument-dropdown').classList.toggle('open');
}

function onInstrumentPresetAll(checked) {
    document.querySelectorAll('#instrument-dropdown input[value]').forEach(cb => {
        const inst = cb.value;
        if (checked) appState.globalInstruments.add(inst);
        else appState.globalInstruments.delete(inst);
        cb.checked = checked;
    });
    syncInstrumentPresets();
    updateInstrumentBtn();
    saveState();
    applyGlobalFilters();
}

const USUAL_INSTRUMENTS = new Set(['ES', 'MES', 'NQ', 'MNQ', 'RTY', 'M2K', 'CL', 'MCL', 'GC', 'MGC', 'YM', 'MYM']);
const MICRO_INSTRUMENTS = new Set(['MES', 'MNQ', 'M2K', 'MCL', 'MGC', 'MYM']);

function onInstrumentToggle(inst, checked) {
    if (checked) appState.globalInstruments.add(inst);
    else appState.globalInstruments.delete(inst);
    syncInstrumentPresets();
    updateInstrumentBtn();
    saveState();
    applyGlobalFilters();
}

function _applyInstrumentPreset(presetSet, checked) {
    document.querySelectorAll('#instrument-dropdown input[value]').forEach(cb => {
        const inst = cb.value;
        if (checked) {
            if (presetSet.has(inst)) { cb.checked = true; appState.globalInstruments.add(inst); }
            else { cb.checked = false; appState.globalInstruments.delete(inst); }
        } else {
            if (presetSet.has(inst)) { cb.checked = false; appState.globalInstruments.delete(inst); }
        }
    });
    syncInstrumentPresets();
    updateInstrumentBtn();
    saveState();
    applyGlobalFilters();
}

function onInstrumentPresetUsual(checked) {
    _applyInstrumentPreset(USUAL_INSTRUMENTS, checked);
}

function onInstrumentPresetMicros(checked) {
    _applyInstrumentPreset(MICRO_INSTRUMENTS, checked);
}

function syncInstrumentPresets() {
    const allCb = document.getElementById('instrument-preset-all');
    const usualCb = document.getElementById('instrument-preset-usual');
    const microsCb = document.getElementById('instrument-preset-micros');
    const allCheckboxes = document.querySelectorAll('#instrument-dropdown input[value]');
    const checkedSet = appState.globalInstruments;

    if (allCb) allCb.checked = allCheckboxes.length > 0 && [...allCheckboxes].every(cb => cb.checked);
    if (usualCb) usualCb.checked = [...USUAL_INSTRUMENTS].some(i => checkedSet.has(i)) && [...USUAL_INSTRUMENTS].filter(i => document.querySelector(`#instrument-dropdown input[value="${i}"]`)).every(i => checkedSet.has(i));
    if (microsCb) microsCb.checked = [...MICRO_INSTRUMENTS].some(i => checkedSet.has(i)) && [...MICRO_INSTRUMENTS].filter(i => document.querySelector(`#instrument-dropdown input[value="${i}"]`)).every(i => checkedSet.has(i));
}

function updateInstrumentBtn() {
    const btn = document.getElementById('instrument-filter-btn');
    const n = appState.globalInstruments.size;
    btn.textContent = n > 0 ? `Instruments (${n})` : 'All Trades';
}

function toggleHourDropdown() {
    document.getElementById('hour-dropdown').classList.toggle('open');
}

const RTH_HOURS = ['09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00'];

function onHourPresetAll(checked) {
    document.querySelectorAll('#hour-dropdown input[value]').forEach(cb => {
        const h = cb.value;
        if (checked) appState.globalHours.add(h);
        else appState.globalHours.delete(h);
        cb.checked = checked;
    });
    syncRTHPreset();
    updateHourBtn();
    saveState();
    applyGlobalFilters();
}

function onHourPresetRTH(checked) {
    RTH_HOURS.forEach(h => {
        if (checked) appState.globalHours.add(h);
        else appState.globalHours.delete(h);
        const cb = document.querySelector(`#hour-dropdown input[value="${h}"]`);
        if (cb) cb.checked = checked;
    });
    syncAllHoursPreset();
    updateHourBtn();
    saveState();
    applyGlobalFilters();
}

function onHourToggle(halfHour, checked) {
    if (checked) appState.globalHours.add(halfHour);
    else appState.globalHours.delete(halfHour);
    syncRTHPreset();
    syncAllHoursPreset();
    updateHourBtn();
    saveState();
    applyGlobalFilters();
}

function syncRTHPreset() {
    const rthCb = document.getElementById('hour-preset-rth');
    if (rthCb) {
        rthCb.checked = RTH_HOURS.every(h => appState.globalHours.has(h));
    }
}

function syncAllHoursPreset() {
    const allCb = document.getElementById('hour-preset-all');
    if (!allCb) return;
    const allCheckboxes = document.querySelectorAll('#hour-dropdown input[value]');
    allCb.checked = allCheckboxes.length > 0 && [...allCheckboxes].every(cb => cb.checked);
}

function updateHourBtn() {
    const btn = document.getElementById('hour-filter-btn');
    const n = appState.globalHours.size;
    btn.textContent = n > 0 ? `Time (${n})` : 'Time';
}

function onDateRangeChange() {
    appState.globalDateFrom = document.getElementById('global-date-from').value;
    appState.globalDateTo = document.getElementById('global-date-to').value;
    saveState();
    applyGlobalFilters();
}

function toggleDatePresets() {
    const dd = document.getElementById('date-presets-dropdown');
    dd.classList.toggle('open');
}

function applyDatePreset(preset) {
    const dr = appState.data.metadata.dateRange;
    // Use the data's end date as "today" reference since this is backtesting data
    const ref = new Date(dr.end + 'T00:00:00');
    let from = '', to = dr.end;

    const pad = (d) => d.toISOString().slice(0, 10);
    const addDays = (d, n) => { const r = new Date(d); r.setDate(r.getDate() + n); return r; };

    switch (preset) {
        case 'today':
            from = dr.end;
            to = dr.end;
            break;
        case 'yesterday': {
            const yd = pad(addDays(ref, -1));
            from = yd;
            to = yd;
            break;
        }
        case 'last7':
            from = pad(addDays(ref, -6));
            break;
        case 'last30':
            from = pad(addDays(ref, -29));
            break;
        case 'thisMonth':
            from = `${ref.getFullYear()}-${String(ref.getMonth() + 1).padStart(2, '0')}-01`;
            break;
        case 'lastMonth': {
            const lm = new Date(ref.getFullYear(), ref.getMonth() - 1, 1);
            const lmEnd = new Date(ref.getFullYear(), ref.getMonth(), 0);
            from = pad(lm);
            to = pad(lmEnd);
            break;
        }
        case 'thisYear':
            from = `${ref.getFullYear()}-01-01`;
            break;
        case 'all':
            from = '';
            to = dr.end;
            break;
    }

    // Clamp to data range
    if (from && from < dr.start) from = dr.start;
    if (to > dr.end) to = dr.end;

    document.getElementById('global-date-from').value = from;
    document.getElementById('global-date-to').value = to;
    appState.globalDateFrom = from;
    appState.globalDateTo = to;

    // Close dropdown
    document.getElementById('date-presets-dropdown').classList.remove('open');

    saveState();
    applyGlobalFilters();
}

// Close dropdowns when clicking outside
document.addEventListener('click', (e) => {
    const presetsDD = document.getElementById('date-presets-dropdown');
    const presetsBtn = document.getElementById('date-presets-btn');
    if (presetsDD && presetsBtn && !presetsDD.contains(e.target) && !presetsBtn.contains(e.target)) {
        presetsDD.classList.remove('open');
    }
    const instDD = document.getElementById('instrument-dropdown');
    const instBtn = document.getElementById('instrument-filter-btn');
    if (instDD && instBtn && !instDD.contains(e.target) && !instBtn.contains(e.target)) {
        instDD.classList.remove('open');
    }
    const hourDD = document.getElementById('hour-dropdown');
    const hourBtn = document.getElementById('hour-filter-btn');
    if (hourDD && hourBtn && !hourDD.contains(e.target) && !hourBtn.contains(e.target)) {
        hourDD.classList.remove('open');
    }
});


function applyGlobalFilters() {
    updateFilterBadge();
    renderSidebar();
    destroyAllCharts();
    renderKPIs();
    tradeLogState.page = 1;
    tradeLogState.filterVariant = '';
    renderTab(appState.activeTab);
}

function clearGlobalFilters() {
    const dr = appState.data.metadata.dateRange;
    document.querySelectorAll('.direction-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.value === '');
    });
    document.getElementById('global-date-from').value = '';
    document.getElementById('global-date-to').value = dr.end;
    appState.globalDirection = '';
    appState.globalDateFrom = '';
    appState.globalDateTo = dr.end;
    appState.globalInstruments.clear();
    document.querySelectorAll('#instrument-dropdown input').forEach(cb => cb.checked = false);
    updateInstrumentBtn();
    appState.globalHours.clear();
    document.querySelectorAll('#hour-dropdown input').forEach(cb => cb.checked = false);
    updateHourBtn();
    saveState();
    updateFilterBadge();
    renderSidebar();
    destroyAllCharts();
    renderKPIs();
    tradeLogState.page = 1;
    tradeLogState.filterVariant = '';
    renderTab(appState.activeTab);
}

function updateFilterBadge() {
    const dr = appState.data.metadata.dateRange;
    const hasDateFilter = (appState.globalDateFrom && appState.globalDateFrom > dr.start)
        || (appState.globalDateTo && appState.globalDateTo < dr.end);
    const hasFilter = appState.globalDirection
        || appState.globalInstruments.size > 0
        || appState.globalHours.size > 0
        || hasDateFilter;
    document.getElementById('clear-filters-btn').style.display = hasFilter ? 'inline-block' : 'none';
    renderFilterSummary();
}

function renderFilterSummary() {
    const el = document.getElementById('filter-summary');
    if (!el) return;
    if (!appState.data) {
        el.innerHTML = '<span class="filter-summary-label">All Trades</span>';
        return;
    }
    const dr = appState.data.metadata.dateRange;
    const chips = [];
    const x = (fn) => `<button class="filter-chip-x" onclick="${fn}" title="Remove filter">&times;</button>`;

    if (appState.globalDirection) {
        chips.push(`<span class="filter-chip filter-chip-direction"><span class="filter-chip-label">Dir:</span> ${appState.globalDirection}${x('clearFilterDirection()')}</span>`);
    }

    if (appState.globalInstruments.size > 0) {
        const list = [...appState.globalInstruments].sort().join(', ');
        chips.push(`<span class="filter-chip filter-chip-instrument"><span class="filter-chip-label">Instr:</span> ${list}${x('clearFilterInstruments()')}</span>`);
    }

    if (appState.globalHours.size > 0) {
        const sorted = [...appState.globalHours].sort();
        let label;
        if (sorted.length <= 4) {
            label = sorted.map(h => formatHalfHourLabel(h)).join(', ');
        } else {
            label = formatHalfHourLabel(sorted[0]) + ' \u2013 ' + formatHalfHourLabel(sorted[sorted.length - 1]) + ` (${sorted.length})`;
        }
        chips.push(`<span class="filter-chip filter-chip-time"><span class="filter-chip-label">Time:</span> ${label}${x('clearFilterTime()')}</span>`);
    }

    const hasDateFrom = appState.globalDateFrom && appState.globalDateFrom > dr.start;
    const hasDateTo = appState.globalDateTo && appState.globalDateTo < dr.end;
    if (hasDateFrom || hasDateTo) {
        const from = hasDateFrom ? formatDateFull(appState.globalDateFrom) : dr.start;
        const to = hasDateTo ? formatDateFull(appState.globalDateTo) : dr.end;
        chips.push(`<span class="filter-chip filter-chip-date"><span class="filter-chip-label">Date:</span> ${from} \u2013 ${to}${x('clearFilterDate()')}</span>`);
    }

    if (chips.length === 0) {
        el.innerHTML = '<span class="filter-summary-label">All Trades</span>';
        return;
    }

    el.innerHTML = '<span class="filter-summary-label">Active</span>' + chips.join('<span class="filter-chip-sep">|</span>');
}

function clearFilterDirection() {
    appState.globalDirection = '';
    document.querySelectorAll('.direction-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.value === '');
    });
    saveState();
    applyGlobalFilters();
}

function clearFilterInstruments() {
    appState.globalInstruments.clear();
    document.querySelectorAll('#instrument-dropdown input').forEach(cb => cb.checked = false);
    updateInstrumentBtn();
    saveState();
    applyGlobalFilters();
}

function clearFilterTime() {
    appState.globalHours.clear();
    document.querySelectorAll('#hour-dropdown input').forEach(cb => cb.checked = false);
    updateHourBtn();
    saveState();
    applyGlobalFilters();
}

function clearFilterDate() {
    const dr = appState.data.metadata.dateRange;
    appState.globalDateFrom = '';
    appState.globalDateTo = dr.end;
    document.getElementById('global-date-from').value = '';
    document.getElementById('global-date-to').value = dr.end;
    saveState();
    applyGlobalFilters();
}

// --- Sidebar ---
function renderSidebar() {
    const strategies = appState.data.metadata.strategies;
    const precomputed = appState.data.strategies;
    const sidebarOverride = getSidebarMetrics(); // null if no global filter

    let items = strategies.map(name => {
        if (sidebarOverride) {
            const fm = sidebarOverride.families[name];
            return {
                key: name, name: name,
                trades: fm ? fm.count : 0,
                pnl: fm ? Math.round(fm.pnl * 100) / 100 : 0,
                winRate: fm ? fm.winRate : 0,
                variants: precomputed[name].subStrategies ? precomputed[name].subStrategies.length : 0,
            };
        }
        return {
            key: name, name: name,
            trades: precomputed[name].tradeCount,
            pnl: precomputed[name].totalPnL,
            winRate: precomputed[name].winRate,
            variants: precomputed[name].subStrategies ? precomputed[name].subStrategies.length : 0,
        };
    });

    // Filter by search
    if (appState.sidebarFilter) {
        const q = appState.sidebarFilter.toLowerCase();
        items = items.filter(i => i.name.toLowerCase().includes(q));
    }

    // Sort
    switch (appState.sidebarSort) {
        case 'name': items.sort((a, b) => a.name.localeCompare(b.name)); break;
        case 'pnl': items.sort((a, b) => b.pnl - a.pnl); break;
        case 'trades': items.sort((a, b) => b.trades - a.trades); break;
        case 'wr': items.sort((a, b) => b.winRate - a.winRate); break;
    }

    // All strategies summary
    const allMeta = sidebarOverride ? sidebarOverride._ALL : { pnl: precomputed._ALL.totalPnL, count: precomputed._ALL.tradeCount };
    const isAllSelected = appState.selectedStrategies.has('_ALL');

    const list = document.getElementById('strategy-list');
    let html = `<li class="strategy-item ${isAllSelected ? 'active' : ''}" onclick="toggleStrategy('_ALL')">
        <span class="strategy-checkbox">${isAllSelected ? '&#9745;' : '&#9744;'}</span>
        <span class="name">All</span>
        <span class="meta"><span class="trades-badge">${allMeta.count}</span> <span class="pnl-badge ${profitClass(allMeta.pnl)}">${formatCurrencyShort(allMeta.pnl)}</span></span>
    </li>`;

    items.forEach(item => {
        if (sidebarOverride && item.trades === 0) return; // hide accounts with no matching trades
        const variantLabel = item.variants > 1 ? `<span style="color:var(--text-muted);font-size:10px"> (${item.variants})</span>` : '';
        const isSelected = appState.selectedStrategies.has(item.key);
        html += `<li class="strategy-item ${isSelected ? 'active' : ''}" onclick="toggleStrategy('${item.key}')">
            <span class="strategy-checkbox">${isSelected ? '&#9745;' : '&#9744;'}</span>
            <span class="name" title="${item.key}">${item.name}${variantLabel}</span>
            <span class="meta"><span class="trades-badge">${item.trades}</span> <span class="pnl-badge ${profitClass(item.pnl)}">${formatCurrencyShort(item.pnl)}</span></span>
        </li>`;
    });

    list.innerHTML = html;

    // Update selection count badge
    updateSelectionBadge();
}

function onSidebarSearch(val) {
    appState.sidebarFilter = val;
    saveState();
    renderSidebar();
}

function onSidebarSort(sortBy) {
    appState.sidebarSort = sortBy;
    saveState();
    document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.sort-btn[data-sort="${sortBy}"]`).classList.add('active');
    renderSidebar();
}

// --- KPI Cards ---
function renderKPIs() {
    if (appState.chartInstances['spark-pnl']) {
        appState.chartInstances['spark-pnl'].destroy();
        delete appState.chartInstances['spark-pnl'];
    }

    const m = getActiveMetrics();
    if (!m || m.tradeCount === 0) {
        ['kpi-pnl', 'kpi-pf', 'kpi-winrate', 'kpi-trades', 'kpi-sharpe', 'kpi-dd'].forEach(id => {
            document.getElementById(id).innerHTML = '<div class="label">-</div><div class="value text-neutral">No trades</div>';
        });
        return;
    }

    document.getElementById('kpi-pnl').innerHTML = `
        <div class="label">Total P&L</div>
        <div class="value ${profitTextClass(m.totalPnL)}">${formatCurrency(m.totalPnL)}</div>
        <div class="sub">Avg: ${formatCurrency(m.avgTrade)}/trade</div>
        <div class="kpi-sparkline"><canvas id="spark-pnl"></canvas></div>`;

    document.getElementById('kpi-pf').innerHTML = `
        <div class="label">Profit Factor</div>
        <div class="value" style="color:${pfColor(m.profitFactor)}">${m.profitFactor >= 9999 ? 'Inf' : m.profitFactor.toFixed(2)}</div>
        <div class="sub">Avg Win: ${formatCurrency(m.avgWin)} | Loss: ${formatCurrency(m.avgLoss)}</div>`;

    document.getElementById('kpi-winrate').innerHTML = `
        <div class="label">Win Rate</div>
        <div class="value" style="color:${m.winRate >= 50 ? 'var(--profit)' : 'var(--loss)'}">${formatPercent(m.winRate)}</div>
        <div class="sub">${m.winCount}W / ${m.lossCount}L</div>`;

    document.getElementById('kpi-trades').innerHTML = `
        <div class="label">Total Trades</div>
        <div class="value text-accent">${m.tradeCount.toLocaleString()}</div>
        <div class="sub">${m.breakEvenCount} breakeven</div>`;

    document.getElementById('kpi-sharpe').innerHTML = `
        <div class="label">Sharpe Ratio</div>
        <div class="value" style="color:${sharpeColor(m.sharpeRatio)}">${m.sharpeRatio != null ? m.sharpeRatio.toFixed(2) : 'N/A'}</div>
        <div class="sub">Annualized (252 days)</div>`;

    document.getElementById('kpi-dd').innerHTML = `
        <div class="label">Max Drawdown</div>
        <div class="value text-loss">${formatCurrency(m.maxDrawdown)}</div>
        <div class="sub">Streaks: ${m.maxConsecWins}W / ${m.maxConsecLosses}L</div>`;

    // P&L sparkline — equity curve
    if (m.equityCurve && m.equityCurve.length > 1) {
        const values = downsampleArray(m.equityCurve.map(d => d[1]), 60);
        appState.chartInstances['spark-pnl'] = renderSparkline('spark-pnl', values, getCSSVar('--accent'),
            (ctx) => formatCurrency(ctx.parsed.y));
    }
}

// --- Tab Switching ---
function switchTab(tabName) {
    appState.activeTab = tabName;
    saveState();
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.tab-btn[data-tab="${tabName}"]`).classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    destroyAllCharts();
    renderKPIs();
    renderTab(tabName);
}

function toggleStrategy(key) {
    if (key === '_ALL') {
        appState.selectedStrategies = new Set(['_ALL']);
    } else {
        appState.selectedStrategies.delete('_ALL');

        if (appState.selectedStrategies.has(key)) {
            appState.selectedStrategies.delete(key);
            if (appState.selectedStrategies.size === 0) {
                appState.selectedStrategies.add('_ALL');
            }
        } else {
            appState.selectedStrategies.add(key);
        }

        const allStrategies = appState.data.metadata.strategies;
        if (appState.selectedStrategies.size === allStrategies.length) {
            appState.selectedStrategies = new Set(['_ALL']);
        }
    }

    saveState();
    renderSidebar();
    destroyAllCharts();
    renderKPIs();
    tradeLogState.page = 1;
    tradeLogState.filterVariant = '';
    renderTab(appState.activeTab);
}

function selectAllStrategies() {
    appState.selectedStrategies = new Set(['_ALL']);
    saveState();
    renderSidebar();
    destroyAllCharts();
    renderKPIs();
    tradeLogState.page = 1;
    tradeLogState.filterVariant = '';
    renderTab(appState.activeTab);
}

function clearStrategySelection() {
    appState.selectedStrategies = new Set(['_ALL']);
    appState.sidebarFilter = '';
    const searchInput = document.querySelector('.sidebar-search');
    if (searchInput) searchInput.value = '';
    saveState();
    renderSidebar();
    destroyAllCharts();
    renderKPIs();
    tradeLogState.page = 1;
    tradeLogState.filterVariant = '';
    renderTab(appState.activeTab);
}

function updateSelectionBadge() {
    const badge = document.getElementById('selection-count');
    if (!badge) return;
    if (appState.selectedStrategies.has('_ALL')) {
        badge.textContent = 'All';
        badge.style.display = 'none';
    } else {
        const count = appState.selectedStrategies.size;
        badge.textContent = `${count} selected`;
        badge.style.display = 'inline-block';
    }
}

function destroyAllCharts() {
    Object.values(appState.chartInstances).forEach(c => {
        if (c && typeof c.destroy === 'function') c.destroy();
    });
    appState.chartInstances = {};
}

function renderTab(tabName) {
    // Strategies tab doesn't need trade metrics
    if (tabName === 'strategies') {
        renderStrategyConfigs('strategy-config-container');
        return;
    }

    const m = getActiveMetrics();
    if (!m || m.tradeCount === 0) return;

    switch (tabName) {
        case 'calendar':
            renderCalendar('calendar-container', m.dailyPnL);
            break;

        case 'overview':
            appState.chartInstances.equity = renderEquityCurve('chart-equity', m.equityCurve);
            renderSubStrategyTable('sub-strategy-container', m.subStrategies);
            appState.chartInstances.dailyPnl = renderDailyPnL('chart-daily-pnl', m.dailyPnL);
            appState.chartInstances.dist = renderProfitDistribution('chart-distribution', m.profitDistribution);
            appState.chartInstances.longShort = renderLongVsShort('chart-long-short', m.longShort);
            appState.chartInstances.longShortWr = renderLongVsShortWinRate('chart-long-short-wr', m.longShort);
            appState.chartInstances.instPnl = renderInstrumentPnLBar('chart-inst-pnl', m.instrumentPnL, true, appState.data.metadata.instruments);
            break;

        case 'time':
            appState.chartInstances.hourlyPnl = renderHourlyPnL('chart-hourly-pnl', m.hourlyPnL, m.hourlyTradeCount);
            appState.chartInstances.hourlyWR = renderHourlyWinRate('chart-hourly-wr', m.hourlyWinRate);
            appState.chartInstances.dowPnl = renderDowPnL('chart-dow-pnl', m.dowPnL);
            renderHeatmapTable('heatmap-container', m.hourDayMatrix);
            break;

        case 'instruments':
            appState.chartInstances.instPnl2 = renderInstrumentPnLBar('chart-inst-pnl2', m.instrumentPnL, true, appState.data.metadata.instruments);
            renderInstrumentTable('inst-table-container', m.instrumentDetails);
            break;

        case 'risk':
            appState.chartInstances.maeScatter = renderScatterMAE('chart-mae', m.maeVsProfit);
            appState.chartInstances.mfeScatter = renderScatterMFE('chart-mfe', m.mfeVsProfit);
            appState.chartInstances.rollingPnl = renderRollingPnL('chart-rolling-pnl', m.rollingPnL20);
            appState.chartInstances.rollingWR = renderRollingWinRate('chart-rolling-wr', m.rollingWinRate50);
            appState.chartInstances.streaks = renderStreaks('chart-streaks', m.streaks);
            break;

        case 'trades':
            renderTradeLog('tab-trades');
            break;
    }
}

// --- Strategy Configurations Table ---
function renderStrategyConfigs(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const configs = (typeof STRATEGY_CONFIGS !== 'undefined') ? STRATEGY_CONFIGS : [];
    if (configs.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted)">No strategy configs loaded. Run <code>python ingest.py --ninjatrader</code> to export.</p>';
        return;
    }

    let html = '<div style="overflow-x:auto"><table class="trade-table">';
    html += '<thead><tr>';
    html += '<th>Strategy</th><th>Instrument</th><th>Timeframe</th>';
    html += '<th>Qty</th><th>Long</th><th>Short</th>';
    html += '<th>Session Start</th><th>Session End</th><th>Account</th>';
    html += '</tr></thead><tbody>';

    for (const c of configs) {
        const longBadge = c.allowLongs
            ? '<span style="color: var(--green)">Yes</span>'
            : '<span style="color: var(--text-muted)">No</span>';
        const shortBadge = c.allowShorts
            ? '<span style="color: var(--red)">Yes</span>'
            : '<span style="color: var(--text-muted)">No</span>';

        html += '<tr>';
        html += `<td><strong>${c.name}</strong></td>`;
        html += `<td>${c.instrument}</td>`;
        html += `<td>${c.timeframe}</td>`;
        html += `<td style="text-align:center">${c.qty}</td>`;
        html += `<td style="text-align:center">${longBadge}</td>`;
        html += `<td style="text-align:center">${shortBadge}</td>`;
        html += `<td>${c.sessionStart}</td>`;
        html += `<td>${c.sessionEnd}</td>`;
        html += `<td>${c.account}</td>`;
        html += '</tr>';
    }

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// --- Sub-Strategy Comparison Table ---
function renderSubStrategyTable(containerId, subStrategies) {
    const container = document.getElementById(containerId);
    const card = container.closest('.chart-card');
    // Hide the entire card when there's 0 or 1 sub-strategy (nothing to compare)
    const validSubs = (subStrategies || []).filter(s => s.trades > 0);
    if (validSubs.length <= 1) {
        container.innerHTML = '';
        if (card) card.style.display = 'none';
        return;
    }
    if (card) card.style.display = '';

    const sel = appState.selectedStrategies;
    const isAll = sel.has('_ALL');
    const isSingle = !isAll && sel.size === 1;
    const nameHeader = (isAll || !isSingle) ? 'Account' : 'Variant';

    const sorted = [...validSubs].sort((a, b) => b.totalPnL - a.totalPnL);

    let html = `<div class="data-table-container"><table class="data-table">
        <thead><tr>
            <th>${nameHeader}</th><th class="num">Trades</th><th class="num">Win Rate</th>
            <th class="num">Avg Trade</th><th class="num">Avg Win</th><th class="num">Avg Loss</th>
            <th class="num">P&L</th><th class="num">PF</th><th class="num">Max DD</th>
        </tr></thead><tbody>`;

    sorted.forEach(s => {
        if (s.trades === 0) return; // skip empty after filtering
        const pf = s.profitFactor >= 9999 ? 'Inf' : s.profitFactor.toFixed(2);
        const displayName = (isAll || !isSingle) ? s.name : s.name.replace('Sim-', '');
        html += `<tr>
            <td class="fw-600">${displayName}</td>
            <td class="num">${s.trades}</td>
            <td class="num ${s.winRate >= 50 ? 'text-profit' : 'text-loss'}">${s.winRate.toFixed(1)}%</td>
            <td class="num ${profitTextClass(s.avgTrade)}">${formatCurrency(s.avgTrade)}</td>
            <td class="num text-profit">${formatCurrency(s.avgWin)}</td>
            <td class="num text-loss">${formatCurrency(s.avgLoss)}</td>
            <td class="num ${profitTextClass(s.totalPnL)} fw-600">${formatCurrency(s.totalPnL)}</td>
            <td class="num" style="color:${pfColor(s.profitFactor)}">${pf}</td>
            <td class="num text-loss">${formatCurrency(s.maxDrawdown)}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    container.innerHTML = html;
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
