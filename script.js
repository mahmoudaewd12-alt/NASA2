/* ==========================================================================
   Lunar Landing Site Navigator — script.js
   Vanilla JS. Loads the real CSV, places markers on the real map image,
   and calls the existing FastAPI backend (POST /predict).
   ========================================================================== */
(() => {
  'use strict';

  /* ======================================================================
     1. CONFIGURATION — edit these values to match your project
     ====================================================================== */
  const CONFIG = {
    // FastAPI backend (started with: uvicorn api:app --reload --port 8000).
    // You can also override at runtime:  index.html?api=http://127.0.0.1:9000
    API_BASE_URL: new URLSearchParams(location.search).get('api') || 'http://127.0.0.1:8000',
    PREDICT_PATH: '/predict',          // POST endpoint defined in api.py
    HEALTH_PATH: '/openapi.json',      // GET; FastAPI serves this by default
    REQUEST_TIMEOUT_MS: 10000,

    // Project assets (keep them in the same folder as index.html)
    CSV_URL: 'nasa_moon_trek_lunar_regions_features.csv',
    MAP_URL: '6037101826668171398_121.jpg',
    /* Map calibration. The image is an equirectangular map (1287 x 652 px).
       Pixel positions of its lat/long grid were measured from the image:
         vertical lines   x =    2 (180°W) ... 643 (0°) ... 1283 (180°E)
         horizontal lines y =    8 (90°N)  ... 328 (0°) ...  650 (90°S)
       x = X0 + lon * KX ;  y = Y0 - lat * KY                              */
    MAP: { width: 1287, height: 652, x0: 642.5, kx: (1283 - 2) / 360, y0: 329, ky: (650 - 8) / 180, maxScale: 5 },

    // Labels printed on the map image that have NO row in the CSV (cannot be selected).
    IMAGE_LABELS_WITHOUT_DATA: ['Okina', 'Hertzsprung', 'Ranger 4', 'Apollo'],

    POLAR_LAT: -80,                    // UI filter only: "south of −80° latitude"
    COMPARE_MAX: 4,
    INTRO_MS: 2600,

    // Class bands from the project's training notebook (the API itself does not return them).
    TRAINING_BANDS: [35, 55, 75],      // Poor < 35 <= Moderate < 55 <= Good < 75 <= Excellent
  };

  /* ======================================================================
     2. DATA SCHEMA (verified against the real CSV header and api.py)
     ====================================================================== */
  const NAME_COL = 'Region_Name';

  // group / label / unit / digits, plus how each field relates to the model
  const FIELDS = [
    { key: 'latitude_deg',             label: 'Latitude',          unit: '°',  digits: 4, group: 'Position',                 model: true,  icon: 'pin' },
    { key: 'east_longitude_deg',       label: 'East Longitude',    unit: '°',  digits: 4, group: 'Position',                 model: true,  icon: 'pin' },
    { key: 'elevation_m',              label: 'Elevation',         unit: ' m', digits: 0, group: 'Terrain',                  model: false, icon: 'mountain' },
    { key: 'slope_deg',                label: 'Slope',             unit: '°',  digits: 1, group: 'Terrain',                  model: false, icon: 'slope' },
    { key: 'illumination_percent',     label: 'Illumination',      unit: '%',  digits: 2, group: 'Illumination & Earth link', model: true,  icon: 'sun' },
    { key: 'sun_elevation_deg',        label: 'Sun Elevation',     unit: '°',  digits: 2, group: 'Illumination & Earth link', model: true,  icon: 'sun' },
    { key: 'sun_azimuth_deg',          label: 'Sun Azimuth',       unit: '°',  digits: 2, group: 'Illumination & Earth link', model: true,  icon: 'compass' },
    { key: 'earth_visibility_percent', label: 'Earth Visibility',  unit: '%',  digits: 1, group: 'Illumination & Earth link', model: true,  icon: 'earth' },
  ];
  const BOOL_COL = 'earth_visible';
  const REQUIRED_COLUMNS = [NAME_COL, ...FIELDS.map(f => f.key), BOOL_COL];

  // Exactly the features api.py's model uses (api.py: bundle["features"]).
  // slope_deg and elevation_m are accepted by the API but NOT used by the model, so they are not sent.
  // The backend derives `earth_visible` itself. All six are required here because the API would
  // otherwise silently substitute defaults (azimuth 180, lat 0, lon 0) for missing optional fields.
  const MODEL_INPUTS = [
    { key: 'sun_elevation_deg',        label: 'Sun elevation',     unit: '°', min: -90,  max: 90 },
    { key: 'sun_azimuth_deg',          label: 'Sun azimuth',       unit: '°', min: 0,    max: 360 },
    { key: 'latitude_deg',             label: 'Latitude',          unit: '°', min: -90,  max: 90 },
    { key: 'east_longitude_deg',       label: 'East longitude',    unit: '°', min: -180, max: 360 },
    { key: 'illumination_percent',     label: 'Illumination',      unit: '%', min: 0,    max: 100 },
    { key: 'earth_visibility_percent', label: 'Earth visibility',  unit: '%', min: 0,    max: 100 },
  ];

  /* ======================================================================
     3. STATE & HELPERS
     ====================================================================== */
  const state = {
    regions: [],
    selectedId: null,
    compare: [],
    analysisId: null,
    predictions: new Map(),     // id -> { status:'loading'|'ok'|'error', payload, data, error, at }
    filter: '', polarOnly: false,
    view: { scale: 1, tx: 0, ty: 0, fit: 1 },
    api: 'checking',
  };

  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const icon = n => `<svg class="icon" aria-hidden="true"><use href="#i-${n}"/></svg>`;
  const NA = 'Not Available';
  const byId = id => state.regions.find(r => r.id === id);
  const announce = msg => { const el = $('#liveRegion'); el.textContent = ''; setTimeout(() => { el.textContent = msg; }, 30); };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  function fmt(value, f) {
    if (value === null || value === undefined || Number.isNaN(value)) return null;
    return Number(value).toFixed(f.digits).replace(/^-/, '−') + f.unit;
  }
  const fmtCoord = (lat, lon) => `${lat.toFixed(2).replace('-', '−')}°, ${lon.toFixed(2).replace('-', '−')}°E`;

  /* ======================================================================
     4. CSV LOADING (real file, parsed at runtime — nothing hardcoded)
     ====================================================================== */
  function parseCSV(text) {
    text = text.replace(/^\uFEFF/, '');
    const rows = []; let row = [], cell = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) {
        if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; }
        else cell += c;
      } else if (c === '"') q = true;
      else if (c === ',') { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(cell); cell = '';
        if (row.some(v => v.trim() !== '')) rows.push(row);
        row = [];
      } else cell += c;
    }
    row.push(cell);
    if (row.some(v => v.trim() !== '')) rows.push(row);
    return rows;
  }
  const toNum = v => { if (v === undefined || v === null || String(v).trim() === '') return null; const n = Number(v); return Number.isFinite(n) ? n : null; };
  const toBool = v => { const s = String(v ?? '').trim().toLowerCase(); if (['true', '1', 'yes'].includes(s)) return true; if (['false', '0', 'no'].includes(s)) return false; return null; };

  function project(lat, lon) { const m = CONFIG.MAP; return { x: m.x0 + lon * m.kx, y: m.y0 - lat * m.ky }; }
  function unproject(x, y) { const m = CONFIG.MAP; return { lon: (x - m.x0) / m.kx, lat: (m.y0 - y) / m.ky }; }

  async function loadRegions() {
    let res;
    try { res = await fetch(CONFIG.CSV_URL, { cache: 'no-store' }); }
    catch (e) { throw new Error(location.protocol === 'file:' ? 'file-protocol' : 'network'); }
    if (!res.ok) throw new Error('http-' + res.status);
    const rows = parseCSV(await res.text());
    if (rows.length < 2) throw new Error('empty');
    const header = rows[0].map(h => h.trim());
    const idx = Object.fromEntries(header.map((h, i) => [h, i]));
    const missing = REQUIRED_COLUMNS.filter(c => !(c in idx));
    if (!(NAME_COL in idx)) throw new Error('no-name-column');

    const regions = rows.slice(1).map((r, i) => {
      const get = k => (k in idx ? r[idx[k]] : undefined);
      const reg = { id: 'r' + i, csvRow: i + 2, name: String(get(NAME_COL) ?? '').trim() || `Region ${i + 1}`, values: {} };
      FIELDS.forEach(f => { reg.values[f.key] = toNum(get(f.key)); });
      reg.values[BOOL_COL] = toBool(get(BOOL_COL));
      const lat = reg.values.latitude_deg, lon = reg.values.east_longitude_deg;
      reg.pos = (lat !== null && lon !== null && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 360) ? project(lat, lon) : null;
      return reg;
    });
    return { regions, missing };
  }

  /* ======================================================================
     5. INTRO
     ====================================================================== */
  function runIntro() {
    const intro = $('#intro');
    $('#introMoon').style.setProperty('--tex', `url("${CONFIG.MAP_URL}")`);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { intro.remove(); document.body.classList.remove('intro-active'); return; }
    intro.classList.add('play');
    let finished = false;
    const finish = () => {
      if (finished) return; finished = true;
      intro.classList.add('done'); document.body.classList.remove('intro-active');
      setTimeout(() => intro.remove(), 700);
      document.removeEventListener('keydown', onKey);
    };
    const onKey = () => finish();
    $('#introSkip').addEventListener('click', e => { e.stopPropagation(); finish(); });
    intro.addEventListener('click', finish);
    document.addEventListener('keydown', onKey);
    setTimeout(finish, CONFIG.INTRO_MS);
  }

  /* ======================================================================
     6. NAVIGATION
     ====================================================================== */
  function initNav() {
    const toggle = $('#navToggle'), links = $('#navLinks');
    toggle.addEventListener('click', () => {
      const open = links.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
    links.addEventListener('click', e => { if (e.target.closest('a')) { links.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false'); } });

    const map = new Map($$('[data-nav]').map(a => [a.dataset.nav, a]));
    const obs = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) { map.forEach(a => a.classList.remove('active')); map.get(en.target.id)?.classList.add('active'); }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    map.forEach((_, id) => { const s = document.getElementById(id); if (s) obs.observe(s); });
  }

  /* ======================================================================
     7. MAP (zoom / pan / markers)
     ====================================================================== */
  const vp = () => $('#mapViewport');
  const stage = () => $('#mapStage');

  function applyView(animate = false) {
    const v = state.view, st = stage();
    st.classList.toggle('animating', animate);
    st.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`;
    st.style.setProperty('--inv', String(1 / v.scale));
    $('#zoomReadout').textContent = Math.round(v.scale / v.fit * 100) + '%';
  }
  function clampView() {
    const v = state.view, w = CONFIG.MAP.width * v.scale, h = CONFIG.MAP.height * v.scale, vw = vp().clientWidth, vh = vp().clientHeight;
    v.tx = w <= vw ? (vw - w) / 2 : clamp(v.tx, vw - w, 0);
    v.ty = h <= vh ? (vh - h) / 2 : clamp(v.ty, vh - h, 0);
  }
  function fitView(animate = false) {
    const v = state.view; v.fit = vp().clientWidth / CONFIG.MAP.width; v.scale = v.fit; v.tx = 0; v.ty = 0; clampView(); applyView(animate);
  }
  function zoomAt(cx, cy, factor, animate = false) {
    const v = state.view, rect = vp().getBoundingClientRect(), px = cx - rect.left, py = cy - rect.top;
    const next = clamp(v.scale * factor, v.fit, CONFIG.MAP.maxScale);
    const ratio = next / v.scale;
    v.tx = px - (px - v.tx) * ratio; v.ty = py - (py - v.ty) * ratio; v.scale = next;
    clampView(); applyView(animate);
  }
  function zoomCenter(factor) { const r = vp().getBoundingClientRect(); zoomAt(r.left + r.width / 2, r.top + r.height / 2, factor, true); }
  function focusRegion(reg) {
    if (!reg.pos) return;
    const v = state.view, target = Math.max(v.scale, Math.min(CONFIG.MAP.maxScale, v.fit * 2.6));
    v.scale = target; v.tx = vp().clientWidth / 2 - reg.pos.x * target; v.ty = vp().clientHeight / 2 - reg.pos.y * target;
    clampView(); applyView(true);
  }

  function buildMarkers() {
    const layer = $('#markerLayer'); layer.innerHTML = '';
    state.regions.forEach(reg => {
      if (!reg.pos) return;
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'marker'; b.dataset.id = reg.id;
      b.style.left = reg.pos.x + 'px'; b.style.top = reg.pos.y + 'px';
      b.setAttribute('aria-label', `${reg.name}, latitude ${reg.values.latitude_deg}, east longitude ${reg.values.east_longitude_deg}`);
      b.innerHTML = `<span class="marker-tip">${esc(reg.name)}</span>`;
      layer.appendChild(b);
    });
    layer.addEventListener('click', e => {
      const m = e.target.closest('.marker'); if (m) selectRegion(m.dataset.id, { focus: false, scroll: false });
    });
  }

  function initMapInteraction() {
    const el = vp(), pointers = new Map();
    let drag = null, pinch = null, moved = false;

    el.addEventListener('wheel', e => {
      const v = state.view, zoomingOut = e.deltaY > 0;
      if (zoomingOut && v.scale <= v.fit + 1e-3) return;          // let the page scroll
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0016)));
    }, { passive: false });

    el.addEventListener('pointerdown', e => {
      if (e.target.closest('.map-msg')) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      stage().classList.remove('animating');
      if (pointers.size === 1) { drag = { x: e.clientX, y: e.clientY, tx: state.view.tx, ty: state.view.ty }; moved = false; }
      else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = { d: Math.hypot(a.x - b.x, a.y - b.y) }; drag = null; moved = true;
        [...pointers.keys()].forEach(id => { try { el.setPointerCapture(id); } catch (_) {} });
      }
    });
    el.addEventListener('pointermove', e => {
      // coordinate readout under the cursor
      const rect = el.getBoundingClientRect(), v = state.view;
      const ix = (e.clientX - rect.left - v.tx) / v.scale, iy = (e.clientY - rect.top - v.ty) / v.scale;
      const { lat, lon } = unproject(ix, iy);
      $('#cursorReadout').textContent = (ix >= 0 && iy >= 0 && ix <= CONFIG.MAP.width && iy <= CONFIG.MAP.height && Math.abs(lat) <= 90.5)
        ? `Cursor: ${lat.toFixed(1).replace('-', '−')}° lat, ${lon.toFixed(1).replace('-', '−')}° E` : 'Cursor: —';

      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2 && pinch) {
        const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch.d > 0) zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, d / pinch.d);
        pinch.d = d; return;
      }
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (!moved && Math.hypot(dx, dy) > 4) { moved = true; el.classList.add('dragging'); try { el.setPointerCapture(e.pointerId); } catch (_) {} }
        if (moved) { v.tx = drag.tx + dx; v.ty = drag.ty + dy; clampView(); applyView(); }
      }
    });
    const end = e => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = null;
      if (pointers.size === 0) { drag = null; el.classList.remove('dragging'); }
    };
    el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => { if (!pointers.size) $('#cursorReadout').textContent = 'Cursor: —'; });
    el.addEventListener('dblclick', e => { if (!e.target.closest('.marker')) zoomAt(e.clientX, e.clientY, 1.8, true); });

    el.addEventListener('keydown', e => {
      if (e.target !== el) return;
      const step = 60, v = state.view;
      const acts = { '+': () => zoomCenter(1.4), '=': () => zoomCenter(1.4), '-': () => zoomCenter(1 / 1.4), '0': () => fitView(true),
        ArrowLeft: () => { v.tx += step; }, ArrowRight: () => { v.tx -= step; }, ArrowUp: () => { v.ty += step; }, ArrowDown: () => { v.ty -= step; } };
      if (acts[e.key]) { e.preventDefault(); acts[e.key](); if (e.key.startsWith('Arrow')) { clampView(); applyView(true); } }
    });

    $('#zoomIn').addEventListener('click', () => zoomCenter(1.5));
    $('#zoomOut').addEventListener('click', () => zoomCenter(1 / 1.5));
    $('#zoomFit').addEventListener('click', () => fitView(true));
    new ResizeObserver(() => { const rel = state.view.scale / state.view.fit; state.view.fit = el.clientWidth / CONFIG.MAP.width; state.view.scale = clamp(state.view.fit * rel, state.view.fit, CONFIG.MAP.maxScale); clampView(); applyView(); }).observe(el);
  }

  /* ======================================================================
     8. REGION LIST + SELECTION
     ====================================================================== */
  const isPolar = r => r.values.latitude_deg !== null && r.values.latitude_deg < CONFIG.POLAR_LAT;
  const matchesFilter = r => (!state.polarOnly || isPolar(r)) && r.name.toLowerCase().includes(state.filter.trim().toLowerCase());

  function renderRegionList() {
    const list = $('#regionList'), shown = state.regions.filter(matchesFilter);
    list.innerHTML = shown.length ? shown.map(r => {
      const v = r.values, has = v.latitude_deg !== null && v.east_longitude_deg !== null;
      return `<li><button type="button" data-id="${r.id}" aria-current="${r.id === state.selectedId}">
        <span class="r-name">${esc(r.name)}</span>
        <span class="r-coord">${has ? fmtCoord(v.latitude_deg, v.east_longitude_deg) : NA}</span></button></li>`;
    }).join('') : '<li class="region-empty">No regions match.</li>';
    $('#regionCount').textContent = `${shown.length} of ${state.regions.length}`;
    $$('.marker').forEach(m => m.classList.toggle('is-dim', !matchesFilter(byId(m.dataset.id))));
  }

  function refreshMarkers() {
    $$('.marker').forEach(m => {
      m.classList.toggle('is-selected', m.dataset.id === state.selectedId);
      m.classList.toggle('in-compare', state.compare.includes(m.dataset.id));
      m.setAttribute('aria-pressed', String(m.dataset.id === state.selectedId));
    });
    $$('#regionList button').forEach(b => b.setAttribute('aria-current', String(b.dataset.id === state.selectedId)));
  }

  function selectRegion(id, { focus = true, scroll = false } = {}) {
    const reg = byId(id); if (!reg) return;
    state.selectedId = id;
    refreshMarkers(); renderRegionPanel();
    if (focus) focusRegion(reg);
    if (scroll) $('#regionPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    announce(`${reg.name} selected`);
  }

  function renderRegionPanel() {
    const reg = byId(state.selectedId), panel = $('#regionPanel');
    if (!reg) return;
    const inCompare = state.compare.includes(reg.id), v = reg.values;
    const cells = FIELDS.map(f => {
      const val = fmt(v[f.key], f);
      return `<div class="metric"><dt>${icon(f.icon)} ${f.label}<span class="use ${f.model ? 'in' : ''}">${f.model ? 'model input' : 'display only'}</span></dt>
        <dd class="${val === null ? 'na' : ''}">${val ?? NA}</dd></div>`;
    }).join('');
    const status = v[BOOL_COL] === null ? null : (v[BOOL_COL] ? 'Visible' : 'Not visible');
    panel.innerHTML = `
      <div class="panel-head">
        <span class="tag tag-data">Dataset value</span>
        <h3>${esc(reg.name)}</h3>
        <div class="panel-meta"><span>Region: ${esc(reg.name)}</span><span>CSV row ${reg.csvRow}</span>${isPolar(reg) ? '<span>South of −80°</span>' : ''}</div>
      </div>
      <dl class="metrics">${cells}
        <div class="metric metrics-wide"><dt>${icon('earth')} Earth Visibility Status</dt><dd class="${status === null ? 'na' : ''}">${status ?? NA}</dd></div>
      </dl>
      <div class="panel-actions">
        <button type="button" class="btn btn-primary" id="analyzeBtn">${icon('cpu')} Analyze Site</button>
        <button type="button" class="btn btn-secondary" id="addCompareBtn">${inCompare ? icon('check') + ' In comparison' : icon('compare') + ' Add to comparison'}</button>
      </div>
      <p class="panel-foot">“Model input” values are sent to the prediction API. “Display only” values are shown for context and are not used by the current model.</p>`;
    panel.classList.remove('flash'); void panel.offsetWidth; panel.classList.add('flash');
    $('#analyzeBtn').addEventListener('click', () => analyzeRegion(reg.id, { scroll: true }));
    $('#addCompareBtn').addEventListener('click', () => { inCompare ? removeCompare(reg.id) : addCompare(reg.id); renderRegionPanel(); });
  }

  /* ======================================================================
     9. API CLIENT
     ====================================================================== */
  class ApiError extends Error { constructor(kind, message, detail) { super(message); this.kind = kind; this.detail = detail; } }

  function buildPayload(reg) {
    const payload = {}, missing = [], invalid = [];
    MODEL_INPUTS.forEach(f => {
      const v = reg.values[f.key];
      if (v === null || v === undefined) missing.push(f.label);
      else if (v < f.min || v > f.max) invalid.push(`${f.label} (${v}${f.unit}) is outside ${f.min} to ${f.max}`);
      else payload[f.key] = v;
    });
    return { payload, missing, invalid };
  }

  async function fetchWithTimeout(url, opts = {}, ms = CONFIG.REQUEST_TIMEOUT_MS) {
    const ctrl = new AbortController(), t = setTimeout(() => ctrl.abort(), ms);
    try { return await fetch(url, { ...opts, signal: ctrl.signal }); }
    catch (e) {
      if (e.name === 'AbortError') throw new ApiError('timeout', `The API did not respond within ${ms / 1000} seconds.`);
      throw new ApiError('network', `Could not reach the API at ${CONFIG.API_BASE_URL}.`, 'Check that the FastAPI server is running and that the URL in script.js (CONFIG.API_BASE_URL) is correct.');
    } finally { clearTimeout(t); }
  }

  async function requestPrediction(payload) {
    const res = await fetchWithTimeout(CONFIG.API_BASE_URL + CONFIG.PREDICT_PATH, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(payload),
    });
    if (!res.ok) {
      let detail = '';
      try {
        const j = await res.json();
        detail = typeof j.detail === 'string' ? j.detail : Array.isArray(j.detail) ? j.detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ') : JSON.stringify(j);
      } catch (_) { /* non-JSON body */ }
      throw new ApiError('http', `The API returned an error (HTTP ${res.status}).`, detail);
    }
    let data;
    try { data = await res.json(); } catch (_) { throw new ApiError('shape', 'The API response was not valid JSON.'); }
    if (typeof data?.landing_score !== 'number' || typeof data?.landing_class !== 'string')
      throw new ApiError('shape', 'The API response did not have the expected fields (landing_score, landing_class).', JSON.stringify(data).slice(0, 300));
    return data;
  }

  async function checkApi() {
    state.api = 'checking'; renderApiStatus();
    try {
      const res = await fetchWithTimeout(CONFIG.API_BASE_URL + CONFIG.HEALTH_PATH, { headers: { Accept: 'application/json' } }, 4000);
      if (!res.ok) throw new Error('http');
      const spec = await res.json();
      state.api = spec.paths && spec.paths[CONFIG.PREDICT_PATH] ? 'online' : 'mismatch';
    } catch (_) { state.api = 'offline'; }
    renderApiStatus();
  }
  function renderApiStatus() {
    const pill = $('#apiStatus'), s = state.api;
    const text = { checking: 'API: checking…', online: 'API: online', offline: 'API: offline', mismatch: 'API: no /predict' }[s];
    pill.dataset.state = s === 'mismatch' ? 'offline' : s;
    $('.api-pill-text', pill).textContent = text;
    pill.title = `${CONFIG.API_BASE_URL}${CONFIG.PREDICT_PATH} — click to re-check`;
    $('#statApi').textContent = { checking: 'Checking…', online: 'Online', offline: 'Offline', mismatch: 'Endpoint missing' }[s];
  }

  /* ======================================================================
     10. ANALYSIS DASHBOARD
     ====================================================================== */
  const classTone = c => ({ excellent: 'excellent', good: 'good', moderate: 'moderate', poor: 'poor' }[String(c).toLowerCase()] || 'neutral');

  async function analyzeRegion(id, { scroll = false, setCurrent = true } = {}) {
    const reg = byId(id); if (!reg) return;
    if (setCurrent) state.analysisId = id;
    if (scroll) $('#analysis').scrollIntoView({ behavior: 'smooth', block: 'start' });

    const { payload, missing, invalid } = buildPayload(reg);
    if (missing.length || invalid.length) {
      state.predictions.set(id, { status: 'error', error: { kind: 'validation', message: 'This region cannot be analyzed with the available data.', detail: [missing.length ? `Missing values: ${missing.join(', ')}.` : '', ...invalid].filter(Boolean).join(' '), retry: false } });
      renderAnalysis(); renderCompare(); return;
    }
    state.predictions.set(id, { status: 'loading', payload });
    renderAnalysis(); renderCompare();
    announce(`Analyzing ${reg.name}`);
    try {
      const data = await requestPrediction(payload);
      state.predictions.set(id, { status: 'ok', payload, data, at: new Date() });
      announce(`${reg.name}: predicted class ${data.landing_class}`);
    } catch (e) {
      state.predictions.set(id, { status: 'error', payload, error: { kind: e.kind || 'unknown', message: e.message, detail: e.detail || '', retry: true } });
      if (e.kind === 'network' || e.kind === 'timeout') { state.api = 'offline'; renderApiStatus(); }
    }
    renderAnalysis(); renderCompare();
  }

  function renderAnalysis() {
    const root = $('#analysisRoot'), reg = byId(state.analysisId);
    if (!reg) {
      root.innerHTML = `<div class="state-box">${icon('chart').replace('class="icon"', 'class="icon icon-lg"')}<h3>No site analyzed yet</h3>
        <p>Select a region on the map and press <strong>Analyze Site</strong>. The model's score and class will appear here.</p>
        <div class="state-actions"><a class="btn btn-secondary" href="#explore">${icon('map')} Go to the map</a></div></div>`;
      return;
    }
    const p = state.predictions.get(reg.id);
    if (!p || p.status === 'loading') {
      root.innerHTML = `<div class="state-box" role="status"><div class="spinner" aria-hidden="true"></div><h3>Analyzing ${esc(reg.name)}…</h3><p>Sending the region's model inputs to the prediction API.</p></div>`; return;
    }
    if (p.status === 'error') {
      const e = p.error;
      root.innerHTML = `<div class="state-box error" role="alert">${icon('alert').replace('class="icon"', 'class="icon icon-lg"')}<h3>Analysis failed for ${esc(reg.name)}</h3><p>${esc(e.message)}</p>
        ${e.detail ? `<div class="err-detail">${esc(e.detail)}</div>` : ''}
        <div class="state-actions">${e.retry !== false ? `<button type="button" class="btn btn-primary" id="retryBtn">${icon('refresh')} Retry</button>` : ''}
        <a class="btn btn-secondary" href="#explore">Choose another region</a></div></div>`;
      $('#retryBtn')?.addEventListener('click', () => analyzeRegion(reg.id));
      return;
    }
    root.innerHTML = renderResults(reg, p);
    requestAnimationFrame(() => requestAnimationFrame(() => { const f = $('.bar-fill', root); if (f) f.style.width = clamp(p.data.landing_score, 0, 100) + '%'; }));
    $('#reanalyzeBtn')?.addEventListener('click', () => analyzeRegion(reg.id));
  }

  function renderResults(reg, p) {
    const d = p.data, tone = classTone(d.landing_class), score = d.landing_score, v = reg.values;
    const [b1, b2, b3] = CONFIG.TRAINING_BANDS;
    const inputsRows = MODEL_INPUTS.map(f => `<tr><th>${f.label}</th><td>${p.payload[f.key].toFixed(2).replace(/\.00$/, '').replace('-', '−')}${f.unit}</td></tr>`).join('');
    const earthFlag = p.payload.earth_visibility_percent > 50;
    const ctxRows = FIELDS.filter(f => !f.model).map(f => `<tr><th>${f.label}</th><td>${fmt(v[f.key], f) ?? NA}</td></tr>`).join('');
    const items = explainResult(reg, p);
    return `<div class="results">
      <section class="card-box" aria-label="Prediction">
        <header><h3>Model prediction</h3><span class="tag tag-model">Model output</span></header>
        <div class="card-body">
          <p class="result-site">${esc(reg.name)}</p>
          <div class="result-top"><span class="badge ${tone}">${icon(d.suitable ? 'check' : 'alert')} ${esc(d.landing_class)}</span>
            <span class="muted">${d.suitable ? 'Model flags this site as suitable (Good or Excellent)' : 'Model does not flag this site as suitable (below Good)'}</span></div>
          <div class="score-wrap">
            <div class="score-line"><span class="score-num">${score.toFixed(1)}</span><span class="score-of">/ 100 landing score</span></div>
            <div class="bar" role="img" aria-label="Landing score ${score.toFixed(1)} out of 100"><div class="bar-fill ${tone}"></div>
              ${[b1, b2, b3].map(b => `<span class="bar-tick" style="left:${b}%"></span>`).join('')}</div>
            <div class="bar-scale" aria-hidden="true"><span style="left:0">0</span><span style="left:${b1}%">${b1}</span><span style="left:${b2}%">${b2}</span><span style="left:${b3}%">${b3}</span><span style="left:100%">100</span></div>
          </div>
          <p class="fine">The score is a model output on a 0–100 scale, not a probability. Tick marks show the class bands used in the training notebook (Poor &lt; ${b1}, Moderate ${b1}–${b2}, Good ${b2}–${b3}, Excellent ≥ ${b3}); the API itself returns only the score, the class and the suitability flag. It does not provide a risk category.</p>
          <div class="state-actions" style="justify-content:flex-start"><button type="button" class="btn btn-ghost btn-sm" id="reanalyzeBtn">${icon('refresh')} Run again</button></div>
        </div>
      </section>
      <section class="card-box" aria-label="Inputs">
        <header><h3>Features sent to the model</h3><span class="tag tag-data">Dataset value</span></header>
        <div class="card-body">
          <table class="kv"><tbody>${inputsRows}
            <tr><th>Earth visible flag<span class="sub">Derived by the backend: Earth visibility &gt; 50%</span></th><td>${earthFlag ? 'Yes' : 'No'}</td></tr></tbody></table>
          <p class="kv-title">Shown for context, not used by the model</p>
          <table class="kv"><tbody>${ctxRows}</tbody></table>
        </div>
      </section>
      <section class="card-box explain" aria-label="Interpretation">
        <header><h3>Scientific interpretation</h3><span class="tag tag-derived">Reading aid</span></header>
        <div class="explain-grid">${items.map(i => `<div class="explain-item"><h4>${icon(i.icon)} ${i.title}</h4><p>${i.text}</p></div>`).join('')}</div>
      </section>
      <div class="notice">${icon('alert')}<p><strong>Not a guarantee of mission success.</strong> This is a screening result from a prototype model trained on three sites with rule-derived labels. Slope and elevation are not part of the current model, so terrain hazards are not reflected in the score. Analysis time: ${p.at.toLocaleTimeString()}.</p></div>
    </div>`;
  }

  // Interpretation text uses ONLY the API result and the region's own values; thresholds
  // quoted are the scoring rule documented in the project's training notebook.
  function explainResult(reg, p) {
    const d = p.data, x = p.payload, v = reg.values, out = [];
    const el = x.sun_elevation_deg;
    let sun;
    if (el <= 0) sun = `The Sun is at or below the horizon (${el.toFixed(1)}°). The training rule gives no credit for sun elevation in this case.`;
    else if (el < 5) sun = `The Sun is very low (${el.toFixed(1)}°). The training rule rates elevations from 0° to 5° on a rising scale, so credit is partial.`;
    else if (el <= 30) sun = `Sun elevation of ${el.toFixed(1)}° lies in the 5–30° band that the training rule rates highest: enough light and power with long shadows that reveal obstacles.`;
    else sun = `Sun elevation of ${el.toFixed(1)}° is above the 5–30° preferred band. The training rule reduces credit gradually for high Sun (more heating, flatter lighting).`;
    out.push({ icon: 'sun', title: 'Sun elevation', text: sun });
    out.push({ icon: 'sun', title: 'Illumination', text: `Illumination is ${x.illumination_percent.toFixed(1)}%. In the training rule, more illumination contributes more to the score (it stands for available solar energy and thermal conditions).` });
    out.push({ icon: 'earth', title: 'Earth visibility', text: x.earth_visibility_percent > 50
      ? `Earth visibility is ${x.earth_visibility_percent.toFixed(1)}%, so the backend sets the Earth-visible flag. This is favourable for direct communication.`
      : `Earth visibility is ${x.earth_visibility_percent.toFixed(1)}%, so the backend does not set the Earth-visible flag. Direct communication would be limited, and a relay may be needed.` });
    const sl = fmt(v.slope_deg, FIELDS[3]), ev = fmt(v.elevation_m, FIELDS[2]);
    out.push({ icon: 'slope', title: 'What the score does not cover', text: `Slope (${sl ?? NA}) and elevation (${ev ?? NA}) are not model inputs, so they do not affect this prediction. Terrain suitability needs to be assessed separately.` });
    out.push({ icon: 'cpu', title: 'Reading the result', text: `The model returned class <strong>${esc(d.landing_class)}</strong> with a score of <strong>${d.landing_score.toFixed(1)}</strong>. Because the training labels were produced by a rule on these same inputs, this result reflects that rule applied to ${esc(reg.name)}'s values rather than independent evidence of landing safety.` });
    return out;
  }

  /* ======================================================================
     11. COMPARISON
     ====================================================================== */
  function addCompare(id) {
    if (state.compare.includes(id)) return;
    if (state.compare.length >= CONFIG.COMPARE_MAX) { announce(`You can compare up to ${CONFIG.COMPARE_MAX} regions.`); flashCompareLimit(); return; }
    state.compare.push(id); afterCompareChange();
  }
  function removeCompare(id) { state.compare = state.compare.filter(x => x !== id); afterCompareChange(); }
  function flashCompareLimit() { renderChips(); }
  function afterCompareChange() { refreshMarkers(); renderChips(); renderCompare(); if (byId(state.selectedId)) renderRegionPanelButtonOnly(); }
  function renderRegionPanelButtonOnly() {
    const btn = $('#addCompareBtn'); if (!btn) return;
    const inC = state.compare.includes(state.selectedId);
    btn.innerHTML = inC ? icon('check') + ' In comparison' : icon('compare') + ' Add to comparison';
  }

  function renderChips() {
    const c = $('#compareChips');
    c.innerHTML = state.compare.map(id => `<li class="chip">${esc(byId(id).name)}<button type="button" data-remove="${id}" aria-label="Remove ${esc(byId(id).name)}">${icon('x')}</button></li>`).join('')
      + (state.compare.length >= CONFIG.COMPARE_MAX ? `<li class="muted" style="align-self:center;font-size:.85rem">Maximum of ${CONFIG.COMPARE_MAX} regions reached.</li>` : '');
    $('#compareAnalyze').disabled = state.compare.length < 1;
    $('#compareAdd').disabled = state.compare.length >= CONFIG.COMPARE_MAX;
  }

  function renderCompare() {
    const out = $('#compareOut'), regs = state.compare.map(byId);
    if (regs.length < 2) {
      out.innerHTML = `<div class="compare-empty">${regs.length === 0 ? 'No regions selected yet.' : 'Add at least one more region to compare.'} Pick regions above, or use “Add to comparison” in the region panel.</div>`; return;
    }
    const numFields = FIELDS;
    const head = `<tr><th scope="col">Metric</th>${regs.map(r => `<th scope="col">${esc(r.name)}<span class="sub">${r.values.latitude_deg !== null && r.values.east_longitude_deg !== null ? fmtCoord(r.values.latitude_deg, r.values.east_longitude_deg) : NA}</span></th>`).join('')}</tr>`;
    const groups = [...new Set(numFields.map(f => f.group))];
    let body = '';
    groups.forEach(g => {
      body += `<tr class="grp"><th colspan="${regs.length + 1}" scope="colgroup">${g}</th></tr>`;
      numFields.filter(f => f.group === g).forEach(f => {
        const base = regs[0].values[f.key];
        body += `<tr><th scope="row">${f.label}</th>${regs.map((r, i) => {
          const val = r.values[f.key], s = fmt(val, f);
          let delta = '';
          if (i > 0 && val !== null && base !== null) { const dv = val - base; delta = `<span class="delta">${dv >= 0 ? '+' : '−'}${Math.abs(dv).toFixed(f.digits)}${f.unit.trim() ? f.unit : ''} vs ${esc(regs[0].name)}</span>`; }
          return `<td class="${s === null ? 'na' : ''}">${s ?? NA}${delta}</td>`;
        }).join('')}</tr>`;
      });
      if (g === 'Illumination & Earth link') {
        body += `<tr><th scope="row">Earth Visibility Status</th>${regs.map(r => { const b = r.values[BOOL_COL]; return `<td class="${b === null ? 'na' : ''}">${b === null ? NA : b ? 'Visible' : 'Not visible'}</td>`; }).join('')}</tr>`;
      }
    });
    body += `<tr class="grp"><th colspan="${regs.length + 1}" scope="colgroup">Model results (from the API)</th></tr>`;
    const cell = (r, fn) => {
      const p = state.predictions.get(r.id);
      if (!p) return `<td class="na">Not analyzed<br><button type="button" class="btn btn-ghost btn-sm" data-analyze="${r.id}">Analyze</button></td>`;
      if (p.status === 'loading') return `<td class="na">Analyzing…</td>`;
      if (p.status === 'error') return `<td class="err-cell">${esc(p.error.message)}${p.error.retry !== false ? `<br><button type="button" class="btn btn-ghost btn-sm" data-analyze="${r.id}">Retry</button>` : ''}</td>`;
      return `<td>${fn(p.data)}</td>`;
    };
    body += `<tr><th scope="row">Landing class</th>${regs.map(r => cell(r, d => `<span class="badge sm ${classTone(d.landing_class)}">${esc(d.landing_class)}</span>`)).join('')}</tr>`;
    const scoreBase = state.predictions.get(regs[0].id);
    body += `<tr><th scope="row">Landing score (0–100)</th>${regs.map((r, i) => cell(r, d => {
      let delta = ''; if (i > 0 && scoreBase?.status === 'ok') { const dv = d.landing_score - scoreBase.data.landing_score; delta = `<span class="delta">${dv >= 0 ? '+' : '−'}${Math.abs(dv).toFixed(1)} vs ${esc(regs[0].name)}</span>`; }
      return `${d.landing_score.toFixed(1)}${delta}`; })).join('')}</tr>`;
    body += `<tr><th scope="row">Model suitability flag</th>${regs.map(r => cell(r, d => d.suitable ? 'Suitable' : 'Not flagged')).join('')}</tr>`;
    out.innerHTML = `<div class="table-wrap"><table class="cmp"><caption class="sr-only">Comparison of selected lunar regions</caption><thead>${head}</thead><tbody>${body}</tbody></table></div>
      <p class="fine">Differences are simple subtractions of the dataset or API values relative to the first region. No combined or invented score is computed. Slope and elevation are shown but are not model inputs.</p>`;
  }

  function initCompare() {
    const sel = $('#compareSelect');
    sel.innerHTML = '<option value="">Choose a region…</option>' + state.regions.map(r => `<option value="${r.id}">${esc(r.name)}</option>`).join('');
    $('#compareAdd').addEventListener('click', () => { if (sel.value) { addCompare(sel.value); sel.value = ''; } });
    $('#compareClear').addEventListener('click', () => { state.compare = []; afterCompareChange(); });
    $('#compareChips').addEventListener('click', e => { const b = e.target.closest('[data-remove]'); if (b) removeCompare(b.dataset.remove); });
    $('#compareOut').addEventListener('click', e => { const b = e.target.closest('[data-analyze]'); if (b) analyzeRegion(b.dataset.analyze, { setCurrent: false }); });
    $('#compareAnalyze').addEventListener('click', async () => {
      state.compare.forEach(id => state.predictions.set(id, { status: 'loading' })); renderCompare();
      await Promise.allSettled(state.compare.map(id => analyzeRegion(id, { setCurrent: false })));
    });
    renderChips(); renderCompare();
  }

  /* ======================================================================
     12. BOOTSTRAP
     ====================================================================== */
  function showMapMessage(html) { const m = $('#mapMsg'); m.innerHTML = html; m.hidden = false; }

  function initImages() {
    ['#heroMap', '#mapImage'].forEach(sel => { $(sel).src = CONFIG.MAP_URL; });
    $('#mapImage').addEventListener('error', () => showMapMessage(`<strong>Map image not found</strong><span>Expected <code>${esc(CONFIG.MAP_URL)}</code> next to index.html. Check <code>CONFIG.MAP_URL</code> in script.js.</span>`));
  }

  function loadErrorMessage(code) {
    const serve = 'Open a terminal in this folder, run <code>python -m http.server 5500</code>, then visit <code>http://localhost:5500</code>.';
    if (code === 'file-protocol') return `<strong>The browser blocked loading the CSV from a file:// address</strong><span>${serve}</span>`;
    if (code.startsWith('http-')) return `<strong>CSV not found (${esc(code.replace('http-', 'HTTP '))})</strong><span>Expected <code>${esc(CONFIG.CSV_URL)}</code> in the same folder as index.html.</span>`;
    if (code === 'no-name-column') return `<strong>The CSV is missing the <code>${NAME_COL}</code> column</strong><span>Regions cannot be identified without it.</span>`;
    return `<strong>The region data could not be loaded</strong><span>${serve}</span>`;
  }

  async function init() {
    runIntro(); initNav(); initImages(); renderApiStatus(); renderAnalysis();
    $('#apiStatus').addEventListener('click', checkApi);
    $('#mapNote').textContent = `Map labels without a dataset row (not selectable): ${CONFIG.IMAGE_LABELS_WITHOUT_DATA.join(', ')}. Marker positions come from the dataset's latitude and east longitude.`;
    initMapInteraction();
    requestAnimationFrame(() => fitView());
    checkApi();

    try {
      const { regions, missing } = await loadRegions();
      state.regions = regions;
      $('#statRegions').textContent = String(regions.length);
      const polar = regions.filter(isPolar).length;
      $('#statPolar').textContent = String(polar);
      $('#aboutCoverage').textContent = `Dataset coverage: ${regions.length} regions loaded from ${CONFIG.CSV_URL}; ${polar} lie south of −80° latitude. ${regions.filter(r => !r.pos).length} could not be placed on the map.`;
      if (missing.length) $('#mapNote').textContent += ` Warning: CSV columns not found: ${missing.join(', ')} (shown as “${NA}”).`;
      buildMarkers(); renderRegionList(); initCompare(); fitView();

      $('#regionList').addEventListener('click', e => { const b = e.target.closest('button[data-id]'); if (b) selectRegion(b.dataset.id, { focus: true }); });
      $('#regionSearch').addEventListener('input', e => { state.filter = e.target.value; renderRegionList(); });
      $('#polarOnly').addEventListener('change', e => { state.polarOnly = e.target.checked; renderRegionList(); });
    } catch (err) {
      $('#statRegions').textContent = '—'; $('#statPolar').textContent = '—';
      showMapMessage(loadErrorMessage(String(err.message)));
      $('#regionList').innerHTML = '<li class="region-empty">Region data unavailable.</li>';
      initCompare();
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
