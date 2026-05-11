/**
 * Polyscriptor PWA Demo — App Logic
 * Self-contained (no imports from main app.js).
 */

// ── LocalStorage keys ──────────────────────────────────────────────────
const LS_ENGINE     = 'pwa_last_engine';
const LS_SEG_METHOD = 'pwa_seg_method';
const LS_MODEL = name => `pwa_last_model_${name}`;

// ── State ──────────────────────────────────────────────────────────────
const state = {
  imageId:        null,
  imageInfo:      null,   // { width, height, filename }
  bboxes:         [],     // [[x1,y1,x2,y2], …]
  lines:          [],     // [{index, text, confidence, bbox}, …]
  engines:        [],     // from /api/engines
  loadedEngine:   null,   // currently active engine name in pool
  engineChangeSeq: 0,     // guards against stale async schema responses
  isSegmenting:   false,
  isTranscribing: false,
  sseAbort:       null,   // AbortController for active SSE
};

// ── DOM refs ───────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const el = {
  btnCamera:        $('btn-camera'),
  btnFile:          $('btn-file'),
  fileCamera:       $('file-camera'),
  filePicker:       $('file-picker'),
  previewWrap:      $('image-preview-wrap'),
  previewImg:       $('preview-img'),
  bboxCanvas:       $('bbox-canvas'),
  previewFilename:  $('preview-filename'),
  btnClearImage:    $('btn-clear-image'),

  engineSelect:     $('engine-select'),
  modelRow:         $('model-row'),
  modelSelect:      $('model-select'),
  btnLoadModel:     $('btn-load-model'),
  modelStatusBadge: $('model-status-badge'),
  enginePill:       $('engine-pill'),
  enginePillText:   $('engine-pill-text'),

  segMethodSelect:  $('seg-method-select'),

  btnSegment:       $('btn-segment'),
  btnTranscribe:    $('btn-transcribe'),
  btnCancel:        $('btn-cancel'),

  progressCard:     $('progress-card'),
  progressBar:      $('progress-bar'),
  statusText:       $('status-text'),

  resultsCard:      $('results-card'),
  resultsList:      $('results-list'),
  lineCount:        $('line-count'),
  btnCopy:          $('btn-copy'),
  btnExportTxt:     $('btn-export-txt'),

  // Photo review overlay
  photoReview:      $('photo-review'),
  reviewImg:        $('review-img'),
  reviewCropCanvas: $('review-crop-canvas'),
  reviewWarn:       $('review-warn'),
  btnRotateCCW:     $('btn-rotate-ccw'),
  btnRotateCW:      $('btn-rotate-cw'),
  btnAutoCrop:      $('btn-auto-crop'),
  btnCropStart:     $('btn-crop-start'),
  btnCropApply:     $('btn-crop-apply'),
  btnCropCancel:    $('btn-crop-cancel'),
  btnRetake:        $('btn-retake'),
  btnUsePhoto:      $('btn-use-photo'),
};

// ── Photo Review State ─────────────────────────────────────────────────
const reviewState = {
  canvas:      null,   // off-screen working canvas (rotated / cropped)
  cropMode:    false,
  cropStart:   null,   // image-coord pointer-down position
  cropRect:    null,   // {x, y, w, h} in image coords
  srcFilename: '',
};

// ── Toast ──────────────────────────────────────────────────────────────
function toast(msg, type = 'info', ms = 4000) {
  const container = $('toast-container');
  const div = document.createElement('div');
  div.className = `toast toast--${type}`;
  div.textContent = msg;
  container.appendChild(div);
  setTimeout(() => div.remove(), ms);
}

// ── API helper ─────────────────────────────────────────────────────────
async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || err.message || `HTTP ${resp.status}`);
  }
  return resp;
}

// ── Engine pill ────────────────────────────────────────────────────────
function setPill(state, text) {
  el.enginePill.className = `engine-pill engine-pill--${state}`;
  el.enginePillText.textContent = text;
}

// ── Engine status (check pool) ─────────────────────────────────────────
async function checkEngineStatus() {
  try {
    const resp = await api('/api/engine/status');
    const data = await resp.json();

    // Response: { loaded: bool, engine_name: str, config: {...} }
    if (data.loaded && data.engine_name) {
      state.loadedEngine = data.engine_name;
      setPill('loaded', data.engine_name);
      setBadge('loaded', 'Model loaded');
      // Pre-select the matching engine in the dropdown
      if (el.engineSelect.querySelector(`option[value="${data.engine_name}"]`)) {
        el.engineSelect.value = data.engine_name;
      }
      // Hide load controls — engine already active
      el.btnLoadModel.hidden = true;
      el.modelRow.hidden = true;
    } else {
      state.loadedEngine = null;
      setPill('unloaded', 'No model');
      setBadge('unloaded', 'No model loaded');
      el.btnLoadModel.hidden = false;
    }
    updateActionButtons();
  } catch {
    setPill('unknown', 'Offline');
    setBadge('loading', 'Checking…');
  }
}

function setBadge(type, text) {
  el.modelStatusBadge.className = `badge badge--${type}`;
  el.modelStatusBadge.textContent = text;
}

// ── Load engines list ──────────────────────────────────────────────────
async function loadEngines() {
  try {
    const resp  = await api('/api/engines');
    const data  = await resp.json();
    // /api/engines returns a plain array
    state.engines = Array.isArray(data) ? data : (data.engines || []);

    el.engineSelect.innerHTML = '';
    const avail = state.engines.filter(e => e.available);

    if (avail.length === 0) {
      el.engineSelect.innerHTML = '<option value="">No engines available</option>';
      return;
    }

    for (const eng of avail) {
      const opt = document.createElement('option');
      opt.value = eng.name;
      opt.textContent = eng.display_name || eng.name;
      el.engineSelect.appendChild(opt);
    }

    // Restore last selection
    const last = localStorage.getItem(LS_ENGINE);
    if (last && el.engineSelect.querySelector(`option[value="${last}"]`)) {
      el.engineSelect.value = last;
    }

    await onEngineChange();
  } catch (e) {
    el.engineSelect.innerHTML = '<option value="">Failed to load engines</option>';
    toast('Could not reach server', 'error');
  }
}

// ── Engine selection changed ───────────────────────────────────────────
async function onEngineChange() {
  const name = el.engineSelect.value;
  if (!name) return;
  const requestSeq = ++state.engineChangeSeq;
  localStorage.setItem(LS_ENGINE, name);

  // If this engine is already the loaded one, hide load controls
  if (name === state.loadedEngine) {
    el.modelRow.hidden = true;
    el.btnLoadModel.hidden = true;
    return;
  }

  el.modelRow.hidden = false;
  el.modelSelect.innerHTML = '<option>Loading…</option>';
  el.btnLoadModel.hidden = false;
  el.btnLoadModel.disabled = true;
  state.modelFieldKey = null;

  try {
    // Use config-schema (same as main app) — it has the full model option list
    const resp = await api(`/api/engine/${encodeURIComponent(name)}/config-schema`);
    const schema = await resp.json();

    if (requestSeq !== state.engineChangeSeq || el.engineSelect.value !== name) {
      return;
    }

    // Find first non-dynamic select field → that's the model selector
    const selectField = (schema.fields || []).find(
      f => f.type === 'select' && !f.dynamic
    );

    el.modelSelect.innerHTML = '';

    if (selectField && (selectField.options || []).length > 0) {
      state.modelFieldKey = selectField.key;
      for (const opt of selectField.options) {
        const o = document.createElement('option');
        o.value = typeof opt === 'object' ? opt.value : opt;
        o.textContent = typeof opt === 'object' ? opt.label : opt;
        el.modelSelect.appendChild(o);
      }
      // Restore last selection or apply schema default
      const lastModel = localStorage.getItem(LS_MODEL(name));
      if (lastModel && el.modelSelect.querySelector(`option[value="${lastModel}"]`)) {
        el.modelSelect.value = lastModel;
      } else if (selectField.default != null) {
        el.modelSelect.value = selectField.default;
      }
    } else {
      // No static options (e.g. API-based engines) — show Default
      state.modelFieldKey = selectField?.key || 'model_path';
      const o = document.createElement('option');
      o.value = '';
      o.textContent = 'Default';
      el.modelSelect.appendChild(o);
    }

    el.btnLoadModel.disabled = false;
  } catch {
    if (requestSeq !== state.engineChangeSeq || el.engineSelect.value !== name) {
      return;
    }
    el.modelSelect.innerHTML = '<option value="">Default</option>';
    state.modelFieldKey = 'model_path';
    el.btnLoadModel.disabled = false;
  }
}

// ── Load model ─────────────────────────────────────────────────────────
async function loadModel() {
  const engineName = el.engineSelect.value;
  if (!engineName) return;

  const modelVal = el.modelSelect.value || '';
  localStorage.setItem(LS_MODEL(engineName), modelVal);

  el.btnLoadModel.disabled = true;
  el.btnLoadModel.textContent = 'Loading…';
  setPill('loading', 'Loading…');
  setBadge('loading', 'Loading…');

  try {
    // Use the field key from the config schema (e.g. 'model_path' for CRNN-CTC/TrOCR/Kraken)
    const fieldKey = state.modelFieldKey || 'model_path';
    const config = modelVal ? { [fieldKey]: modelVal } : {};
    await api('/api/engine/load', {
      method: 'POST',
      body: JSON.stringify({ engine_name: engineName, config }),
    });

    state.loadedEngine = engineName;
    setPill('loaded', engineName);
    setBadge('loaded', 'Model loaded');
    el.btnLoadModel.hidden = true;
    el.modelRow.hidden = true;
    toast(`${engineName} loaded`, 'success');
  } catch (e) {
    setPill('unloaded', 'Load failed');
    setBadge('unloaded', 'Load failed');
    toast(`Load failed: ${e.message}`, 'error');
  } finally {
    el.btnLoadModel.disabled = false;
    el.btnLoadModel.textContent = 'Load Model';
    updateActionButtons();
  }
}

// ── Update action button states ────────────────────────────────────────
function updateActionButtons() {
  const hasImage  = !!state.imageId;
  const hasEngine = !!state.loadedEngine;
  const busy      = state.isSegmenting || state.isTranscribing;

  el.btnSegment.disabled   = !hasImage || !hasEngine || busy;
  el.btnTranscribe.disabled = !hasImage || !hasEngine || busy;
  el.btnCancel.hidden       = !busy;
}

// ── File upload ────────────────────────────────────────────────────────
async function uploadFile(file) {
  if (!file) return;

  const fd = new FormData();
  fd.append('file', file);

  setStatus('Uploading…');
  el.progressCard.hidden = false;
  setProgress(0);

  try {
    const resp = await fetch('/api/image/upload?max_dim=2400', { method: 'POST', body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await resp.json();

    if (data.is_pdf) {
      // PDF: use first page
      const first = data.pages[0];
      state.imageId   = first.image_id;
      state.imageInfo = { width: first.width, height: first.height, filename: first.filename };
      toast(`PDF uploaded — using page 1 of ${data.pages.length}`, 'info');
    } else {
      state.imageId   = data.image_id;
      state.imageInfo = { width: data.width, height: data.height, filename: data.filename };
    }

    // Show preview
    el.previewImg.src              = `/api/image/${state.imageId}`;
    el.previewFilename.textContent = state.imageInfo.filename || file.name;
    el.previewWrap.hidden          = false;
    clearBboxes();

    // Clear old results
    hideResults();
    setStatus('Image ready');
    setProgress(100);
    setTimeout(() => { el.progressCard.hidden = true; }, 800);
    updateActionButtons();
  } catch (e) {
    toast(`Upload failed: ${e.message}`, 'error');
    setStatus('');
    el.progressCard.hidden = true;
  }
}

// ── Clear image ────────────────────────────────────────────────────────
function clearImage() {
  state.imageId   = null;
  state.imageInfo = null;
  state.bboxes    = [];
  state.lines     = [];
  el.previewWrap.hidden = true;
  el.previewImg.src     = '';
  clearBboxes();
  hideResults();
  updateActionButtons();
}

// ── BBox canvas ────────────────────────────────────────────────────────
function clearBboxes() {
  const canvas = el.bboxCanvas;
  const ctx    = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  state.bboxes = [];
}

// Draw bounding boxes scaled to displayed image size
function drawBboxes(bboxes, highlightIdx = -1) {
  const img    = el.previewImg;
  const canvas = el.bboxCanvas;
  const ctx    = canvas.getContext('2d');

  // Match canvas to displayed size
  canvas.width  = img.offsetWidth;
  canvas.height = img.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!bboxes || bboxes.length === 0 || !state.imageInfo) return;

  const scaleX = img.offsetWidth  / state.imageInfo.width;
  const scaleY = img.offsetHeight / state.imageInfo.height;

  // Color palette for lines — use distinct hues
  const COLORS = [
    'rgba(59,130,246,', // blue
    'rgba(99,102,241,', // indigo
    'rgba(34,197,94,',  // green
    'rgba(245,158,11,', // amber
    'rgba(239,68,68,',  // red
    'rgba(168,85,247,', // purple
    'rgba(20,184,166,', // teal
    'rgba(249,115,22,', // orange
  ];

  bboxes.forEach((bbox, i) => {
    const [x1, y1, x2, y2] = bbox;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;

    const colorBase = COLORS[i % COLORS.length];
    const isHighlighted = i === highlightIdx;
    const fillAlpha    = isHighlighted ? 0.25 : 0.10;
    const strokeAlpha  = isHighlighted ? 1.0  : 0.7;

    ctx.fillStyle   = `${colorBase}${fillAlpha})`;
    ctx.strokeStyle = `${colorBase}${strokeAlpha})`;
    ctx.lineWidth   = isHighlighted ? 2 : 1.5;

    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    // Line number label
    ctx.font      = 'bold 10px monospace';
    ctx.fillStyle = `${colorBase}0.9)`;
    const label   = String(i + 1);
    const pad     = 3;
    const tw      = ctx.measureText(label).width + pad * 2;
    ctx.fillStyle = `${colorBase}0.85)`;
    ctx.fillRect(x, y - 14, tw, 14);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + pad, y - 3);
  });
}

// ── Segment ────────────────────────────────────────────────────────────
async function segmentImage() {
  if (!state.imageId) return;

  state.isSegmenting = true;
  updateActionButtons();
  el.progressCard.hidden = false;
  setProgress(0);
  setStatus('Detecting lines…');
  clearBboxes();

  const method = el.segMethodSelect.value || 'kraken';
  localStorage.setItem(LS_SEG_METHOD, method);

  try {
    const url  = `/api/image/${state.imageId}/segment?method=${encodeURIComponent(method)}&device=cuda%3A0`;
    const resp = await api(url);
    const data = await resp.json();

    state.bboxes = data.bboxes || [];
    drawBboxes(state.bboxes);

    setStatus(`${state.bboxes.length} line${state.bboxes.length !== 1 ? 's' : ''} detected`);
    setProgress(100);
    toast(`${state.bboxes.length} lines detected`, 'success', 2500);
  } catch (e) {
    toast(`Segmentation failed: ${e.message}`, 'error');
    setStatus('Segmentation failed');
  } finally {
    state.isSegmenting = false;
    updateActionButtons();
    setTimeout(() => { if (!state.isTranscribing) el.progressCard.hidden = true; }, 1500);
  }
}

// ── Transcribe (SSE) ───────────────────────────────────────────────────
async function startTranscription() {
  if (!state.imageId || !state.loadedEngine) return;

  state.isTranscribing = true;
  state.lines          = [];
  updateActionButtons();

  el.progressCard.hidden = false;
  setProgress(0);
  setStatus('Starting transcription…');
  el.resultsCard.hidden  = true;
  el.resultsList.innerHTML = '';

  const method = el.segMethodSelect.value || 'kraken';

  const body = JSON.stringify({
    image_id:   state.imageId,
    seg_method: method,
    seg_device: 'cuda:0',
  });

  const abort = new AbortController();
  state.sseAbort = abort;

  try {
    const resp = await fetch('/api/transcribe', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: abort.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Transcription failed');
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop(); // last part may be incomplete

      for (const part of parts) {
        const eventLine = part.split('\n').find(l => l.startsWith('event:'));
        const dataLine  = part.split('\n').find(l => l.startsWith('data:'));
        if (!dataLine) continue;

        const event   = eventLine ? eventLine.slice(7).trim() : 'message';
        const payload = JSON.parse(dataLine.slice(5).trim());

        handleSSEEvent(event, payload);
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      toast(`Transcription error: ${e.message}`, 'error');
      setStatus('Error');
    }
  } finally {
    state.isTranscribing = false;
    state.sseAbort       = null;
    updateActionButtons();
  }
}

function handleSSEEvent(event, payload) {
  switch (event) {
    case 'status':
      setStatus(payload.message || '');
      break;

    case 'segmentation': {
      state.bboxes = payload.bboxes || [];
      drawBboxes(state.bboxes);
      setStatus(`${state.bboxes.length} lines detected — transcribing…`);
      break;
    }

    case 'progress': {
      const { current, total, line } = payload;
      setProgress(total > 0 ? (current / total) * 100 : 0);
      setStatus(`Transcribing line ${current} / ${total}…`);

      if (line) {
        state.lines.push(line);
        appendResultLine(line);
        // Highlight corresponding bbox
        drawBboxes(state.bboxes, line.index);
      }

      // Show results card on first result
      if (el.resultsCard.hidden && state.lines.length === 1) {
        el.resultsCard.hidden = false;
        el.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      break;
    }

    case 'complete': {
      setProgress(100);
      const count = (payload.lines || []).length;
      const secs  = payload.total_time_s ? ` in ${payload.total_time_s}s` : '';
      setStatus(`Done — ${count} lines${secs}`);
      el.lineCount.textContent = `${count} lines`;
      el.lineCount.className   = 'badge badge--info';

      // Redraw all bboxes without highlight
      drawBboxes(state.bboxes);
      toast(`Transcription complete (${count} lines)`, 'success');
      setTimeout(() => { el.progressCard.hidden = true; }, 1200);
      break;
    }

    case 'cancelled':
      setStatus('Cancelled');
      toast('Transcription cancelled', 'warn', 2500);
      setTimeout(() => { el.progressCard.hidden = true; }, 1000);
      break;

    case 'error':
      toast(`Error: ${payload.message}`, 'error');
      setStatus('Error');
      break;
  }
}

// ── Result line DOM ────────────────────────────────────────────────────
function appendResultLine(line) {
  const div = document.createElement('div');
  div.className = 'result-line';

  const numSpan = document.createElement('span');
  numSpan.className = 'line-num';
  numSpan.textContent = String(line.index + 1);

  const textSpan = document.createElement('span');
  textSpan.className = 'line-text';
  textSpan.textContent = line.text || '';

  div.appendChild(numSpan);
  div.appendChild(textSpan);

  if (line.confidence !== null && line.confidence !== undefined) {
    const pct = Math.round(line.confidence * 100);
    const confSpan = document.createElement('span');
    confSpan.className = `line-conf ${pct >= 90 ? 'conf-high' : pct >= 75 ? 'conf-mid' : 'conf-low'}`;
    confSpan.textContent = `${pct}%`;
    div.appendChild(confSpan);
  }

  el.resultsList.appendChild(div);
  // Auto-scroll to latest
  el.resultsList.scrollTop = el.resultsList.scrollHeight;
}

// ── Cancel ─────────────────────────────────────────────────────────────
async function cancelTranscription() {
  if (state.sseAbort) state.sseAbort.abort();
  try {
    await api('/api/transcribe/cancel', { method: 'POST', body: '{}' });
  } catch { /* ignore */ }
}

// ── Progress helpers ───────────────────────────────────────────────────
function setProgress(pct) {
  el.progressBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

function setStatus(msg) {
  el.statusText.textContent = msg;
}

// ── Hide results ───────────────────────────────────────────────────────
function hideResults() {
  el.resultsCard.hidden    = true;
  el.resultsList.innerHTML = '';
  state.lines              = [];
  el.lineCount.textContent = '';
}

// ── Copy all ───────────────────────────────────────────────────────────
function copyAll() {
  const text = state.lines.map(l => l.text || '').join('\n');
  if (!text) { toast('Nothing to copy', 'warn', 2000); return; }
  navigator.clipboard.writeText(text)
    .then(() => toast('Copied to clipboard', 'success', 2000))
    .catch(() => toast('Copy failed', 'error'));
}

// ── Export TXT ─────────────────────────────────────────────────────────
function exportTxt() {
  const text = state.lines.map(l => l.text || '').join('\n');
  if (!text) { toast('Nothing to export', 'warn', 2000); return; }
  const blob  = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url   = URL.createObjectURL(blob);
  const a     = document.createElement('a');
  a.href      = url;
  a.download  = (state.imageInfo?.filename?.replace(/\.[^.]+$/, '') || 'transcription') + '.txt';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Redraw bboxes on image resize ──────────────────────────────────────
function onImageResize() {
  if (state.bboxes.length > 0) drawBboxes(state.bboxes);
}

// ── Photo Review ────────────────────────────────────────────────────────

function openPhotoReview(file) {
  reviewState.srcFilename = file.name || 'photo.jpg';
  reviewState.cropMode    = false;
  reviewState.cropStart   = null;
  reviewState.cropRect    = null;

  const img = new Image();
  const url = URL.createObjectURL(file);
  img.onload = () => {
    URL.revokeObjectURL(url);
    const canvas = document.createElement('canvas');
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    reviewState.canvas = canvas;
    updateReviewDisplay();
    el.photoReview.hidden = false;
    document.body.style.overflow = 'hidden';
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    toast('Could not load photo', 'error');
  };
  img.src = url;
}

function closePhotoReview() {
  el.photoReview.hidden = true;
  document.body.style.overflow = '';
  reviewState.canvas   = null;
  reviewState.cropMode = false;
  reviewState.cropRect = null;
  resetCropUI();
}

function updateReviewDisplay() {
  if (!reviewState.canvas) return;
  el.reviewImg.onload = () => {
    syncCropCanvas();
    checkReviewOrientation();
  };
  el.reviewImg.src = reviewState.canvas.toDataURL('image/jpeg', 0.9);
}

function checkReviewOrientation() {
  const landscape = reviewState.canvas.width > reviewState.canvas.height;
  el.reviewWarn.hidden = !landscape;
}

function syncCropCanvas() {
  const c    = el.reviewCropCanvas;
  const rect = el.reviewImg.getBoundingClientRect();
  if (!rect.width) return;
  c.width  = Math.round(rect.width);
  c.height = Math.round(rect.height);
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

// ── Auto-Crop (adaptive page detection) ────────────────────────────────

function autoDetectAndCrop() {
  if (!reviewState.canvas) return;
  exitCropMode();

  const canvas = reviewState.canvas;
  const { width, height } = canvas;
  const data = canvas.getContext('2d').getImageData(0, 0, width, height).data;

  // Single pass: accumulate page-likelihood per row and per column.
  // Heuristic: white paper is typically bright with low saturation.
  const rowSum = new Float32Array(height);
  const colSum = new Float32Array(width);
  let borderSum = 0;
  let borderCount = 0;

  const borderBandY = Math.max(1, Math.floor(height * 0.08));
  const borderBandX = Math.max(1, Math.floor(width * 0.08));

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      const v = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const s = v === 0 ? 0 : (v - min) / v;

      const pageScore = v - (s * 90);
      rowSum[y] += pageScore;
      colSum[x] += pageScore;

      const isBorderPixel = y < borderBandY || y >= (height - borderBandY) || x < borderBandX || x >= (width - borderBandX);
      if (isBorderPixel) {
        borderSum += pageScore;
        borderCount += 1;
      }
    }
  }

  const borderMean = borderCount > 0 ? (borderSum / borderCount) : 40;
  const THRESHOLD = Math.min(230, borderMean + 14);
  const PAD       = 12;

  let top = 0, bottom = height - 1, left = 0, right = width - 1;
  for (let y = 0;          y < height; y++) { if (rowSum[y] / width  > THRESHOLD) { top    = y; break; } }
  for (let y = height - 1; y >= 0;    y--) { if (rowSum[y] / width  > THRESHOLD) { bottom = y; break; } }
  for (let x = 0;          x < width; x++) { if (colSum[x] / height > THRESHOLD) { left   = x; break; } }
  for (let x = width - 1;  x >= 0;    x--) { if (colSum[x] / height > THRESHOLD) { right  = x; break; } }

  // Apply padding and clamp
  top    = Math.max(0,         top    - PAD);
  bottom = Math.min(height - 1, bottom + PAD);
  left   = Math.max(0,         left   - PAD);
  right  = Math.min(width - 1, right  + PAD);

  const w = right - left;
  const h = bottom - top;

  // Sanity check: don't crop to less than 20% of original
  if (w < width * 0.2 || h < height * 0.2) {
    toast('Page not detected clearly - please crop manually', 'warn');
    return;
  }

  const dst = document.createElement('canvas');
  dst.width  = w;
  dst.height = h;
  dst.getContext('2d').drawImage(canvas, left, top, w, h, 0, 0, w, h);
  reviewState.canvas = dst;
  updateReviewDisplay();
}

// ── Rotate ─────────────────────────────────────────────────────────────

function rotateReview(angle) {
  if (!reviewState.canvas) return;
  exitCropMode();
  const src = reviewState.canvas;
  const dst = document.createElement('canvas');
  dst.width  = src.height;
  dst.height = src.width;
  const ctx = dst.getContext('2d');
  ctx.translate(dst.width / 2, dst.height / 2);
  ctx.rotate(angle * Math.PI / 180);
  ctx.drawImage(src, -src.width / 2, -src.height / 2);
  reviewState.canvas = dst;
  updateReviewDisplay();
}

// ── Crop ───────────────────────────────────────────────────────────────

function enterCropMode() {
  reviewState.cropMode  = true;
  reviewState.cropRect  = null;
  reviewState.cropStart = null;
  el.btnCropStart.hidden  = true;
  el.btnCropApply.hidden  = true;
  el.btnCropCancel.hidden = false;
  el.reviewCropCanvas.style.pointerEvents = 'auto';
  syncCropCanvas();
}

function exitCropMode() {
  reviewState.cropMode  = false;
  reviewState.cropStart = null;
  reviewState.cropRect  = null;
  el.reviewCropCanvas.style.pointerEvents = 'none';
  resetCropUI();
  syncCropCanvas();
}

function resetCropUI() {
  el.btnCropStart.hidden  = false;
  el.btnCropApply.hidden  = true;
  el.btnCropCancel.hidden = true;
}

function pointerToImageCoords(e) {
  const c    = el.reviewCropCanvas;
  const rect = c.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(reviewState.canvas.width,  (e.clientX - rect.left) * (reviewState.canvas.width  / rect.width))),
    y: Math.max(0, Math.min(reviewState.canvas.height, (e.clientY - rect.top)  * (reviewState.canvas.height / rect.height))),
  };
}

function onCropPointerDown(e) {
  if (!reviewState.cropMode) return;
  e.preventDefault();
  el.reviewCropCanvas.setPointerCapture(e.pointerId);
  reviewState.cropStart = pointerToImageCoords(e);
  reviewState.cropRect  = null;
  el.btnCropApply.hidden = true;
}

function onCropPointerMove(e) {
  if (!reviewState.cropMode || !reviewState.cropStart) return;
  e.preventDefault();
  const cur = pointerToImageCoords(e);
  reviewState.cropRect = {
    x: Math.min(reviewState.cropStart.x, cur.x),
    y: Math.min(reviewState.cropStart.y, cur.y),
    w: Math.abs(cur.x - reviewState.cropStart.x),
    h: Math.abs(cur.y - reviewState.cropStart.y),
  };
  drawCropOverlay();
}

function onCropPointerUp(e) {
  if (!reviewState.cropMode) return;
  e.preventDefault();
  reviewState.cropStart = null;
  const r = reviewState.cropRect;
  if (r && r.w > 20 && r.h > 20) {
    el.btnCropApply.hidden = false;
  }
}

function drawCropOverlay() {
  const c    = el.reviewCropCanvas;
  const ctx  = c.getContext('2d');
  const r    = reviewState.cropRect;
  if (!r) return;

  const scaleX = c.width  / reviewState.canvas.width;
  const scaleY = c.height / reviewState.canvas.height;
  const rx = r.x * scaleX, ry = r.y * scaleY;
  const rw = r.w * scaleX, rh = r.h * scaleY;

  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.clearRect(rx, ry, rw, rh);
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.lineWidth   = 2;
  ctx.strokeRect(rx, ry, rw, rh);
}

function applyReviewCrop() {
  const r = reviewState.cropRect;
  if (!r || r.w < 20 || r.h < 20) return;
  const dst = document.createElement('canvas');
  dst.width  = Math.round(r.w);
  dst.height = Math.round(r.h);
  dst.getContext('2d').drawImage(
    reviewState.canvas,
    Math.round(r.x), Math.round(r.y), Math.round(r.w), Math.round(r.h),
    0, 0, Math.round(r.w), Math.round(r.h)
  );
  reviewState.canvas = dst;
  exitCropMode();
  updateReviewDisplay();
}

// ── Confirm / Retake ────────────────────────────────────────────────────

function retakePhoto() {
  closePhotoReview();
  el.fileCamera.value = '';
  el.fileCamera.click();
}

function confirmPhoto() {
  if (!reviewState.canvas) return;
  el.btnUsePhoto.disabled = true;
  reviewState.canvas.toBlob(blob => {
    if (!blob) {
      toast('Error while processing photo', 'error');
      el.btnUsePhoto.disabled = false;
      return;
    }
    const baseName = reviewState.srcFilename.replace(/\.[^.]+$/, '');
    const file = new File([blob], baseName + '.jpg', { type: 'image/jpeg' });
    closePhotoReview();
    el.btnUsePhoto.disabled = false;
    uploadFile(file);
  }, 'image/jpeg', 0.92);
}

// ── Register service worker ─────────────────────────────────────────────
async function detectPwaVersion() {
  try {
    const resp = await fetch('/static/pwa/demo.js', {
      method: 'HEAD',
      cache: 'no-store',
    });
    const lastModified = resp.headers.get('last-modified');
    if (lastModified) {
      const ts = Date.parse(lastModified);
      if (Number.isFinite(ts) && ts > 0) return String(ts);
    }
  } catch {
    // Fallback below
  }
  return 'dev';
}

if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const version = await detectPwaVersion();
      const reg = await navigator.serviceWorker.register(`/sw.js?v=${encodeURIComponent(version)}`, { scope: '/' });
      reg.update().catch(() => {});
    } catch (e) {
      console.warn('SW registration failed:', e);
    }
  });
}

// ── Init ───────────────────────────────────────────────────────────────
function init() {
  // Camera button — open review overlay instead of uploading directly
  el.btnCamera.addEventListener('click', () => el.fileCamera.click());
  el.fileCamera.addEventListener('change', () => {
    if (el.fileCamera.files[0]) openPhotoReview(el.fileCamera.files[0]);
    el.fileCamera.value = '';
  });

  // Photo review
  el.btnRotateCCW.addEventListener('click',  () => rotateReview(-90));
  el.btnRotateCW.addEventListener('click',   () => rotateReview(90));
  el.btnAutoCrop.addEventListener('click',   autoDetectAndCrop);
  el.btnCropStart.addEventListener('click',  enterCropMode);
  el.btnCropApply.addEventListener('click',  applyReviewCrop);
  el.btnCropCancel.addEventListener('click', exitCropMode);
  el.btnRetake.addEventListener('click',     retakePhoto);
  el.btnUsePhoto.addEventListener('click',   confirmPhoto);
  el.reviewCropCanvas.addEventListener('pointerdown', onCropPointerDown);
  el.reviewCropCanvas.addEventListener('pointermove', onCropPointerMove);
  el.reviewCropCanvas.addEventListener('pointerup',   onCropPointerUp);

  // File picker button
  el.btnFile.addEventListener('click', () => el.filePicker.click());
  el.filePicker.addEventListener('change', () => {
    if (el.filePicker.files[0]) uploadFile(el.filePicker.files[0]);
    el.filePicker.value = '';
  });

  // Clear image
  el.btnClearImage.addEventListener('click', clearImage);

  // Engine select
  el.engineSelect.addEventListener('change', onEngineChange);

  // Load model
  el.btnLoadModel.addEventListener('click', loadModel);

  // Segment
  el.btnSegment.addEventListener('click', segmentImage);

  // Transcribe
  el.btnTranscribe.addEventListener('click', startTranscription);

  // Cancel
  el.btnCancel.addEventListener('click', cancelTranscription);

  // Export
  el.btnCopy.addEventListener('click', copyAll);
  el.btnExportTxt.addEventListener('click', exportTxt);

  // Seg method persistence
  const savedSeg = localStorage.getItem(LS_SEG_METHOD);
  if (savedSeg && el.segMethodSelect.querySelector(`option[value="${savedSeg}"]`)) {
    el.segMethodSelect.value = savedSeg;
  }
  el.segMethodSelect.addEventListener('change', () => {
    localStorage.setItem(LS_SEG_METHOD, el.segMethodSelect.value);
  });

  // Redraw bboxes on layout changes (image resize)
  const ro = new ResizeObserver(onImageResize);
  ro.observe(el.previewImg);

  // Initial data load
  loadEngines().then(checkEngineStatus);
}

document.addEventListener('DOMContentLoaded', init);
