/**
 * Polyscriptor Web UI — Main application entry point
 *
 * Central state + event bus, wires up components.
 * No framework, no build step — native ES modules.
 */

import { initEnginePanel } from './components/engine-panel.js';
import { initImageViewer } from './components/image-viewer.js';
import { initTranscriptionPanel } from './components/transcription-panel.js';
import { initBatchPanel } from './components/batch-panel.js';

// ── Global state ───────────────────────────────────────────────────────
export const state = {
    engines: [],
    currentEngine: null,
    engineLoaded: false,
    imageId: null,
    imageInfo: null,
    lines: [],           // [{index, text, confidence, bbox, region}]
    regions: [],         // [{id, bbox, num_lines}] — from latest segmentation
    isProcessing: false,
};

// ── Event bus ──────────────────────────────────────────────────────────
export const events = new EventTarget();
export function emit(name, detail) {
    events.dispatchEvent(new CustomEvent(name, { detail }));
}
export function on(name, fn) {
    events.addEventListener(name, e => fn(e.detail));
}

// ── API helper ─────────────────────────────────────────────────────────
export async function api(path, options = {}) {
    const resp = await fetch(path, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || err.message || 'API error');
    }
    return resp;
}

// ── Toast notifications ────────────────────────────────────────────────
export function toast(message, type = 'info', durationMs = 4000) {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), durationMs);
}

// ── GPU status widget ──────────────────────────────────────────────────
function shortName(name) {
    // Abbreviate long GPU names for the header
    return name
        .replace('NVIDIA ', '')
        .replace('GeForce ', '')
        .replace('Tesla ', '')
        .replace('Quadro ', '');
}

async function updateGpuStatus() {
    const widget = document.getElementById('gpu-status');
    try {
        const resp = await api('/api/gpu');
        const data = await resp.json();

        if (!data.available || data.gpus.length === 0) {
            widget.innerHTML = '<span class="gpu-card-name"><span>GPU: N/A</span></span>';
            return;
        }

        widget.innerHTML = data.gpus.map(g => {
            const usedPct = Math.round((g.memory_used_mb / g.memory_total_mb) * 100);
            const fillClass = usedPct >= 85 ? 'hot' : usedPct >= 60 ? 'warm' : '';
            const usedGb   = (g.memory_used_mb / 1000).toFixed(1);
            const totalGb  = (g.memory_total_mb / 1000).toFixed(0);
            const utilHtml = g.utilization_gpu_pct != null
                ? `<span class="gpu-util-pct">${g.utilization_gpu_pct}%</span>` : '';

            return `<div class="gpu-card">
                <div class="gpu-card-name">
                    <span title="${g.name}">${shortName(g.name)}</span>${utilHtml}
                </div>
                <div class="gpu-mem-bar">
                    <div class="gpu-mem-fill ${fillClass}" style="width:${usedPct}%"></div>
                </div>
                <div class="gpu-mem-label">${usedGb}/${totalGb} GB VRAM</div>
            </div>`;
        }).join('');
    } catch {
        widget.innerHTML = '<span style="font-size:.75rem;color:var(--text-muted)">GPU: error</span>';
    }
}

// ── Zoom controls ──────────────────────────────────────────────────────
let zoomLevel = 1.0;
const ZOOM_STEP = 0.25;
let   ZOOM_MIN  = 0.25; // updated per image in fitZoom() so large images are always reachable
const ZOOM_MAX  = 4.0;

function applyZoom(level) {
    const img    = document.getElementById('page-image');
    const canvas = document.getElementById('overlay-canvas');
    if (!img || !img.naturalWidth) return;

    zoomLevel = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, level));
    const w = Math.round(img.naturalWidth  * zoomLevel);
    const h = Math.round(img.naturalHeight * zoomLevel);

    img.style.width  = w + 'px';
    img.style.height = h + 'px';
    canvas.style.width  = w + 'px';
    canvas.style.height = h + 'px';

    document.getElementById('zoom-level').textContent =
        Math.round(zoomLevel * 100) + '%';
}

export function fitZoom() {
    const img    = document.getElementById('page-image');
    const scroll = document.getElementById('viewer-scroll');
    if (!img || !img.naturalWidth || !scroll) return;
    const scaleW = scroll.clientWidth  / img.naturalWidth;
    const scaleH = scroll.clientHeight / img.naturalHeight;
    const fit = Math.min(scaleW, scaleH, 1.0); // never zoom in beyond 100% on fit
    // Ensure the fit level is always reachable: lower ZOOM_MIN for large images (min 5%)
    ZOOM_MIN = Math.min(0.25, Math.max(0.05, fit));
    applyZoom(fit);
}

function initZoomControls() {
    document.getElementById('btn-zoom-in') .addEventListener('click', () => applyZoom(zoomLevel + ZOOM_STEP));
    document.getElementById('btn-zoom-out').addEventListener('click', () => applyZoom(zoomLevel - ZOOM_STEP));
    document.getElementById('btn-zoom-fit').addEventListener('click', fitZoom);

    // Mouse-wheel zoom in viewer — multiplicative for smooth feel
    document.getElementById('viewer-scroll').addEventListener('wheel', e => {
        if (!e.ctrlKey && !e.metaKey) return;
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.10 : 1 / 1.10;
        applyZoom(zoomLevel * factor);
    }, { passive: false });

    on('image-uploaded', () => {
        document.getElementById('zoom-toolbar').classList.remove('hidden');
        // fit after short delay to let image render
        setTimeout(fitZoom, 80);
    });

    // Also show toolbar when a batch item is displayed in the viewer
    on('batch-item-start', () => {
        document.getElementById('zoom-toolbar').classList.remove('hidden');
    });
}

// ── Sticky engine config (localStorage) ───────────────────────────────
const LS_ENGINE = 'polyscriptor_last_engine';
const LS_CONFIG = name => `polyscriptor_config_${name}`;

export function saveEngineConfig(engineName, configObj) {
    try {
        localStorage.setItem(LS_ENGINE, engineName);
        localStorage.setItem(LS_CONFIG(engineName), JSON.stringify(configObj));
    } catch { /* storage full or private mode */ }
}

export function loadSavedEngineName() {
    try { return localStorage.getItem(LS_ENGINE); } catch { return null; }
}

export function loadSavedEngineConfig(engineName) {
    try {
        const raw = localStorage.getItem(LS_CONFIG(engineName));
        return raw ? JSON.parse(raw) : null;
    } catch { return null; }
}

// ── Mobile tab helper ───────────────────────────────────────────────────
function mobileActivateTab(target) {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panels  = document.querySelectorAll('[data-panel]');
    if (!tabBtns.length) return;
    tabBtns.forEach(b => b.classList.toggle('active', b.dataset.target === target));
    panels.forEach(p => p.classList.toggle('panel-active', p.dataset.panel === target));
}

// ── Resizable panels ───────────────────────────────────────────────────
const LS_PANEL_LEFT  = 'polyscriptor_panel_left';
const LS_PANEL_RIGHT = 'polyscriptor_panel_right';

function initResizablePanels() {
    const app = document.getElementById('app');
    const handleLeft  = document.getElementById('resize-left');
    const handleRight = document.getElementById('resize-right');
    if (!handleLeft || !handleRight) return;

    // Restore saved widths
    const savedLeft  = localStorage.getItem(LS_PANEL_LEFT);
    const savedRight = localStorage.getItem(LS_PANEL_RIGHT);
    if (savedLeft)  document.documentElement.style.setProperty('--panel-left',  savedLeft);
    if (savedRight) document.documentElement.style.setProperty('--panel-right', savedRight);

    function startDrag(handle, isLeft) {
        handle.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        const onMove = (e) => {
            const appRect = app.getBoundingClientRect();
            const x = (e.touches ? e.touches[0].clientX : e.clientX) - appRect.left;
            const totalW = appRect.width;

            if (isLeft) {
                const w = Math.max(160, Math.min(x, totalW * 0.4));
                const val = Math.round(w) + 'px';
                document.documentElement.style.setProperty('--panel-left', val);
                localStorage.setItem(LS_PANEL_LEFT, val);
            } else {
                const w = Math.max(200, Math.min(totalW - x, totalW * 0.5));
                const val = Math.round(w) + 'px';
                document.documentElement.style.setProperty('--panel-right', val);
                localStorage.setItem(LS_PANEL_RIGHT, val);
            }
        };

        const onUp = () => {
            handle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            document.removeEventListener('touchmove', onMove);
            document.removeEventListener('touchend', onUp);
        };

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
        document.addEventListener('touchmove', onMove, { passive: true });
        document.addEventListener('touchend', onUp);
    }

    handleLeft.addEventListener('mousedown',  e => { e.preventDefault(); startDrag(handleLeft,  true); });
    handleRight.addEventListener('mousedown', e => { e.preventDefault(); startDrag(handleRight, false); });
    handleLeft.addEventListener('touchstart',  e => startDrag(handleLeft,  true),  { passive: true });
    handleRight.addEventListener('touchstart', e => startDrag(handleRight, false), { passive: true });
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────
function initKeyboardShortcuts() {
    document.addEventListener('keydown', e => {
        // Ignore when typing in an input / textarea / contenteditable
        const tag = e.target.tagName;
        const editable = e.target.isContentEditable;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || editable) return;

        // Ctrl+Enter — transcribe
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById('btn-transcribe')?.click();
            return;
        }

        // ArrowLeft / ArrowRight — batch prev/next
        if (e.key === 'ArrowLeft')  { e.preventDefault(); document.getElementById('btn-nav-prev')?.click(); }
        if (e.key === 'ArrowRight') { e.preventDefault(); document.getElementById('btn-nav-next')?.click(); }
    });
}

// ── Prevent browser from opening dropped files in a new tab ────────────
function initGlobalDropBlocker() {
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop',     e => e.preventDefault());
}

// ── Init ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initEnginePanel();
    initImageViewer();
    initTranscriptionPanel();
    initBatchPanel();
    initZoomControls();
    initResizablePanels();
    initKeyboardShortcuts();
    initGlobalDropBlocker();
    updateGpuStatus();
    setInterval(updateGpuStatus, 15000); // refresh every 15s

    // On mobile: auto-switch tab after key events
    on('image-uploaded',        () => mobileActivateTab('image'));
    on('segment-preview',       () => mobileActivateTab('image'));
    on('transcription-start',   () => mobileActivateTab('results'));
});
