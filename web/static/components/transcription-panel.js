/**
 * Transcription Panel — SSE progress, results, export
 */

import { state, emit, on, toast } from '../app.js';

const $ = id => document.getElementById(id);

// ── Font selector ───────────────────────────────────────────────────────
const LS_FONT = 'polyscriptor_results_font';

const FONTS = [
    { label: 'Monospace (default)',  value: '' },
    { label: 'Monomakh Unicode ✦',  value: 'Monomakh',         local: true            },
    { label: 'Old Standard TT',     value: 'Old Standard TT',  gf: 'Old+Standard+TT'  },
    { label: 'Noto Serif',          value: 'Noto Serif',       gf: 'Noto+Serif'        },
    { label: 'Crimson Pro',         value: 'Crimson Pro',      gf: 'Crimson+Pro'       },
    { label: 'IM Fell English',     value: 'IM Fell English',  gf: 'IM+Fell+English'   },
];

const _loadedFonts = new Set();

function _loadGoogleFont(gfParam) {
    const url = `https://fonts.googleapis.com/css2?family=${gfParam}&display=swap`;
    if (_loadedFonts.has(url)) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = url;
    document.head.appendChild(link);
    _loadedFonts.add(url);
}

function applyFont(value) {
    const f = FONTS.find(f => f.value === value);
    if (!f) return;
    if (f.gf) _loadGoogleFont(f.gf);
    if (f.value) {
        document.documentElement.style.setProperty(
            '--font-results', `"${f.value}", Georgia, serif`);
    } else {
        document.documentElement.style.removeProperty('--font-results');
    }
}

export function initTranscriptionPanel() {
    let _transcribeStart = null;
    let _numRegions = 1;
    let _columnMode = false;

    // Confidence threshold slider
    const slider = $('conf-threshold');
    const sliderVal = $('conf-threshold-val');
    slider.addEventListener('input', () => {
        const threshold = parseInt(slider.value, 10);
        sliderVal.textContent = threshold + '%';
        applyConfidenceFilter(threshold);
    });

    // Search / filter
    const searchInput = $('results-search');
    searchInput.addEventListener('input', () => applySearch(searchInput.value));
    // Clear search on new transcription
    function resetSearch() {
        searchInput.value = '';
        $('results-search-row').classList.add('hidden');
        $('results-search-count').textContent = '';
    }

    // Font selector — populate, restore, handle changes
    const fontSel = $('font-select');
    for (const f of FONTS) {
        const o = document.createElement('option');
        o.value = f.value;
        o.textContent = f.label;
        fontSel.appendChild(o);
    }
    const savedFont = (() => { try { return localStorage.getItem(LS_FONT) || ''; } catch { return ''; } })();
    fontSel.value = savedFont;
    if (savedFont) applyFont(savedFont);
    fontSel.addEventListener('change', () => {
        applyFont(fontSel.value);
        try { localStorage.setItem(LS_FONT, fontSel.value); } catch { /* private mode */ }
    });

    // Column layout toggle
    $('btn-col-layout').addEventListener('click', () => {
        _columnMode = !_columnMode;
        $('btn-col-layout').classList.toggle('active', _columnMode);
        if (_columnMode) renderAllColumns();
        else renderAllFlat();
    });

    on('transcription-start', () => {
        state.lines = [];
        _transcribeStart = null;
        _numRegions = 1;
        _columnMode = false;
        $('btn-col-layout').classList.add('hidden');
        $('btn-col-layout').classList.remove('active');
        $('transcription-lines').innerHTML = '';
        $('transcription-lines').classList.remove('col-layout');
        $('progress-container').classList.remove('hidden');
        $('results-footer').classList.add('hidden');
        $('conf-filter-row').classList.add('hidden');
        resetSearch();
        $('progress-fill').style.width = '0%';
        $('progress-fill').style.background = '';  // reset error colour
        $('progress-text').textContent = 'Segmenting...';
    });

    // Highlight line in transcription panel when a bbox is clicked (or line clicked)
    on('highlight-line', ({ index }) => {
        const container = $('transcription-lines');
        container.querySelectorAll('.line-active').forEach(el => el.classList.remove('line-active'));
        const target = container.querySelector(`[data-index="${index}"]`);
        if (target) {
            target.classList.add('line-active');
            target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    });

    on('sse-status', data => {
        $('progress-text').textContent = data.message;
    });

    on('sse-segmentation', data => {
        if (data.source === 'page') {
            $('progress-text').textContent = 'Processing full page...';
        } else {
            $('progress-text').textContent = `${data.num_lines} lines found. Transcribing...`;
        }
    });

    on('sse-progress', data => {
        const pct = Math.round((data.current / data.total) * 100);
        $('progress-fill').style.width = pct + '%';

        // ETA
        const now = Date.now();
        if (!_transcribeStart) _transcribeStart = now;
        const elapsed = (now - _transcribeStart) / 1000;
        const rate = data.current / elapsed;  // lines/s
        const remaining = rate > 0 ? Math.round((data.total - data.current) / rate) : null;
        const etaStr = remaining != null
            ? ` · ~${remaining < 60 ? remaining + 's' : Math.round(remaining / 60) + 'min'} left`
            : '';
        let tokenStr = '';
        if (data.token_usage) {
            const tu = data.token_usage;
            const parts = [];
            if (tu.prompt_tokens  != null) parts.push(`in:${tu.prompt_tokens}`);
            if (tu.output_tokens  != null) parts.push(`out:${tu.output_tokens}`);
            if (tu.thinking_tokens != null && tu.thinking_tokens > 0) parts.push(`think:${tu.thinking_tokens}`);
            if (parts.length) tokenStr = ` | ${parts.join(' ')} tok`;
        }
        $('progress-text').textContent = `${data.current} / ${data.total} lines${etaStr}${tokenStr}`;

        _numRegions = Math.max(_numRegions, (data.line.region ?? 0) + 1);
        state.lines.push(data.line);
        appendLine(data.line);
    });

    on('sse-complete', data => {
        $('progress-container').classList.add('hidden');
        $('results-footer').classList.remove('hidden');
        $('btn-export-xml').classList.remove('hidden');
        let summary = `${data.lines.length} lines in ${data.total_time_s}s (${data.engine})`;
        if (data.token_usage) {
            const tu = data.token_usage;
            const parts = [];
            if (tu.prompt_tokens != null)   parts.push(`in: ${tu.prompt_tokens}`);
            if (tu.output_tokens != null)   parts.push(`out: ${tu.output_tokens}`);
            if (tu.thinking_tokens != null && tu.thinking_tokens > 0)
                parts.push(`think: ${tu.thinking_tokens}`);
            if (parts.length) summary += ` | tokens: ${parts.join(', ')}`;
        }
        $('results-summary').textContent = summary;
        // Show confidence filter if any line has confidence data
        if (state.lines.some(l => l.confidence != null)) {
            $('conf-filter-row').classList.remove('hidden');
            slider.value = 0;
            sliderVal.textContent = '0%';
        }
        // Show search if there are results
        if (state.lines.length > 0) {
            $('results-search-row').classList.remove('hidden');
        }
        // Show column layout toggle if multiple regions detected
        if (_numRegions > 1) {
            $('btn-col-layout').classList.remove('hidden');
        }
        emit('transcription-complete', data);
    });

    on('sse-cancelled', () => {
        $('progress-text').textContent = 'Cancelled';
        $('progress-fill').style.width = '0%';
        // Show footer if we have partial results
        if (state.lines.length > 0) {
            $('results-footer').classList.remove('hidden');
            $('results-summary').textContent = `Cancelled — ${state.lines.length} lines transcribed`;
        }
        emit('transcription-complete', {});
    });

    on('sse-error', data => {
        $('progress-text').textContent = `Error: ${data.message}`;
        $('progress-fill').style.width = '0%';
        $('progress-fill').style.background = 'var(--danger)';
        emit('transcription-complete', {});
    });

    on('transcription-error', data => {
        $('progress-text').textContent = `Error: ${data.message}`;
        emit('transcription-complete', {});
    });

    // Also hide Export XML when a new transcription starts
    on('transcription-start', () => {
        $('btn-export-xml').classList.add('hidden');
    });

    $('btn-copy-text').addEventListener('click', copyText);
    $('btn-export-txt').addEventListener('click', exportTxt);
    $('btn-export-csv').addEventListener('click', exportCsv);
    $('btn-export-xml').addEventListener('click', exportXml);
}

function renderAllFlat() {
    const container = $('transcription-lines');
    container.innerHTML = '';
    container.classList.remove('col-layout');
    state.lines.forEach(line => appendLine(line));
}

function renderAllColumns() {
    const container = $('transcription-lines');
    container.innerHTML = '';
    container.classList.add('col-layout');

    const maxRegion = state.lines.reduce((m, l) => Math.max(m, l.region ?? 0), 0);
    const groups = Array.from({ length: maxRegion + 1 }, () => []);
    state.lines.forEach(line => groups[line.region ?? 0].push(line));

    groups.forEach((lines, r) => {
        const col = document.createElement('div');
        col.className = 'region-column';

        const hdr = document.createElement('div');
        hdr.className = 'region-col-header';

        const title = document.createElement('span');
        title.textContent = `Column ${r + 1}  (${lines.length})`;
        hdr.appendChild(title);

        const closeBtn = document.createElement('button');
        closeBtn.className = 'region-col-close';
        closeBtn.textContent = '×';
        closeBtn.title = 'Hide this column';
        closeBtn.addEventListener('click', e => { e.stopPropagation(); col.remove(); });
        hdr.appendChild(closeBtn);

        col.appendChild(hdr);
        lines.forEach(line => appendLine(line, col));
        container.appendChild(col);
    });
}

function appendLine(line, container = null) {
    container = container || $('transcription-lines');
    const div = document.createElement('div');
    div.className = 'line-result';
    div.dataset.index = line.index;
    if (line.confidence != null) {
        div.dataset.confidence = Math.round(line.confidence * 100);
    }

    // Line number
    const numSpan = document.createElement('span');
    numSpan.className = 'line-num';
    numSpan.textContent = line.index + 1;

    // Editable text span
    const textSpan = document.createElement('span');
    textSpan.className = 'line-text';
    textSpan.textContent = line.text;

    // Confidence badge
    let confSpan = null;
    if (line.confidence != null) {
        const pct = Math.round(line.confidence * 100);
        const cls = pct >= 90 ? 'conf-high' : pct >= 75 ? 'conf-mid' : 'conf-low';
        confSpan = document.createElement('span');
        confSpan.className = `confidence ${cls}`;
        confSpan.textContent = pct + '%';
    }

    div.appendChild(numSpan);
    div.appendChild(textSpan);
    if (confSpan) div.appendChild(confSpan);

    // Thinking text (Gemini reasoning) — collapsible per line
    if (line.thinking_text) {
        const details = document.createElement('details');
        details.className = 'thinking-block';
        const summary = document.createElement('summary');
        summary.className = 'thinking-toggle';
        summary.textContent = 'reasoning';
        const pre = document.createElement('pre');
        pre.className = 'thinking-text';
        pre.textContent = line.thinking_text;
        details.appendChild(summary);
        details.appendChild(pre);
        div.appendChild(details);
    }

    // Single click → highlight bbox on image
    div.addEventListener('click', e => {
        if (textSpan.contentEditable === 'true') return; // don't interfere while editing
        emit('highlight-line', { index: line.index });
    });

    // Double-click → start inline editing
    textSpan.addEventListener('dblclick', e => {
        e.stopPropagation();
        textSpan.contentEditable = 'true';
        textSpan.focus();
        // Select all text for easy replacement
        const range = document.createRange();
        range.selectNodeContents(textSpan);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
    });

    // Save on blur or Enter
    const saveEdit = () => {
        textSpan.contentEditable = 'false';
        const newText = textSpan.textContent;
        if (newText !== line.text) {
            state.lines[line.index].text = newText;
            div.classList.add('line-edited');
        }
    };
    textSpan.addEventListener('blur', saveEdit);
    textSpan.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); saveEdit(); }
        if (e.key === 'Escape') {
            textSpan.textContent = state.lines[line.index].text; // revert
            textSpan.contentEditable = 'false';
        }
    });

    container.appendChild(div);
    // Auto-scroll only for the main flat container (not column sub-divs)
    if (container === $('transcription-lines')) {
        container.scrollTop = container.scrollHeight;
    }
}

function applyConfidenceFilter(threshold) {
    $('transcription-lines').querySelectorAll('.line-result').forEach(div => {
        const conf = parseInt(div.dataset.confidence ?? '100', 10);
        div.classList.toggle('line-dimmed', conf < threshold);
    });
}

function applySearch(query) {
    const lines = $('transcription-lines').querySelectorAll('.line-result');
    const q = query.trim().toLowerCase();
    let matchCount = 0;

    lines.forEach(div => {
        const textSpan = div.querySelector('.line-text');
        if (!textSpan) return;
        // Use state.lines for the canonical text (survives inline edits and search markup)
        const lineIdx = parseInt(div.dataset.index ?? '-1', 10);
        const raw = lineIdx >= 0 && state.lines[lineIdx]
            ? state.lines[lineIdx].text
            : textSpan.textContent;

        if (!q) {
            // Clear search: restore plain text, remove hidden
            textSpan.textContent = raw;
            div.classList.remove('line-hidden');
            return;
        }

        const lc = raw.toLowerCase();
        const idx = lc.indexOf(q);
        if (idx === -1) {
            div.classList.add('line-hidden');
        } else {
            div.classList.remove('line-hidden');
            matchCount++;
            // Highlight match with <mark> using safe DOM manipulation
            const before = raw.slice(0, idx);
            const match  = raw.slice(idx, idx + q.length);
            const after  = raw.slice(idx + q.length);
            textSpan.textContent = '';
            textSpan.appendChild(document.createTextNode(before));
            const mark = document.createElement('mark');
            mark.textContent = match;
            textSpan.appendChild(mark);
            textSpan.appendChild(document.createTextNode(after));
        }
    });

    const countEl = $('results-search-count');
    countEl.textContent = q ? `${matchCount} match${matchCount !== 1 ? 'es' : ''}` : '';
}

// (escapeHtml no longer needed — we use textContent/DOM directly)

async function copyText() {
    if (state.lines.length === 0) return;
    const text = state.lines.map(l => l.text).join('\n');
    try {
        await navigator.clipboard.writeText(text);
        const btn = $('btn-copy-text');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1500);
    } catch {
        toast('Clipboard not available — use Export TXT instead', 'error');
    }
}

function exportTxt() {
    if (state.lines.length === 0) return;
    const text = state.lines.map(l => l.text).join('\n');
    downloadFile('transcription.txt', text, 'text/plain');
}

function exportCsv() {
    if (state.lines.length === 0) return;
    const header = 'Line,Text,Confidence,X1,Y1,X2,Y2\n';
    const rows = state.lines.map(l => {
        const conf = l.confidence != null ? l.confidence.toFixed(4) : '';
        const bbox = l.bbox ? l.bbox.join(',') : ',,,';
        return `${l.index + 1},"${l.text.replace(/"/g, '""')}",${conf},${bbox}`;
    }).join('\n');
    downloadFile('transcription.csv', header + rows, 'text/csv');
}

function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

async function exportXml() {
    if (!state.imageId) return;
    try {
        const resp = await fetch(`/api/image/${state.imageId}/export-xml`, { method: 'POST' });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            toast(`XML export failed: ${err.detail || resp.statusText}`, 'error');
            return;
        }
        const blob = await resp.blob();
        // Use filename from Content-Disposition if provided, else fall back
        let filename = 'transcription.xml';
        const cd = resp.headers.get('Content-Disposition');
        if (cd) {
            const m = cd.match(/filename="([^"]+)"/);
            if (m) filename = m[1];
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        toast(`XML export error: ${err.message}`, 'error');
    }
}
