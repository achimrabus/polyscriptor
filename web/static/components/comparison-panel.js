/**
 * Comparison Panel — secondary engine runs on cached line segments.
 *
 * V1 keeps the main transcription flow unchanged. Users create a base
 * transcription first, then switch the engine/model in the left panel and run
 * additional comparison passes against the same segmented lines.
 */

import { state, emit, on, api, toast } from '../app.js';

const $ = id => document.getElementById(id);

function resetComparisonState() {
    state.comparison = {
        base: null,
        runs: [],
        selectedSlotId: null,
        isRunning: false,
        expanded: false,
    };
}

function setStatus(message = '', type = '') {
    $('comparison-status').textContent = message;
    const badge = $('comparison-status-badge');
    if (!type) {
        badge.className = 'status-badge hidden';
        badge.textContent = '';
        return;
    }
    badge.className = `status-badge ${type}`;
    badge.classList.remove('hidden');
    badge.textContent = type === 'status-loaded' ? 'Ready' : 'Running';
}

// The comparison workspace is opt-in: it only appears when the user clicks the
// "Compare" toggle in the results header. We never auto-expand it after a plain
// transcription so users who just want to transcribe aren't interrupted.
function updateComparisonVisibility() {
    const base = state.comparison.base;
    const toggle = $('btn-compare-toggle');
    const panel = $('comparison-panel');
    const resultsPanel = $('results-panel');

    if (!base) {
        toggle.classList.add('hidden');
        toggle.classList.remove('active');
        panel.classList.add('hidden');
        resultsPanel.classList.remove('compare-mode');
        state.comparison.expanded = false;
        return;
    }

    toggle.classList.remove('hidden');
    const expanded = !!state.comparison.expanded;
    toggle.classList.toggle('active', expanded);
    panel.classList.toggle('hidden', !expanded);
    // Expanded comparison takes over the results column full-height: the plain
    // transcription list is redundant here (its text is the base column).
    resultsPanel.classList.toggle('compare-mode', expanded);
}

function updateBaseSummary() {
    const base = state.comparison.base;
    const summary = $('comparison-base-summary');
    const help = $('comparison-help');
    if (!base) {
        summary.textContent = '';
        help.textContent = '';
        return;
    }

    if (base.segSource === 'page') {
        summary.textContent = `${base.label} uses a page-level result. Run a segmented base transcription first to enable comparison.`;
        help.textContent = 'Comparison V1 uses cached line segmentation from the base transcription so all engines can be aligned line-by-line.';
    } else {
        summary.textContent = `Base result: ${base.label} · ${base.lineCount} lines`;
        help.textContent = 'To add a comparison, switch to another engine on the left, load its model, and run a comparison on the same cached line segments.';
    }
}

function collectLiveOverrides() {
    const overrides = {};
    for (const el of $('config-form').querySelectorAll('[data-key]')) {
        if (el.dataset.saveFor) continue;
        if (el.dataset.passwordField) continue;
        const key = el.dataset.key;
        if (el.type === 'checkbox') overrides[key] = el.checked;
        else if (el.type === 'number') overrides[key] = Number(el.value);
        else overrides[key] = el.value;
    }
    return overrides;
}

function renderSlotSelect() {
    const select = $('comparison-slot-select');
    const label = document.querySelector('label[for="comparison-slot-select"]');
    select.innerHTML = '';

    if (state.comparison.runs.length === 0) {
        select.classList.add('hidden');
        label.classList.add('hidden');
        return;
    }

    for (const run of state.comparison.runs) {
        const opt = document.createElement('option');
        opt.value = run.comparison_slot.slot_id;
        opt.textContent = run.comparison_slot.label;
        select.appendChild(opt);
    }

    select.value = state.comparison.selectedSlotId || state.comparison.runs[0].comparison_slot.slot_id;
    select.classList.remove('hidden');
    label.classList.remove('hidden');
}

function summaryCard(label, value, note = '') {
    return `<div class="comparison-summary-card">
        <span class="label">${label}</span>
        <span class="value">${value}</span>
        ${note ? `<span class="note">${note}</span>` : ''}
    </div>`;
}

function renderSummary(run) {
    const box = $('comparison-summary');
    if (!run) {
        box.classList.add('hidden');
        box.innerHTML = '';
        return;
    }

    const labels = run.labels;
    const summary = run.summary;
    const identical = summary.identical_lines ?? 0;
    const identicalPct = summary.line_count
        ? ` (${((identical / summary.line_count) * 100).toFixed(0)}%)`
        : '';
    box.innerHTML = [
        summaryCard(labels.macro_char_rate, `${summary.macro_char_rate.toFixed(2)}%`, `${summary.line_count} lines`),
        summaryCard(labels.micro_char_rate, `${summary.micro_char_rate.toFixed(2)}%`, `${summary.total_edit_distance} edits`),
        summaryCard(labels.macro_word_rate, `${summary.macro_word_rate.toFixed(2)}%`),
        summaryCard(`Avg ${labels.match_rate}`, `${summary.avg_match_percent.toFixed(2)}%`, run.comparison_slot.label),
        summaryCard('Identical lines', `${identical} / ${summary.line_count}`, `agree${identicalPct}`),
    ].join('');
    box.classList.remove('hidden');
}

function renderLines(run) {
    const container = $('comparison-lines');
    if (!run) {
        container.classList.add('hidden');
        container.innerHTML = '';
        return;
    }

    const labels = run.labels;
    const [low, mid] = labels.color_thresholds || [15, 35];
    container.innerHTML = '';

    // Column labels rendered once as a sticky header instead of per row.
    // Three columns: a narrow gutter (line # + mismatch %), then base/comparison.
    const header = document.createElement('div');
    header.className = 'comparison-col-header';
    const gutterTitle = document.createElement('span');
    gutterTitle.className = 'comparison-col-title comparison-gutter-title';
    gutterTitle.textContent = '#';
    gutterTitle.title = `Per-line ${labels.char_rate.toLowerCase()}`;
    const baseTitle = document.createElement('span');
    baseTitle.className = 'comparison-col-title';
    baseTitle.textContent = run.base_slot.label;
    const compTitle = document.createElement('span');
    compTitle.className = 'comparison-col-title';
    compTitle.textContent = run.comparison_slot.label;
    header.appendChild(gutterTitle);
    header.appendChild(baseTitle);
    header.appendChild(compTitle);
    container.appendChild(header);

    for (const row of run.lines) {
        const block = document.createElement('div');
        block.className = `comparison-line${row.has_disagreement ? ' comparison-line-disagree' : ' comparison-line-match'}`;
        block.title = `${labels.char_rate} ${row.metrics.char_rate.toFixed(1)}% · ${labels.word_rate} ${row.metrics.word_rate.toFixed(1)}% · ${labels.match_rate} ${row.metrics.match_percent.toFixed(1)}%`;

        block.appendChild(createGutter(row, low, mid));
        block.appendChild(createCell('base', row));
        block.appendChild(createCell('comparison', row));
        container.appendChild(block);
    }
    container.classList.remove('hidden');
}

// Narrow left gutter: line number + per-line mismatch %, colour-coded by the
// engine-comparison thresholds. Replaces the old full-width per-line header row.
function createGutter(row, low, mid) {
    const gutter = document.createElement('div');
    gutter.className = 'comparison-gutter';

    const num = document.createElement('span');
    num.className = 'comparison-line-num';
    num.textContent = row.index + 1;

    const pct = document.createElement('span');
    const value = row.metrics.char_rate;
    const level = value <= low ? 'good' : value <= mid ? 'warn' : 'bad';
    pct.className = `comparison-mismatch ${level}`;
    pct.textContent = `${value.toFixed(0)}%`;

    gutter.appendChild(num);
    gutter.appendChild(pct);
    gutter.addEventListener('click', () => emit('highlight-line', { index: row.index }));
    return gutter;
}

function createCell(side, row) {
    const cell = document.createElement('div');
    cell.className = 'comparison-cell';

    const value = document.createElement('div');
    value.className = 'comparison-cell-text';

    const text = side === 'base' ? row.base_text : row.comparison_text;
    if (!text) {
        value.textContent = '(empty)';
        value.classList.add('empty');
    } else if (row.diff_ops && row.diff_ops.length) {
        renderDiff(value, row.diff_ops, side);
    } else {
        value.textContent = text;
    }

    value.addEventListener('click', () => emit('highlight-line', { index: row.index }));
    cell.appendChild(value);
    return cell;
}

/**
 * Render a char-level diff into `target`.
 * Base column shows what was removed/changed (delete + replace);
 * comparison column shows what was added/changed (insert + replace).
 */
function renderDiff(target, diffOps, side) {
    for (const op of diffOps) {
        if (op.op === 'equal') {
            target.appendChild(document.createTextNode(side === 'base' ? op.r : op.h));
        } else if (op.op === 'replace') {
            appendDiffSpan(target, side === 'base' ? op.r : op.h, 'diff-sub');
        } else if (op.op === 'delete' && side === 'base') {
            appendDiffSpan(target, op.r, 'diff-del');
        } else if (op.op === 'insert' && side === 'comparison') {
            appendDiffSpan(target, op.h, 'diff-ins');
        }
    }
}

function appendDiffSpan(target, ch, cls) {
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = ch;
    target.appendChild(span);
}

function renderSelectedRun() {
    const run = state.comparison.runs.find(
        entry => entry.comparison_slot.slot_id === state.comparison.selectedSlotId,
    ) || null;
    renderSummary(run);
    renderLines(run);
}

function updateControls() {
    updateComparisonVisibility();
    updateBaseSummary();
    renderSlotSelect();
    renderSelectedRun();

    const btn = $('btn-run-comparison');
    const base = state.comparison.base;
    const ready = !!base && base.segSource !== 'page' && state.engineLoaded && state.imageId && !state.comparison.isRunning;
    btn.disabled = !ready;
    if (!base) {
        setStatus('', '');
    }
}

async function onRunComparison() {
    if (state.comparison.isRunning) return;
    const base = state.comparison.base;
    if (!base || !state.imageId) return;
    if (base.segSource === 'page') {
        toast('Comparison requires a segmented base transcription.', 'error');
        return;
    }
    if (!state.engineLoaded) {
        toast('Load the comparison engine in the left panel first.', 'error');
        return;
    }

    state.comparison.isRunning = true;
    emit('comparison-start', {});
    setStatus(`Running comparison with current engine…`, 'status-loading');
    $('btn-run-comparison').classList.add('loading');
    $('btn-run-comparison').textContent = 'Comparing…';
    updateControls();

    try {
        const resp = await fetch('/api/compare/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.imageId,
                engine_config_overrides: collectLiveOverrides(),
            }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Comparison failed');
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const parts = buffer.split('\n\n');
            buffer = parts.pop() || '';
            for (const part of parts) {
                if (!part.trim()) continue;
                const eventMatch = part.match(/event: (\w+)/);
                const dataMatch = part.match(/data: (.+)/s);
                if (!eventMatch || !dataMatch) continue;
                const eventName = eventMatch[1];
                const data = JSON.parse(dataMatch[1]);
                emit(`comparison-sse-${eventName}`, data);
            }
        }
    } catch (err) {
        emit('comparison-error', { message: err.message });
    }
}

export function initComparisonPanel() {
    resetComparisonState();

    $('btn-compare-toggle').addEventListener('click', () => {
        state.comparison.expanded = !state.comparison.expanded;
        updateControls();
        if (state.comparison.expanded) {
            $('comparison-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });

    $('btn-run-comparison').addEventListener('click', onRunComparison);
    $('comparison-slot-select').addEventListener('change', e => {
        state.comparison.selectedSlotId = e.target.value;
        renderSelectedRun();
    });

    on('image-uploaded', () => {
        resetComparisonState();
        updateControls();
    });
    on('batch-item-start', () => {
        resetComparisonState();
        updateControls();
    });
    on('transcription-start', () => {
        resetComparisonState();
        updateControls();
    });
    on('engine-loaded', () => updateControls());

    on('transcription-complete', data => {
        if (!data?.lines?.length || !data?.result_slot) return;
        state.comparison.base = {
            slotId: data.result_slot.slot_id,
            label: data.result_slot.label,
            engineName: data.result_slot.engine_name,
            segSource: data.seg_source || data.result_slot.seg_source || 'unknown',
            lineCount: data.result_slot.line_count || data.lines.length,
        };
        state.comparison.runs = [];
        state.comparison.selectedSlotId = null;
        updateControls();
    });

    on('comparison-sse-status', data => {
        setStatus(data.message, 'status-loading');
    });

    on('comparison-sse-progress', data => {
        setStatus(`Comparing ${data.current} / ${data.total} lines…`, 'status-loading');
    });

    on('comparison-sse-complete', data => {
        state.comparison.isRunning = false;
        $('btn-run-comparison').classList.remove('loading');
        $('btn-run-comparison').textContent = 'Run Comparison';

        const existingIndex = state.comparison.runs.findIndex(
            run => run.comparison_slot.slot_id === data.comparison_slot.slot_id,
        );
        if (existingIndex >= 0) state.comparison.runs[existingIndex] = data;
        else state.comparison.runs.push(data);
        state.comparison.selectedSlotId = data.comparison_slot.slot_id;

        setStatus(`Comparison ready: ${data.comparison_slot.label} (${data.total_time_s}s)`, 'status-loaded');
        updateControls();
    });

    on('comparison-sse-error', data => {
        state.comparison.isRunning = false;
        $('btn-run-comparison').classList.remove('loading');
        $('btn-run-comparison').textContent = 'Run Comparison';
        setStatus(`Comparison failed: ${data.message}`, '');
        updateControls();
        toast(`Comparison failed: ${data.message}`, 'error');
    });

    on('comparison-error', data => {
        state.comparison.isRunning = false;
        $('btn-run-comparison').classList.remove('loading');
        $('btn-run-comparison').textContent = 'Run Comparison';
        setStatus(`Comparison failed: ${data.message}`, '');
        updateControls();
        toast(`Comparison failed: ${data.message}`, 'error');
    });

    on('comparison-start', () => {
        state.comparison.isRunning = true;
        updateControls();
    });

    updateControls();
}
