/**
 * Batch Panel — multi-image queue, sequential processing, combined export
 *
 * Activated when the user selects/drops multiple images.
 * Each item is processed using the existing upload + transcribe flow.
 * Results are stored per-item and can be exported as combined TXT or CSV.
 */

import { state, emit, on, api, toast } from '../app.js';

const $ = id => document.getElementById(id);

// Batch state (separate from state.lines which tracks the current single image)
const batch = {
    items: [],      // { file, imageId, status, lines, filename }
    running: false,
    cancelled: false,
    currentIndex: -1,     // item currently shown in the viewer
    processingIndex: -1,  // item currently being transcribed (may differ when user navigates away)
    userNavigated: false, // user manually navigated away from auto-advance
    abortController: null,
};

export function initBatchPanel() {
    // Hook into the file input to detect multiple files, PDFs, or second image.
    // Use capture:true so this fires before image-viewer's bubble listener, letting us
    // stopImmediatePropagation() and own the upload when batch-panel takes over.
    const fileInput = $('file-input');
    fileInput.addEventListener('change', e => {
        const files = Array.from(fileInput.files);
        const hasPdf = files.some(f => f.name.toLowerCase().endsWith('.pdf'));
        // Intercept: multiple files, PDF, or single image when one is already loaded
        if (files.length > 1 || hasPdf || (files.length === 1 && !hasPdf && state.imageId)) {
            e.stopImmediatePropagation(); // prevent image-viewer from also uploading the PDF
            handleMultipleFiles(files);
            fileInput.value = '';
        }
        // Single non-PDF image with no existing image → handled by image-viewer.js
    }, true); // capture:true — fires before image-viewer's non-capture listener

    // Multiple XML selection from the Upload XML button
    const xmlInput = $('xml-input');
    xmlInput.addEventListener('change', e => {
        if (xmlInput.files.length <= 1) return; // single XML → image-viewer handles normally
        e.stopImmediatePropagation();
        uploadXmlFiles(Array.from(xmlInput.files));
        xmlInput.value = '';
    }, true); // capture — fires before image-viewer's listener

    // Drag-drop: intercept multiple images/PDFs or any drop when image already loaded
    const uploadArea = $('upload-area');
    uploadArea.addEventListener('drop', e => {
        const files = Array.from(e.dataTransfer.files);
        const xmlFiles = files.filter(f => f.name.toLowerCase().endsWith('.xml'));
        const nonXml  = files.filter(f => !f.name.toLowerCase().endsWith('.xml'));
        const hasPdf  = nonXml.some(f => f.name.toLowerCase().endsWith('.pdf'));

        // Take over if: multiple images, a PDF, a second image on top of existing, or multiple XMLs
        const takeBatch = nonXml.length > 1 || hasPdf || (nonXml.length === 1 && state.imageId);
        const takeXml   = xmlFiles.length > 1 || (xmlFiles.length === 1 && batch.items.length > 0);

        if (takeBatch || takeXml) {
            e.preventDefault();
            e.stopImmediatePropagation();
            if (nonXml.length > 0) handleMultipleFiles(nonXml);
            if (xmlFiles.length > 0) uploadXmlFiles(xmlFiles);
        }
    }, true); // capture phase — fires before image-viewer's bubble handler

    // PDF pages from single-PDF drop on image-viewer — add to batch
    on('pdf-pages-ready', data => {
        const existing = new Set(batch.items.map(i => i.filename));
        for (const page of data.pages) {
            if (!existing.has(page.filename)) {
                batch.items.push({
                    file: null,
                    imageId: page.image_id,
                    status: 'pending',
                    lines: [],
                    filename: page.filename,
                    preUploaded: true,
                });
                existing.add(page.filename);
            }
        }
        if (batch.items.length > 0) {
            renderQueue();
            // PDF pages are already uploaded — always preview the first one directly,
            // bypassing the state.imageId guard in previewFirstBatchItem().
            const first = batch.items[0];
            if (first && first.preUploaded && first.imageId) {
                batch.currentIndex = 0;
                emit('batch-item-start', { imageId: first.imageId, filename: first.filename });
                updateNavButtons();
            }
        }
    });

    $('btn-process-batch').addEventListener('click', processBatch);
    $('btn-clear-batch').addEventListener('click', clearBatch);
    $('btn-export-batch-txt').addEventListener('click', exportAllTxt);
    $('btn-export-batch-csv').addEventListener('click', exportAllCsv);
    $('btn-export-batch-xml').addEventListener('click', exportAllXml);

    $('btn-nav-prev').addEventListener('click', () => navigate(-1));
    $('btn-nav-next').addEventListener('click', () => navigate(+1));

    // Persist PAGE XML and resume checkboxes across sessions
    const usePageXmlEl = $('batch-use-pagexml');
    const resumeEl     = $('batch-resume');
    const savedPageXml = localStorage.getItem('batch_use_pagexml');
    const savedResume  = localStorage.getItem('batch_resume');
    if (savedPageXml !== null) usePageXmlEl.checked = savedPageXml === 'true';
    if (savedResume  !== null) resumeEl.checked     = savedResume  === 'true';
    usePageXmlEl.addEventListener('change', () => localStorage.setItem('batch_use_pagexml', usePageXmlEl.checked));
    resumeEl.addEventListener('change',     () => localStorage.setItem('batch_resume',      resumeEl.checked));

    // Cancel during batch: abort current SSE + stop the queue loop
    $('btn-cancel').addEventListener('click', () => {
        if (!batch.running) return;
        batch.cancelled = true;
        batch.abortController?.abort();
    }, { capture: true });
}

// ── XML matching for batch ────────────────────────────────────────────────────

// Match XML files to batch items by filename stem (e.g. page001.xml → page001.jpg)
async function uploadXmlFiles(xmlFiles) {
    if (!xmlFiles.length) return;
    const stem = name => name.replace(/\.[^/.]+$/, '').toLowerCase();

    let matched = 0, deferred = 0, skipped = 0;

    for (const xml of xmlFiles) {
        const xmlStem = stem(xml.name);
        const item = batch.items.find(it => stem(it.filename) === xmlStem);
        if (!item) { skipped++; continue; }

        if (item.imageId) {
            // Already uploaded → send to server immediately
            try {
                const fd = new FormData();
                fd.append('file', xml);
                const resp = await fetch(`/api/image/${item.imageId}/xml`, { method: 'POST', body: fd });
                if (!resp.ok) throw new Error((await resp.json()).detail);
                item.xmlUploaded = true;
                matched++;
            } catch (err) {
                toast(`XML ${xml.name}: ${err.message}`, 'error');
            }
        } else {
            // Image not yet uploaded — store XML, send during processBatch
            item.xmlFile = xml;
            deferred++;
        }
    }

    const parts = [];
    if (matched  > 0) parts.push(`${matched} uploaded`);
    if (deferred > 0) parts.push(`${deferred} queued for batch`);
    if (skipped  > 0) parts.push(`${skipped} unmatched`);
    toast(`XML files: ${parts.join(', ')}`, matched + deferred > 0 ? 'success' : 'error');
}

// ── Queue management ─────────────────────────────────────────────────────────

function handleMultipleFiles(files) {
    // If a single image is already loaded (not yet in batch), add it first
    if (batch.items.length === 0 && state.imageId) {
        batch.items.push({
            file: null,
            imageId: state.imageId,
            status: 'pending',
            lines: state.lines.length ? state.lines : [],
            filename: (state.imageInfo && state.imageInfo.filename) || 'current image',
            preUploaded: true,
        });
    }
    // Add new files (skip duplicates by name)
    const existing = new Set(batch.items.map(i => i.filename));
    const added = files.filter(f => !existing.has(f.name));
    added.forEach(f => {
        batch.items.push({ file: f, imageId: null, status: 'pending', lines: [], filename: f.name });
    });
    if (batch.items.length > 0) { renderQueue(); previewFirstBatchItem(); }
}

// Auto-preview the first batch item (upload if needed) so the viewer isn't blank
async function previewFirstBatchItem() {
    if (state.imageId || batch.running) return;
    const first = batch.items[0];
    if (!first) return;
    if (first.preUploaded && first.imageId) {
        batch.currentIndex = 0;
        emit('batch-item-start', { imageId: first.imageId, filename: first.filename });
        updateNavButtons();
    } else if (first.file) {
        try {
            const fd = new FormData();
            fd.append('file', first.file);
            const resp = await fetch('/api/image/upload', { method: 'POST', body: fd });
            if (!resp.ok) return;
            const data = await resp.json();
            if (data.is_pdf) {
                // Expand PDF into page items immediately (same as processSingleItem does)
                const newItems = data.pages.map(p => ({
                    file: null, imageId: p.image_id, status: 'pending',
                    lines: [], filename: p.filename, preUploaded: true,
                }));
                batch.items.splice(0, 1, ...newItems);
                renderQueue();
                const firstPage = batch.items[0];
                if (firstPage) {
                    batch.currentIndex = 0;
                    emit('batch-item-start', { imageId: firstPage.imageId, filename: firstPage.filename });
                    updateNavButtons();
                }
                return;
            }
            first.imageId = data.image_id;
            first.preUploaded = true;
            batch.currentIndex = 0;
            emit('batch-item-start', { imageId: first.imageId, filename: first.filename });
            updateNavButtons();
        } catch { /* non-fatal */ }
    }
}

function clearBatch() {
    if (batch.running) return;
    batch.items = [];
    batch.currentIndex = -1;
    $('batch-queue-section').classList.add('hidden');
    $('batch-export-row').classList.add('hidden');
    updateNavButtons();
}

let _dragSrcIndex = null;

function renderQueue() {
    const section = $('batch-queue-section');
    const list = $('batch-list');
    section.classList.remove('hidden');
    list.innerHTML = '';
    batch.items.forEach((item, i) => {
        const row = document.createElement('div');
        row.className = 'batch-item';
        row.id = `batch-item-${i}`;
        row.dataset.index = i;

        // Drag handle
        const handle = document.createElement('span');
        handle.className = 'batch-drag-handle';
        handle.textContent = '⠿';
        handle.title = 'Drag to reorder';

        const name = document.createElement('span');
        name.className = 'batch-item-name';
        name.title = item.filename;
        name.textContent = item.filename;

        const status = document.createElement('span');
        status.className = 'batch-status';
        status.id = `batch-status-${i}`;
        _setStatusEl(status, item.status, item.lines.length);

        row.appendChild(handle);
        row.appendChild(name);
        row.appendChild(status);

        // Click a done item to reload it, or a preUploaded pending item to load for manual transcription
        const canPreview = item.status === 'done' || (item.preUploaded && item.imageId);
        if (canPreview) {
            row.style.cursor = 'pointer';
            row.addEventListener('click', e => {
                if (e.target === handle) return; // don't trigger on drag handle click
                if (item.status === 'done') {
                    loadBatchItem(i);
                } else {
                    // Load preUploaded pending page so user can manually segment/transcribe it
                    batch.currentIndex = i;
                    emit('batch-item-start', { imageId: item.imageId, filename: item.filename });
                    updateNavButtons();
                }
            });
        }

        // Drag-to-reorder (only when not running)
        if (!batch.running) {
            row.draggable = true;
            row.addEventListener('dragstart', e => {
                _dragSrcIndex = i;
                e.dataTransfer.effectAllowed = 'move';
                row.classList.add('batch-dragging');
            });
            row.addEventListener('dragend', () => {
                row.classList.remove('batch-dragging');
                list.querySelectorAll('.batch-item').forEach(r => r.classList.remove('batch-drag-over'));
            });
            row.addEventListener('dragover', e => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                list.querySelectorAll('.batch-item').forEach(r => r.classList.remove('batch-drag-over'));
                row.classList.add('batch-drag-over');
            });
            row.addEventListener('dragleave', () => row.classList.remove('batch-drag-over'));
            row.addEventListener('drop', e => {
                e.preventDefault();
                row.classList.remove('batch-drag-over');
                const destIndex = parseInt(row.dataset.index, 10);
                if (_dragSrcIndex == null || _dragSrcIndex === destIndex) return;

                // Reorder batch.items
                const [moved] = batch.items.splice(_dragSrcIndex, 1);
                batch.items.splice(destIndex, 0, moved);

                // Fix currentIndex if it pointed to a moved item
                if (batch.currentIndex === _dragSrcIndex) {
                    batch.currentIndex = destIndex;
                } else if (_dragSrcIndex < destIndex) {
                    if (batch.currentIndex > _dragSrcIndex && batch.currentIndex <= destIndex) batch.currentIndex--;
                } else {
                    if (batch.currentIndex >= destIndex && batch.currentIndex < _dragSrcIndex) batch.currentIndex++;
                }

                _dragSrcIndex = null;
                renderQueue();
            });
        }

        list.appendChild(row);
    });

    // Show export row if any item is done
    const anyDone = batch.items.some(i => i.status === 'done');
    $('batch-export-row').classList.toggle('hidden', !anyDone);
    updateNavButtons();
}

function _setStatusEl(el, status, lineCount) {
    el.className = 'batch-status';
    if (status === 'pending')    { el.textContent = 'pending'; }
    else if (status === 'active'){ el.textContent = 'running…'; el.classList.add('active'); }
    else if (status === 'done')  { el.textContent = `✓ ${lineCount} lines`; el.classList.add('done'); }
    else if (status === 'error') { el.textContent = 'error'; el.classList.add('error'); }
}

function updateItemStatus(index, status, lineCount = 0) {
    batch.items[index].status = status;
    const el = $(`batch-status-${index}`);
    if (el) _setStatusEl(el, status, lineCount);
}

function updateOverallProgress(current = null, total = null) {
    const el = $('batch-overall-progress');
    if (current == null) {
        el.classList.add('hidden');
        el.textContent = '';
    } else {
        el.textContent = `${current} / ${total}`;
        el.classList.remove('hidden');
    }
}

function updateNavButtons() {
    const done = batch.items.filter(i => i.status === 'done');
    const hasBatch = done.length > 0;
    const idx = batch.currentIndex;
    // Allow navigation to done items even while batch is running
    const prevDone = hasBatch && batch.items.slice(0, idx).some(i => i.status === 'done');
    const nextDone = hasBatch && batch.items.slice(idx + 1).some(i => i.status === 'done');
    $('btn-nav-prev').disabled = !prevDone;
    $('btn-nav-next').disabled = !nextDone;
    const label = $('batch-nav-label');
    if (hasBatch && idx >= 0) {
        const pos = done.indexOf(batch.items[idx]) + 1;
        label.textContent = `${pos}/${done.length}`;
    } else {
        label.textContent = '';
    }
}

function navigate(delta) {
    const indices = batch.items
        .map((item, i) => item.status === 'done' ? i : -1)
        .filter(i => i >= 0);
    if (indices.length < 2) return;
    const cur = indices.indexOf(batch.currentIndex);
    const next = indices[cur + delta];
    if (next != null) loadBatchItem(next);
}

// ── Processing ───────────────────────────────────────────────────────────────

async function processBatch() {
    if (batch.running || !state.engineLoaded) {
        if (!state.engineLoaded) toast('Load an engine first', 'error');
        return;
    }
    batch.running = true;
    batch.cancelled = false;
    batch.userNavigated = false;  // reset: auto-advance viewer from scratch
    $('btn-process-batch').disabled = true;
    $('btn-cancel').classList.remove('hidden');

    const segMethod   = $('seg-method').value;
    const segDevice   = $('seg-device').value;
    const maxColumns  = parseInt($('seg-max-columns')?.value || '6', 10);
    const splitWidth  = parseFloat($('seg-split-width')?.value || '40') / 100;
    const usePageXml  = $('batch-use-pagexml').checked;
    const resume      = $('batch-resume').checked;
    const pending = batch.items.filter(i => resume ? i.status === 'pending' : i.status !== 'done').length;
    let doneThisRun = 0;
    updateOverallProgress(0, pending);

    for (let i = 0; i < batch.items.length; i++) {
        if (batch.cancelled) {
            // Mark remaining pending items back to pending (they stay pending)
            break;
        }

        const item = batch.items[i];
        if (item.status === 'done') {
            // Resume mode: skip done; non-resume mode: also skip done
            continue;
        }

        batch.processingIndex = i;
        updateItemStatus(i, 'active');
        updateNavButtons();

        try {
            // 1. Upload image (skip if already uploaded, e.g. PDF page pre-rendered by server)
            if (item.preUploaded && item.imageId) {
                // Already registered server-side — no upload needed
            } else {
                const fd = new FormData();
                fd.append('file', item.file);
                const upResp = await fetch('/api/image/upload', { method: 'POST', body: fd });
                if (!upResp.ok) throw new Error(`Upload failed: ${upResp.statusText}`);
                const upData = await upResp.json();
                // PDF uploaded directly: expand into sub-items and skip this placeholder
                if (upData.is_pdf) {
                    const newItems = upData.pages.map(p => ({
                        file: null, imageId: p.image_id, status: 'pending',
                        lines: [], filename: p.filename, preUploaded: true,
                    }));
                    batch.items.splice(i + 1, 0, ...newItems);
                    updateItemStatus(i, 'done', 0);
                    renderQueue();
                    continue;
                }
                item.imageId = upData.image_id;
            }

            // Upload deferred XML if one was matched earlier
            if (item.xmlFile && item.imageId) {
                try {
                    const fd = new FormData();
                    fd.append('file', item.xmlFile);
                    await fetch(`/api/image/${item.imageId}/xml`, { method: 'POST', body: fd });
                    item.xmlUploaded = true;
                } catch { /* non-fatal */ }
            }

            // Show in viewer — skip if user manually navigated to a different item
            if (!batch.userNavigated) {
                batch.currentIndex = i;
                emit('batch-item-start', { imageId: item.imageId, filename: item.filename });
            }

            // 2. Transcribe via SSE (abortable)
            batch.abortController = new AbortController();
            const lines = await transcribeSSE(
                item.imageId, segMethod, segDevice, maxColumns, splitWidth, usePageXml, batch.abortController.signal
            );
            item.lines = lines;
            updateItemStatus(i, 'done', lines.length);
            doneThisRun++;
            updateOverallProgress(doneThisRun, pending);
            // Fire sse-complete so the panel shows footer, column toggle, confidence filter, etc.
            if (batch.currentIndex === i) {
                emit('sse-complete', { lines: item.lines, total_time_s: 0, engine: '(batch)' });
            }

        } catch (err) {
            if (err.name === 'AbortError' || batch.cancelled) {
                updateItemStatus(i, 'pending');
            } else {
                updateItemStatus(i, 'error');
                toast(`${item.filename}: ${err.message}`, 'error');
            }
        }

        // Re-render to make done items clickable
        renderQueue();
    }

    batch.running = false;
    batch.processingIndex = -1;
    batch.userNavigated = false;
    batch.abortController = null;
    $('btn-process-batch').disabled = false;
    $('btn-cancel').classList.add('hidden');
    $('batch-export-row').classList.remove('hidden');
    updateOverallProgress(null);
    updateNavButtons();

    const doneCount = batch.items.filter(i => i.status === 'done').length;
    if (batch.cancelled) {
        toast(`Batch cancelled — ${doneCount} image(s) done`, 'info', 4000);
    } else {
        toast(`Batch complete: ${doneCount}/${batch.items.length} images`, 'success', 5000);
    }
    emit('batch-complete', { items: batch.items });
}

function transcribeSSE(imageId, segMethod, segDevice, maxColumns, splitWidthFraction = 0.4, usePageXml = true, signal = null) {
    return new Promise((resolve, reject) => {
        const lines = [];
        const body = JSON.stringify({
            image_id: imageId, seg_method: segMethod,
            seg_device: segDevice, max_columns: maxColumns,
            split_width_fraction: splitWidthFraction,
            use_pagexml: usePageXml,
        });

        fetch('/api/transcribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
            signal,
        }).then(resp => {
            if (!resp.ok) return reject(new Error(resp.statusText));
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';

            const pump = () => reader.read().then(({ done, value }) => {
                if (done) { resolve(lines); return; }
                buf += decoder.decode(value, { stream: true });
                const parts = buf.split('\n\n');
                buf = parts.pop();
                for (const chunk of parts) {
                    const evLine  = chunk.split('\n').find(l => l.startsWith('event:'));
                    const dataLine = chunk.split('\n').find(l => l.startsWith('data:'));
                    if (!evLine || !dataLine) continue;
                    const event = evLine.slice(7).trim();
                    const data  = JSON.parse(dataLine.slice(5).trim());
                    if (event === 'progress') {
                        lines.push(data.line);
                        // Only stream to panel when user is watching this item
                        if (batch.currentIndex === batch.processingIndex) emit('sse-progress', data);
                    } else if (event === 'segmentation') {
                        // Store bboxes/regions so loadBatchItem can restore them later
                        if (batch.items[batch.processingIndex]) {
                            batch.items[batch.processingIndex].bboxes  = data.bboxes  || [];
                            batch.items[batch.processingIndex].regions = data.regions || [];
                        }
                        if (batch.currentIndex === batch.processingIndex) emit('sse-segmentation', data);
                    } else if (event === 'complete') {
                        resolve(lines);
                    } else if (event === 'error') {
                        reject(new Error(data.message));
                    } else if (event === 'cancelled') {
                        resolve(lines);
                    }
                }
                pump();
            }).catch(reject);
            pump();
        }).catch(reject);
    });
}

// Load a completed batch item back into the viewer / results panel
function loadBatchItem(index) {
    const item = batch.items[index];
    if (item.status !== 'done') return;
    batch.currentIndex = index;
    batch.userNavigated = true;  // user left auto-advance mode
    emit('batch-item-start', { imageId: item.imageId, filename: item.filename });
    updateNavButtons();
    // Restore segmentation data so line-click highlighting works.
    // batch-item-start clears currentBboxes in the image viewer; re-populate them here.
    const bboxes  = item.bboxes  || [];
    const regions = item.regions || [];
    emit('sse-segmentation', { num_lines: item.lines.length, bboxes, regions, source: 'batch-restore' });
    // Re-populate state.lines so exports and confidence filter work
    state.lines = item.lines.map((l, i) => ({ ...l, index: i }));
    // Re-emit each line to rebuild the transcription panel
    $('transcription-lines').innerHTML = '';
    $('conf-filter-row').classList.add('hidden');
    state.lines.forEach(l => emit('sse-progress', {
        current: l.index + 1, total: state.lines.length, line: l
    }));
    emit('sse-complete', { lines: state.lines, total_time_s: 0, engine: '(batch)' });
}

// ── Export ────────────────────────────────────────────────────────────────────

function exportAllTxt() {
    const done = batch.items.filter(i => i.status === 'done');
    if (!done.length) return;
    const text = done.map(item =>
        `=== ${item.filename} ===\n` + item.lines.map(l => l.text).join('\n')
    ).join('\n\n');
    downloadFile('batch_transcription.txt', text, 'text/plain');
}

function exportAllCsv() {
    const done = batch.items.filter(i => i.status === 'done');
    if (!done.length) return;
    const header = 'File,Line,Text,Confidence\n';
    const rows = done.flatMap(item =>
        item.lines.map(l => {
            const conf = l.confidence != null ? l.confidence.toFixed(4) : '';
            return `"${item.filename.replace(/"/g,'""')}",${l.index + 1},"${l.text.replace(/"/g,'""')}",${conf}`;
        })
    );
    downloadFile('batch_transcription.csv', header + rows.join('\n'), 'text/csv');
}

async function exportAllXml() {
    const done = batch.items.filter(i => i.status === 'done' && i.imageId);
    if (!done.length) return;
    try {
        const resp = await fetch('/api/batch/export-xml', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_ids: done.map(i => i.imageId) }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'batch_export.zip'; a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        toast(`XML export failed: ${err.message}`, 'error');
    }
}

function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
}
