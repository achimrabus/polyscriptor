/**
 * Engine Panel — engine selection, dynamic config form, model loading
 */

import { state, emit, on, api, saveEngineConfig, loadSavedEngineName, loadSavedEngineConfig, toast } from '../app.js';

const $ = id => document.getElementById(id);

// --- API Key localStorage helpers (keys never stored on server) ---
const _KEY_PREFIX = 'polyscriptor_key_';

function _loadBrowserKey(slot) {
    try { return localStorage.getItem(_KEY_PREFIX + slot) || ''; }
    catch (_) { return ''; }
}

function _saveBrowserKey(slot, key) {
    try {
        if (key) localStorage.setItem(_KEY_PREFIX + slot, key);
        else localStorage.removeItem(_KEY_PREFIX + slot);
    } catch (_) { /* private browsing etc. */ }
}

function _hasBrowserKey(slot) {
    return !!_loadBrowserKey(slot);
}

export function initEnginePanel() {
    loadEngines();

    $('engine-select').addEventListener('change', onEngineSelected);
    $('btn-load-model').addEventListener('click', onLoadModel);
    $('btn-transcribe').addEventListener('click', onTranscribe);
    $('btn-segment').addEventListener('click', onSegment);

    // Show/hide blla-specific options
    const segMethodSel = $('seg-method');
    const bllaopts = $('blla-options');
    const syncBllaOpts = () => {
        if (bllaopts) bllaopts.style.display = segMethodSel.value === 'kraken-blla' ? '' : 'none';
    };
    segMethodSel.addEventListener('change', syncBllaOpts);
    syncBllaOpts();

    // Cancel button — visible during transcription
    $('btn-cancel').addEventListener('click', async () => {
        try {
            await fetch('/api/transcribe/cancel', { method: 'POST' });
        } catch (_) { /* ignore */ }
    });

    // Enable transcribe/segment buttons when image is ready
    on('engine-loaded',     () => { updateTranscribeBtn(); updateSegmentBtn(); });
    on('image-uploaded',    () => { updateTranscribeBtn(); updateSegmentBtn(); });
    on('batch-item-start',  () => { updateTranscribeBtn(); updateSegmentBtn(); });
    on('transcription-complete', () => {
        state.isProcessing = false;
        $('btn-transcribe').classList.remove('loading');
        $('btn-transcribe').textContent = 'Transcribe';
        $('btn-cancel').classList.add('hidden');
        updateTranscribeBtn();
        updateSegmentBtn();
    });

    // Region list — appears after segmentation, cleared on new image/transcription
    on('sse-segmentation', data => renderRegionList(data.regions || []));
    on('image-uploaded',   () => { $('seg-regions-list').classList.add('hidden'); $('seg-regions-list').innerHTML = ''; });
}

async function loadEngines() {
    try {
        const resp = await api('/api/engines');
        state.engines = await resp.json();

        const select = $('engine-select');
        select.innerHTML = '';

        const available = state.engines.filter(e => e.available);
        const unavailable = state.engines.filter(e => !e.available);

        if (available.length === 0) {
            select.innerHTML = '<option>No engines available</option>';
            return;
        }

        const savedEngine = loadSavedEngineName();

        for (const eng of available) {
            const opt = document.createElement('option');
            opt.value = eng.name;
            opt.textContent = eng.name;
            select.appendChild(opt);
        }

        if (unavailable.length > 0) {
            const group = document.createElement('optgroup');
            group.label = 'Unavailable';
            for (const eng of unavailable) {
                const opt = document.createElement('option');
                opt.value = eng.name;
                opt.textContent = `${eng.name} (${eng.unavailable_reason || 'missing deps'})`;
                opt.disabled = true;
                group.appendChild(opt);
            }
            select.appendChild(group);
        }

        // Restore last used engine if available
        if (savedEngine && available.find(e => e.name === savedEngine)) {
            select.value = savedEngine;
        }
        select.disabled = false;
        onEngineSelected();
    } catch (err) {
        $('engine-description').textContent = `Error loading engines: ${err.message}`;
    }
}

async function onEngineSelected() {
    const name = $('engine-select').value;
    const eng = state.engines.find(e => e.name === name);
    state.currentEngine = eng;

    // Description
    $('engine-description').textContent = eng?.description || '';

    // Show/hide segmentation controls based on engine capability
    updateSegmentationVisibility(eng);

    // Load config schema
    const configForm = $('config-form');
    configForm.innerHTML = '';

    if (!eng) return;

    try {
        const resp = await api(`/api/engine/${encodeURIComponent(name)}/config-schema`);
        const schema = await resp.json();

        for (const field of schema.fields || []) {
            configForm.appendChild(createField(field));
        }

        // Restore saved config values for this engine (skip password fields for security)
        const savedCfg = loadSavedEngineConfig(name);
        if (savedCfg) {
            for (const el of configForm.querySelectorAll('[data-key]')) {
                if (el.dataset.passwordField) continue;  // never prefill secrets
                const val = savedCfg[el.dataset.key];
                if (val == null) continue;
                if (el.type === 'checkbox') el.checked = !!val;
                else el.value = val;
            }
        }

        $('btn-load-model').disabled = false;

        // For Commercial APIs: when provider changes, swap model list and update key hint
        const providerSel = $('cfg-provider');
        const modelSel    = $('cfg-model');
        if (providerSel && modelSel) {
            const syncModelList = async () => {
                // Clear model list and auto-fetch from live API if a key is available
                _populateSelect(modelSel, []);  // show "— click ↻ to load —"
                modelSel.dispatchEvent(new Event('change'));

                // Auto-trigger fetch if we have a browser key for this provider
                const prov = providerSel.value.toLowerCase();
                const keyEl = $('cfg-api_key');
                const hasBrowser = _hasBrowserKey(prov);
                const hasTyped   = keyEl?.value?.trim().length > 0;
                if (hasBrowser || hasTyped) {
                    const refreshBtn = modelSel.closest('.config-field')?.querySelector('.btn-refresh');
                    if (refreshBtn) refreshBtn.click();
                }
            };
            providerSel.addEventListener('change', syncModelList);
            syncModelList();  // run once on load to match default provider
        }

        const keyInput = $('cfg-api_key');
        if (providerSel && keyInput) {
            const updateKeyHint = () => {
                const slot = providerSel.value.toLowerCase();
                const hasBrowser = _hasBrowserKey(slot);
                const saveRow = keyInput.closest('.config-field')?.querySelector('.key-save-row');
                const saveBox = saveRow?.querySelector('input[type="checkbox"]');
                if (hasBrowser) {
                    keyInput.placeholder = '••••••••  (saved in browser — leave blank to keep)';
                    keyInput.dataset.hasBrowser = 'true';
                    keyInput.disabled = false;
                    if (saveRow) { saveRow.style.display = ''; saveRow.querySelector('label').textContent = 'Key saved in browser'; }
                    if (saveBox) saveBox.checked = true;
                } else {
                    keyInput.placeholder = 'Paste API key here';
                    keyInput.disabled = false;
                    delete keyInput.dataset.hasBrowser;
                    if (saveRow) { saveRow.style.display = ''; saveRow.querySelector('label').textContent = 'Save key in browser'; }
                    if (saveBox) saveBox.checked = false;
                }
            };
            providerSel.addEventListener('change', updateKeyHint);
            updateKeyHint();  // run once on load
        }

        // Kraken: show preset dropdown and load preset list
        const krakenPresetRow = $('kraken-preset-row');
        if (krakenPresetRow) {
            if (name === 'Kraken') {
                krakenPresetRow.classList.remove('hidden');
                _loadKrakenPresets();
            } else {
                krakenPresetRow.classList.add('hidden');
            }
        }

        // Auto-load model if this engine was previously configured.
        // Skip engines with dynamic model lists (need live fetch first — user loads manually).
        const hasDynamic = schema.fields?.some(f => f.dynamic);
        if (savedCfg && !hasDynamic) {
            onLoadModel();
        }
    } catch (err) {
        configForm.innerHTML = `<p class="muted">Error: ${err.message}</p>`;
    }
}

let _krakenPresetsLoaded = false;
async function _loadKrakenPresets() {
    if (_krakenPresetsLoaded) return;
    const sel = $('kraken-preset-select');
    const status = $('kraken-preset-status');
    if (!sel) return;
    try {
        const resp = await fetch('/api/kraken/presets');
        const data = await resp.json();
        sel.innerHTML = '';
        const blank = document.createElement('option');
        blank.value = '';
        blank.textContent = '— use model path above —';
        sel.appendChild(blank);
        for (const p of data.presets || []) {
            const opt = document.createElement('option');
            opt.value = p.id;
            const icon = p.source === 'local' ? '📁' : '⬇️';
            opt.textContent = `${icon} ${p.label} (${p.language})`;
            sel.appendChild(opt);
        }
        _krakenPresetsLoaded = true;
    } catch (e) {
        if (status) status.textContent = 'Could not load presets';
    }
    sel.addEventListener('change', () => {
        const status = $('kraken-preset-status');
        const modelPathEl = $('cfg-model_path');
        const val = sel.value;
        if (!val) {
            if (status) status.textContent = '';
            return;
        }
        if (status) {
            status.textContent = val === 'blla-local'
                ? '📁 Local model — loads instantly'
                : '⬇️ Auto-downloads from Zenodo on first use (~30–120s)';
        }
        // Pre-fill model_path field with the preset ID so server knows what to load
        if (modelPathEl) modelPathEl.value = '';  // clear — preset_id takes priority
    });
}

/**
 * Show or hide segmentation controls depending on whether the selected engine
 * requires line segmentation. Page-level engines (VLMs, Commercial APIs, etc.)
 * do their own segmentation internally — showing these controls is misleading.
 */
function updateSegmentationVisibility(eng) {
    const needsSeg = eng ? eng.requires_line_segmentation : true;
    const segControls = $('seg-controls');
    if (segControls) {
        segControls.style.display = needsSeg ? '' : 'none';
    }
}

function createField(field) {
    const wrapper = document.createElement('div');

    if (field.type === 'checkbox') {
        wrapper.className = 'config-field config-field-checkbox';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = `cfg-${field.key}`;
        input.dataset.key = field.key;
        input.checked = field.default ?? false;

        const label = document.createElement('label');
        label.htmlFor = input.id;
        label.textContent = field.label;

        wrapper.appendChild(input);
        wrapper.appendChild(label);
    } else {
        wrapper.className = 'config-field';
        const label = document.createElement('label');
        label.htmlFor = `cfg-${field.key}`;
        label.textContent = field.label;
        wrapper.appendChild(label);

        if (field.type === 'select') {
            // Row: select + optional refresh button
            const selectRow = document.createElement('div');
            selectRow.className = 'select-row';

            const select = document.createElement('select');
            select.id = `cfg-${field.key}`;
            select.dataset.key = field.key;
            if (field.per_provider_options) {
                // Store for later use when provider changes
                select.dataset.perProviderOptions = JSON.stringify(field.per_provider_options);
            }
            _populateSelect(select, field.options || [], field.default);
            selectRow.appendChild(select);

            // Dynamic refresh button — fetches live model list from server
            if (field.dynamic) {
                const hint = document.createElement('span');
                hint.className = 'dynamic-hint muted';
                hint.textContent = field.dynamic_hint || 'Click ↻ to load models';

                const refreshBtn = document.createElement('button');
                refreshBtn.type = 'button';
                refreshBtn.className = 'btn-refresh';
                refreshBtn.title = 'Refresh model list from server';
                refreshBtn.textContent = '↻';
                refreshBtn.addEventListener('click', async () => {
                    const engineName = $('engine-select').value;
                    const providerEl = $('cfg-provider');
                    const keyEl = $('cfg-api_key');
                    const provider = providerEl?.value?.toLowerCase() || 'openai';
                    const apiKey = keyEl?.value?.trim() || _loadBrowserKey(provider);

                    refreshBtn.textContent = '…';
                    refreshBtn.disabled = true;
                    try {
                        const baseUrlEl = $('cfg-base_url');
                    const baseUrl = baseUrlEl?.value?.trim() || '';
                    const params = new URLSearchParams({ provider, api_key: apiKey, base_url: baseUrl });
                        const resp = await fetch(
                            `/api/engine/${encodeURIComponent(engineName)}/models?${params}`
                        );
                        const data = await resp.json();
                        if (data.error) {
                            hint.textContent = `Error: ${data.error}`;
                        } else if (data.models.length === 0) {
                            hint.textContent = 'No models found';
                        } else {
                            const current = select.value;
                            // Build options, keep __custom__ at the end if present
                            const newOpts = data.models.map(m => ({ label: m, value: m }));
                            if (field.custom_key) newOpts.push({ label: 'Custom model ID…', value: '__custom__' });
                            _populateSelect(select, newOpts, current);
                            hint.textContent = `${data.models.length} models loaded`;
                        }
                    } catch (e) {
                        hint.textContent = `Error: ${e.message}`;
                    } finally {
                        refreshBtn.textContent = '↻';
                        refreshBtn.disabled = false;
                    }
                });
                selectRow.appendChild(refreshBtn);
                wrapper.appendChild(selectRow);
                wrapper.appendChild(hint);
            } else {
                wrapper.appendChild(selectRow);
            }

            // If this select can have a __custom__ sentinel, wire up a
            // hidden text input that appears when "__custom__" is chosen.
            if (field.custom_key) {
                const customInput = document.createElement('input');
                customInput.type = 'text';
                customInput.id = `cfg-${field.custom_key}`;
                customInput.dataset.key = field.custom_key;
                customInput.placeholder = field.custom_placeholder || 'Enter custom value';
                customInput.style.marginTop = '4px';

                // Show/hide based on current select value
                const syncCustomVisibility = () => {
                    const isCustom = select.value === '__custom__';
                    customInput.style.display = isCustom ? '' : 'none';
                    customInput.required = isCustom;
                };
                select.addEventListener('change', syncCustomVisibility);
                syncCustomVisibility();  // run once on creation

                wrapper.appendChild(customInput);
            }

            // Upload button — lets users upload a local .mlmodel file from their machine
            if (field.upload) {
                const uploadRow = document.createElement('div');
                uploadRow.className = 'upload-model-row';
                uploadRow.style.cssText = 'display:flex;align-items:center;gap:6px;margin-top:6px;';

                const fileInput = document.createElement('input');
                fileInput.type = 'file';
                fileInput.accept = '.mlmodel';
                fileInput.style.display = 'none';

                const uploadBtn = document.createElement('button');
                uploadBtn.type = 'button';
                uploadBtn.className = 'btn-secondary btn-sm';
                uploadBtn.textContent = 'Upload .mlmodel…';
                uploadBtn.title = 'Upload a Kraken model file from your computer';

                const uploadStatus = document.createElement('span');
                uploadStatus.className = 'muted';
                uploadStatus.style.fontSize = '0.85em';

                uploadBtn.addEventListener('click', () => fileInput.click());

                fileInput.addEventListener('change', async () => {
                    const f = fileInput.files[0];
                    if (!f) return;
                    uploadStatus.textContent = `Uploading ${f.name}…`;
                    uploadBtn.disabled = true;
                    try {
                        const fd = new FormData();
                        fd.append('file', f);
                        const resp = await fetch('/api/models/upload', { method: 'POST', body: fd });
                        if (!resp.ok) {
                            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                            throw new Error(err.detail || resp.statusText);
                        }
                        const data = await resp.json();
                        // Repopulate select with fresh options returned by server
                        const newPath = data.path;
                        _populateSelect(select, data.options, newPath);
                        uploadStatus.textContent = `Uploaded: ${data.filename}`;
                        // Re-run custom visibility sync (new value might not be __custom__)
                        if (field.custom_key) {
                            const isCustom = select.value === '__custom__';
                            const ci = document.getElementById(`cfg-${field.custom_key}`);
                            if (ci) { ci.style.display = isCustom ? '' : 'none'; ci.required = isCustom; }
                        }
                    } catch (e) {
                        uploadStatus.textContent = `Upload failed: ${e.message}`;
                    } finally {
                        uploadBtn.disabled = false;
                        fileInput.value = '';
                    }
                });

                uploadRow.appendChild(fileInput);
                uploadRow.appendChild(uploadBtn);
                uploadRow.appendChild(uploadStatus);
                wrapper.appendChild(uploadRow);
            }
        } else if (field.type === 'number') {
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `cfg-${field.key}`;
            input.dataset.key = field.key;
            if (field.min != null) input.min = field.min;
            if (field.max != null) input.max = field.max;
            input.value = field.default ?? '';
            wrapper.appendChild(input);
        } else if (field.type === 'password') {
            const input = document.createElement('input');
            input.type = 'password';
            input.id = `cfg-${field.key}`;
            input.dataset.key = field.key;
            input.dataset.passwordField = 'true';

            // Determine effective key slot for localStorage lookup
            function _getKeySlot() {
                const providerEl = $('cfg-provider');
                if (providerEl) return providerEl.value.toLowerCase();
                const engineEl = $('engine-select');
                if (engineEl?.value === 'OpenWebUI') return 'openwebui';
                return field.key;
            }

            function applyKeyHint() {
                const slot = _getKeySlot();
                const hasBrowser = _hasBrowserKey(slot);
                if (hasBrowser) {
                    input.placeholder = '••••••••  (saved in browser — leave blank to keep)';
                    input.dataset.hasBrowser = 'true';
                } else {
                    input.placeholder = field.placeholder || 'Paste API key here';
                    delete input.dataset.hasBrowser;
                }
                input.disabled = false;
            }
            applyKeyHint();
            wrapper.appendChild(input);

            // "Save key in browser" checkbox
            const saveRow = document.createElement('div');
            saveRow.className = 'key-save-row';
            const saveBox = document.createElement('input');
            saveBox.type = 'checkbox';
            saveBox.id = `cfg-${field.key}-save`;
            saveBox.dataset.saveFor = field.key;
            const slot = _getKeySlot();
            saveBox.checked = _hasBrowserKey(slot);
            const saveLabel = document.createElement('label');
            saveLabel.htmlFor = saveBox.id;
            saveLabel.textContent = _hasBrowserKey(slot)
                ? 'Key saved in browser' : 'Save key in browser';
            saveRow.appendChild(saveBox);
            saveRow.appendChild(saveLabel);
            wrapper.appendChild(saveRow);
        } else if (field.type === 'textarea') {
            const ta = document.createElement('textarea');
            ta.id = `cfg-${field.key}`;
            ta.dataset.key = field.key;
            ta.rows = field.rows || 3;
            ta.value = field.default ?? '';
            if (field.placeholder) ta.placeholder = field.placeholder;
            ta.style.width = '100%';
            ta.style.resize = 'vertical';
            wrapper.appendChild(ta);
            if (field.hint) {
                const hint = document.createElement('small');
                hint.textContent = field.hint;
                hint.style.color = 'var(--text-muted, #888)';
                wrapper.appendChild(hint);
            }
        } else {
            // text
            const input = document.createElement('input');
            input.type = 'text';
            input.id = `cfg-${field.key}`;
            input.dataset.key = field.key;
            input.value = field.default ?? '';
            if (field.placeholder) input.placeholder = field.placeholder;
            wrapper.appendChild(input);
        }
    }
    return wrapper;
}

function collectConfig() {
    const config = {};
    const fields = $('config-form').querySelectorAll('[data-key]');
    for (const el of fields) {
        const key = el.dataset.key;
        if (el.dataset.saveFor) continue;  // "save key" checkboxes are not config
        if (el.type === 'checkbox') {
            config[key] = el.checked;
        } else if (el.type === 'number') {
            config[key] = Number(el.value);
        } else if (el.dataset.passwordField && !el.value.trim()) {
            // Blank password field — inject key from browser localStorage
            const providerEl = $('cfg-provider');
            let slot = key;
            if (providerEl) slot = providerEl.value.toLowerCase();
            else if ($('engine-select')?.value === 'OpenWebUI') slot = 'openwebui';
            const browserKey = _loadBrowserKey(slot);
            config[key] = browserKey;  // may be empty — server will check env next
        } else {
            config[key] = el.value;
        }
    }
    return config;
}

function _persistNewKeys(engineName) {
    // For engines with password fields: if user typed a new key AND checked "save",
    // persist it to browser localStorage (never to server).
    const saveBoxes = $('config-form').querySelectorAll('[data-save-for]');
    for (const box of saveBoxes) {
        const keyField = $(`cfg-${box.dataset.saveFor}`);
        const newKey = keyField?.value?.trim();

        // Determine slot from engine name
        const slotMap = {
            'OpenWebUI': 'openwebui',
            'Commercial APIs': null,  // slot depends on selected provider
        };
        let slot = slotMap[engineName];
        if (engineName === 'Commercial APIs') {
            const providerEl = $('cfg-provider');
            slot = providerEl?.value?.toLowerCase() || 'openai';
        }
        if (!slot) continue;

        if (box.checked && newKey) {
            // Save to browser localStorage
            _saveBrowserKey(slot, newKey);
            keyField.placeholder = '••••••••  (saved in browser — leave blank to keep)';
            keyField.dataset.hasBrowser = 'true';
            const label = box.nextElementSibling;
            if (label) label.textContent = 'Key saved in browser';
        } else if (!box.checked) {
            // Unchecked = delete from browser
            _saveBrowserKey(slot, '');
            delete keyField?.dataset?.hasBrowser;
        }
    }
}

async function onLoadModel() {
    const name = $('engine-select').value;
    _persistNewKeys(name);   // save any new keys to browser localStorage
    const config = collectConfig();
    // Attach Kraken preset ID if one is selected
    if (name === 'Kraken') {
        const presetSel = $('kraken-preset-select');
        if (presetSel?.value) config.preset_id = presetSel.value;
    }
    const btn = $('btn-load-model');
    const status = $('engine-status');

    btn.classList.add('loading');
    btn.textContent = 'Loading...';
    status.className = 'status-badge status-loading';
    status.textContent = `Loading ${name}...`;
    status.classList.remove('hidden');

    try {
        const resp = await api('/api/engine/load', {
            method: 'POST',
            body: JSON.stringify({ engine_name: name, config }),
        });
        const data = await resp.json();

        state.engineLoaded = true;
        status.className = 'status-badge status-loaded';
        status.textContent = `${name} loaded (${data.load_time_s}s)`;

        // Persist engine + config for next session
        saveEngineConfig(name, collectConfig());

        emit('engine-loaded', data);
    } catch (err) {
        status.className = 'status-badge';
        status.style.color = 'var(--danger)';
        status.textContent = `Error: ${err.message}`;
        state.engineLoaded = false;
    } finally {
        btn.classList.remove('loading');
        btn.textContent = 'Load Model';
    }
}

async function onTranscribe() {
    if (state.isProcessing) return;
    if (!state.engineLoaded || !state.imageId) return;

    state.isProcessing = true;
    const btn = $('btn-transcribe');
    btn.classList.add('loading');
    btn.textContent = 'Transcribing...';
    btn.disabled = true;
    $('btn-cancel').classList.remove('hidden');

    const segMethod = $('seg-method').value;
    const segDevice = $('seg-device').value;
    const maxColumns = parseInt($('seg-max-columns')?.value || '6', 10);
    const splitWidth = parseFloat($('seg-split-width')?.value || '40') / 100;

    emit('transcription-start');

    try {
        // Collect live config overrides — non-password form fields are sent at
        // transcription time so changes (e.g. custom_prompt, thinking_mode) take
        // effect immediately without requiring a model reload.
        const liveOverrides = {};
        for (const el of $('config-form').querySelectorAll('[data-key]')) {
            if (el.dataset.saveFor) continue;       // skip "save key" checkboxes
            if (el.dataset.passwordField) continue; // never resend secrets
            const key = el.dataset.key;
            if (el.type === 'checkbox')      liveOverrides[key] = el.checked;
            else if (el.type === 'number')   liveOverrides[key] = Number(el.value);
            else                             liveOverrides[key] = el.value;
        }

        const resp = await fetch('/api/transcribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: state.imageId,
                seg_method: segMethod,
                seg_device: segDevice,
                max_columns: maxColumns,
                split_width_fraction: splitWidth,
                engine_config_overrides: liveOverrides,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Transcription failed');
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const parts = buffer.split('\n\n');
            buffer = parts.pop(); // keep incomplete

            for (const part of parts) {
                if (!part.trim()) continue;
                const eventMatch = part.match(/event: (\w+)/);
                const dataMatch = part.match(/data: (.+)/s);
                if (eventMatch && dataMatch) {
                    const eventName = eventMatch[1];
                    const data = JSON.parse(dataMatch[1]);
                    emit(`sse-${eventName}`, data);
                }
            }
        }
    } catch (err) {
        emit('transcription-error', { message: err.message });
    }
}

function updateTranscribeBtn() {
    $('btn-transcribe').disabled = !(state.engineLoaded && state.imageId && !state.isProcessing);
}

function updateSegmentBtn() {
    $('btn-segment').disabled = !(state.imageId && !state.isProcessing);
}

async function onSegment() {
    if (!state.imageId || state.isProcessing) return;

    const btn = $('btn-segment');
    const segMethod   = $('seg-method').value;
    const segDevice   = $('seg-device').value;
    const maxColumns  = parseInt($('seg-max-columns')?.value || '6', 10);
    const splitWidth  = parseFloat($('seg-split-width')?.value || '40') / 100;

    btn.classList.add('loading');
    btn.textContent = 'Segmenting…';
    btn.disabled = true;

    try {
        const params = new URLSearchParams({
            method: segMethod, device: segDevice,
            max_columns: maxColumns, split_width_fraction: splitWidth,
        });
        const resp = await api(`/api/image/${state.imageId}/segment?${params}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
        }
        const data = await resp.json();
        // Reuse the same event the transcription flow uses — draws bboxes on canvas
        emit('sse-segmentation', data);
        if (data.source !== 'page') {
            toast(`${data.num_lines} lines found (${data.source})`, 'success', 3000);
        }
        emit('segment-preview');  // switch mobile tab to image view
    } catch (err) {
        toast(`Segmentation failed: ${err.message}`, 'error');
    } finally {
        btn.classList.remove('loading');
        btn.textContent = 'Segment';
        updateSegmentBtn();
    }
}

/**
 * Populate a <select> with an array of options.
 * Each option may be a string or {label, value}.
 * Tries to restore previousValue after repopulating.
 */
function _populateSelect(select, options, previousValue) {
    select.innerHTML = '';
    if (options.length === 0) {
        const o = document.createElement('option');
        o.value = '';
        o.textContent = '— click ↻ to load —';
        select.appendChild(o);
        return;
    }
    for (const opt of options) {
        const o = document.createElement('option');
        o.value = typeof opt === 'object' ? opt.value : opt;
        o.textContent = typeof opt === 'object' ? opt.label : opt;
        select.appendChild(o);
    }
    if (previousValue != null) {
        // Restore previous selection if it still exists
        const match = Array.from(select.options).find(o => o.value === previousValue);
        if (match) select.value = previousValue;
    }
}

// Same palette as image-viewer.js REGION_COLORS
const _REGION_COLORS = [
    'rgba(255,160,30,0.9)', 'rgba(46,213,115,0.9)', 'rgba(232,65,24,0.9)',
    'rgba(52,172,224,0.9)', 'rgba(162,16,213,0.9)', 'rgba(255,211,42,0.9)',
    'rgba(18,203,196,0.9)', 'rgba(253,89,166,0.9)',
];

function renderRegionList(regions) {
    const list = $('seg-regions-list');
    list.innerHTML = '';
    if (!regions.length) { list.classList.add('hidden'); return; }
    list.classList.remove('hidden');

    const hdr = document.createElement('div');
    hdr.className = 'seg-regions-header';
    hdr.textContent = `Regions (${regions.length})`;
    list.appendChild(hdr);

    regions.forEach((r, i) => {
        const row = document.createElement('div');
        row.className = 'seg-region-row';

        const dot = document.createElement('span');
        dot.className = 'seg-region-dot';
        dot.style.background = _REGION_COLORS[i % _REGION_COLORS.length];

        const label = document.createElement('span');
        label.className = 'seg-region-label';
        label.textContent = `R${i + 1}`;

        const count = document.createElement('span');
        count.className = 'seg-region-count';
        count.textContent = `${r.num_lines} line${r.num_lines !== 1 ? 's' : ''}`;

        const delBtn = document.createElement('button');
        delBtn.className = 'seg-region-del btn-icon';
        delBtn.textContent = '×';
        delBtn.title = 'Delete this region';
        delBtn.addEventListener('click', async () => {
            delBtn.disabled = true;
            try {
                const resp = await api(`/api/image/${state.imageId}/region/${i}`, { method: 'DELETE' });
                const data = await resp.json();
                emit('sse-segmentation', data);
                toast(`Region R${i + 1} removed`, 'info', 2000);
            } catch (err) {
                toast(`Delete failed: ${err.message}`, 'error');
                delBtn.disabled = false;
            }
        });

        row.appendChild(dot);
        row.appendChild(label);
        row.appendChild(count);
        row.appendChild(delBtn);
        list.appendChild(row);
    });
}
