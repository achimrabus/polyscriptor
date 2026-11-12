// jsdom run of the web UI: a second image dropped onto a transcribed one (12.11.2026).
// Expected: both go to the batch queue, the new image is shown, the transcribed one is
// 'done' and comes back with its lines; a manual transcription of a queued image is kept.
//
//   NODE_PATH=<dir with jsdom> node web/tests/test_batch_drop.mjs
//
// The static files are copied to a temp dir with {"type":"module"} so Node loads them as
// ES modules; fetch is a stub, no server is needed.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { JSDOM } = require('jsdom');

const STATIC = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'static');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'polyscriptor-static-'));
fs.cpSync(STATIC, tmp, { recursive: true });
fs.writeFileSync(path.join(tmp, 'package.json'), '{"type":"module"}');

const dom = new JSDOM(fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8'),
                      { url: 'http://localhost/', pretendToBeVisual: true });
const w = dom.window;
for (const k of ['window', 'document', 'localStorage', 'CustomEvent', 'Event', 'EventTarget', 'HTMLElement',
                 'Node', 'FormData', 'Blob', 'File', 'getComputedStyle', 'requestAnimationFrame',
                 'matchMedia', 'navigator']) {
    try { globalThis[k] = k === 'window' ? w : w[k]; } catch (e) { /* read-only in Node */ }
}
w.matchMedia = w.matchMedia || (() => ({ matches: false, addEventListener() {}, addListener() {} }));
globalThis.matchMedia = w.matchMedia;
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: () => () => {} });

const uploads = [];
globalThis.fetch = w.fetch = async (url, opts = {}) => {
    const json = body => ({ ok: true, status: 200, json: async () => body, text: async () => '' });
    if (url === '/api/image/upload') {
        const f = opts.body.get('file');
        uploads.push(f.name);
        return json({ image_id: `id-${f.name}`, filename: f.name, width: 100, height: 200 });
    }
    if (url.startsWith('/api/engines')) return json([]);
    return json({});
};

const app = await import(pathToFileURL(path.join(tmp, 'app.js')).href);
const { state, emit } = app;
w.document.dispatchEvent(new w.Event('DOMContentLoaded'));
const $ = id => w.document.getElementById(id);
const flush = () => new Promise(r => setTimeout(r, 50));

function drop(name) {
    const ev = new w.Event('drop', { bubbles: true, cancelable: true });
    Object.defineProperty(ev, 'dataTransfer', { value: { files: [new w.File(['x'], name, { type: 'image/jpeg' })] } });
    $('upload-area').dispatchEvent(ev);
}
function transcribe(texts) {   // what a finished single-image transcription leaves behind
    state.lines = texts.map((t, i) => ({ index: i, text: t, confidence: 0.9, bbox: [0, 10 * i, 50, 10 * i + 8] }));
    emit('sse-complete', { lines: state.lines, total_time_s: 1, engine: 'stub' });
}

let fehler = 0;
const check = (ok, msg) => { console.log((ok ? 'ok   ' : 'FAIL ') + msg); if (!ok) fehler++; };

drop('A.jpg'); await flush();
check(state.imageId === 'id-A.jpg', `first drop shows A (imageId ${state.imageId})`);
transcribe(['a1', 'a2']);

drop('B.jpg'); await flush();
check(uploads.includes('B.jpg'), 'B uploaded');
check(state.imageId === 'id-B.jpg', `second drop shows B (imageId ${state.imageId})`);
check($('image-info').textContent.includes('B.jpg'), `viewer label names B ("${$('image-info').textContent}")`);
check(($('batch-status-0') || {}).textContent === '✓ 2 lines', `A is done in the queue ("${($('batch-status-0') || {}).textContent}")`);

transcribe(['b1']);
drop('C.jpg'); await flush();
check(state.imageId === 'id-C.jpg', `third drop shows C (imageId ${state.imageId})`);
check(($('batch-status-1') || {}).textContent === '✓ 1 lines', `B's manual transcription kept ("${($('batch-status-1') || {}).textContent}")`);

$('batch-item-0').click(); await flush();
check(state.imageId === 'id-A.jpg' && state.lines.map(l => l.text).join() === 'a1,a2',
      `clicking A restores its lines (${state.lines.map(l => l.text).join()})`);

console.log(fehler ? `${fehler} FAILED` : 'all passed');
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(fehler ? 1 : 0);
