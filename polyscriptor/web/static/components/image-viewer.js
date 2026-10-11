/**
 * Image Viewer — upload, display, bbox overlay
 */

import { state, emit, on, api, fitZoom } from '../app.js';

const $ = id => document.getElementById(id);

export function initImageViewer() {
    const uploadArea = $('upload-area');
    const fileInput = $('file-input');
    const xmlInput = $('xml-input');

    // Click to browse image
    uploadArea.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
    });

    // Drag & drop — accept image, PDF, and XML
    uploadArea.addEventListener('dragover', e => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        const img = files.find(f =>
            f.type.startsWith('image/') || f.name.toLowerCase().endsWith('.pdf'));
        const xml = files.find(f => f.name.endsWith('.xml'));
        if (img) uploadFile(img);
        if (xml) uploadXml(xml);  // queued after image upload sets imageId
    });

    // XML file picker
    xmlInput.addEventListener('change', () => {
        if (xmlInput.files.length > 0) uploadXml(xmlInput.files[0]);
    });

    // Batch panel: load a completed item's image into the viewer
    on('batch-item-start', ({ imageId, filename }) => {
        state.imageId = imageId;
        // Clear bboxes immediately for the new item
        currentBboxes  = [];
        currentRegions = [];
        const img = $('page-image');
        img.src = `/api/image/${imageId}`;
        $('image-container').classList.remove('hidden');
        $('viewer-placeholder').classList.add('hidden');
        img.onload = () => {
            const canvas = $('overlay-canvas');
            canvas.width  = img.naturalWidth;
            canvas.height = img.naturalHeight;
            fitZoom();
            // Redraw any bboxes that arrived before the image finished loading
            if (currentBboxes.length > 0) {
                drawBboxes(currentBboxes, -1, currentRegions);
            } else {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        };
        $('image-info').textContent = filename;
        $('xml-upload-row').classList.remove('hidden');
        $('xml-status').textContent = 'No PAGE XML';
        $('xml-status').classList.remove('xml-ok');
        emit('transcription-start', {});
    });

    // Draw bboxes after segmentation; keep state.regions in sync
    on('sse-segmentation', data => {
        state.regions = data.regions || [];
        if (data.source === 'page') {
            // Page-level engine: clear any old line bboxes, don't draw full-page box
            drawBboxes([], -1, []);
        } else {
            drawBboxes(data.bboxes, -1, state.regions);
        }
        if (data.source === 'pagexml') {
            $('xml-status').textContent = `PAGE XML: ${data.num_lines} lines`;
        }
    });

    // Highlight line on click from transcription panel
    on('highlight-line', ({ index }) => highlightBbox(index));

    // Click on canvas → highlight the clicked bbox and emit highlight-line
    const canvas = $('overlay-canvas');
    canvas.addEventListener('click', e => {
        if (currentBboxes.length === 0) return;

        const img = $('page-image');
        // Scale factor: natural image coords / displayed canvas coords
        const scaleX = img.naturalWidth / img.clientWidth;
        const scaleY = img.naturalHeight / img.clientHeight;

        const rect = canvas.getBoundingClientRect();
        const clickX = (e.clientX - rect.left) * scaleX;
        const clickY = (e.clientY - rect.top) * scaleY;

        for (let i = 0; i < currentBboxes.length; i++) {
            const [x1, y1, x2, y2] = currentBboxes[i];
            if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
                emit('highlight-line', { index: i });
                break;
            }
        }
    });
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    $('image-info').textContent = 'Uploading...';

    try {
        const resp = await fetch('/api/image/upload', {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail);
        }
        const data = await resp.json();

        // PDF: redirect all pages to batch panel
        if (data.is_pdf) {
            $('image-info').textContent = `PDF: ${data.num_pages} page(s) — added to batch queue`;
            emit('pdf-pages-ready', data);
            return;
        }

        state.imageId = data.image_id;
        state.imageInfo = data;

        // Display image — show container, hide placeholder
        const img = $('page-image');
        img.src = `/api/image/${data.image_id}`;
        $('image-container').classList.remove('hidden');
        $('viewer-placeholder').classList.add('hidden');

        // Wait for image to load to size canvas and fit zoom
        img.onload = () => {
            const canvas = $('overlay-canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            fitZoom();  // sets img.style.width/height and canvas display size
        };

        $('image-info').textContent = `${data.filename} (${data.width}×${data.height})`;
        // Show XML upload row
        $('xml-upload-row').classList.remove('hidden');
        $('xml-status').textContent = 'No PAGE XML';
        $('xml-status').classList.remove('xml-ok');
        emit('image-uploaded', data);
    } catch (err) {
        $('image-info').textContent = `Error: ${err.message}`;
    }
}

async function uploadXml(file) {
    if (!state.imageId) {
        // Will retry after image upload finishes
        on('image-uploaded', () => uploadXml(file), { once: true });
        return;
    }
    const xmlStatus = $('xml-status');
    xmlStatus.textContent = 'Uploading XML...';
    xmlStatus.classList.remove('xml-ok');
    try {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch(`/api/image/${state.imageId}/xml`, {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail);
        }
        xmlStatus.textContent = `✓ ${file.name}`;
        xmlStatus.classList.add('xml-ok');
        emit('xml-uploaded', { filename: file.name });
    } catch (err) {
        xmlStatus.textContent = `XML error: ${err.message}`;
    }
}

let currentBboxes = [];
let currentRegions = [];

// Distinct colours for up to 8 regions (cycling)
const REGION_COLORS = [
    'rgba(255, 160,  30, 0.55)',  // orange
    'rgba( 46, 213, 115, 0.55)',  // green
    'rgba(232,  65, 24,  0.55)',  // red
    'rgba( 52, 172, 224, 0.55)',  // blue
    'rgba(162,  16, 213, 0.55)',  // purple
    'rgba(255, 211,  42, 0.55)',  // yellow
    'rgba( 18, 203, 196, 0.55)',  // teal
    'rgba(253,  89, 166, 0.55)',  // pink
];

function drawBboxes(bboxes, highlightIndex = -1, regions = []) {
    currentBboxes = bboxes;
    currentRegions = regions;
    const canvas = $('overlay-canvas');
    const img = $('page-image');
    const ctx = canvas.getContext('2d');

    // Keep canvas display size in sync with zoom-controlled img size
    canvas.style.width  = img.style.width  || img.clientWidth  + 'px';
    canvas.style.height = img.style.height || img.clientHeight + 'px';

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw region outlines first (underneath line boxes)
    regions.forEach((r, ri) => {
        const [x1, y1, x2, y2] = r.bbox;
        const color = REGION_COLORS[ri % REGION_COLORS.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);
        // Subtle fill
        ctx.fillStyle = color.replace('0.55', '0.07');
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        // Region label
        ctx.fillStyle = color.replace('0.55', '0.9');
        ctx.font = 'bold 13px sans-serif';
        ctx.fillText(`R${ri + 1} (${r.num_lines} lines)`, x1 + 4, y1 + 16);
    });

    // Draw line boxes on top
    for (let i = 0; i < bboxes.length; i++) {
        const [x1, y1, x2, y2] = bboxes[i];
        const isHighlighted = i === highlightIndex;

        ctx.strokeStyle = isHighlighted ? '#e94560' : 'rgba(58, 134, 255, 0.6)';
        ctx.lineWidth = isHighlighted ? 3 : 1.5;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        if (isHighlighted) {
            ctx.fillStyle = 'rgba(233, 69, 96, 0.1)';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }
    }
}

function highlightBbox(index) {
    if (currentBboxes.length > 0) {
        drawBboxes(currentBboxes, index, currentRegions);
    }
}
