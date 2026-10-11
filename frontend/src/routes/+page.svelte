<script lang="ts">
	import { api, transcribeStream, type LineResult, type SegResult } from '$lib/api';
	import EnginePanel from '$lib/components/EnginePanel.svelte';
	import ImageViewer from '$lib/components/ImageViewer.svelte';
	import TranscriptionPanel from '$lib/components/TranscriptionPanel.svelte';
	import BatchPanel from '$lib/components/BatchPanel.svelte';
	import ComparePanel from '$lib/components/ComparePanel.svelte';

	interface Slot {
		id: string;
		label: string;
		engine: string;
		kind: string;
		lines: LineResult[];
	}

	let tab = $state<'workspace' | 'batch' | 'compare'>('workspace');

	let images = $state<{ image_id: string; filename: string }[]>([]);
	let currentId = $state<string | null>(null);
	let cache = new Map<string, { seg: SegResult | null; lines: LineResult[]; slots: Slot[]; active: string | null }>();

	let seg = $state<SegResult | null>(null);
	let lines = $state<LineResult[]>([]);
	let slots = $state<Slot[]>([]);
	let activeSlot = $state<string | null>(null);

	let busy = $state(false);
	let progress = $state<{ current: number; total: number } | null>(null);
	let statusMsg = $state('');
	let error = $state('');

	// segmentation options
	let segMethod = $state('kraken-blla');
	let segDevice = $state('cpu');
	let segDirection = $state('horizontal-lr');
	let segModel = $state('');

	let fileInput = $state<HTMLInputElement | null>(null);
	const current = $derived(images.find((i) => i.image_id === currentId));

	$effect(() => {
		window.__polyscriptor_current_image = currentId;
	});
	$effect(() => {
		const h = (e: KeyboardEvent) => {
			if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !busy) run();
		};
		window.addEventListener('keydown', h);
		return () => window.removeEventListener('keydown', h);
	});

	async function onFiles(files: FileList | null) {
		if (!files) return;
		for (const f of Array.from(files)) {
			try {
				const up = await api.upload(f);
				images = [...images, { image_id: up.image_id, filename: up.filename }];
				selectImage(up.image_id);
			} catch (e) {
				error = String(e);
			}
		}
	}

	function store() {
		if (!currentId) return;
		cache.set(currentId, { seg, lines, slots, active: activeSlot });
	}

	function selectImage(id: string) {
		store();
		currentId = id;
		const c = cache.get(id);
		seg = c?.seg ?? null;
		lines = c?.lines ?? [];
		slots = c?.slots ?? [];
		activeSlot = c?.active ?? null;
		error = '';
	}

	async function runSegment() {
		if (!currentId) return;
		busy = true;
		error = '';
		try {
			const params: Record<string, string> = { method: segMethod, device: segDevice, text_direction: segDirection };
			if (segModel) params['model_path'] = segModel;
			seg = await api.segment(currentId, params);
			lines = seg.bboxes.map((b, i) => ({
				index: i,
				text: '',
				confidence: null,
				error: null,
				bbox: b,
				region: 0
			}));
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
		}
	}

	async function run() {
		if (!currentId) {
			error = 'Upload an image first';
			return;
		}
		busy = true;
		error = '';
		progress = null;
		lines = [];
		slots = [];
		activeSlot = null;
		try {
			await transcribeStream(
				{
					image_id: currentId,
					seg_method: seg ? undefined : segMethod,
					seg_device: segDevice,
					text_direction: segDirection
				},
				{
					onEvent: (ev, data) => {
						if (ev === 'segmentation') {
							seg = data;
						} else if (ev === 'status') {
							statusMsg = data.message ?? '';
						} else if (ev === 'progress') {
							progress = { current: data.current ?? 0, total: data.total ?? 0 };
						} else if (ev === 'complete') {
							lines = (data.lines ?? []) as LineResult[];
							const slot: Slot = {
								id: data.result_slot?.id ?? 'primary',
								label: data.result_slot?.label ?? data.engine ?? 'primary',
								engine: data.engine ?? '',
								kind: 'primary',
								lines
							};
							slots = [slot];
							activeSlot = slot.id;
							statusMsg = `done in ${data.total_time_s}s`;
						} else if (ev === 'error') {
							error = data.message ?? 'transcription failed';
						} else if (ev === 'cancelled') {
							statusMsg = 'cancelled';
						}
					}
				}
			);
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
			progress = null;
		}
	}

	async function cancel() {
		try {
			await fetch('/api/transcribe/cancel', { method: 'POST', credentials: 'same-origin' });
		} catch {
			/* ignore */
		}
	}

	async function resegment(rects: number[][]) {
		if (!currentId) return;
		busy = true;
		error = '';
		try {
			seg = await api.resegment(currentId, {
				rects,
				device: segDevice === 'cuda:0' ? 'cuda' : segDevice,
				text_direction: segDirection,
				model_path: segModel || null
			});
			lines = seg.bboxes.map((b, i) => ({ index: i, text: '', confidence: null, error: null, bbox: b, region: 0 }));
			slots = [];
			activeSlot = null;
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
		}
	}

	async function deleteRegion(i: number) {
		if (!currentId) return;
		try {
			seg = await api.deleteRegion(currentId, i);
			lines = seg.bboxes.map((b, idx) => ({ index: idx, text: '', confidence: null, error: null, bbox: b, region: 0 }));
		} catch (e) {
			error = String(e);
		}
	}

	async function exportXml(fmt: string) {
		if (!currentId || fmt !== 'xml') return;
		try {
			const blob = await api.exportXml(currentId);
			const a = document.createElement('a');
			a.href = URL.createObjectURL(blob);
			a.download = (current?.filename ?? 'page').replace(/\.[^.]+$/, '') + '.xml';
			a.click();
			URL.revokeObjectURL(a.href);
		} catch (e) {
			error = String(e);
		}
	}

	function switchSlot(id: string) {
		activeSlot = id;
		const slot = slots.find((s) => s.id === id);
		if (slot) lines = slot.lines;
	}

	async function runCompare() {
		if (!currentId) return;
		busy = true;
		error = '';
		statusMsg = 'Running comparison engine…';
		try {
			const res = await api.compareRun(currentId);
			const slot: Slot = {
				id: res.slot?.id ?? `slot-${slots.length}`,
				label: res.slot?.label ?? 'comparison',
				engine: res.slot?.engine ?? 'comparison',
				kind: 'comparison',
				lines: (res.slot?.lines ?? []) as LineResult[]
			};
			slots = [...slots, slot];
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
			statusMsg = '';
		}
	}
</script>

<div class="flex h-full flex-col">
	<!-- toolbar -->
	<div class="flex flex-wrap items-center gap-2 border-b border-zinc-200 bg-white px-3 py-1.5 text-xs dark:border-zinc-800 dark:bg-zinc-900">
		<input bind:this={fileInput} type="file" multiple accept="image/*,application/pdf" class="hidden" onchange={(e) => onFiles(e.currentTarget.files)} />
		<button class="btn !py-1" onclick={() => fileInput?.click()}>📤 Upload</button>
		{#if images.length > 1}
			<select class="input !w-auto !py-1" value={currentId ?? ''} onchange={(e) => selectImage(e.currentTarget.value)}>
				{#each images as im}
					<option value={im.image_id}>{im.filename}</option>
				{/each}
			</select>
		{/if}
		<span class="mx-1 h-4 w-px bg-zinc-300 dark:bg-zinc-700"></span>
		<select class="input !w-auto !py-1" bind:value={segMethod} title="Segmentation method">
			<option value="kraken-blla">Kraken Neural (blla)</option>
			<option value="kraken">Kraken classical</option>
			<option value="hpp">HPP (fast)</option>
			<option value="none">pre-segmented</option>
		</select>
		<select class="input !w-auto !py-1" bind:value={segDevice} title="Device">
			<option value="cpu">cpu</option>
			<option value="cuda:0">cuda:0</option>
		</select>
		<select class="input !w-auto !py-1" bind:value={segDirection} title="Reading direction">
			<option value="horizontal-lr">→ LTR</option>
			<option value="horizontal-rl">← RTL</option>
			<option value="vertical-lr">↓ vertical</option>
		</select>
		<input class="input !w-36 !py-1" placeholder="custom .mlmodel (optional)" bind:value={segModel} />
		<button class="btn !py-1" onclick={runSegment} disabled={!currentId || busy}>Segment</button>
		<div class="ml-auto flex rounded border border-zinc-300 dark:border-zinc-700">
			{#each [['workspace', 'Workspace'], ['compare', 'Compare'], ['batch', 'Batch']] as [t, label]}
				<button
					class="px-3 py-1 {tab === t ? 'bg-sky-600 text-white' : 'hover:bg-zinc-100 dark:hover:bg-zinc-800'}"
					onclick={() => (tab = t)}>{label}</button>
			{/each}
		</div>
	</div>

	{#if error}
		<div class="border-b border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
			{error}
		</div>
	{/if}

	<div class="grid min-h-0 flex-1 grid-cols-[280px_minmax(0,1fr)_minmax(320px,420px)] gap-2 p-2 max-lg:grid-cols-1">
		<EnginePanel onloaded={() => (statusMsg = '')} />
		<ImageViewer
			src={currentId ? api.imageUrl(currentId) : null}
			{seg}
			segments={lines.map((l) => l.bbox) as number[][]}
			currentSegment={-1}
			{busy}
			ondelete={deleteRegion}
			onresegment={resegment}
		/>
		{#if tab === 'workspace'}
			<TranscriptionPanel
				{lines}
				{busy}
				{progress}
				{statusMsg}
				onrun={run}
				oncancel={cancel}
				onexport={exportXml}
				onjump={() => {}}
			/>
		{:else if tab === 'compare'}
			<ComparePanel {slots} activeSlotId={activeSlot} hasImage={!!currentId} onswitch={switchSlot} onruncompare={runCompare} />
		{:else}
			<BatchPanel />
		{/if}
	</div>
</div>
