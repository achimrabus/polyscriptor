<script lang="ts">
	import type { SegResult } from '$lib/api';

	let {
		src,
		seg,
		segments,
		currentSegment,
		busy,
		ondelete,
		onresegment
	}: {
		src: string | null;
		seg: SegResult | null;
		segments: number[][] | null;
		currentSegment: number;
		busy: boolean;
		ondelete: (regionIndex: number) => void;
		onresegment: (rects: number[][]) => Promise<void>;
	} = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);
	let wrap = $state<HTMLDivElement | null>(null);
	let img: HTMLImageElement | null = null;

	let view = { scale: 1, ox: 0, oy: 0 };
	let drawMode = $state(false);
	let rects = $state<number[][]>([]);
	let drag: { x0: number; y0: number; cur?: { x: number; y: number } } | null = null;
	let pan: { x0: number; y0: number; ox: number; oy: number } | null = null;

	$effect(() => {
		if (!src) return;
		img = new Image();
		img.onload = () => {
			fit();
			redraw();
		};
		img.src = src;
	});
	$effect(() => {
		// Touch every reactive input redraw() consumes *before* calling it: redraw
		// returns early while the image is still loading, and an early return skips
		// the prop reads, so Svelte would never track seg/segments changes.
		void seg;
		void segments;
		void currentSegment;
		void rects;
		redraw();
	});

	function fit() {
		if (!img || !wrap || !canvas) return;
		const { clientWidth: w, clientHeight: h } = wrap;
		canvas.width = w;
		canvas.height = h;
		view.scale = Math.min(w / img.width, h / img.height) * 0.98;
		view.ox = (w - img.width * view.scale) / 2;
		view.oy = (h - img.height * view.scale) / 2;
	}

	function toImage(x: number, y: number): [number, number] {
		return [(x - view.ox) / view.scale, (y - view.oy) / view.scale];
	}

	function redraw() {
		if (!canvas || !img) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const { width: w, height: h } = canvas;
		ctx.clearRect(0, 0, w, h);
		ctx.imageSmoothingEnabled = view.scale < 4;
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.drawImage(img, view.ox, view.oy, img.width * view.scale, img.height * view.scale);

		const scale = view.scale;
		const T = (x: number, y: number): [number, number] => [x * scale + view.ox, y * scale + view.oy];

		if (seg?.regions) {
			for (const r of seg.regions) {
				const [x1, y1] = T(r.bbox[0], r.bbox[1]);
				const [x2, y2] = T(r.bbox[2], r.bbox[3]);
				ctx.strokeStyle = 'rgba(16,185,129,.9)';
				ctx.lineWidth = 1.5;
				ctx.setLineDash([]);
				ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
			}
		}
		if (segments) {
			for (let i = 0; i < segments.length; i++) {
				const b = segments[i];
				const [x1, y1] = T(b[0], b[1]);
				const [x2, y2] = T(b[2], b[3]);
				ctx.strokeStyle =
					i === currentSegment ? 'rgba(239,68,68,1)' : 'rgba(59,130,246,.85)';
				ctx.lineWidth = i === currentSegment ? 2.5 : 1;
				ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
			}
		}
		ctx.strokeStyle = 'rgba(245,158,11,.95)';
		ctx.setLineDash([6, 4]);
		ctx.lineWidth = 1.5;
		for (const r of rects) {
			const [x1, y1] = T(r[0], r[1]);
			const [x2, y2] = T(r[2], r[3]);
			ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
		}
		if (drag && drag.cur) {
			ctx.beginPath();
			const [x1, y1] = T(...toImage(drag.x0, drag.y0));
			const [x2, y2] = T(...toImage(drag.cur.x, drag.cur.y));
			ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
			ctx.stroke();
		}
		ctx.setLineDash([]);
	}

	function canvasPos(e: MouseEvent): [number, number] {
		const r = canvas!.getBoundingClientRect();
		return [e.clientX - r.left, e.clientY - r.top];
	}

	function onwheel(e: WheelEvent) {
		e.preventDefault();
		const [mx, my] = canvasPos(e);
		const [ix, iy] = toImage(mx, my);
		const f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
		view.scale *= f;
		view.ox = mx - ix * view.scale;
		view.oy = my - iy * view.scale;
		redraw();
	}

	function ondown(e: MouseEvent) {
		const [x, y] = canvasPos(e);
		if (drawMode) {
			drag = { x0: x, y0: y };
		} else {
			pan = { x0: e.clientX, y0: e.clientY, ox: view.ox, oy: view.oy };
		}
	}

	function onmove(e: MouseEvent) {
		if (drag) {
			const [x, y] = canvasPos(e);
			drag.cur = { x, y };
			redraw();
		} else if (pan) {
			view.ox = pan.ox + (e.clientX - pan.x0);
			view.oy = pan.oy + (e.clientY - pan.y0);
			redraw();
		}
	}

	function onup(e: MouseEvent) {
		if (drag) {
			const [x, y] = canvasPos(e);
			const [ix1, iy1] = toImage(drag.x0, drag.y0);
			const [ix2, iy2] = toImage(x, y);
			const r = [
				Math.round(Math.min(ix1, ix2)),
				Math.round(Math.min(iy1, iy2)),
				Math.round(Math.max(ix1, ix2)),
				Math.round(Math.max(iy1, iy2))
			];
			if (r[2] - r[0] >= 10 && r[3] - r[1] >= 10) rects = [...rects, r];
			drag = null;
			redraw();
		}
		pan = null;
	}

	async function applyResegment() {
		if (!rects.length) return;
		await onresegment(rects);
		rects = [];
		redraw();
	}

	function deleteFirstRegion() {
		if (seg?.regions?.length) ondelete(0);
	}
</script>

<svelte:window onkeyup={(e) => e.key === 'Escape' && ((rects = []), (drag = null), redraw())} />

<div class="panel flex h-full flex-col">
	<div class="panel-title flex items-center gap-2">
		<span>Page</span>
		<div class="ml-auto flex items-center gap-1.5">
			<button
				class="btn !px-2 !py-0.5 text-xs"
				class:!bg-amber-600={drawMode}
				class:!text-white={drawMode}
				onclick={() => (drawMode = !drawMode)}
				title="Draw boxes to re-run segmentation inside them"
			>
				✏ Box re-seg
			</button>
			{#if rects.length}
				<button class="btn !px-2 !py-0.5 text-xs btn-primary" onclick={applyResegment} disabled={busy}>
					Re-segment {rects.length} ✕
				</button>
				<button class="btn !px-2 !py-0.5 text-xs" onclick={() => ((rects = []), redraw())}>✕</button>
			{/if}
			{#if seg?.regions?.length}
				<select
					class="input !w-auto !py-0.5 text-xs"
					onchange={(e) => {
						const v = e.currentTarget.value;
						if (v) ondelete(Number(v));
						e.currentTarget.value = '';
					}}
				>
					<option value="">Delete region…</option>
					{#each seg.regions as r, i}
						<option value={i}>region {i} ({r.num_lines} lines)</option>
					{/each}
				</select>
			{/if}
			<button class="btn !px-2 !py-0.5 text-xs" onclick={() => (fit(), redraw())}>fit</button>
		</div>
	</div>
	<div bind:this={wrap} class="relative min-h-0 flex-1 overflow-hidden bg-zinc-200 dark:bg-zinc-950">
		<canvas
			bind:this={canvas}
			class="absolute inset-0 {drawMode ? 'cursor-crosshair' : 'cursor-grab'}"
			onwheel={onwheel}
			onmousedown={ondown}
			onmousemove={onmove}
			onmouseup={onup}
			aria-label="page image with segmentation overlay"
		></canvas>
		{#if !src}
			<div class="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">
				Upload an image or PDF to begin
			</div>
		{/if}
	</div>
	<div class="flex items-center gap-2 border-t border-zinc-200 px-3 py-1.5 text-xs text-zinc-500 dark:border-zinc-800">
		{#if seg}
			{seg.num_lines} lines · {seg.regions?.length ?? 0} regions · via {seg.source}
		{:else}
			Scroll to zoom · drag to pan{drawMode ? ' · drag to draw boxes, Esc to clear' : ''}
		{/if}
	</div>
</div>
