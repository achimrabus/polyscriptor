<script lang="ts">
	import type { LineResult } from '$lib/api';

	let {
		lines,
		busy,
		progress,
		statusMsg,
		onrun,
		oncancel,
		onexport,
		onjump
	}: {
		lines: LineResult[];
		busy: boolean;
		progress: { current: number; total: number } | null;
		statusMsg: string;
		onrun: () => void;
		oncancel: () => void;
		onexport: (fmt: 'txt' | 'csv' | 'xml') => void;
		onjump: (index: number) => void;
	} = $props();

	let minConf = $state(0);

	function confColor(c: number | null): string {
		if (c === null) return 'bg-zinc-300 dark:bg-zinc-700';
		if (c > 0.9) return 'bg-emerald-500';
		if (c > 0.7) return 'bg-lime-500';
		if (c > 0.5) return 'bg-amber-500';
		return 'bg-red-500';
	}

	const visible = $derived(lines.filter((l) => l.confidence === null || l.confidence >= minConf));
	const errors = $derived(lines.filter((l) => l.error).length);
	const avgConf = $derived.by(() => {
		const cs = lines.map((l) => l.confidence).filter((c): c is number => c !== null);
		return cs.length ? (cs.reduce((a, b) => a + b, 0) / cs.length).toFixed(3) : '—';
	});

	function download(name: string, content: string, type: string) {
		const a = document.createElement('a');
		a.href = URL.createObjectURL(new Blob([content], { type }));
		a.download = name;
		a.click();
		URL.revokeObjectURL(a.href);
	}

	function exportTxt() {
		download('transcription.txt', lines.map((l) => l.text).join('\n'), 'text/plain');
	}
	function exportCsv() {
		const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;
		const rows = ['index,text,confidence,error'].concat(
			lines.map((l) => `${l.index},${esc(l.text)},${l.confidence ?? ''},${esc(l.error ?? '')}`)
		);
		download('transcription.csv', rows.join('\n'), 'text/csv');
	}
</script>

<div class="panel flex h-full flex-col">
	<div class="panel-title flex items-center gap-2">
		Transcription
		{#if lines.length}
			<span class="ml-auto text-[10px] normal-case">
				{lines.length} lines · avg conf {avgConf}
				{#if errors}<span class="text-red-500">· {errors} errors</span>{/if}
			</span>
		{/if}
	</div>

	{#if busy}
		<div class="px-3 pt-2">
			<div class="h-1.5 w-full overflow-hidden rounded bg-zinc-200 dark:bg-zinc-800">
				<div
					class="h-full bg-sky-500 transition-all"
					style="width: {progress && progress.total ? (progress.current / progress.total) * 100 : 5}%"></div>
			</div>
			<p class="mt-1 text-[11px] text-zinc-500">{statusMsg}</p>
		</div>
	{/if}

	<div class="flex items-center gap-2 px-3 py-2">
		<button class="btn btn-primary !py-1 text-xs" onclick={onrun} disabled={busy}>
			{busy ? 'Transcribing…' : '▶ Transcribe (Ctrl+↵)'}
		</button>
		{#if busy}
			<button class="btn !py-1 text-xs" onclick={oncancel}>Cancel</button>
		{/if}
		{#if lines.length && !busy}
			<div class="ml-auto flex items-center gap-1 text-[11px] text-zinc-500">
				<button class="btn !px-2 !py-0.5" onclick={exportTxt}>TXT</button>
				<button class="btn !px-2 !py-0.5" onclick={exportCsv}>CSV</button>
				<button class="btn !px-2 !py-0.5" onclick={() => onexport('xml')}>PAGE XML</button>
			</div>
		{/if}
	</div>

	{#if lines.length > 8}
		<div class="flex items-center gap-2 px-3 pb-1 text-[11px] text-zinc-500">
			dim &lt; {(minConf * 100).toFixed(0)}% conf
			<input type="range" min="0" max="0.95" step="0.05" bind:value={minConf} class="flex-1" />
		</div>
	{/if}

	<div class="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
		{#each visible as line (line.index)}
			<div
				class="group mb-1 flex items-start gap-2 rounded p-1 hover:bg-zinc-100 dark:hover:bg-zinc-800/60"
				class:opacity-35={line.confidence !== null && line.confidence < minConf}
			>
				<button
					class="mt-1 w-8 shrink-0 text-right font-mono text-[10px] text-zinc-400 hover:text-sky-500"
					onclick={() => onjump(line.index)}
					title="locate in image"
				>
					{line.index}
				</button>
				<span class="mt-1.5 h-2 w-2 shrink-0 rounded-full {confColor(line.confidence)}"
					title={line.confidence !== null ? `confidence ${(line.confidence * 100).toFixed(1)}%` : 'no confidence'}></span>
				{#if line.error}
					<span class="flex-1 rounded bg-red-100 px-2 py-1 text-xs text-red-700 dark:bg-red-950 dark:text-red-300"
						title={line.error}>{line.text}</span>
				{:else}
					<textarea
						class="field-input flex-1 resize-none border-0 bg-transparent px-1 py-0.5 text-sm leading-5 focus:outline-none focus:ring-1 focus:ring-sky-400 rounded"
						rows={Math.max(1, Math.ceil(line.text.length / 80))}
						bind:value={line.text}
					></textarea>
				{/if}
			</div>
		{/each}
	</div>

	{#if !lines.length && !busy}
		<div class="flex flex-1 items-center justify-center px-6 pb-6 text-center text-xs text-zinc-400">
			Segmentation lines and transcription results appear here.
		</div>
	{/if}
</div>
