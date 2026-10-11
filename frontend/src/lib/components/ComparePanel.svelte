<script lang="ts">
	import { api } from '$lib/api';

	let {
		slots,
		activeSlotId,
		hasImage,
		onswitch,
		onruncompare
	}: {
		slots: { id: string; label: string; engine: string; kind: string; lines: unknown[] }[];
		activeSlotId: string | null;
		hasImage: boolean;
		onswitch: (id: string) => void;
		onruncompare: () => Promise<void>;
	} = $props();

	let gtFile = $state<File | null>(null);
	let gtInfo = $state<{ filename: string; line_count: number } | null>(null);
	let scores = $state<Record<string, unknown>[]>([]);
	let busy = $state(false);
	let error = $state('');

	async function uploadGt() {
		if (!gtFile) return;
		error = '';
		try {
			const currentId = window.__polyscriptor_current_image;
			if (!currentId) {
				error = 'Upload an image first';
				return;
			}
			const res = await api.uploadGroundTruth(currentId, gtFile);
			gtInfo = { filename: gtFile.name, line_count: res.line_count };
			await score();
		} catch (e) {
			error = String(e);
		}
	}

	async function score() {
		busy = true;
		error = '';
		try {
			const currentId = window.__polyscriptor_current_image;
			if (!currentId) return;
			const res = await api.compareGroundTruth(currentId);
			const lb = (res as Record<string, unknown>).leaderboard;
			scores = Array.isArray(lb) ? (lb as Record<string, unknown>[]) : [];
		} catch (e) {
			error = String(e);
		} finally {
			busy = false;
		}
	}

	const metricKeys = ['micro_cer', 'macro_cer', 'macro_wer', 'micro_wer'];
	function fmt(v: unknown): string {
		return typeof v === 'number' ? (v * 100).toFixed(2) + '%' : String(v ?? '—');
	}
</script>

<div class="panel flex h-full flex-col">
	<div class="panel-title">Comparison workspace</div>

	<div class="flex-1 space-y-3 overflow-y-auto p-3">
		{#if !hasImage}
			<p class="text-xs text-zinc-400">Upload an image and run a transcription first.</p>
		{/if}

		<div>
			<p class="field-label">Result slots</p>
			{#if !slots.length}
				<p class="text-xs text-zinc-400">none yet — run a transcription, then compare another engine</p>
			{:else}
				{#each slots as s (s.id)}
					<button
						class="mb-1 flex w-full items-center gap-2 rounded border px-2 py-1.5 text-left text-xs
						{activeSlotId === s.id
							? 'border-sky-500 bg-sky-50 dark:bg-sky-950/40'
							: 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800'}"
						onclick={() => onswitch(s.id)}
					>
						<span
							class="rounded px-1 py-0.5 text-[9px] uppercase"
							class:bg-sky-100={s.kind === 'primary'}
							class:bg-amber-100={s.kind !== 'primary'}
							class:text-sky-700={s.kind === 'primary'}
							class:text-amber-700={s.kind !== 'primary'}>{s.kind}</span>
						<span class="flex-1 truncate">{s.label}</span>
						<span class="text-zinc-400">{s.lines.length}</span>
					</button>
				{/each}
			{/if}
			<button class="btn mt-1 w-full !py-1 text-xs" onclick={onruncompare} disabled={busy || !slots.length}>
				Add current loaded engine as comparison
			</button>
		</div>

		<div>
			<p class="field-label">Ground truth (PAGE XML)</p>
			<input
				type="file"
				accept=".xml"
				class="input !py-1 text-xs"
				onchange={(e) => (gtFile = e.currentTarget?.files?.[0] ?? null)}
			/>
			<button class="btn mt-1 w-full !py-1 text-xs" onclick={uploadGt} disabled={!gtFile || busy}>
				Attach &amp; score all slots
			</button>
			{#if gtInfo}
				<p class="mt-1 text-[11px] text-zinc-500">GT: {gtInfo.filename} ({gtInfo.line_count} lines)</p>
			{/if}
		</div>

		{#if scores.length}
			<table class="w-full text-left text-xs">
				<thead>
					<tr class="border-b border-zinc-200 text-zinc-500 dark:border-zinc-700">
						<th class="py-1 pr-2 font-medium">result</th>
						{#each metricKeys as k}
							<th class="py-1 pr-2 font-medium">{k.replace(/_/g, ' ')}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each scores as row, i (i)}
						<tr class="border-b border-zinc-100 dark:border-zinc-800">
							<td class="py-1 pr-2 truncate">
								{row['label'] ?? row['engine'] ?? row['slot'] ?? `#${i}`}
							</td>
							{#each metricKeys as k}
								<td class="py-1 pr-2 font-mono">{fmt(row[k])}</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
			<p class="text-[10px] text-zinc-400">lower is better; CER on the ground-truth line set</p>
		{/if}

		{#if error}
			<p class="rounded bg-red-100 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">{error}</p>
		{/if}
	</div>
</div>
