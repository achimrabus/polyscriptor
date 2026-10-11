<script lang="ts">
	import { api, transcribeStream } from '$lib/api';

	type Item = {
		file: File;
		image_id?: string;
		status: 'queued' | 'processing' | 'done' | 'error';
		nlines?: number;
		error?: string;
	};

	let items = $state<Item[]>([]);
	let running = $state(false);
	let cancelled = false;
	let inputEl = $state<HTMLInputElement | null>(null);

	const pending = $derived(items.filter((i) => i.status === 'done').length);
	const exportIds = $derived(items.filter((i) => i.image_id && i.status === 'done').map((i) => i.image_id!));

	function addFiles(files: FileList | File[]) {
		for (const f of Array.from(files)) {
			if (!/\.(png|jpe?g|tif?f|bmp|webp|pdf)$/i.test(f.name)) continue;
			items = [...items, { file: f, status: 'queued' }];
		}
	}

	async function run() {
		running = true;
		cancelled = false;
		for (const item of items.filter((i) => i.status === 'queued')) {
			if (cancelled) break;
			item.status = 'processing';
			try {
				const up = await api.upload(item.file);
				item.image_id = up.image_id;
				await api.segment(item.image_id, { method: 'kraken-blla', device: 'cpu' });
				let count = 0;
				await transcribeStream(
					{ image_id: item.image_id, seg_method: 'kraken-blla', seg_device: 'cpu' },
					{
						onEvent: (ev, data) => {
							if (ev === 'complete') count = data.lines?.length ?? 0;
						}
					}
				);
				item.nlines = count;
				item.status = 'done';
			} catch (e) {
				item.status = 'error';
				item.error = String(e).slice(0, 200);
			}
		}
		running = false;
	}

	async function exportZip(fmt: 'txt' | 'csv' | 'xml') {
		const blob = await api.exportBatch(exportIds, fmt);
		const a = document.createElement('a');
		a.href = URL.createObjectURL(blob);
		a.download = `polyscriptor_batch_${fmt}.zip`;
		a.click();
		URL.revokeObjectURL(a.href);
	}

	function ondrop(e: DragEvent) {
		e.preventDefault();
		if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
	}
</script>

<div class="panel flex h-full flex-col">
		<div class="panel-title flex items-center gap-2">
			Batch queue
			<span class="ml-auto text-[10px] normal-case">{pending}/{items.length} done</span>
		</div>

		<div
			class="m-2 flex-1 overflow-y-auto rounded border-2 border-dashed border-zinc-300 p-2 dark:border-zinc-700"
			ondrop={ondrop}
			ondragover={(e) => e.preventDefault()}
			onclick={() => inputEl?.click()}
			role="button"
			tabindex="0"
			aria-label="Add files to batch queue"
		>
			<input
				bind:this={inputEl}
				type="file"
				multiple
				accept="image/*,application/pdf"
				class="hidden"
				onchange={(e) => (addFiles(e.currentTarget.files ?? []), (e.currentTarget.value = ''))}
			/>
			{#if !items.length}
				<div class="flex h-full items-center justify-center text-xs text-zinc-400">
					Drop images / PDFs here
				</div>
			{:else}
				{#each items as it, i (it.file.name + i)}
					<div class="mb-1 flex items-center gap-2 rounded bg-zinc-50 px-2 py-1 text-xs dark:bg-zinc-800/60">
						<span
							class="h-2 w-2 rounded-full"
							class:bg-zinc-300={it.status === 'queued'}
							class:bg-sky-500={it.status === 'processing'}
							class:bg-emerald-500={it.status === 'done'}
							class:bg-red-500={it.status === 'error'}></span>
						<span class="flex-1 truncate">{it.file.name}</span>
						{#if it.nlines}<span class="text-zinc-400">{it.nlines} lines</span>{/if}
						{#if it.error}<span class="text-red-500" title={it.error}>error</span>{/if}
					</div>
				{/each}
			{/if}
		</div>

		<div class="flex gap-2 border-t border-zinc-200 p-2 dark:border-zinc-800">
			<button class="btn btn-primary !py-1 text-xs" onclick={run} disabled={running || !items.some((i) => i.status === 'queued')}>
				{running ? 'Running…' : 'Process queue'}
			</button>
			{#if running}
				<button class="btn !py-1 text-xs" onclick={() => (cancelled = true)}>Stop</button>
			{/if}
			<div class="ml-auto flex gap-1">
				{#each ['txt', 'csv', 'xml'] as fmt}
					<button class="btn !px-2 !py-0.5 text-xs" onclick={() => exportZip(fmt)} disabled={!exportIds.length}>
						ZIP {fmt}
					</button>
				{/each}
			</div>
			{#if items.length && !running}
				<button class="btn !px-2 !py-0.5 text-xs" onclick={() => (items = [])}>Clear</button>
			{/if}
		</div>
	</div>
