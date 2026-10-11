<script lang="ts">
	import { api, type EngineInfo, type EngineStatus, type SchemaField } from '$lib/api';

	let {
		onloaded
	}: { onloaded: () => void } = $props();

	let engines = $state<EngineInfo[]>([]);
	let selected = $state('');
	let schema = $state<SchemaField[]>([]);
	let config = $state<Record<string, unknown>>({});
	let status = $state<EngineStatus | null>(null);
	let loading = $state(false);
	let error = $state('');

	const engine = $derived(engines.find((e) => e.name === selected));

	$effect(() => {
		refresh();
		const t = setInterval(refreshStatus, 5000);
		return () => clearInterval(t);
	});

	async function refresh() {
		try {
			const res = await api.engines();
			engines = Array.isArray(res) ? res : (res as { engines: EngineInfo[] }).engines;
			if (!selected) {
				const first = engines.find((e) => e.available);
				if (first) selectEngine(first.name);
			}
			await refreshStatus();
		} catch (e) {
			error = String(e);
		}
	}

	async function refreshStatus() {
		try {
			status = await api.engineStatus();
			if (status?.loaded && status.engine_name && !selected) {
				selected = status.engine_name;
			}
		} catch {
			/* server busy */
		}
	}

	async function selectEngine(name: string) {
		selected = name;
		error = '';
		config = {};
		if (!name) return;
		try {
			const s = await api.configSchema(name);
			schema = s.fields ?? [];
			for (const f of schema) {
				if (f.default !== undefined) config[f.key] = f.default;
				if (f.type === 'checkbox' && config[f.key] === undefined) config[f.key] = false;
			}
		} catch (e) {
			schema = [];
			error = String(e);
		}
	}

	function isCustomSelect(f: SchemaField): boolean {
		return !!f.custom_key;
	}

	function selectValue(f: SchemaField): string {
		const v = String(config[f.key] ?? '');
		if (f.options && !f.options.some((o) => o.value === v) && v !== '__custom__') {
			return f.options[0]?.value ?? '';
		}
		return v || f.options?.[0]?.value || '';
	}

	function onCustomChange(f: SchemaField, value: string) {
		config[f.custom_key!] = value;
	}

	async function load() {
		loading = true;
		error = '';
		try {
			const cfg = { ...config };
			for (const f of schema) {
				if (f.custom_key && String(cfg[f.key]) === '__custom__') {
					cfg[f.key] = cfg[f.custom_key];
					delete cfg[f.custom_key];
				}
			}
			const res = await api.loadEngine(selected, cfg);
			if (res && res.success === false) error = res.error ?? 'load failed';
			await refreshStatus();
			onloaded();
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	async function unload() {
		try {
			await api.unloadEngine();
			await refreshStatus();
		} catch (e) {
			error = String(e);
		}
	}

	function customActive(f: SchemaField): boolean {
		return String(config[f.key] ?? selectValue(f)) === '__custom__';
	}
</script>

<div class="panel flex h-full flex-col">
	<div class="panel-title flex items-center justify-between">
		Engine
		{#if status?.loaded}
			<span class="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">
				{status.engine_name} loaded
			</span>
		{:else}
			<span class="rounded bg-zinc-200 px-1.5 py-0.5 text-[10px] text-zinc-500 dark:bg-zinc-800">no model</span>
		{/if}
	</div>

	<div class="flex-1 space-y-3 overflow-y-auto p-3">
		<select class="input" value={selected} onchange={(e) => selectEngine(e.currentTarget.value)}>
			{#each engines as e}
				<option value={e.name} disabled={!e.available}>
					{e.name}{e.available ? '' : ' — unavailable'}
				</option>
			{/each}
		</select>

		{#if engine && !engine.available}
			<p class="text-xs text-amber-600">{engine.unavailable_reason ?? 'engine unavailable'}</p>
		{/if}
		{#if engine}<p class="text-xs text-zinc-500">{engine.description}</p>{/if}

		{#each schema as f (f.key)}
			{#if f.type === 'select' && f.options}
				<div>
					<label class="field-label" for={`f-${f.key}`}>{f.label}</label>
					<select
						id={`f-${f.key}`}
						class="input"
						value={selectValue(f)}
						onchange={(e) => (config[f.key] = e.currentTarget.value)}
					>
						{#each f.options as o}
							<option value={o.value}>{o.label}</option>
						{/each}
						{#if isCustomSelect(f)}
							<option value="__custom__">Custom…</option>
						{/if}
					</select>
					{#if isCustomSelect(f) && customActive(f)}
						<input
							class="input mt-1"
							placeholder={f.custom_placeholder}
							value={String(config[f.custom_key!] ?? '')}
							oninput={(e) => onCustomChange(f, e.currentTarget.value)}
						/>
					{/if}
					{#if f.hint}<p class="mt-0.5 text-[11px] text-zinc-400">{f.hint}</p>{/if}
				</div>
			{:else if f.type === 'number'}
				<div>
					<label class="field-label" for={`f-${f.key}`}>{f.label}</label>
					<input
						id={`f-${f.key}`}
						class="input"
						type="number"
						min={f.min}
						max={f.max}
						value={String(config[f.key] ?? f.default ?? '')}
						oninput={(e) => (config[f.key] = Number(e.currentTarget.value))}
					/>
					{#if f.hint}<p class="mt-0.5 text-[11px] text-zinc-400">{f.hint}</p>{/if}
				</div>
			{:else if f.type === 'checkbox'}
				<label class="flex items-center gap-2 text-sm">
					<input type="checkbox" checked={Boolean(config[f.key])} onchange={(e) => (config[f.key] = e.currentTarget.checked)} />
					{f.label}
				</label>
				{#if f.hint}<p class="text-[11px] text-zinc-400">{f.hint}</p>{/if}
			{:else if f.type === 'password'}
				<div>
					<label class="field-label" for={`f-${f.key}`}>{f.label}</label>
					<input
						id={`f-${f.key}`}
						class="input"
						type="password"
						value={String(config[f.key] ?? '')}
						oninput={(e) => (config[f.key] = e.currentTarget.value)}
					/>
				</div>
			{:else}
				<div>
					<label class="field-label" for={`f-${f.key}`}>{f.label}</label>
					<input
						id={`f-${f.key}`}
						class="input"
						value={String(config[f.key] ?? '')}
						oninput={(e) => (config[f.key] = e.currentTarget.value)}
					/>
				</div>
			{/if}
		{/each}

		{#if error}
			<p class="rounded bg-red-100 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">{error}</p>
		{/if}
	</div>

	<div class="flex gap-2 border-t border-zinc-200 p-3 dark:border-zinc-800">
		<button class="btn btn-primary flex-1" onclick={load} disabled={!selected || loading}>
			{loading ? 'Loading…' : status?.engine_name === selected && status.loaded ? 'Reload model' : 'Load model'}
		</button>
		{#if status?.loaded}
			<button class="btn" onclick={unload}>Unload</button>
		{/if}
	</div>
</div>
